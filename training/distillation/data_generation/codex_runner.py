#!/usr/bin/env python3
"""Run auditable image-annotation jobs through an interactive Codex login.

No OpenAI API key is read. The subprocess inherits the user's persisted
`codex login` session and receives images read-only. Full runs are blocked until
the user explicitly approves a completed smoke review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


APPROVAL_TEXT = "I reviewed the smoke outputs and approve the full run"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def check_login() -> str:
    command = ["codex", "login", "status"]
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.pop("CODEX_API_KEY", None)
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    combined = (result.stdout + "\n" + result.stderr).strip()
    if result.returncode or "Logged in" not in combined:
        raise RuntimeError("Codex is not logged in. Run `codex login` interactively first.")
    if "API key" in combined:
        raise RuntimeError("This runner requires ChatGPT/Codex account login, not API-key login.")
    return combined.splitlines()[-1]


def codex_version() -> str:
    result = subprocess.run(["codex", "--version"], text=True, capture_output=True, check=True)
    return result.stdout.strip()


def validate_job(job: dict[str, Any], base: Path) -> dict[str, Any]:
    job_id = str(job.get("job_id", "")).strip()
    if not job_id or "/" in job_id or ".." in job_id:
        raise ValueError(f"Unsafe or missing job_id: {job_id!r}")
    prompt = str(job.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"Job {job_id} has no prompt")
    images = []
    for value in job.get("images", []):
        candidate = Path(str(value)).expanduser()
        path = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append(str(path))
    if not images:
        raise ValueError(f"Job {job_id} has no images")
    return {**job, "job_id": job_id, "prompt": prompt, "images": images}


def validate_output(value: Any, schema: dict[str, Any]) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("Install jsonschema to validate Codex outputs") from exc
    model_value = value
    if isinstance(value, dict) and "_provenance" in value:
        model_value = {key: item for key, item in value.items() if key != "_provenance"}
    jsonschema.validate(model_value, schema)


def enforce_expected_identity(value: dict[str, Any], job: dict[str, Any]) -> None:
    """Reject output that changes identifiers frozen by the job builder."""
    payload = job.get("payload", {})
    expected = payload.get("expected_identity", {})
    if not isinstance(expected, dict):
        raise ValueError(f"Job {job['job_id']} expected_identity must be an object")
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"Job {job['job_id']} output changed frozen identity fields: "
            f"{canonical_json(mismatches)}"
        )


def smoke_digest(results: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(results.glob("*.json"))
    if not files:
        raise ValueError(f"No smoke results under {results}")
    for path in files:
        digest.update(path.name.encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def run_jobs(args: argparse.Namespace) -> None:
    login = check_login()
    version = codex_version()
    jobs_path = args.jobs.expanduser().resolve()
    schema_path = args.schema.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jobs = [validate_job(job, jobs_path.parent) for job in read_jsonl(jobs_path)]
    ids = [job["job_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate job_id values")
    if args.stage == "smoke":
        jobs = jobs[: args.smoke_size]
        if len(jobs) < args.smoke_size:
            raise ValueError(f"Smoke needs {args.smoke_size} jobs, got {len(jobs)}")
    else:
        approval = run_root / "smoke/APPROVED.json"
        if not approval.is_file():
            raise RuntimeError(
                "Full run is blocked. Review smoke/results and run the approve-smoke command first."
            )
        approved = json.loads(approval.read_text(encoding="utf-8"))
        current = smoke_digest(run_root / "smoke/results")
        if approved.get("smoke_results_sha256") != current:
            raise RuntimeError("Smoke outputs changed after approval; review and approve them again.")
    if args.limit is not None:
        jobs = jobs[: args.limit]

    stage_root = run_root / args.stage
    results_root = stage_root / "results"
    logs_root = stage_root / "logs"
    results_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 1, "stage": args.stage, "model": args.model,
        "codex_version": version, "login": login, "jobs": str(jobs_path),
        "jobs_sha256": sha256(jobs_path), "output_schema": str(schema_path),
        "output_schema_sha256": sha256(schema_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "uses_openai_api_key": False, "smoke_size": args.smoke_size,
    }
    write_json(stage_root / "RUN_METADATA.json", metadata)

    completed = 0
    for index, job in enumerate(jobs):
        job_id = job["job_id"]
        destination = results_root / f"{job_id}.json"
        if destination.is_file():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            validate_output(existing, schema)
            enforce_expected_identity(existing, job)
            completed += 1
            continue
        partial = results_root / f".{job_id}.partial.json"
        event_log = logs_root / f"{job_id}.events.jsonl"
        instruction = "\n\n".join((
            "You are producing one auditable training-data annotation. Inspect only the attached image(s).",
            job["prompt"],
            "Job payload:\n" + canonical_json(job.get("payload", {})),
            "Return only the JSON object required by the supplied output schema. Do not edit files, "
            "call external services, or infer hidden/test labels.",
        ))
        command = [
            "codex", "exec", "-", "--ephemeral", "--ignore-user-config",
            "--skip-git-repo-check", "--sandbox", "read-only", "--model", args.model,
            "--output-schema", str(schema_path), "--output-last-message", str(partial),
            "--json", "--color", "never", "--cd", str(jobs_path.parent),
            "--image", *job["images"],
        ]
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("CODEX_API_KEY", None)
        with event_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, input=instruction, text=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        value = json.loads(partial.read_text(encoding="utf-8"))
        validate_output(value, schema)
        enforce_expected_identity(value, job)
        value.setdefault("_provenance", {})
        value["_provenance"].update({
            "job_id": job_id, "job_index": index, "model": args.model,
            "codex_version": version,
            "job_payload_sha256": hashlib.sha256(
                canonical_json(job.get("payload", {})).encode("utf-8")
            ).hexdigest(),
            "image_sha256": {Path(path).name: sha256(Path(path)) for path in job["images"]},
        })
        validate_output(value, schema)
        write_json(destination, value)
        partial.unlink()
        completed += 1
        print(f"[{completed}/{len(jobs)}] {job_id}", flush=True)
    metadata.update({"completed": completed, "finished_at": datetime.now(timezone.utc).isoformat()})
    write_json(stage_root / "RUN_METADATA.json", metadata)


def approve_smoke(args: argparse.Namespace) -> None:
    if args.confirmation != APPROVAL_TEXT:
        raise ValueError(f"Confirmation must be exactly: {APPROVAL_TEXT!r}")
    run_root = args.run_root.expanduser().resolve()
    results = run_root / "smoke/results"
    files = sorted(results.glob("*.json"))
    if len(files) < args.smoke_size:
        raise ValueError(f"Need at least {args.smoke_size} reviewed smoke results, found {len(files)}")
    reviewer = args.reviewer.strip()
    if not reviewer:
        raise ValueError("Reviewer must not be empty")
    approval = {
        "approved": True, "reviewer": reviewer, "notes": args.notes,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "smoke_size": args.smoke_size, "smoke_results_sha256": smoke_digest(results),
        "confirmation": args.confirmation,
    }
    write_json(run_root / "smoke/APPROVED.json", approval)
    print(json.dumps(approval, ensure_ascii=False, indent=2))


def status(args: argparse.Namespace) -> None:
    root = args.run_root.expanduser().resolve()
    value = {}
    for stage in ("smoke", "full"):
        results = root / stage / "results"
        value[stage] = len(list(results.glob("*.json"))) if results.is_dir() else 0
    value["smoke_approved"] = (root / "smoke/APPROVED.json").is_file()
    print(json.dumps(value, indent=2))


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate a planned run without checking login or invoking Codex."""
    jobs_path = args.jobs.expanduser().resolve()
    schema_path = args.schema.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        import jsonschema
    except ImportError as exc:
        raise RuntimeError("Install jsonschema to validate the output schema") from exc
    jsonschema.Draft202012Validator.check_schema(schema)
    jobs = [validate_job(job, jobs_path.parent) for job in read_jsonl(jobs_path)]
    ids = [job["job_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate job_id values")
    selected = jobs
    if args.stage == "smoke":
        selected = jobs[: args.smoke_size]
        if len(selected) < args.smoke_size:
            raise ValueError(f"Smoke needs {args.smoke_size} jobs, got {len(selected)}")
    else:
        approval = run_root / "smoke/APPROVED.json"
        if not approval.is_file():
            raise RuntimeError("Full run is blocked until the completed smoke is approved.")
        approved = json.loads(approval.read_text(encoding="utf-8"))
        current = smoke_digest(run_root / "smoke/results")
        if approved.get("smoke_results_sha256") != current:
            raise RuntimeError("Smoke outputs changed after approval; review and approve them again.")
    if args.limit is not None:
        selected = selected[: args.limit]
    print(json.dumps({
        "valid": True,
        "stage": args.stage,
        "jobs_available": len(jobs),
        "jobs_selected": len(selected),
        "images_selected": sum(len(job["images"]) for job in selected),
        "jobs_sha256": sha256(jobs_path),
        "output_schema_sha256": sha256(schema_path),
        "codex_invoked": False,
        "login_checked": False,
    }, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--jobs", type=Path, required=True)
    run.add_argument("--schema", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--stage", choices=("smoke", "full"), required=True)
    run.add_argument("--smoke-size", type=int, default=20)
    run.add_argument("--limit", type=int)
    run.add_argument("--model", default=os.environ.get("CODEX_DATA_MODEL", "gpt-5.6-sol"))
    validate = sub.add_parser("validate")
    validate.add_argument("--jobs", type=Path, required=True)
    validate.add_argument("--schema", type=Path, required=True)
    validate.add_argument("--run-root", type=Path, required=True)
    validate.add_argument("--stage", choices=("smoke", "full"), required=True)
    validate.add_argument("--smoke-size", type=int, default=20)
    validate.add_argument("--limit", type=int)
    approve = sub.add_parser("approve-smoke")
    approve.add_argument("--run-root", type=Path, required=True)
    approve.add_argument("--smoke-size", type=int, default=20)
    approve.add_argument("--reviewer", required=True)
    approve.add_argument("--notes", default="")
    approve.add_argument("--confirmation", required=True)
    inspect = sub.add_parser("status")
    inspect.add_argument("--run-root", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "run":
        run_jobs(args)
    elif args.command == "validate":
        validate_inputs(args)
    elif args.command == "approve-smoke":
        approve_smoke(args)
    else:
        status(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
