"""Versioned manifest schema for corrected remote-sensing evaluation samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "1.0"
TaskType = Literal[
    "caption",
    "open_vqa",
    "bbox",
    "single_choice",
    "multi_choice",
    "change_caption",
]
CleanStatus = Literal["keep", "corrected", "manual_review", "exclude"]


@dataclass(frozen=True)
class ImageRef:
    path: str
    role: str
    width: int
    height: int
    transform: str = "none"

    def validate(self, *, check_exists: bool = False) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Invalid image size for {self.path}: {self.width}x{self.height}"
            )
        if check_exists and not Path(self.path).is_file():
            raise FileNotFoundError(self.path)


@dataclass
class Sample:
    id: str
    dataset: str
    subtask: str
    task_type: TaskType
    prompt: str
    images: list[ImageRef]
    references: list[str]
    choices: dict[str, str] = field(default_factory=dict)
    answer_labels: list[str] = field(default_factory=list)
    accepted_labels: list[str] = field(default_factory=list)
    clean_status: CleanStatus = "keep"
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    split: str = ""
    source: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self, *, check_images: bool = False) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        if not self.id or not self.dataset or not self.subtask:
            raise ValueError("Sample id, dataset, and subtask must be non-empty")
        if not self.prompt.strip():
            raise ValueError(f"Sample {self.id} has an empty prompt")
        if not self.images:
            raise ValueError(f"Sample {self.id} has no images")
        for image in self.images:
            image.validate(check_exists=check_images)
        if self.task_type == "change_caption" and len(self.images) != 2:
            raise ValueError(
                f"Change-caption sample {self.id} must contain exactly two images"
            )
        if self.task_type in {"single_choice", "multi_choice"}:
            if not self.choices or not self.answer_labels:
                raise ValueError(
                    f"Choice sample {self.id} lacks choices or answer labels"
                )
            unknown = set(self.answer_labels) - set(self.choices)
            if unknown:
                raise ValueError(
                    f"Sample {self.id} has unknown answer labels: {sorted(unknown)}"
                )
            accepted = self.accepted_labels or self.answer_labels
            if set(accepted) - set(self.choices):
                raise ValueError(f"Sample {self.id} has unknown accepted labels")
        elif not self.references:
            raise ValueError(f"Sample {self.id} has no references")
        if (
            self.clean_status == "keep"
            and self.issues
            and any(issue.startswith("exclude:") for issue in self.issues)
        ):
            raise ValueError(
                f"Sample {self.id} is marked keep despite an exclusion issue"
            )


    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(
        cls, raw: dict[str, Any], *, manifest_dir: Path | None = None
    ) -> "Sample":
        data = dict(raw)
        schema_ver = data.get("schema_version", "1.0")
        if schema_ver == "2.0-full-clean":
            return cls._from_v2_record(data, manifest_dir=manifest_dir)
        images = []
        for raw_image in data["images"]:
            image = dict(raw_image)
            image_path = Path(image["path"])
            if manifest_dir is not None and not image_path.is_absolute():
                image["path"] = str((manifest_dir / image_path).resolve())
            images.append(ImageRef(**image))
        data["images"] = images
        sample = cls(**data)
        sample.validate()
        return sample

    @classmethod
    def _from_v2_record(
        cls, data: dict[str, Any], *, manifest_dir: Path | None = None
    ) -> "Sample":
        images = []
        for img in data["images"]:
            p = img["path"]
            if manifest_dir is not None and not Path(p).is_absolute():
                p = str((manifest_dir / p).resolve())
            images.append(
                ImageRef(
                    path=p,
                    role=img["role"],
                    width=img["width"],
                    height=img["height"],
                    transform=img.get("transform", "none"),
                )
            )
        metadata = dict(data.get("metadata", {}))
        if "split" in data:
            metadata["split"] = data["split"]
        if "source" in data:
            metadata["source"] = data["source"]
        sample = cls(
            id=data["id"],
            dataset=data["dataset"],
            subtask=data["subtask"],
            task_type=data["task_type"],
            prompt=data["prompt"],
            images=images,
            references=data.get("references", []),
            choices=data.get("choices", {}),
            answer_labels=data.get("answer_labels", []),
            accepted_labels=data.get("accepted_labels", []),
            clean_status=data.get("clean_status", "keep"),
            issues=data.get("issues", []),
            metadata=metadata,
        )
        sample.validate()
        return sample
