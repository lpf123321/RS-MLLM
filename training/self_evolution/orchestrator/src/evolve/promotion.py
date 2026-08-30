from __future__ import annotations

from typing import Any


def evaluate_candidate(
    baseline: dict[str, Any], candidate: dict[str, Any],
    baseline_sam: dict[str, Any], candidate_sam: dict[str, Any],
    rules: dict[str, Any], regression_drop: float = 0.0,
) -> dict[str, Any]:
    """Evaluate bbox-free promotion gates; pseudo box metrics are diagnostic only."""
    accuracy_gain = float(candidate["accuracy"]) - float(baseline["accuracy"])
    correct_gain = int(candidate["correct"]) - int(baseline["correct"])
    baseline_pop = float(baseline["mean_pop"])
    candidate_pop = float(candidate["mean_pop"])
    baseline_tree = int(baseline.get("tree_search_samples", baseline.get("routes", {}).get("2", 0)))
    candidate_tree = int(candidate.get("tree_search_samples", candidate.get("routes", {}).get("2", 0)))
    equal_accuracy_efficiency = (
        correct_gain == 0
        and (baseline_pop == 0.0 == candidate_pop or candidate_pop <= baseline_pop * 0.95)
        and candidate_tree <= baseline_tree
    )
    benefit = correct_gain >= 1 or equal_accuracy_efficiency
    baseline_fpr = float(baseline_sam.get("negative_fpr_at_0_5", 0.0))
    candidate_fpr = float(candidate_sam.get("negative_fpr_at_0_5", baseline_fpr))
    checks = {
        "correct_not_lower": correct_gain >= 0,
        "effective_benefit": benefit,
        "negative_fpr_not_higher": candidate_fpr <= baseline_fpr,
        "direct_accuracy": regression_drop <= float(rules["max_direct_accuracy_drop"]),
    }
    return {
        "accepted": all(checks.values()), "checks": checks,
        "accuracy_gain": accuracy_gain, "correct_gain": correct_gain,
        "mean_pop_change": candidate_pop - baseline_pop,
        "tree_search_change": candidate_tree - baseline_tree,
        "negative_fpr_change": candidate_fpr - baseline_fpr,
        "pseudo_metrics_are_diagnostic_only": True,
    }
