"""Candidate task-adaptive pruning policy for the Delta expert router."""

# Keep ratio is the fraction of visual tokens retained.  Referring receives
# more tokens because its accuracy degrades sharply under aggressive pruning.
CANDIDATE_TASK_PRUNE_CONFIG = {
    "vqa": {"method": "l2norm", "keep_ratio": 0.50},
    "mcq": {"method": "l2norm", "keep_ratio": 0.50},
    "change": {"method": "l2norm", "keep_ratio": 0.50},
    "caption": {"method": "l2norm", "keep_ratio": 0.50},
    "referring": {"method": "scope_l2", "keep_ratio": 0.75},
}
