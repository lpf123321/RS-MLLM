"""Offline-derived task-adaptive pruning policies for the Delta Router.

R is the visual-token keep ratio.  The strict policy uses the smallest tested
R whose primary metric is within epsilon=0.05 of the R=1 baseline.  Grounding
uses L2Norm so both VRSBench Referring and full-resolution XLRS Grounding share
one inexpensive selector in evaluation and serving.
"""

THRESHOLD_EPSILON = 0.05

# Empirical retention thresholds (R*) for L2Norm, the deterministic default,
# computed on the exp7 R sweep with epsilon=0.05:
#   VQA    (VRSBench Accuracy):     R* = 0.25  (drop 0.035)
#   MME    (MCQ Accuracy):          R* = 0.25  (drop 0.024)
#   XLRS   (MCQ Accuracy):          R* = 0.10  (drop 0.004)
#   Change (LEVIR-CC CIDEr):        R* = 0.50  (drop 0.043)
#   Referring (Acc@0.5):            L2Norm R* = 0.75 (grounding-wide policy)
# XLRS is routed to "mcq" (same task key as MME), so the binding MME threshold
# R=0.25 also covers XLRS.
THRESHOLD_TASK_PRUNE_CONFIG = {
    "vqa": {"method": "l2norm", "keep_ratio": 0.25},
    "mcq": {"method": "l2norm", "keep_ratio": 0.25},
    "change": {"method": "l2norm", "keep_ratio": 0.50},
    "referring": {"method": "l2norm", "keep_ratio": 0.75},
    # Caption threshold is provisional until the corrected XLRS Caption sweep
    # is complete; VRSBench Caption is stable at this operating point.
    "caption": {"method": "l2norm", "keep_ratio": 0.50},
}

# Optional compression-focused policy keeps the same selector but uses the
# more aggressive Referring operating point.
COMPRESSION_TASK_PRUNE_CONFIG = dict(THRESHOLD_TASK_PRUNE_CONFIG)
COMPRESSION_TASK_PRUNE_CONFIG["referring"] = {
    "method": "l2norm", "keep_ratio": 0.50,
}

# The default Router is conservative and follows the formal threshold rule.
CANDIDATE_TASK_PRUNE_CONFIG = THRESHOLD_TASK_PRUNE_CONFIG
