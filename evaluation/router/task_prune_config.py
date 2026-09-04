"""Offline-derived task-adaptive pruning policies for the Delta Router.

R is the visual-token keep ratio.  The strict policy uses the smallest tested
R whose primary metric is within epsilon=0.05 of the R=1 baseline.  The
compression policy deliberately relaxes the Referring constraint to use the
better ScopeL2 point at R=0.50.
"""

THRESHOLD_EPSILON = 0.05

# Empirical retention thresholds (R*) for L2Norm, the deterministic default,
# computed on the exp7 R sweep with epsilon=0.05:
#   VQA    (VRSBench Accuracy):     R* = 0.50  (drop 0.035)
#   MME    (MCQ Accuracy):          R* = 0.25  (drop 0.024)
#   XLRS   (MCQ Accuracy):          R* = 0.10  (drop 0.004)
#   Change (LEVIR-CC CIDEr):        R* = 0.50  (drop 0.043)
#   Referring (Acc@0.5):            ScopeL2 R* = 0.75 (kept per task choice)
# XLRS is routed to "mcq" (same task key as MME), so the binding MME threshold
# R=0.25 also covers XLRS.
THRESHOLD_TASK_PRUNE_CONFIG = {
    "vqa": {"method": "l2norm", "keep_ratio": 0.50},
    "mcq": {"method": "l2norm", "keep_ratio": 0.25},
    "change": {"method": "l2norm", "keep_ratio": 0.50},
    "referring": {"method": "scope_l2", "keep_ratio": 0.75},
    # Caption threshold is provisional until the corrected XLRS Caption sweep
    # is complete; VRSBench Caption is stable at this operating point.
    "caption": {"method": "l2norm", "keep_ratio": 0.50},
}

# Optional compression-focused policy: ScopeL2 gives the strongest Referring
# score among the aggressive R=0.50 operating points.
COMPRESSION_TASK_PRUNE_CONFIG = dict(THRESHOLD_TASK_PRUNE_CONFIG)
COMPRESSION_TASK_PRUNE_CONFIG["referring"] = {
    "method": "scope_l2", "keep_ratio": 0.50,
}

# The default Router is conservative and follows the formal threshold rule.
CANDIDATE_TASK_PRUNE_CONFIG = THRESHOLD_TASK_PRUNE_CONFIG
