# Fixed-teacher off-policy distillation

This directory adds teacher-trajectory distillation without modifying the
full Vision-OPD source tree. Set `VISION_OPD_ROOT` to that checkout at runtime.

The completed on-policy experiment samples responses from the current 4B
student.  This experiment samples responses once from the frozen full-data-SFT
9B teacher and replays those token trajectories through VERL.  At every teacher
prefix, the existing Vision-OPD actor computes teacher and student distributions
and updates only the student's rank-8 LoRA.

Pipeline:

1. `scripts/run_offpolicy_teacher_trajectory_generation.sh`
2. `scripts/run_formal_offpolicy_distillation.sh`

The formal comparison intentionally retains the completed on-policy run's loss,
optimizer, batch size, response length, data order, image policy, and random
seed. Only the response trajectory source changes from the student to the
teacher. Generated trajectories, checkpoints, and reports are written below
`ARTIFACT_ROOT` and are not part of this source submission.
