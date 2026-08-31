from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from verl.trainer.ppo.core_algos import compute_self_distillation_loss


class OPSDLossTests(unittest.TestCase):
    def test_topk_region_global_jsd_is_finite_and_differentiable(self) -> None:
        student_logits = torch.tensor(
            [[[-0.2, -1.2, -2.2], [-1.4, -0.4, -2.4]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        teacher_logits = torch.tensor(
            [[[-0.4, -1.0, -2.0], [-1.0, -0.6, -2.2]]],
            dtype=torch.float32,
        )
        student_topk = torch.log_softmax(student_logits, dim=-1)
        teacher_topk = torch.log_softmax(teacher_logits, dim=-1)
        response_mask = torch.tensor([[1.0, 1.0]])
        config = SimpleNamespace(
            full_logit_distillation=True,
            distillation_topk=3,
            distillation_add_tail=True,
            alpha=0.5,
            is_clip=None,
        )

        loss, metrics = compute_self_distillation_loss(
            student_log_probs=student_topk[..., 0],
            teacher_log_probs=teacher_topk[..., 0],
            response_mask=response_mask,
            self_distillation_config=config,
            student_topk_log_probs=student_topk,
            teacher_topk_log_probs=teacher_topk,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()), 0.0)
        self.assertEqual(metrics["self_distillation/num_distill_tokens"], 2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(student_logits.grad).all())


if __name__ == "__main__":
    unittest.main()
