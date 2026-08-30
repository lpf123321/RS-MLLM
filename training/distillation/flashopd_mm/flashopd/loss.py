"""KL 散度损失函数：forward / reverse / JSD / top-k sparse，支持温度缩放."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kl_type: str = "reverse",
    temperature: float = 1.0,
    top_k: int = 0,
    chunk_size: int = 128,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算 token 级 KL 散度，自动分块避免 OOM.

    Args:
        student_logits: (B, L, V)
        teacher_logits: (B, L, V)
        kl_type: "reverse" = KL(s||t), "forward" = KL(t||s)
        temperature: softmax 温度
        top_k: >0 时仅在 teacher top-k token 上计算
        chunk_size: 沿 seq 维分块大小

    Returns:
        标量 KL loss（已做 T² 缩放）
    """
    min_v = min(student_logits.shape[-1], teacher_logits.shape[-1])
    s = student_logits[:, :, :min_v]
    t = teacher_logits[:, :, :min_v]

    B, L, _ = s.shape
    # KL 在 Student/Teacher 分布很接近时容易因 BF16 消减误差变成微小负数，
    # 概率与对数运算统一提升到 FP32。
    total = torch.zeros((), device=s.device, dtype=torch.float32)
    denominator = (
        token_mask.to(device=s.device, dtype=s.dtype).sum().clamp_min(1.0)
        if token_mask is not None
        else B * L
    )

    for i in range(0, L, chunk_size):
        j = min(i + chunk_size, L)
        sc, tc = s[:, i:j].float(), t[:, i:j].float()

        if 0 < top_k < sc.shape[-1]:
            idx = tc.topk(top_k, dim=-1).indices
            sc = sc.gather(-1, idx)
            tc = tc.gather(-1, idx)

        s_lp = F.log_softmax(sc / temperature, dim=-1)
        t_lp = F.log_softmax(tc / temperature, dim=-1)

        if kl_type == "forward":
            t_p = F.softmax(tc / temperature, dim=-1).detach()
            token_kl = (t_p * (t_lp.detach() - s_lp)).sum(dim=-1)
        else:  # reverse
            s_p = F.softmax(sc / temperature, dim=-1)
            token_kl = (s_p * (s_lp - t_lp.detach())).sum(dim=-1)
        token_kl = token_kl.clamp_min(0.0)
        if token_mask is not None:
            token_kl = token_kl * token_mask[:, i:j].to(token_kl.dtype)
        total = total + token_kl.sum()

    return total / denominator * (temperature ** 2)


def jsd_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    alpha: float = 0.5,
    top_k: int = 0,
    chunk_size: int = 128,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Jensen-Shannon 散度 = alpha * KL(s||m) + (1-alpha) * KL(t||m)."""
    min_v = min(student_logits.shape[-1], teacher_logits.shape[-1])
    s = student_logits[:, :, :min_v]
    t = teacher_logits[:, :, :min_v]

    B, L, _ = s.shape
    total = torch.zeros((), device=s.device, dtype=torch.float32)
    denominator = (
        token_mask.to(device=s.device, dtype=s.dtype).sum().clamp_min(1.0)
        if token_mask is not None
        else B * L
    )

    for i in range(0, L, chunk_size):
        j = min(i + chunk_size, L)
        sc, tc = s[:, i:j].float(), t[:, i:j].float()

        if 0 < top_k < sc.shape[-1]:
            idx = tc.topk(top_k, dim=-1).indices
            sc = sc.gather(-1, idx)
            tc = tc.gather(-1, idx)

        s_p = F.softmax(sc / temperature, dim=-1)
        t_p = F.softmax(tc / temperature, dim=-1).detach()
        m_p = alpha * s_p + (1 - alpha) * t_p
        m_lp = m_p.log()

        kl_s = (s_p * (s_p.log() - m_lp)).sum(dim=-1)
        kl_t = (t_p * (t_p.log() - m_lp)).sum(dim=-1)
        token_jsd = (alpha * kl_s + (1 - alpha) * kl_t).clamp_min(0.0)
        if token_mask is not None:
            token_jsd = token_jsd * token_mask[:, i:j].to(token_jsd.dtype)
        total = total + token_jsd.sum()

    return total / denominator * (temperature ** 2)


def kl_from_sparse_logprobs(
    student_logits: torch.Tensor,
    teacher_top_ids: torch.Tensor,
    teacher_top_logprobs: torch.Tensor,
    kl_type: str = "reverse",
    temperature: float = 1.0,
    token_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """使用 API 返回的 sparse logprobs 计算 KL（无需 teacher 完整 logits）.

    Args:
        student_logits: (B, L, V)
        teacher_top_ids: (B, L, K)
        teacher_top_logprobs: (B, L, K) — 已是 log prob
    """
    T = temperature
    B, L, K = teacher_top_ids.shape

    s_lp = F.log_softmax(student_logits.float() / T, dim=-1)
    s_lp_at_t = s_lp.gather(-1, teacher_top_ids)
    teacher_top_logprobs = teacher_top_logprobs.float()

    if T != 1.0:
        t_lp = teacher_top_logprobs / T
        t_lp = t_lp - torch.logsumexp(t_lp, dim=-1, keepdim=True)
    else:
        t_lp = teacher_top_logprobs

    if kl_type == "forward":
        t_p = t_lp.exp().detach()
        token_kl = (t_p * (t_lp.detach() - s_lp_at_t)).sum(dim=-1)
    else:
        s_p = s_lp_at_t.exp()
        token_kl = (s_p * (s_lp_at_t - t_lp.detach())).sum(dim=-1)
    token_kl = token_kl.clamp_min(0.0)

    if token_mask is not None:
        mask = token_mask.to(device=token_kl.device, dtype=token_kl.dtype)
        token_kl = token_kl * mask
        denominator = mask.sum().clamp_min(1.0)
    else:
        denominator = B * L
    return token_kl.sum() / denominator * (T ** 2)


def clip_kl(kl: torch.Tensor, kl_min: float = 0.0, kl_max: float = 0.0) -> torch.Tensor:
    """对 KL loss 做上下界截断."""
    if kl_min > 0:
        kl = torch.clamp(kl, min=kl_min)
    if kl_max > 0:
        kl = torch.clamp(kl, max=kl_max)
    return kl
