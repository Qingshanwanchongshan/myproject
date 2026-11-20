# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""EA-DPO equations (3)-(10) and a single-layer attention observer.

The observer leaves the model's attention output unchanged. Statistics are
detached, computed in FP32, and accumulated in query blocks, not stored as a
full batch of attention matrices. See docs/source/ea_dpo.md for conventions.
"""

import copy
import importlib
import math
from contextlib import contextmanager

import torch
from transformers import AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


@torch.no_grad()
def causal_credit(query, key, value, completion_mask, attention_mask, output_projection, scaling, chunk_size, delta):
    """Compute Eqs. (4)-(6) from post-RoPE Q/K and unweighted V.

    Q is [B, H, S, D]; K/V may have fewer heads (GQA). The head-averaged
    attention is restricted to completion tokens and row-normalized BEFORE
    norm gating. Self attention participates in the denominator, but not in C.
    """
    batch, heads, length, _ = query.shape
    if completion_mask.shape != (batch, length) or attention_mask.shape != (batch, length):
        raise ValueError("EA-DPO requires un-packed full sequences; Q/K and token masks must align.")
    if key.shape[-2] != length or value.shape[-2] != length or heads % key.shape[1]:
        raise ValueError("EA-DPO requires full-sequence MHA/GQA without a KV cache.")
    groups = heads // key.shape[1]
    credit = torch.zeros((batch, length), device=query.device, dtype=torch.float32)
    completion_mask = completion_mask.bool()
    attention_mask = attention_mask.bool()
    if (completion_mask & ~attention_mask).any():
        raise ValueError("Completion tokens cannot be padding.")

    for b in range(batch):
        positions = completion_mask[b].nonzero(as_tuple=True)[0]
        if not positions.numel():
            raise ValueError("EA-DPO requires at least one completion token in each sequence.")
        # Repeat each KV head for its query-head group, then concatenate heads.
        # Project the UNWEIGHTED per-token V, not the attention-mixed output.
        values = value[b, :, positions, :].repeat_interleave(groups, dim=0)
        values = values.transpose(0, 1).reshape(positions.numel(), -1)
        transformed = output_projection(values)
        # f(x)=v W^O excludes output-projection bias. This also honors PEFT and
        # quantized projection modules without reading a possibly packed weight.
        bias = output_projection(torch.zeros_like(values[:1]))
        norms = (transformed.float() - bias.float()).norm(dim=-1)
        keys = key[b].repeat_interleave(groups, dim=0).float()
        all_positions = torch.arange(length, device=query.device)
        for start in range(0, positions.numel(), chunk_size):
            rows = positions[start : start + chunk_size]
            with torch.autocast(device_type=query.device.type, enabled=False):
                logits = query[b, :, rows, :].float() @ keys.transpose(-1, -2)
                logits *= scaling
                allowed = (all_positions[None, :] <= rows[:, None]) & attention_mask[b][None, :]
                logits.masked_fill_(~allowed[None, :, :], -torch.inf)
                # Average full-context probabilities across heads first, then
                # condition on completion keys. Log space prevents underflow
                # when almost all attention mass is on a long prompt.
                log_probs = logits.log_softmax(dim=-1)
                log_a = torch.logsumexp(log_probs[:, :, positions], dim=0) - math.log(heads)
                a = (log_a - torch.logsumexp(log_a, dim=-1, keepdim=True)).exp()
                energy = a * norms[None, :]
                row_sum = energy.sum(dim=-1, keepdim=True)
                effective = torch.where(row_sum > delta, energy / row_sum.clamp_min(delta), 0.0)
                effective *= positions[None, :] < rows[:, None]
                credit[b, positions] += effective.sum(dim=0)
    return credit


@torch.no_grad()
def token_weights(
    entropy,
    credit,
    mask,
    *,
    redistribution,
    tau_chosen,
    tau_rejected,
    kappa_chosen,
    kappa_rejected,
    gamma,
    cap_quantile,
    epsilon,
):
    """Eqs. (3), (7)-(10), with rows ordered [chosen, rejected].

    Returns detached FP32 weights, zero outside valid completion tokens.
    Each non-empty completion has mean weight one. At lambda=1 weights are
    non-negative, not necessarily strictly positive in floating point.
    """
    if entropy.shape != credit.shape or entropy.shape != mask.shape or entropy.shape[0] % 2:
        raise ValueError("EA-DPO expects equally shaped [2*batch, tokens] signals in chosen/rejected order.")
    valid = mask.bool()
    lengths = valid.sum(-1, keepdim=True)
    if (lengths == 0).any():
        raise ValueError("Empty completions after truncation cannot be used for EA-DPO.")
    h = entropy.detach().float().masked_fill(~valid, 0.0)
    c = credit.detach().float().masked_fill(~valid, 0.0)
    if not torch.isfinite(h).all() or not torch.isfinite(c).all() or (h < 0).any() or (c < 0).any():
        raise ValueError("EA-DPO entropy and causal credit must be finite and non-negative.")
    half = h.shape[0] // 2
    for row in range(half, h.shape[0]):
        cap = torch.quantile(h[row, valid[row]], cap_quantile)
        h[row] = torch.minimum(h[row], (2 * cap - h[row]).clamp_min(0))
    h = h / (h.sum(-1, keepdim=True) / lengths + epsilon)
    c = c / (c.sum(-1, keepdim=True) / lengths + epsilon)
    kappa = h.new_tensor([kappa_chosen] * half + [kappa_rejected] * half)[:, None]
    tau = h.new_tensor([tau_chosen] * half + [tau_rejected] * half)[:, None]
    information = h + kappa * c + gamma * h * c
    probabilities = (information / tau).masked_fill(~valid, -torch.inf).softmax(-1)
    weights = (1 - redistribution) + redistribution * lengths * probabilities
    return weights.masked_fill(~valid, 0.0)


def select_attention(model, target_layer):
    """Select a full-attention Llama, Qwen2/2.5, or Qwen3 text layer."""
    if model.config.model_type not in {"llama", "qwen2", "qwen3"}:
        raise ValueError("EA-DPO currently supports text Llama, Qwen2/2.5 and Qwen3 models only.")
    layers = model.model.layers
    index = math.ceil(0.8 * len(layers)) - 1 if target_layer is None else target_layer
    if not 0 <= index < len(layers):
        raise ValueError(f"ea_target_layer must be a zero-based index in [0, {len(layers) - 1}].")
    attention = layers[index].self_attn
    if not type(attention).__module__.startswith("transformers.models."):
        raise ValueError("EA-DPO requires a native Transformers attention module.")
    if model.config.model_type in {"qwen2", "qwen3"} and attention.sliding_window is not None:
        raise ValueError("EA-DPO requires a full-attention target layer; select a non-sliding-window layer.")
    if type(attention).__name__ not in {"LlamaAttention", "Qwen2Attention", "Qwen3Attention"}:
        raise ValueError("EA-DPO requires the native Transformers attention module, without replacement kernels.")
    return attention, index


class AttentionCredit:
    """Observe one policy forward; reference and checkpoint recomputations are unobserved."""

    def __init__(self, attention, completion_mask, attention_mask, chunk_size, delta):
        self.attention = attention
        self.completion_mask = completion_mask
        self.attention_mask = attention_mask
        self.chunk_size = chunk_size
        self.delta = delta
        self.credit = None

    @contextmanager
    def capture(self):
        original_config = self.attention.config
        backend = original_config._attn_implementation
        if backend == "eager":
            module = importlib.import_module(type(self.attention).__module__)
            original_attention = module.eager_attention_forward
        elif backend in {"sdpa", "flash_attention_2", "flash_attention_3"}:
            original_attention = ALL_ATTENTION_FUNCTIONS[backend]
        else:
            raise ValueError(f"EA-DPO does not support attention backend {backend!r}.")

        def observed_attention(module, query, key, value, attention_mask, **kwargs):
            self.credit = causal_credit(
                query.detach(),
                key.detach(),
                value.detach(),
                self.completion_mask,
                self.attention_mask,
                module.o_proj,
                kwargs.get("scaling", module.scaling),
                self.chunk_size,
                self.delta,
            )
            return original_attention(module, query, key, value, attention_mask, **kwargs)

        # A private config copy on this attention module preserves the model's
        # original mask construction and saved config. Restore even on failure.
        # A unique name avoids cross-talk between simultaneously alive trainers.
        name = f"ea_dpo_{id(self)}"
        AttentionInterface.register(name, observed_attention)
        self.attention.config = copy.copy(original_config)
        self.attention.config._attn_implementation = name
        try:
            yield self
            if self.credit is None:
                raise RuntimeError("EA-DPO target attention was not observed during the policy forward.")
        finally:
            self.attention.config = original_config
            # GeneralInterface.__delitem__ only deletes local overrides; the
            # public register method writes the shared mapping.
            AttentionInterface._global_mapping.pop(name, None)
