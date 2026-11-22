# Copyright 2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Offline numerical and real tiny-model tests for the local EA-DPO method."""

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from trl import DPOConfig, DPOTrainer
from trl.trainer.ea_dpo import AttentionCredit, causal_credit, select_attention, token_weights


def weight_options(**overrides):
    options = dict(
        redistribution=0.5,
        tau_chosen=1.0,
        tau_rejected=0.7,
        kappa_chosen=1.0,
        kappa_rejected=1.0,
        gamma=1.0,
        cap_quantile=0.95,
        epsilon=1e-8,
    )
    return {**options, **overrides}


def tiny_model(family="qwen2", backend="eager"):
    config_type, model_type = {
        "llama": (LlamaConfig, LlamaForCausalLM),
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[family]
    config = config_type(
        vocab_size=20,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    config._attn_implementation = backend
    return model_type(config)


def tokenizer_and_data():
    words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "user", "assistant", "question", "good", "bad", "answer", "yes", "no"]
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]"
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}"
        "{% if message['role'] == 'assistant' %}{{ eos_token + ' ' }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    )
    data = Dataset.from_dict(
        {
            "prompt": [[{"role": "user", "content": "question"}]] * 2,
            "chosen": [
                [{"role": "assistant", "content": "good answer yes"}],
                [{"role": "assistant", "content": "yes good"}],
            ],
            "rejected": [
                [{"role": "assistant", "content": "bad no"}],
                [{"role": "assistant", "content": "bad answer no"}],
            ],
        }
    )
    return tokenizer, data


def trainer_at(path, *, enabled=True, redistribution=0.5, top_fraction=1.0,
               lora=False, checkpointing=False, backend="eager"):
    torch.manual_seed(12)
    model = tiny_model(backend=backend)
    reference = copy.deepcopy(model)
    # A non-identical policy/reference pair catches the common error of applying
    # token weights to the policy only. Initial DPO log ratios otherwise vanish.
    with torch.no_grad():
        model.lm_head.weight.add_(torch.randn_like(model.lm_head.weight) * 0.03)
    tokenizer, data = tokenizer_and_data()
    args = DPOConfig(
        output_dir=str(path),
        use_cpu=True,
        bf16=False,
        report_to="none",
        max_length=32,
        per_device_train_batch_size=2,
        max_steps=2,
        logging_steps=1,
        save_strategy="no",
        gradient_checkpointing=checkpointing,
        ea_dpo_enabled=enabled,
        ea_lambda=redistribution,
        ea_top_fraction=top_fraction,
        ea_attention_chunk_size=2,
        disable_tqdm=True,
        learning_rate=1e-3,
    )
    peft_config = (
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM")
        if lora
        else None
    )
    return DPOTrainer(
        model=model,
        ref_model=None if lora else reference,
        args=args,
        processing_class=tokenizer,
        train_dataset=data,
        peft_config=peft_config,
    )


def dense_credit(attentions, projected_values, mask):
    """Independent dense oracle, deliberately using explicit PDF operations."""
    result = torch.zeros_like(mask, dtype=torch.float32)
    for b in range(mask.shape[0]):
        positions = mask[b].bool().nonzero(as_tuple=True)[0]
        a = attentions[b].float().mean(0)[positions][:, positions]
        a = a / a.sum(-1, keepdim=True)
        e = a * projected_values[b, positions].float().norm(dim=-1)[None, :]
        row_sum = e.sum(-1, keepdim=True)
        effective = torch.where(row_sum > 1e-8, e / row_sum.clamp_min(1e-8), 0)
        result[b, positions] = effective.tril(-1).sum(0)
    return result


def test_credit_uniform_attention_gating_and_padding():
    # One prompt, three completions, one padding token. Uniform attention and
    # identity projection, with V norms [1,2,3] on completion tokens.
    q = torch.zeros(1, 1, 5, 2)
    k = q.clone()
    v = torch.tensor([[[[999.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [999.0, 0.0]]]])
    mask = torch.tensor([[0, 1, 1, 1, 0]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    for chunk in (1, 2, 9):
        actual = causal_credit(q, k, v, mask, attention, torch.nn.Identity(), 1.0, chunk, 1e-8)
        torch.testing.assert_close(actual, torch.tensor([[0.0, 1 / 3 + 1 / 6, 2 / 6, 0.0, 0.0]]))
        assert actual.sum() / 3 < 1
        zero = causal_credit(q, k, v * 0, mask, attention, torch.nn.Identity(), 1.0, chunk, 1e-8)
        assert torch.count_nonzero(zero) == 0


def test_weights_equations_fold_mask_budget_and_stop_gradient():
    h = torch.tensor([[1.0, 2.0, 9.0, 777.0], [1.0, 2.0, 9.0, 777.0]], requires_grad=True)
    c = torch.tensor([[1.0, 0.5, 0.0, 777.0], [1.0, 0.5, 0.0, 777.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    weights = token_weights(h, c, mask, **weight_options(cap_quantile=0.5))
    probed_weights, (probed_h, probed_c, info, probabilities) = token_weights(
        h, c, mask, return_details=True, **weight_options(cap_quantile=0.5)
    )
    torch.testing.assert_close(probed_weights, weights, rtol=0, atol=0)
    # Rejected median=2: entropy 9 folds to zero; chosen entropy is unchanged.
    for row, folded, tau in [(0, torch.tensor([1.0, 2.0, 9.0]), 1.0), (1, torch.tensor([1.0, 2.0, 0.0]), 0.7)]:
        hn = folded / (folded.mean() + 1e-8)
        cn = c[row, :3].detach() / (c[row, :3].detach().mean() + 1e-8)
        expected = 0.5 + 0.5 * 3 * ((hn + cn + hn * cn) / tau).softmax(-1)
        torch.testing.assert_close(probed_h[row, :3], hn)
        torch.testing.assert_close(probed_c[row, :3], cn)
        torch.testing.assert_close(info[row, :3], hn + cn + hn * cn)
        torch.testing.assert_close(probabilities[row, :3], ((hn + cn + hn * cn) / tau).softmax(-1))
        torch.testing.assert_close(weights[row, :3], expected)
    torch.testing.assert_close(weights.sum(-1), torch.tensor([3.0, 3.0]))
    assert not weights.requires_grad and torch.count_nonzero(weights[:, 3]) == 0
    uniform = token_weights(h, c, mask, **weight_options(redistribution=0.0))
    torch.testing.assert_close(uniform, mask.float())
    single = token_weights(torch.ones(2, 1), torch.zeros(2, 1), torch.ones(2, 1), **weight_options())
    torch.testing.assert_close(single, torch.ones(2, 1))
    with pytest.raises(ValueError, match="Empty"):
        token_weights(h, c, torch.zeros_like(mask), **weight_options())


def test_tolerates_bf16_rounding_in_entropy():
    # 真实训练里熵来自 fused LM head 的 bf16 -(p log p) 求和，极低熵位置会溢出成
    # -1e-3 量级的负数（4090 上跑到 step ~1820 时就这么崩过一次）。这是舍入不是错误。
    h = torch.tensor([[-3e-3, 1.0, 2.0], [-3e-3, 1.0, 2.0]], dtype=torch.bfloat16)
    c = torch.tensor([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=torch.bfloat16)
    mask = torch.ones(2, 3, dtype=torch.long)
    weights = token_weights(h, c, mask, **weight_options())
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.tensor([3.0, 3.0]))

    # 明显越界的负值 / NaN 仍然必须报错
    bad = torch.tensor([[-0.5, 1.0, 2.0], [-0.5, 1.0, 2.0]])
    with pytest.raises(ValueError, match="negative"):
        token_weights(bad, c.float(), mask, **weight_options())
    nan = torch.tensor([[float("nan"), 1.0, 2.0], [float("nan"), 1.0, 2.0]])
    with pytest.raises(ValueError, match="finite"):
        token_weights(nan, c.float(), mask, **weight_options())


@pytest.mark.parametrize("length", [1, 2, 7, 10, 210, 500])
def test_top_fraction_selects_scores_preserves_budget_and_masks_gradients(length):
    width = length + 3
    entropy = torch.ones(2, width)
    credit = torch.arange(width, 0, -1).float().repeat(2, 1)
    mask = torch.zeros(2, width, dtype=torch.long)
    mask[:, :length] = 1
    weights = token_weights(
        entropy, credit, mask, **weight_options(top_fraction=0.3, gamma=0.0)
    )
    count = math.ceil(length * 0.3)
    assert (weights > 0).sum(1).tolist() == [count, count]
    assert (weights[:, :count] > 0).all()
    assert torch.count_nonzero(weights[:, count:]) == 0
    torch.testing.assert_close(weights.sum(1), torch.full((2,), float(length)))
    logps = torch.zeros(2, width, requires_grad=True)
    chosen, rejected = (weights * logps).sum(1)
    (-F.logsigmoid(chosen - rejected)).backward()
    assert torch.count_nonzero(logps.grad[:, count:]) == 0
    assert (logps.grad[:, :count] != 0).all()


def test_top_fraction_ties_are_deterministic():
    mask = torch.ones(2, 10, dtype=torch.long)
    weights = token_weights(
        torch.ones(2, 10), torch.zeros(2, 10), mask,
        **weight_options(top_fraction=0.3),
    )
    torch.testing.assert_close(weights[:, :3], torch.full((2, 3), 10 / 3))
    assert torch.count_nonzero(weights[:, 3:]) == 0


def test_evaluation_metrics_are_invariant_to_partial_batches(tmp_path):
    trainer = trainer_at(tmp_path)
    _, data = tokenizer_and_data()
    data = data.select([0, 1, 0])
    trainer.args.per_device_eval_batch_size = 1
    individual = trainer.evaluate(eval_dataset=data)
    trainer.args.per_device_eval_batch_size = 2
    batched = trainer.evaluate(eval_dataset=data)
    for metric in (
        "rewards/accuracies", "rewards/chosen", "rewards/rejected", "rewards/margins",
        "logps/chosen", "logps/rejected", "entropy", "mean_token_accuracy",
        "logits/chosen", "logits/rejected",
    ):
        key = f"eval_{metric}"
        assert batched[key] == pytest.approx(individual[key], abs=1e-5), key


@pytest.mark.parametrize("family", ["llama", "qwen2", "qwen3"])
@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_observer_preserves_model_and_matches_dense_oracle(family, backend):
    torch.manual_seed(7)
    model = tiny_model(family, backend).eval()
    oracle = copy.deepcopy(model)
    oracle.set_attn_implementation("eager")
    ids = torch.tensor([[0, 1, 4, 6, 7, 8], [1, 4, 6, 7, 8, 0]])
    attention_mask = (ids != 0).long()
    mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 0]])
    layer, index = select_attention(model, None)
    oracle_layer, _ = select_attention(oracle, index)
    projected = []

    def record_values(module, args, output):
        b, s, _ = output.shape
        values = output.view(b, s, 2, 4).repeat_interleave(2, dim=2).reshape(b, s, 16)
        projected.append(oracle_layer.o_proj(values).detach())

    handle = oracle_layer.v_proj.register_forward_hook(record_values)
    with torch.no_grad():
        dense = oracle(input_ids=ids, attention_mask=attention_mask, output_attentions=True, use_cache=False)
        expected = dense_credit(dense.attentions[index], projected[0], mask)
        before = model(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits
        registry = set(ALL_ATTENTION_FUNCTIONS)
        observer = AttentionCredit(layer, mask, attention_mask, 2, 1e-8)
        with observer.capture():
            actual = model(input_ids=ids, attention_mask=attention_mask, use_cache=False).logits
        torch.testing.assert_close(actual, before, rtol=0, atol=0)
        torch.testing.assert_close(observer.credit, expected, atol=2e-6, rtol=2e-5)
        assert set(ALL_ATTENTION_FUNCTIONS) == registry
        assert layer.config._attn_implementation == backend
        assert not observer.credit.requires_grad
    handle.remove()
    with pytest.raises(RuntimeError, match="deliberate"):
        with observer.capture():
            raise RuntimeError("deliberate")
    assert layer.config._attn_implementation == backend
    assert set(ALL_ATTENTION_FUNCTIONS) == registry


@pytest.mark.parametrize("top_fraction,redistribution", [(1.0, 0.5), (0.3, 0.5), (0.3, 0.0)])
def test_trainer_loss_and_gradients_match_equation_12(tmp_path, top_fraction, redistribution):
    trainer = trainer_at(tmp_path, top_fraction=top_fraction, redistribution=redistribution)
    batch = next(iter(trainer.get_train_dataloader()))
    model = trainer.model
    layer = trainer._ea_attention
    captured = []

    def save_value(module, args, output):
        b, s, _ = output.shape
        values = output.view(b, s, 2, 4).repeat_interleave(2, dim=2).reshape(b, s, 16)
        captured.append(layer.o_proj(values).detach())

    handle = layer.v_proj.register_forward_hook(save_value)
    inputs = {k: batch[k] for k in ("input_ids", "attention_mask")}
    output = model(**inputs, output_attentions=True, use_cache=False)
    handle.remove()
    mask = batch["completion_mask"][:, 1:].bool()
    logits = output.logits[:, :-1].float()
    log_dist = logits.log_softmax(-1)
    entropy = -(log_dist.exp() * log_dist).sum(-1)
    credit = dense_credit(output.attentions[trainer._ea_layer_index], captured[0], batch["completion_mask"])
    weights = token_weights(
        entropy, credit[:, 1:], mask,
        **weight_options(top_fraction=top_fraction, redistribution=redistribution),
    )
    labels = batch["input_ids"][:, 1:]
    policy_logps = log_dist.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        ref_logits = trainer.ref_model(**inputs, use_cache=False).logits[:, :-1].float()
        ref_logps = ref_logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    rewards = trainer.args.beta * (weights * (policy_logps - ref_logps)).sum(-1)
    chosen, rejected = rewards.chunk(2)
    expected = -F.logsigmoid(chosen - rejected).mean()
    expected.backward()
    expected_grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    model.zero_grad(set_to_none=True)
    actual = trainer.compute_loss(model, batch)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.backward()
    for name, param in model.named_parameters():
        if name in expected_grads:
            torch.testing.assert_close(param.grad, expected_grads[name], atol=2e-6, rtol=2e-4)
    assert all(p.grad is None for p in trainer.ref_model.parameters())


def test_lambda_zero_recovers_standard_dpo_loss_and_gradient(tmp_path):
    baseline = trainer_at(tmp_path / "base", enabled=False)
    ea = trainer_at(tmp_path / "ea", redistribution=0)
    batch = next(iter(baseline.get_train_dataloader()))
    base_loss = baseline.compute_loss(baseline.model, batch)
    ea_loss = ea.compute_loss(ea.model, batch)
    torch.testing.assert_close(ea_loss, base_loss, atol=0, rtol=0)
    base_loss.backward()
    ea_loss.backward()
    for left, right in zip(baseline.model.parameters(), ea.model.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_two_training_steps_save_and_reload(tmp_path, lora, checkpointing):
    trainer = trainer_at(tmp_path, lora=lora, checkpointing=checkpointing, backend="sdpa")
    before = {name: p.detach().clone() for name, p in trainer.model.named_parameters() if p.requires_grad}
    output = trainer.train()
    assert math.isfinite(output.training_loss) and trainer.state.global_step == 2
    assert any(not torch.equal(before[name], p) for name, p in trainer.model.named_parameters() if name in before)
    trainer.save_model(str(tmp_path / "saved"))
    config_path = tmp_path / "saved" / ("adapter_config.json" if lora else "config.json")
    assert "ea_dpo_" not in config_path.read_text(encoding="utf-8")
    if not lora:
        reloaded = Qwen2ForCausalLM.from_pretrained(tmp_path / "saved")
        assert reloaded.config.model_type == "qwen2"
    assert any("ea/weight_mean_chosen" in row for row in trainer.state.log_history)
    for row in trainer.state.log_history:
        if "ea/weight_mean_chosen" in row:
            assert row["ea/weight_mean_chosen"] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "overrides",
    [
        {"ea_lambda": -1},
        {"ea_lambda": float("nan")},
        {"ea_top_fraction": 0},
        {"ea_top_fraction": 1.1},
        {"ea_top_fraction": float("nan")},
        {"ea_tau_rejected": 2},
        {"ea_kappa_rejected": 0.5},
        {"ea_entropy_cap_quantile": 0},
        {"ea_attention_chunk_size": 0},
        {"precompute_ref_log_probs": True},
        {"loss_type": "ipo"},
        {"ld_alpha": 1},
        {"use_weighting": True},
        {"sync_ref_model": True},
        {"padding_free": True},
        {"disable_dropout": False},
        {"torch_compile": True},
    ],
)
def test_incompatible_configurations_fail_early(tmp_path, overrides):
    with pytest.raises(ValueError, match="EA-DPO"):
        DPOConfig(
            output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to="none", ea_dpo_enabled=True, **overrides
        )


@pytest.mark.parametrize("lora", [False, True])
def test_reentrant_checkpointing(tmp_path, lora):
    trainer = trainer_at(tmp_path, lora=lora, checkpointing=True, backend="sdpa")
    trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    result = trainer.train()
    assert math.isfinite(result.training_loss)


def test_invalid_model_and_target_layer():
    model = tiny_model()
    with pytest.raises(ValueError, match="zero-based"):
        select_attention(model, 10)
    with pytest.raises(ValueError, match="zero-based"):
        select_attention(model, -1)
    model.model.layers[0].self_attn.sliding_window = 3
    with pytest.raises(ValueError, match="sliding"):
        select_attention(model, 0)


def test_mean_budget_with_zero_signals_and_extreme_scores():
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    zeros = torch.zeros(2, 3)
    uniform = token_weights(zeros, zeros, mask, **weight_options(redistribution=1))
    torch.testing.assert_close(uniform, mask.float())
    h = torch.tensor([[1e10, 1.0, 0.0], [1.0, 1.0, 1e10]])
    weights = token_weights(h, zeros, mask, **weight_options(redistribution=1, tau_rejected=1e-4))
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    torch.testing.assert_close(weights.sum(-1), mask.sum(-1).float())


def test_rejects_missing_prompt_and_empty_completion(tmp_path):
    trainer = trainer_at(tmp_path)
    batch = next(iter(trainer.get_train_dataloader()))
    invalid = {key: value.clone() for key, value in batch.items()}
    invalid["completion_mask"][0] = 0
    with pytest.raises(ValueError, match="prompt token"):
        trainer.compute_loss(trainer.model, invalid)
    first = batch["attention_mask"][0].nonzero(as_tuple=True)[0][0]
    batch["completion_mask"][0, first] = 1
    with pytest.raises(ValueError, match="prompt token"):
        trainer.compute_loss(trainer.model, batch)
