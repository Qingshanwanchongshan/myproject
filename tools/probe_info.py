#!/usr/bin/env python3
"""探针：EA-DPO 的权重为什么这么集中？

在真实训练的第 1-2 个微批上，记录 H~ / C~ / I / p_max / w_max 的分布，
判断是"熵"还是"因果入度"贡献了离群值。

    python tools/probe_info.py --config configs/ea_dpo.yaml --max_steps 2 \
        --eval_strategy no --save_strategy no --output_dir outputs/probe-ea
"""

import runpy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trl-1.18.1"))
import trl.trainer.dpo_trainer as dpo_trainer

_original = dpo_trainer.token_weights
_calls = 0
_MAX_REPORT = 3


def wrapped(entropy, credit, mask, **kwargs):
    global _calls
    if _calls >= _MAX_REPORT:
        return _original(entropy, credit, mask, **kwargs)

    weights, (h, c, info, probs) = _original(entropy, credit, mask, return_details=True, **kwargs)
    valid = mask.bool()
    half = h.shape[0] // 2
    print(f"\n[probe] 第 {_calls + 1} 个微批  lambda={kwargs['redistribution']:.2f} gamma={kwargs['gamma']:.2f}")
    for name, rows in (("chosen", slice(half)), ("rejected", slice(half, None))):
        sel = valid[rows]
        hh, cc = h[rows][sel], c[rows][sel]
        ii, pp, ww = info[rows][sel], probs[rows][sel], weights[rows][sel]
        tau = kwargs[f"tau_{name}"]
        length = sel.sum(-1).float().mean()
        print(f"  {name:8s} tau={tau:.2f} H~ max={hh.max():.2f} C~ p99={torch.quantile(cc, 0.99):.2f} "
              f"C~ max={cc.max():.2f} I max={ii.max():.2f}")
        print(f"  {'':8s} p_max={pp.max():.4f} 平均长度={length:.1f} w_max={ww.max():.2f} w 中位={ww.median():.2f}")
        print(f"  {'':8s} mean(H~)={hh.mean():.2f} mean(C~)={cc.mean():.2f} mean(H~C~)={(hh * cc).mean():.2f}")
    _calls += 1
    return weights


if __name__ == "__main__":
    dpo_trainer.token_weights = wrapped
    sys.argv = ["train_dpo.py"] + sys.argv[1:]
    runpy.run_path("train_dpo.py", run_name="__main__")
