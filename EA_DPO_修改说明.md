# EA-DPO：已修改内容与服务器操作

详细的逐公式映射、代码清单、数值约定和限制见 [完整实现说明](trl-1.18.1/docs/source/ea_dpo.md)。

## 你需要关注的文件

| 文件 | 作用 |
|---|---|
| `trl-1.18.1/trl/trainer/ea_dpo.py` | 你的理论的核心数学与单层 attention 统计 |
| `trl-1.18.1/trl/trainer/dpo_trainer.py` | 将上述统计和 Token 权重接入训练 loss |
| `trl-1.18.1/trl/trainer/dpo_config.py` | 所有新增超参数与开关 |
| `train_dpo.py` | 训练启动，确保加载本目录的修改版 TRL |
| `configs/ea_dpo.yaml` | EA-DPO 10 步冒烟配置 |
| `configs/dpo_baseline.yaml` | 同设置的标准 DPO 对照配置 |
| `trl-1.18.1/tests/test_ea_dpo.py` | 公式和实际训练测试 |

这次已接入真实的注意力、value 投影、熵、权重和损失，默认开关仍为关闭。用 EA-DPO 配置启动才会开启。

## 上传

上传以下内容，保持相对目录关系：

```text
你的服务器实验目录/
├── train_dpo.py
├── configs/
│   ├── ea_dpo.yaml
│   └── dpo_baseline.yaml
└── trl-1.18.1/
```

`.venv-ea-test/` 是本地测试环境，`tmp/` 是测试输出，均不必上传；`__pycache__/`、`.pytest_cache/` 也不需要。原始 `trl-v1.18.1.zip` 保持为官方备份，**不包含此次修改**，不要只传这个旧 ZIP。

## 服务器安装与验证

使用 Python 3.11 以上，在训练环境先安装适配服务器 CUDA 的 PyTorch（TRL 要求 >=2.6）。然后在上述实验目录运行：

```bash
python -m pip install -e './trl-1.18.1[peft]' 'transformers==5.23.0' pytest
python -c "import trl; print(trl.__file__)"
python -m pytest trl-1.18.1/tests/test_ea_dpo.py -q
```

输出路径应指向上传的源码目录。不要用 `pip install -U trl` 覆盖成本地未修改的官方包。

若 `pytest` 或训练时报 `ImportError: apply_chat_template requires jinja2>=3.1.6`，再补一条
`python -m pip install -U 'jinja2>=3.1.6'`：transformers 5.x 只在运行时检查这个版本下限，pip 不会主动升级已装的旧版。
这套组合（Python 3.11、torch 2.8、transformers 5.23.0）已复跑，33 个测试全部通过；Windows 控制台为 GBK 时也能正常结束并保存模型。

先运行两组 10 步验证：

```bash
python train_dpo.py --config configs/dpo_baseline.yaml
python train_dpo.py --config configs/ea_dpo.yaml
```

示例使用 Qwen2.5-0.5B-Instruct 和 ultrafeedback_binarized，仅作为流程验证；后续将两份配置中的模型/数据一起换成正式实验对象。配置使用 BF16，要求 GPU 支持；不支持时同步修改 dtype、bf16/fp16。首次运行会从 Hugging Face 下载模型和数据，需要联网或事先准备本地缓存。

跑通后可以覆盖步数与输出路径：

```bash
python train_dpo.py --config configs/ea_dpo.yaml --max_steps 1000 --output_dir outputs/ea-dpo-run1
```

冒烟配置关闭中间保存，但入口仍在训练结束后保存模型。正式长训练建议设置 `--save_strategy steps --save_steps 100`，以便断点恢复。两组比较时保持模型、数据顺序、batch、学习率、训练步数、随机种子一致。

默认层为约 80% 深度；默认 lambda=0.5、tau_w=1、tau_l=0.7、kappa_w=kappa_l=1、gamma=1、H_cap 分位数=0.95。这些数值不是调参结论，需要消融与验证。`--ea_lambda 0` 可验证恢复标准 DPO。

## 需要知道的边界

当前适配纯文本 Llama/Qwen2/Qwen2.5/Qwen3 的全注意力层。CPU 小模型测试验证算法和训练链路；大模型 GPU 训练、FlashAttention、量化、多卡吞吐和实际效果仍需要服务器验证。精确的注意力统计按块计算，减少峰值存储，但额外计算仍是平方级，不能把它表述为零开销或线性复杂度。
