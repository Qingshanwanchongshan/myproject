# EA-DPO

在 DPO 损失里引入 **token 级加权 + 注意力统计** 的偏好优化实现，基于 **TRL v1.18.1** 的修改版，
可直接在服务器上复现冒烟实验。

> 研究代码，尚未发表。仓库里的默认超参数是**可跑通的起点，不是调参结论**。

## 仓库结构

```
.
├── train_dpo.py            # 训练入口；强制加载本目录下的修改版 TRL
├── configs/
│   ├── ea_dpo.yaml         # EA-DPO 10 步冒烟配置
│   └── dpo_baseline.yaml   # 同设置的标准 DPO 对照配置
├── trl-1.18.1/             # 修改版 TRL（核心改动见下）
│   ├── trl/trainer/ea_dpo.py          # 理论核心数学 + 单层 attention 统计
│   ├── trl/trainer/dpo_trainer.py     # 把统计与 token 权重接入训练 loss
│   ├── trl/trainer/dpo_config.py      # 新增超参数与开关
│   ├── tests/test_ea_dpo.py           # 公式与训练链路测试
│   └── docs/source/ea_dpo.md          # 逐公式映射、代码清单、数值约定与限制
├── EA_DPO_修改说明.md      # 修改清单、服务器操作步骤、已知边界
└── 准备跑实验.pdf          # 实验准备文档
```

所有改动都以 `trl/trainer/` 为中心，TRL 其余部分保持与官方 v1.18.1 一致；
上游开源协议见 `trl-1.18.1/LICENSE`（Apache-2.0）。

## 环境要求

- Python >= 3.11
- PyTorch >= 2.6（按服务器 CUDA 版本装）
- `transformers==5.23.0`

```bash
python -m pip install -e './trl-1.18.1[peft]' 'transformers==5.23.0' pytest
python -c "import trl; print(trl.__file__)"   # 输出必须指向本仓库内的 trl-1.18.1
python -m pytest trl-1.18.1/tests/test_ea_dpo.py -q
```

> 不要用 `pip install -U trl` —— 那会装上未修改的官方包，覆盖掉这里的实现。

## 快速开始

两组 10 步冒烟实验（Qwen2.5-0.5B-Instruct + ultrafeedback_binarized，仅验证流程）：

```bash
python train_dpo.py --config configs/dpo_baseline.yaml
python train_dpo.py --config configs/ea_dpo.yaml
```

跑通后可覆盖步数与输出路径：

```bash
python train_dpo.py --config configs/ea_dpo.yaml --max_steps 1000 --output_dir outputs/ea-dpo-run1
```

配置默认使用 BF16，需要 GPU 支持；不支持时同步修改 dtype 与 bf16/fp16。
首次运行会从 Hugging Face 下载模型与数据。

## 默认开关与边界

- EA-DPO 的开关**默认关闭**，只有用 `configs/ea_dpo.yaml` 启动才会打开；`--ea_lambda 0` 可验证是否退回标准 DPO。
- 默认作用层约为 80% 深度；默认 `lambda=0.5`、`tau_w=1`、`tau_l=0.7`、`kappa_w=kappa_l=1`、`gamma=1`、`H_cap` 分位数 0.95。
- 当前只适配纯文本 Llama / Qwen2 / Qwen2.5 / Qwen3 的**全注意力层**；大模型 GPU 训练、FlashAttention、量化、多卡吞吐与实际效果仍待在服务器上验证。
- 注意力统计按块计算以降低峰值显存，但额外计算量仍是**平方级**，不是零开销。

详见 [EA_DPO_修改说明.md](EA_DPO_修改说明.md) 与 [trl-1.18.1/docs/source/ea_dpo.md](trl-1.18.1/docs/source/ea_dpo.md)。

## 许可

本仓库对 TRL 的修改沿袭上游 **Apache-2.0**；如后续公开发表，请补充引用信息。
