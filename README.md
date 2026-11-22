# EA-DPO

新服务器迁移入口：[migration/START_HERE.txt](migration/START_HERE.txt)。包含与旧服务器一致的核心实现、环境版本及下一轮配置；数据集、历史 LoRA 权重和实验结果通过 Git LFS 提供；克隆后执行 `git lfs pull` 和 `python migration/restore_assets.py`。公开基础模型需另行下载。

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](requirements.txt)
[![TRL 1.18.1](https://img.shields.io/badge/TRL-1.18.1-orange.svg)](trl-1.18.1)

在 DPO 损失里引入 **Token 级熵 + 注意力因果入度加权** 的偏好优化实现，基于 **TRL v1.18.1** 的修改版。
仓库自带一套可直接在单卡 4090 上跑完的实验流程：第一轮用 50% 数据看趋势，第二轮再上全量。

> 研究代码，尚未发表。所有超参数都是**可跑通的起点，不是调参结论**。

---

## 1. 目录结构

```
.
├── train_dpo.py                 # 训练入口：强制加载本目录的修改版 TRL，支持 --resume_from_checkpoint auto
├── requirements.txt             # 依赖清单（torch 按 CUDA 版本单独装，见文件头）
├── LICENSE                      # Apache-2.0（沿袭上游 TRL）
├── tests/
│   └── test_tools.py            # 自检：配置检查器、四个工具、续训解析、脚本卫生（约 20 个用例）
├── configs/
│   ├── ea_dpo.yaml              # 第一轮 EA-DPO（50% 数据，约 3813 步）
│   ├── dpo_baseline.yaml        # 第一轮标准 DPO 对照（除 ea_* 外逐字段相同）
│   ├── ea_dpo_full.yaml         # 第二轮 EA-DPO（全量 1 epoch）
│   └── dpo_baseline_full.yaml   # 第二轮标准 DPO 对照
├── scripts/
│   ├── setup_env.sh             # 一次性环境准备（torch / trl / 校验 / 单测 / 自检）
│   ├── prepare_data.py          # 落盘固定数据集：30500 对训练集 + 512 对固定评测集
│   ├── run_first_round.sh       # 第一轮全流程：冒烟 → 测速 → 两组训练 → 汇总
│   └── run_full.sh              # 第二轮全量
├── tools/
│   ├── check_configs.py         # 静态检查：配置字段、两组可比性、脚本参数
│   ├── report.py                # 单组训练健康度汇总
│   ├── compare_runs.py          # 两组曲线并排对比（标注哪些数字能横向比）
│   ├── eval_implicit_reward.py   # 同口径离线评测：不加权的标准 DPO 隐式奖励胜率
│   ├── estimate_time.py         # 从日志实测速度换算两轮耗时
│   └── watch_gpu.sh             # 采样 GPU 利用率/显存（另开终端，可选）
├── trl-1.18.1/                  # 修改版 TRL，核心改动见下
│   ├── trl/trainer/ea_dpo.py            # 理论核心数学 + 单层 attention 统计
│   ├── trl/trainer/dpo_trainer.py       # 把统计与 Token 权重接入训练 loss
│   ├── trl/trainer/dpo_config.py        # 新增超参数与组合校验
│   ├── tests/test_ea_dpo.py             # 公式与训练链路测试（33 个）
│   └── docs/source/ea_dpo.md            # 逐公式映射、数值约定与限制
├── EA_DPO_修改说明.md           # 改动清单、实验方案、上机步骤、已知边界
├── 待验证清单.md                 # ⚠️ 出问题先看这里：论文未规定/未验证的约定与可疑点
└── 准备跑实验.pdf               # 方法说明（原始手稿）
```

上游 TRL 其余部分与官方 v1.18.1 一致；协议见 `LICENSE`（Apache-2.0，沿袭 `trl-1.18.1/LICENSE`）。

---

## 1.5 不用 GPU 就能跑的自检

```bash
python -m pytest tests/ -q          # 约 20 个用例：检查器、四个工具、续训解析、脚本卫生
python tools/check_configs.py       # 配置字段 / 两组可比性 / 脚本参数（0.1 秒）
```

这套自检不需要 GPU、不需要 transformers、不下载任何模型 —— 改完配置或脚本先跑一遍，
比等模型下载完再报错省时间。上了服务器之后再跑 `trl-1.18.1/tests/test_ea_dpo.py`（33 个，
需要 torch，验证公式与训练链路）。

---

## 2. 快速开始（4090 单卡，三条命令）

```bash
bash scripts/setup_env.sh          # 装 torch/trl/transformers，跑 33 个单测，检查 GPU 与 bf16
python scripts/prepare_data.py     # 生成 data/uf-subset（30500 对）与 data/uf-full
bash scripts/run_first_round.sh    # 冒烟 → 测速 → 两组各约 3813 步 → 出对比表
```

`run_first_round.sh` 是幂等的：某组跑完后再执行会跳过；训练中断后重跑会
`--resume_from_checkpoint auto` 从最新存档继续。日志在 `logs/`，产物在 `outputs/`。

想手动跑单条命令（与脚本里的完全相同）：

```bash
# 10 步冒烟：写一个只有 adapter 的存档，好让 tools/report.py 精确校验权重均值
python train_dpo.py --config configs/ea_dpo.yaml --max_steps 10 --eval_strategy no \
  --save_strategy steps --save_steps 10 --save_total_limit 1 --save_only_model true \
  --report_to none --output_dir outputs/smoke-ea
python tools/report.py outputs/smoke-ea

# 正式一组（中断后重跑会自动续上）
python train_dpo.py --config configs/ea_dpo.yaml --resume_from_checkpoint auto
```

---

## 3. 第一轮的关键设置，以及为什么

第一轮 = 全量数据的 50%（30500 对）、约 3813 步，目标是**看趋势是否分叉**，不是出结论。改数据量只改 `prepare_data.py --train-size`，配置里是 `max_steps: -1`，步数会自动跟着变。

| 设置 | 值 | 为什么 |
|---|---|---|
| 模型 | Qwen2.5-3B-Instruct | 3B 是第一个真实规模；bf16 LoRA 预计占 11-14 GiB（以 `timing` 阶段实测为准） |
| 数据 | `data/uf-subset`，train 30500 对 / test 512 对 | 30500 是全量的 50%；**两组共用同一份落盘数据**，样本顺序完全一致 |
| 评测集 | 512 对固定子集 | 全量 test 约 2000 对，一次评测要 12-15 分钟，15 次就是 3-4 小时；512 对 3-4 分钟就够看趋势 |
| 步数 | 3813 = 30500 / 8 | 正好是这个子集的一个 epoch（配置里 `max_steps: -1` 自动算） |
| batch | 1 × 8 = 有效 8 对 | 24G 卡上留足显存余量；两组一致就不影响可比性 |
| 学习率 | 1.0e-5，线性衰减，3% warmup | all-linear LoRA 的常用量级。对照组：`LR=5.0e-6 TAG=lr5e-6 bash scripts/run_first_round.sh` |
| β | 0.1（显式写入） | DPO 标准值；写进配置是为了复现，不靠默认值 |
| LoRA | r=16, α=32, `all-linear` | α/r=2；显式用 all-linear 而不是 peft 的 q/v 默认映射，是为了让两种方法都有足够容量把差异表现出来（容量太窄会把方法差异一起压平） |
| 长度 | max_length 1024 | TRL 的 DPO 惯例；超长样本被截断，prompt 就占满长度的样本会被训练器丢弃，不会报错 |
| 评测/存档 | 每 500 步，保留 3 个 | 3813 步拿到 8 个评测点；一次评测约 3-4 分钟，总开销约半小时 |
| 梯度检查点 | 显式 true | TRL 本来默认就是 true，写出来是因为它会**让反向重算前向**，EA 的注意力统计因此每步算两遍（成本可接受，但要知道） |
| 日志 | `report_to: tensorboard` | 出曲线；没装 tensorboard 时会自动跳过，不会让训练失败 |
| 随机种子 | 两组都 42 | 唯一的变量只剩 `ea_*` |

EA-DPO 专有参数全部保持论文默认值：λ=0.5、τ_chosen=1.0、τ_rejected=0.7、κ=1.0/1.0、γ=1.0、
H_cap 分位数 0.95、目标层自动 = `ceil(0.8 × 36) - 1 = 28`（第 29 层）。
第一轮**不调这些**——调了就看不出方法本身的效果。

---

## 4. 结果怎么读（这条最容易被误读）

**先说清楚两者的关系**：标准 DPO 和 EA-DPO 报的是**同一种量** —— 隐式奖励之差：

```
r(y)    = β · Σ_t [log π(y_t) − log π_ref(y_t)]          基线：权重全 1
r_EA(y) = β · Σ_t w_t · [log π(y_t) − log π_ref(y_t)]    EA：w_t 样本自适应，均值恒为 1
```

形式、量纲、尺度都一样，数值高度相关。所以**论文里能并排比较这些变体**，这一点你说得对。

**需要小心的是尺子**：`w_t` 由样本自己的熵和注意力因果入度算出来，等于说它是 π 的函数。
于是 EA 的胜率是 `f_EA(π)`、基线的胜率是 `f_dpo(π)` —— 同一个模型换把尺子数会变，
两个模型各用各的尺子比，**口径差就混进了方法差里**。口径差通常只有 1-3 个百分点，
但一篇方法论文要证明的效应量也常常就是 1-2 个百分点，两者同量级。

**所以做法是：用第三把尺子同时量两边**，而不是各自报各自的。三步，按性价比排：

1. **同一把尺子重算（本仓库已带，`report` 阶段自动跑）**

   ```bash
   python tools/eval_implicit_reward.py \
     --adapter outputs/dpo-3b-20pct:DPO --adapter outputs/ea-3b-20pct:EA-DPO
   ```

   在同一批固定评测集上，对两个 adapter 都重算**不加权**的标准 DPO 隐式奖励与胜率。
   它还会把每个 adapter **自己日志里的加权口径**并排列出来，两列相减就是**这一组的口径差
   到底多大** —— 不用猜。口径差小到可忽略，直接看训练日志也够；大到你想要的效应量那么大，
   就必须用这一列。4090 上 256 对约 6-8 分钟。

2. **训练日志里本来就同口径的部分**：`entropy`、`mean_token_accuracy`、`logits/*`
   （定义与权重无关，可以直接比；也和基线训练日志里的同名指标一一对应）。

3. **生成质量**：两组 adapter 在同一批固定 prompt 上各生成一次，裁判模型 pairwise 判定。
   这是"有没有用"的最终证据，但需要一个能调裁判模型的环境。

另外有个便宜的自洽性检查：`--ea_lambda 0` 应该**严格退化成标准 DPO**（代码里有这条路径），
跑 10 步对比 loss 曲线即可确认实现没有额外偏置。

```bash
python tools/compare_runs.py outputs/dpo-3b-20pct outputs/ea-3b-20pct --labels DPO EA-DPO
python tools/report.py outputs/ea-3b-20pct          # 单组健康度
tensorboard --logdir outputs --port 6006            # 曲线
```

---

## 5. 环境要求

- **目标平台：Linux**（Ubuntu 22.04 / 24.04 + NVIDIA 驱动 + 单卡 4090）。所有脚本是 `bash`，
  全部 LF 行尾、无 BOM，不含任何 Windows 专有内容
- Python >= 3.11（租来的镜像常是 3.10，`setup_env.sh` 会先拦下来）
- PyTorch >= 2.6（按服务器 CUDA 版本装；`scripts/setup_env.sh` 会先检查已有的 torch）
- `transformers==5.23.0`、`jinja2>=3.1.6`
- 单卡 >= 24 GiB（4090 / A5000 / L40S / A100）。16 GiB 卡风险高；V100 没有 bf16

国内机器建议 `export HF_ENDPOINT=https://hf-mirror.com`（脚本已默认设置）。

---

## 6. 已知边界

- 只适配纯文本 Llama / Qwen2 / Qwen2.5 / Qwen3 的**全注意力层**；`sdpa` 或 `eager` 后端。
- 注意力统计是**精确的分块实现**，额外计算仍是平方级，不是零开销，也不能称为 FlashAttention 融合核。
- 不支持 FSDP / DeepSpeed / 张量并行 / padding-free / torch.compile / VLM / 量化路径，
  配置里已做前置校验，踩到会直接报错而不是静默错位。
- 本仓库的静态检查与工具脚本在无 GPU 机器上验证过；**真实训练速度、显存占用、效果
  仍需在服务器上确认**（`run_first_round.sh` 的测速阶段会给出实际数字）。

详见 [EA_DPO_修改说明.md](EA_DPO_修改说明.md) 与 [trl-1.18.1/docs/source/ea_dpo.md](trl-1.18.1/docs/source/ea_dpo.md)。

---

## 7. 引用

方法与实现**尚未发表**。在它变成论文之前，引用这个仓库即可：

```bibtex
@misc{ea_dpo_2026,
  title  = {EA-DPO: Entropy-Attention Weighted Direct Preference Optimization},
  author = {TODO},
  year   = {2026},
  note   = {Research code, unpublished. Repository: <你的仓库地址>},
}
```

方法描述的来源是本目录下的 `准备跑实验.pdf`（2026-11-18，公式 (2)-(12)）；
逐公式与代码的对应关系见 [trl-1.18.1/docs/source/ea_dpo.md](trl-1.18.1/docs/source/ea_dpo.md)。

发表之后请把上表换成正式的 arXiv/会议条目——`EA_DPO_修改说明.md` 里已经列出投稿前
还缺的三件事（外部生成评测、多种子、消融）。

---

## 8. 许可

本仓库对 TRL 的修改沿袭上游 **Apache-2.0**，见 [LICENSE](LICENSE)。
`trl-1.18.1/` 之外新增的代码（scripts / tools / tests / configs）同样以 Apache-2.0 发布。
