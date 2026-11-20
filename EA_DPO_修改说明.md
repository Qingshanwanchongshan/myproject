# EA-DPO：改动清单与上机操作

方法本身的逐公式映射、数值约定和限制见
[完整实现说明](trl-1.18.1/docs/source/ea_dpo.md)。本文只讲"改了什么、怎么跑、出了问题怎么查"。

---

## 1. 这一版改了什么

**核心算法**（`trl-1.18.1/trl/trainer/`）：熵、注意力因果入度、非对称保均值权重全部接入真实的
注意力、value 投影与损失；`ea_dpo_enabled` 默认关闭，只有用 EA 配置启动才打开。

**实验流程**（仓库根目录）：把"能不能一次跑通"当成需求重做了一遍。

| 文件 | 作用 |
|---|---|
| `configs/ea_dpo.yaml` | 第一轮 EA-DPO：20% 数据、1500 步、4090 单卡 |
| `configs/dpo_baseline.yaml` | 第一轮对照：与上一行**逐字段相同**，只有 `ea_*` 不同 |
| `configs/ea_dpo_full.yaml` / `dpo_baseline_full.yaml` | 第二轮全量 1 epoch（7642 步） |
| `scripts/setup_env.sh` | 装 torch/trl/transformers → 校验 trl 来源 → 跑自检与 33 个单测 → 报告 GPU/bf16/磁盘 |
| `scripts/prepare_data.py` | 落盘 `data/uf-20pct`（12000 对）与 `data/uf-full`，两者共用同一批 512 条评测样本 |
| `scripts/run_first_round.sh` | 冒烟 → 测速 → 两组训练 → 汇总，幂等、可续跑，并把配置副本与机器信息写进产物目录 |
| `scripts/run_full.sh` | 第二轮全量 |
| `tests/test_tools.py` | 28 个自检用例（无 GPU 可跑）：配置、四个工具、续训解析、脚本与文件卫生 |
| `requirements.txt` / `LICENSE` | 依赖清单（torch 单独按 CUDA 装）/ Apache-2.0 |
| `tools/check_configs.py` | 0.1 秒内查出配置字段拼错、两组不可比、脚本参数拼错 |
| `tools/compare_runs.py` | 两组曲线并排，并把"能横向比"和"不能横向比"的指标分开 |
| `tools/eval_implicit_reward.py` | 同一把尺子重算不加权的标准 DPO 隐式奖励胜率（可横向比），并量化口径差 |
| `tools/estimate_time.py` | 从测速日志换算出两组/全量的预计耗时 |
| `train_dpo.py` | 新增 `--resume_from_checkpoint auto｜true｜false｜<路径>` |

---

## 2. 与公式的对应（细节见 docs/source/ea_dpo.md）

1. 预测熵 H：复用 TRL fused LM head 的 `outputs.entropy`，与公式 (2) 的 `pi_theta` 同源。
2. 劣质回答折返：每个 rejected completion 独立取有效 Token 熵的 95% 分位 `H_cap`，再 `min(H, max(0, 2H_cap-H))`；chosen 不折返，prompt/padding 不进分位。
3. A 与 f(x)：取目标层过 RoPE 与 QK-norm 后的真实 Q/K/V；先算全上下文注意力概率，跨 query head 算术平均，再裁出 completion-to-completion 子矩阵并逐行归一化；V 按头拼接后过真实 `o_proj` 并去 bias 取 L2 范数。
4. 因果入度 C：`E=A·‖f(x)‖`，行和 ≤ `delta_row` 整行归零，仅累加严格下三角 `j>t`，最后一个 completion Token 的 C 恒为 0。
5. 归一化与交互：逐序列在有效 completion Token 上做均值归一，`I = H̃ + κC̃ + γH̃C̃`。
6. 保均值权重：`w = (1-λ) + λ·T·softmax(I/τ)`，每个 completion 的权重均值为 1。
7. 加权损失：策略与冻结参考使用**同一个**停止梯度的 w 分别求和，再算原始 sigmoid DPO。

**最容易出错的 Token 对齐**：输入位置 s 的 Token 由位置 s-1 的 LM 输出预测，因此
`entropy[:, s-1]`、`log_prob[:, s-1]` 与 `C[:, s]` 对齐；代码用 `observer.credit[:, 1:]`。

---

## 3. 上传到服务器

保持相对目录关系，只传这些（`outputs/`、`logs/`、`data/`、`.venv*`、`__pycache__/`、
两个 `.zip` 原始备份都不必传）：

```text
你的实验目录/
├── train_dpo.py
├── configs/
├── scripts/
├── tools/
└── trl-1.18.1/
```

`trl-v1.18.1.zip` 是官方原始备份，**不包含**本次修改；`EA-DPO-TRL-v1.18.1.zip` 是上一版打包件，
也不含本次流程改动。不要只传 ZIP。

如果通过 Windows 工具中转导致换行变成 CRLF，Linux 上会报 `$'\r': command not found`，修一行即可：

```bash
sed -i 's/\r$//' scripts/*.sh
```

---

## 4. 上机步骤

```bash
# 0) 进目录
cd /root/EA-DPO

# 1) 环境（幂等，可重复执行）
bash scripts/setup_env.sh
#    它会：检查/安装 torch(>=2.6, cu124) → pip install -e './trl-1.18.1[peft]'
#    'transformers==5.23.0' jinja2 pytest tensorboard → 校验 import trl 指向本仓库
#    → 跑 tests/test_ea_dpo.py（33 个）→ 报告 GPU/bf16/磁盘

# 2) 数据（幂等，可重复执行）
export HF_ENDPOINT=https://hf-mirror.com      # 脚本里已默认设置
python scripts/prepare_data.py                # 约 2-5 分钟，含 3B 之外的 200MB 数据下载
#    --skip-full  只生成第一轮需要的 20% 数据
#    --force      重新生成

# 3) 第一轮：冒烟 → 测速 → 两组训练 → 对比
bash scripts/run_first_round.sh
```

分阶段执行（中途想看一眼再继续时用）：

```bash
bash scripts/run_first_round.sh smoke    # 10 步冒烟，约 3-5 分钟
bash scripts/run_first_round.sh timing   # 40 步测速，给出两轮耗时预估
bash scripts/run_first_round.sh dpo      # 只跑对照组
bash scripts/run_first_round.sh ea       # 只跑 EA-DPO
bash scripts/run_first_round.sh report   # 汇总 + 同口径离线评测
```

学习率对照组（可选，产物写到独立目录，不会覆盖主实验）：

```bash
LR=5.0e-6 TAG=lr5e-6 bash scripts/run_first_round.sh
# 主实验 lr=1.0e-5；这一组用 5.0e-6 再跑一遍两组，用来确认结论不是学习率造出来的
```

耗时预估（4090 24G，实际以 `timing` 阶段的实测为准）：

| 阶段 | 步数 | 预计 |
|---|---|---|
| 冒烟（两组） | 10 + 10 | < 10 分钟（含首次下载 3B 模型约 6 GB） |
| 第一轮一组 | 1500 | 3.5-4.5 小时（+ 6 次评测约 20 分钟） |
| 第一轮两组 | 3000 | 7-9 小时 |
| 同口径离线评测 | 256 对 × 2 adapter | 6-8 分钟（`report` 阶段自动跑） |
| 第二轮一组 | 7642 | 18-22 小时 |

---

## 5. 断点续训

```bash
python train_dpo.py --config configs/ea_dpo.yaml --resume_from_checkpoint auto
python train_dpo.py --config configs/ea_dpo.yaml --resume_from_checkpoint outputs/ea-3b-20pct/checkpoint-750
python train_dpo.py --config configs/ea_dpo.yaml --resume_from_checkpoint false   # 明确从头
```

`auto` = 自动挑 `output_dir` 下步数最大的 `checkpoint-*`；没有存档时等价于从头开始。
两个 shell 脚本里已经默认带 `auto`，所以**训练中断后直接重跑同一个脚本即可**。
脚本还会跳过已经跑完的那一组（判据：`output_dir/adapter_config.json` 存在）。

---

## 6. 跑之前/跑的时候的自检

```bash
python -m pytest tests/ -q            # 28 个用例，不需要 GPU/transformers，约 1 秒
python tools/check_configs.py        # 0.1 秒：配置字段、两组可比性、脚本参数
python tools/report.py outputs/ea-3b-20pct
python tools/compare_runs.py outputs/dpo-3b-20pct outputs/ea-3b-20pct --labels DPO EA-DPO
python tools/eval_implicit_reward.py --adapter outputs/dpo-3b-20pct:DPO \
                                    --adapter outputs/ea-3b-20pct:EA-DPO --limit 256
```

`tests/test_tools.py` 覆盖：配置合法性与可比性、四个工具在合成数据上的行为、续训参数解析、
shell 脚本结构（shebang / `set -euo` / `if`-`fi` 配平 / `fi` 前必须有 `;`）、全仓库文本文件的
UTF-8 与 LF 行尾、README 里的路径引用是否真实存在、根目录是否有 LICENSE 与 requirements.txt。
改完任何配置或脚本先跑它，比等模型下载完再报错省时间。

`check_configs.py` 校验四件事：

1. YAML 里每个键都是 `DPOConfig / _BaseConfig / ModelConfig / ScriptArguments` 的真实字段
   （`TrlParser` 遇到未知键会拒绝启动，但那是模型开始下载之后才发现）；
2. 同一轮的对照组与 EA 组**除 `ea_*` 和 `output_dir` 外逐字段相同**（可比性的前提）；
3. 20% 轮与全量轮的模型/优化器/LoRA/长度设置一致；
4. 脚本里传给 `train_dpo.py` 的每个 `--flag` 都真实存在。

冒烟阶段会自动核对三件事（都从 `trainer_state.json` 读，不依赖控制台打印了什么）：

- `EA-DPO enabled at zero-based layer 28` —— 3B 共 36 层，默认目标层是第 29 层；
- `ea/weight_mean_rejected == 1` —— 公式 (10) 的保均值约束，偏离 1 说明归一化有问题，别往下跑；
- 对照组日志里**没有任何 `ea/` 指标** —— 确认 `ea_dpo_enabled: false` 真的生效了。

冒烟之所以写 checkpoint（配 `--save_only_model true`，每个存档只有 adapter），就是因为
`ea/weight_mean_*` 只在 `trainer_state.json` 里是精确值，控制台日志不一定打全。
冒烟产物在 `outputs/smoke-*`，跑完可以直接删。

---

## 7. 数据量与训练长度

ultrafeedback_binarized 全量约 61135 对偏好数据，`per_device_train_batch_size 1 ×
gradient_accumulation_steps 8` 下 **7642 步 = 1 epoch**。第一轮的 `data/uf-20pct` 取前 12000 对，
1500 步正好一轮。

三种改法都只动命令行，不动代码：

```bash
--max_steps 3000                  # 约 40% 数据
STEPS=750 bash scripts/run_first_round.sh
--num_train_epochs 0.25           # 直接按比例
```

**不建议超过 2 epoch**：DPO 过拟合后常见回答长度暴涨与 reward hacking，3B 上尤其明显。

---

## 8. 训练监控与异常判据

控制台每 5 步打印一次；`trainer_state.json` 每 250 步落盘一次（`save_strategy: no` 时**没有**这个文件）。

| 指标 | 健康 | 异常信号 |
|---|---|---|
| `loss` | 0.693 起，缓降到 0.3-0.5 | 前 100 步就 <0.2：lr 过大；不降：lr 太小 |
| `grad_norm` | 稳定 < 5 | EA 组长期被打到 1.0 裁剪上限 = 权重方差太大，被 clip 的效果削弱 |
| `entropy` | 缓降 | 骤降到 < 0.5 = 模式坍塌 |
| `ea/weight_mean_*` | 恒等于 1 | ≠1 说明实现有问题 |
| `ea/weight_max_rejected` | 5-15 | 长期 > 20-30，下一轮把 `ea_tau_rejected` 从 0.7 调到 0.85 |
| `ea/target_layer` | 28（3B） | 不是 28 说明模型换错了 |
| 平均生成长度 | 温和增长 | 翻倍 = 长度 reward hacking，该停 |

想看曲线：`tensorboard --logdir outputs --port 6006`（配置里已 `report_to: tensorboard`；
没装 tensorboard 会被自动跳过，不会让训练失败）。GPU 采样：
`bash tools/watch_gpu.sh 5 ~/gpu.csv`。

---

## 9. 已知边界与未验证项

**实现层面**（细节见 docs/source/ea_dpo.md）：

> 论文没规定、由本实现选定的 5 处约定，以及它们各自的判定方法，集中记在
> [待验证清单.md](待验证清单.md)。出问题先看那个文件。

- 仅纯文本 Llama / Qwen2（含 2.5）/ Qwen3 的原生全注意力层；`eager`、`sdpa` 后端。
  包装器保留 FlashAttention 2/3 分支，但 CUDA 路径未验证。
- 注意力统计按 query block 累积，峰值 workspace 约 `O(H·chunk·S)`；**额外计算仍是平方级**，
  不能表述为零开销或线性复杂度。
- 不支持 FSDP / DeepSpeed / 张量·上下文并行 / padding-free / torch.compile / 虚拟 Token
  prompt tuning / VLM / 滑动窗口目标层 —— 这些在 `DPOConfig.__post_init__` 与 `DPOTrainer.__init__`
  里前置报错，不会静默错位。
- 必须固定参考模型 + sigmoid loss + reverse KL + 无 label smoothing；不能用
  `precompute_ref_log_probs`（动态 Token 权重不能从两个预先求和的标量里恢复）。

**这一版在本机能验证到什么程度**（Windows、无 GPU、无 transformers）：| 已验证 | 方式 |
|---|---|
| 所有 Python 文件语法 | `py_compile` |
| 配置字段、两组可比性、脚本参数 | `tools/check_configs.py`（并用 4 个反例确认它真能报错） |
| `tools/report.py`、`tools/compare_runs.py` | 用合成的 `trainer_state.json` 实跑，含缺失文件、无评测点等分支 |
| `tools/estimate_time.py` | 用 4 种日志格式（标准块 / 仅 runtime / JSON 风格 / 无信息）实跑 |
| `tools/eval_implicit_reward.py` 的纯逻辑 | 27 项单测：路径解析、对话格式归一化、chat template 前缀一致性、keep_start 截断、`logits_to_keep` 生效/不生效两种索引、胜率与长度统计 |
| `tools/check_configs.py` 的两条校验路径 | 用假模块模拟"已装 transformers"的运行时分支与"全局 TRL"回退分支 |
| `train_dpo.py` 的续训解析 | 直接 import 后单测 `auto/true/false/路径/缺值/空目录` 六种情况 |
| shell 脚本 | 结构检查（`if/fi`、`case/esac`、`do/done` 配平、`fi` 前必须有 `;`、引号配对、heredoc 闭合、LF 行尾、无 BOM） |

`check_configs.py` 在装了 transformers 的机器上会**直接用运行时字段表**校验（这是权威的），
本机没装时才退回源码解析。所以服务器上 `run_first_round.sh` 的自检阶段就能确认
`warmup_ratio`、`save_only_model` 这类 TrainingArguments 字段真的存在，不会跑到一半才发现。

| 未验证（必须上机确认） | 在哪一步暴露 |
|---|---|
| 真实显存占用、吞吐 | `run_first_round.sh timing` |
| `transformers==5.23.0` + torch 2.x 的真实安装 | `scripts/setup_env.sh` |
| 33 个单测在服务器环境通过 | `scripts/setup_env.sh` 第 4 步 |
| TensorBoard 曲线、多卡、其他模型 | 第一轮跑完之后 |

---

## 10. 常见报错

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: 'trl'` 或 `import trl` 指向 site-packages | 没装可编辑版 | `python -m pip install -e './trl-1.18.1[peft]'` |
| `ImportError: apply_chat_template requires jinja2>=3.1.6` | transformers 5.x 只在运行时检查这个下限 | `python -m pip install -U 'jinja2>=3.1.6'` |
| 下载模型/数据卡住或 401 | 国内直连 HuggingFace | `export HF_ENDPOINT=https://hf-mirror.com` |
| `$'\r': command not found` | 文件被中转成了 CRLF | `sed -i 's/\r$//' scripts/*.sh` |
| `CUDA out of memory` | 显存不足或已有进程占卡 | `nvidia-smi` 看占用；把 `max_length` 降到 768；确认 batch 仍是 1 |
| `Dataset.map`/`filter` 多进程报错 | 容器 /dev/shm 或进程数限制 | `--dataset_num_proc 1` |
| `EA-DPO requires a full-attention target layer` | 模型不是 Qwen2.5 全注意力 / 换了架构 | 换回 `Qwen/Qwen2.5-3B-Instruct` |
| `EA-DPO requires padding_free=False, torch_compile=False...` | 配置冲突 | 保持配置里的默认值，不要加 `--padding_free` / `--torch_compile` |
| 权重均值不等于 1 | 实现被改动 | 别继续长训练，先跑单测确认 |

---

## 11. 第一轮之后的消融计划（按这个顺序）

第一轮只回答"两种方法的曲线是否分叉"。要下结论，按优先级补：

1. **同口径离线评测**：`python tools/eval_implicit_reward.py --adapter outputs/dpo-3b-20pct:DPO
   --adapter outputs/ea-3b-20pct:EA-DPO`。它用不加权的标准 DPO 隐式奖励作分母，两组可以横向比，
   也可以顺手换 checkpoint 看随步数的变化。
2. **生成质量**：512 条固定 prompt、两组 adapter 各生成一次，裁判 pairwise，同时记录长度。
   这是"有没有用"的最终证据。
3. **种子**：对照组换 seed=43 再跑一遍，看方法差异是否大于跑次方差
   （512 对样本上胜率的抽样标准差约 2.2%，两组差距小于它时先别下结论）。
4. **学习率对照**：`LR=5.0e-6 TAG=lr5e-6 bash scripts/run_first_round.sh`
   （产物写到 `outputs/*-lr5e-6`，不会覆盖主实验）。用来确认结论不是学习率造出来的。
5. **λ**：`--ea_lambda 0.3 / 0.8`，配合 `--ea_lambda 0`（应严格退化为标准 DPO）确认实现无偏。
6. **γ**：`--ea_gamma 0` 去掉交互项。当前 κ=γ=1 且 H̃、C̃ 都归一化到均值 1，交互项 `H̃C̃`
   量纲上是平方，通常是三项里最大的一项 —— 不做这一组就无法归因。
7. **τ_rejected**：0.7 → 0.85，配合 `ea/weight_max_rejected` 读数。

注意：τ_chosen=1.0 且 κ_chosen=1.0 时，chosen 侧的 softmax 接近均匀，权重几乎贴着 1 走。
也就是说当前这组参数实际上等价于"**只对 rejected 的 completion Token 做集中加权**"，
写结论时应当这样表述，而不是"两侧都加权"。

**关于"训练日志里的 win_rate 能不能比"**：两者是**同一种量**（都是隐式奖励之差，
`β·Σ(权重)·log(π/π_ref)`），形式、量纲、尺度都一致，方向上完全可以并排看 —— 论文里比较
DPO 变体也是这么做的。真正要小心的是**尺子**：EA 的权重 `w_t` 由样本自身的熵和注意力入度
算出，等于尺子随模型变，严格说 EA 的数是 `f_EA(π)`、基线的是 `f_dpo(π)`，不是同一个函数。
这个口径差一般只有 1-3 个百分点，但你要证明的效应量也常在 1-2 个百分点。

所以跨组比较时用第 1 条的离线脚本统一尺子。它会把每组"日志口径 vs 不加权口径"的差
直接打出来 —— **先量一量口径差有多大**：小到可忽略就直接看日志，大到和效应同量级就必须
用统一尺子的那一列。`report` 阶段会自动跑它。
