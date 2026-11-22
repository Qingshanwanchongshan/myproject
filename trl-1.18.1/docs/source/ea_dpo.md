# EA-DPO 本地实现说明

依据用户提供的《准备跑实验.pdf》（2026-11-18）公式 (2)-(12)，在官方 TRL v1.18.1 上增加可切换的 EA-DPO。研究假设的实际效果需要后续实验验证；下列默认超参数是实验起点。

## 修改范围

| 文件 | 修改及用途 |
|---|---|
| `trl/trainer/ea_dpo.py` | 新增因果信用、范数门控、折返熵、均值归一化、非对称权重，以及单层注意力采集 |
| `trl/trainer/dpo_trainer.py` | 在策略前向采集统计；用相同、停止梯度的权重加权策略/参考逐 Token log-prob；增加权重日志 |
| `trl/trainer/dpo_config.py` | 新增 EA-DPO 参数及组合校验；默认关闭，保留标准 DPO |
| `tests/test_ea_dpo.py` | 无需下载权重的数值、小模型、损失/梯度、LoRA、检查点和保存测试 |
| `docs/source/paper_index.md` | 登记本地、未发表的方法及文档入口 |
| 项目根目录 `train_dpo.py` | 优先加载旁边的修改版 TRL；未知参数直接报错，避免参数拼写错误被忽略 |
| 项目根目录 `configs/*.yaml` | 同模型、数据、优化器和随机种子的标准 DPO / EA-DPO 冒烟配置 |

## 公式与代码的对应

1. **预测熵 H，公式 (2)**：复用 TRL fused LM head 的 `outputs.entropy`，它来自当前策略最终输出的词表分布，与公式中的 `pi_theta` 一致。目标深层用于注意力与 value 统计，不把中间隐藏状态伪装成预测分布。
2. **劣质回答折返，公式 (3)**：每个 rejected completion 独立计算有效 Token 熵的 95% 分位数（线性插值，可配置），再使用 `min(H, max(0, 2*H_cap-H))`。Chosen 熵不做折返。Prompt、Padding 不进入分位数。
3. **A 与 f(x)，公式 (4)**：取目标层经过 RoPE、以及模型自带 Q/K normalization 后的实际 Q/K/V。先计算原始全上下文注意力概率，跨 query heads 算术平均，再裁出 Completion-to-Completion 子矩阵并逐行归一化。GQA 按 query-head 分组重复 KV heads。将未被注意力混合的 V 按头拼接后送入实际 `o_proj`，减去投影 bias，得到 `f(x)=vW^O` 的 L2 范数；包含当前 LoRA 对该投影的修改。
4. **有效因果入度 C，公式 (5)-(6)**：`E[j,t]=A[j,t]*norm(f(x_t))`，包括对角线在内计算行和；行和不超过 `delta_row` 时整行归零。之后仅累加严格下三角 `j>t`，保证最后一个 completion token 的 C 为 0。
5. **归一化与交互，公式 (7)-(8)**：逐序列、仅在有效 completion tokens 上计算 `H/(mean(H)+eps)` 与 `C/(mean(C)+eps)`，再算 `I=H+kappa*C+gamma*H*C`。全零信号保持零。加入 epsilon 后均值近似为 1，这是原公式的数值性质。
6. **非对称保均值权重，公式 (9)-(10)**：chosen/rejected 分别使用温度与 kappa；`w=(1-lambda)+lambda*T*softmax(I/tau)`，每个 completion 的权重均值为 1。lambda=1 时有限精度下权重可以为 0。权重总和保持 T，不意味着任意模型上的实际梯度范数一定与标准 DPO 相同。
7. **加权奖励与损失，公式 (11)-(12)**：对当前策略及冻结参考的逐 Token log-prob 使用**同一个**停止梯度的 w，分别求和后计算原始 sigmoid DPO。参考模型不反传，熵/注意力/权重构建不反传；梯度只通过策略 log-prob 回传。

## 最容易出错的 Token 对齐

输入位置 s 的 Token 由位置 s-1 的 LM 输出预测。因此 `entropy[:, s-1]`、`log_prob[:, s-1]` 与注意力 key 位置 s 的 `C[:, s]` 对齐。代码使用 `observer.credit[:, 1:]` 与 `completion_mask[:, 1:]`。

EOS 若属于 TRL 处理后的 completion，会作为有效 Token 计入；不额外排除。空 completion、截断后没有有效 completion、首个有效 Token 没有 prompt 前驱，会主动报错。

## PDF 未完全指定、此实现明确采用的约定

- 多头聚合：先平均全上下文 attention probabilities，再裁出并归一化 completion 子矩阵；不是分别在每个 head 内先去掉 Prompt 后再平均，两者一般不相等。
- 多头 value 变换：按 query-head 顺序拼接（GQA 重复）后，使用完整 W^O；不是对每头范数求平均，也不是对混合后的 attention output 求范数。
- 目标层：默认 `ceil(0.8 * num_layers) - 1`，零起始编号。例如 24 层模型默认使用索引 19（第 20 层）。可显式指定。
- H_cap：每条 rejected completion 的有效熵分位数，不是整个 batch 或数据集的分位数。

## 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `ea_dpo_enabled` | false | EA-DPO 开关 |
| `ea_lambda` | 0.5 | 重分配强度；保留全部 token 时，0 退化为标准 DPO |
| `ea_top_fraction` | 1.0 | 每条回答按 EA 信息分数保留的 token 比例 |
| `ea_tau_chosen` | 1.0 | chosen 温度 |
| `ea_tau_rejected` | 0.7 | rejected 温度，必须不大于 chosen |
| `ea_kappa_chosen` | 1.0 | chosen 因果主效应系数 |
| `ea_kappa_rejected` | 1.0 | rejected 系数，必须不小于 chosen |
| `ea_gamma` | 1.0 | 交互项系数，非负 |
| `ea_entropy_cap_quantile` | 0.95 | rejected 的 H_cap 分位数 |
| `ea_target_layer` | 自动 | 目标层的零起始索引 |
| `ea_attention_chunk_size` | 128 | 每块计算多少个 completion queries |
| `ea_row_delta` | 1e-8 | 休眠行阈值 |
| `ea_epsilon` | 1e-8 | 均值归一化常数 |

`ea/weight_mean_chosen` 和 `ea/weight_mean_rejected` 应接近 1；`ea/weight_max_*` 记录每序列最大权重的跨样本均值；`ea/target_layer` 记录实际层号。原有奖励和 log-prob 日志在启用 EA-DPO 时对应加权后的数值。

设置 `ea_top_fraction: 0.3` 可运行稀疏加权变体。每条 chosen/rejected 回答独立保留信息分数最高的
`ceil(0.3 * T)` 个有效 completion token，同分按输入位置稳定排序。其余位置的直接 log-prob
损失权重为零，完整回答仍作为上下文参与前向计算。选中权重重新缩放至总和 T，策略和参考使用同一组
停止梯度的权重。保留比例为 1 时沿用完整 EA-DPO，且 lambda=0 时严格退化为标准 DPO。
保留比例小于 1 时，lambda=0 仍会执行稀疏选择。日志中的 `ea/selected_fraction_*` 记录实际比例，
短回答的比例会因向上取整而超过设定值。这一变体仍计算完整序列的熵与注意力统计。

## 支持范围与实际计算代价

- 已适配纯文本 Llama、Qwen2（包含 Qwen2.5）、Qwen3 的原生 Transformers 全注意力层。
- 原模型可以使用 `eager`、`sdpa`；包装器也保留 FlashAttention 2/3 后端，但本机 CPU 测试未验证 FlashAttention CUDA 路径。模型注意力结果仍由所选原后端计算。
- 统计按 query block 累积，避免存储整个 batch 的 `B*H*S*S` 矩阵。每样本额外注意力计算仍约为 `O(H*T_completion*S*head_dim)`，峰值 attention workspace 约 `O(H*chunk_size*S)`。另有 value 输出投影成本。**这是精确的分块统计实现，不是已经证明低耗时的融合 FlashAttention kernel，也不能称为线性时间实现。**
- 单卡全参数与 LoRA、非 reentrant/reentrant 梯度检查点支持；普通 DDP 代码路径保留，但多 GPU 尚需服务器验证。
- 当前不支持 FSDP、DeepSpeed、张量/上下文并行、padding-free、torch.compile、prompt/prefix 虚拟 Token、VLM、非原生 attention 替换、滑动窗口目标层。主动报错避免统计错位。
- EA-DPO 必须使用固定 reference、标准 sigmoid/reverse-KL，不能叠加 LD-DPO、WPO、混合 loss 或 label smoothing。不能启用序列级 `precompute_ref_log_probs`：动态 Token 权重无法从两个预先求和的参考标量恢复。
- QLoRA 投影沿用实际模块调用，但量化精度和 CUDA/bitsandbytes 路径未在本机验证。

## 测试

从本仓库目录执行：

```bash
python -m pip install pytest
python -m pytest tests/test_ea_dpo.py -q
```

测试使用真正的随机初始化微型 Llama/Qwen2/Qwen3、实际 tokenizer、对话偏好数据和 DPOTrainer，无需下载预训练权重。对照公式的完整矩阵、手工 log-softmax loss 和梯度，并检查 lambda=0 与标准 DPO 的严格一致性。训练测试包含 LoRA、梯度检查点以及保存。CPU 单元测试不能替代服务器上的 GPU 数值、显存、吞吐和正式效果评估。

## 官方接口依据

统计入口通过 Transformers 的 [AttentionInterface](https://huggingface.co/docs/transformers/main/en/attention_interface) 取得后端收到的 Q/K/V；仅在当前 policy forward 临时修改目标 attention module 的 config，并在异常时同样恢复。模型顶层 config 不被写入自定义后端名称，因此保存文件不依赖临时注册名。
