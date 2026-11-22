#!/usr/bin/env python3
"""静态检查配置与训练命令，不需要 GPU、不需要 transformers。

它回答三个问题：

1. YAML 里的每个键，真的是 TRL 的 DPOConfig / ModelConfig / ScriptArguments 的字段吗？
   （`TrlParser` 遇到未知键会直接报错，但那是训练启动几十秒后、模型都开始下载了才发现。
   这里改成 0.1 秒内报错。）
2. 两组配置除 `ea_*`（和 output_dir）之外，是否逐字段相同？
   这是"差异只可能来自 EA-DPO"的前提。
3. 脚本里传给 `train_dpo.py` 的每个 `--flag`，是不是真实存在的字段？
   防止 `--eval_stategy` 这类拼写错误在长跑开始后才爆炸。

用法：
    python tools/check_configs.py
    python tools/check_configs.py --quiet     # 只输出结论
"""

import argparse
import ast
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml 是 datasets 的依赖，正常都有
    print("[fail] 缺少 pyyaml：python -m pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# TrainingArguments 的字段没有本地源码可解析（transformers 只在服务器上装），
# 这里列出本项目实际用到的那一小撮，全部是 transformers 长期稳定的字段。
TRAINING_ARGUMENT_FIELDS = {
    "bf16",
    "data_seed",
    "dataloader_num_workers",
    "eval_steps",
    "eval_strategy",
    "gradient_accumulation_steps",
    "gradient_checkpointing_kwargs",
    "hub_model_id",
    "learning_rate",
    "logging_steps",
    "logging_first_step",
    "lr_scheduler_type",
    "max_grad_norm",
    "max_steps",
    "num_train_epochs",
    "optim",
    "output_dir",
    "per_device_eval_batch_size",
    "per_device_train_batch_size",
    "report_to",
    "save_only_model",
    "save_steps",
    "save_strategy",
    "save_total_limit",
    "seed",
    "warmup_steps",
}

# TrlParser 自己消费、不属于任何 dataclass 的参数
PARSER_ONLY_FLAGS = {"config", "resume_from_checkpoint"}

# 同一轮里允许不同的键：EA 开关块 + 输出目录
ALLOWED_PAIR_DIFF = {"output_dir"}

# 跨轮（20% 与全量）必须一致的键：换数据量/步数之外，其余设置不能漂
CROSS_ROUND_SAME = [
    "model_name_or_path",
    "attn_implementation",
    "dtype",
    "bf16",
    "use_peft",
    "lora_r",
    "lora_alpha",
    "lora_target_modules",
    "learning_rate",
    "beta",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "warmup_steps",
    "max_length",
    "per_device_eval_batch_size",
    "gradient_checkpointing",
    "disable_dropout",
    "seed",
]


def fields_of(path: Path, *class_names: str) -> set:
    """收集若干个 dataclass 的注解字段名。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = set(class_names)
    found = set()
    seen = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in wanted:
            seen.add(node.name)
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    found.add(item.target.id)
    missing = wanted - seen
    if missing:
        raise SystemExit(f"[fail] {path} 里找不到类 {sorted(missing)}")
    return found


def load_known_keys(trl_dir: Path) -> tuple:
    """返回 (已知字段集合, 来源说明)。

    优先用运行时导入的 dataclass 字段（服务器上装了 transformers/trl 时最权威）；
    导入不了（本地无依赖）时退回源码解析 + 内置的 TrainingArguments 字段列表。
    只信任**本仓库内**的 trl：官方版 TRL 没有 ea_* 字段，用它校验会误报。
    """
    try:
        import transformers
        import trl

        if Path(trl.__file__).resolve().is_relative_to(trl_dir.resolve()):
            from transformers import TrainingArguments
            from trl import DPOConfig, DatasetMixtureConfig, ModelConfig, ScriptArguments

            keys = set(PARSER_ONLY_FLAGS)
            for cls in (DPOConfig, ModelConfig, ScriptArguments, DatasetMixtureConfig, TrainingArguments):
                keys |= set(cls.__dataclass_fields__)
            return keys, f"运行时导入（trl {trl.__version__} / transformers {transformers.__version__}）"
    except Exception:  # 没装依赖、或本仓库的 trl 不可导入
        pass

    trainer = trl_dir / "trl" / "trainer"
    scripts = trl_dir / "trl" / "scripts" / "utils.py"
    keys = set()
    keys |= fields_of(trainer / "dpo_config.py", "DPOConfig")
    keys |= fields_of(trainer / "base_config.py", "_BaseConfig")
    keys |= fields_of(trainer / "model_config.py", "ModelConfig")
    keys |= fields_of(scripts, "ScriptArguments", "DatasetMixtureConfig")
    keys |= TRAINING_ARGUMENT_FIELDS
    keys |= PARSER_ONLY_FLAGS
    return keys, "源码解析 + 内置 TrainingArguments 字段列表（未安装 transformers）"


def read_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"[fail] {path} 顶层不是映射")
    return data


def check_yaml(configs: list, known: set, quiet: bool) -> list:
    problems = []
    seen_dirs = {}
    for path, cfg in configs:
        unknown = sorted(k for k in cfg if k not in known)
        if unknown:
            problems.append(f"{path}: 未知字段 {unknown}（TrlParser 会拒绝启动）")
        missing = [k for k in ("model_name_or_path", "dataset_name", "output_dir") if k not in cfg]
        if missing:
            problems.append(f"{path}: 缺少必需字段 {missing}")
        out = cfg.get("output_dir")
        if out in seen_dirs:
            problems.append(f"{path}: output_dir={out} 与 {seen_dirs[out]} 重复，两组实验会互相覆盖")
        seen_dirs[out] = path
        if not quiet:
            print(f"  [ok] {path.name:<24} {len(cfg):>2} 个字段全部合法")
    return problems


def check_pairs(pairs: list, quiet: bool) -> list:
    problems = []
    for base_path, base, ea_path, ea in pairs:
        diffs = sorted(
            k for k in set(base) | set(ea) if base.get(k) != ea.get(k) and not k.startswith("ea_") and k not in ALLOWED_PAIR_DIFF
        )
        if diffs:
            detail = ", ".join(f"{k}: {base.get(k)!r} vs {ea.get(k)!r}" for k in diffs)
            problems.append(f"{base_path.name} 与 {ea_path.name} 除了 ea_* 还有差异 -> {detail}")
        elif not quiet:
            print(f"  [ok] {base_path.name} vs {ea_path.name}：只有 ea_* 和 output_dir 不同")

        if ea.get("ea_dpo_enabled") is not True:
            problems.append(f"{ea_path.name}: ea_dpo_enabled 必须是 true")
        if base.get("ea_dpo_enabled") is not False:
            problems.append(f"{base_path.name}: ea_dpo_enabled 必须是 false")
        stray = [k for k in base if k.startswith("ea_") and k != "ea_dpo_enabled"]
        if stray:
            problems.append(f"{base_path.name}: 对照组不该出现 {stray}")
    return problems


def check_cross_round(configs: list, quiet: bool) -> list:
    problems = []
    by_name = {p.name: c for p, c in configs}
    for left, right in [("dpo_baseline.yaml", "dpo_baseline_full.yaml"), ("ea_dpo.yaml", "ea_dpo_full.yaml")]:
        if left not in by_name or right not in by_name:
            continue
        a, b = by_name[left], by_name[right]
        for key in CROSS_ROUND_SAME:
            if a.get(key) != b.get(key):
                problems.append(f"{left} 与 {right} 的 {key} 不一致：{a.get(key)!r} vs {b.get(key)!r}")
    if not problems and not quiet:
        print("  [ok] 20% 轮与全量轮的模型/优化器/LoRA/长度设置一致")
    return problems


def check_scripts(script_dir: Path, known: set, quiet: bool) -> list:
    problems = []
    flag_pattern = re.compile(r"(?<![\w-])--([A-Za-z0-9_]+)")
    assign_pattern = re.compile(r'^([A-Z_][A-Z0-9_]*)="([^"]*)"\s*$', re.MULTILINE)
    for script in sorted(script_dir.glob("*.sh")):
        text = script.read_text(encoding="utf-8")
        assignments = dict(assign_pattern.findall(text))
        # 把反斜杠续行拼成逻辑行，再只看调用了 train_dpo.py 的那些
        logical = text.replace("\\\n", " ").splitlines()
        commands = [line for line in logical if "train_dpo.py" in line]
        if not commands:
            continue
        for index, line in enumerate(commands, 1):
            for flag in flag_pattern.findall(line):
                if flag not in known:
                    problems.append(f"{script.name}（第 {index} 条训练命令）: 未知参数 --{flag}")
        # --config 指向的配置文件必须真实存在。
        # 变量形式要么能在脚本里找到字面量赋值，要么在脚本里有 [ -f "$var" ] 的运行时守卫。
        for value in re.findall(r'--config\s+"?([^"\s]+)"?', " ".join(commands)):
            if not value.startswith("$"):
                if not (script_dir.parent / value).is_file():
                    problems.append(f"{script.name}: --config {value} 不存在")
                continue
            var = value[1:]
            resolved = assignments.get(var, "")
            if resolved:
                if not (script_dir.parent / resolved).is_file():
                    problems.append(f"{script.name}: --config {resolved} 不存在")
            elif not re.search(rf'\[ -f "\${re.escape(var)}" \]', text):
                problems.append(
                    f"{script.name}: --config {value} 既没有字面量赋值，也没有 [ -f \"{value}\" ] 守卫"
                )
        if not quiet:
            print(f"  [ok] {script.name}: {len(commands)} 条训练命令的参数全部合法")
    return problems


def summarize(configs: list) -> None:
    keys = ["max_steps", "num_train_epochs", "learning_rate", "beta", "eval_steps", "save_steps"]
    print("\n关键设置")
    header = f"  {'配置':<24}" + "".join(f"{k:>14}" for k in keys)
    print(header)
    for path, cfg in configs:
        row = f"  {path.name:<24}"
        for key in keys:
            value = cfg.get(key, "-")
            row += f"{value!s:>14}"
        print(row)


def resolve(value, root: Path) -> Path:
    """相对路径按仓库根目录解析；绝对路径原样使用（测试时用临时目录）。"""
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trl-dir", default="trl-1.18.1", help="修改版 TRL 目录")
    parser.add_argument("--configs-dir", default="configs", help="配置目录")
    parser.add_argument("--scripts-dir", default="scripts", help="脚本目录")
    parser.add_argument("--quiet", action="store_true", help="只输出结论")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    trl_dir = resolve(args.trl_dir, root)
    configs_dir = resolve(args.configs_dir, root)
    scripts_dir = resolve(args.scripts_dir, root)
    if not trl_dir.is_dir():
        print(f"[fail] 找不到 {trl_dir}", file=sys.stderr)
        return 1
    if not configs_dir.is_dir():
        print(f"[fail] 找不到 {configs_dir}", file=sys.stderr)
        return 1

    known, source = load_known_keys(trl_dir)
    if not args.quiet:
        print(f"已知字段 {len(known)} 个，来源：{source}")

    paths = sorted(configs_dir.glob("*.yaml"))
    if not paths:
        print(f"[fail] {configs_dir} 里没有 yaml", file=sys.stderr)
        return 1
    configs = [(path, read_config(path)) for path in paths]
    by_name = {path.name: cfg for path, cfg in configs}

    print("\n1. 字段合法性")
    problems = check_yaml(configs, known, args.quiet)

    print("\n2. 同一轮的对照组 vs EA-DPO")
    pairs = []
    for base, ea in [
        ("dpo_baseline.yaml", "ea_dpo.yaml"),
        ("dpo_baseline_full.yaml", "ea_dpo_full.yaml"),
    ]:
        if base in by_name and ea in by_name:
            pairs.append((configs_dir / base, by_name[base], configs_dir / ea, by_name[ea]))
    if not pairs:
        problems.append("找不到成对的配置（dpo_baseline*.yaml / ea_dpo*.yaml）")
    problems += check_pairs(pairs, args.quiet)

    print("\n3. 跨轮一致性")
    problems += check_cross_round(configs, args.quiet)

    print("\n4. 脚本里的训练参数")
    if scripts_dir.is_dir():
        problems += check_scripts(scripts_dir, known, args.quiet)
    else:
        print(f"  [skip] 没有 {scripts_dir}")

    summarize(configs)

    print()
    if problems:
        for line in problems:
            print(f"[fail] {line}")
        print(f"\n{len(problems)} 个问题")
        return 1
    print("[ok] 配置检查全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
