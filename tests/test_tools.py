#!/usr/bin/env python3
"""项目自带的自检：不需要 GPU、不需要 transformers、不下载任何模型。

    python -m pytest tests/ -q          # 推荐
    python -m pytest tests/ -v

覆盖三层：
  1. 静态检查器本身（tools/check_configs.py 能通过、也能抓到真问题）
  2. 四个分析工具在合成数据上的行为（report / compare_runs / estimate_time / eval_implicit_reward）
  3. 训练入口的断点续训解析、shell 脚本的行尾与语法结构
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(ROOT))

CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


def run(*args, cwd=ROOT):
    """跑一个 python 脚本，返回 (退出码, stdout, stderr)。"""
    result = subprocess.run([sys.executable, *map(str, args)], cwd=cwd,
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode, result.stdout, result.stderr


def make_state(path: Path, ea: bool = False, final: int = 500, eval_every: int = 250):
    """造一个假的 trainer_state.json。"""
    history = []
    for step in range(5, final + 1, 5):
        row = {
            "loss": 0.69 - 0.0002 * step,
            "grad_norm": 0.5,
            "learning_rate": 1e-5,
            "entropy": 3.4 - 0.0001 * step,
            "mean_token_accuracy": 0.55,
            "logits/chosen": -1.2,
            "rewards/chosen": 0.01,
            "rewards/rejected": -0.01,
            "rewards/accuracies": 0.6,
            "rewards/margins": 0.02,
            "step": step,
        }
        if ea:
            row["ea/weight_mean_chosen"] = 1.0000001
            row["ea/weight_mean_rejected"] = 1.0
            row["ea/weight_max_rejected"] = 6.5
            row["ea/target_layer"] = 28.0
        history.append(row)
        if step % eval_every == 0:
            history.append({
                "eval_loss": 0.68 - 0.0001 * step,
                "eval_entropy": 3.3,
                "eval_mean_token_accuracy": 0.6,
                "eval_logits/chosen": -1.1,
                "eval_rewards/accuracies": 0.65 + 0.0001 * step,
                "eval_rewards/margins": 0.03,
                "step": step,
            })
    path.mkdir(parents=True, exist_ok=True)
    (path / "trainer_state.json").write_text(
        json.dumps({"log_history": history, "global_step": final, "epoch": 1.0}), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 1. 配置静态检查
# ---------------------------------------------------------------------------
def test_configs_are_valid():
    """四个配置都必须通过 check_configs 的全部检查。"""
    code, out, err = run(TOOLS / "check_configs.py", "--quiet")
    assert code == 0, f"check_configs 失败：\n{out}\n{err}"
    assert "配置检查全部通过" in out


def test_configs_exist_and_parse():
    """配置目录非空，且每份都是合法 YAML 映射。"""
    assert len(CONFIGS) == 4, f"期望 4 份配置，实际 {[p.name for p in CONFIGS]}"
    for path in CONFIGS:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{path.name} 顶层不是映射"
        for key in ("model_name_or_path", "dataset_name", "output_dir"):
            assert key in data, f"{path.name} 缺 {key}"


def test_paired_configs_differ_only_in_ea_suffix():
    """同一轮的两份配置，除 ea_* 与 output_dir 外逐字段相同（可比性前提）。"""
    for base_name, ea_name in [("dpo_baseline.yaml", "ea_dpo.yaml"),
                               ("dpo_baseline_full.yaml", "ea_dpo_full.yaml")]:
        base = yaml.safe_load((ROOT / "configs" / base_name).read_text(encoding="utf-8"))
        ea = yaml.safe_load((ROOT / "configs" / ea_name).read_text(encoding="utf-8"))
        diff = {k for k in set(base) | set(ea)
                if base.get(k) != ea.get(k) and not k.startswith("ea_") and k != "output_dir"}
        assert not diff, f"{base_name} vs {ea_name} 多余的差异：{sorted(diff)}"
        assert ea["ea_dpo_enabled"] is True
        assert base["ea_dpo_enabled"] is False


def test_ea_constraints_hold():
    """论文对 EA 参数的约束：tau_rejected <= tau_chosen，kappa_rejected >= kappa_chosen。"""
    for name in ("ea_dpo.yaml", "ea_dpo_full.yaml"):
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
        assert 0 <= cfg["ea_lambda"] <= 1
        assert 0 < cfg["ea_tau_rejected"] <= cfg["ea_tau_chosen"]
        assert 0 <= cfg["ea_kappa_chosen"] <= cfg["ea_kappa_rejected"]
        assert cfg["ea_gamma"] >= 0
        assert 0 < cfg["ea_entropy_cap_quantile"] <= 1


def test_check_configs_rejects_unknown_key(tmp_path):
    """未知字段必须被拦（TrlParser 要到模型开始下载才报错）。"""
    for path in CONFIGS:
        (tmp_path / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    bad = tmp_path / "dpo_baseline.yaml"
    bad.write_text(bad.read_text(encoding="utf-8") + "\neval_stategy: steps\n", encoding="utf-8")
    code, out, _ = run(TOOLS / "check_configs.py", "--configs-dir", tmp_path, "--quiet")
    assert code != 0
    assert "未知字段" in out


def test_check_configs_rejects_pair_mismatch(tmp_path):
    """对照组偷改一个超参，必须被可比性检查抓到。"""
    for path in CONFIGS:
        (tmp_path / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    ea = tmp_path / "ea_dpo.yaml"
    ea.write_text(ea.read_text(encoding="utf-8").replace("learning_rate: 1.0e-5",
                                                        "learning_rate: 9.9e-5"), encoding="utf-8")
    code, out, _ = run(TOOLS / "check_configs.py", "--configs-dir", tmp_path, "--quiet")
    assert code != 0
    assert "learning_rate" in out


def test_check_configs_rejects_bad_flag(tmp_path):
    """脚本里拼错的参数名必须被拦。"""
    script = tmp_path / "bad.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n"
                      "python train_dpo.py --config configs/ea_dpo.yaml --learning_rat 1e-5\n",
                      encoding="utf-8")
    code, out, _ = run(TOOLS / "check_configs.py", "--scripts-dir", tmp_path, "--quiet")
    assert code != 0
    assert "--learning_rat" in out


# ---------------------------------------------------------------------------
# 2. 分析工具
# ---------------------------------------------------------------------------
def test_report_tool(tmp_path):
    make_state(tmp_path / "run", ea=True)
    code, out, err = run(TOOLS / "report.py", tmp_path / "run")
    assert code == 0, err
    assert "train loss:" in out
    assert "weight_mean stays at 1" in out


def test_report_tool_missing_state(tmp_path):
    code, out, _ = run(TOOLS / "report.py", tmp_path / "nothing")
    assert code == 1
    assert "no trainer_state.json" in out


def test_compare_runs(tmp_path):
    make_state(tmp_path / "dpo", ea=False)
    make_state(tmp_path / "ea", ea=True)
    code, out, err = run(TOOLS / "compare_runs.py", tmp_path / "dpo", tmp_path / "ea",
                         "--labels", "DPO", "EA-DPO", "--csv", tmp_path / "cmp.csv")
    assert code == 0, err
    assert "同口径" in out and "不同口径" in out and "训练健康度" in out
    assert "target_layer 最新 28" in out
    assert (tmp_path / "cmp.csv").is_file()


def test_compare_runs_missing_run_is_clean_error(tmp_path):
    make_state(tmp_path / "dpo", ea=False)
    code, out, err = run(TOOLS / "compare_runs.py", tmp_path / "dpo", tmp_path / "nope")
    assert code == 1
    assert "Traceback" not in err


@pytest.mark.parametrize("fmt,expect", [
    ("***** Train metrics *****\n  train_steps_per_second   =       0.801\n", 0),
    ("***** Train metrics *****\n  train_runtime            =        0:05:12.34\n", 0),
    ("{'train_runtime': 312.4, 'train_steps_per_second': 0.128}\n", 0),
    ("***** Running training *****\n  Num examples = 12,000\n", 1),
])
def test_estimate_time(tmp_path, fmt, expect):
    log = tmp_path / "train.log"
    log.write_text(fmt, encoding="utf-8")
    code, out, _ = run(TOOLS / "estimate_time.py", log)
    assert code == expect
    if expect == 0:
        assert "步/秒" in out


# ---------------------------------------------------------------------------
# 3. eval_implicit_reward 的纯逻辑
# ---------------------------------------------------------------------------
def test_eval_tool_parsing(tmp_path):
    import eval_implicit_reward as ev

    assert ev.parse_adapter("outputs/ea-3b-20pct:EA-DPO") == ("outputs/ea-3b-20pct", "EA-DPO")
    assert ev.parse_adapter("outputs/ea-3b-20pct")[1] == "ea-3b-20pct"
    assert ev.parse_adapter(str(ROOT / "configs"))[0] == str(ROOT / "configs")

    assert ev.as_messages("hi", "user") == [{"role": "user", "content": "hi"}]
    assert ev.as_messages([{"content": "x"}], "assistant") == [{"role": "assistant", "content": "x"}]
    with pytest.raises(TypeError):
        ev.as_messages(123, "user")


class FakeTokenizer:
    """prompt 长度 p，completion 长度 c 的假模板。"""

    def __init__(self, p, c):
        self.p, self.c = p, c

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        prompt = list(range(1, self.p + 1))
        if add_generation_prompt:
            return prompt
        return prompt + list(range(90, 90 + self.c))


EXAMPLE = {"prompt": [{"role": "user", "content": "q"}],
           "chosen": [{"role": "assistant", "content": "a"}],
           "rejected": [{"role": "assistant", "content": "b"}]}


def test_eval_tool_build_pair():
    import eval_implicit_reward as ev

    built = ev.build_pair(FakeTokenizer(5, 3), EXAMPLE, max_length=100)
    assert built is not None and len(built) == 2
    assert built[0][1] == 5, "completion 起点应等于 prompt 长度"
    assert len(built[0][0]) == 8

    assert ev.build_pair(FakeTokenizer(100, 3), EXAMPLE, max_length=100) is None
    assert ev.build_pair(FakeTokenizer(99, 3), EXAMPLE, max_length=100) is not None

    truncated = ev.build_pair(FakeTokenizer(5, 100), EXAMPLE, max_length=20)
    assert truncated is not None and len(truncated[0][0]) == 20
    assert ev.build_pair(FakeTokenizer(20, 5), EXAMPLE, max_length=20) is None


def test_eval_tool_completion_rows():
    """logits_to_keep 生效 / 不生效两种形状下，取到的行数都必须等于 completion 长度。"""
    import eval_implicit_reward as ev

    for logits_len, length, start in [(7, 10, 4), (10, 10, 4), (10, 10, 1), (3, 10, 8), (10, 10, 8)]:
        rows = list(range(logits_len))[ev.completion_rows(logits_len, length, start)]
        assert len(rows) == length - start, (logits_len, length, start, rows)


def test_eval_tool_summarize():
    import eval_implicit_reward as ev

    results = [{"chosen": -10.0, "rejected": -20.0, "chosen_len": 8, "rejected_len": 6},
               {"chosen": -30.0, "rejected": -12.0, "chosen_len": 5, "rejected_len": 9}]
    refs = [{"chosen": -12.0, "rejected": -21.0, "chosen_len": 8, "rejected_len": 6},
            {"chosen": -31.0, "rejected": -13.0, "chosen_len": 5, "rejected_len": 9}]
    summary = ev.summarize(results, refs, beta=0.1)
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["margin_mean"] == pytest.approx(0.05)
    assert summary["chosen_len_mean"] == pytest.approx(6.5)


def test_eval_tool_logged_win_rate(tmp_path):
    import eval_implicit_reward as ev

    empty = tmp_path / "none"
    empty.mkdir()
    assert ev.logged_win_rate(empty) is None

    make_state(tmp_path / "root", final=250)
    assert ev.logged_win_rate(tmp_path / "root") is not None

    ck = tmp_path / "ck"
    make_state(ck / "checkpoint-250", final=250)
    make_state(ck / "checkpoint-500", final=500)
    got = ev.logged_win_rate(ck)
    assert got is not None and got > 0.65, "应取步数最大的 checkpoint"

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "trainer_state.json").write_text("{not json", encoding="utf-8")
    assert ev.logged_win_rate(broken) is None

    mixed = tmp_path / "mixed"
    mixed.mkdir()
    state = mixed / "trainer_state.json"
    state.write_text(json.dumps({"log_history": [
        {"step": 100, "eval_rewards/accuracies": 0.7},
        {"step": 105, "rewards/accuracies": 0.9},
    ]}), encoding="utf-8")
    assert ev.logged_win_rate(mixed) == 0.7
    state.write_text(json.dumps({"log_history": [{"rewards/accuracies": 0.9}]}), encoding="utf-8")
    assert ev.logged_win_rate(mixed) is None


def test_eval_tool_uses_shared_completion_boundary():
    import eval_implicit_reward as ev

    class BoundaryTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
            if add_generation_prompt:
                return [1, 2, 3, 4, 5]
            if messages[-1]["content"] == "a":
                return [1, 2, 3, 4, 5, 90]
            return [1, 2, 3, 4, 99, 91]

    pairs = ev.build_pair(BoundaryTokenizer(), EXAMPLE, max_length=5)
    assert pairs is not None
    assert pairs[0][1] == pairs[1][1] == 4
    assert pairs[0][0] == [1, 2, 3, 4, 5]
    assert pairs[1][0] == [1, 2, 3, 4, 99]


# ---------------------------------------------------------------------------
# 4. 训练入口的续训解析
# ---------------------------------------------------------------------------
def test_resume_argument_parsing(tmp_path):
    import train_dpo

    resume, rest = train_dpo.split_resume_args(["--config", "a.yaml", "--max_steps", "10"])
    assert resume is None and rest == ["--config", "a.yaml", "--max_steps", "10"]

    resume, rest = train_dpo.split_resume_args(["--config", "a.yaml", "--resume_from_checkpoint", "auto"])
    assert resume == "auto" and rest == ["--config", "a.yaml"]

    resume, rest = train_dpo.split_resume_args(["--resume_from_checkpoint=outputs/x/checkpoint-500"])
    assert resume == "outputs/x/checkpoint-500" and rest == []

    with pytest.raises(SystemExit):
        train_dpo.split_resume_args(["--resume_from_checkpoint"])


def test_resume_resolution(tmp_path):
    import train_dpo

    run = tmp_path / "run"
    for step in (250, 1000, 500):
        (run / f"checkpoint-{step}").mkdir(parents=True)
    (run / "checkpoint-tmp").mkdir()

    assert train_dpo.latest_checkpoint(run) == str(run / "checkpoint-1000")
    assert train_dpo.latest_checkpoint(tmp_path / "empty") is None
    assert train_dpo.resolve_resume("auto", run) == str(run / "checkpoint-1000")
    assert train_dpo.resolve_resume("auto", tmp_path / "empty") is None
    assert train_dpo.resolve_resume("true", run) is True
    assert train_dpo.resolve_resume("false", run) is None
    assert train_dpo.resolve_resume(None, run) is None
    assert train_dpo.resolve_resume("outputs/x/checkpoint-5", run) == "outputs/x/checkpoint-5"


# ---------------------------------------------------------------------------
# 5. 脚本与文件卫生
# ---------------------------------------------------------------------------
TEXT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".md", ".txt", ".cff"}
SKIP_DIRS = {"trl-1.18.1", "outputs", "logs", "data", "__pycache__"}


def text_files():
    """仓库内需要保持 UTF-8 / LF 的文本文件（跳过上游 TRL、产物与隐藏目录）。"""
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        parts = path.relative_to(ROOT).parts
        if any(part in SKIP_DIRS or part.startswith(".") for part in parts):
            continue
        yield path


def test_text_files_are_utf8_lf_without_bom():
    """Windows 中转最容易破坏的两件事：CRLF（bash 报 $'\\r'）和 BOM（shebang 失效）。"""
    problems = []
    for path in text_files():
        raw = path.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            problems.append(f"{path.name}: 有 UTF-8 BOM")
        if b"\r\n" in raw:
            problems.append(f"{path.name}: 含 CRLF")
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            problems.append(f"{path.name}: 不是合法 UTF-8 ({exc})")
    assert not problems, problems


def shell_body(text):
    """去掉 heredoc 内容与注释，返回可做结构统计的行。"""
    body, in_heredoc, terminator = [], False, None
    for line in text.splitlines():
        if in_heredoc:
            if line.strip() == terminator:
                in_heredoc = False
            continue
        marker = None
        for token in line.split():
            if token.startswith("<<") and len(token) > 2:
                marker = token.strip("<>'\"").lstrip("-")
                break
        if marker:
            in_heredoc, terminator = True, marker
            line = line.split("<<")[0]
        cut = line.find(" #")
        body.append(line[:cut] if cut > 0 else line)
    return body


def count_word(text, word):
    """按词边界计数，避免 "config" 里的 fi 被算成 fi。"""
    import re

    return len(re.findall(rf"(?<![\w-]){word}(?![\w-])", text))


@pytest.mark.parametrize("script", sorted((ROOT / "scripts").glob("*.sh")), ids=lambda p: p.name)
def test_shell_scripts_structure(script):
    """没有 bash 也能查的几条：shebang、set -euo、if/fi 与 case/esac 配平、fi 前必须有 ;。"""
    text = script.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines and lines[0].startswith("#!"), "缺 shebang"
    assert "set -euo pipefail" in text, "缺 set -euo pipefail"

    body = shell_body(text)
    words = "\n".join(body)
    # 每个 if 恰好对应一个 fi；elif 自带 else 语义、不额外配 fi，所以不能并入 if 计数
    assert count_word(words, "if") == count_word(words, "fi"), "if/fi 不配平"
    assert count_word(words, "case") == count_word(words, "esac"), "case/esac 不配平"
    assert count_word(words, "do") == count_word(words, "done"), "do/done 不配平"

    for number, line in enumerate(body, 1):
        for keyword in ("fi", "done", "esac"):
            match = __import__("re").search(rf"(?<![\w-]){keyword}(?![\w-])", line)
            if match is None:
                continue
            before = line[: match.start()].rstrip()
            if not before:
                continue
            assert before.endswith((";", "&", "|", "}")), f"{script.name}:{number} {keyword} 前缺少 ; -> {line}"


def test_docs_reference_existing_files():
    """README 里提到的仓库内路径都必须真实存在。"""
    import re

    text = (ROOT / "README.md").read_text(encoding="utf-8")
    referenced = set(re.findall(r"\]\(([^)#]+)\)", text)) | set(re.findall(r"`((?:configs|scripts|tools)/[^`\s]+)`", text))
    missing = [ref for ref in referenced
               if not ref.startswith("http") and not (ROOT / ref).exists()]
    assert not missing, f"README 引用了不存在的路径：{missing}"


def test_root_has_license_and_requirements():
    """开源仓库的两个最低要求：许可证与依赖清单。"""
    assert (ROOT / "LICENSE").is_file(), "缺 LICENSE"
    assert "Apache License" in (ROOT / "LICENSE").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "transformers" in requirements and "torch" in requirements
