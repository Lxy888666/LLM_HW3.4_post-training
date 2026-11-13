"""Cross-check the report against the pinned sources and recorded run artifacts."""
import ast
import hashlib
import json
import math
import re
import struct
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from analyze_optimized import main as audit_improved


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT.parent / "LLM强化学习后训练代码分析与运行报告.md"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def close(left, right):
    assert math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-7), (left, right)


def triple(item):
    return item["a"], item["b"], item["c"]


def reward(text, answer):
    matched = re.fullmatch(r"\s*([+-]?\d+)\s*", text)
    return float(matched is not None and int(matched.group(1)) == answer)


def main():
    report = REPORT.read_text(encoding="utf-8")
    assert "\ufffd" not in report
    assert not re.search(r"\?{3,}", report)
    assert report.count("```python\n") == 2
    fences = re.findall(r"^```.*$", report, re.MULTILINE)
    assert len(fences) % 2 == 0
    math_delimiters = re.findall(r"^\$\$$", report, re.MULTILINE)
    assert len(math_delimiters) == 12
    assert not re.search(r"^#{1,6}\s*$", report, re.MULTILINE)

    body, references = report.split("## 参考文献", 1)
    references = references.split("## 附录", 1)[0]
    entries = {int(n) for n in re.findall(r"^(\d+)\. ", references, re.MULTILINE)}
    clean_body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    clean_body = re.sub(r"`[^`]*`", "", clean_body)
    clean_body = re.sub(r"\$\$.*?\$\$", "", clean_body, flags=re.DOTALL)
    cited = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", clean_body):
        cited.update(int(number) for number in re.findall(r"\d+", group))
    assert entries == set(range(1, 12)), entries
    assert cited == entries, {"cited": sorted(cited), "entries": sorted(entries)}

    core = read_json(ROOT / "results/core_verification.json")
    assert core["status"] == "all_passed" and core["probe_count"] == len(core["results"]) == 8
    for relative, digest in core["source"]["files"].items():
        assert hashlib.sha256((ROOT / "sources" / relative).read_bytes()).hexdigest() == digest
    snapshot = read_json(ROOT / "results/upstream_source_manifest.json")
    assert snapshot["revisions"] == core["source"]["revisions"]
    for relative, digest in snapshot["files"].items():
        assert hashlib.sha256((ROOT / "sources" / relative).read_bytes()).hexdigest() == digest

    source_index = read_json(ROOT / "results/source_index.json")
    for item in source_index:
        path = ROOT / "sources" / item["project"] / item["file"]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert any(
            isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name == item["symbol"]
            and node.lineno == item["line"]
            and node.end_lineno == item["end_line"]
            for node in ast.walk(tree)
        ), item
    table_locations = re.findall(r"^\| (OpenRLHF|verl) \| `([^`]+):(\d+)`", report, re.MULTILINE)
    for project, relative, line in table_locations:
        assert any(item["project"] == project and item["file"] == relative
                   and item["line"] == int(line) for item in source_index), (project, relative, line)
    excerpts = re.findall(r"```python\n(.*?)\n```", report, re.DOTALL)
    excerpt_locations = [("OpenRLHF/openrlhf/models/loss.py", 92, 95),
                         ("verl/verl/trainer/ppo/core_algos.py", 301, 306)]
    for excerpt, (relative, begin, end) in zip(excerpts, excerpt_locations):
        lines = (ROOT / "sources" / relative).read_text(encoding="utf-8").splitlines()
        assert textwrap.dedent("\n".join(lines[begin - 1:end])).strip() == excerpt.strip()

    directory = ROOT / "results/grpo_run"
    config = read_json(directory / "config.json")
    summary = read_json(directory / "summary.json")
    metrics = read_json(directory / "analysis_metrics.json")
    dataset = read_json(directory / "dataset.json")
    before = read_json(directory / "evaluation_before.json")
    after = read_json(directory / "evaluation_after.json")
    checkpoint = read_json(directory / "checkpoint_verification.json")
    logs = read_jsonl(directory / "training.jsonl")
    rollouts = read_jsonl(directory / "rollouts.jsonl")
    assert config["source"] == core["source"]
    assert len(logs) == summary["optimizer_steps"] == 48
    assert len(rollouts) == summary["rollout_completions"] == 192
    assert len(dataset["train"]) == 48 and len(dataset["test"]) == 32
    train_ids = {triple(x) for x in dataset["train"]}
    test_ids = {triple(x) for x in dataset["test"]}
    assert len(train_ids) == 48 and len(test_ids) == 32 and not train_ids & test_ids
    assert all(x["answer"] == x["a"] * x["b"] + x["c"]
               and 10 <= x["a"] <= 49 and 2 <= x["b"] <= 9 and 1 <= x["c"] <= 9
               for x in dataset["train"] + dataset["test"])

    grouped = defaultdict(list)
    for row in rollouts:
        close(row["reward"], reward(row["response"], row["answer"]))
        step = row["step"]
        assert triple(row) in {triple(x) for x in dataset["train"][(step-1)*2:step*2]}
        grouped[(step, triple(row))].append(row["reward"])
    distribution = Counter()
    for rewards in grouped.values():
        assert len(rewards) == 4
        label = "mixed" if min(rewards) != max(rewards) else "all_correct" if rewards[0] == 1 else "all_wrong"
        distribution[label] += 1
    assert dict(distribution) == metrics["group_distribution"]
    assert distribution["mixed"] == summary["informative_groups"] == 5
    assert sum(distribution.values()) == summary["total_groups"] == 48
    close(sum(x["reward"] for x in rollouts), metrics["correct_rollouts"])
    assert sum(x["grad_norm_before_clipping"] > 0 for x in logs) == summary["nonzero_gradient_steps"] == 34
    for index, row in enumerate(logs):
        assert row["optimizer_step"] == index + 1
        step = row["rollout_step"]
        assert step == index // 2 + 1 and row["update_in_rollout"] == index % 2 + 1
        assert all(math.isfinite(value) for value in row.values() if isinstance(value, (float, int)))
        close(row["total_loss"], row["policy_loss"] + config["kl_coefficient"] * row["reference_kl"])
        close(row["mean_reward"], sum(x["reward"] for x in rollouts if x["step"] == step) / 8)
        assert row["informative_groups"] == sum(
            min(values) != max(values) for (s, _), values in grouped.items() if s == step)

    for label, evaluation in [("before", before), ("after", after)]:
        assert [triple(row) for row in evaluation["samples"]] == [triple(row) for row in dataset["test"]]
        n = evaluation["n"]
        correct = sum(reward(row["response"], row["answer"]) for row in evaluation["samples"])
        assert n == len(evaluation["samples"]) == 32
        assert correct == evaluation["correct"] == summary[label + "_correct"]
        close(correct / n, evaluation["accuracy"])
        close(evaluation["accuracy"], summary[label + "_accuracy"])
        close(sum(row["response_tokens"] for row in evaluation["samples"]) / n,
              evaluation["mean_response_tokens"])
        close(evaluation["mean_response_tokens"], metrics[label + "_mean_response_tokens"])
        assert all(bool(reward(row["response"], row["answer"])) == row["correct"]
                   and re.fullmatch(r"\s*([+-]?\d+)\s*", row["response"]) is not None
                   and row["format_valid"] for row in evaluation["samples"])
    repairs = sum(not x["correct"] and y["correct"] for x, y in zip(before["samples"], after["samples"]))
    breaks = sum(x["correct"] and not y["correct"] for x, y in zip(before["samples"], after["samples"]))
    assert repairs == metrics["repair_count"] == 3 and breaks == metrics["break_count"] == 0
    assert checkpoint == {"status": "all_passed", "matching_answers": 32,
                          "test_size": 32, "correct": 6, "accuracy": .1875}
    adapter = directory / "adapter/adapter_model.safetensors"
    with adapter.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
    saved_parameters = sum(math.prod(value["shape"]) for key, value in header.items() if key != "__metadata__")
    assert saved_parameters == config["trainable_parameters"] == 540672
    adapter_config = read_json(directory / "adapter/adapter_config.json")
    assert adapter_config["base_model_name_or_path"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert 0 < summary["changed_parameter_elements"] <= saved_parameters
    assert summary["completed"] and summary["parameter_delta_l2"] > 0
    base_manifest = read_json(ROOT / "results/base_model_manifest.json")
    assert base_manifest["revision"] == config["base_model_revision"]
    assert len(base_manifest["files"]) == 7

    for value in ["540,672", "494,573,440", "540,671", "9.375%", "18.750%",
                  "3.5625", "3.59375", "0.5679", "39 组全部错误", "4 组全部正确"]:
        assert value in report, value
    assert "32 条贪心输出均与更新后记录一致" in report
    audit_improved(quiet=True)
    improved = read_json(ROOT / "results/grpo_improved/summary.json")
    reloaded = read_json(ROOT / "results/grpo_improved/checkpoint_verification.json")
    assert reloaded["status"] == "all_passed" and reloaded["matching_answers"] == 32
    assert reloaded["correct"] == 28 and reloaded["fresh_test"]["matching_answers"] == 256
    assert reloaded["fresh_test"]["correct"] == 221
    for value in ["28/32", "221/256", "220/256", "86.328125%", "8,798,208",
                  "sft_epoch_4", "210 组全部正确", "40 组奖励有差异"]:
        assert value in report, value
    tables = sum(bool(re.match(r"^\|[-: ]+\|", line)) for line in report.splitlines())
    result = {
        "status": "all_passed",
        "report_file": REPORT.name,
        "report_sha256": hashlib.sha256(REPORT.read_bytes()).hexdigest(),
        "reference_entries": len(entries),
        "references_cited": sorted(cited),
        "source_symbols_checked": len(source_index),
        "source_table_locations_checked": len(table_locations),
        "exact_source_excerpts_checked": len(excerpts),
        "source_snapshot_files_checked": len(snapshot["files"]),
        "markdown_tables": tables,
        "display_equations": len(math_delimiters) // 2,
        "core_probe_groups": core["probe_count"],
        "optimizer_calls": len(logs),
        "nonzero_gradient_calls": summary["nonzero_gradient_steps"],
        "rollout_completions": len(rollouts),
        "group_distribution": dict(distribution),
        "before_correct": before["correct"],
        "after_correct": after["correct"],
        "checkpoint_matching_answers": checkpoint["matching_answers"],
        "saved_adapter_parameters": saved_parameters,
        "improved_selected_stage": improved["selected_stage"],
        "improved_test_correct": reloaded["correct"],
        "improved_fresh_test_correct": reloaded["fresh_test"]["correct"],
    }
    (ROOT / "results/submission_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
