"""Validate the optimized experiment and derive paired evaluation comparisons."""
import json
import hashlib
import math
import re
from collections import Counter, defaultdict

from source_loader import ROOT


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def triple(item):
    return item["a"], item["b"], item["c"]


def reward(text, answer):
    match = re.fullmatch(r"\s*([+-]?\d+)\s*", text)
    return float(match is not None and int(match.group(1)) == answer)


def comparison(before, after):
    assert [triple(x) for x in before["samples"]] == [triple(x) for x in after["samples"]]
    return {"before_correct": before["correct"], "after_correct": after["correct"],
            "repaired": sum(not x["correct"] and y["correct"]
                            for x,y in zip(before["samples"],after["samples"])),
            "broken": sum(x["correct"] and not y["correct"]
                           for x,y in zip(before["samples"],after["samples"]))}


def main(quiet=False):
    directory = ROOT / "results/grpo_improved"
    config = read_json(directory/"config.json")
    data = read_json(directory/"dataset.json")
    summary = read_json(directory/"summary.json")
    assert summary["completed"]
    assert hashlib.sha256((directory/"dataset.json").read_bytes()).hexdigest() == config["dataset_sha256"]
    sizes={"train":2048,"validation":128,"test":32,"fresh_test":256}
    split_ids={}
    for split, size in sizes.items():
        assert len(data[split]) == size
        split_ids[split]={triple(x) for x in data[split]}
        assert len(split_ids[split]) == size
        assert all(x["answer"] == x["a"]*x["b"]+x["c"] for x in data[split])
    for left in sizes:
        for right in sizes:
            if left != right:
                assert not split_ids[left]&split_ids[right]
    original=read_json(ROOT/"results/grpo_run/dataset.json")
    assert data["test"] == original["test"]
    evaluations={}
    for stage in ("base","sft","grpo_final","selected"):
        evaluations[stage]={}
        for split in ("test","fresh_test"):
            value=read_json(directory/f"evaluation_{stage}_{split}.json")
            assert value["n"]==sizes[split]
            assert [triple(x) for x in value["samples"]] == [triple(x) for x in data[split]]
            correct=sum(reward(x["response"],x["answer"]) for x in value["samples"])
            assert correct == value["correct"]
            assert correct/value["n"] == value["accuracy"]
            assert all(bool(reward(x["response"],x["answer"]))==x["correct"] for x in value["samples"])
            evaluations[stage][split]=value
    history=read_json(directory/"validation_history.json")
    assert len(history)==9
    best_sft=max(history[:5],key=lambda x:x["validation_correct"])
    best_all=max(history,key=lambda x:x["validation_correct"])
    selected=(f"sft_epoch_{best_all['epoch']}" if best_all["stage"]=="sft" else
              f"grpo_step_{best_all['rollout_step']}" if best_all["stage"]=="grpo" else "sft_epoch_0")
    assert summary["selected_stage"]==selected
    assert summary["selected_validation_correct"]==best_all["validation_correct"]
    assert summary["best_sft_epoch"]==best_sft.get("epoch",0)
    sft_logs=read_jsonl(directory/"sft_training.jsonl")
    assert len(sft_logs)==summary["sft_optimizer_steps"]==1024
    assert all(math.isfinite(row["loss"]) and math.isfinite(row["gradient_norm"]) for row in sft_logs)
    assert [row["optimizer_step"] for row in sft_logs]==list(range(1,1025))
    logs=read_jsonl(directory/"training.jsonl")
    rollouts=read_jsonl(directory/"rollouts.jsonl")
    assert len(logs)==summary["grpo_optimizer_steps"]==256
    assert len(rollouts)==summary["grpo_rollout_completions"]==2048
    groups=defaultdict(list)
    for row in rollouts:
        assert triple(row) in split_ids["train"]
        assert reward(row["response"],row["answer"])==row["reward"]
        groups[(row["step"],triple(row))].append(row["reward"])
    distribution=Counter()
    for scores in groups.values():
        assert len(scores)==8
        distribution["mixed" if min(scores)!=max(scores) else "all_correct" if scores[0] else "all_wrong"]+=1
    assert len(groups)==256 and distribution["mixed"]==summary["informative_groups"]
    for index,row in enumerate(logs):
        assert row["optimizer_step"]==index+1 and row["rollout_step"]==index//2+1
        assert all(math.isfinite(x) for x in row.values() if isinstance(x,(int,float)))
        assert math.isclose(row["total_loss"],row["policy_loss"]+.01*row["reference_kl"],abs_tol=1e-7)
        assert row["informative_groups"]==sum(min(x)!=max(x) for (step,_),x in groups.items()
                                              if step==row["rollout_step"])
    result={"status":"all_passed","disjoint_split_sizes":sizes,
            "sft_optimizer_steps":len(sft_logs),"grpo_optimizer_steps":len(logs),
            "grpo_nonzero_gradient_calls":sum(x["grad_norm_before_clipping"]>0 for x in logs),
            "group_distribution":dict(distribution),
            "rollout_mean_reward":sum(x["reward"] for x in rollouts)/len(rollouts),
            "informative_group_rate":distribution["mixed"]/len(groups),
            "selected_stage":selected,"paired_comparisons":{}}
    for split in ("test","fresh_test"):
        result["paired_comparisons"][split]={}
        for left,right in [("base","sft"),("sft","grpo_final"),("base","selected")]:
            result["paired_comparisons"][split][f"{left}_to_{right}"]=comparison(
                evaluations[left][split],evaluations[right][split])
    (directory/"analysis_metrics.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    if not quiet:
        print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
