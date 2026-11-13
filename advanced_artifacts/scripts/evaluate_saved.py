"""Reload the saved adapter and compare greedy outputs with the recorded results."""
import argparse
import json

from run_grpo import ROOT, AutoModelForCausalLM, AutoTokenizer, evaluate, torch
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="grpo_run")
    args = parser.parse_args()
    torch.set_num_threads(4)
    directory = ROOT / "results" / args.run
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    records = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    previous = json.loads((directory / "evaluation_after.json").read_text(encoding="utf-8"))
    model_path = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, directory / "adapter").to("cuda")
    actual = evaluate(model, tokenizer, records["test"], "cuda", config["max_new_tokens"])
    equal = [x["response"] == y["response"] for x, y in zip(actual["samples"], previous["samples"])]
    if not all(equal) or len(equal) != len(previous["samples"]):
        raise AssertionError("Reloaded checkpoint does not reproduce every recorded greedy answer")
    result = {"status": "all_passed", "matching_answers": sum(equal),
              "test_size": actual["n"], "correct": actual["correct"],
              "accuracy": actual["accuracy"]}
    if "fresh_test" in records:
        previous_fresh = json.loads((directory / "evaluation_selected_fresh_test.json").read_text(encoding="utf-8"))
        fresh = evaluate(model, tokenizer, records["fresh_test"], "cuda", config["max_new_tokens"])
        equal_fresh = [x["response"] == y["response"] for x,y in zip(fresh["samples"],previous_fresh["samples"])]
        if not all(equal_fresh) or len(equal_fresh) != len(previous_fresh["samples"]):
            raise AssertionError("Reloaded checkpoint differs from the fresh-test greedy answers")
        result["fresh_test"] = {"matching_answers": sum(equal_fresh), "test_size": fresh["n"],
                                "correct": fresh["correct"], "accuracy": fresh["accuracy"]}
    (directory / "checkpoint_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
