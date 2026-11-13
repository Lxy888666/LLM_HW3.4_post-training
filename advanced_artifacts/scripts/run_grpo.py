"""Single-GPU post-training using original verl v0.5.0 GRPO/loss bodies.

This runner does not launch either project's distributed trainer.
It performs genuine LoRA parameter updates on a public pretrained LLM.
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from source_loader import load_algorithms, source_manifest


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def task(item):
    return f"Compute {item['a']} * {item['b']} + {item['c']}. Reply with only the integer."


def grade(text, answer):
    import re
    match = re.fullmatch(r"\s*([+-]?\d+)\s*", text)
    valid = match is not None
    return float(valid and int(match.group(1)) == answer), valid


def make_data(seed, train_size, test_size):
    rng = random.Random(seed)
    triples = [(a, b, c) for a in range(10, 50) for b in range(2, 10) for c in range(1, 10)]
    rng.shuffle(triples)
    records = [{"a": a, "b": b, "c": c, "answer": a * b + c} for a, b, c in triples]
    return records[:train_size], records[train_size:train_size + test_size]


def encode(tokenizer, items, device):
    prompts = [tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a calculator. Output only the final integer, without explanation."},
        {"role": "user", "content": task(x)}], tokenize=False, add_generation_prompt=True) for x in items]
    return tokenizer(prompts, padding=True, return_tensors="pt").to(device)


@torch.no_grad()
def generate(model, tokenizer, items, device, limit, sample):
    model.eval()
    encoded = encode(tokenizer, items, device)
    kwargs = dict(max_new_tokens=limit, do_sample=sample, top_p=1., top_k=0,
                  repetition_penalty=1., pad_token_id=tokenizer.pad_token_id,
                  use_cache=True)
    if sample:
        kwargs["temperature"] = 1.
    sequence = model.generate(**encoded, **kwargs)
    width = encoded.input_ids.shape[1]
    responses = sequence[:, width:]
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    is_eos = torch.zeros_like(responses, dtype=torch.bool)
    for eos_id in eos_ids:
        is_eos |= responses.eq(eos_id)
    # Include the first EOS, exclude subsequent padding.
    mask = ((is_eos.long().cumsum(-1) - is_eos.long()) == 0).float()
    texts = tokenizer.batch_decode(responses, skip_special_tokens=True)
    attention = torch.cat([encoded.attention_mask, mask.long()], dim=-1)
    return sequence, attention, responses, mask, texts


def log_probs(model, sequence, attention, responses):
    positions = attention.long().cumsum(-1) - 1
    positions.masked_fill_(attention == 0, 0)
    output = model(input_ids=sequence, attention_mask=attention, position_ids=positions,
                   use_cache=False, logits_to_keep=responses.shape[1] + 1)
    logits = output.logits[:, :-1, :].float()
    return logits.log_softmax(-1).gather(-1, responses.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def evaluate(model, tokenizer, items, device, limit):
    rows = []
    for offset in range(0, len(items), 4):
        batch = items[offset:offset + 4]
        _, _, _, masks, texts = generate(model, tokenizer, batch, device, limit, False)
        for item, text, length in zip(batch, texts, masks.sum(-1).tolist()):
            correct, valid = grade(text, item["answer"])
            rows.append({**item, "prompt": task(item), "response": text,
                         "correct": bool(correct), "format_valid": valid,
                         "response_tokens": int(length)})
    return {"n": len(rows), "correct": sum(x["correct"] for x in rows),
            "accuracy": sum(x["correct"] for x in rows) / len(rows),
            "format_valid_rate": sum(x["format_valid"] for x in rows) / len(rows),
            "mean_response_tokens": sum(x["response_tokens"] for x in rows) / len(rows),
            "samples": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--output", default="grpo_run")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This run configuration requires a CUDA GPU")
    alg = load_algorithms()
    device = "cuda"
    out = ROOT / "results" / args.output
    if (out / "config.json").exists():
        raise FileExistsError(f"Use a fresh --output name; existing run is preserved: {out}")
    out.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.,
                                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model.to(device)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    initial = {n: p.detach().cpu().clone() for n, p in trainable}
    optimizer = torch.optim.AdamW([p for _, p in trainable], lr=1e-4, weight_decay=0.)
    train, test = make_data(args.seed, args.steps * 2, 32)
    dump(out / "dataset.json", {"train": train, "test": test})
    settings = {"seed": args.seed, "rollout_steps": args.steps, "prompts_per_step": 2,
                "group_size": 4, "updates_per_rollout": 2, "max_new_tokens": 12,
                "temperature": 1., "top_p": 1., "top_k": 0, "repetition_penalty": 1.,
                "learning_rate": 1e-4, "cliprange": .2, "clip_ratio_c": 3.,
                "kl_coefficient": .01, "kl_type": "low_var_kl", "loss_agg_mode": "token-mean",
                "lora_rank": 8, "lora_alpha": 16, "lora_targets": ["q_proj", "v_proj"],
                "reward": "1 if entire stripped response is a correct integer, otherwise 0",
                "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "base_model_revision": json.loads((model_path / "download_manifest.json").read_text())["revision"],
                "trainable_parameters": sum(p.numel() for _, p in trainable),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "torch": torch.__version__, "transformers": transformers.__version__,
                "execution_scope": "Single-GPU runner using extracted original verl functions; not native distributed trainer",
                "source": source_manifest()}
    dump(out / "config.json", settings)
    start = time.perf_counter()
    before = evaluate(model, tokenizer, test, device, settings["max_new_tokens"])
    dump(out / "evaluation_before.json", before)
    print("before", before["correct"], "/", before["n"], flush=True)
    logs = []
    training_start = time.perf_counter()
    with (out / "training.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(args.steps):
            items = train[step * 2:step * 2 + 2]
            repeated = [item for item in items for _ in range(settings["group_size"])]
            sequence, attention, responses, mask, texts = generate(
                model, tokenizer, repeated, device, settings["max_new_tokens"], True)
            scores = torch.tensor([grade(t, x["answer"])[0] for t, x in zip(texts, repeated)],
                                  device=device)
            token_rewards = torch.zeros_like(mask)
            last = mask.sum(-1).long() - 1
            token_rewards.scatter_(1, last.unsqueeze(1), scores.unsqueeze(1))
            ids = np.repeat(np.arange(2), settings["group_size"])
            # Execute the original CPU-side driver advantage function.
            advantages, _ = alg.grpo(token_rewards.cpu(), mask.cpu(), ids)
            advantages = advantages.to(device)
            with torch.no_grad():
                old = log_probs(model, sequence, attention, responses).detach()
                with model.disable_adapter():
                    reference = log_probs(model, sequence, attention, responses).detach()
            group_rewards = scores.reshape(2, settings["group_size"])
            informative = int((group_rewards.max(-1).values > group_rewards.min(-1).values).sum().item())
            for update in range(settings["updates_per_rollout"]):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                current = log_probs(model, sequence, attention, responses)
                pg, clipfrac, ppo_kl, lower_clip = alg.verl_policy(
                    old, current, advantages, mask, cliprange=settings["cliprange"],
                    clip_ratio_c=settings["clip_ratio_c"], loss_agg_mode=settings["loss_agg_mode"])
                kl = alg.agg_loss(alg.kl_penalty(current, reference, "low_var_kl"), mask, "token-mean")
                loss = pg + settings["kl_coefficient"] * kl
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("Non-finite gradient")
                optimizer.step()
                row = {"rollout_step": step + 1, "optimizer_step": len(logs) + 1,
                       "update_in_rollout": update + 1, "mean_reward": scores.mean().item(),
                       "informative_groups": informative, "policy_loss": pg.item(),
                       "reference_kl": kl.item(), "total_loss": loss.item(),
                       "grad_norm_before_clipping": grad_norm.item(), "clip_fraction": clipfrac.item(),
                       "old_current_approx_kl": ppo_kl.item(), "dual_clip_fraction": lower_clip.item(),
                       "response_tokens": int(mask.sum().item()),
                       "elapsed_seconds": time.perf_counter() - training_start}
                logs.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
            with (out / "rollouts.jsonl").open("a", encoding="utf-8") as handle:
                for item, text, reward in zip(repeated, texts, scores.tolist()):
                    handle.write(json.dumps({"step": step + 1, **item, "response": text,
                                             "reward": reward}) + "\n")
            print("step", step + 1, "reward", round(scores.mean().item(), 3),
                  "informative", informative, "grad", round(grad_norm.item(), 5), flush=True)
    training_seconds = time.perf_counter() - training_start
    after = evaluate(model, tokenizer, test, device, settings["max_new_tokens"])
    dump(out / "evaluation_after.json", after)
    model.peft_config["default"].base_model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"
    model.save_pretrained(out / "adapter")
    (out / "adapter" / "README.md").write_text(
        "# 单卡 GRPO 算术实验适配器\n\n"
        "基础模型：`Qwen/Qwen2.5-0.5B-Instruct`。\n\n"
        "本目录保存核心函数级单卡 GRPO 实验的 LoRA 适配器，"
        "目标模块为 `q_proj` 与 `v_proj`，秩为 8，`lora_alpha=16`。\n\n"
        "训练配置、数据、评测与日志见上一级目录；"
        "使用方法见附件根目录的 `README.md`。加载时需配合指定版本的基础模型。\n\n"
        "适配器用于复查本次小规模算术实验，不代表通用数学能力的性能结论。\n",
        encoding="utf-8")
    squared_change = sum((p.detach().cpu().float() - initial[n].float()).square().sum().item()
                         for n, p in trainable)
    changed = sum(int(torch.count_nonzero(p.detach().cpu() - initial[n])) for n, p in trainable)
    summary = {"completed": True, "optimizer_steps": len(logs),
               "nonzero_gradient_steps": sum(x["grad_norm_before_clipping"] > 0 for x in logs),
               "rollout_completions": args.steps * 8,
               "informative_groups": sum(x["informative_groups"] for x in logs[::2]),
               "total_groups": args.steps * 2,
               "changed_parameter_elements": changed, "parameter_delta_l2": squared_change ** .5,
               "before_accuracy": before["accuracy"], "after_accuracy": after["accuracy"],
               "before_correct": before["correct"], "after_correct": after["correct"],
               "test_size": len(test), "before_format_valid_rate": before["format_valid_rate"],
               "after_format_valid_rate": after["format_valid_rate"],
               "training_seconds": training_seconds, "total_seconds": time.perf_counter() - start}
    assert changed > 0, "No trainable parameter changed"
    dump(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
