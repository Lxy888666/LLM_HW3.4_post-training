"""Prespecified SFT warm start followed by validation-selected GRPO updates."""
import argparse
import copy
import hashlib
import json
import math
import os
import random
import time

from run_grpo import (ROOT, AutoModelForCausalLM, AutoTokenizer, LoraConfig,
                      dump, encode, evaluate, generate, get_peft_model,
                      grade, load_algorithms, log_probs, np, source_manifest,
                      torch, transformers)
from peft import get_peft_model_state_dict, set_peft_model_state_dict

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def adapter_state(model):
    return {key: value.detach().cpu().clone()
            for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}


def save_adapter(model, directory):
    model.peft_config["default"].base_model_name_or_path = "Qwen/Qwen2.5-0.5B-Instruct"
    model.save_pretrained(directory, selected_adapters=["default"], save_embedding_layers=False)
    (directory / "README.md").write_text(
        "# 算术任务 LoRA 适配器\n\n"
        "基础模型为 `Qwen/Qwen2.5-0.5B-Instruct`。"
        "本目录保存训练检查点的 LoRA 权重与配置。"
        "实验配置、训练过程和评测记录见上一级目录。\n", encoding="utf-8")


def split_data(seed):
    legacy = json.loads((ROOT / "results/grpo_run/dataset.json").read_text(encoding="utf-8"))
    legacy_test = legacy["test"]
    excluded = {(x["a"], x["b"], x["c"]) for x in legacy_test}
    items = [{"a": a, "b": b, "c": c, "answer": a*b+c}
             for a in range(10, 50) for b in range(2, 10) for c in range(1, 10)
             if (a, b, c) not in excluded]
    random.Random(seed).shuffle(items)
    # No split is chosen using model outputs or reward values.
    holdout, validation, pool = items[:256], items[256:384], items[384:]
    return {"train": pool[:2048], "validation": validation,
            "test": legacy_test, "fresh_test": holdout}


def supervised_loss(model, tokenizer, items, device):
    encoded = encode(tokenizer, items, device)
    targets = [tokenizer.encode(str(x["answer"]), add_special_tokens=False)
               + [tokenizer.eos_token_id] for x in items]
    width = max(map(len, targets))
    responses = torch.full((len(items), width), tokenizer.pad_token_id,
                           dtype=torch.long, device=device)
    mask = torch.zeros_like(responses, dtype=torch.float32)
    for index, target in enumerate(targets):
        responses[index, :len(target)] = torch.tensor(target, device=device)
        mask[index, :len(target)] = 1
    sequence = torch.cat([encoded.input_ids, responses], dim=-1)
    attention = torch.cat([encoded.attention_mask, mask.long()], dim=-1)
    return -(log_probs(model, sequence, attention, responses) * mask).sum() / mask.sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="grpo_improved")
    args = parser.parse_args()
    out = ROOT / "results" / args.output
    if out.exists():
        raise FileExistsError("Select a new output name; the recorded experiment is preserved")
    out.mkdir(parents=True)
    seed = 20261113
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    alg = load_algorithms()
    data = split_data(seed)
    dump(out / "dataset.json", data)
    config = {"seed": seed, "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
              "base_model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
              "train_size": 2048, "validation_size": 128, "test_size": 32,
              "fresh_test_size": 256, "sft_epochs": 4, "sft_batch_size": 8,
              "sft_peak_lr": 5e-4, "sft_final_lr_fraction": .1,
              "sft_warmup_fraction": .05, "sft_optimizer_steps": 1024,
              "rollout_steps": 128, "prompts_per_step": 2, "group_size": 8,
              "updates_per_rollout": 2, "max_new_tokens": 12,
              "temperature": 1., "top_p": 1., "top_k": 0,
              "learning_rate": 2e-5, "cliprange": .2, "clip_ratio_c": 3.,
              "kl_coefficient": .01, "kl_type": "low_var_kl",
              "loss_agg_mode": "token-mean", "lora_rank": 16, "lora_alpha": 32,
              "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
              "reference": "Frozen adapter from the best validation SFT epoch",
              "selection_rule": "Maximum validation accuracy; ties retain the earlier checkpoint",
              "grpo_validation_interval": 32,
              "reward": "1 if the full stripped response parses as the correct integer, otherwise 0",
              "test_policy": "Evaluate legacy and fresh tests only at the base, selected SFT, final GRPO, and selected stages; never select by test metrics",
              "torch": torch.__version__, "transformers": transformers.__version__,
              "execution_scope": "Single-GPU runner using original verl algorithm bodies",
              "source": source_manifest(),
              "dataset_sha256": hashlib.sha256((out / "dataset.json").read_bytes()).hexdigest()}
    dump(out / "config.json", config)
    device = "cuda"
    model_path = ROOT / "models/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    base = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.,
        target_modules=config["lora_targets"], task_type="CAUSAL_LM"))
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    config["trainable_parameters"] = sum(p.numel() for p in trainable)
    dump(out / "config.json", config)
    start = time.perf_counter()
    for split in ("test", "fresh_test"):
        dump(out / f"evaluation_base_{split}.json", evaluate(model, tokenizer, data[split], device, 12))
    base_validation = evaluate(model, tokenizer, data["validation"], device, 12)
    history = [{"stage": "base", "validation_correct": base_validation["correct"]}]
    best_sft_state = adapter_state(model)
    best_sft_correct = base_validation["correct"]
    best_sft_epoch = 0
    optimizer = torch.optim.AdamW(trainable, lr=config["sft_peak_lr"], weight_decay=0.)
    sft_step = 0
    sft_start = time.perf_counter()
    with (out / "sft_training.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(config["sft_epochs"]):
            order = list(data["train"])
            random.Random(seed + epoch).shuffle(order)
            for offset in range(0, len(order), config["sft_batch_size"]):
                sft_step += 1
                progress = sft_step / config["sft_optimizer_steps"]
                if progress < config["sft_warmup_fraction"]:
                    scale = progress / config["sft_warmup_fraction"]
                else:
                    decay = (progress-config["sft_warmup_fraction"]) / (1-config["sft_warmup_fraction"])
                    scale = .1 + .9 * .5 * (1 + math.cos(math.pi*decay))
                optimizer.param_groups[0]["lr"] = config["sft_peak_lr"] * scale
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss = supervised_loss(model, tokenizer, order[offset:offset+8], device)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(trainable, 1.)
                if not torch.isfinite(loss) or not torch.isfinite(gradient):
                    raise RuntimeError("Non-finite SFT update")
                optimizer.step()
                row = {"epoch": epoch+1, "optimizer_step": sft_step, "loss": loss.item(),
                       "gradient_norm": gradient.item(), "learning_rate": optimizer.param_groups[0]["lr"]}
                stream.write(json.dumps(row)+"\n")
                if sft_step % 64 == 0:
                    stream.flush()
                    print(f"SFT step={sft_step}/1024 loss={loss.item():.4f}", flush=True)
            validation = evaluate(model, tokenizer, data["validation"], device, 12)
            history.append({"stage": "sft", "epoch": epoch+1, "validation_correct": validation["correct"]})
            dump(out / "validation_history.json", history)
            print(f"SFT epoch={epoch+1} validation={validation['correct']}/128", flush=True)
            if validation["correct"] > best_sft_correct:
                best_sft_correct = validation["correct"]
                best_sft_state = adapter_state(model)
                best_sft_epoch = epoch+1
    sft_seconds = time.perf_counter()-sft_start
    set_peft_model_state_dict(model, best_sft_state, adapter_name="default")
    save_adapter(model, out / "adapter_sft")
    for split in ("test", "fresh_test"):
        dump(out / f"evaluation_sft_{split}.json", evaluate(model, tokenizer, data[split], device, 12))
    # A separate adapter retains the warm-start reference without a second base model.
    model.add_adapter("reference", copy.deepcopy(model.peft_config["default"]))
    set_peft_model_state_dict(model, best_sft_state, adapter_name="reference")
    model.set_adapter("default")
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"], weight_decay=0.)
    best_state = best_sft_state
    best_correct = best_sft_correct
    selected_stage = f"sft_epoch_{best_sft_epoch}"
    grpo_logs = []
    grpo_start = time.perf_counter()
    order = list(data["train"])
    random.Random(seed+100).shuffle(order)
    with (out / "training.jsonl").open("w", encoding="utf-8") as stream:
        for step in range(config["rollout_steps"]):
            items = order[2*step:2*step+2]
            repeated = [x for x in items for _ in range(config["group_size"])]
            sequence, attention, responses, mask, texts = generate(model, tokenizer, repeated, device, 12, True)
            scores = torch.tensor([grade(t, x["answer"])[0] for t,x in zip(texts,repeated)], device=device)
            rewards = torch.zeros_like(mask)
            rewards.scatter_(1,(mask.sum(-1).long()-1).unsqueeze(1),scores.unsqueeze(1))
            advantages,_ = alg.grpo(rewards.cpu(),mask.cpu(),np.repeat(np.arange(2),8))
            advantages = advantages.to(device)
            with torch.no_grad():
                old = log_probs(model,sequence,attention,responses).detach()
                model.set_adapter("reference")
                reference = log_probs(model,sequence,attention,responses).detach()
                model.set_adapter("default")
            group_scores=scores.reshape(2,8)
            informative=int((group_scores.max(-1).values>group_scores.min(-1).values).sum())
            for update in range(2):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                current=log_probs(model,sequence,attention,responses)
                pg,clip_fraction,ppo_kl,lower_clip=alg.verl_policy(old,current,advantages,mask,
                    cliprange=.2,clip_ratio_c=3.,loss_agg_mode="token-mean")
                kl=alg.agg_loss(alg.kl_penalty(current,reference,"low_var_kl"),mask,"token-mean")
                loss=pg+.01*kl
                loss.backward()
                gradient=torch.nn.utils.clip_grad_norm_(trainable,1.)
                if not torch.isfinite(loss) or not torch.isfinite(gradient):
                    raise RuntimeError("Non-finite GRPO update")
                optimizer.step()
                row={"rollout_step":step+1,"optimizer_step":len(grpo_logs)+1,
                     "update_in_rollout":update+1,"mean_reward":scores.mean().item(),
                     "informative_groups":informative,"policy_loss":pg.item(),
                     "reference_kl":kl.item(),"total_loss":loss.item(),
                     "grad_norm_before_clipping":gradient.item(),"clip_fraction":clip_fraction.item(),
                     "old_current_approx_kl":ppo_kl.item(),"dual_clip_fraction":lower_clip.item()}
                grpo_logs.append(row)
                stream.write(json.dumps(row)+"\n")
            with (out/"rollouts.jsonl").open("a",encoding="utf-8") as handle:
                for item,text,score in zip(repeated,texts,scores.tolist()):
                    handle.write(json.dumps({"step":step+1,**item,"response":text,"reward":score})+"\n")
            if (step+1)%16==0:
                stream.flush()
                print(f"GRPO step={step+1}/128 reward={scores.mean().item():.3f}",flush=True)
            if (step+1)%32==0:
                validation=evaluate(model,tokenizer,data["validation"],device,12)
                history.append({"stage":"grpo","rollout_step":step+1,
                                "validation_correct":validation["correct"]})
                dump(out/"validation_history.json",history)
                print(f"GRPO step={step+1} validation={validation['correct']}/128",flush=True)
                if validation["correct"]>best_correct:
                    best_correct=validation["correct"]
                    best_state=adapter_state(model)
                    selected_stage=f"grpo_step_{step+1}"
    grpo_seconds=time.perf_counter()-grpo_start
    save_adapter(model,out/"adapter_grpo_final")
    for split in ("test","fresh_test"):
        dump(out/f"evaluation_grpo_final_{split}.json",evaluate(model,tokenizer,data[split],device,12))
    set_peft_model_state_dict(model,best_state,adapter_name="default")
    save_adapter(model,out/"adapter")
    for split in ("test","fresh_test"):
        result=evaluate(model,tokenizer,data[split],device,12)
        dump(out/f"evaluation_selected_{split}.json",result)
        if split=="test":
            dump(out/"evaluation_after.json",result)
    base_test=json.loads((out/"evaluation_base_test.json").read_text(encoding="utf-8"))
    dump(out/"evaluation_before.json",base_test)
    summary={"completed":True,"sft_optimizer_steps":sft_step,"best_sft_epoch":best_sft_epoch,
             "grpo_optimizer_steps":len(grpo_logs),"grpo_rollout_completions":128*16,
             "informative_groups":sum(x["informative_groups"] for x in grpo_logs[::2]),
             "total_groups":256,"selected_stage":selected_stage,"selected_validation_correct":best_correct,
             "sft_seconds":sft_seconds,"grpo_seconds":grpo_seconds,
             "total_seconds":time.perf_counter()-start,"evaluations":{}}
    for stage in ("base","sft","grpo_final","selected"):
        summary["evaluations"][stage]={}
        for split in ("test","fresh_test"):
            value=json.loads((out/f"evaluation_{stage}_{split}.json").read_text(encoding="utf-8"))
            summary["evaluations"][stage][split]={key:value[key] for key in
                ("n","correct","accuracy","format_valid_rate","mean_response_tokens")}
    dump(out/"summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
