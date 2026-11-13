"""Behavioral probes for the actual upstream PPO and GRPO algorithm definitions."""
import json
from pathlib import Path

import numpy as np
import torch

from source_loader import ROOT, load_algorithms, source_manifest


def main():
    alg = load_algorithms()
    results = {}
    current = torch.log(torch.tensor([[1.3, 0.7, 1.1, 0.9, 8.0]])).requires_grad_()
    mask = torch.tensor([[1., 1., 1., 1., 0.]])
    advantage = torch.tensor([[1., -1., 1., -1., 999.]])
    loss = alg.ppo_policy()(current, torch.zeros_like(current), advantage, mask)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(-0.15), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(current.grad, torch.tensor([[0., 0., -.275, .225, 0.]]),
                               atol=1e-6, rtol=1e-5)
    results["ppo_clipping_and_mask"] = {"loss": loss.item(), "gradient": current.grad.tolist()}

    rewards = alg.reward(torch.tensor([1., 2.]), .2,
                         torch.tensor([[.1, .1, 0.], [.1, .1, .1]]),
                         torch.tensor([[1, 1, 0], [1, 1, 1]]))
    torch.testing.assert_close(rewards, torch.tensor([[-.02, .98, 0.], [-.02, -.02, 1.98]]))
    results["terminal_reward_placement"] = rewards.tolist()

    with torch.no_grad():
        advantages, returns = alg.gae(None, torch.zeros(1, 3), torch.tensor([[0., 0., 1.]]),
                                      torch.ones(1, 3), .9, .8)
    torch.testing.assert_close(advantages, torch.tensor([[.5184, .72, 1.]]))
    results["gae_backward_propagation"] = advantages.tolist()

    vloss = alg.ppo_value(clip_eps=.2)(torch.tensor([[.5, .7, 999.]]), torch.zeros(1, 3),
                                      torch.tensor([[1., 1., 0.]]), torch.tensor([[1., 1., 0.]]))
    torch.testing.assert_close(vloss, torch.tensor(.32))
    results["clipped_value_loss"] = vloss.item()

    scores = torch.tensor([1., 0., 0., 0., 0., 0., 1., 1.]).unsqueeze(1)
    ids = np.array(["mixed"] * 4 + ["wrong"] * 2 + ["correct"] * 2)
    adv, _ = alg.grpo(scores, torch.ones_like(scores), ids)
    torch.testing.assert_close(adv[:, 0], torch.tensor([1.5, -.5, -.5, -.5, 0., 0., 0., 0.]),
                               atol=4e-6, rtol=1e-5)
    centered, _ = alg.grpo(scores, torch.ones_like(scores), ids, norm_adv_by_std_in_grpo=False)
    torch.testing.assert_close(centered[:4, 0], torch.tensor([.75, -.25, -.25, -.25]))
    permutation = torch.tensor([4, 2, 7, 0, 3, 6, 1, 5])
    shuffled, _ = alg.grpo(scores[permutation], torch.ones_like(scores), ids[permutation.numpy()])
    torch.testing.assert_close(shuffled, adv[permutation])
    results["grpo_grouping_and_normalization"] = {"normalized": adv[:, 0].tolist(),
                                                    "centered": centered[:, 0].tolist(),
                                                    "uid_permutation_invariant": True}

    singleton, _ = alg.grpo(torch.tensor([[7.]]), torch.ones(1, 1), np.array(["single"]))
    torch.testing.assert_close(singleton, torch.tensor([[7. / (1. + 1e-6)]]))
    results["grpo_singleton_fallback"] = singleton.item()

    loss_mat = torch.tensor([[1., 100.], [3., 3.]])
    lmask = torch.tensor([[1., 0.], [1., 1.]])
    token = alg.agg_loss(loss_mat, lmask, "token-mean")
    sequence = alg.agg_loss(loss_mat, lmask, "seq-mean-token-mean")
    torch.testing.assert_close(token, torch.tensor(7. / 3))
    torch.testing.assert_close(sequence, torch.tensor(2.))
    results["loss_aggregation"] = {"token_mean": token.item(), "sequence_mean": sequence.item()}

    old = torch.zeros(1, 1)
    new = torch.tensor([[4.]]).log()
    neg_adv = -torch.ones(1, 1)
    open_loss = alg.ppo_policy()(new, old, neg_adv, torch.ones(1, 1))
    verl_loss = alg.verl_policy(old, new, neg_adv, torch.ones(1, 1), cliprange=.2)[0]
    assert open_loss.item() == 4. and verl_loss.item() == 3.
    results["dual_clip_difference"] = {"OpenRLHF": open_loss.item(), "verl": verl_loss.item()}

    payload = {"status": "all_passed", "probe_count": len(results),
               "torch_version": torch.__version__, "source": source_manifest(), "results": results}
    output = ROOT / "results" / "core_verification.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
