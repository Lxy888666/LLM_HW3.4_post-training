"""Execute pinned upstream algorithm bodies without distributed-framework imports.

Only definition decorators are removed; function/class bodies remain upstream code.
Upstream licensing remains in sources/OpenRLHF/LICENSE and sources/verl/LICENSE.
"""
import ast
import copy
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "OpenRLHF": "a542dd98d29efacef89a4f694366e66c6e6d419b",
    "verl": "8fdc4d3f202f41461f4de9f42a637228e342668b",
}


def verify_sources():
    manifest_path = ROOT / "results" / "upstream_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["revisions"] != REVISIONS:
        raise RuntimeError("The source snapshot manifest has unexpected revisions")
    for project, expected in REVISIONS.items():
        directory = ROOT / "sources" / project
        if (directory / ".git").exists():
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
            if actual != expected:
                raise RuntimeError(f"Source revision mismatch: {project}: {actual}")
            dirty = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=directory, text=True)
            if dirty.strip():
                raise RuntimeError(f"Tracked upstream files were modified: {project}")
    for relative, expected_hash in manifest["files"].items():
        path = ROOT / "sources" / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Source snapshot checksum mismatch: {relative}")


def definition(project, relative_path, name, namespace, parent=None):
    path = ROOT / "sources" / project / relative_path
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    nodes = tree.body
    if parent:
        nodes = next(x for x in nodes if isinstance(x, ast.ClassDef) and x.name == parent).body
    node = copy.deepcopy(next(x for x in nodes if getattr(x, "name", None) == name))
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def load_algorithms():
    verify_sources()
    ns = {"torch": torch, "nn": nn, "np": np, "defaultdict": defaultdict,
          "Optional": Optional, "Tuple": Tuple, "Union": Union}
    masked = definition("OpenRLHF", "openrlhf/models/utils.py", "masked_mean", ns)
    reward = definition("OpenRLHF", "openrlhf/models/utils.py", "compute_reward", ns)
    policy = definition("OpenRLHF", "openrlhf/models/loss.py", "PolicyLoss", ns)
    value = definition("OpenRLHF", "openrlhf/models/loss.py", "ValueLoss", ns)
    gae = definition("OpenRLHF", "openrlhf/trainer/ppo_utils/experience_maker.py",
                     "get_advantages_and_returns", ns, parent="RemoteExperienceMaker")
    vns = {"torch": torch}
    definition("verl", "verl/utils/torch_functional.py", "masked_sum", vns)
    vmasked = definition("verl", "verl/utils/torch_functional.py", "masked_mean", vns)
    ns["verl_F"] = SimpleNamespace(masked_mean=vmasked)
    core = "verl/trainer/ppo/core_algos.py"
    agg = definition("verl", core, "agg_loss", ns)
    grpo = definition("verl", core, "compute_grpo_outcome_advantage", ns)
    vpolicy = definition("verl", core, "compute_policy_loss", ns)
    kl = definition("verl", core, "kl_penalty", ns)
    return SimpleNamespace(ppo_policy=policy, ppo_value=value, reward=reward,
                           gae=gae, masked_mean=masked, grpo=grpo,
                           verl_policy=vpolicy, agg_loss=agg, kl_penalty=kl)


def source_manifest():
    manifest = {"revisions": REVISIONS, "files": {}}
    paths = {
        "OpenRLHF": ["openrlhf/models/utils.py", "openrlhf/models/loss.py",
                     "openrlhf/trainer/ppo_utils/experience_maker.py"],
        "verl": ["verl/trainer/ppo/core_algos.py", "verl/utils/torch_functional.py"],
    }
    for project, files in paths.items():
        for relative in files:
            data = (ROOT / "sources" / project / relative).read_bytes()
            manifest["files"][f"{project}/{relative}"] = hashlib.sha256(data).hexdigest()
    return manifest
