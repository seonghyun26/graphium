"""Load every pairmixer config and instantiate a layer to verify it parses."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pathlib import Path
import yaml

import torch
from graphium.nn.pyg_layers.pairmixer_pyg import PairMixerLayerPyg


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_defaults(cfg_path, base_dir):
    """Shallow-merge via hydra-style `defaults:` key (one level only)."""
    cfg = load_yaml(cfg_path)
    defaults = cfg.pop("defaults", [])
    merged = {}
    for d in defaults:
        if isinstance(d, str):
            parent_path = base_dir / f"{d}.yaml"
            parent = resolve_defaults(parent_path, base_dir)
            _deep_update(merged, parent)
    _deep_update(merged, cfg)
    return merged


def _deep_update(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v


def main():
    base = Path("/home/shpark/prj-molrepr/graphium/expts/hydra-configs/model")
    files = sorted(base.glob("pairmixer*.yaml"))
    print(f"# Found {len(files)} pairmixer configs")
    for f in files:
        cfg = resolve_defaults(f, base)
        gnn = cfg["architecture"]["gnn"]
        lk = dict(gnn.get("layer_kwargs", {}))
        # Pass what the layer ctor expects
        lk.setdefault("in_dim", gnn["in_dim"])
        lk.setdefault("out_dim", gnn["out_dim"])
        lk.setdefault("in_dim_edges", gnn.get("in_dim_edges"))
        lk["normalization"] = "none"
        # Layer may have moe kwargs — leave them be
        lk["layer_idx"] = 1  # use fast path
        try:
            layer = PairMixerLayerPyg(**lk).cuda()
            ck = lk.get("use_checkpoint", "?")
            cm = lk.get("compile_mode", "?")
            print(f"  {f.name:45s}  ckpt={ck!s:5}  compile={cm!s}")
            del layer
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {f.name:45s}  FAIL: {e}")


if __name__ == "__main__":
    main()
