"""Checkpoint configuration regressions for Titans v2."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from memlm import AdapterCfg, _titans_memory_kwargs


def test_v2_checkpoint_preserves_titans_inner_lr_cap():
    cfg = AdapterCfg.from_ckpt({"cfg": {
        "kind": "titans",
        "titans_inner_lr_cap": 0.01,
        "titans_kwargs": {
            "init_decay_bias": -5.0,
            "init_momentum_bias": 2.0,
            "qk_rmsnorm": True,
        },
    }})

    assert cfg.titans_inner_lr_cap == 0.01
    assert _titans_memory_kwargs(cfg)["default_step_transform_max_lr"] == 0.01


def test_legacy_checkpoint_uses_previous_titans_lr_default():
    cfg = AdapterCfg.from_ckpt({"cfg": {"kind": "titans", "titans_kwargs": {}}})

    assert _titans_memory_kwargs(cfg)["default_step_transform_max_lr"] == 1.0

