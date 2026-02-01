"""Compatibility helpers for running mmseg_custom backbones without mmseg/mmcv.

These adapters keep the ViT-Adapter backbones importable inside the Prismatic
environment (torch2.x) without requiring the full mmseg/mmcv stack. If mmseg/mmcv
are available, we defer to the real implementations.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import torch
import torch.nn as nn


class _DummyRegistry:
    def register_module(self, *args: Any, **kwargs: Any) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            return cls

        return decorator


try:  # mmseg registry
    from mmseg.models.builder import BACKBONES as _MMSEG_BACKBONES
except Exception:  # pragma: no cover - optional dependency
    _MMSEG_BACKBONES = _DummyRegistry()

BACKBONES = _MMSEG_BACKBONES


try:  # mmseg logging helper
    from mmseg.utils import get_root_logger as _get_root_logger
except Exception:  # pragma: no cover - optional dependency
    def _get_root_logger() -> logging.Logger:
        return logging.getLogger("mmseg_custom")

get_root_logger = _get_root_logger


try:  # mmcv BaseModule
    from mmcv.runner import BaseModule as _BaseModule
except Exception:  # pragma: no cover - optional dependency
    class _BaseModule(nn.Module):
        pass

BaseModule = _BaseModule


try:  # mmcv checkpoint loader
    from mmcv.runner import load_checkpoint as _mmcv_load_checkpoint
except Exception:  # pragma: no cover - optional dependency
    def _mmcv_load_checkpoint(
        model: nn.Module,
        filename: str,
        map_location: str | torch.device = "cpu",
        strict: bool = False,
        logger: logging.Logger | None = None,
    ) -> Dict[str, Any]:
        if filename is None:
            return {}
        state = torch.load(filename, map_location=map_location)
        if isinstance(state, dict):
            state = state.get("state_dict", state)
        incompatible = model.load_state_dict(state, strict=strict)
        if logger and (incompatible.missing_keys or incompatible.unexpected_keys):
            logger.warning(
                "Checkpoint loaded with mismatched keys: missing=%s unexpected=%s",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )
        return state if isinstance(state, dict) else {}

load_checkpoint = _mmcv_load_checkpoint
