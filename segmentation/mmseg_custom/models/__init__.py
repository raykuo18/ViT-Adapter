# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401,F403

try:  # optional full mmseg/mmcv stack
    from .builder import (MASK_ASSIGNERS, MATCH_COST, TRANSFORMER, build_assigner,
                          build_match_cost)
    from .decode_heads import *  # noqa: F401,F403
    from .losses import *  # noqa: F401,F403
    from .plugins import *  # noqa: F401,F403
    from .segmentors import *  # noqa: F401,F403

    __all__ = [
        'MASK_ASSIGNERS', 'MATCH_COST', 'TRANSFORMER', 'build_assigner',
        'build_match_cost'
    ]
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name is None or not exc.name.startswith("mmcv"):
        raise
    __all__ = []
