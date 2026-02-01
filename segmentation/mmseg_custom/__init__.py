try:  # allow importing backbones without the full mmseg stack
    from .core import *  # noqa: F401,F403
    from .datasets import *  # noqa: F401,F403
    from .models import *  # noqa: F401,F403
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name is None or not exc.name.startswith("mmseg"):
        raise
