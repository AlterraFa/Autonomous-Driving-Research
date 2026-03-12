from .manual    import run_manual
from .replay    import run_replay
from .inference import run_inference

MODE_RUNNERS = {
    "manual"    : run_manual,
    "replay"    : run_replay,
    "inference" : run_inference,
}

__all__ = [
    "run_manual",
    "run_replay",
    "run_inference",
    "MODE_RUNNERS",
]
