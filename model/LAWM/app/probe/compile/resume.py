import os
import torch
import glob
from utils.logger import Logger

logger = Logger(__name__)

def load_checkpoint(
    model,
    optimizer,
    checkpoint_dir,
    checkpoint_name="probe.pt",
    prefer_best=True,
    map_location=None,
):
    basename = "checkpoint.pt"
    meta_path = os.path.join(checkpoint_dir, basename)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = torch.load(meta_path, map_location=map_location, weights_only=False)
    score = meta.get("score")
    start_epoch = meta.get("epoch", 0)

    prefix = "best_" if prefer_best else "last_"
    model_path = os.path.join(checkpoint_dir, f"{prefix}{checkpoint_name}")
    if not os.path.exists(model_path):
        candidates = sorted(glob.glob(os.path.join(checkpoint_dir, f"{prefix}*.pt")))
        if not candidates:
            raise FileNotFoundError(f"Missing checkpoint weights under {checkpoint_dir} with prefix '{prefix}'")
        model_path = candidates[-1]

    loaded_state = torch.load(model_path, map_location=map_location, weights_only=False)
    if isinstance(loaded_state, dict):
        _load_state_dict_compat(model, loaded_state)
    elif hasattr(loaded_state, "state_dict"):
        _load_state_dict_compat(model, loaded_state.state_dict())
    else:
        raise TypeError(f"Unsupported checkpoint payload type for model weights: {type(loaded_state)}")

    if optimizer is not None:
        optimizer_payload = meta.get("optimizer_state_dict", None)
        if optimizer_payload is None:
            optimizer_payload = meta.get("optimizer", None)
        if optimizer_payload is not None:
            if isinstance(optimizer_payload, dict):
                optimizer.load_state_dict(optimizer_payload)
            elif hasattr(optimizer_payload, "state_dict"):
                optimizer.load_state_dict(optimizer_payload.state_dict())

    return model, optimizer, start_epoch + 1, score, meta


def restore_resume_state(
    resume_meta: dict,
    scaler,
    criterion,
    lr_scheduler,
    wd_scheduler,
    start_epoch: int,
    ipe: int,
    rank: int,
    run_idx: int,
    resume_prefer_best: bool,
):
    scaler_payload = resume_meta.get("scaler", None)
    if scaler is not None and scaler_payload is not None:
        if isinstance(scaler_payload, dict):
            scaler.load_state_dict(scaler_payload)
        elif hasattr(scaler_payload, "state_dict"):
            scaler.load_state_dict(scaler_payload.state_dict())

    criterion_payload = resume_meta.get("criterion", None)
    if criterion_payload is not None:
        try:
            if isinstance(criterion_payload, dict):
                _load_state_dict_compat(criterion, criterion_payload)
            elif hasattr(criterion_payload, "state_dict"):
                _load_state_dict_compat(criterion, criterion_payload.state_dict())
        except Exception:
            logger.WARNING("Could not restore criterion state from checkpoint metadata. Continuing with current criterion state.")

    resumed_iters = max(0, int(start_epoch) * int(ipe))
    for _ in range(resumed_iters):
        lr_scheduler.step()
        wd_scheduler.step()

    if rank == 0:
        logger.INFO(
            f"Resumed run{run_idx} from epoch {start_epoch} "
            f"(prefer_best={resume_prefer_best}, restored_iters={resumed_iters})."
        )

    return resumed_iters


def _load_state_dict_compat(model, state_dict: dict):
    adjusted = _normalize_state_dict_for_model(model, state_dict)
    try:
        model.load_state_dict(adjusted)
        return
    except RuntimeError:
        if hasattr(model, "module"):
            raw_state = {k.removeprefix("module."): v for k, v in adjusted.items()}
            model.module.load_state_dict(raw_state)
            return
        raise


def _normalize_state_dict_for_model(model, state_dict: dict) -> dict:
    target_keys = list(model.state_dict().keys())
    target_has_module = any(k.startswith("module.") for k in target_keys)
    source_keys = list(state_dict.keys())
    source_has_module = any(k.startswith("module.") for k in source_keys)

    if target_has_module and not source_has_module:
        return {f"module.{k}": v for k, v in state_dict.items()}
    if not target_has_module and source_has_module:
        return {k.removeprefix("module."): v for k, v in state_dict.items()}
    return state_dict