import os, sys
import resource
import time
import gc
import glob
from functools import partial
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
FOLDER_DIR = os.path.dirname(os.path.dirname(__file__))

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .compile import (
    compile_model,
    compile_transform,
    compile_dataloader,
    compile_opt,
    compile_loss,
)
from model.training_logger import (
    get_next_run,
    create_supervised_logger,
    NoOpLogger
)
from utils.distributed import init_distributed
from utils.logger import Logger
from model.early_stop import EarlyStopping

logger = Logger(__name__)


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

def gpu_timer(funct, log_timming = True):
    log_timming = log_timming and torch.cuda.is_available()
    
    elapsed_time = -1.0
    if log_timming:
        start = torch.cuda.Event(enable_timing = True)
        end = torch.cuda.Event(enable_timing = True)
        start.record()
        
    result = funct()
    if log_timming:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    
    return result, elapsed_time


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


def _resolve_action_key(action_map: dict, task_name: str) -> str:
    aliases = {
        "velocity": ["velocity", "vel", "speed"],
        "steer": ["steer", "steering", "steering_angle"],
        "lateral_error": ["lateral_error", "cte", "cross_track_error", "lateral"],
    }
    for key in aliases.get(task_name, [task_name]):
        if key in action_map:
            return key
    raise KeyError(
        f"Could not map task '{task_name}' to action keys. "
        f"Available keys: {sorted(action_map.keys())}"
    )


def _normalize_target_shape(target: torch.Tensor, batch_size: int) -> torch.Tensor:
    if target.ndim == 0:
        target = target.view(1, 1).expand(batch_size, 1)
    elif target.ndim == 1:
        target = target.unsqueeze(0)
    elif target.ndim >= 3:
        if target.shape[0] == 1 and target.shape[1] == batch_size:
            target = target[0]
        else:
            target = target.reshape(batch_size, -1)

    if target.ndim == 2 and target.shape[0] != batch_size and target.shape[1] == batch_size:
        target = target.transpose(0, 1)

    if target.ndim != 2:
        target = target.reshape(batch_size, -1)

    return target


def _format_targets(pred: torch.Tensor, action_input, enabled_tasks):
    action_map = action_input
    if isinstance(action_input, list):
        if len(action_input) == 0:
            raise ValueError("Received empty action list")
        if isinstance(action_input[0], dict):
            action_map = action_input[0]
        else:
            raise TypeError(f"Unsupported action list element type: {type(action_input[0])}")

    if not isinstance(action_map, dict):
        raise TypeError(f"Unsupported action container type: {type(action_map)}")

    B, T_pred, _ = pred.shape
    target_map = {}

    for task_name in enabled_tasks:
        action_key = _resolve_action_key(action_map, task_name)
        raw_target = action_map[action_key].to(device=pred.device, dtype=pred.dtype)
        target = _normalize_target_shape(raw_target, B)

        if target.shape[1] != T_pred:
            target = F.interpolate(
                target.unsqueeze(1),
                size=T_pred,
                mode="linear",
                align_corners=False,
            ).squeeze(1)

        target_map[task_name] = target

    return target_map
    
GLOBAL_SEED = 12
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

def main(args: dict, yaml_path: str):
    
    train_cfg: dict = args.get("train", {})
    crop_size    = train_cfg.get('crop_size', 224)
    nclips       = train_cfg.get('nclips', 1)
    
    loader_cfg: dict = args.get('loader_setup', {})
    num_workers        = loader_cfg.get('num_workers', 1)
    persistent_workers = loader_cfg.get('persistent_workers', False)
    pin_mem            = loader_cfg.get('pin_mem', False)

    model_cfg: dict = args.get("model", {})
    enc_cfg   = model_cfg.get('enc', {})
    probe_cfg = model_cfg.get('probe', {})

    augment_cfg: dict = args.get('data_aug', {})
    auto_augment        = augment_cfg.get('auto_augment', False)
    horizontal_flip     = augment_cfg.get('horizontal_flip', False)
    motion_shift        = augment_cfg.get('motion_shift', False)
    random_aspect_ratio = augment_cfg.get('random_resize_aspect_ratio', (1.0, 1.0))
    random_resize_scale = augment_cfg.get('random_resize_scale', (1.0, 1.0))
    reprob              = augment_cfg.get('reprob', 0.0)
    
    optim_cfg: dict = args.get('optimization', {})
    anneal       = optim_cfg.get('annel', optim_cfg.get('anneal', 1))
    epochs       = optim_cfg.get('epochs', 100)
    final_lr     = optim_cfg.get('final_lr', 0.0)
    final_wd     = optim_cfg.get("final_weight_decay", 0.0)
    ipe          = optim_cfg.get('ipe', 100)
    lr           = optim_cfg.get('lr', 1e-3)
    start_lr     = optim_cfg.get('start_lr', 1e-3)
    warmup       = optim_cfg.get('warmup', 10)
    weight_decay = optim_cfg.get('weight_decay', 0.0)
    betas        = optim_cfg.get('betas', (0.9, 0.999))
    eps          = optim_cfg.get('eps', 1.0e-8)

    loss_cfg: dict = args.get('loss', {})
    normalize_rep = loss_cfg.get('normalize_rep', False)

    meta_cfg: dict = args.get('meta', {})
    dtype = meta_cfg.get('dtype', 'float32')
    save_freq = meta_cfg.get('save_every_freq', 2)
    sync_gc   = meta_cfg.get('sync_gc', False)
    continue_train = bool(meta_cfg.get('continue_train', False))
    continue_from_run = meta_cfg.get('continue_from_run', None)
    resume_prefer_best = bool(meta_cfg.get('resume_prefer_best', True))

    logging_cfg: dict = args.get('logging', {})
    progress_type = logging_cfg.get('progress_type', 'table')
    save_csv = logging_cfg.get('save_csv', True)
    save_batch_csv = logging_cfg.get('save_batch_csv', False)
    save_epoch_csv = logging_cfg.get('save_epoch_csv', True)
    

    world_size, rank = init_distributed()
    logger.CUSTOM("SUCCESS", f"Initialized distributed on rank {rank}")
    
    
    if dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    
    
    torch.cuda.set_device(rank)
    device_type = f'cuda:{rank}'
    device = torch.device(device_type)
    encoder, probe= compile_model(
        enc_cfg = enc_cfg,
        probe_cfg = probe_cfg,
        device = device
    )

    if model_cfg.get('compile', False):
        logger.INFO("Compiling model")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        probe.compile()

    encoder = DDP(encoder, static_graph = True)
    probe   = DDP(probe, static_graph = False, find_unused_parameters = True)
    for p in encoder.parameters():
        p.requires_grad = False
    
    transform = compile_transform(
        random_horizontal_flip = horizontal_flip,
        random_resize_aspect_ratio = random_aspect_ratio,
        random_resize_scale = random_resize_scale,
        reprob = reprob,
        auto_augment = auto_augment,
        motion_shift = motion_shift,
        crop_size    = crop_size,
    )
    
    video_loader, val_loader, video_sampler, val_sampler = compile_dataloader(
        train_cfg, 
        nclips = nclips,
        transform = transform,
        collate_fn = torch.utils.data.default_collate,
        num_workers  = num_workers,
        persistance_workers = persistent_workers,
        pin_memory = pin_mem,
        world_sz = world_size,
        rank = rank
    )
    
    optim, scaler, lr_scheduler, wd_scheduler = compile_opt(
        encoder              = encoder,
        probe                = probe,
        iterations_per_epoch = ipe,
        start_lr             = start_lr,
        warmup               = warmup, 
        anneal               = anneal,
        num_epochs           = epochs,
        wd                   = weight_decay,
        final_lr             = final_lr,
        mixed_precision      = mixed_precision,
        betas                = betas,
        eps                  = eps,
        ref_lr               = lr,
        final_wd             = final_wd
    )

    criterion = compile_loss(loss_cfg = loss_cfg, device = device)
    optim.add_param_group({
        "params": list(criterion.parameters()),
        "weight_decay": 0.0,
    })
    logger.INFO("Added uncertainty loss parameters to optimizer")

    log_dir = os.path.join(FOLDER_DIR, "../Experiment/probe/")
    if rank == 0:
        next_run_idx = get_next_run(log_dir)
        if continue_train:
            resolved_run_idx = int(continue_from_run) if continue_from_run is not None else max(1, next_run_idx - 1)
            logger.INFO(f"Resuming requested. Selected run index: run{resolved_run_idx}")
        else:
            resolved_run_idx = next_run_idx
        run_idx_tensor = torch.tensor([resolved_run_idx], dtype=torch.long, device=device)
    else:
        run_idx_tensor = torch.tensor([0], dtype=torch.long, device=device)

    if dist.is_initialized():
        dist.broadcast(run_idx_tensor, src=0)
    run_idx = int(run_idx_tensor.item())

    start_epoch = 0
    resume_score = None

    if continue_train:
        resume_dir = os.path.join(log_dir, f"run{run_idx}", "weights")
        probe, optim, start_epoch, resume_score, resume_meta = load_checkpoint(
            model=probe,
            optimizer=optim,
            checkpoint_dir=resume_dir,
            checkpoint_name="probe.pt",
            prefer_best=resume_prefer_best,
            map_location=device,
        )

        scaler_payload = resume_meta.get("scaler", None)
        if scaler is not None and scaler_payload is not None:
            if isinstance(scaler_payload, dict):
                scaler.load_state_dict(scaler_payload)
            elif hasattr(scaler_payload, "state_dict"):
                scaler.load_state_dict(scaler_payload.state_dict())

        criterion_payload = resume_meta.get("criterion", None)
        if criterion_payload is not None and hasattr(criterion_payload, "state_dict"):
            try:
                criterion.load_state_dict(criterion_payload.state_dict())
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
    
    loader = iter(video_loader)

    # Only create logger and run directories for rank 0 to avoid race conditions
    if rank == 0:
        log_stats = create_supervised_logger(
            log_dir = log_dir,
            epochs = epochs,
            run_name = f"run{run_idx}",
            progress_type = progress_type,
            save_csv = save_csv,
            save_batch_csv = save_batch_csv,
            save_epoch_csv = save_epoch_csv,
        )
        probe_save = EarlyStopping(patience = epochs, freq = save_freq, min_delta = 0, path = os.path.join(log_dir, f"run{run_idx}/weights/probe.pt"), weights_only = True)
        if resume_score is not None:
            probe_save.best_loss = resume_score
        if not continue_train:
            os.system(f"cp {yaml_path} {os.path.join(log_dir, f'run{run_idx}')}")
    else:
        log_stats = NoOpLogger()
   
    if sync_gc:
        gc.disable()
        gc.collect()


    def train_step(clips, actions):
        _new_lr = lr_scheduler.step()
        _new_wd = wd_scheduler.step()
        
        def forward_target(c: torch.Tensor):
            with torch.no_grad():
                h: torch.Tensor = encoder(c)
                if normalize_rep:
                    h = F.layer_norm(h, (h.size(-1), ))
            return h

        def forward_prediction(h: torch.Tensor):
            _a = probe(h)
            return _a

        def regression(pred: torch.Tensor, target: dict[str, torch.Tensor]):
            loss, detail = criterion(pred, target)
            return loss, detail

        with torch.amp.autocast(device_type, dtype = dtype, enabled = mixed_precision):
            h = forward_target(clips)
            a = forward_prediction(h)
            targets = _format_targets(a, actions, criterion.enabled_tasks)
            loss, detail = regression(a, targets)
            
            
        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
        else:
            loss.backward()
            
        if mixed_precision:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
        optim.zero_grad()
        
        
        loss_details = {
            "Loss|Total": float(detail["total_loss"].item()),
        }
        for task_name, task_detail in detail["per_task"].items():
            task_title = task_name.replace("_", " ").title().replace(" ", "")
            loss_details[f"{task_title}"] = float(task_detail["weighted_loss"].item())

        return (
            loss.item(),
            _new_lr,
            _new_wd,
            loss_details,
        )

    @torch.no_grad()
    def val_step(clips, actions):
        with torch.amp.autocast(device_type, dtype = dtype, enabled = mixed_precision):
            h = encoder(clips)
            if normalize_rep:
                h = F.layer_norm(h, (h.size(-1), ))
            a = probe(h)
            targets = _format_targets(a, actions, criterion.enabled_tasks)
            loss, detail = criterion(a, targets)

        loss_details = {
            "Total Loss": float(detail["total_loss"].item()),
        }
        for task_name, task_detail in detail["per_task"].items():
            task_title = task_name.replace("_", " ").title().replace(" ", "")
            loss_details[f"{task_title}"] = float(task_detail["weighted_loss"].item())

        return float(loss.item()), loss_details
    
    with log_stats:
        log_stats.start_training("Training Latent Action WM")
        video_sampler.set_epoch(0)
        last_loss = 0.0
        last_val_loss = 0.0
        curr_lr, curr_wd = 0.0, 0.0
        for epoch in range(start_epoch, epochs):
            
            log_stats.start_epoch(epoch, len(video_loader), desc = "Training")
            
            for _ in log_stats.batch_iterator(range(ipe)):
                
                iter_retries = 0
                iter_success = False
                while not iter_success:
                    try:
                        sample = next(loader)
                        iter_success = True
                    except StopIteration:
                        loader = iter(video_loader)
                        video_sampler.set_epoch(epoch)
                    except Exception as e:
                        NUM_RETRIES = 5
                        if iter_retries < NUM_RETRIES:
                            logger.WARNING(f"Encountered an error while iterating loader: {e}")
                            iter_retries += 1
                            time.sleep(5)
                        else:
                            logger.ERROR("Exceeded maximum retries when iterating dataloader. Please check for error", exit_code = 5, full_traceback = e)
                        
                def load_clips():
                    clips = sample[0][0].to(device, non_blocking = True)
                    actions = [{key: value.to(device, non_blocking=True) for key, value in action_dict.items()} for action_dict in sample[1]]
                    return clips, actions
                
                clips, actions = load_clips()
                
                (loss, curr_lr, curr_wd, loss_details), elapsed_time = gpu_timer(partial(train_step, clips, actions[0]))
                last_loss = loss

                if np.isnan(loss) or np.isinf(loss):
                    logger.ERROR(
                        f"Model failed to converge. {'nan' if np.isnan(loss) else 'inf'} detected",
                        exit_code=-213,
                    )

                batch_metrics = {
                    "LR": curr_lr,
                    "WD": curr_wd,
                    "GPU Timer": elapsed_time,
                    **loss_details,
                }
                log_stats.log_batch(batch_metrics, phase="train", phase_agnostic=["LR", "WD", "GPU Timer"])

            if val_loader is not None and len(val_loader) > 0:
                val_sampler.set_epoch(epoch)
                log_stats.start_phase(len(val_loader), desc="Validation")

                for sample in log_stats.batch_iterator(val_loader):
                    clips = sample[0][0].to(device, non_blocking=True)
                    actions = [{key: value.to(device, non_blocking=True) for key, value in action_dict.items()} for action_dict in sample[1]]

                    val_loss, val_details = val_step(clips, actions[0])
                    last_val_loss = val_loss

                    val_metrics = {
                        **val_details,
                    }
                    log_stats.log_batch(val_metrics, phase="val")

            log_stats.log_epoch(extra_metrics={
                "GPU": torch.cuda.max_memory_allocated() / 1024.0 ** 2
            })

            gc.collect()

            if rank == 0:
                metric_loss = log_stats.get_metric("Total Loss", "val")
                if metric_loss is None:
                    metric_loss = log_stats.get_metric("Total Loss", "train")
                if metric_loss is None:
                    metric_loss = last_val_loss if last_val_loss > 0 else last_loss
                probe_save(
                    metric_loss,
                    probe,
                    epoch=epoch,
                    optimizer=optim,
                    scaler=scaler,
                    loss=last_loss,
                    lr=curr_lr,
                    criterion=criterion
                )
    