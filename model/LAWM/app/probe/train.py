import os, sys
import resource
import time
import gc
from ruamel.yaml import YAML
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
    compile_grad_optimizer,
    compile_loss, format_targets,
    load_checkpoint, restore_resume_state
)
from utils.training_logger import (
    get_next_run,
    create_supervised_logger,
    NoOpLogger
)
from utils.distributed import init_distributed
from utils.logger import Logger
from utils.early_stop import EarlyStopping

logger = Logger(__name__)


def _unwrap_module(module):
    return module.module if hasattr(module, "module") else module


def _task_metric_name(task_name: str) -> str:
    return task_name.replace("_", " ").title()


def _collect_affine_metrics(module, enabled_tasks=None, max_dims: int = 4, affine_modality: str = None):
    core = _unwrap_module(module)
    metrics = {}
    task_names = list(enabled_tasks) if enabled_tasks is not None else []

    scale_param = getattr(core, "scale", None)
    shift_param = getattr(core, "shift", None)
    if affine_modality is not None and affine_modality in task_names:
        idx = task_names.index(affine_modality)
        label = _task_metric_name(affine_modality)
        if scale_param is not None:
            raw_scale = scale_param.detach().float().flatten().cpu().tolist()
            eff_scale = F.softplus(scale_param.detach()).float().flatten().cpu().tolist()
            metrics[f"Raw Scale {label}"] = float(raw_scale[idx])
            metrics[f"Scale {label}"] = float(eff_scale[idx])
        if shift_param is not None:
            shift_vals = shift_param.detach().float().flatten().cpu().tolist()
            metrics[f"Shift {label}"] = float(shift_vals[idx])
        return metrics

    # Default: log all modalities
    if scale_param is not None:
        raw_scale = scale_param.detach().float().flatten().cpu()
        eff_scale = F.softplus(scale_param.detach()).float().flatten().cpu()
        metrics["Scale Mean"] = float(eff_scale.mean().item())
        for i, value in enumerate(raw_scale[:max_dims]):
            if i < len(task_names):
                task_label = _task_metric_name(task_names[i])
                metrics[f"Raw Scale {task_label}"] = float(value.item())
            else:
                metrics[f"Raw Scale[{i}]"] = float(value.item())
        for i, value in enumerate(eff_scale[:max_dims]):
            if i < len(task_names):
                task_label = _task_metric_name(task_names[i])
                metrics[f"Scale {task_label}"] = float(value.item())
            else:
                metrics[f"Scale[{i}]"] = float(value.item())

    if shift_param is not None:
        shift_vals = shift_param.detach().float().flatten().cpu()
        metrics["Shift Mean"] = float(shift_vals.mean().item())
        for i, value in enumerate(shift_vals[:max_dims]):
            if i < len(task_names):
                task_label = _task_metric_name(task_names[i])
                metrics[f"Shift {task_label}"] = float(value.item())
            else:
                metrics[f"Shift[{i}]"] = float(value.item())

    return metrics

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

def save_config_pretty(config_dict, save_path):
    yaml = YAML()
    # Basic formatting
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    
    # This turns the dict into a 'ruamel' internal dict that supports comments/spacing
    from ruamel.yaml.comments import CommentedMap
    
    def dict_to_commented(d):
        if isinstance(d, dict):
            cm = CommentedMap()
            for k, v in d.items():
                cm[k] = dict_to_commented(v)
            return cm
        return d

    pretty_data = dict_to_commented(config_dict)

    # Add a blank line before every top-level key for readability
    first = True
    for key in pretty_data.keys():
        if not first:
            pretty_data.yaml_set_comment_before_after_key(key, before='\n')
        first = False

    with open(save_path, 'w') as f:
        yaml.dump(pretty_data, f)


    
GLOBAL_SEED = 12
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

def main(args: dict, *noargs, **nokwargs):
    meta_cfg: dict = args.get('meta', {})
    affine_modality = meta_cfg.get('affine_modality', None)
    
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
    normalize_targets   = augment_cfg.get('normalize_targets', False)
    
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
    
    grad_optim_cfg: dict = optim_cfg.get('gradient_optimizer', {})
    grad_optim_name = grad_optim_cfg.get('type', 'normal')
    grad_optim_params = grad_optim_cfg.get('params', {})

    loss_cfg: dict = args.get('loss', {})
    normalize_rep = loss_cfg.get('normalize_rep', False)

    meta_cfg: dict = args.get('meta', {})
    dtype                     = meta_cfg.get('dtype', 'float32')
    save_freq                 = meta_cfg.get('save_every_freq', 2)
    save_root_dir_cfg         = meta_cfg.get('save_root_dir', "./Experiment")
    sync_gc                   = meta_cfg.get('sync_gc', False)
    continue_from_path = meta_cfg.get('continue_from_path', None)
    continue_train                 = bool(continue_from_path)
    resume_prefer_best             = bool(meta_cfg.get('resume_prefer_best', True))

    checkpoint_cfg: dict = args.get('checkpoint', {})
    patience = checkpoint_cfg.get('patience', epochs)
    min_delta = checkpoint_cfg.get('min_delta', 0.0)

    logging_cfg: dict = args.get('logging', {})
    progress_type  = logging_cfg.get('progress_type', 'table')
    save_csv       = logging_cfg.get('save_csv', True)
    save_batch_csv = logging_cfg.get('save_batch_csv', False)
    save_epoch_csv = logging_cfg.get('save_epoch_csv', True)
    

    world_size, rank = init_distributed()
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        logger.CUSTOM("SUCCESS", f"DDP enabled (world_size={world_size}, rank={rank})")
    else:
        logger.INFO("DDP disabled (single-GPU/single-process mode)")
    
    
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
    
    criterion = compile_loss(loss_cfg = loss_cfg, device = device)
    if dist.is_initialized() and world_size > 1:
        criterion = DDP(criterion, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    criterion_core = _unwrap_module(criterion)
    enabled_tasks = list(getattr(criterion_core, "enabled_tasks", []))
    
    transform = compile_transform(
        random_horizontal_flip = horizontal_flip,
        random_resize_aspect_ratio = random_aspect_ratio,
        random_resize_scale = random_resize_scale,
        reprob = reprob,
        auto_augment = auto_augment,
        motion_shift = motion_shift,
        crop_size    = crop_size,
    )
    
    video_loader, val_loader, video_sampler, val_sampler, stats = compile_dataloader(
        train_cfg, 
        nclips = nclips,
        transform = transform,
        collate_fn = torch.utils.data.default_collate,
        num_workers  = num_workers,
        persistance_workers = persistent_workers,
        pin_memory = pin_mem,
        world_sz = world_size,
        rank = rank,
        normalize_targets = normalize_targets,
    )
    
    
    inv_softplus = lambda x: np.where(x > 20, x, np.log(np.exp(np.clip(x, 1e-9, 20)) - 1.0))
    probe_cfg['init_scales'] = [] if probe_cfg['init_scales'] == "auto" else None
    probe_cfg['init_shifts'] = [] if probe_cfg['init_shifts'] == "auto" else None
    for task in enabled_tasks:
        mod_stats = stats[task]
        if probe_cfg['init_scales'] is not None:
            probe_cfg['init_scales'] += [float(inv_softplus(mod_stats['std']))]
        if probe_cfg['init_shifts'] is not None:
            probe_cfg['init_shifts'] += [mod_stats['mean']]
    
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

    if dist.is_initialized() and world_size > 1:
        encoder = DDP(encoder, static_graph = True)
        probe   = DDP(probe, static_graph = False, find_unused_parameters = False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    
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


    n_tasks = len(criterion_core.enabled_tasks) if hasattr(criterion_core, 'enabled_tasks') else 1
    optim.add_param_group({
        "params": list(criterion.parameters()),
        "lr_scale": 0.1,
        "weight_decay": 0.0,
    })
    logger.INFO("Added uncertainty loss parameters to optimizer (lr_scale=0.1)")

    # Initialize gradient optimizer for multi-task learning
    grad_optim = compile_grad_optimizer(
        base_optimizer = optim,
        optimizer_name = grad_optim_name,
        n_tasks        = n_tasks,
        device         = device_type,
        **grad_optim_params
    )

    log_dir = os.path.join(save_root_dir_cfg, "probe")
    logger.INFO(f"Probe save root directory: {log_dir}")

    continue_run_dir = None
    continue_run_name = None
    if continue_train:
        continue_run_dir = os.path.abspath(os.path.expanduser(continue_from_path))
        if os.path.basename(continue_run_dir) == "weights":
            continue_run_dir = os.path.dirname(continue_run_dir)
        if not os.path.isdir(continue_run_dir):
            raise FileNotFoundError(f"continue_from_path does not exist: {continue_from_path}")
        continue_run_name = os.path.basename(continue_run_dir)
        if not continue_run_name.startswith("run"):
            raise ValueError(
                f"Expected continue_from_path to point to a run directory like '.../run1', got: {continue_run_dir}"
            )

    if rank == 0:
        if continue_train:
            resolved_run_idx = int(continue_run_name.removeprefix("run"))
            logger.INFO(f"Resuming requested. Selected run directory: {continue_run_dir}")
        else:
            next_run_idx = get_next_run(log_dir)
            resolved_run_idx = next_run_idx
        run_idx_tensor = torch.tensor([resolved_run_idx], dtype=torch.long, device=device)
    else:
        run_idx_tensor = torch.tensor([0], dtype=torch.long, device=device)

    if dist.is_initialized() and world_size > 1:
        dist.broadcast(run_idx_tensor, src=0)
    run_idx = int(run_idx_tensor.item())

    start_epoch = 0
    resume_score = None
    run_name = f"run{run_idx}"
    run_dir = os.path.join(log_dir, run_name)

    if continue_train and continue_run_dir is not None:
        run_dir = continue_run_dir
        run_name = os.path.basename(run_dir)

    if continue_train:
        resume_dir = os.path.join(run_dir, "weights")
        probe, optim, start_epoch, resume_score, resume_meta = load_checkpoint(
            model=probe,
            optimizer=optim,
            checkpoint_dir=resume_dir,
            checkpoint_name="probe.pt",
            prefer_best=resume_prefer_best,
            map_location=device,
        )
        restore_resume_state(
            resume_meta=resume_meta,
            scaler=scaler,
            criterion=criterion,
            lr_scheduler=lr_scheduler,
            wd_scheduler=wd_scheduler,
            start_epoch=start_epoch,
            ipe=ipe,
            rank=rank,
            run_idx=run_idx,
            resume_prefer_best=resume_prefer_best,
        )
    

    # Only create logger and run directories for rank 0 to avoid race conditions
    if rank == 0:
        log_stats = create_supervised_logger(
            log_dir = log_dir,
            epochs = epochs,
            run_name = run_name,
            progress_type = progress_type,
            save_csv = save_csv,
            save_batch_csv = save_batch_csv,
            save_epoch_csv = save_epoch_csv,
        )
        probe_save = EarlyStopping(
            patience = patience,
            freq = save_freq,
            min_delta = min_delta,
            path = os.path.join(run_dir, "weights/probe.pt"),
            weights_only = True,
        )
        if resume_score is not None:
            probe_save.best_loss = resume_score
        if not continue_train:
            yaml_name = f"{args['app']}-{probe_cfg['name']}-{args['common']['crop_size']}.px.yaml"
            save_config_pretty(args, os.path.join(run_dir, yaml_name))
    else:
        log_stats = NoOpLogger()
   
    if sync_gc:
        gc.disable()
        gc.collect()


    def train_step(clips, actions):
        _new_lr = lr_scheduler.step()
        _new_wd = wd_scheduler.step()
        use_multi_task_grad = grad_optim_name.lower() in {"pcgrad", "gradnorm", "famo"}
        
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
            targets = format_targets(a, actions, criterion_core.enabled_tasks)
            loss, detail = regression(a, targets)
            task_loss_map = criterion_core.compute_task_losses(a, targets, weighted=True) if use_multi_task_grad else None

        grad_optim.zero_grad()

        if use_multi_task_grad and task_loss_map is not None:
            task_losses = [task_loss_map[task_name] for task_name in criterion_core.enabled_tasks]
            if len(task_losses) == 1:
                task_losses = [loss]

            # For gradient-surgery methods, run native backward so task-specific
            # hook-captured gradients remain consistent.
            grad_optim.backward(*task_losses)
            grad_optim.step()
        else:
            if mixed_precision:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                grad_optim.step()
        
        loss_details = {
            "Total Loss": float(detail["total_loss"].item()),
        }
        for task_name, task_detail in detail["per_task"].items():
            task_title = task_name.replace("_", " ").title()
            loss_details[f"{task_title}"] = float(task_detail["base_loss"].item())
        for task_name, task_detail in detail["per_task"].items():
            task_title = task_name.replace("_", " ").title()
            loss_details[f"Weight {task_title}"] = float(task_detail["weighted_loss"].item())
        
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
            a  = probe(h)
            targets = format_targets(a, actions, criterion_core.enabled_tasks)
            loss, detail = criterion(a, targets)

        loss_details = {
            "Total Loss": float(detail["total_loss"].item()),
        }
        for task_name, task_detail in detail["per_task"].items():
            task_title = task_name.replace("_", " ").title()
            loss_details[f"{task_title}"] = float(task_detail["base_loss"].item())

        return float(loss.item()), loss_details
    
    loader = iter(video_loader)
    with log_stats:
        log_stats.start_training("Training Latent Action WM")
        video_sampler.set_epoch(0)
        last_loss = 0.0
        last_val_loss = 0.0
        curr_lr, curr_wd = 0.0, 0.0
        for epoch in range(start_epoch, epochs):
            
            # ==================================== #
            #               TRAINING
            # ==================================== #
            log_stats.start_epoch(epoch, ipe, desc = "Training")
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
                log_stats.log_batch(batch_metrics, phase="train", phase_agnostic=["LR", "WD", "GPU Timer", "Weight Lat Err", "Weight Velocity", "Weight Steer"])

            # ==================================== #
            #               VALUATING
            # ==================================== #
            if val_loader is not None and len(val_loader) > 0:
                probe.eval()
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

                probe.train()

            log_stats.log_epoch()

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
                    criterion=_unwrap_module(criterion).state_dict()
                )

            should_stop = False
            if rank == 0:
                should_stop = bool(probe_save.early_stop)

            if dist.is_initialized() and world_size > 1:
                stop_tensor = torch.tensor([int(should_stop)], device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                break