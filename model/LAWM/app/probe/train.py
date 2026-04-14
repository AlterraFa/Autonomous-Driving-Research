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
    compile_fdat_loss,
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


def gpu_timer(funct, log_timming=True):
    log_timming = log_timming and torch.cuda.is_available()
    elapsed_time = -1.0
    if log_timming:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    result = funct()
    if log_timming:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
    return result, elapsed_time


def save_config_pretty(config_dict, save_path):
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.default_flow_style = False

    from ruamel.yaml.comments import CommentedMap

    def dict_to_commented(d):
        if isinstance(d, dict):
            cm = CommentedMap()
            for k, v in d.items():
                cm[k] = dict_to_commented(v)
            return cm
        return d

    pretty_data = dict_to_commented(config_dict)
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
    # ======================== Config unpacking ========================
    train_cfg: dict = args.get("train", {})
    crop_size = train_cfg.get('crop_size', 256)
    fpcs = train_cfg.get('fpcs', 12)

    loader_cfg: dict = args.get('loader_setup', {})
    num_workers = loader_cfg.get('num_workers', 4)
    persistent_workers = loader_cfg.get('persistent_workers', True)
    pin_mem = loader_cfg.get('pin_mem', True)

    model_cfg: dict = args.get("model", {})
    enc_cfg = model_cfg.get('enc', {})
    probe_cfg = model_cfg.get('probe', {})
    world_model_cfg = model_cfg.get('world_model', {})

    augment_cfg: dict = args.get('data_aug', {})
    auto_augment = augment_cfg.get('auto_augment', False)
    horizontal_flip = augment_cfg.get('horizontal_flip', False)
    motion_shift = augment_cfg.get('motion_shift', False)
    random_aspect_ratio = augment_cfg.get('random_resize_aspect_ratio', (1.0, 1.0))
    random_resize_scale = augment_cfg.get('random_resize_scale', (1.0, 1.0))
    reprob = augment_cfg.get('reprob', 0.0)

    optim_cfg: dict = args.get('optimization', {})
    anneal = optim_cfg.get('anneal', optim_cfg.get('annel', 15))
    epochs = optim_cfg.get('epochs', 50)
    final_lr = optim_cfg.get('final_lr', 0.0)
    final_wd = optim_cfg.get("final_weight_decay", 0.0)
    ipe = optim_cfg.get('ipe', 100)
    lr = optim_cfg.get('lr', 1e-3)
    start_lr = optim_cfg.get('start_lr', 1e-4)
    warmup = optim_cfg.get('warmup', 5)
    weight_decay = optim_cfg.get('weight_decay', 0.01)
    betas = optim_cfg.get('betas', (0.9, 0.999))
    eps = optim_cfg.get('eps', 1.0e-8)

    loss_cfg: dict = args.get('loss', {})

    meta_cfg: dict = args.get('meta', {})
    dtype = meta_cfg.get('dtype', 'bfloat16')
    save_freq = meta_cfg.get('save_every_freq', 5)
    save_root_dir_cfg = meta_cfg.get('save_root_dir', "./Experiment")
    sync_gc = meta_cfg.get('sync_gc', False)
    seed = meta_cfg.get('seed', 239)
    continue_from_path = meta_cfg.get('continue_from_path', None)
    continue_train = bool(continue_from_path)
    resume_prefer_best = bool(meta_cfg.get('resume_prefer_best', True))

    checkpoint_cfg: dict = args.get('checkpoint', {})
    patience = checkpoint_cfg.get('patience', epochs)
    min_delta = checkpoint_cfg.get('min_delta', 0.0)

    logging_cfg: dict = args.get('logging', {})
    progress_type = logging_cfg.get('progress_type', 'table')
    save_csv = logging_cfg.get('save_csv', True)
    save_batch_csv = logging_cfg.get('save_batch_csv', False)
    save_epoch_csv = logging_cfg.get('save_epoch_csv', True)

    # ======================== Distributed setup ========================
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

    torch.manual_seed(seed)
    torch.cuda.set_device(rank)
    device_type = f'cuda:{rank}'
    device = torch.device(device_type)

    # ======================== Compile components ========================
    criterion = compile_fdat_loss(loss_cfg=loss_cfg, device=device)

    transform = compile_transform(
        random_horizontal_flip=horizontal_flip,
        random_resize_aspect_ratio=random_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    video_loader, val_loader, video_sampler, val_sampler, _ = compile_dataloader(
        train_cfg,
        nclips=1,
        transform=transform,
        collate_fn=torch.utils.data.default_collate,
        num_workers=num_workers,
        persistance_workers=persistent_workers,
        pin_memory=pin_mem,
        world_sz=world_size,
        rank=rank,
        dataset_type="straightening_probe",
    )

    world_model, decoder = compile_model(
        enc_cfg=enc_cfg,
        probe_cfg=probe_cfg,
        world_model_cfg=world_model_cfg,
        device=device,
    )

    if model_cfg.get('compile', False):
        logger.INFO("Compiling decoder")
        torch._dynamo.config.optimize_ddp = False
        decoder = torch.compile(decoder)

    if dist.is_initialized() and world_size > 1:
        decoder = DDP(decoder, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    optim, scaler, lr_scheduler, wd_scheduler = compile_opt(
        encoder=world_model,  # unused for param groups, but kept for interface compat
        probe=decoder,
        iterations_per_epoch=ipe,
        start_lr=start_lr,
        warmup=warmup,
        anneal=anneal,
        num_epochs=epochs,
        wd=weight_decay,
        final_lr=final_lr,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
        ref_lr=lr,
        final_wd=final_wd,
    )

    # ======================== Run directory ========================
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

    if rank == 0:
        if continue_train:
            resolved_run_idx = int(continue_run_name.removeprefix("run"))
        else:
            resolved_run_idx = get_next_run(log_dir)
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

    # Resume decoder checkpoint
    if continue_train:
        resume_dir = os.path.join(run_dir, "weights")
        ckpt_path = os.path.join(resume_dir, "best_decoder.pt" if resume_prefer_best else "last_decoder.pt")
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            core = decoder.module if hasattr(decoder, 'module') else decoder
            core.load_state_dict(state)
            logger.INFO(f"Resumed decoder from {ckpt_path}")

        meta_path = os.path.join(resume_dir, "checkpoint.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location=device, weights_only=False)
            start_epoch = meta.get("epoch", 0) + 1
            resume_score = meta.get("score")
            if meta.get("optimizer_state_dict"):
                optim.load_state_dict(meta["optimizer_state_dict"])
            # Advance schedulers
            for _ in range(start_epoch * ipe):
                lr_scheduler.step()
                wd_scheduler.step()
            logger.INFO(f"Resumed from epoch {start_epoch}, score={resume_score}")

    # ======================== Logger + Early stopping ========================
    if rank == 0:
        log_stats = create_supervised_logger(
            log_dir=log_dir,
            epochs=epochs,
            run_name=run_name,
            progress_type=progress_type,
            save_csv=save_csv,
            save_batch_csv=save_batch_csv,
            save_epoch_csv=save_epoch_csv,
        )
        decoder_save = EarlyStopping(
            patience=patience,
            freq=save_freq,
            min_delta=min_delta,
            path=os.path.join(run_dir, "weights/decoder.pt"),
            weights_only=True,
        )
        if resume_score is not None:
            decoder_save.best_loss = resume_score
        if not continue_train:
            yaml_name = f"probe-action-{crop_size}px.yaml"
            save_config_pretty(args, os.path.join(run_dir, yaml_name))
    else:
        log_stats = NoOpLogger()

    if sync_gc:
        gc.disable()
        gc.collect()

    # ======================== Training Loop ========================
    n_waypoints = train_cfg.get('n_waypoints', 12)
    decoder_type = probe_cfg.get('decoder', {}).get('type', 'ActionDecoder')

    def train_step(clips, gt):
        _new_lr = lr_scheduler.step()
        _new_wd = wd_scheduler.step()

        with torch.amp.autocast(device_type, dtype=dtype, enabled=mixed_precision):
            # Forward frozen world model to get action latents
            with torch.no_grad():
                a_latent = world_model(clips)  # [B, T_act, action_embed_dim]

            # Decode action latents to waypoints
            if decoder_type == 'EfficientProbe':
                # EfficientProbe expects [B, N, D] and outputs [B, output_dim]
                pred_flat = decoder(a_latent)  # [B, n_waypoints*2]
                pred_wp = pred_flat.view(-1, n_waypoints, 2)
            else:
                pred_wp = decoder(a_latent)  # [B, n_waypoints, 2]

            gt_wp = gt['midlane_wp'].to(device=clips.device, dtype=pred_wp.dtype)
            gate_score = gt['gate_score'].to(device=clips.device, dtype=pred_wp.dtype)

            loss_dict = criterion(pred_wp, gt_wp, gate_score=gate_score)
            loss = loss_dict['total'].mean()

        if mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            scaler.step(optim)
            scaler.update()
        else:
            loss.backward()
            optim.step()
        optim.zero_grad()

        details = {
            "Total": loss.item(),
            "Frenet": loss_dict['frenet'].mean().item(),
            "Heading": loss_dict['heading'].mean().item(),
            "Smooth": loss_dict['smooth'].mean().item(),
        }

        return loss.item(), _new_lr, _new_wd, details

    @torch.no_grad()
    def val_step(clips, gt):
        with torch.amp.autocast(device_type, dtype=dtype, enabled=mixed_precision):
            a_latent = world_model(clips)

            if decoder_type == 'EfficientProbe':
                pred_flat = decoder(a_latent)
                pred_wp = pred_flat.view(-1, n_waypoints, 2)
            else:
                pred_wp = decoder(a_latent)

            gt_wp = gt['midlane_wp'].to(device=clips.device, dtype=pred_wp.dtype)
            gate_score = gt['gate_score'].to(device=clips.device, dtype=pred_wp.dtype)

            loss_dict = criterion(pred_wp, gt_wp, gate_score=gate_score)
            loss = loss_dict['total'].mean()

        details = {
            "Total": loss.item(),
            "Frenet": loss_dict['frenet'].mean().item(),
            "Heading": loss_dict['heading'].mean().item(),
            "Smooth": loss_dict['smooth'].mean().item(),
        }
        return loss.item(), details

    loader = iter(video_loader)
    ipe = len(video_loader)
    with log_stats:
        log_stats.start_training("Training Action Latent Waypoint Probe")
        video_sampler.set_epoch(0)
        last_loss = 0.0
        last_val_loss = 0.0
        curr_lr, curr_wd = 0.0, 0.0

        for epoch in range(start_epoch, epochs):

            # ==================================== #
            #               TRAINING
            # ==================================== #
            decoder.train()
            log_stats.start_epoch(epoch, ipe, desc="Training")
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
                            logger.WARNING(f"Dataloader error: {e}")
                            iter_retries += 1
                            time.sleep(2)
                        else:
                            logger.ERROR("Exceeded max retries on dataloader", exit_code=5, full_traceback=e)

                clips, gt = sample
                clips = clips.to(device)

                (last_loss, curr_lr, curr_wd, details), elapsed = gpu_timer(
                    partial(train_step, clips, gt)
                )

                if np.isnan(last_loss) or np.isinf(last_loss):
                    logger.ERROR(f"Diverged: {'nan' if np.isnan(last_loss) else 'inf'}", exit_code=-213)

                log_stats.log_batch({
                    "LR": curr_lr,
                    "WD": curr_wd,
                    **details,
                })

            # ==================================== #
            #              VALIDATION
            # ==================================== #
            decoder.eval()
            val_losses = []
            val_details_accum = {}
            for val_batch in val_loader:
                val_clips, val_gt = val_batch
                val_clips = val_clips.to(device)
                v_loss, v_details = val_step(val_clips, val_gt)
                val_losses.append(v_loss)
                for k, v in v_details.items():
                    val_details_accum.setdefault(k, []).append(v)

            last_val_loss = np.mean(val_losses) if val_losses else 0.0
            val_metrics = {f"Val {k}": np.mean(v) for k, v in val_details_accum.items()}

            log_stats.end_epoch({
                "Train Loss": last_loss,
                "Val Loss": last_val_loss,
                **val_metrics,
            })

            # ==================================== #
            #            CHECKPOINTING
            # ==================================== #
            if rank == 0:
                core_decoder = decoder.module if hasattr(decoder, 'module') else decoder
                decoder_save(
                    last_val_loss,
                    core_decoder,
                    epoch=epoch,
                    optimizer=optim.state_dict(),
                    scaler=scaler.state_dict() if scaler is not None else None,
                )

            if sync_gc:
                gc.collect()
    #             break