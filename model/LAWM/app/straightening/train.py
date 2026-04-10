import os, sys
import resource
import yaml
import time
import gc
import glob
import re
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
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distributed import all_gather, all_reduce

from .compile import (
    compile_model,
    compile_transform,
    compile_dataloader,
    compile_opt
)
from utils.training_logger import (
    get_next_run,
    create_self_supervised_logger,
    NoOpLogger
)
from utils.distributed import init_distributed
from utils.logger import Logger
from utils.early_stop import EarlyStopping, MultiModuleEarlyStopping

logger = Logger(__name__)

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

def load_ckpt(
    models_dict,
    optimizer,
    scaler,
    checkpoint_dir,
    prefer_best=True,
    map_location=None,
):
    meta_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = torch.load(meta_path, map_location=map_location)
    score = meta.get("score")
    best_loss = meta.get("best_loss", score)
    start_epoch = meta.get("epoch", 0)

    prefix = "best_" if prefer_best else "last_"
    missing_models = []
    for name, model in models_dict.items():
        model_path = os.path.join(checkpoint_dir, f"{prefix}{name}.pt")
        if not os.path.exists(model_path):
            missing_models.append(model_path)
            continue

        model_state = torch.load(model_path, map_location=map_location)
        model.load_state_dict(model_state)

    if missing_models:
        missing = "\n".join(missing_models)
        raise FileNotFoundError(
            f"Missing expected resume weights in {checkpoint_dir}:\n{missing}"
        )

    if optimizer is not None and meta.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(meta["optimizer_state_dict"])
    if scaler is not None and meta.get("scaler_state_dict") is not None:
        scaler.load_state_dict(meta["scaler_state_dict"])

    return models_dict, optimizer, scaler, start_epoch + 1, score, best_loss

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
ACTION_LOSS = {}
def loss_registry(name):
    def decorator(fn):
        ACTION_LOSS[name] = fn
        return fn
    return decorator

def main(args: dict, yaml_path: str):
    
    train_cfg: dict = args.get("train", {})
    patch_size   = train_cfg.get('patch_size', 16)
    tubelet_size = train_cfg.get('tubelet_size', 2)
    crop_size    = train_cfg.get('crop_size', 224)
    fpcs = train_cfg.get('fpcs', 16)
    
    loader_cfg: dict = args.get('loader_setup', {})
    num_workers        = loader_cfg.get('num_workers', train_cfg.get('num_workers', 1))
    persistent_workers = loader_cfg.get('persistent_workers', train_cfg.get('persistent_workers', False))
    pin_mem            = loader_cfg.get('pin_mem', train_cfg.get('pin_mem', False))

    model_cfg: dict    = args.get("model", {})
    enc_cfg       = model_cfg.get('enc', {})
    pred_cfg      = model_cfg.get('pred', {})
    act_cfg       = model_cfg.get('action', {})
    filter_cfg    = model_cfg.get('filter', {})
    common_cfg    = model_cfg.get('common', {})
    action_pframe = common_cfg.get('action_pframe', 1)

    augment_cfg: dict = args.get('data_aug', {})
    auto_augment        = augment_cfg.get('auto_augment', False)
    horizontal_flip     = augment_cfg.get('horizontal_flip', False)
    motion_shift        = augment_cfg.get('motion_shift', False)
    random_aspect_ratio = augment_cfg.get('random_resize_aspect_ratio', (1.0, 1.0))
    random_resize_scale = augment_cfg.get('random_resize_scale', (1.0, 1.0))
    reprob              = augment_cfg.get('reprob', 0.0)
    
    optim_cfg: dict = args.get('optimization', {})
    anneal       = optim_cfg.get('anneal', 15)
    num_epochs   = optim_cfg.get('epochs', 100)
    final_lr     = optim_cfg.get('final_lr', 0.0)
    final_wd     = optim_cfg.get("final_weight_decay", 0.0)
    ipe          = optim_cfg.get('ipe', 100)
    lr           = optim_cfg.get('lr', 1e-3)
    start_lr     = optim_cfg.get('start_lr', 1e-3)
    warmup       = optim_cfg.get('warmup', 10)
    weight_decay = optim_cfg.get('weight_decay', 0.0)
    betas        = optim_cfg.get('betas', (0.9, 0.999))
    eps          = optim_cfg.get('eps', 1.0e-8)
    ema          = optim_cfg.get('ema', [0.9, 1.0])

    init_step = 2
    loss_cfg: dict = args.get('loss', {})
    auto_steps        = min(init_step + loss_cfg.get('auto_steps', 0), fpcs // tubelet_size)
    loss_exp          = loss_cfg.get("loss_exp", 1.0)
    normalize_reps    = loss_cfg.get('normalize_reps', False)
    normalize_actions = loss_cfg.get('normalize-actions', False)
    reg_type          = loss_cfg.get("reg_name", "energy")
    l1_energy         = loss_cfg.get('l1', 1.0)
    l2_energy         = loss_cfg.get('l2', 0.0)
    lv_vcm            = loss_cfg.get('lv', 0.0)
    lc_vcm            = loss_cfg.get('lc', 0.0)
    lm_vcm            = loss_cfg.get('lm', 0.0)
    l_curve           = loss_cfg.get('lcurve', 1.0)
    sig_weight        = loss_cfg.get("weight", 1.0)
    num_proj          = loss_cfg.get('num_proj', 128)
    samp_range        = loss_cfg.get('samp_range', [-1, 1])
    samp_sz           = loss_cfg.get('samp_sz', 16)
    cov_coeff         = loss_cfg.get('cov_coeff', 0.4)
    std_coeff         = loss_cfg.get('std_coeff', 0.4)

    
    meta_cfg: dict = args.get('meta', {})
    dtype              = meta_cfg.get('dtype', 'float32')
    save_freq          = meta_cfg.get('save_every_freq', 2)
    seed               = meta_cfg.get('seed', 0)
    sync_gc            = meta_cfg.get('sync_gc', False)
    save_root_dir      = meta_cfg.get('save_root_dir', "./")
    continue_from_path = meta_cfg.get('continue_from_path', None)
    continue_train     = bool(continue_from_path)
    resume_prefer_best = bool(meta_cfg.get('resume_prefer_best', True))
    tokens_pframe      = (crop_size // patch_size) ** 2

    checkpoint_cfg: dict = args.get('checkpoint', {})
    patience = checkpoint_cfg.get('patience', num_epochs)
    min_delta = checkpoint_cfg.get('min_delta', 0.0)

    logging_cfg: dict = args.get('logging', {})
    progress_type         = logging_cfg.get('progress_type', 'table')
    save_csv              = logging_cfg.get('save_csv', True)
    save_batch_csv        = logging_cfg.get('save_batch_csv', False)
    save_epoch_csv        = logging_cfg.get('save_epoch_csv', True)
    log_batch_tensorboard = logging_cfg.get('log_batch_tensorboard', False)
    log_model_graph       = logging_cfg.get('log_model_graph', False)

    logger.WARNING(f"Autostep currently selected as {auto_steps} steps")

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
    encoder, filterer, target_filterer, agg, lpred, apred = compile_model(
        enc_cfg = enc_cfg,
        lpred_cfg = pred_cfg,
        apred_cfg = act_cfg,
        filter_cfg = filter_cfg,
        device = device
    )

    if model_cfg.get('compile', False):
        logger.INFO("Compiling model")
        torch._dynamo.config.optimize_ddp = False
        target_filterer.compile()
        agg.compile()
        filterer.compile()
        encoder.compile()
        lpred.compile()
        apred.compile()

    if dist.is_initialized() and world_size > 1:
        agg              = DDP(agg, static_graph = False, find_unused_parameters = True)
        encoder          = DDP(encoder, static_graph = True)
        target_filterer  = DDP(target_filterer, static_graph = True)
        filterer         = DDP(filterer, static_graph = False, find_unused_parameters = True)
        lpred            = DDP(lpred, static_graph = False, find_unused_parameters = True)
        apred            = DDP(apred, static_graph = False, find_unused_parameters = True)
    for p in target_filterer.parameters():
        p.requires_grad = False
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
    
    video_loader, video_sampler = compile_dataloader(
        train_cfg, 
        transform = transform,
        collate_fn = torch.utils.data.default_collate,
        num_workers  = num_workers,
        persistance_workers = persistent_workers,
        pin_memory = pin_mem,
        world_sz = world_size,
        rank = rank
    )
    
    optim, scaler, lr_scheduler, wd_scheduler = compile_opt(
        filterer             = filterer,
        apred                = apred,
        lpred                = lpred,
        iterations_per_epoch = ipe,
        start_lr             = start_lr,
        warmup               = warmup, 
        anneal               = anneal,
        num_epochs           = num_epochs,
        wd                   = weight_decay,
        final_lr             = final_lr,
        mixed_precision      = mixed_precision,
        betas                = betas,
        eps                  = eps,
        ref_lr               = lr,
        final_wd             = final_wd
    )
    
    log_dir = os.path.join(save_root_dir, "straightening")
    logger.INFO(f"Straightening save root directory: {log_dir}")

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
            resolved_run_idx = get_next_run(log_dir)
        run_idx_tensor = torch.tensor([resolved_run_idx], dtype=torch.long, device=device)
    else:
        run_idx_tensor = torch.tensor([0], dtype=torch.long, device=device)

    if dist.is_initialized() and world_size > 1:
        dist.broadcast(run_idx_tensor, src=0)
    run_idx = int(run_idx_tensor.item())

    start_epoch = 0
    resume_score = None
    resume_best_loss = None
    run_name = f"run{run_idx}"
    run_dir = os.path.join(log_dir, run_name)

    if continue_train and continue_run_dir is not None:
        run_dir = continue_run_dir
        run_name = os.path.basename(run_dir)
        models_to_resume = {
            "filter": filterer,
            "target_filter": target_filterer,
            "agg": agg,
            "lpred": lpred,
            "apred": apred,
        }
        (
            resumed_models,
            optim,
            scaler,
            start_epoch,
            resume_score,
            resume_best_loss,
        ) = load_ckpt(
            models_dict=models_to_resume,
            optimizer=optim,
            scaler=scaler,
            checkpoint_dir=os.path.join(run_dir, "weights"),
            prefer_best=resume_prefer_best,
            map_location=device,
        )
        filterer = resumed_models["filter"]
        target_filterer = resumed_models["target_filter"]
        agg = resumed_models["agg"]
        lpred = resumed_models["lpred"]
        apred = resumed_models["apred"]
        logger.INFO(
            f"Resumed straightening from {run_dir} at epoch {start_epoch} "
            f"using {'best' if resume_prefer_best else 'latest last'} checkpoints"
        )

    # Only create logger and run directories for rank 0 to avoid race conditions.
    if rank == 0:
        log_stats = create_self_supervised_logger(
            log_dir = log_dir,
            epochs = num_epochs,
            run_name = run_name,
            progress_type = progress_type,
            save_csv = save_csv,
            save_batch_csv = save_batch_csv,
            save_epoch_csv = save_epoch_csv,
            log_batch_tensorboard = log_batch_tensorboard,
        )
        saver = MultiModuleEarlyStopping(
            patience = patience,
            freq = save_freq,
            path_root = os.path.join(run_dir, "weights"),
            weights_only = True
        )
        if resume_best_loss is not None:
            saver.best_loss = resume_best_loss
        elif resume_score is not None:
            saver.best_loss = resume_score
        if not continue_train:
            yaml_name = f"{args['app']}-{model_cfg.get('filter', {}).get('name', 'model')}-{reg_type}-{crop_size}px.yaml"
            save_config_pretty(args, os.path.join(run_dir, yaml_name))

        if log_model_graph:
            class _FilterTraceWrapper(torch.nn.Module):
                def __init__(self, model: torch.nn.Module, H: int, W: int):
                    super().__init__()
                    self.model = model
                    self.H = H
                    self.W = W

                def forward(self, latent: torch.Tensor):
                    return self.model(latent, self.H, self.W)

            model_dim = filter_cfg.get('filter_dim', pred_cfg.get('pred_embed_dim', 768))
            action_dim = act_cfg.get('action_embed_dim', 64)
            n_frames = max(1, fpcs // tubelet_size)
            n_ctx_frames = max(1, n_frames - 1)
            h_patches = crop_size // patch_size
            w_patches = crop_size // patch_size

            # -- apred(h_ctx, h_goal)
            apred_ctx = torch.randn(1, n_ctx_frames * tokens_pframe, model_dim, device=device)
            apred_goal = torch.randn(1, tokens_pframe, model_dim, device=device)

            # -- lpred(h_ctx, a_ctx)
            lpred_h = torch.randn(1, n_ctx_frames * tokens_pframe, model_dim, device=device)
            lpred_a = torch.randn(1, n_ctx_frames * action_pframe, action_dim, device=device)

            # -- filterer(latent, H, W)
            filter_latent = torch.randn(1, n_frames * tokens_pframe, model_dim, device=device)

            apred_model = apred.module if isinstance(apred, DDP) else apred
            lpred_model = lpred.module if isinstance(lpred, DDP) else lpred
            filter_model = filterer.module if isinstance(filterer, DDP) else filterer
            filter_trace_model = _FilterTraceWrapper(filter_model, h_patches, w_patches)

            log_stats.log_model_graph(apred_model, (apred_ctx, apred_goal))
            log_stats.log_model_graph(lpred_model, (lpred_h, lpred_a))
            log_stats.log_model_graph(filter_trace_model, (filter_latent,))
            logger.INFO("Logged model graphs to TensorBoard: filterer, lpred, apred")
    else:
        log_stats = NoOpLogger()
   
    if sync_gc:
        gc.disable()
        gc.collect()

    loader = iter(video_loader)

    momentum_scheduler = (
        ema[0] + i * (ema[1] - ema[0]) / (ipe * num_epochs)
        for i in range(int(ipe * num_epochs) + 1)
    )

    def train_step(clips):
        _new_lr = lr_scheduler.step()
        _new_wd = wd_scheduler.step()

        def forward_context(latent: torch.Tensor, H: int, W: int):
            h: torch.Tensor = filterer(latent, H, W)
            if normalize_reps:
                h = F.layer_norm(h, (h.size(-1), ))
            return h
        
        def forward_target(latent: torch.Tensor, H: int, W: int):
            with torch.no_grad():
                h: torch.Tensor = target_filterer(latent, H, W)
                if normalize_reps:
                    h = F.layer_norm(h, (h.size(-1), ))
                return h

        def to_latent(c: torch.Tensor):
            with torch.no_grad():
                latent: torch.Tensor = encoder(c)
                if normalize_reps:
                    latent = F.layer_norm(latent, (latent.size(-1), ))
                return latent

        def forward_prediction(h_ctx: torch.Tensor, h_goal: torch.Tensor, T: int):
            def _step_action(h, g):
                _a: torch.Tensor = apred(h, g, T = T)
                if normalize_actions:
                    _a = F.layer_norm(_a, (_a.size(-1), ))
                return _a
            
            def _step_prediction(h, a):
                _z: torch.Tensor = lpred(h, a)
                if normalize_reps:
                    _z = F.layer_norm(_z, (_z.size(-1), ))
                return _z

            # -- Teacher forcing entire timestep action + prediction
            _a_tf = _step_action(h_ctx, h_goal)
            _z_tf = _step_prediction(h_ctx, _a_tf)


            # -- Autoregressive rollout of each timestep action and prediction
            z_ctx = torch.cat([h_ctx[:, :tokens_pframe], _z_tf[:, :tokens_pframe]], dim = 1)
            for n in range(init_step, auto_steps):
                # -- Consider chunking?
                # -- Since the latent is predicted on action, the action must not drift
                a_ctx = _step_action(z_ctx, h_goal)
                # a_ctx = _a_tf[:, :n * action_pframe]

                # -- Prediction shifting all frames to 1 timestep to the future
                z_nxt = _step_prediction(z_ctx, a_ctx)[:, -tokens_pframe: ]
                z_ctx = torch.cat([z_ctx, z_nxt], dim = 1)
            _z_ar = z_ctx[:, tokens_pframe: ]
            
            return _z_tf, _z_ar, _a_tf
            
        def latent_loss(h, z):
            sub_h = h[:, tokens_pframe: z.size(1) + tokens_pframe]
            return torch.mean(torch.abs(z - sub_h) ** loss_exp) / loss_exp

        @loss_registry("sigreg")
        def sigreg(a: torch.Tensor):
            device = a.device

            t = torch.linspace(*samp_range, samp_sz, device = device)
            exp_f = torch.exp(-0.5 * (t**2))
            g = torch.Generator(device=device).manual_seed(42) 
            u = torch.randn(a.size(2), num_proj, device = device, generator = g)
            u /= u.norm(p = 2, dim = 0)
            
            proj = (a @ u) # -- B, N, M
            ecf = (1j * proj.unsqueeze(-1) * t).exp().mean(dim = (0, 1))
            
            ecf = all_reduce(ecf)
            
            err = ((ecf - exp_f).abs() ** 2) * exp_f
            area = torch.trapz(err, t, dim = 1)
            return area.mean() * sig_weight, None, None

        @loss_registry("energy")
        def action_loss(a):
            def energy(a):
                
                """High energy landscape with sparse, defined actions"""
                
                D = a.size(-1)
                # -- Prevents vanishing signals
                hinge = torch.relu(D ** 0.5 - (a ** 2).sum(-1)) * l2_energy
                # -- Sparse action (clearly defined action)
                sparsity = torch.abs(a).sum(-1) * l1_energy

                return (hinge + sparsity).mean(), hinge.mean().item(), sparsity.mean().item()

            def vcm(a):

                """Laziness not permitted"""

                a = all_gather(a)

                N, D = a.size(0) * a.size(1), a.size(2)
                a = a.reshape(N, D)

                # -- Ensure each sample in batch is different (prevent collapse)
                # -- A pure hinge saturates at 1.0 when std->0; the log barrier keeps pressure near collapse.
                std = torch.std(a, dim = 0, unbiased = False)
                variance = torch.mean((1 - std) ** 2) * lv_vcm

                # -- Prevent static action to have value different than 0
                mean = a.mean().abs() * lm_vcm

                # -- Ensure each variable is independent (maximize information capacity)
                # -- Cov is rank deficient => Condition N >> D must satisfied
                a = a - a.mean(dim = 0)
                cov = (a.T @ a) / (N - 1)
                diag_mask = ~torch.eye(D, device = a.device).bool()
                covariance = cov[diag_mask].pow(2).mean() * lc_vcm
                return (
                    covariance + mean + variance,
                    covariance.item(),
                    mean.item(),
                    variance.item(),
                    std.mean().item()
                )
                
            vcm_loss, covariance, mean, variance, std = vcm(a)
            energy_loss, hinge, sparsity = energy(a)
            return vcm_loss + energy_loss, (hinge, sparsity), (variance, covariance, mean, std)

        def straighten_loss(h: torch.Tensor):
            B, _, D = h.shape
            _h = h.view(B, tokens_pframe, -1, D)
            _h = agg(_h)
            v = torch.diff(_h, dim = 2)
            v0 = v[:, :, :-1]
            v1 = v[:, :, 1:]
            cos_sim = torch.cosine_similarity(v0, v1, dim=-1)
            return (1 - cos_sim).mean() * l_curve
        
        def collapse_loss(enc_h: torch.Tensor, filter_h: torch.Tensor):
            enc_h = enc_h.float()
            filter_h = filter_h.float()

            with torch.no_grad():
                enc_flat = enc_h.reshape(-1, latent_ctx.size(-1))
                enc_flat = enc_flat - enc_flat.mean(dim=0)
                
                target_std = torch.sqrt(enc_flat.var(dim=0) + 1e-6)
                
            filter_flat = filter_h.reshape(-1, filter_h.size(-1))
            filter_flat_centered = filter_flat - filter_flat.mean(dim=0)
            filter_std = torch.sqrt(filter_flat.var(dim=0) + 1e-6)
            std_loss = torch.mean(torch.relu(target_std - filter_std))
            
            N, D = filter_flat.shape
            filter_cov = (filter_flat_centered.T @ filter_flat_centered) / (N - 1)
            diag_mask = torch.eye(D, device=filter_h.device).bool()
            cov_loss = filter_cov[~diag_mask].pow(2).mean()

            return (std_loss * std_coeff) + (cov_loss * cov_coeff)
                

        with torch.amp.autocast(device_type, dtype = dtype, enabled = mixed_precision):

            latent_ctx  = to_latent(clips[:, :, :-1])
            latent_goal = to_latent(clips[:, :, -1:])
            H_patches = clips.shape[3] // patch_size
            W_patches = clips.shape[4] // patch_size
            T = (latent_ctx.shape[1] + latent_goal.shape[1]) // tokens_pframe
            
            
            h_ctx  = forward_context(latent_ctx, H_patches, W_patches)
            h_goal = forward_target(latent_goal, H_patches, W_patches)
            h = torch.concat([h_ctx, h_goal], dim = 1)

            
            z_tf, z_ar, a_tf = forward_prediction(h_ctx, h_goal, T)
            loss_tf                    = latent_loss(h, z_tf)
            loss_ar                    = latent_loss(h, z_ar)
            loss_straight              = straighten_loss(h)
            loss_collapse              = collapse_loss(latent_ctx, h_ctx)
            loss_act, energy, vcm = ACTION_LOSS[reg_type](a_tf)
            loss = loss_tf + loss_ar + loss_act + loss_straight + loss_collapse
            
            
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

        m = next(momentum_scheduler)
        with torch.no_grad():
            params_k = []
            params_q = []
            for param_q, param_k in zip(filterer.parameters(), target_filterer.parameters()):
                params_k.append(param_k)
                params_q.append(param_q)
            torch._foreach_mul_(params_k, m)
            torch._foreach_add_(params_k, params_q, alpha=1 - m)
        
        loss = loss.item()
        loss_tf = loss_tf.item()
        loss_ar = loss_ar.item()
        loss_act = loss_act.item()
        
        return (
            loss, 
            loss_tf,
            loss_ar,
            loss_act,
            loss_straight,
            loss_collapse,
            energy,
            vcm,
            _new_lr,
            _new_wd
        )
        
    
    with log_stats:
        log_stats.start_training("Training Filtering Latent Action WM")
        video_sampler.set_epoch(0)
        for epoch in range(num_epochs):
            
            log_stats.start_epoch(epoch, ipe, desc = "Training")
            
            for itr in log_stats.batch_iterator([i for i in range(ipe)]):
                
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
                            logger.ERROR("Exceeded maximum retries when iterating dataloade. Please check for error", exit_code = 5, full_traceback = e)
                        
                clips = sample.to(device)
                
                (loss, loss_tf, loss_ar, loss_act, loss_collapse, loss_straight, energy, vcm, curr_lr, curr_wd), elapsed_time = gpu_timer(partial(train_step, clips))

                if np.isnan(loss) or np.isinf(loss):
                    logger.ERROR(f"Model failed to converge. {'nan' if np.isnan(loss) else 'inf' if np.isinf(loss) else ''} detected", exit_code = -213)
                
                if energy and vcm:
                    log_stats.log_batch({
                        "LR": curr_lr,
                        "WD": curr_wd, 
                        "Loss": loss,
                        "Teach Force|Z": loss_tf,
                        "Autoregressive|Z": loss_ar,
                        "Action": loss_act,
                        "Collapse": loss_collapse,
                        "Straight": loss_straight,
                        "Hinge|Erg": energy[0],
                        "Sparsity|Erg": energy[1],
                        "Variance|VCM": vcm[0],
                        "Covariance|VCM": vcm[1],
                        "Mean|VCM": vcm[2],
                        "Std": vcm[3],
                    })
                else:
                    log_stats.log_batch({
                        "LR": curr_lr,
                        "WD": curr_wd, 
                        "Loss": loss,
                        "Teach Force|Z": loss_tf,
                        "Autoregressive|Z": loss_ar,
                        "Action | SIGReg": loss_act,
                        "Collapse": loss_collapse,
                        "Straight": loss_straight,
                    })
                    
            
            log_stats.log_epoch()
            
            gc.collect()
            
            if rank == 0:
                models_to_save = {
                    "filter": filterer,
                    "target_filter": target_filterer,
                    "agg": agg,
                    "lpred": lpred,
                    "apred": apred
                }
                saver(
                    score=log_stats.get_metric("Loss", "train"), 
                    models_dict=models_to_save, 
                    optimizer=optim, 
                    scaler=scaler, 
                    epoch=epoch
                )
                
                if saver.early_stop:
                    logger.INFO("Early stopping triggered")

            should_stop = False
            if rank == 0:
                should_stop = bool(saver.early_stop)

            if dist.is_initialized() and world_size > 1:
                stop_tensor = torch.tensor([int(should_stop)], device=device)
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                break