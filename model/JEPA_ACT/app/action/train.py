import os, sys
import resource
import yaml
import time
import gc
from functools import partial
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
FOLDER_DIR = os.path.dirname(os.path.dirname(__file__))

import torch
import torch.nn.functional as F

from model.JEPA_ACT.train_script.action.compile import (
    compile_model,
    compile_transform,
    compile_dataloader,
    compile_opt
)
from model.training_logger import (
    TrainingLogger, 
    get_next_run,
    create_self_supervised_logger
)
from utils.messages.logger import Logger
from model.early_stop import EarlyStopping
from torch.nn import functional as F

logger = Logger()

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


def load_checkpoint(model, optimizer, checkpoint_dir, prefer_best=True, map_location=None):
    basename = "checkpoint.pt"
    meta_path = os.path.join(checkpoint_dir, basename)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = torch.load(meta_path, map_location=map_location)
    score = meta.get("score")
    start_epoch = meta.get("epoch", 0)

    prefix = "best_" if prefer_best else "last_"
    model_path = os.path.join(checkpoint_dir, f"{prefix}{basename}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}")

    loaded_model = torch.load(model_path, map_location=map_location)
    model.load_state_dict(loaded_model.state_dict())
    if optimizer is not None and meta.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(meta["optimizer_state_dict"])

    return model, optimizer, start_epoch + 1, score

def load_multi_checkpoint(
    models: dict,            # {"encoder": enc_model, "target": tgt_model, "predictor": pred_model}
    checkpoint_dir: str,
    prefer_best: bool = True,
    optimizer: torch.optim.Optimizer | None = None,
    map_location=None,
):
    meta_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = torch.load(meta_path, map_location=map_location)
    score = meta.get("score")
    start_epoch = meta.get("epoch", 0)
    prefix = "best_" if prefer_best else "last_"

    def _load_model_file(model_key: str, filename_suffix: str):
        model_path = os.path.join(checkpoint_dir, f"{prefix}checkpoint{filename_suffix}.pt")
        if not os.path.exists(model_path):
            return  # allow missing optional models
        loaded = torch.load(model_path, map_location=map_location)
        models[model_key].load_state_dict(loaded.state_dict())

    _load_model_file("encoder", "")

    if "target" in models:
        _load_model_file("target", "_target")
    if "predictor" in models:
        _load_model_file("predictor", "_predictor")

    if optimizer is not None and meta.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(meta["optimizer_state_dict"])

    return models, optimizer, start_epoch + 1, score

def main(args: dict, yaml_path: str):
    
    train_cfg: dict = args.get("train", {})
    patch_size   = train_cfg.get('patch_size', 16)
    tubelet_size = train_cfg.get('tubelet_size', 2)
    crop_size    = train_cfg.get('crop_size', 224)
    fps          = train_cfg.get('fps', 2)
    nclips       = train_cfg.get('nclips', 1)
    ctx_fpcs     = train_cfg.get('ctx_fpcs', 8)
    pred_fpcs    = train_cfg.get('pred_fpcs', 8)
    
    loader_cfg: dict = args.get('loader_setup', {})
    num_workers        = loader_cfg.get('num_workers', 1)
    persistent_workers = loader_cfg.get('persistent_workers', False)
    pin_mem            = loader_cfg.get('pin_mem', False)

    model_cfg: dict = args.get("model", {})
    enc_cfg   = model_cfg.get('enc', {})
    pred_cfg  = model_cfg.get('pred', {})
    act_cfg   = model_cfg.get('action', {})
    common_cfg = model_cfg.get('common', {})
    action_pframe = common_cfg.get('action_pframe', 1)

    augment_cfg: dict = args.get('data_aug', {})
    auto_augment        = augment_cfg.get('auto_augment', False)
    horizontal_flip     = augment_cfg.get('horizontal_flip', False)
    motion_shift        = augment_cfg.get('motion_shift', False)
    random_aspect_ratio = augment_cfg.get('random_resize_aspect_ratio', (1.0, 1.0))
    random_resize_scale = augment_cfg.get('random_resize_scale', (1.0, 1.0))
    reprob              = augment_cfg.get('reprob', 0.0)
    
    optim_cfg: dict = args.get('optimization', {})
    annel        = optim_cfg.get('annel', 1)
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

    init_step = 2
    loss_cfg: dict = args.get('loss', {})
    auto_steps    = min(init_step + loss_cfg.get('auto_steps', 0), (ctx_fpcs + pred_fpcs) // tubelet_size)
    loss_exp      = loss_cfg.get("loss_exp", 1.0)
    normalize_rep = loss_cfg.get('normalize_rep', False)
    reg_coeff     = loss_cfg.get('reg_coeff', 0.0)
    l1_energy     = loss_cfg.get('l1', 1.0)
    l2_energy     = loss_cfg.get('l2', 0.0)
    lv_vcm        = loss_cfg.get('lv', 0.0)
    lc_vcm        = loss_cfg.get('lc', 0.0)
    lm_vcm        = loss_cfg.get('lm', 0.0)

    meta_cfg: dict = args.get('meta', {})
    dtype = meta_cfg.get('dtype', 'float32')
    save_freq = meta_cfg.get('save_every_freq', 2)
    seed      = meta_cfg.get('seed', 0)
    
    
    
    if dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False
    
    tokens_pframe = (crop_size // patch_size) ** 2
    
    device_type = 'cuda'
    device = torch.device(device_type)
    encoder, lpred, apred = compile_model(
        enc_cfg = enc_cfg,
        lpred_cfg = pred_cfg,
        apred_cfg = act_cfg,
        device = device
    )

    if model_cfg.get('compile', False):
        print("Compiling model")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        lpred.compile()
        apred.compile()
    
    transform = compile_transform(
        random_horizontal_flip = horizontal_flip,
        random_resize_aspect_ratio = random_aspect_ratio,
        random_resize_scale = random_resize_scale,
        reprob = reprob,
        auto_augment = auto_augment,
        motion_shift = motion_shift,
        crop_size    = crop_size,
    )
    
    video_loader = compile_dataloader(
        train_cfg, 
        nclips = nclips,
        transform = transform,
        collate_fn = torch.utils.data.default_collate,
        num_workers  = num_workers,
        persistance_workers = persistent_workers,
        pin_memory = pin_mem
    )
    
    optim, scaler, lr_scheduler, wd_scheduler = compile_opt(
        encoder = encoder,
        apred   = apred,
        lpred   = lpred,
        iterations_per_epoch = ipe,
        start_lr = start_lr,
        warmup = warmup, 
        anneal = annel,
        num_epochs = epochs,
        wd = weight_decay,
        final_lr = final_lr,
        mixed_precision = mixed_precision,
        betas = betas,
        eps = eps,
        ref_lr = lr,
        final_wd = final_wd
    )
    
    loader = iter(video_loader)

    log_dir = os.path.join(FOLDER_DIR, "../Experiment/action/")
    run_idx = get_next_run(log_dir)
    log_stats = create_self_supervised_logger(
        log_dir = log_dir,
        epochs = epochs,
        run_name = f"run{run_idx}",
        progress_type = "table"
    )
    
    gc.disable()
    gc.collect()

    def train_step(clips):
        _new_lr = lr_scheduler.step()
        _new_wd = wd_scheduler.step()
        
        def forward_target(c: torch.Tensor):
            with torch.no_grad():
                h: torch.Tensor = encoder(c)
                if normalize_rep:
                    h = F.layer_norm(h, (h.size(-1), ))
            return h

        def forward_prediction(h: torch.Tensor):
            def _step_action(h):
                _a = apred(h)
                return _a
            
            def _step_prediction(h, a):
                _z: torch.Tensor = lpred(h, a)
                if normalize_rep:
                    _z = F.layer_norm(_z, (_z.size(-1), ))
                return _z

            # -- Teacher forcing entire timestep action + prediction
            h_ctx = h[:, :-tokens_pframe, :]
            _a_tf = _step_action(h_ctx)
            _z_tf = _step_prediction(h_ctx, _a_tf)
                
            # -- Autoregressive rollout of each timestep action and prediction
            h_ctx = torch.cat([h_ctx[:, :tokens_pframe], _z_tf[:, :tokens_pframe]], dim = 1)
            a_ctx = _a_tf[:, : action_pframe] 
            for n in range(init_step, auto_steps):
                # -- Consider chunking?
                # -- Since the latent is predicted on action, the action must not drift
                a_ctx = _a_tf[:, :n * action_pframe]

                h_nxt = _step_prediction(h_ctx, a_ctx)[:, -tokens_pframe: ]
                h_ctx = torch.cat([h_ctx, h_nxt], dim = 1)
            _z_ar = h_ctx[:, tokens_pframe: ]
            
            
            return _z_tf, _z_ar, _a_tf
            
        def latent_loss(h, z):
            sub_h = h[:, tokens_pframe: z.size(-2) + tokens_pframe]
            return torch.mean(torch.abs(z - sub_h) ** loss_exp) / loss_exp
        
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
                
                N, D = a.size(0) * a.size(1), a.size(2)
                a = a.reshape(N, D)

                # -- Ensure each sample in batch is different (prevent collapse) 
                variance = torch.relu(1 - torch.std(a, dim = 0)).mean() * lv_vcm

                # -- Prevent static action to have value different than 0
                mean = a.mean() * lm_vcm

                # -- Ensure each variable is independent (maximize information capacity)
                a = a - a.mean(dim = 0)
                cov = (a.T @ a) / (N - 1)
                diag_mask = ~torch.eye(D, device = a.device).bool()
                covariance = cov[diag_mask].pow(2).mean() * lc_vcm
                return covariance + mean + variance, covariance.item(), mean.item(), variance.item()
                
            vcm_loss, covariance, mean, variance = vcm(a)
            energy_loss, hinge, sparsity = energy(a)
            return vcm_loss + energy_loss, (hinge, sparsity), (variance, covariance, mean)

        with torch.amp.autocast(device_type, dtype = dtype, enabled = mixed_precision):
            h = forward_target(clips)
            z_tf, z_ar, a_tf = forward_prediction(h)
            loss_tf  = latent_loss(h, z_tf)
            loss_ar  = latent_loss(h, z_ar)
            loss_act, energy, vcm = action_loss(a_tf)            
            loss = loss_tf + loss_ar + loss_act
            
            
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
        
        loss = loss.item()
        loss_tf = loss_tf.item()
        loss_ar = loss_ar.item()
        loss_act = loss_act.item()
        
        return (
            loss, 
            loss_tf,
            loss_ar,
            loss_act,
            energy,
            vcm
        )
    
    with log_stats:
        log_stats.start_training("Training Latent Action WM")
        
        for epoch in range(epochs):
            
            log_stats.start_epoch(epoch, len(video_loader), desc = "Training")
            
            for itr in log_stats.batch_iterator([i for i in range(ipe)]):
                
                iter_retries = 0
                iter_success = False
                while not iter_success:
                    try:
                        sample = next(loader)
                        iter_success = True
                    except StopIteration:
                        loader = iter(video_loader)
                    except Exception as e:
                        NUM_RETRIES = 5
                        if iter_retries < NUM_RETRIES:
                            logger.WARNING(f"Encountered an error while iterating loader: {e}")
                            iter_retries += 1
                            time.sleep(5)
                        else:
                            logger.ERROR("Exceeded maximum retries when iterating dataloade. Please check for error", exit_code = 5, full_traceback = e)
                        
                def load_clips():
                    clips = torch.concat(
                        [
                            sample[0][0].to(device, non_blocking = True), 
                            sample[1][0].to(device, non_blocking = True)
                        ], dim = 2) 
                    
                    actions = [{key: value.to(device, non_blocking=True) for key, value in action_dict.items()} for action_dict in sample[2]]
                    return clips, actions
                
                clips, actions = load_clips()
                
                (loss, loss_tf, loss_ar, loss_act, energy, vcm), elapsed_time = gpu_timer(partial(train_step, clips))

                log_stats.log_batch({
                    "Loss": loss,
                    "Teach Force": loss_tf,
                    "Autoregressive": loss_ar,
                    "Action": loss_act,
                    "Hinge": energy[0],
                    "Sparsity": energy[1],
                    "Variance": vcm[0],
                    "Covariance": vcm[1],
                    "Mean": vcm[2],
                    "GPU timer": elapsed_time 
                })
            
    
if __name__ == "__main__":
    yaml_path = "./JEPA_ACT/cfgs/action-224px-1024.24e.yaml"
    with open(yaml_path, "r") as f:
        args = yaml.safe_load(f)
        
    main(args, yaml_path)