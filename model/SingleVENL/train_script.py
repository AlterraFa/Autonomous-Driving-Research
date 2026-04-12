import os, sys
import torch
import torch.optim as optim
import warnings
import yaml

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, ConstantLR
from model.SingleVENL.schedulers import CosineSchedule, CosineWDSchedule

# ----------------------------------------------------
# Resolve project root from this file's location
# ----------------------------------------------------
FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../.."))

sys.path.append(PROJECT_ROOT)
from model.SingleVENL.data_loader import CarlaDatasetLoader, get_next_run
from model.SingleVENL.model import SingleVENL
from model.SingleVENL.core_trainer import single_epoch_training, single_epoch_val
from model.early_stop import EarlyStopping
from model.training_logger import TrainingLogger

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)


yaml_dir = FILE_DIR + "/model_cfg.yaml"
with open(yaml_dir, "r") as f:
    config = yaml.safe_load(f)


# --- DATA SECTION ---
crop_v = config['data']['crop']['vertical']      # [150, 720]
crop_h = config['data']['crop']['horizontal']    # [370, 1130]

# --- CONFIG SECTION ---
init_lr   = float(config['config']['initLR'])    # 0.001
target_lr = float(config['config']['targetLR'])  # 1e-07
init_wd   = float(config['config']['initWD'])
target_wd = float(config['config']['targetWD'])
epochs    = config['config']['epochs']           # 300

# --- MODEL SECTION ---
num_components = config['model']['components']   # 5
drop_rate      = config['model']['droprate']
drop_route     = config['model']['drop_route']
drop_all       = config['model']['drop_all']
output_names   = config['model']['output_names']
num_waypoints  = config['model']['num_waypoints']

# Metadata Shapes
input_metadata = config['model']['input_metadata']

# --- LOSS CONTRIB SECTION ---
mse_w      = config['loss_contrib']['mse_contrib']
nll_w      = config['loss_contrib']['nll_contrib']
std_reg_w  = config['loss_contrib']['std_reg_contrib']
entropy_w  = config['loss_contrib']['entropy_contrib']
repul_w    = config['loss_contrib']['repulsion_contrib']
l1_loss_w  = config['loss_contrib']['l1_contrib']
l2_loss_w  = config['loss_contrib']['l2_contrib']

# --- CHECKPOINTS SECTION ---
save_best    = config['checkpoints']['save_best_only']
ckpt_mode    = config['checkpoints']['mode']
save_weights = config['checkpoints']['save_weights_only']
ckpt_freq    = config['checkpoints']['frequency']
patience     = config['checkpoints']['patience']
min_delta    = config['checkpoints']['min_delta']



if __name__ == "__main__":
    
    gpu = torch.device("cuda")
    torch.manual_seed(45)

    model = SingleVENL(
        num_waypoints = num_waypoints, 
        components = num_components, 
        droprate = drop_rate, 
        input_metadata = input_metadata,
        output_names = output_names,
        drop_all = drop_all,
        drop_route = drop_route
    ).to(gpu)
    images_key = list(model.input_metadata.keys())

    dataset = CarlaDatasetLoader(
        [
            "./data/SingleVENL/recording_20251025_142727_best_temporal/", 
            "./data/SingleVENL/recording_20251019_161905_best_temporal/", 
            # "./data/SingleVENL/recording_20251029_163431_extra_temporal/",
            # "./data/SingleVENL/recording_20251029_140108_extra_temporal/",
            "./data/SingleVENL/recording_20251029_203531_extra_temporal/"
        ], 
        images_key = images_key, 
        downsize_ratio = 1, 
        load_size = -1,
        input_metadata = input_metadata
    )
    train, val, test = dataset.split(train = 0.85, val = 0.15)
    train_loader     = DataLoader(train, batch_size = 64, shuffle = True, collate_fn = dataset.collate_fn, num_workers = 4, persistent_workers = True)
    val_loader       = DataLoader(val, batch_size = 256, shuffle = False, collate_fn = dataset.collate_fn, num_workers = 4, persistent_workers = True)


    dummy = {}
    for key, value in model.input_metadata.items():
        dummy.update({key: torch.zeros(value).to(gpu)})
    model.initialize_module(**dummy)  # Initialize dummy layers
    


    optimizer = optim.AdamW(model.parameters(), lr = init_lr, betas = (0.95, 0.999))

    lr_scheduler = CosineSchedule(
        optimizer = optimizer,
        ref_lr = init_lr,
        T_max = int(epochs * len(train_loader)),
        final_lr = target_lr
    )
    wd_scheduler = CosineWDSchedule(
        optimizer = optimizer,
        ref_wd = init_wd,
        T_max = int(epochs * len(train_loader)),
        final_wd = target_wd
    )
    

    log_dir = f"{FILE_DIR}/Experiment/"
    run = get_next_run(FILE_DIR)
    log_stats = TrainingLogger(
        log_dir = log_dir,
        epochs  = epochs,
        run_name = f"run{run}",
        progress_type = "table"
    ) 
    os.system(f"cp {yaml_dir} {os.path.join(log_dir, f'run{run}')}")
    earlystop = EarlyStopping(
        patience, 
        min_delta, 
        freq = ckpt_freq, 
        path = f"{log_dir}/run{run}/weights/{model._get_name()}.pt", 
        mode = ckpt_mode, 
        verbose = False,
        weights_only = True
    )
    
    with log_stats:
        log_stats.start_training("Training SingleVENL")
        for epoch in range(epochs):
            
            log_stats.start_epoch(epoch, len(train_loader), desc = "Training")    
            train_metrics = single_epoch_training(
                model = model, 
                loader = train_loader, 
                lr_scheduler = lr_scheduler,
                wd_scheduler = wd_scheduler,
                optimizer = optimizer, 
                log_stats = log_stats
            )

            log_stats.start_phase(len(val_loader), desc = "Validating")
            val_metrics   = single_epoch_val(
                model = model,
                loader = val_loader,
                log_stats = log_stats
            )

            current_lr = optimizer.param_groups[0]['lr']
            current_wd = optimizer.param_groups[0]['weight_decay']
            log_stats.log_epoch(extra_metrics={
                "Current LR": current_lr,
                "Current WD": current_wd
            })
            
            earlystop(log_stats.get_metric('Total', "val"), model, epoch, optimizer)
            if earlystop.early_stop:
                break
