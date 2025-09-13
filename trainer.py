import os, sys
import re
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, ConstantLR

# ----------------------------------------------------
# Resolve project root from this file's location
# ----------------------------------------------------
FILE_DIR   = os.path.dirname(os.path.abspath(__file__))           # .../model/PilotNet
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../.."))   # .../ (CARLAPython root)

sys.path.append(PROJECT_ROOT)
from model.PilotNet.model import PilotNetStatic, single_epoch_training_static, single_epoch_val_static
from utils.data_processor import CarlaDatasetLoader
from utils.helper import EarlyStopping


def get_next_run(base_dir: str = "PilotNetExperiment") -> int:
    """
    Detects the highest run number in base_dir and returns the next available run index.
    """
    exp_dir = os.path.join(FILE_DIR + "/model/PilotNet", base_dir)  # anchor to project root
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        return 1

    runs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    run_nums = []
    for r in runs:
        match = re.match(r"run(\d+)", r)
        if match:
            run_nums.append(int(match.group(1)))

    return max(run_nums, default=0) + 1


if __name__ == "__main__":
    
    gpu = torch.device("cuda")
    torch.manual_seed(45)

    dataset          = CarlaDatasetLoader("./data/recording_20250905_231635_best_spatial_roi", downsize_ratio = 1, load_size = -1)
    train, val, test = dataset.split(train = 0.8, val = 0.2)
    train_loader     = DataLoader(train, batch_size = 60, shuffle = True, collate_fn = dataset.collate_fn)
    val_loader       = DataLoader(val, batch_size = 50, shuffle = True, collate_fn = dataset.collate_fn)
    
    experiment_name = "PilotNetExperiment"
    mode  = "steer"
    run   = get_next_run(experiment_name)

    dummy = torch.zeros((1, 3, 300, 150)).to(gpu)
    model = PilotNetStatic(mode = mode, num_waypoints = 5, num_cmd = 4, droprate = 0.25).to(gpu) 
    model(dummy)  # Initialize dummy layers

    log_dir = f"{FILE_DIR}/model/PilotNet/{experiment_name}/run{run}"
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model, dummy)
    writer.flush()


    initLR = 5e-4; targetLR = 1e-9
    epochs = 100; l1 = 1e-4; l2 = 1e-4; 

    optimizer = optim.AdamW(model.parameters(), lr = initLR, betas = (0.95, 0.999))

    sched1 = CosineAnnealingLR(optimizer, T_max = epochs // 2, eta_min = targetLR)
    sched2 = ConstantLR(optimizer, factor = targetLR / initLR, total_iters = epochs // 2)  # keep constant
    scheduler = SequentialLR(
        optimizer,
        schedulers=[sched1, sched2],
        milestones=[epochs // 2]
    )

    criterion = nn.HuberLoss(delta = 0.05)
    earlystop = EarlyStopping(20, 1e-5, path = f"{FILE_DIR}/model/PilotNet/{experiment_name}/run{run}/best_{model._get_name()}_run{run}.pt", verbose = True)
    
    pbar = tqdm(range(epochs), desc="Training Epochs", position = 0)
    for epoch in pbar:
        train_metrics = single_epoch_training_static(model, mode, train_loader, criterion, optimizer, l1 = l1, l2 = l2)
        val_metrics   = single_epoch_val_static(model, mode, val_loader, criterion, l1, l2)

        scheduler.step()
        currentLr = optimizer.param_groups[0]['lr']

        tqdm.write(
            f"Epoch {epoch+1}/{epochs} — "
            f"Sup: {train_metrics['Supervised']:.4f}, "
            f"Total: {train_metrics['Total']:.4f}, "
            f"Val Loss: {val_metrics['Cost']:.4f}, "
            f"Total Val Loss: {val_metrics['Total']:.4f}, "
            f"LR: {currentLr:.1e}, "
            f"No update: {earlystop.counter}/{earlystop.patience}"
        )

        writer.add_scalar("Loss/Supervised", train_metrics["Supervised"], epoch+1)
        writer.add_scalar("Loss/Total",      train_metrics["Total"],      epoch+1)
        writer.add_scalar("Loss/Validation", val_metrics["Cost"],         epoch+1)
        writer.add_scalar("Loss/Total Validation", val_metrics["Total"],         epoch+1)
        writer.add_scalar("Misc/LearningRate", currentLr,                 epoch+1)

        earlystop(val_metrics['Cost'], model)
        if earlystop.early_stop:
            print(f"STOPPED AT EPOCH {epoch}")
            break
