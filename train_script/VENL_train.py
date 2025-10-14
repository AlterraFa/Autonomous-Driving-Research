import os, sys
import re
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, ast
import warnings

from configparser import ConfigParser
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
from model.VENL.model import VENL, single_epoch_training, single_epoch_val, CarlaDatasetLoader
from utils.others.helper import EarlyStopping

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

def get_next_run(model_name: str) -> int:
    """
    Detects the highest run number in base_dir and returns the next available run index.
    """
    exp_dir = os.path.join(FILE_DIR + f"/../model/{model_name}", "Experiment")  # anchor to project root
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

config = ConfigParser()
config.read(FILE_DIR + "/../config/config.ini")

if __name__ == "__main__":
    
    gpu = torch.device("cuda")
    torch.manual_seed(45)

    model = VENL.waypoint(num_waypoints = 5, droprate = 0.25).to(gpu) 
    images_key = list(model.input_metadata.keys())

    dataset = CarlaDatasetLoader(
        "./data/recording_20251012_205458_best_temporal/", 
        images_key = images_key, 
        downsize_ratio = 1, 
        load_size = -1
    )
    train, val, test = dataset.split(train = 0.8, val = 0.2)
    train_loader     = DataLoader(train, batch_size = 120, shuffle = True, collate_fn = dataset.collate_fn)
    val_loader       = DataLoader(val, batch_size = 200, shuffle = True, collate_fn = dataset.collate_fn)


    model_name = "VENL"
    run        = get_next_run(model_name)

    dummy = {}
    for key, value in model.input_metadata.items():
        dummy.update({key: torch.zeros(value).to(gpu)})
    model.initialize_module(**dummy)  # Initialize dummy layers

    log_dir = f"{FILE_DIR}/../model/{model_name}/Experiment/run{run}"
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model, list(dummy.values()))
    writer.flush()

    
    initLR       = float(config["VENL"]["initLR"])
    targetLR     = float(config["VENL"]["targetLR"])
    epochs       = int(config["VENL"]["epochs"])
    l1           = float(config["VENL"]["l1"])
    l2           = float(config["VENL"]["l2"])
    l1_weight    = float(config["VENL"]["l1_weight"])
    l2_weight    = float(config["VENL"]["l2_weight"])
    target_spread = torch.tensor(ast.literal_eval(config["VENL"]["target_spread"]))
    patience      = int(config["VENL"]["patience"])

    optimizer = optim.AdamW(model.parameters(), lr = initLR, betas = (0.95, 0.999))

    sched1 = CosineAnnealingLR(optimizer, T_max = epochs // 2, eta_min = targetLR)
    sched2 = ConstantLR(optimizer, factor = targetLR / initLR, total_iters = epochs // 2)  # keep constant
    scheduler = SequentialLR(
        optimizer,
        schedulers=[sched1, sched2],
        milestones=[epochs // 2]
    )

    earlystop = EarlyStopping(patience, 1e-5, path = f"{FILE_DIR}/../model/{model_name}/Experiment/run{run}/{model._get_name()}_run{run}.pt", verbose = True)
    
    pbar = tqdm(range(epochs), desc="Training Epochs", position = 0)
    for epoch in pbar:
        train_metrics = single_epoch_training(
            model, 
            train_loader, 
            optimizer, 
            target_spread = target_spread, 
            l1_weight = l1_weight, 
            l2_weight = l2_weight, 
            l1 = l1, l2 = l2
        )
        val_metrics   = single_epoch_val(
            model = model,
            loader = val_loader,
            target_spread = target_spread,
            l1_weight = l1_weight,
            l2_weight = l2_weight
        )

        scheduler.step()
        currentLr = optimizer.param_groups[0]['lr']

        tqdm.write(
            f"Epoch {epoch+1}/{epochs} — "
            f"Total: {train_metrics['Total']:.4f}, "
            f"MSE: {train_metrics['MSE']:.4f}, "
            f"NLL: {train_metrics['NLL']:.4f}, "
            f"STD Reg: {train_metrics['STD Reg']:.4f}, "
            f"Val Total: {val_metrics['Total']:.4f}, "
            f"Val MSE: {val_metrics['MSE']:.4f}, "
            f"Val NLL: {val_metrics['NLL']:.4f}, "
            f"Val STD Reg: {val_metrics['STD Reg']:.4f}, "
            f"LR: {currentLr:.1e}, "
            f"No update: {earlystop.counter}/{earlystop.patience}"
        )

        writer.add_scalar("Loss/Train Total", train_metrics["Total"], epoch+1)
        writer.add_scalar("Loss/Train MSE", train_metrics["MSE"], epoch+1)
        writer.add_scalar("Loss/Train NLL", train_metrics["NLL"], epoch+1)
        writer.add_scalar("Loss/Train STD Reg", train_metrics["STD Reg"], epoch+1)

        writer.add_scalar("Loss/Val Total", val_metrics["Total"], epoch+1)
        writer.add_scalar("Loss/Val MSE", val_metrics["MSE"], epoch+1)
        writer.add_scalar("Loss/Val NLL", val_metrics["NLL"], epoch+1)
        writer.add_scalar("Loss/Val STD Reg", val_metrics["STD Reg"], epoch+1)

        writer.add_scalar("Misc/LearningRate", currentLr, epoch+1)
        writer.flush()

        earlystop(val_metrics['Total'], model)
        if earlystop.early_stop:
            print(f"STOPPED AT EPOCH {epoch}")
            break