import os, sys
import re
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, ast
import warnings
import yaml

from configparser import ConfigParser
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, ConstantLR

# ----------------------------------------------------
# Resolve project root from this file's location
# ----------------------------------------------------
FILE_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../.."))

sys.path.append(PROJECT_ROOT)
from model.VENL.model import VENL
from model.data_loader import CarlaDatasetLoader, get_next_run
from model.VENL.core_trainer import single_epoch_training, single_epoch_val
from model.early_stop import EarlyStopping

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)

yaml_dir = FILE_DIR + "/model_cfg.yaml"
with open(yaml_dir, "r") as f:
    config = yaml.safe_load(f)

venl_config = config["config"]
venl_loss = config["loss_contrib"]

initLR       = float(venl_config["initLR"])
targetLR     = float(venl_config["targetLR"])
epochs       = int(venl_config["epochs"])
components   = int(venl_config["components"])
patience      = int(venl_config["patience"])
droprate      = float(venl_config["droprate"])
if __name__ == "__main__":
    
    run = get_next_run(FILE_DIR)
    gpu = torch.device("cuda")
    torch.manual_seed(45)

    model = VENL.waypoint(num_waypoints = 6, components = components, droprate = droprate).to(gpu) 
    images_key = list(model.input_metadata.keys())

    dataset = CarlaDatasetLoader(
        [
            "./data/VENL/recording_20251025_142727_best_temporal/", 
            "./data/VENL/recording_20251019_161905_best_temporal/", 
            "./data/VENL/recording_20251029_163431_extra_temporal/",
            "./data/VENL/recording_20251029_175725_temporal/",
            "./data/VENL/recording_20251029_203531_temporal"
        ], 
        images_key = images_key, 
        downsize_ratio = 1, 
        load_size = -1  
    )
    train, val, test = dataset.split(train = 0.85, val = 0.15)
    train_loader     = DataLoader(train, batch_size = 128, shuffle = True, collate_fn = dataset.collate_fn, num_workers = 4, persistent_workers = True)
    val_loader       = DataLoader(val, batch_size = 300, shuffle = True, collate_fn = dataset.collate_fn, num_workers = 4, persistent_workers = True)


    dummy = {}
    for key, value in model.input_metadata.items():
        dummy.update({key: torch.zeros(value).to(gpu)})
    model.initialize_module(**dummy)  # Initialize dummy layers

    log_dir = f"{FILE_DIR}/Experiment/run{run}"
    writer = SummaryWriter(log_dir=log_dir)
    writer.add_graph(model, list(dummy.values()))
    writer.flush()
    os.system(f"cp {yaml_dir} {log_dir}")

    optimizer = optim.AdamW(model.parameters(), lr = initLR, betas = (0.95, 0.999))

    sched1 = CosineAnnealingLR(optimizer, T_max = epochs // 2, eta_min = targetLR)
    sched2 = ConstantLR(optimizer, factor = targetLR / initLR, total_iters = epochs // 2)  # keep constant
    scheduler = SequentialLR(
        optimizer,
        schedulers=[sched1, sched2],
        milestones=[epochs // 2]
    )

    earlystop = EarlyStopping(patience, 1e-5, path = f"{log_dir}/{model._get_name()}_run{run}.pt", verbose = False)
    
    pbar = tqdm(range(epochs), desc = f"Training Epochs - Early stop at: {earlystop.counter} / {earlystop.patience}", position = 0)
    for epoch in pbar:
        
        desc_str = f"Training Epochs - Early stop at: {earlystop.counter} / {earlystop.patience}. {'Has improved' if earlystop.improved else 'No improvement'}"
        pbar.set_description(desc = desc_str)

        train_metrics = single_epoch_training(
            model, 
            train_loader, 
            optimizer, 
        )
        val_metrics   = single_epoch_val(
            model = model,
            loader = val_loader,
        )

        scheduler.step()
        currentLr = optimizer.param_groups[0]['lr']

        tqdm.write(
            f"Epoch {epoch+1}/{epochs} — "
            f"Total: {train_metrics['Total']:.4f}, "
            f"MSE: {train_metrics['MSE']:.4f}, "
            f"NLL: {train_metrics['NLL']:.4f}, "
            f"STD Reg: {train_metrics['STD Reg']:.4f}, "
            f"ENT Weight: {train_metrics['Weights Entropy']:.4f}, "
            f"Val Total: {val_metrics['Total']:.4f}, "
            f"Val MSE: {val_metrics['MSE']:.4f}, "
            f"Val NLL: {val_metrics['NLL']:.4f}, "
            f"Val STD Reg: {val_metrics['STD Reg']:.4f}, "
            f"Val ENT Weight: {val_metrics['Weights Entropy']:.4f}, "
            f"LR: {currentLr:.1e}, "
            f"No update: {earlystop.counter}/{earlystop.patience}"
        )

        writer.add_scalar("Loss/Train Total", train_metrics["Total"], epoch+1)
        writer.add_scalar("Loss/Train MSE", train_metrics["MSE"], epoch+1)
        writer.add_scalar("Loss/Train NLL", train_metrics["NLL"], epoch+1)
        writer.add_scalar("Loss/Train STD Reg", train_metrics["STD Reg"], epoch+1)
        writer.add_scalar("Loss/Train Entropy", train_metrics["Weights Entropy"], epoch + 1)
        writer.add_scalar("Loss/Train Repulsion", train_metrics["Repulsion"], epoch + 1)

        writer.add_scalar("Loss/Val Total", val_metrics["Total"], epoch+1)
        writer.add_scalar("Loss/Val MSE", val_metrics["MSE"], epoch+1)
        writer.add_scalar("Loss/Val NLL", val_metrics["NLL"], epoch+1)
        writer.add_scalar("Loss/Val STD Reg", val_metrics["STD Reg"], epoch+1)
        writer.add_scalar("Loss/Val Repulsion", val_metrics["Repulsion"], epoch+1)
        writer.add_scalar("Loss/Val Entropy", val_metrics["Weights Entropy"], epoch + 1)
        
        writer.add_scalar("Grad Monitor/Routed", train_metrics["Grad_Routed"], epoch + 1)
        writer.add_scalar("Grad Monitor/Unrouted", train_metrics["Grad_Unrouted"], epoch + 1)
        writer.add_scalar("Grad Monitor/MultiCam0", train_metrics["Grad_MultiCam0"], epoch + 1)
        writer.add_scalar("Grad Monitor/MultiCam1", train_metrics["Grad_MultiCam1"], epoch + 1)
        writer.add_scalar("Grad Monitor/MultiCam2", train_metrics["Grad_MultiCam2"], epoch + 1)
        
        writer.add_scalar("Misc/LearningRate", currentLr, epoch+1)
        writer.flush()

        earlystop(val_metrics['Total'], model, epoch = epoch, optimizer = optimizer)
        if earlystop.early_stop:
            print(f"STOPPED AT EPOCH {epoch}")
            break