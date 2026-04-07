import torch
import os
import numpy as np
from torch import distributed as dist

class EarlyStopping:
    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.0, 
                 freq: int = 0, 
                 path: str = "checkpoint.pt",
                 mode: str = "min",
                 verbose: bool = False, 
                 weights_only =  False):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = path
        self.verbose   = verbose
        self.counter   = 1
        self.best_loss = None
        self.early_stop = False
        self.weights_only = weights_only
        
        basename = os.path.basename(path)
        self.parent_folder = os.path.dirname(path)
        self.best_name = "/best_" + basename
        self.last_name = "/last_" + basename
        self.improved = False
        
        self.save_freq  = freq
        self.iter_count = 0

        self.mode = mode
        if self.mode == 'min':
            self.monitor_op = np.less_equal
            self.min_delta = -self.min_delta
        elif self.mode == 'max':
            self.monitor_op = np.greater_equal
            self.min_delta = self.min_delta
        else:
            raise ValueError(f"EarlyStopping mode {mode} is unknown!")
        
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def __call__(self, score: float, model: torch.nn.Module, **other):
        # check if loss improved by at least min_delta
        
        if dist.is_initialized():
            score_t = torch.tensor(score, device=torch.cuda.current_device())
            dist.all_reduce(score_t, op=dist.ReduceOp.SUM)
            score = score_t.item() / dist.get_world_size()
        
        if self.best_loss is None:
            self.best_loss = score
            path = self.parent_folder + self.best_name
            if self.rank == 0:
                self._save_checkpoint(path, score, model, other)
        
        elif self.monitor_op(score, self.best_loss + self.min_delta):
            self.best_loss = score
            self.counter   = 1
            path = self.parent_folder + self.best_name
            if self.rank == 0:
                self._save_checkpoint(path, score, model, other)
            self.improved = True
            if self.verbose:
                print(f"Validation loss improved to {score:.4f}. Saved model to {path}")
        else:
            self.counter += 1
            self.improved = False
            if self.verbose:
                print(f"No improvement in val loss for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                
        if self.save_freq != 0 and self.iter_count % self.save_freq == 0:
            name, ext = self.last_name.split(".")
            name += f"_{self.iter_count}"
            last_name = name + "." + ext 
            path  = self.parent_folder + last_name
            if self.rank == 0:
                self._save_checkpoint(path, score, model, other)
        self.iter_count += 1
                
    def _save_checkpoint(self, path, score, model, other):
        raw_model = model.module if hasattr(model, 'module') else model
        
        checkpoint = {'score': score} | other
        
        torch.save(checkpoint, os.path.join(self.parent_folder, "checkpoint.pt"))
        torch.save(raw_model.state_dict() if self.weights_only else raw_model, path)

class MultiModuleEarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, freq=0, path_root="weights", mode="min", weights_only=True):
        self.patience = patience
        self.min_delta = min_delta
        self.path_root = path_root # Directory where all weights go
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.weights_only = weights_only
        self.save_freq = freq
        self.iter_count = 0
        self.mode = mode
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        if not os.path.exists(self.path_root):
            os.makedirs(self.path_root, exist_ok=True)

    def __call__(self, score, models_dict, optimizer, scaler, epoch):
        if dist.is_initialized():
            score_t = torch.tensor(score, device=torch.cuda.current_device())
            dist.all_reduce(score_t, op=dist.ReduceOp.SUM)
            score = score_t.item() / dist.get_world_size()

        is_best = False
        if self.best_loss is None:
            self.best_loss = score
            is_best = True
        elif (score < self.best_loss - self.min_delta if self.mode == 'min' else score > self.best_loss + self.min_delta):
            self.best_loss = score
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        # Save Logic
        if self.rank == 0:
            # 1. Always save metadata
            meta = {
                'epoch': epoch,
                'score': score,
                'best_loss': self.best_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None
            }
            torch.save(meta, os.path.join(self.path_root, "checkpoint.pt"))

            # 2. Save models
            for name, model in models_dict.items():
                raw_model = model.module if hasattr(model, 'module') else model
                state = raw_model.state_dict() if self.weights_only else raw_model
                
                # Save as "last"
                torch.save(state, os.path.join(self.path_root, f"last_{name}.pt"))
                
                # Save as "best"
                if is_best:
                    torch.save(state, os.path.join(self.path_root, f"best_{name}.pt"))

            # 3. Periodic checkpoint (last_X_iter.pt)
            if self.save_freq > 0 and epoch % self.save_freq == 0:
                for name, model in models_dict.items():
                    raw_model = model.module if hasattr(model, 'module') else model
                    state = raw_model.state_dict() if self.weights_only else raw_model
                    torch.save(state, os.path.join(self.path_root, f"epoch_{epoch}_{name}.pt"))