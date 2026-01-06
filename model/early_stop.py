import torch
import os
import numpy as np

class EarlyStopping:
    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.0, 
                 freq: int = 0, 
                 path: str = "checkpoint.pt",
                 mode: str = "min",
                 verbose: bool = False):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = path
        self.verbose   = verbose
        self.counter   = 1
        self.best_loss = None
        self.early_stop = False
        
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

    def __call__(self, score: float, model: torch.nn.Module, epoch, optimizer=None):
        # check if loss improved by at least min_delta
        
        if self.best_loss is None:
            self.best_loss = score
            self._save_checkpoint(self.parent_folder + self.best_name, score, model, epoch, optimizer)
        
        elif self.monitor_op(score, self.best_loss + self.min_delta):
            self.best_loss = score
            self.counter   = 1
            self._save_checkpoint(self.parent_folder + self.best_name, score, model, epoch, optimizer)
            self.improved = True
            if self.verbose:
                print(f"Validation loss improved to {score:.4f}. Saved model to {self.parent_folder + self.best_name}")
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
            self._save_checkpoint(self.parent_folder + last_name, score, model, epoch, optimizer)
        self.iter_count += 1
                
    def _save_checkpoint(self, path, score, model, epoch, optimizer):
        
        checkpoint = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'score': score
        }
        
        torch.save(checkpoint, self.parent_folder + "/checkpoint.pt")
        torch.save(model, path)