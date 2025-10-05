import torch
import os

class EarlyStopping:
    def __init__(self, 
                 patience: int = 5, 
                 min_delta: float = 0.0, 
                 path: str = "checkpoint.pt",
                 verbose: bool = False):
        self.patience  = patience
        self.min_delta = min_delta
        self.path      = path
        self.verbose   = verbose
        self.counter   = 0
        self.best_loss = torch.inf
        self.early_stop = False
        basename = os.path.basename(path)
        self.parent_folder = os.path.dirname(path)
        self.best_name = "/best_" + basename
        self.last_name = "/last_" + basename

    def __call__(self, val_loss: float, model: torch.nn.Module):
        # check if loss improved by at least min_delta
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model, self.parent_folder + self.best_name)
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}. Saved model to {self.parent_folder + self.best_name}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in val loss for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
        torch.save(model, self.parent_folder + self.last_name)
                