import os
import csv
import sys
from typing import Dict, Optional, List, Literal, Union, Tuple
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter

# Try to import progress-table
try:
    from progress_table import ProgressTable
    PROGRESS_TABLE_AVAILABLE = True
except ImportError:
    PROGRESS_TABLE_AVAILABLE = False

IS_KAGGLE_COMMIT = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Batch'

if IS_KAGGLE_COMMIT:
    from tqdm import tqdm
else:
    from tqdm.auto import tqdm

class NoOpLogger:
    """
    No-operation logger for distributed training - provides the same interface
    as TrainingLogger but performs no actual logging operations.
    
    This is used for non-zero ranks in distributed training to avoid:
    - Race conditions in directory creation
    - Duplicate logging output
    - Unnecessary computational overhead
    
    All methods are no-ops and return appropriate default values.
    """
    
    def __init__(self, *args, **kwargs):
        """No-op constructor - accepts any arguments to match TrainingLogger interface."""
        self.current_epoch = 0
        self.use_validation = kwargs.get('use_validation', True)
    
    def __enter__(self):
        """Context manager entry - returns self."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - no cleanup needed."""
        return False
    
    def start_training(self, description: str = "Training"):
        """No-op training initialization."""
        pass
    
    def start_epoch(self, epoch: int, num_batches: int, desc: str = "Training"):
        """No-op epoch initialization."""
        self.current_epoch = epoch
    
    def start_phase(self, num_batches: int, desc: str = "Phase"):
        """No-op phase initialization."""
        pass
    
    def batch_iterator(self, dataloader):
        """Returns the dataloader as-is without progress tracking."""
        return dataloader
    
    def log_batch(
        self,
        metrics: Dict[str, float],
        phase: str = "train",
        step: Optional[int] = None,
        phase_agnostic: Optional[List[str]] = None,
    ):
        """No-op batch logging."""
        pass
    
    def log_epoch(self, train_metrics: Optional[Dict[str, float]] = None, 
                  val_metrics: Optional[Dict[str, float]] = None,
                  extra_metrics: Optional[Dict[str, float]] = None):
        """No-op epoch logging."""
        pass
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]):
        """No-op model graph logging."""
        pass
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """No-op histogram logging."""
        pass
    
    def get_epoch_metrics(self, phase: str = "val") -> Dict[str, float]:
        """Returns empty metrics dictionary."""
        return {}
    
    def get_metric(self, metric_name: str, phase: str = "val") -> Optional[float]:
        """Returns None for any metric request."""
        return None
    
    def save_checkpoint(self, models: Dict[str, torch.nn.Module], optimizer: torch.optim.Optimizer,
                       scheduler: Optional[object] = None, filename: Optional[str] = None, 
                       extra_state: Optional[Dict] = None):
        """No-op checkpoint saving."""
        pass
    
    def close(self):
        """No-op cleanup."""
        pass
    
    def set_validation_mode(self, use_validation: bool):
        """No-op validation mode setting."""
        self.use_validation = use_validation


class TrainingLogger:
    """
    A training logger for models that integrates TensorBoard SummaryWriter
    with progress bars (tqdm or progress-table) for monitoring training progress.
    
    Args:
        log_dir: Directory to save TensorBoard logs
        epochs: Total number of training epochs
        run_name: Optional name for the run (will be appended to log_dir)
        metrics_to_track: List of metric names to track (optional, auto-detected if not provided)
        progress_type: Either "tqdm" or "table" for progress display style
        use_validation: Whether to expect validation metrics (set to False for self-supervised training)
    """
    
    @staticmethod
    def _format_value(value: float) -> str:
        """Format a numeric value, using scientific notation for small values."""
        if isinstance(value, int):
            return str(value)
        if abs(value) < 0.0001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.4f}"
    
    def __init__(
        self,
        log_dir: str,
        epochs: int,
        run_name: Optional[str] = None,
        metrics_to_track: Optional[List[str]] = None,
        progress_type: Literal["tqdm", "table"] = "tqdm",
        use_validation: bool = True,
        save_csv: bool = True,
        save_batch_csv: bool = False,
        save_epoch_csv: bool = True,
        log_batch_tensorboard: bool = False,
        resume_epoch: int = 0,
    ):
        self.epochs = epochs
        self.current_epoch = 0
        self.metrics_to_track = metrics_to_track or []
        self.progress_type = progress_type
        self.use_validation = use_validation
        self.save_csv = save_csv
        self.save_batch_csv = save_batch_csv
        self.save_epoch_csv = save_epoch_csv
        self.log_batch_tensorboard = log_batch_tensorboard
        
        # Global iteration counter for per-batch TensorBoard logging
        self._global_step = 0
        
        # Validate progress_table availability
        if progress_type == "table" and not PROGRESS_TABLE_AVAILABLE:
            print("Warning: progress-table not installed, falling back to tqdm")
            print("Install with: pip install progress-table")
            self.progress_type = "tqdm"
        
        # Setup log directory
        if run_name:
            self.log_dir = os.path.join(log_dir, run_name)
        else:
            self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Progress bars (tqdm)
        self.epoch_pbar: Optional[tqdm] = None
        self.batch_pbar: Optional[tqdm] = None
        
        # Progress table
        self.progress_table: Optional["ProgressTable"] = None
        self._table_columns_added = False
        self._table_pbar = None  # For table mode progress bar
        
        # Metric accumulators for epoch averaging
        self._train_metrics_accum = defaultdict(list)
        self._val_metrics_accum = defaultdict(list)
        self._misc_metrics_accum = defaultdict(list)

        # CSV logging buffers and schema
        self._csv_batch_rows: List[Dict[str, Union[str, float, int]]] = []
        self._csv_epoch_rows: List[Dict[str, Union[str, float, int]]] = []
        self._csv_batch_fields: List[str] = []
        self._csv_epoch_fields: List[str] = []
        self._csv_batch_path = os.path.join(self.log_dir, "batch_metrics.csv")
        self._csv_epoch_path = os.path.join(self.log_dir, "epoch_metrics.csv")
        
        # Resume support: reload existing CSV data and set global step offset
        if resume_epoch > 0:
            self._global_step = resume_epoch * (epochs // max(epochs, 1))  # rough default
            self.current_epoch = resume_epoch
            # Reload existing epoch CSV rows so new rows are appended
            if self.save_csv and self.save_epoch_csv and os.path.exists(self._csv_epoch_path):
                with open(self._csv_epoch_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    self._csv_epoch_fields = list(reader.fieldnames or [])
                    for row in reader:
                        # Keep only rows up to resume_epoch
                        try:
                            if int(row.get("epoch", 0)) <= resume_epoch:
                                self._csv_epoch_rows.append(dict(row))
                        except (ValueError, TypeError):
                            self._csv_epoch_rows.append(dict(row))
            # Reload existing batch CSV rows
            if self.save_csv and self.save_batch_csv and os.path.exists(self._csv_batch_path):
                with open(self._csv_batch_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    self._csv_batch_fields = list(reader.fieldnames or [])
                    for row in reader:
                        try:
                            if int(row.get("epoch", 0)) <= resume_epoch:
                                self._csv_batch_rows.append(dict(row))
                        except (ValueError, TypeError):
                            self._csv_batch_rows.append(dict(row))
                self._global_step = len(self._csv_batch_rows)
        
        # Track state for table display
        self._current_phase = "train"
        self._num_batches = 0
        self._current_batch = 0
        
        os.makedirs(os.path.join(self.log_dir, "weights"), exist_ok=True)
        
    def start_training(self, description: str = "Training"):
        """Initialize progress display at the start of training."""
        if self.progress_type == "tqdm":
            self.epoch_pbar = tqdm(
                total=self.epochs,
                desc=description,
                position=0,
                leave=True,
                file=sys.stdout,
                mininterval=30,
                initial=self.current_epoch,
            )
        else:
            # Initialize progress table with proper settings
            # pbar_embedded=False keeps progress bar on the right side
            self.progress_table = ProgressTable(
                num_decimal_places=4,
                pbar_embedded=False,
                pbar_show_progress=True,
                pbar_show_eta=True,
                pbar_show_throughput=True
            )
            self.progress_table.add_column("Epoch", color="bold")
            self._table_columns_added = False
        
    def start_epoch(self, epoch: int, num_batches: int, desc: str = "Training"):
        """
        Start a new epoch and initialize progress display.
        Clears metric accumulators for fresh epoch tracking.
        
        Args:
            epoch: Current epoch number (0-indexed)
            num_batches: Total number of batches in the epoch
            desc: Description for the progress display
        """
        self.current_epoch = epoch
        self._train_metrics_accum.clear()
        self._val_metrics_accum.clear()
        self._misc_metrics_accum.clear()
        self._current_phase = desc.lower()
        self._num_batches = num_batches
        self._current_batch = 0
        
        if self.progress_type == "tqdm":
            self.batch_pbar = tqdm(
                total=num_batches,
                desc=f"Epoch {self.current_epoch + 1}/{self.epochs} - {desc}",
                position=1,
                leave=False,
                file=sys.stdout,
                mininterval=30
            )
        else:
            # For table mode, store num_batches for batch_iterator to use
            self._table_num_batches = num_batches
            # Update epoch display immediately
            if self.progress_table is not None:
                self.progress_table["Epoch"] = f"{self.current_epoch + 1}/{self.epochs}"
    
    def start_phase(self, num_batches: int, desc: str = "Phase"):
        """
        Start a new phase (e.g., validation) within the same epoch.
        Does NOT clear metric accumulators, preserving train metrics.
        
        Args:
            num_batches: Total number of batches in this phase
            desc: Description for the progress display
        """
        self._current_phase = desc.lower()
        self._num_batches = num_batches
        self._current_batch = 0
        
        if self.progress_type == "tqdm":
            if self.batch_pbar is not None:
                self.batch_pbar.close()
                
            self.batch_pbar = tqdm(
                total=num_batches,
                desc=f"Epoch {self.current_epoch + 1}/{self.epochs} - {desc}",
                position=1,
                leave=False,
                file=sys.stdout,
                mininterval=30
            )
        else:
            # For table mode, store num_batches for batch_iterator to use
            self._table_num_batches = num_batches
    
    def batch_iterator(self, dataloader):
        """
        Returns a progress-tracked iterator over the dataloader.
        Use this instead of manually calling log_batch for automatic progress updates.
        
        Args:
            dataloader: The dataloader to iterate over
            
        Yields:
            Items from the dataloader with progress tracking
            
        Example:
            for batch in logger.batch_iterator(train_loader):
                loss = model(batch)
                logger.log_batch({"Loss": loss.item()})
        """
        if self.progress_type == "tqdm":
            if self.batch_pbar is not None:
                # Don't call update here - log_batch will do it
                yield from dataloader
            else:
                yield from dataloader
        else:
            # Use table(dataloader) directly - it wraps the dataloader with progress
            if self.progress_table is not None:
                yield from self.progress_table(dataloader)
            else:
                yield from dataloader

    @staticmethod
    def _normalize_csv_value(value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return str(value.detach().cpu().tolist())
        if isinstance(value, (list, tuple, dict)):
            return str(value)
        return value

    @staticmethod
    def _upsert_fieldnames(fieldnames: List[str], row: Dict[str, Union[str, float, int]]) -> List[str]:
        fields = list(fieldnames)
        for key in row.keys():
            if key not in fields:
                fields.append(key)
        return fields

    @staticmethod
    def _write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, Union[str, float, int]]]):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
        
    def log_batch(
        self,
        metrics: Dict[str, float],
        phase: str = "train",
        step: Optional[int] = None,
        phase_agnostic: Optional[List[str]] = None,
    ):
        """
        Log metrics for a single batch and update progress display.
        
        Args:
            metrics: Dictionary of metric names to values
            phase: Either "train" or "val"
            step: Optional global step (if None, uses internal counter)
        """
        phase_agnostic = set(phase_agnostic or [])

        clean_metrics: Dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            clean_metrics[key] = value

        # Accumulate metrics for epoch averaging
        train_or_val_accum = self._train_metrics_accum if phase == "train" else self._val_metrics_accum
        for key, value in clean_metrics.items():
            if key in phase_agnostic:
                self._misc_metrics_accum[key].append(value)
            else:
                train_or_val_accum[key].append(value)

        if self.save_csv and self.save_batch_csv:
            row = {
                "epoch": self.current_epoch + 1,
                "batch": self._current_batch + 1,
                "phase": phase,
            }
            row.update({k: self._normalize_csv_value(v) for k, v in clean_metrics.items()})
            self._csv_batch_rows.append(row)
            self._csv_batch_fields = self._upsert_fieldnames(self._csv_batch_fields, row)
            self._write_csv(self._csv_batch_path, self._csv_batch_fields, self._csv_batch_rows)
        
        self._current_batch += 1
        self._global_step += 1

        # Per-iteration TensorBoard logging
        if self.log_batch_tensorboard:
            tb_prefix = f"Iter_Train" if phase == "train" else f"Iter_Val"
            for key, value in clean_metrics.items():
                if key in phase_agnostic:
                    self.writer.add_scalar(f"Iter_Misc/{key}", value, self._global_step)
                else:
                    self.writer.add_scalar(f"{tb_prefix}/{key}", value, self._global_step)
        
        if self.progress_type == "tqdm":
            # Update batch progress bar with current metrics
            if self.batch_pbar is not None:
                display_metrics = {k: self._format_value(v) for k, v in metrics.items()}
                self.batch_pbar.set_postfix(display_metrics)
                self.batch_pbar.update(1)
        else:
            # For table mode, update running averages in the table
            if self.progress_table is not None:
                # Add metric columns dynamically on first batch
                if not self._table_columns_added:
                    for key in clean_metrics.keys():
                        if key in phase_agnostic:
                            self.progress_table.add_column(f"{key}", color="magenta")
                        elif self.use_validation:
                            self.progress_table.add_column(f"Train | {key}", color="blue")
                            self.progress_table.add_column(f"Val | {key}", color="green")
                        else:
                            # No validation mode - just show metric name without 'Train' prefix
                            self.progress_table.add_column(key, color="blue")
                    self._table_columns_added = True
                
                # Update running average in the table
                for key in clean_metrics.keys():
                    if key in phase_agnostic:
                        avg_value = sum(self._misc_metrics_accum[key]) / len(self._misc_metrics_accum[key])
                        self.progress_table[f"{key}"] = self._format_value(avg_value)
                    else:
                        if self.use_validation:
                            prefix = "Train | " if phase == "train" else "Val | "
                        else:
                            prefix = ""
                        avg_value = sum(train_or_val_accum[key]) / len(train_or_val_accum[key])
                        column_name = f"{prefix}{key}" if prefix else key
                        self.progress_table[column_name] = self._format_value(avg_value)
            
    def log_epoch(
        self,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None,
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log epoch-level metrics to TensorBoard and update epoch progress bar.
        
        Args:
            train_metrics: Training metrics dict (if None, uses accumulated batch metrics)
            val_metrics: Validation metrics dict (if None, uses accumulated batch metrics)
            extra_metrics: Additional metrics to log (e.g., learning rate, EMA momentum)
        """
        epoch = self.current_epoch + 1  # 1-indexed for display
        
        # Use accumulated metrics if not provided
        if train_metrics is None:
            train_metrics = {k: sum(v) / len(v) for k, v in self._train_metrics_accum.items() if v}
        if val_metrics is None and self.use_validation:
            val_metrics = {k: sum(v) / len(v) for k, v in self._val_metrics_accum.items() if v}
        elif val_metrics is None:
            val_metrics = {}
            
        # Log training metrics
        for key, value in train_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f"Train/{key}", value, epoch)
            
        # Log validation metrics (only if validation is enabled and metrics exist)
        if self.use_validation and val_metrics:
            for key, value in val_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(f"Val/{key}", value, epoch)
            
        # Log extra metrics (learning rate, EMA, etc.)
        misc_metrics = {k: sum(v) / len(v) for k, v in self._misc_metrics_accum.items() if v}
        if misc_metrics:
            for key, value in misc_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(f"Misc/{key}", value, epoch)

        if extra_metrics:
            for key, value in extra_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(f"Misc/{key}", value, epoch)

        if self.save_csv and self.save_epoch_csv:
            row = {"epoch": epoch}
            for key, value in train_metrics.items():
                row[f"Train/{key}"] = self._normalize_csv_value(value)
            if self.use_validation and val_metrics:
                for key, value in val_metrics.items():
                    row[f"Val/{key}"] = self._normalize_csv_value(value)
            for key, value in misc_metrics.items():
                row[f"Misc/{key}"] = self._normalize_csv_value(value)
            if extra_metrics:
                for key, value in extra_metrics.items():
                    row[f"Misc/{key}"] = self._normalize_csv_value(value)

            self._csv_epoch_rows.append(row)
            self._csv_epoch_fields = self._upsert_fieldnames(self._csv_epoch_fields, row)
            self._write_csv(self._csv_epoch_path, self._csv_epoch_fields, self._csv_epoch_rows)
        
        self.writer.flush()
        
        if self.progress_type == "tqdm":
            # Close batch progress bar
            if self.batch_pbar is not None:
                self.batch_pbar.close()
                self.batch_pbar = None
                
            # Update epoch progress bar
            if self.epoch_pbar is not None:
                self.epoch_pbar.update(1)
        else:
            # For table mode, add extra metric columns and finalize row
            if self.progress_table is not None:
                # Add extra metric columns on first epoch (train/val columns added in log_batch)
                if extra_metrics and not hasattr(self, '_extra_columns_added'):
                    for key in extra_metrics.keys():
                        color = "yellow" if "lr" in key.lower() else "magenta"
                        self.progress_table.add_column(f"{key}", color=color)
                    self._extra_columns_added = True
                
                # Update extra metrics
                if extra_metrics:
                    for key, value in extra_metrics.items():
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        self.progress_table[f"{key}"] = self._format_value(value)
                
                self.progress_table.next_row()
            
        # Print epoch summary
        self._print_epoch_summary(epoch, train_metrics, val_metrics, extra_metrics)
        
    def _print_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """Print a formatted summary of the epoch (tqdm mode only, table shows in-line)."""
        # Skip for table mode - the table already displays all info
        if self.progress_type == "table":
            return
            
        parts = [f"Epoch {epoch}/{self.epochs}"]
        
        # Add train metrics (only show 'Train' prefix if validation is enabled)
        for key, value in train_metrics.items():
            if self.use_validation:
                parts.append(f"Train {key}: {self._format_value(value)}")
            else:
                parts.append(f"{key}: {self._format_value(value)}")
            
        # Add val metrics (only if validation is enabled and metrics exist)
        if self.use_validation and val_metrics:
            for key, value in val_metrics.items():
                parts.append(f"Val {key}: {self._format_value(value)}")
            
        # Add extra metrics
        if extra_metrics:
            for key, value in extra_metrics.items():
                parts.append(f"{key}: {self._format_value(value)}")
        
        tqdm.write(" | ".join(parts))
        
    def log_model_graph(
        self, 
        model: torch.nn.Module, 
        input_sample: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]
    ):
        """
        Log model architecture graph to TensorBoard.
        
        Args:
            model: The model to log
            input_sample: Input sample(s) for the model. Can be:
                - A single Tensor for single-input models
                - A Tuple of Tensors for multi-input models
                - A Dict mapping input names to Tensors (will be converted to tuple)
        """
        try:
            # Handle different input types
            if isinstance(input_sample, dict):
                # Convert dict to tuple of tensors (preserving insertion order)
                input_sample = tuple(input_sample.values())
            
            model.eval()
            self.writer.add_graph(model, input_sample, use_strict_trace=False)
            self.writer.flush()
        except Exception as e:
            msg = f"Warning: Could not log model graph: {e}"
            if self.progress_type == "tqdm":
                tqdm.write(msg)
            else:
                print(msg)
            
    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """Log a histogram of values (e.g., weights, gradients)."""
        step = step or self.current_epoch + 1
        self.writer.add_histogram(tag, values, step)
    
    def get_epoch_metrics(self, phase: str = "val") -> Dict[str, float]:
        """
        Get the averaged metrics for the current epoch.
        Useful for early stopping or other decisions based on epoch metrics.
        
        Args:
            phase: Either "train" or "val"
            
        Returns:
            Dictionary of metric names to averaged values
        """
        accum = self._train_metrics_accum if phase == "train" else self._val_metrics_accum
        if phase == "misc":
            accum = self._misc_metrics_accum
        return {k: sum(v) / len(v) for k, v in accum.items() if v}
    
    def get_metric(self, metric_name: str, phase: str = "val") -> Optional[float]:
        """
        Get a specific averaged metric for the current epoch.
        Useful for early stopping checks.
        
        Args:
            metric_name: Name of the metric to retrieve
            phase: Either "train" or "val"
            
        Returns:
            The averaged metric value, or None if not found
        """
        accum = self._train_metrics_accum if phase == "train" else self._val_metrics_accum
        if phase == "misc":
            accum = self._misc_metrics_accum
        if metric_name in accum and accum[metric_name]:
            return sum(accum[metric_name]) / len(accum[metric_name])
        return None
        
    def log_image(self, tag: str, image: torch.Tensor, step: Optional[int] = None):
        """Log an image to TensorBoard."""
        step = step or self.current_epoch + 1
        self.writer.add_image(tag, image, step)
        
    def save_checkpoint(
        self,
        models: Dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        filename: Optional[str] = None,
        extra_state: Optional[Dict] = None
    ):
        """
        Save a training checkpoint.
        
        Args:
            models: Dictionary of model names to model instances
            optimizer: The optimizer
            scheduler: Optional learning rate scheduler
            filename: Optional filename (defaults to checkpoint_epoch_{epoch}.pt)
            extra_state: Additional state to save
        """
        if filename is None:
            filename = f"checkpoint_epoch_{self.current_epoch + 1}.pt"
            
        checkpoint = {
            "epoch": self.current_epoch,
            "optimizer_state_dict": optimizer.state_dict(),
        }
        
        # Save model states
        for name, model in models.items():
            checkpoint[f"{name}_state_dict"] = model.state_dict()
            
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            
        if extra_state:
            checkpoint.update(extra_state)
            
        save_path = os.path.join(self.log_dir, "weights", filename)
        torch.save(checkpoint, save_path)
        
        msg = f"Checkpoint saved: {save_path}"
        if self.progress_type == "tqdm":
            tqdm.write(msg)
        else:
            print(msg)
        
    def close(self):
        """Clean up resources."""
        if self.progress_type == "tqdm":
            if self.batch_pbar is not None:
                self.batch_pbar.close()
            if self.epoch_pbar is not None:
                self.epoch_pbar.close()
        else:
            if self.progress_table is not None:
                self.progress_table.close()
        self.writer.close()
        
    def set_validation_mode(self, use_validation: bool):
        """
        Enable or disable validation mode dynamically.
        
        Args:
            use_validation: Whether to expect validation metrics
        """
        self.use_validation = use_validation
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Exception: {exc_type}, {exc_val}")
        self.close()
        return False


def get_next_run(exp_dir: str) -> int:
    """Get the next run number for experiment logging."""
    os.makedirs(exp_dir, exist_ok=True)
    
    existing_runs = [
        d for d in os.listdir(exp_dir)
        if os.path.isdir(os.path.join(exp_dir, d)) and d.startswith("run")
    ]
    
    if not existing_runs:
        return 1
    
    run_numbers = []
    for run in existing_runs:
        try:
            num = int(run.replace("run", ""))
            run_numbers.append(num)
        except ValueError:
            continue
            
    return max(run_numbers) + 1 if run_numbers else 1


def create_supervised_logger(
    log_dir: str,
    epochs: int,
    run_name: Optional[str] = None,
    progress_type: Literal["tqdm", "table"] = "tqdm",
    save_csv: bool = True,
    save_batch_csv: bool = False,
    save_epoch_csv: bool = True,
    log_batch_tensorboard: bool = False,
    resume_epoch: int = 0,
) -> TrainingLogger:
    """
    Create a TrainingLogger configured for supervised training (with validation).
    
    Args:
        log_dir: Directory to save TensorBoard logs
        epochs: Total number of training epochs
        run_name: Optional name for the run (will be appended to log_dir)
        progress_type: Either "tqdm" or "table" for progress display style
        save_csv: Enable CSV export
        save_batch_csv: Save per-batch CSV rows
        save_epoch_csv: Save per-epoch CSV rows
        log_batch_tensorboard: Log every iteration to TensorBoard (not just epoch averages)
        
    Returns:
        TrainingLogger instance with use_validation=True
    """
    return TrainingLogger(
        log_dir=log_dir,
        epochs=epochs,
        run_name=run_name,
        progress_type=progress_type,
        use_validation=True,
        save_csv=save_csv,
        save_batch_csv=save_batch_csv,
        save_epoch_csv=save_epoch_csv,
        log_batch_tensorboard=log_batch_tensorboard,
        resume_epoch=resume_epoch,
    )


def create_self_supervised_logger(
    log_dir: str,
    epochs: int,
    run_name: Optional[str] = None,
    progress_type: Literal["tqdm", "table"] = "tqdm",
    save_csv: bool = True,
    save_batch_csv: bool = False,
    save_epoch_csv: bool = True,
    log_batch_tensorboard: bool = False,
    resume_epoch: int = 0,
) -> TrainingLogger:
    """
    Create a TrainingLogger configured for self-supervised training (no validation).
    
    Args:
        log_dir: Directory to save TensorBoard logs
        epochs: Total number of training epochs
        run_name: Optional name for the run (will be appended to log_dir)
        progress_type: Either "tqdm" or "table" for progress display style
        save_csv: Enable CSV export
        save_batch_csv: Save per-batch CSV rows
        save_epoch_csv: Save per-epoch CSV rows
        log_batch_tensorboard: Log every iteration to TensorBoard (not just epoch averages)
        
    Returns:
        TrainingLogger instance with use_validation=False
    """
    return TrainingLogger(
        log_dir=log_dir,
        epochs=epochs,
        run_name=run_name,
        progress_type=progress_type,
        use_validation=False,
        save_csv=save_csv,
        save_batch_csv=save_batch_csv,
        save_epoch_csv=save_epoch_csv,
        log_batch_tensorboard=log_batch_tensorboard,
        resume_epoch=resume_epoch,
    )


def test_logger(progress_type: str = "tqdm", use_validation: bool = True):
    """Test function for TrainingLogger with specified progress type and validation mode."""
    import tempfile
    import time
    
    mode_str = "with validation" if use_validation else "no validation (self-supervised)"
    print("=" * 70)
    print(f"Testing TrainingLogger with progress_type='{progress_type}' ({mode_str})")
    print("=" * 70)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize logger
        logger = TrainingLogger(
            log_dir=tmpdir,
            epochs=5,
            run_name=f"test_run_{progress_type}_{mode_str.replace(' ', '_')}",
            progress_type=progress_type,
            use_validation=use_validation
        )
        
        # Simulate training
        with logger:
            logger.start_training(f"Testing Logger ({progress_type} mode)")
            
            for epoch in range(5):
                num_train_batches = 300
                num_val_batches = 100
                
                # Simulate a fake dataloader (just a range)
                fake_train_loader = range(num_train_batches)
                fake_val_loader = range(num_val_batches)
                
                # Training phase - using batch_iterator for progress bar
                logger.start_epoch(epoch, num_train_batches, desc="Training")
                
                for batch_idx in logger.batch_iterator(fake_train_loader):
                    # Simulate some metrics
                    train_metrics = {
                        "Loss": 1.0 - (epoch * 0.1 + batch_idx * 0.01),
                        "L1_Loss": 0.5 - (epoch * 0.05 + batch_idx * 0.005),
                    }
                    logger.log_batch(train_metrics, phase="train")
                    time.sleep(0.02)  # Simulate computation time
                
                # Validation phase (only if validation is enabled)
                if use_validation:
                    logger.start_phase(num_val_batches, desc="Validation")
                    
                    for batch_idx in logger.batch_iterator(fake_val_loader):
                        val_metrics = {
                            "Loss": 1.1 - epoch * 0.15 - batch_idx * 0.01,
                            "L1_Loss": 0.55 - epoch * 0.06 - batch_idx * 0.005,
                        }
                        logger.log_batch(val_metrics, phase="val")
                        time.sleep(0.02)
                
                # Log epoch with extra metrics
                logger.log_epoch(
                    extra_metrics={
                        "LearningRate": 1e-3 * (0.9 ** epoch),
                        "EMA_Momentum": 0.996 + epoch * 0.001
                    }
                )
                
        print("\n" + "=" * 60)
        print(f"Test completed! Logs saved to: {tmpdir}")
        print("=" * 60)
        
        # List created files
        print("\nCreated files:")
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                print(f"  - {os.path.join(root, f)}")


if __name__ == "__main__":
    import sys
    
    # Check command line argument for progress type
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ("tqdm", "table"):
            print(f"\nTesting {mode} mode with validation:")
            test_logger(mode, use_validation=True)
            print(f"\nTesting {mode} mode without validation (self-supervised):")
            test_logger(mode, use_validation=False)
        else:
            print(f"Unknown progress type: {mode}. Use 'tqdm' or 'table'.")
    else:
        # Test both progress types and validation modes
        print("\n" + "=" * 70)
        print("TESTING TQDM MODE WITH VALIDATION")
        print("=" * 70 + "\n")
        test_logger("tqdm", use_validation=True)
        
        print("\n" + "=" * 70)
        print("TESTING TQDM MODE WITHOUT VALIDATION (SELF-SUPERVISED)")
        print("=" * 70 + "\n")
        test_logger("tqdm", use_validation=False)
        
        print("\n\n")
        
        print("=" * 70)
        print("TESTING PROGRESS-TABLE MODE WITH VALIDATION")
        print("=" * 70 + "\n")
        test_logger("table", use_validation=True)
        
        print("\n" + "=" * 70)
        print("TESTING PROGRESS-TABLE MODE WITHOUT VALIDATION (SELF-SUPERVISED)")
        print("=" * 70 + "\n")
        test_logger("table", use_validation=False)
