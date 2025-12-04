import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    """
    Unified logging for training with console output and optional tensorboard.
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Create run with timestamp as name (for tensorboard run list)
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / experiment_name / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup console logging
        self.log_file = self.run_dir / "training.log"
        
        # Setup tensorboard - write directly to experiment dir for multi-run view
        # Each run gets its own subdirectory named by timestamp
        experiment_tb_dir = self.log_dir / experiment_name
        run_tb_dir = experiment_tb_dir / run_name
        self.writer = SummaryWriter(log_dir=str(run_tb_dir))
        
        # Log tensorboard instructions
        self.log_message(f"Tensorboard logging to: {run_tb_dir}")
        self.log_message(f"  View all runs with: tensorboard --logdir={experiment_tb_dir}")
        self.log_message(f"  Then open: http://localhost:6006\n")
        
        # Save config
        if config is not None:
            config_path = self.run_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
        
        # Log startup info
        self.log_message(f"{'='*80}")
        self.log_message(f"Experiment: {experiment_name}")
        self.log_message(f"Run directory: {self.run_dir}")
        self.log_message(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"{'='*80}\n")
    
    def log_message(self, message: str):
        """Log a message to console and file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to tensorboard if enabled."""
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration as text."""
        if self.writer is not None:
            config_str = json.dumps(config, indent=2)
            self.writer.add_text('config', config_str, 0)
    
    def log_scalars(self, tag: str, scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars with same tag."""
        if self.writer is not None:
            self.writer.add_scalars(tag, scalar_dict, step)
    
    def flush(self):
        """Flush tensorboard writer."""
        if self.writer is not None:
            self.writer.flush()
    
    def close(self):
        """Close logger and tensorboard writer."""
        if self.writer is not None:
            self.writer.close()
        
        self.log_message(f"\n{'='*80}")
        self.log_message(f"Training finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"{'='*80}")
    
    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory for this run."""
        ckpt_dir = self.run_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        return ckpt_dir
