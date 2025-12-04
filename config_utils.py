import json
import os
from typing import Any, Dict
from pathlib import Path


class Config:
    """Configuration manager for neural subspace training."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
        
    @classmethod
    def from_file(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(config_dict)
    
    def __getitem__(self, key: str) -> Any:
        """Access config sections like config['training']."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        return self._config.get(key, default)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self._config, f, indent=4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return raw dictionary."""
        return self._config.copy()
    
    # Convenience accessors for nested values
    @property
    def system_name(self) -> str:
        return self._config['system']['name']
    
    @property
    def problem_name(self) -> str:
        return self._config['system']['problem_name']
    
    @property
    def subspace_dim(self) -> int:
        return self._config['subspace']['dim']
    
    @property
    def shape_space_dim(self) -> int:
        return self._config['subspace']['shape_space_dim']
    
    @property
    def batch_size(self) -> int:
        return self._config['training']['batch_size']
    
    @property
    def n_train_iters(self) -> int:
        return self._config['training']['n_train_iters']
    
    @property
    def learning_rate(self) -> float:
        return self._config['optimizer']['learning_rate']
    
    @property
    def output_dir(self) -> str:
        return self._config['logging']['output_dir']
    
    @property
    def experiment_name(self) -> str:
        return self._config['logging']['experiment_name']


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from file or return default.
    
    Args:
        config_path: Path to config JSON file. If None, looks for 'config.json'
    
    Returns:
        Config object
    """
    if config_path is None:
        config_path = 'config.json'
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return Config.from_file(config_path)
