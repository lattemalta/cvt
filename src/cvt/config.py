from dataclasses import dataclass
from typing import Dict, Any


class ModelConfig:
    def __init__(self) -> None:
        self.num_epochs = 50
        self.batch_size = 32
        self.lr = 0.01
        self.img_size = 32
        self.patch_size = 16
        self.num_inputlayer_units = 512
        self.num_heads = 4
        self.num_mlp_units = 512
        self.num_layers = 6
        self.batch_size = 32


@dataclass
class TrainingConfig:
    """Configuration for DBNet++ training"""
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Image parameters
    img_size: tuple[int, int] = (640, 640)
    
    # Model parameters
    backbone: str = 'resnet18'  # 'resnet18' or 'resnet34'
    
    # Loss weights
    loss_weights: Dict[str, float] = None
    
    # Training settings
    save_interval: int = 10
    log_interval: int = 50
    num_workers: int = 4
    
    # Device settings
    device: str = 'auto'  # 'auto', 'cuda', 'mps', 'cpu'
    mixed_precision: bool = True
    
    # Checkpoint settings
    checkpoint_dir: str = './checkpoints'
    resume_from: str = None
    save_best_only: bool = True
    
    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    
    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {'alpha': 1.0, 'beta': 10.0, 'gamma': 1.0}
