import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any


class MetricsTracker:
    """Track training metrics during DBNet++ training"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new epoch"""
        self.metrics = defaultdict(list)
        self.epoch_metrics = {}
    
    def update(self, **kwargs):
        """Update metrics with new values"""
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].append(value)
    
    def get_epoch_summary(self) -> Dict[str, float]:
        """Get average metrics for the current epoch"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = np.mean(values)
        return summary
    
    def save_epoch(self, epoch: int):
        """Save current epoch metrics and reset"""
        self.epoch_metrics[epoch] = self.get_epoch_summary()
        self.reset()
    
    def get_best_metric(self, metric_name: str, minimize: bool = True) -> tuple[int, float]:
        """Get the best epoch and value for a specific metric"""
        if not self.epoch_metrics:
            return -1, float('inf') if minimize else float('-inf')
        
        values = [(epoch, metrics.get(metric_name, float('inf') if minimize else float('-inf'))) 
                 for epoch, metrics in self.epoch_metrics.items()]
        
        if minimize:
            best_epoch, best_value = min(values, key=lambda x: x[1])
        else:
            best_epoch, best_value = max(values, key=lambda x: x[1])
        
        return best_epoch, best_value
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric across epochs"""
        return [metrics.get(metric_name, 0.0) for metrics in self.epoch_metrics.values()]


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __repr__(self):
        return f'{self.name}: {self.avg:.4f} (current: {self.val:.4f})'


def compute_text_detection_metrics(pred_binary: torch.Tensor, 
                                 gt_text: torch.Tensor,
                                 threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute basic text detection metrics
    
    Args:
        pred_binary: Predicted binary map (B, 1, H, W)
        gt_text: Ground truth text mask (B, 1, H, W)  
        threshold: Threshold for binarization
        
    Returns:
        Dictionary of metrics
    """
    # Binarize predictions
    pred_binary = (pred_binary > threshold).float()
    gt_text = (gt_text > 0.5).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1)
    gt_flat = gt_text.view(-1)
    
    # Calculate basic metrics
    tp = (pred_flat * gt_flat).sum().item()
    fp = (pred_flat * (1 - gt_flat)).sum().item()
    fn = ((1 - pred_flat) * gt_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - gt_flat)).sum().item()
    
    # Compute metrics
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'iou': iou,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def format_metrics_string(metrics: Dict[str, float], prefix: str = '') -> str:
    """Format metrics dictionary as a readable string"""
    if prefix:
        prefix = f"{prefix}_"
    
    formatted = []
    for key, value in metrics.items():
        formatted.append(f"{prefix}{key}: {value:.4f}")
    
    return " | ".join(formatted)