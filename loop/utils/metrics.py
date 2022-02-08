from typing import Dict

import torch
import segmentation_models_pytorch as smp

from config import DLConfig


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# --- metrics factory ---
METRICS = {
    "iou": smp.utils.metrics.IoU(threshold=0.5)
}


def calc_metrics(cfg: DLConfig, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Main function calling from train for metrics calculation.

    Args:
        cfg (DLConfig): train config;
        outputs (torch.Tensor): prediction strings;
        targets (torch.Tensor): ground truth strings;
    Returns:
        metrics (Dict[str, torch.Tensor]): metrics dict.
    """
    metrics = {}

    for metric in cfg.evaluation.metrics:
        if metric in METRICS:
            metrics[metric] = torch.FloatTensor([METRICS[metric](outputs, targets)])  # noqa
            metrics[metric] = metrics[metric].to(f"cuda:{cfg.local_rank}")
        else:
            raise ValueError(f"WTF metric: {metric}")
    return metrics
