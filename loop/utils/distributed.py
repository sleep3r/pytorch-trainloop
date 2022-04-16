import os

import torch
import torch.multiprocessing as mp
from torch import distributed as dist

from config import DLConfig


def init_distributed(cfg: DLConfig) -> bool:
    """Initializes distributed training if cfg.training.distributed == true."""
    if not cfg.training.distributed:
        distributed = False
        cfg.local_rank, cfg.world_size = get_dist_info()
    else:
        distributed = True

        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method('spawn')

        cfg.local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(**cfg.training.dist_params)

        # re-set gpu_ids with distributed training mode
        _, cfg.world_size = get_dist_info()
    return distributed


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False

    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)  # noqa
    rt /= n
    return rt


def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)  # noqa
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)  # noqa
