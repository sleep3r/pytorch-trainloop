import os
import random

import torch
import numpy as np


def set_random_seed(seed: int = 228, precision: int = 10, deterministic: bool = False) -> None:
    """
    Seeds all.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for CUDNN backend,
                              i.e., set `torch.backends.cudnn.deterministic` to True and
                              `torch.backends.cudnn.benchmark` to False. Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=precision)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


def reset_all_weights(model: torch.nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: torch.nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)
