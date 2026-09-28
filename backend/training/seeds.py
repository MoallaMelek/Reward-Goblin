"""Seed every source of randomness we control."""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    import torch
    from stable_baselines3.common.utils import set_random_seed

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # single-threaded CPU math is both faster here and reproducible
    set_random_seed(seed)


def seed_list(n: int, base: int = 1) -> list[int]:
    return [base + i for i in range(n)]
