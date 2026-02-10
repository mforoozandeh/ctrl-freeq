import os
import random

import numpy as np
import torch
from ctrl_freeq.utils.colored_logging import setup_colored_logging

# Initialize logger for optimization process
logger = setup_colored_logging(level="INFO")


def generate_instances(param, sigma, snapshots):
    param_instances = []

    if isinstance(param, (list, np.ndarray)) and isinstance(sigma, (list, np.ndarray)):
        if len(param) != len(sigma):
            raise ValueError("param and sigma must have the same length")

        if all(x == 0 for x in sigma):
            param_instance = np.array([par for par in param])
            param_instances.append(param_instance)
        else:
            for _ in range(snapshots):
                param_instance = np.array(
                    [random.gauss(par, sig) for par, sig in zip(param, sigma)]
                )
                param_instances.append(param_instance)
    else:
        raise ValueError("param and sigma must be lists or numpy arrays")

    return param_instances


def convert_attributes_to_numpy(obj):
    for attr in dir(obj):
        if not attr.startswith("__") and not callable(getattr(obj, attr)):
            value = getattr(obj, attr)
            if isinstance(value, list):
                if all(isinstance(i, (int, float, complex, np.ndarray)) for i in value):
                    setattr(obj, attr, np.array(value))


def set_cores(num_cores=None):
    max_cores = os.cpu_count() or 1

    # Default policy: use max(1, total_cores - 1)
    if num_cores is None:
        num_cores = max(1, max_cores - 1)
    else:
        try:
            num_cores = int(num_cores)
        except Exception:
            num_cores = max(1, max_cores - 1)
        # clamp to [1, max_cores]
        if num_cores < 1:
            num_cores = 1
        if num_cores > max_cores:
            num_cores = max_cores

    torch.set_num_threads(num_cores)

    logger.info(
        f"Number of threads set to {num_cores} out of {max_cores} available cores."
    )
