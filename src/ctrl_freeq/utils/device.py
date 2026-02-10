import os
import logging
from typing import Optional, Tuple

import torch
from ctrl_freeq.utils.colored_logging import setup_colored_logging

logger = setup_colored_logging(level="INFO")


def select_device(compute_resource: str) -> Tuple[torch.device, str]:
    """
    Select compute device based on requested resource.

    Returns (device, backend_str) where backend_str is "cuda" or "cpu".
    Falls back to CPU if CUDA is unavailable and logs a warning.
    """
    choice = (compute_resource or "cpu").lower()
    if choice == "gpu":
        if torch.cuda.is_available():
            logger.info("Compute resource set to GPU (CUDA available).")
            return torch.device("cuda"), "cuda"
        else:
            message = "CUDA not available; falling back to CPU. To use GPU, run on a CUDA-enabled environment."
            # Log via our colored logger
            logger.warning(message)
            # Also log via root logger so pytest caplog can capture the warning
            logging.warning(message)
            return torch.device("cpu"), "cpu"
    # default CPU
    return torch.device("cpu"), "cpu"


def resolve_cpu_cores(requested: Optional[int]) -> int:
    """
    Resolve effective CPU core count.
    - Default: max(1, total_cores - 1)
    - If requested provided: clamp to [1, total_cores]
    """
    total = os.cpu_count() or 1
    default_cores = max(1, total - 1)

    if requested is None:
        return default_cores

    # clamp requested value
    try:
        req = int(requested)
    except Exception:
        req = default_cores

    if req < 1:
        req = 1
    if req > total:
        req = total
    return req
