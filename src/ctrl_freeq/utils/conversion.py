import collections

import numpy as np
import torch


def array_to_tensor(input_data, device="cpu", dtype=None):
    """
    Convert input data to a torch tensor on the specified device and with the specified dtype.

    Args:
    - input_data: The input data to convert. Can be a scalar, list, tuple, NumPy array, or torch tensor.
    - device (str or torch.device): The device to place the tensor on ('cpu' or 'cuda').
    - dtype (torch.dtype, optional): The desired data type of the tensor.

    Returns:
    - torch.Tensor: A tensor on the specified device and with the specified dtype.

    Raises:
    - TypeError: If the input data type is not supported.
    """
    if isinstance(input_data, torch.Tensor):
        return input_data.to(device=device, dtype=dtype)

    elif isinstance(input_data, np.ndarray):
        if dtype is None:
            if np.iscomplexobj(input_data):
                dtype = torch.complex128
            elif np.issubdtype(input_data.dtype, np.floating):
                dtype = torch.float64
            elif np.issubdtype(input_data.dtype, np.integer):
                dtype = torch.int32
        return torch.from_numpy(input_data).to(device=device, dtype=dtype)

    elif isinstance(
        input_data, (int, float, complex, np.integer, np.floating, np.complexfloating)
    ):
        if isinstance(input_data, (np.integer, np.floating, np.complexfloating)):
            input_data = input_data.item()  # Convert NumPy scalar to Python scalar
        if dtype is None:
            if isinstance(input_data, float):
                dtype = torch.float64
            elif isinstance(input_data, complex):
                dtype = torch.complex128
            elif isinstance(input_data, int):
                dtype = torch.int32
        return torch.tensor(input_data, device=device, dtype=dtype)

    elif isinstance(input_data, collections.abc.Iterable):
        input_array = np.array(input_data)
        if dtype is None:
            if np.iscomplexobj(input_array):
                dtype = torch.complex128
            elif np.issubdtype(input_array.dtype, np.floating):
                dtype = torch.float64
            elif np.issubdtype(input_array.dtype, np.integer):
                dtype = torch.int32
        return torch.from_numpy(input_array).to(device=device, dtype=dtype)

    else:
        raise TypeError(
            f"Unsupported input type: {type(input_data)}. Cannot convert to torch tensor."
        )
