import torch


def waveform_gen_piecewise_cart(para: torch.Tensor, n_pulse: int) -> tuple:
    """
    Generate waveforms for piecewise cartesian method.

    Args:
        para (torch.Tensor): Parameters tensor of shape (2 * n_pulse,)
        n_pulse (int): Number of pulse segments

    Returns:
        tuple: (amp, phi, cx, cy) tensors of shape (n_pulse, 1)
    """
    # Reshape parameters to cx and cy components
    cx = para[:n_pulse].reshape(n_pulse, 1)
    cy = para[n_pulse : 2 * n_pulse].reshape(n_pulse, 1)

    # Calculate amplitude and phase
    amp = torch.sqrt(cx**2 + cy**2)
    phi = torch.atan2(cy, cx)

    return amp, phi, cx, cy


def waveform_gen_piecewise_polar(para: torch.Tensor, n_pulse: int) -> tuple:
    """
    Generate waveforms for piecewise polar method.

    Args:
        para (torch.Tensor): Parameters tensor of shape (2 * n_pulse,)
        n_pulse (int): Number of pulse segments

    Returns:
        tuple: (amp, phi, cx, cy) tensors of shape (n_pulse, 1)
    """
    # Reshape parameters to amplitude and phase components
    amp = para[:n_pulse].reshape(n_pulse, 1)
    phi = para[n_pulse : 2 * n_pulse].reshape(n_pulse, 1)

    # Calculate cx and cy from amplitude and phase
    cx = amp * torch.cos(phi)
    cy = amp * torch.sin(phi)

    return amp, phi, cx, cy


def waveform_gen_piecewise_polar_phase(para: torch.Tensor, n_pulse: int) -> tuple:
    """
    Generate waveforms for piecewise polar_phase method.

    Args:
        para (torch.Tensor): Parameters tensor of shape (n_pulse + 1,)
        n_pulse (int): Number of pulse segments

    Returns:
        tuple: (amp, phi, cx, cy) tensors of shape (n_pulse, 1)
    """
    # First parameter is global amplitude, rest are phases
    global_amp = para[0]
    phi = para[1 : n_pulse + 1].reshape(n_pulse, 1)

    # Create constant amplitude
    amp = global_amp * torch.ones(n_pulse, 1)

    # Calculate cx and cy from amplitude and phase
    cx = amp * torch.cos(phi)
    cy = amp * torch.sin(phi)

    return amp, phi, cx, cy


def create_identity_basis_for_piecewise(
    n_pulse: int, n_params: int, dtype=torch.float32
) -> list:
    """
    Create identity-like basis matrices for piecewise optimization.

    For piecewise optimization, we want each parameter to directly correspond to a pulse segment,
    so we use identity matrices as the "basis" to pass to existing CtrlFreeQ waveform generators.

    Args:
        n_pulse (int): Number of pulse segments.
        n_params (int): Number of parameters (basis size).
        dtype (torch.dtype): Data type for the basis matrices.

    Returns:
        list: List containing basis matrices for use with CtrlFreeQ waveform generators.
    """
    # For piecewise, we want direct parameter-to-segment mapping
    # So we use identity matrices (or truncated identity if n_params < n_pulse)
    basis_size = min(n_pulse, n_params)

    # Create two basis matrices for cx and cy components (or amplitude and phase)
    basis_matrices = []
    for _ in range(2):
        # Create identity matrix
        basis = torch.eye(n_pulse, basis_size, dtype=dtype)
        basis_matrices.append(basis)

    return basis_matrices


def get_piecewise_parameter_count(n_pulse: int, method: str) -> int:
    """
    Get the number of parameters required for piecewise optimization.

    Args:
        n_pulse (int): Number of pulse segments.
        method (str): Optimization method ('cart', 'polar', or 'polar_phase').

    Returns:
        int: Number of parameters required.
    """
    if method == "cart":
        return 2 * n_pulse  # cx and cy for each segment
    elif method == "polar":
        return 2 * n_pulse  # amplitude and phase for each segment
    elif method == "polar_phase":
        return n_pulse + 1  # 1 amplitude parameter + n_pulse phase parameters
    else:
        raise ValueError(f"Unknown piecewise method: {method}")


def compare_parameter_efficiency(n_pulse: int, basis_size: int, method: str) -> dict:
    """
    Compare parameter efficiency between CtrlFreeQ (basis) and piecewise methods.

    Args:
        n_pulse (int): Number of pulse segments.
        basis_size (int): Number of basis functions used in CtrlFreeQ.
        method (str): Piecewise method ('cart', 'polar', or 'polar_phase').

    Returns:
        dict: Dictionary containing parameter counts and efficiency ratio.
    """
    piecewise_params = get_piecewise_parameter_count(n_pulse, method)
    ctrlfreeq_params = 2 * basis_size if method != "polar_phase" else basis_size + 1

    return {
        "n_pulse": n_pulse,
        "piecewise_params": piecewise_params,
        "ctrlfreeq_params": ctrlfreeq_params,
        "efficiency_ratio": piecewise_params / ctrlfreeq_params,
        "method": method,
    }
