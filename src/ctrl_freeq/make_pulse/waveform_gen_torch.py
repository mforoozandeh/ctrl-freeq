import torch


def waveform_gen_polar_phase(para: torch.Tensor, mat: list) -> tuple:
    """
    Generate waveform parameters using polar phase parameterization.

    Args:
        para (torch.Tensor): Parameter tensor of shape (N+1,).
        mat (list of torch.Tensor): List containing two matrices:
            - mat[0]: Tensor of shape (M, N).
            - mat[1]: Tensor of shape (M, N).

    Returns:
        tuple: A tuple containing (amp, phi, cx, cy) where each is a torch.Tensor.
    """
    # Compute amplitude
    amp = mat[0][:, 0] * para[0]
    amp = amp.unsqueeze(1)  # Convert to shape (M, 1)

    # Compute phase
    phi = mat[1] @ (para[1:] * torch.pi)
    phi = phi.unsqueeze(1)  # Convert to shape (M, 1)

    # Compute Cartesian coordinates
    cx = amp * torch.cos(phi)
    cy = amp * torch.sin(phi)

    return amp, phi, cx, cy


def waveform_gen_polar(para: torch.Tensor, mat: list) -> tuple:
    """
    Generate waveform parameters using polar parameterization.

    Args:
        para (torch.Tensor): Parameter tensor of shape (2*L,).
        mat (list of torch.Tensor): List containing two matrices:
            - mat[0]: Tensor of shape (M, L).
            - mat[1]: Tensor of shape (N, L).

    Returns:
        tuple: A tuple containing (amp, phi, cx, cy) where each is a torch.Tensor.
    """
    # Split parameters into two halves
    ll = para.size(0) // 2
    para = para.unsqueeze(1)  # Convert to shape (2*L, 1)
    c1 = para[:ll]  # First half for amplitude
    c2 = para[ll:]  # Second half for phase

    # Compute amplitude and phase
    amp = mat[0] @ c1  # Shape: (M, 1)
    phi = mat[1] @ (c2 * torch.pi)  # Shape: (N, 1)

    # Compute Cartesian coordinates
    cx = amp * torch.cos(phi)
    cy = amp * torch.sin(phi)

    return amp, phi, cx, cy


def waveform_gen_cart(para: torch.Tensor, mat: list) -> tuple:
    """
    Generate waveform parameters using Cartesian parameterization.

    Args:
        para (torch.Tensor): Parameter tensor of shape (2*L,).
        mat (list of torch.Tensor): List containing two matrices:
            - mat[0]: Tensor of shape (M, L) for x-components.
            - mat[1]: Tensor of shape (M, L) for y-components.

    Returns:
        tuple: A tuple containing (amp, phi, cx, cy) where each is a torch.Tensor.
    """
    # Split parameters into two halves
    ll = para.size(0) // 2
    para = para.unsqueeze(1)  # Convert to shape (2*L, 1)
    c1 = para[:ll]  # Parameters for x-components
    c2 = para[ll:]  # Parameters for y-components

    # Compute Cartesian coordinates
    cx = mat[0] @ c1  # Shape: (M, 1)
    cy = mat[1] @ c2  # Shape: (M, 1)

    # Compute amplitude and phase
    amp = torch.sqrt(cx**2 + cy**2)
    phi = torch.atan2(cy, cx)

    return amp, phi, cx, cy
