import torch


def h0_omega_1_iterator_torch(H0, n_rabi, initials, targets):
    """
    Generate expanded H0, initials, targets tensors for all combinations of H0 and Omega_R snapshots.

    Args:
        H0 (torch.Tensor): Tensor of shape [Q, N, N], where Q is the number of H0 matrices.
        initials (torch.Tensor): Tensor of shape [Q, N, N] or [Q, N], initial states.
        targets (torch.Tensor): Tensor of shape [Q, N, N] or [Q, N], target states.

    Returns:
        H0_combined (torch.Tensor): Tensor of shape [M * Q, N, N].
        initials_combined (torch.Tensor): Tensor of shape [M * Q, N, N] or [M * Q, N], matching the shape of initials.
        targets_combined (torch.Tensor): Tensor of shape [M * Q, N, N] or [M * Q, N], matching the shape of targets.
    """

    # Get the sizes
    Q = H0.shape[0]  # Number of H0 matrices
    N = H0.shape[1]  # Size of the matrices/vectors
    M = n_rabi  # Number of Omega_R snapshots

    # Check that initials and targets have the same shape
    if initials.shape != targets.shape:
        raise ValueError("Initials and targets must have the same shape.")

    # Determine the shape of initials and targets
    if not (
        (initials.dim() == 2 and initials.shape == (Q, N))
        or (initials.dim() == 3 and initials.shape == (Q, N, N))
    ):
        raise ValueError("Initials and targets must have shape [Q, N] or [Q, N, N].")

    # Create meshgrid of indices
    H0_idx = torch.arange(Q)
    omega_1_idx = torch.arange(M)
    grid_H0_idx, _ = torch.meshgrid(H0_idx, omega_1_idx, indexing="ij")

    # Flatten indices
    H0_idx_flat = grid_H0_idx.reshape(-1)  # Shape [M * Q]

    # Gather values
    H0_combined = H0[H0_idx_flat]  # Shape [M * Q, N, N]
    initials_combined = initials[H0_idx_flat]  # Shape depends on initials_shape
    targets_combined = targets[H0_idx_flat]  # Shape depends on initials_shape

    return H0_combined, initials_combined, targets_combined
