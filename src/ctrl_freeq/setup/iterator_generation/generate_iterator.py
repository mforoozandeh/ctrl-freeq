def h0_omega_1_iterator_torch(H0, n_rabi, initials, targets):
    """
    Generate expanded H0, initials, targets tensors for all combinations of H0 and Omega_R snapshots.

    H0 is identical across Rabi frequencies, so we use expand() (a memory view
    that shares the underlying storage) instead of copying with fancy indexing.
    Initials and targets are small (state vectors/density matrices) so we use
    repeat() for them since downstream code may write to them.

    Args:
        H0 (torch.Tensor): Tensor of shape [Q, N, N], where Q is the number of H0 matrices.
        initials (torch.Tensor): Tensor of shape [Q, N, N] or [Q, N], initial states.
        targets (torch.Tensor): Tensor of shape [Q, N, N] or [Q, N], target states.

    Returns:
        H0_combined (torch.Tensor): Tensor of shape [M * Q, N, N] (view, no copy).
        initials_combined (torch.Tensor): Tensor of shape [M * Q, N, N] or [M * Q, N].
        targets_combined (torch.Tensor): Tensor of shape [M * Q, N, N] or [M * Q, N].
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

    # Expand H0 using a view (no memory copy) — H0 is the same for all Rabi
    # frequencies, so we insert a dimension and expand along it, then reshape
    # to the flat [M*Q, N, N] layout the rest of the code expects.
    # expand() returns a read-only view sharing the same memory.
    H0_combined = H0.unsqueeze(1).expand(Q, M, N, N).reshape(Q * M, N, N)

    # Initials/targets are small; use repeat() which copies but is negligible
    # compared to H0 (state vectors are N-dimensional, H0 matrices are N×N).
    if initials.dim() == 2:
        initials_combined = initials.unsqueeze(1).expand(Q, M, N).reshape(Q * M, N)
        targets_combined = targets.unsqueeze(1).expand(Q, M, N).reshape(Q * M, N)
    else:
        initials_combined = (
            initials.unsqueeze(1).expand(Q, M, N, N).reshape(Q * M, N, N)
        )
        targets_combined = targets.unsqueeze(1).expand(Q, M, N, N).reshape(Q * M, N, N)

    return H0_combined, initials_combined, targets_combined
