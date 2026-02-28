import torch

from ctrl_freeq.conditions.stopping_conds import OptimizationInterrupted
from ctrl_freeq.utils.colored_logging import setup_colored_logging


class CtrlFreeQ:
    def __init__(
        self,
        n_para,
        n_qubits,
        op,
        rabi_freq,
        n_pulse,
        n_h0,
        n_rabi,
        mat,
        H0,
        dt,
        initial_state,
        target_state,
        wf_fun,
        u_fun,
        state_fun,
        fid_fun,
        targ_fid,
        me,
        collapse_ops=None,
        hamiltonian_model=None,
        control_ops=None,
    ):
        self.n_para = n_para
        self.n_qubits = n_qubits
        self.op = op
        self.rabi_freq = rabi_freq
        self.n_pulse = n_pulse
        self.n_h0 = n_h0
        self.n_rabi = n_rabi
        self.mat = mat
        self.H0 = H0
        self.dt = dt
        self.initial_state = initial_state
        self.target_state = target_state
        self.wf_fun = wf_fun
        self.u_fun = u_fun
        self.state_fun = state_fun
        self.fid_fun = fid_fun
        self.iter = 0
        self.exit_val = targ_fid
        self.me = me
        self.collapse_ops = collapse_ops

        # Model-based (generic) path: when a HamiltonianModel is provided,
        # pulse_hamiltonian_generic is used instead of the legacy function.
        self.hamiltonian_model = hamiltonian_model
        self.control_ops = control_ops

        self.fid = None
        self.pen = None
        self.fidelity_history = []  # Track fidelity per iteration

        # Flag for COBYLA early termination (avoids exception-based termination)
        self.early_termination_flag = False
        self.early_termination_solution = None

        # Initialize logger for optimization progress
        self.logger = setup_colored_logging(level="INFO")

    def objective_function(self, para):
        parameters = torch.split(para, list(self.n_para))
        amps, phis, cxs, cys = pulse_para(
            self.n_qubits, parameters, self.mat, self.wf_fun, self.me
        )
        self.pen = penalty(amps)

        if self.hamiltonian_model is not None:
            # Generic path: works for any HamiltonianModel
            u = self.hamiltonian_model.control_amplitudes(
                cxs, cys, self.rabi_freq, self.n_h0
            )
            Hp = pulse_hamiltonian_generic(u, self.control_ops)
        else:
            # Legacy spin-chain path (backward compatible)
            Hp = pulse_hamiltonian(
                cxs,
                cys,
                self.rabi_freq,
                self.op,
                self.n_pulse,
                self.n_h0,
                self.n_rabi,
                self.n_qubits,
            )

        # Use optimized simulator (main optimization - 6.28x speedup from vectorized matrix exponentials)
        state = simulator_optimized(
            self.H0,
            Hp,
            self.dt,
            self.initial_state,
            self.u_fun,
            self.state_fun,
            collapse_ops=self.collapse_ops,
        )
        self.fid = self.fid_fun(state, self.target_state)
        self.cost = -self.fid + self.pen
        return self.cost

    def callback_function(self, para):
        if self.iter == 0:
            self.logger.info("=" * 33)
            self.logger.info(f"{'Iteration':<10} | {'Fidelity':<10} | {'Penalty':<10}")
            self.logger.info("=" * 33)

        self.iter += 1
        current_fidelity = -self.cost  # cost = -fidelity + penalty, so fidelity = -cost
        self.fidelity_history.append(
            current_fidelity.item()
            if hasattr(current_fidelity, "item")
            else float(current_fidelity)
        )
        self.logger.info(
            f"{self.iter:<10} | {current_fidelity:<10.4f} | {self.pen:<10.4f}"
        )

        if current_fidelity >= self.exit_val:
            raise OptimizationInterrupted(
                f"Objective function reached target fidelity = {self.exit_val}, exiting optimization...",
                para,
            )

    def callback_function_cobyla(self, para):
        """
        COBYLA-specific callback function that uses flag-based termination
        instead of raising exceptions to avoid COBYLA callback failures.
        """
        # Skip callback if early termination has already been triggered
        if self.early_termination_flag:
            return

        if self.iter == 0:
            self.logger.info("=" * 33)
            self.logger.info(f"{'Iteration':<10} | {'Fidelity':<10} | {'Penalty':<10}")
            self.logger.info("=" * 33)

        self.iter += 1
        current_fidelity = -self.cost  # cost = -fidelity + penalty, so fidelity = -cost
        self.fidelity_history.append(
            current_fidelity.item()
            if hasattr(current_fidelity, "item")
            else float(current_fidelity)
        )
        self.logger.info(
            f"{self.iter:<10} | {current_fidelity:<10.4f} | {self.pen:<10.4f}"
        )

        if current_fidelity >= self.exit_val:
            self.early_termination_flag = True
            self.early_termination_solution = para
            self.logger.warning(
                f"Objective function reached target fidelity = {self.exit_val}, exiting optimization..."
            )


def pulse_para(n_qubits, parameters, mat, wf_fun, me):
    amps_list = []
    phis_list = []
    cxs_list = []
    cys_list = []

    for i in range(n_qubits):
        amp, phi, cx, cy = wf_fun[i](parameters[i], mat[i])
        amps_list.append(amp)  # Each amp has shape [100, 1]
        phis_list.append(phi)
        cxs_list.append(cx)
        cys_list.append(cy)

    # Concatenate along dim=1 to get shape [100, n_qubits]
    amps = torch.cat(amps_list, dim=1)
    phis = torch.cat(phis_list, dim=1)
    cxs = torch.cat(cxs_list, dim=1)
    cys = torch.cat(cys_list, dim=1)

    cxs, cys = modulate_waveforms(cxs, cys, me)

    return amps, phis, cxs, cys


def pulse_hamiltonian(cx, cy, rabi_freq, op_tensor, n_pulse, n_h0, n_rabi, n_qubits):
    """
    Generate the pulse Hamiltonian for N qubits using op_tensor directly.

    Args:
        cx (torch.Tensor): Tensor of shape (n_pulse, N).
        cy (torch.Tensor): Tensor of shape (n_pulse, N).
        rabi_freq (torch.Tensor): Tensor of shape (n_rabi, N).
        op_tensor (torch.Tensor): Tensor of shape (3, N, D, D).
        n_pulse (int): Number of pulses.
        n_h0 (int): Number of H0 configurations.
        n_rabi (int): Number of Rabi frequencies.
        n_qubits (int): Number of qubits.

    Returns:
        torch.Tensor: Tensor of shape (n_pulse, n_rabi * n_h0, D, D).
    """
    D = 2**n_qubits

    # Ensure that cx and cy have shape (n_pulse, n_qubits)
    cx = cx.reshape(n_pulse, n_qubits)
    cy = cy.reshape(n_pulse, n_qubits)

    # Ensure that rabi_freq has shape (n_rabi, n_qubits)
    rabi_freq = rabi_freq.reshape(n_rabi, n_qubits)

    # Initialize the tensor of shape (n_pulse, n_rabi * n_h0, D, D)
    Hp_total = torch.zeros(
        n_pulse, n_rabi * n_h0, D, D, dtype=op_tensor.dtype, device=op_tensor.device
    )

    for i in range(n_qubits):
        # Operators for qubit i
        X_i = op_tensor[0, i].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, D)
        Y_i = op_tensor[1, i].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, D)

        # Pulse parameters for qubit i
        cx_i = (
            cx[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        )  # Shape: (n_pulse, 1, 1, 1)
        cy_i = (
            cy[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        )  # Shape: (n_pulse, 1, 1, 1)

        # Rabi frequencies for qubit i
        rabi_i = (
            rabi_freq[:, i].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )  # Shape: (1, n_rabi, 1, 1)

        # Expand rabi_i to match n_h0
        rabi_i = rabi_i.repeat(1, n_h0, 1, 1)  # Shape: (1, n_rabi * n_h0, 1, 1)

        # Compute Hamiltonian contribution for qubit i
        Hp_i = cx_i * X_i + cy_i * Y_i  # Shape: (n_pulse, 1, D, D)

        # Expand Hp_i to match n_rabi * n_h0
        Hp_i = Hp_i.expand(
            -1, n_rabi * n_h0, -1, -1
        )  # Shape: (n_pulse, n_rabi * n_h0, D, D)

        # Multiply by rabi frequencies
        Hp_i = Hp_i * rabi_i  # Shape: (n_pulse, n_rabi * n_h0, D, D)

        # Sum contributions from all qubits
        Hp_total += Hp_i

    return Hp_total  # Shape: (n_pulse, n_rabi * n_h0, D, D)


def pulse_hamiltonian_generic(amplitudes, control_ops_tensor):
    """Build the pulse Hamiltonian using the generic bilinear control formulation.

    Computes  Hp(t, b) = sum_k  u_k(t, b) * H_ctrl_k  for every time step t
    and batch element b using a single ``einsum``.

    This replaces the spin-chain-specific ``pulse_hamiltonian`` and works for
    any Hamiltonian model (spin chain, superconducting, etc.).

    Args:
        amplitudes (torch.Tensor): Control amplitudes of shape
            ``(n_pulse, n_batch, n_controls)`` as returned by
            ``HamiltonianModel.control_amplitudes()``.
        control_ops_tensor (torch.Tensor): Fixed control operators of shape
            ``(n_controls, D, D)`` as returned by
            ``HamiltonianModel.control_ops_tensor()``.

    Returns:
        torch.Tensor: Pulse Hamiltonian of shape ``(n_pulse, n_batch, D, D)``.
    """
    # Promote real amplitudes to complex so einsum can contract with complex operators
    if not amplitudes.is_complex() and control_ops_tensor.is_complex():
        amplitudes = amplitudes.to(control_ops_tensor.dtype)
    return torch.einsum("tbk,kij->tbij", amplitudes, control_ops_tensor)


def simulator(H0, Hp, dt, initial_state, u_fun, state_fun):
    """
    Simulates the quantum system with flattened batch dimensions.

    Parameters:
    - H0: Tensor of shape (batch_size, D, D)
    - Hp: Tensor of shape (n_pulse, batch_size, D, D)
    - dt: Time step (float)
    - initial_state: Tensor of shape (batch_size, D)

    Returns:
    - state: Tensor of shape (n_h0, n_rabi, D)
    """
    state = initial_state  # Shape: (batch_size, D)

    for n in range(Hp.shape[0]):
        H = H0 + Hp[n]  # Shape: (batch_size, D, D)
        U = u_fun(H, dt)  # Shape: (batch_size, D, D)
        state = state_fun(U, state)

    return state


def simulator_optimized(H0, Hp, dt, initial_state, u_fun, state_fun, collapse_ops=None):
    """
    Optimized simulator that vectorizes matrix exponential computation in chunks.

    Processes time steps in chunks to cap peak memory usage at
    ``chunk_size * batch_size * D * D`` instead of allocating the full
    ``n_pulse * batch_size * D * D`` tensor all at once.

    Parameters:
    - H0: Tensor of shape (batch_size, D, D)
    - Hp: Tensor of shape (n_pulse, batch_size, D, D)
    - dt: Time step (float)
    - initial_state: Tensor of shape (batch_size, D) or (batch_size, D, D) for density matrices
    - u_fun: Matrix exponential function
    - state_fun: State evolution function (state_hilbert, state_liouville, or state_lindblad)
    - collapse_ops: Optional tensor of shape (n_ops, D, D) for Lindblad dissipation

    Returns:
    - state: Tensor of shape (batch_size, D) or (batch_size, D, D)
    """
    n_pulse, batch_size, D, _ = Hp.shape

    # Chunk size balances vectorisation benefit vs. peak memory.
    # For small problems the whole batch fits; for large problems we cap it.
    # 32 time-steps per chunk keeps peak allocation reasonable while still
    # benefiting from batched matrix_exp.
    chunk_size = min(n_pulse, 32)

    state = initial_state

    # Build extra kwargs for state_fun (only state_lindblad needs dt and collapse_ops)
    use_lindblad = collapse_ops is not None

    for start in range(0, n_pulse, chunk_size):
        end = min(start + chunk_size, n_pulse)
        chunk_len = end - start

        # Slice the pulse Hamiltonian for this chunk: (chunk_len, batch_size, D, D)
        Hp_chunk = Hp[start:end]

        # Add H0 via broadcasting — H0 is (batch_size, D, D), Hp_chunk is
        # (chunk_len, batch_size, D, D).  The addition broadcasts H0 over the
        # time dimension without an explicit expand + separate allocation.
        H_chunk = Hp_chunk + H0.unsqueeze(0)  # (chunk_len, batch_size, D, D)

        # Flatten for batched matrix exponential
        H_flat = H_chunk.reshape(chunk_len * batch_size, D, D)
        U_flat = u_fun(H_flat, dt)  # (chunk_len * batch_size, D, D)
        U_chunk = U_flat.reshape(chunk_len, batch_size, D, D)

        # Apply time evolution for this chunk
        if use_lindblad:
            for n in range(chunk_len):
                state = state_fun(U_chunk[n], state, dt, collapse_ops)
        else:
            for n in range(chunk_len):
                state = state_fun(U_chunk[n], state)

    return state


def fidelity_hilbert(a_mat, b_mat):
    """
        Calculate the fidelity between two matrices a_mat and b_mat using
    torch.mean.

        Parameters:
        a_mat (torch.Tensor): A matrix of shape (batch_size, D)
        b_mat (torch.Tensor): A matrix of shape (batch_size, D)

        Returns:
        torch.Tensor: The fidelity value
    """
    fidelity = torch.abs((a_mat * b_mat.conj()).sum(dim=1)) ** 2
    return torch.mean(fidelity)


def fidelity_liouville(rho, sigma):
    """
    Computes the Uhlmann-Jozsa fidelity for two batches of density matrices.

    Parameters:
    rho (torch.Tensor): A tensor of shape (batch_size, D, D)
    sigma (torch.Tensor): A tensor of shape (batch_size, D, D)

    Returns:
    torch.Tensor: A tensor containing the mean fidelity over the batch.
    """
    sqrt_rho = matrix_square_root(rho)  # shape: (batch_size, D, D)
    intermediate = torch.bmm(sqrt_rho, torch.bmm(sigma, sqrt_rho))
    sqrtm = matrix_square_root(intermediate)
    trace_sqrtm = torch.diagonal(sqrtm, dim1=-2, dim2=-1).sum(
        -1
    )  # shape: (batch_size,)

    fidelity = trace_sqrtm.real**2  # shape: (batch_size,)

    return torch.mean(fidelity)


def penalty(amp):
    amp = amp.t().reshape(-1)
    max_amp = torch.max(torch.abs(amp))
    if max_amp > 1:
        pen = (max_amp - 1) ** 2
    else:
        # Ensure the zero tensor is on the same device and dtype as amp to avoid device/dtype mismatch
        pen = torch.tensor(0.0, dtype=amp.dtype, device=amp.device)
    return pen


def exp_mat_exact(H, dt):
    """
    Compute the exact matrix exponential of a 2x2 Hermitian matrix H.

    Parameters:
    H (torch.Tensor): A tensor of shape (..., 2, 2) containing Hermitian matrices.
    dt (float): Time step.

    Returns:
    torch.Tensor: The matrix exponential of H.
    """
    # Extract components from the Hermitian matrix H
    h1 = torch.real(H[:, 0, 1])
    h2 = torch.imag(H[:, 0, 1])
    h3 = H[:, 0, 0]

    # Calculate the magnitude of the vector h
    h_magnitude = torch.sqrt(h1**2 + h2**2 + h3**2)

    # Compute the unitary operator U(t)
    cos_term = torch.cos(dt * h_magnitude).unsqueeze(-1).unsqueeze(-1) * torch.eye(
        2, dtype=H.dtype, device=H.device
    ).unsqueeze(0)
    sin_term = (
        -1j
        * torch.sin(dt * h_magnitude).unsqueeze(-1).unsqueeze(-1)
        / h_magnitude.unsqueeze(-1).unsqueeze(-1)
        * H
    )

    U_t = cos_term + sin_term

    return U_t


def exp_mat_torch(H, dt):
    """
    Compute the matrix exponential of a DxD Hermitian matrix H using torch.linalg.matrix_exp.

    Parameters:
    H (torch.Tensor): A tensor of shape (..., D, D) containing Hermitian matrices.
    dt (float): Time step.

    Returns:
    torch.Tensor: The matrix exponential of H.
    """
    return torch.linalg.matrix_exp(-1j * H * dt)  # Shape: (batch_size, D, D)


def state_hilbert(U, state):
    """
    Apply the unitary operator U to the state.

    Parameters:
    U (torch.Tensor): A tensor of shape (batch_size, D, D) containing the unitary operator.
    state (torch.Tensor): A tensor of shape (batch_size, D) containing the state vector.

    Returns:
    torch.Tensor: The state vector of shape (batch_size, D) after applying the unitary operator U.

    """
    state = torch.matmul(U, state.unsqueeze(-1)).squeeze(-1)  # Shape: (batch_size, D)
    return state


def state_liouville(U, state):
    """
    Apply the Liouville operator U to the state (density matrix).

    Parameters:
    U (torch.Tensor): A tensor of shape (batch_size, D, D) containing the Liouville operator.
    state (torch.Tensor): A tensor of shape (batch_size, D, D) containing density matrices.

    Returns:
    torch.Tensor: The state vector of shape (batch_size, D, D) after applying the Liouville operator U.
    """

    state = U @ state @ U.conj().transpose(-2, -1)  # Shape: (batch_size, D, D)
    return state


def lindblad_dissipator(rho, collapse_ops):
    """
    Compute the Lindblad dissipator: sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho}).

    Parameters:
    rho (torch.Tensor): Density matrices of shape (batch_size, D, D).
    collapse_ops (torch.Tensor): Collapse operators of shape (n_ops, D, D).

    Returns:
    torch.Tensor: The dissipator contribution of shape (batch_size, D, D).
    """
    # collapse_ops: (n_ops, D, D) -> unsqueeze for batch: (n_ops, 1, D, D)
    L = collapse_ops.unsqueeze(1)  # (n_ops, 1, D, D)
    L_dag = L.conj().transpose(-2, -1)  # (n_ops, 1, D, D)
    L_dag_L = L_dag @ L  # (n_ops, 1, D, D)

    # rho: (batch_size, D, D) -> unsqueeze for ops: (1, batch_size, D, D)
    rho_expanded = rho.unsqueeze(0)  # (1, batch_size, D, D)

    # L rho L^dag: (n_ops, batch_size, D, D)
    term1 = L @ rho_expanded @ L_dag

    # 0.5 * {L^dag L, rho} = 0.5 * (L^dag L rho + rho L^dag L)
    term2 = 0.5 * (L_dag_L @ rho_expanded + rho_expanded @ L_dag_L)

    # Sum over all collapse operators: (batch_size, D, D)
    return (term1 - term2).sum(dim=0)


def state_lindblad(U, state, dt, collapse_ops):
    """
    Apply the Lindblad evolution: unitary step followed by dissipative step.

    Uses Euler splitting: rho(t+dt) = U rho U^dag + dt * L[U rho U^dag]

    Parameters:
    U (torch.Tensor): Unitary operators of shape (batch_size, D, D).
    state (torch.Tensor): Density matrices of shape (batch_size, D, D).
    dt (float or torch.Tensor): Time step.
    collapse_ops (torch.Tensor): Collapse operators of shape (n_ops, D, D).

    Returns:
    torch.Tensor: Updated density matrices of shape (batch_size, D, D).
    """
    # Unitary evolution step
    rho = U @ state @ U.conj().transpose(-2, -1)

    # Dissipative step (Euler)
    rho = rho + dt * lindblad_dissipator(rho, collapse_ops)

    return rho


def matrix_square_root(mat):
    """
    Compute the matrix square root of a batch of matrices using eigen decomposition.

    Parameters:
    mat (torch.Tensor): A tensor of shape (batch_size, D, D)

    Returns:
    torch.Tensor: A tensor of shape (batch_size, D, D) containing the matrix square roots.
    """
    eigvals, eigvecs = torch.linalg.eig(mat)
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_mat = eigvecs @ torch.diag_embed(sqrt_eigvals) @ torch.linalg.inv(eigvecs)
    return sqrt_mat


def modulate_waveforms(cx, cy, me):
    """
    Modulate the waveforms for each qubit.

    Args:
        cx (torch.Tensor): Real tensor of shape (n_pulse, N)
        cy (torch.Tensor): Real tensor of shape (n_pulse, N)
        me (torch.Tensor): Complex tensor of shape (n_pulse, N)

    Returns:
        modulated_cx (torch.Tensor): Real tensor of shape (n_pulse, N)
        modulated_cy (torch.Tensor): Real tensor of shape (n_pulse, N)
    """

    c = torch.complex(cx, cy)  # Shape: (n_pulse, N)
    modulated_waveform = c * me  # Complex multiplication
    modulated_cx = modulated_waveform.real
    modulated_cy = modulated_waveform.imag

    return modulated_cx, modulated_cy
