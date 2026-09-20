import math

import torch

from ctrl_freeq.conditions.stopping_conds import OptimizationInterrupted

from ctrl_freeq.utils.colored_logging import setup_colored_logging


def _as_float(value):
    """Return *value* as a plain Python float, detaching tensors if needed."""
    return value.item() if hasattr(value, "item") else float(value)


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
        # Physical fidelity, amplitude penalty and the penalized score
        # (fidelity - penalty) are tracked as three separate series.
        self.fidelity_history = []
        self.penalty_history = []
        self.score_history = []

        # Flag for COBYLA early termination (avoids exception-based termination)
        self.early_termination_flag = False
        self.early_termination_solution = None

        # Initialize logger for optimization progress
        self.logger = setup_colored_logging(level="INFO")

    def objective_function(self, para):
        parameters = torch.split(para, list(self.n_para))

        amps, cxs, cys = pulse_para(
            self.n_qubits, parameters, self.mat, self.wf_fun, self.me
        )
        self.pen = penalty(amps)
        # Kept for the amplitude-limit report; the modulation carrier has unit
        # magnitude, so |I+iQ| is the command amplitude either way.
        self.last_cx = cxs.detach()
        self.last_cy = cys.detach()

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

    def _record_iteration(self):
        """Record physical fidelity, penalty and penalized score separately.

        ``cost = -fidelity + penalty``, so ``-cost`` is the *penalized score*,
        not the fidelity.  Reporting it under the name "fidelity" understates
        the physical fidelity by exactly the penalty.

        Returns:
            tuple: ``(fidelity, penalty, score)`` as plain floats.
        """
        if self.iter == 0:
            self.logger.info("=" * 46)
            self.logger.info(
                f"{'Iteration':<10} | {'Fidelity':<10} | {'Penalty':<10} | "
                f"{'Score':<10}"
            )
            self.logger.info("=" * 46)

        self.iter += 1
        fidelity = _as_float(self.fid)
        penalty_value = _as_float(self.pen)
        score = fidelity - penalty_value

        self.fidelity_history.append(fidelity)
        self.penalty_history.append(penalty_value)
        self.score_history.append(score)

        self.logger.info(
            f"{self.iter:<10} | {fidelity:<10.4f} | {penalty_value:<10.4f} | "
            f"{score:<10.4f}"
        )
        return fidelity, penalty_value, score

    def callback_function(self, para):
        _fidelity, _penalty, score = self._record_iteration()

        # The stopping criterion is deliberately the *penalized* score, so a
        # solution that only reaches the target by exceeding the amplitude
        # limit does not stop the optimisation early.
        if score >= self.exit_val:
            raise OptimizationInterrupted(
                f"Penalized score reached target = {self.exit_val}, exiting optimization...",
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

        _fidelity, _penalty, score = self._record_iteration()

        if score >= self.exit_val:
            self.early_termination_flag = True
            self.early_termination_solution = para
            self.logger.warning(
                f"Penalized score reached target = {self.exit_val}, exiting optimization..."
            )


def pulse_para(n_qubits, parameters, mat, wf_fun, me):
    amps_list = []
    cxs_list = []
    cys_list = []

    for i in range(n_qubits):
        amp, _phi, cx, cy = wf_fun[i](parameters[i], mat[i])
        amps_list.append(amp)  # Each amp has shape [100, 1]
        cxs_list.append(cx)
        cys_list.append(cy)

    # Concatenate along dim=1 to get shape [100, n_qubits]
    amps = torch.cat(amps_list, dim=1)
    cxs = torch.cat(cxs_list, dim=1)
    cys = torch.cat(cys_list, dim=1)

    cxs, cys = modulate_waveforms(cxs, cys, me)

    return amps, cxs, cys


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

    # Strang splitting: (C_half U C_half)^N = C_half U (C_full U)^(N-1) C_half.
    # The interior half-channels merge into full-step channels, so N unitary
    # steps need only N+1 channel applications.  Both channels are built once.
    half_channel = None
    full_channel = None
    if use_lindblad:
        half_channel = dissipative_channel(collapse_ops, dt / 2)
        full_channel = dissipative_channel(collapse_ops, dt)
        state = apply_channel(half_channel, state)

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
                U = U_chunk[n]
                state = U @ state @ U.conj().transpose(-2, -1)
                if start + n < n_pulse - 1:
                    state = apply_channel(full_channel, state)
        else:
            for n in range(chunk_len):
                state = state_fun(U_chunk[n], state)

    if use_lindblad:
        state = apply_channel(half_channel, state)

    return state


def simulate_trajectory(H0, Hp, dt, initial_state, u_fun, state_fun, collapse_ops=None):
    """Propagate and record every state boundary, including the initial state.

    Uses exactly the same propagator, control Hamiltonian and dissipative
    channel as :func:`simulator_optimized`, so an analysis replay reproduces
    the optimizer's physics rather than a second, divergent implementation.
    The interior dissipative half-channels are *not* merged here: each of the
    ``n_pulse + 1`` recorded states is the physical state at time ``k * dt``.

    Parameters:
    - H0: ``(batch, D, D)`` drift Hamiltonians.
    - Hp: ``(n_pulse, batch, D, D)`` pulse Hamiltonians.
    - dt: time step ``T / n_pulse``.
    - initial_state: ``(batch, D)`` or ``(batch, D, D)``.
    - collapse_ops: optional ``(n_ops, D, D)`` collapse operators.

    Returns:
        torch.Tensor of shape ``(n_pulse + 1, batch, ...)``; entry 0 is the
        initial state and entry ``n_pulse`` the state at exactly ``T``.
    """
    n_pulse = Hp.shape[0]
    half_channel = (
        dissipative_channel(collapse_ops, dt / 2) if collapse_ops is not None else None
    )

    state = initial_state
    states = [state]
    for n in range(n_pulse):
        U = u_fun(H0 + Hp[n], dt)
        if half_channel is not None:
            state = apply_channel(half_channel, state)
            state = U @ state @ U.conj().transpose(-2, -1)
            state = apply_channel(half_channel, state)
        else:
            state = state_fun(U, state)
        states.append(state)

    return torch.stack(states)


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


def _assert_pure_targets(sigma, atol=1e-6):
    """Raise unless every target density matrix is pure with unit trace."""
    trace = torch.diagonal(sigma, dim1=-2, dim2=-1).sum(-1)
    purity = torch.einsum("bij,bji->b", sigma, sigma)
    if not torch.allclose(
        trace.real, torch.ones_like(trace.real), atol=atol
    ) or not torch.allclose(purity.real, torch.ones_like(purity.real), atol=atol):
        raise ValueError(
            "fidelity_liouville implements the Uhlmann fidelity against a "
            "*pure* target, for which it reduces to Re Tr(rho sigma). The "
            "supplied targets are not pure unit-trace density matrices "
            f"(max |Tr - 1| = {(trace.real - 1).abs().max():.3e}, "
            f"max |Tr(sigma^2) - 1| = {(purity.real - 1).abs().max():.3e}). "
            "A mixed-target Uhlmann fidelity needs an explicit API decision; "
            "gate objectives use the channel metric in fidelity_gate_liouville."
        )


def fidelity_liouville(rho, sigma):
    r"""Uhlmann-Jozsa fidelity against **pure** target density matrices.

    For a pure target :math:`\sigma = |\psi\rangle\langle\psi|` the general
    expression :math:`\bigl(\mathrm{Tr}\sqrt{\sqrt\rho\,\sigma\sqrt\rho}\bigr)^2`
    reduces exactly to :math:`\langle\psi|\rho|\psi\rangle =
    \mathrm{Re}\,\mathrm{Tr}(\rho\sigma)`.  Evaluating the closed form
    directly avoids the eigendecomposition of a rank-deficient intermediate,
    whose ``torch.linalg.eig`` backward pass fails with a complex
    eigenvector-phase error — the objective is smooth even where that
    intermediate is not.

    Parameters:
    rho (torch.Tensor): A tensor of shape (batch_size, D, D)
    sigma (torch.Tensor): Pure target density matrices, shape (batch_size, D, D)

    Returns:
    torch.Tensor: A tensor containing the mean fidelity over the batch.
    """
    _assert_pure_targets(sigma)
    fidelity = torch.einsum("bij,bji->b", rho, sigma).real
    return torch.mean(fidelity)


def fidelity_gate_hilbert(states, targets, n_rows, d, projector=None):
    r"""Average gate fidelity over the whole computational subspace.

    The computational basis is propagated as *columns*, so relative phases
    are preserved.  With :math:`V` the isometry embedding the ``d`` computational
    states in the model Hilbert space, :math:`U` the full-space evolution and
    :math:`G` the target gate, :math:`M = G^\dagger V^\dagger U V` and

    .. math::

        F_{\text{avg}} = \frac{\mathrm{Tr}(M^\dagger M) + |\mathrm{Tr}\,M|^2}
                              {d\,(d+1)}

    Population that leaks out of the computational subspace is *not*
    renormalised away: it simply reduces :math:`\mathrm{Tr}(M^\dagger M)`.

    Args:
        states: ``(n_rows * n_batch, D)`` propagated basis columns, ordered
            row-major (``row * n_batch + batch``).
        targets: ``(n_rows * n_batch, D)`` embedded ``G|r>`` targets.
        n_rows: number of computational basis columns (must equal ``d``).
        d: computational dimension ``2**n_qubits``.
        projector: optional ``(D, d)`` isometry ``V``; ``None`` when ``D == d``.

    Returns:
        torch.Tensor: mean average gate fidelity over the ensemble.
    """
    if n_rows != d:
        raise ValueError(
            f"Average gate fidelity needs all {d} computational basis columns, "
            f"got {n_rows} rows."
        )
    total = states.shape[0]
    if total % n_rows:
        raise ValueError(
            f"State batch of {total} is not divisible by {n_rows} rows; the "
            f"objective arrays are misaligned."
        )
    n_batch = total // n_rows
    psi = states.reshape(n_rows, n_batch, -1)
    tgt = targets.reshape(n_rows, n_batch, -1)

    if projector is None:
        comp = psi
    else:
        comp = torch.einsum("rbi,ia->rba", psi, projector.conj())

    tr_MdagM = (comp.conj() * comp).real.sum(-1).sum(0)  # (n_batch,)
    tr_M = (tgt.conj() * psi).sum(-1).sum(0)  # (n_batch,)

    fidelity = (tr_MdagM + tr_M.abs() ** 2) / (d * (d + 1))
    return torch.mean(fidelity)


def fidelity_gate_liouville(states, targets, n_rows, d):
    r"""Average gate fidelity of a channel, from Pauli-string overlaps.

    With unnormalised Pauli strings :math:`P_j` (``Tr(P_j P_k) = d delta_jk``,
    :math:`P_0 = I`) and :math:`x_j = \mathrm{Tr}[(G P_j G^\dagger)\,
    \mathcal{E}_{\text{comp}}(P_j)]`,

    .. math::

        F_{\text{avg}} = \frac{d\,x_0 + \sum_j x_j}{d^2 (d+1)}

    The :math:`d\,x_0` term replaces the usual constant :math:`d^2` and so
    also counts population lost from the computational subspace.  State
    fidelity on a handful of input states cannot detect that, and Pauli
    strings are not density matrices: they must never be passed to an
    Uhlmann fidelity.

    Args:
        states: ``(n_rows * n_batch, D, D)`` propagated embedded Pauli strings,
            ordered row-major, with the identity string first.
        targets: ``(n_rows * n_batch, D, D)`` embedded ``G P_j G^dag``.
        n_rows: number of Pauli strings (must equal ``d**2``).
        d: computational dimension ``2**n_qubits``.

    Returns:
        torch.Tensor: mean average gate fidelity over the ensemble.
    """
    if n_rows != d * d:
        raise ValueError(
            f"Channel gate fidelity needs all {d * d} Pauli strings, got {n_rows} rows."
        )
    total = states.shape[0]
    if total % n_rows:
        raise ValueError(
            f"State batch of {total} is not divisible by {n_rows} rows; the "
            f"objective arrays are misaligned."
        )
    n_batch = total // n_rows
    D = states.shape[-1]
    fin = states.reshape(n_rows, n_batch, D, D)
    tgt = targets.reshape(n_rows, n_batch, D, D)

    x = torch.einsum("rbij,rbji->rb", tgt, fin).real  # (n_rows, n_batch)
    fidelity = (d * x[0] + x.sum(0)) / (d * d * (d + 1))
    return torch.mean(fidelity)


def amplitude_limit_report(cx, cy, omega_r_max, rabi_samples=None):
    r"""Per-qubit report of the sampled peak drive amplitude against its limit.

    The amplitude limit in this optimiser is a **soft penalty** (see
    :func:`penalty`): exceeding it costs objective value but nothing prevents
    the returned waveform from doing so.  This report makes the actual
    violation explicit instead of leaving it folded into the objective.

    The peak is taken as :math:`\max_t \sqrt{I(t)^2 + Q(t)^2}`, which is the
    magnitude of the command amplitude.  Taking a signed polar amplitude at
    face value would understate the peak of a waveform whose amplitude
    coefficient goes negative.

    The peak is over the **sampled** waveform.  It is not automatically a
    bound on an independently interpolated continuous waveform, which can
    overshoot between samples.

    Args:
        cx: ``(n_pulse, n_qubits)`` in-phase command amplitudes.
        cy: ``(n_pulse, n_qubits)`` quadrature command amplitudes.
        omega_r_max: per-qubit configured maximum Rabi rate (rad/s).
        rabi_samples: optional ``(n_rabi, n_qubits)`` sampled Rabi gains, used
            to report the uncertain *physical* peak separately from the
            nominal command amplitude.

    Returns:
        list[dict]: one entry per qubit with ``normalized_peak`` (1.0 is the
        limit), ``excess`` (amount above 1, else 0), ``within_limit``,
        ``nominal_peak_rad_per_s`` and, when sampled gains are supplied,
        ``sampled_peak_rad_per_s`` as a ``(min, max)`` pair.
    """
    magnitude = torch.sqrt(cx**2 + cy**2)
    peaks = magnitude.max(dim=0).values.detach().cpu()
    omega = torch.as_tensor(omega_r_max, dtype=peaks.dtype).reshape(-1)

    report = []
    for i in range(peaks.shape[0]):
        normalized = float(peaks[i])
        entry = {
            "qubit": i,
            "normalized_peak": normalized,
            "excess": max(0.0, normalized - 1.0),
            "within_limit": normalized <= 1.0,
            "omega_r_max": float(omega[i]),
            "nominal_peak_rad_per_s": normalized * float(omega[i]),
        }

        if rabi_samples is not None:
            # A negative sampled gain is a phase reversal, not a negative drive
            # magnitude, so the physical peak range comes from |gain|.  Signed
            # extrema would report (-2, 1) for gains [-2, 1] and understate the
            # largest drive the pulse actually reaches.
            gains = (
                torch.as_tensor(rabi_samples, dtype=peaks.dtype)
                .reshape(-1, peaks.shape[0])[:, i]
                .abs()
            )
            entry["sampled_peak_rad_per_s"] = (
                normalized * float(gains.min()),
                normalized * float(gains.max()),
            )
        report.append(entry)
    return report


def format_amplitude_limit_report(report):
    """Render :func:`amplitude_limit_report` as human-readable lines."""
    lines = [
        "Amplitude limit is a SOFT penalty: the returned waveform is not "
        "clipped or constrained to it.",
        "Peaks are over the sampled waveform and do not bound an "
        "independently interpolated continuous waveform.",
    ]
    for entry in report:
        status = "OK" if entry["within_limit"] else "EXCEEDS LIMIT"
        line = (
            f"  qubit {entry['qubit']}: peak |I+iQ| = "
            f"{entry['normalized_peak']:.4f} x Omega_R_max "
            f"({entry['nominal_peak_rad_per_s']:.4g} rad/s) "
            f"[{status}"
        )
        if not entry["within_limit"]:
            line += f", {entry['excess']:.4f} above the limit"
        line += "]"
        if "sampled_peak_rad_per_s" in entry:
            low, high = entry["sampled_peak_rad_per_s"]
            line += (
                f" | physical peak over sampled Rabi gains: {low:.4g}-{high:.4g} rad/s"
            )
        lines.append(line)
    return "\n".join(lines)


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

    Uses the Rodrigues-like formula:
        exp(-i H dt) = cos(|h| dt) I  -  i sin(|h| dt) / |h| * H

    where |h| is the magnitude of the Bloch vector.  The sinc-like
    ratio sin(x)/x is evaluated via ``torch.sinc`` (which computes
    sin(pi x)/(pi x)), so the division is safe when |h| -> 0 and
    the result correctly reduces to the identity matrix.

    Parameters:
    H (torch.Tensor): A tensor of shape (batch, 2, 2) containing Hermitian matrices.
    dt (float): Time step.

    Returns:
    torch.Tensor: The matrix exponential of H, shape (batch, 2, 2).
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

    # sin(|h| dt) / |h|  =  dt * sin(|h| dt) / (|h| dt)
    #                     =  dt * sinc(|h| dt / pi)
    # torch.sinc(x) computes sin(pi x) / (pi x), so we pass |h| dt / pi.
    sinc_term = dt * torch.sinc(h_magnitude * dt / torch.pi)
    sin_term = -1j * sinc_term.unsqueeze(-1).unsqueeze(-1) * H

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


def precompute_collapse_products(collapse_ops):
    """Precompute L, L†, L†L for Lindblad dissipation (call once, not per step).

    Parameters:
    collapse_ops (torch.Tensor): Collapse operators of shape (n_ops, D, D).

    Returns:
    tuple: (L, L_dag, L_dag_L) each of shape (n_ops, 1, D, D), ready for
        batched broadcasting in :func:`lindblad_dissipator`.
    """
    L = collapse_ops.unsqueeze(1)  # (n_ops, 1, D, D)
    L_dag = L.conj().transpose(-2, -1)  # (n_ops, 1, D, D)
    L_dag_L = L_dag @ L  # (n_ops, 1, D, D)
    return L, L_dag, L_dag_L


def lindblad_dissipator(rho, collapse_ops, _precomputed=None):
    """
    Compute the Lindblad dissipator: sum_k (L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho}).

    Parameters:
    rho (torch.Tensor): Density matrices of shape (batch_size, D, D).
    collapse_ops (torch.Tensor): Collapse operators of shape (n_ops, D, D).
    _precomputed (tuple, optional): Pre-computed (L, L_dag, L_dag_L) from
        :func:`precompute_collapse_products`.  When provided, ``collapse_ops``
        is ignored and the pre-computed values are used directly, avoiding
        redundant matmuls in the inner time-step loop.

    Returns:
    torch.Tensor: The dissipator contribution of shape (batch_size, D, D).
    """
    if _precomputed is not None:
        L, L_dag, L_dag_L = _precomputed
    else:
        L, L_dag, L_dag_L = precompute_collapse_products(collapse_ops)

    # rho: (batch_size, D, D) -> unsqueeze for ops: (1, batch_size, D, D)
    rho_expanded = rho.unsqueeze(0)  # (1, batch_size, D, D)

    # L rho L^dag: (n_ops, batch_size, D, D)
    term1 = L @ rho_expanded @ L_dag

    # 0.5 * {L^dag L, rho} = 0.5 * (L^dag L rho + rho L^dag L)
    term2 = 0.5 * (L_dag_L @ rho_expanded + rho_expanded @ L_dag_L)

    # Sum over all collapse operators: (batch_size, D, D)
    return (term1 - term2).sum(dim=0)


def _validate_step(dt, name="dt"):
    """Reject non-finite or negative channel durations.

    Only inspects *dt*; callers must keep using the original object so that a
    duration carrying ``requires_grad`` stays connected to the graph.  A zero
    step is allowed (it yields the identity channel); pulse durations
    themselves remain strictly positive and are validated by the setup code.
    """
    value = dt.item() if isinstance(dt, torch.Tensor) else float(dt)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")


def dissipator_superoperator(collapse_ops):
    r"""Build the Lindblad dissipator as a superoperator matrix.

    Returns ``S`` of shape ``(D**2, D**2)`` acting on the **row-major**
    vectorisation of the density matrix, i.e. ``vec(D[rho]) = S vec(rho)``
    with ``vec`` = ``rho.reshape(D*D)``.  With that convention
    ``vec(A rho B) = (A kron B^T) vec(rho)``, so

    .. math::

        S = \sum_k L_k \otimes L_k^{*}
            - \tfrac12 (L_k^\dagger L_k) \otimes I
            - \tfrac12 I \otimes (L_k^\dagger L_k)^{T}

    Args:
        collapse_ops (torch.Tensor): ``(n_ops, D, D)`` collapse operators.

    Returns:
        torch.Tensor: ``(D**2, D**2)`` superoperator matrix.
    """
    n_ops, D, _ = collapse_ops.shape
    eye = torch.eye(D, dtype=collapse_ops.dtype, device=collapse_ops.device)
    S = torch.zeros(D * D, D * D, dtype=collapse_ops.dtype, device=collapse_ops.device)
    for k in range(n_ops):
        L = collapse_ops[k]
        L_dag_L = L.conj().transpose(-2, -1) @ L
        S = (
            S
            + torch.kron(L, L.conj())
            - 0.5 * torch.kron(L_dag_L, eye)
            - 0.5 * torch.kron(eye, L_dag_L.transpose(-2, -1).contiguous())
        )
    return S


def dissipative_channel(collapse_ops, dt):
    r"""Return the exact dissipative channel ``exp(dt * D)``.

    Unlike an Euler step ``rho + dt * D[rho]``, the matrix exponential of the
    dissipator is completely positive and trace preserving for *every* step
    size.  The Euler step is only an approximation to it and produces
    unphysical states (negative populations, trace > 1) as soon as
    ``dt`` is comparable to the relaxation times.


    Args:
        collapse_ops (torch.Tensor): ``(n_ops, D, D)`` collapse operators.
        dt (float or torch.Tensor): channel duration, finite and non-negative.
            A tensor duration is used as-is, so gradients with respect to the
            pulse duration propagate through the channel; substituting the
            validated Python scalar would silently detach them.

    Returns:
        torch.Tensor: ``(D**2, D**2)`` channel in the row-major vec convention.
    """
    _validate_step(dt)
    return torch.linalg.matrix_exp(dissipator_superoperator(collapse_ops) * dt)


def apply_channel(channel, rho):
    """Apply a row-major vec superoperator to a batch of density matrices.

    Args:
        channel (torch.Tensor): ``(D**2, D**2)`` superoperator.
        rho (torch.Tensor): ``(batch, D, D)`` density matrices.

    Returns:
        torch.Tensor: ``(batch, D, D)`` transformed density matrices.
    """
    batch, D, _ = rho.shape
    flat = rho.reshape(batch, D * D)
    return (flat @ channel.transpose(-2, -1)).reshape(batch, D, D)


def state_lindblad(U, state, dt, collapse_ops, _precomputed=None):
    """
    Apply one Strang-split Lindblad step: half-channel, unitary, half-channel.

    The dissipative half-steps use the exact channel ``exp((dt/2) * D)``, so
    the result is completely positive and trace preserving, and the splitting
    is second-order accurate in ``dt``.

    Parameters:
    U (torch.Tensor): Unitary operators of shape (batch_size, D, D).
    state (torch.Tensor): Density matrices of shape (batch_size, D, D).
    dt (float or torch.Tensor): Time step.
    collapse_ops (torch.Tensor): Collapse operators of shape (n_ops, D, D).
    _precomputed (torch.Tensor, optional): Pre-built half-step channel from
        :func:`dissipative_channel`, reused to avoid rebuilding it per step.

    Returns:
    torch.Tensor: Updated density matrices of shape (batch_size, D, D).
    """
    half = (
        _precomputed
        if _precomputed is not None
        else dissipative_channel(collapse_ops, dt / 2)
    )
    rho = apply_channel(half, state)
    rho = U @ rho @ U.conj().transpose(-2, -1)
    return apply_channel(half, rho)


def matrix_square_root(mat):
    """
    Compute the matrix square root of a batch of matrices via eigendecomposition.

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
