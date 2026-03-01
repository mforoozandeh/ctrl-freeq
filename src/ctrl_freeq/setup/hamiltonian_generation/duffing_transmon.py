r"""Three-level (Duffing oscillator) transmon model with leakage detection.

This model represents each transmon as a 3-level system (|0>, |1>, |2>)
in the per-qubit rotating frame with the RWA applied, enabling the detection
of leakage into the non-computational |2> state.

The Hilbert space dimension is 3^n_qubits.
"""

from __future__ import annotations

import numpy as np
import torch

from ctrl_freeq.setup.hamiltonian_generation.base import (
    HamiltonianModel,
    register_hamiltonian,
)


def _annihilation_3() -> np.ndarray:
    """3-level annihilation operator: a|n> = sqrt(n)|n-1>."""
    return np.array([[0, 1, 0], [0, 0, np.sqrt(2)], [0, 0, 0]], dtype=complex)


@register_hamiltonian("duffing_transmon")
class DuffingTransmonModel(HamiltonianModel):
    r"""Three-level transmon (Duffing oscillator) Hamiltonian.

    Each transmon is modelled as a 3-level system with states
    :math:`|0\rangle, |1\rangle, |2\rangle`, enabling the detection of
    population leakage outside the computational subspace
    :math:`\{|0\rangle, |1\rangle\}`.

    Reference frame & approximations
    ---------------------------------
    Same as :class:`SuperconductingQubitModel`: per-qubit rotating frame with
    RWA.  The ``frequency_instances`` values are rotating-frame detunings
    :math:`\delta_i` in rad/s.

    Drift Hamiltonian
    -----------------
    .. math::

        H_{\text{drift}} = \sum_i \bigl[
            \delta_i\,\hat{n}_i
            + \tfrac{\alpha_i}{2}\,\hat{n}_i(\hat{n}_i - I)\bigr]
            + \sum_{i<j} g_{ij}\bigl(a_i^\dagger a_j + a_i a_j^\dagger\bigr)

    where :math:`\hat{n} = a^\dagger a`, :math:`\alpha_i` is the per-qubit
    anharmonicity (negative for transmons), and :math:`g_{ij}` is the
    exchange coupling strength.

    Control Hamiltonian
    -------------------
    .. math::

        H_p(t) = \sum_i \Omega_{d,i}\bigl[
            I_i(t)\,(a_i + a_i^\dagger)/2
            + Q_i(t)\,(-\mathrm{i}\,a_i + \mathrm{i}\,a_i^\dagger)/2\bigr]

    Anharmonicities are **required** — the whole point of this model is to
    capture the non-equidistant level spacing that gives rise to leakage.
    """

    def __init__(
        self,
        n_qubits: int,
        coupling_type: str = "XY",
        anharmonicities: np.ndarray | list | None = None,
    ):
        """
        Args:
            n_qubits: number of transmon qubits.
            coupling_type: coupling type (only ``"XY"`` is currently supported
                for the 3-level model).
            anharmonicities: per-qubit anharmonicities α_i (rad/s).  **Required.**
                Raises ``ValueError`` if ``None``.

        Raises:
            ValueError: If ``anharmonicities`` is not provided.
        """
        if anharmonicities is None:
            raise ValueError(
                "DuffingTransmonModel requires anharmonicities — "
                "the 3-level structure is meaningless without them."
            )
        self.n_qubits = n_qubits
        self.coupling_type = coupling_type
        self.anharmonicities = np.asarray(anharmonicities, dtype=float)
        self._build_operators()

    # ------------------------------------------------------------------
    # Operator construction
    # ------------------------------------------------------------------

    def _build_operators(self):
        """Build 3-level operators in the full Hilbert space."""
        a_local = _annihilation_3()  # (3, 3)
        adag_local = a_local.conj().T
        n_local = adag_local @ a_local  # diag(0, 1, 2)

        # Drive operators (local, before embedding)
        x_local = (a_local + adag_local) / 2  # "X-like"
        y_local = (-1j * a_local + 1j * adag_local) / 2  # "Y-like"

        self._a = []
        self._adag = []
        self._n_op = []
        self._X = []
        self._Y = []

        for i in range(self.n_qubits):
            self._a.append(self._embed_operator(a_local, i, self.n_qubits))
            self._adag.append(self._embed_operator(adag_local, i, self.n_qubits))
            self._n_op.append(self._embed_operator(n_local, i, self.n_qubits))
            self._X.append(self._embed_operator(x_local, i, self.n_qubits))
            self._Y.append(self._embed_operator(y_local, i, self.n_qubits))

    # ------------------------------------------------------------------
    # HamiltonianModel interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return 3**self.n_qubits

    @property
    def n_controls(self) -> int:
        return 2 * self.n_qubits  # I, Q per qubit

    def build_drift(
        self,
        frequency_instances: list[np.ndarray],
        coupling_instances: list[np.ndarray] | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        r"""Build drift Hamiltonian snapshots for the 3-level model.

        Args:
            frequency_instances: list of detuning arrays ``(n_qubits,)`` in
                rad/s, one per snapshot.
            coupling_instances: list of exchange-coupling matrices
                ``(n_qubits, n_qubits)`` in rad/s, one per snapshot.
            **kwargs: Reserved for future extensions.

        Returns:
            List of ``(D, D)`` complex numpy arrays where ``D = 3^n_qubits``.
        """
        D = self.dim
        H0_list = []

        for idx, omega in enumerate(frequency_instances):
            H = np.zeros((D, D), dtype=complex)

            # Single-qubit terms: delta_i * n_i + (alpha_i/2) * n_i*(n_i - I)
            for i in range(self.n_qubits):
                H += omega[i] * self._n_op[i]
                H += (self.anharmonicities[i] / 2) * (
                    self._n_op[i] @ self._n_op[i] - self._n_op[i]
                )

            # Coupling: g_{ij} (a†_i a_j + a_i a†_j)
            if coupling_instances is not None and self.n_qubits > 1:
                g = (
                    coupling_instances[idx]
                    if idx < len(coupling_instances)
                    else coupling_instances[-1]
                )
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        if g[i, j] != 0:
                            H += g[i, j] * (
                                self._adag[i] @ self._a[j] + self._a[i] @ self._adag[j]
                            )

            H0_list.append(H)

        return H0_list

    def build_control_ops(self) -> list[np.ndarray]:
        """Return ``[(a+a†)/2, (-ia+ia†)/2]`` per qubit, embedded in full space."""
        ops = []
        for i in range(self.n_qubits):
            ops.append(self._X[i])
            ops.append(self._Y[i])
        return ops

    def control_amplitudes(
        self,
        cx: torch.Tensor,
        cy: torch.Tensor,
        rabi_freq: torch.Tensor,
        n_h0: int,
    ) -> torch.Tensor:
        r"""Map I/Q waveforms to control amplitudes.

        Identical structure to the 2-level superconducting model:
        ``[I_0*Ω_0, Q_0*Ω_0, I_1*Ω_1, Q_1*Ω_1, ...]``.

        Args:
            cx: ``(n_pulse, n_qubits)`` — in-phase waveform.
            cy: ``(n_pulse, n_qubits)`` — quadrature waveform.
            rabi_freq: ``(n_rabi, n_qubits)`` — drive amplitude snapshots.
            n_h0: number of drift Hamiltonian snapshots.

        Returns:
            ``(n_pulse, n_rabi * n_h0, 2 * n_qubits)``
        """
        n_pulse = cx.shape[0]

        # Interleave: [I_0, Q_0, I_1, Q_1, ...]
        u = torch.stack([cx, cy], dim=-1).reshape(n_pulse, 2 * self.n_qubits)

        # Drive amplitude scaling
        rabi_expanded = rabi_freq.repeat_interleave(2, dim=-1)
        rabi_batch = rabi_expanded.repeat(n_h0, 1)

        u = u.unsqueeze(1) * rabi_batch.unsqueeze(0)

        return u

    # ------------------------------------------------------------------
    # Computational subspace embedding
    # ------------------------------------------------------------------

    def _computational_projector(self) -> np.ndarray:
        r"""Build the projector from 2^n computational space into 3^n space.

        Returns:
            ``P`` of shape ``(3^n, 2^n)`` such that ``P @ |ψ_comp> = |ψ_full>``.
        """
        n = self.n_qubits
        d_comp = 2**n
        d_full = 3**n

        # Embedding: map computational basis index (binary) to 3^n index (ternary)
        # |b_{n-1} ... b_0>_comp  →  |b_{n-1} ... b_0>_full
        # where each binary digit stays the same (0 or 1) in base-3.
        P = np.zeros((d_full, d_comp), dtype=complex)
        for comp_idx in range(d_comp):
            # Convert comp_idx to binary digits, then interpret as base-3 index
            full_idx = 0
            for bit in range(n):
                if comp_idx & (1 << bit):
                    full_idx += 3**bit
            P[full_idx, comp_idx] = 1.0
        return P

    def embed_computational_state(self, state: np.ndarray) -> np.ndarray:
        r"""Embed a 2^n state vector into the 3^n Hilbert space.

        Maps each computational basis state to the corresponding
        state in the 3-level space (|2> subspace gets zero amplitude).

        Args:
            state: ``(2^n,)`` complex state vector.

        Returns:
            ``(3^n,)`` complex state vector.
        """
        P = self._computational_projector()
        return P @ state

    def embed_computational_gate(self, gate: np.ndarray) -> np.ndarray:
        r"""Embed a 2^n × 2^n gate into the 3^n Hilbert space.

        The embedded gate acts as the original gate on the computational
        subspace and as identity on the leakage subspace:

        .. math::

            U_{\text{embed}} = P \, U_{\text{gate}} \, P^\dagger
                + (I - P P^\dagger)

        Args:
            gate: ``(2^n, 2^n)`` unitary matrix.

        Returns:
            ``(3^n, 3^n)`` unitary matrix.
        """
        P = self._computational_projector()
        D = self.dim
        PP_dag = P @ P.conj().T  # projector onto computational subspace
        return P @ gate @ P.conj().T + (np.eye(D, dtype=complex) - PP_dag)

    def leakage(self, state: np.ndarray | torch.Tensor) -> float:
        r"""Compute population outside the computational subspace.

        .. math::

            L = 1 - \sum_{i \in \text{comp}} |c_i|^2

        Args:
            state: ``(D,)`` state vector in the 3^n Hilbert space.

        Returns:
            Leakage fraction in [0, 1].
        """
        P = self._computational_projector()
        if isinstance(state, torch.Tensor):
            P_t = torch.as_tensor(P, dtype=state.dtype, device=state.device)
            proj = P_t.conj().T @ state
            pop_comp = torch.sum(torch.abs(proj) ** 2).item()
        else:
            proj = P.conj().T @ state
            pop_comp = np.sum(np.abs(proj) ** 2)
        return 1.0 - float(pop_comp)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, n_qubits: int, params: dict) -> DuffingTransmonModel:
        """Construct from config dict.

        ``anharmonicities`` are required and specified in Hz in the config
        (converted to rad/s here).
        """
        coupling_type = params.get("coupling_type", "XY")
        anharmonicities = params.get("anharmonicities", None)
        if anharmonicities is not None:
            anharmonicities = 2 * np.pi * np.array(anharmonicities)
        return cls(
            n_qubits,
            coupling_type=coupling_type,
            anharmonicities=anharmonicities,
        )

    @classmethod
    def default_config(cls, n_qubits: int) -> dict:
        """Return a complete runnable 3-level transmon configuration.

        Provides sensible defaults with anharmonicities of −330 MHz
        (typical for fixed-frequency transmons).
        """
        qubits = [f"q{i + 1}" for i in range(n_qubits)]

        omegas = [(i + 1) * 10e6 for i in range(n_qubits)]

        g = [[0.0] * n_qubits for _ in range(n_qubits)]
        for i in range(n_qubits - 1):
            g[i][i + 1] = 1.047e7

        if n_qubits == 1:
            initial_states = [["Z"]]
            target_states = {"Axis": [["-Z"]]}
        else:
            initial_states = [["-Z", "Z"] + ["-Z"] * max(0, n_qubits - 2)]
            target_states = {"Gate": ["iSWAP"]}

        return {
            "hamiltonian_type": "duffing_transmon",
            "qubits": qubits,
            "compute_resource": "cpu",
            "parameters": {
                "Delta": omegas,
                "sigma_Delta": [0.0] * n_qubits,
                "Omega_R_max": [40e6] * n_qubits,
                "sigma_Omega_R_max": [0.0] * n_qubits,
                "pulse_duration": [200e-9] * n_qubits,
                "point_in_pulse": [100] * n_qubits,
                "wf_type": ["cheb"] * n_qubits,
                "wf_mode": ["cart"] * n_qubits,
                "amplitude_envelope": ["gn"] * n_qubits,
                "amplitude_order": [1] * n_qubits,
                "coverage": ["broadband"] * n_qubits,
                "sw": [5e6] * n_qubits,
                "pulse_offset": [0.0] * n_qubits,
                "pulse_bandwidth": [5e5] * n_qubits,
                "ratio_factor": [0.5] * n_qubits,
                "profile_order": [2] * n_qubits,
                "n_para": [16] * n_qubits,
                "J": g,
                "sigma_J": 0.0,
                "coupling_type": "XY",
                "anharmonicities": [-330e6] * n_qubits,
            },
            "initial_states": initial_states,
            "target_states": target_states,
            "optimization": {
                "space": "hilbert",
                "H0_snapshots": 1,
                "Omega_R_snapshots": 1,
                "algorithm": "l-bfgs",
                "max_iter": 300,
                "targ_fid": 0.999,
            },
        }
