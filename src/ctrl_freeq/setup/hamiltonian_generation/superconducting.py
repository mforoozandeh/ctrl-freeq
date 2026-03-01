from __future__ import annotations

import numpy as np
import torch

from ctrl_freeq.setup.hamiltonian_generation.base import (
    HamiltonianModel,
    register_hamiltonian,
)


@register_hamiltonian("superconducting")
class SuperconductingQubitModel(HamiltonianModel):
    """Superconducting transmon qubit Hamiltonian.

    Models fixed-frequency transmon qubits with always-on capacitive coupling,
    truncated to the qubit subspace (two levels per transmon).

    Drift Hamiltonian:
        H_drift = sum_i (omega_i / 2) * Z_i  +  sum_{i<j} g_{ij} (X_i X_j + Y_i Y_j)

    where omega_i are qubit frequencies and g_{ij} are coupling strengths.
    The coupling term is the transverse (exchange) interaction that arises
    from capacitive coupling between transmons in the rotating frame.

    Control Hamiltonian:
        Hp(t) = sum_i [ I_i(t) * X_i  +  Q_i(t) * Y_i ] * Omega_d_i

    where I_i(t) and Q_i(t) are the in-phase and quadrature components of
    the microwave drive on qubit i, and Omega_d_i is the drive amplitude.

    This is structurally similar to the spin-chain model but with different
    physical parameters and conventions:
        - Qubit frequencies (omega) instead of chemical shifts (Delta)
        - Coupling strengths (g) instead of J-couplings
        - Exchange coupling (XX + YY) is the standard form
        - Anharmonicity can introduce Z-Z coupling (cross-resonance)
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
            coupling_type: type of qubit-qubit coupling.
                ``"XY"`` – exchange coupling (default, from capacitive coupling).
                ``"ZZ"`` – static ZZ coupling (from anharmonicity-mediated shifts).
                ``"XY+ZZ"`` – both exchange and ZZ terms.
            anharmonicities: per-qubit anharmonicities (rad/s), used to compute
                the static ZZ shift when ``coupling_type`` includes ``"ZZ"``.
                If ``None``, ZZ coupling must be specified directly via the
                coupling matrix.
        """
        self.n_qubits = n_qubits
        self.coupling_type = coupling_type
        self.anharmonicities = (
            np.asarray(anharmonicities, dtype=float)
            if anharmonicities is not None
            else None
        )
        self._build_operators()

    def _build_operators(self):
        """Build Pauli operators in the full Hilbert space."""
        # Pauli matrices (with 1/2 factor to match spin-1/2 convention)
        X = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
        Y = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)

        self._X = []
        self._Y = []
        self._Z = []

        for i in range(self.n_qubits):
            self._X.append(self._embed_operator(X, i, self.n_qubits))
            self._Y.append(self._embed_operator(Y, i, self.n_qubits))
            self._Z.append(self._embed_operator(Z, i, self.n_qubits))

    @property
    def dim(self) -> int:
        return 2**self.n_qubits

    @property
    def n_controls(self) -> int:
        return 2 * self.n_qubits  # I and Q per qubit

    def build_drift(
        self,
        frequency_instances: list[np.ndarray],
        coupling_instances: list[np.ndarray] | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        """Build drift Hamiltonian snapshots.

        Args:
            frequency_instances: list of qubit-frequency arrays ``(n_qubits,)``,
                one per snapshot.  In rad/s.
            coupling_instances: list of coupling matrices ``(n_qubits, n_qubits)``,
                one per snapshot.  Upper-triangular, in rad/s.
            **kwargs: Optional ``zz_instances`` for explicit ZZ coupling values.

        Returns:
            List of ``(D, D)`` complex numpy arrays.
        """
        zz_instances = kwargs.get("zz_instances", None)

        D = self.dim
        H0_list = []

        for idx, omega in enumerate(frequency_instances):
            H = np.zeros((D, D), dtype=complex)

            # Qubit frequencies: sum_i (omega_i/2) Z_i
            for i in range(self.n_qubits):
                H += omega[i] * self._Z[i]

            # Coupling terms
            if coupling_instances is not None and self.n_qubits > 1:
                g = (
                    coupling_instances[idx]
                    if idx < len(coupling_instances)
                    else coupling_instances[-1]
                )
                use_xy = "XY" in self.coupling_type
                use_zz = "ZZ" in self.coupling_type

                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        if use_xy and g[i, j] != 0:
                            # Exchange coupling: g_{ij} (X_i X_j + Y_i Y_j)
                            H += g[i, j] * (
                                self._X[i] @ self._X[j] + self._Y[i] @ self._Y[j]
                            )

                if use_zz:
                    if zz_instances is not None:
                        zz = (
                            zz_instances[idx]
                            if idx < len(zz_instances)
                            else zz_instances[-1]
                        )
                    elif self.anharmonicities is not None:
                        # Approximate static ZZ from anharmonicity:
                        # zeta_{ij} ~ 2 * g_{ij}^2 * (1/alpha_i + 1/alpha_j)
                        alpha = self.anharmonicities
                        zz = np.zeros((self.n_qubits, self.n_qubits))
                        for i in range(self.n_qubits):
                            for j in range(i + 1, self.n_qubits):
                                if alpha[i] != 0 and alpha[j] != 0:
                                    zz[i, j] = (
                                        2
                                        * g[i, j] ** 2
                                        * (1.0 / alpha[i] + 1.0 / alpha[j])
                                    )
                    else:
                        zz = np.zeros((self.n_qubits, self.n_qubits))

                    for i in range(self.n_qubits):
                        for j in range(i + 1, self.n_qubits):
                            if zz[i, j] != 0:
                                H += zz[i, j] * (self._Z[i] @ self._Z[j])

            H0_list.append(H)

        return H0_list

    def build_control_ops(self) -> list[np.ndarray]:
        """Return ``[X_0, Y_0, X_1, Y_1, ...]`` (I/Q drive per qubit)."""
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
        """Map I/Q waveforms to control amplitudes.

        Same structure as SpinChainModel since both use per-qubit X/Y control:
            cx -> I_i(t) (in-phase)
            cy -> Q_i(t) (quadrature)
            rabi_freq -> drive amplitude Omega_d_i

        Args:
            cx: ``(n_pulse, n_qubits)``
            cy: ``(n_pulse, n_qubits)``
            rabi_freq: ``(n_rabi, n_qubits)``
            n_h0: number of drift Hamiltonian snapshots.

        Returns:
            ``(n_pulse, n_rabi * n_h0, 2 * n_qubits)``
        """
        n_pulse = cx.shape[0]

        # Interleave: [I_0, Q_0, I_1, Q_1, ...]
        u = torch.stack([cx, cy], dim=-1).reshape(n_pulse, 2 * self.n_qubits)

        # Drive amplitude scaling: [Omega_d_0, Omega_d_0, Omega_d_1, ...]
        rabi_expanded = rabi_freq.repeat_interleave(2, dim=-1)
        rabi_batch = rabi_expanded.repeat(n_h0, 1)

        u = u.unsqueeze(1) * rabi_batch.unsqueeze(0)

        return u  # (n_pulse, n_rabi * n_h0, 2 * n_qubits)

    @classmethod
    def from_config(cls, n_qubits: int, params: dict) -> SuperconductingQubitModel:
        """Construct from config dict, extracting superconducting-specific params."""
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
        """Return a complete runnable superconducting qubit configuration.

        Provides sensible transmon defaults: XY coupling, iSWAP gate target
        (the natural entangling gate for XY-coupled transmons), Chebyshev
        waveforms.
        """
        qubits = [f"q{i + 1}" for i in range(n_qubits)]

        # Qubit frequencies: 10, 20, 30, … MHz (in rotating frame)
        omegas = [(i + 1) * 10e6 for i in range(n_qubits)]

        # Capacitive coupling matrix (upper-triangular, nearest-neighbour)
        g = [[0.0] * n_qubits for _ in range(n_qubits)]
        for i in range(n_qubits - 1):
            g[i][i + 1] = 1.047e7

        # Target gate: iSWAP for multi-qubit, axis flip for single qubit
        if n_qubits == 1:
            initial_states = [["Z"]]
            target_states = {"Axis": [["-Z"]]}
        else:
            initial_states = [["-Z", "Z"] + ["-Z"] * max(0, n_qubits - 2)]
            target_states = {"Gate": ["iSWAP"]}

        return {
            "hamiltonian_type": "superconducting",
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
