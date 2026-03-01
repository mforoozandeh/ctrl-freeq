from __future__ import annotations

import numpy as np
import torch

from ctrl_freeq.setup.hamiltonian_generation.base import (
    HamiltonianModel,
    register_hamiltonian,
)


@register_hamiltonian("superconducting")
class SuperconductingQubitModel(HamiltonianModel):
    r"""Two-level transmon qubit Hamiltonian in the per-qubit rotating frame.

    Models fixed-frequency transmon qubits with always-on capacitive coupling,
    truncated to the computational subspace (two lowest levels per transmon).
    For a 3-level (Duffing oscillator) model that captures leakage to the
    :math:`|2\rangle` state, see :class:`DuffingTransmonModel`.

    Reference frame & approximations
    ---------------------------------
    The model operates in a **per-qubit rotating frame** with the rotating-wave
    approximation (RWA) applied.  All fast-oscillating carrier terms at the
    drive frequencies are absorbed, so the ``frequency_instances`` values
    represent **detunings** :math:`\delta_i = \omega_{\text{qubit}} -
    \omega_{\text{frame}}`, *not* bare lab-frame transition frequencies.

    Sign / scaling conventions
    --------------------------
    * All frequencies (:math:`\delta`, *g*, :math:`\zeta`, :math:`\Omega_d`)
      are in **rad/s**.
    * Pauli matrices carry the spin-½ factor:
      :math:`X_i = \tfrac{1}{2}\sigma_x^{(i)}`,
      :math:`Y_i = \tfrac{1}{2}\sigma_y^{(i)}`,
      :math:`Z_i = \tfrac{1}{2}\sigma_z^{(i)}`.
      Therefore ``omega * Z[i]`` contributes :math:`(\omega/2)\,\sigma_z`.

    Drift Hamiltonian
    -----------------
    .. math::

        H_{\text{drift}} = \sum_i \frac{\delta_i}{2}\,\sigma_z^{(i)}
            \;+\; \sum_{i<j} g_{ij}\bigl(X_i X_j + Y_i Y_j\bigr)
            \;+\; \sum_{i<j} \zeta_{ij}\,Z_i Z_j

    * :math:`\delta_i` — qubit detuning in the rotating frame (rad/s).
    * :math:`g_{ij}` — exchange (XY) coupling from the capacitive interaction
      (``coupling_instances``).  This drives energy-conserving swap transitions.
    * :math:`\zeta_{ij}` — residual static ZZ rate (cross-Kerr).  Physically
      distinct from *g*: it is a perturbative diagonal shift, not an exchange
      term.  Can be supplied as a calibrated matrix (``zz_crosstalk``), per-
      snapshot values (``zz_instances`` kwarg), or approximated from
      anharmonicities via
      :math:`\zeta_{ij} \approx 2\,g_{ij}^2 (1/\alpha_i + 1/\alpha_j)`.

    Control Hamiltonian
    -------------------
    .. math::

        H_p(t) = \sum_i \Omega_{d,i}\bigl[
            I_i(t)\,X_i + Q_i(t)\,Y_i\bigr]

    where :math:`I_i(t)` and :math:`Q_i(t)` are the in-phase and quadrature
    components of the microwave drive envelope on qubit *i*, and
    :math:`\Omega_{d,i}` is the maximum drive amplitude.

    When ``stark_shift_coeffs`` is provided, an additional AC Stark (light-
    shift) term is included:

    .. math::

        H_{\text{Stark}}(t) = \sum_i \frac{s_i}{2}\,
            \bigl(I_i^2(t) + Q_i^2(t)\bigr)\,\Omega_{d,i}^2\;\sigma_z^{(i)}

    This is implemented as an extra Z control channel per qubit whose
    amplitude is :math:`s_i\,(I_i^2+Q_i^2)\,\Omega_{d,i}^2`.
    """

    def __init__(
        self,
        n_qubits: int,
        coupling_type: str = "XY",
        anharmonicities: np.ndarray | list | None = None,
        zz_crosstalk: np.ndarray | list | None = None,
        stark_shift_coeffs: np.ndarray | list | None = None,
    ):
        """
        Args:
            n_qubits: number of transmon qubits.
            coupling_type: type of qubit-qubit coupling.
                ``"XY"`` – exchange coupling (default, from capacitive coupling).
                ``"ZZ"`` – static ZZ coupling (from anharmonicity-mediated shifts).
                ``"XY+ZZ"`` – both exchange and ZZ terms.
            anharmonicities: per-qubit anharmonicities α_i (rad/s), used to
                compute the perturbative static ZZ shift when
                ``coupling_type`` includes ``"ZZ"`` and no calibrated ZZ
                matrix is provided.  If ``None``, ZZ coupling must be
                specified via ``zz_crosstalk`` or the ``zz_instances`` kwarg.
            zz_crosstalk: calibrated static ZZ coupling matrix
                ``(n_qubits, n_qubits)`` in rad/s.  Upper-triangular.  When
                provided, this takes priority over the perturbative formula
                derived from ``anharmonicities``.  A runtime ``zz_instances``
                kwarg to :meth:`build_drift` still takes highest priority.
            stark_shift_coeffs: per-qubit AC Stark shift coefficients s_i.
                When provided, an additional Z control channel per qubit is
                added with amplitude ``s_i * (I² + Q²) * Ω_d²``, modelling
                the drive-dependent frequency shift (light shift).
        """
        self.n_qubits = n_qubits
        self.coupling_type = coupling_type
        self.anharmonicities = (
            np.asarray(anharmonicities, dtype=float)
            if anharmonicities is not None
            else None
        )
        self.zz_crosstalk = (
            np.asarray(zz_crosstalk, dtype=float) if zz_crosstalk is not None else None
        )
        self.stark_shift_coeffs = (
            np.asarray(stark_shift_coeffs, dtype=float)
            if stark_shift_coeffs is not None
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
        """Number of control channels: 2 per qubit (I, Q), or 3 when Stark shift is enabled (I, Q, Z)."""
        if self.stark_shift_coeffs is not None:
            return 3 * self.n_qubits
        return 2 * self.n_qubits

    def build_drift(
        self,
        frequency_instances: list[np.ndarray],
        coupling_instances: list[np.ndarray] | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        r"""Build drift Hamiltonian snapshots.

        The ZZ coupling follows a priority chain:

        1. Runtime ``zz_instances`` kwarg (per-snapshot, highest priority).
        2. Constructor ``zz_crosstalk`` (calibrated, fixed across snapshots).
        3. Perturbative formula from ``anharmonicities``.
        4. Zero matrix (no ZZ coupling).

        Args:
            frequency_instances: list of detuning arrays ``(n_qubits,)`` in
                rad/s, one per snapshot.  These are rotating-frame detunings
                :math:`\delta_i`, not bare transition frequencies.
            coupling_instances: list of exchange-coupling matrices
                ``(n_qubits, n_qubits)`` in rad/s, one per snapshot.
                Upper-triangular.
            **kwargs: Optional ``zz_instances`` — list of ZZ coupling matrices
                ``(n_qubits, n_qubits)`` in rad/s, one per snapshot.  Overrides
                both ``zz_crosstalk`` and the perturbative formula.

        Returns:
            List of ``(D, D)`` complex numpy arrays.
        """
        zz_instances = kwargs.get("zz_instances", None)

        D = self.dim
        H0_list = []

        for idx, omega in enumerate(frequency_instances):
            H = np.zeros((D, D), dtype=complex)

            # Qubit detunings: sum_i delta_i * Z_i  (= delta_i/2 * sigma_z)
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
                    # ZZ priority chain: zz_instances > zz_crosstalk > formula
                    if zz_instances is not None:
                        zz = (
                            zz_instances[idx]
                            if idx < len(zz_instances)
                            else zz_instances[-1]
                        )
                    elif self.zz_crosstalk is not None:
                        zz = self.zz_crosstalk
                    elif self.anharmonicities is not None:
                        # Perturbative static ZZ from anharmonicity:
                        # zeta_{ij} ~ 2 g_{ij}^2 (1/alpha_i + 1/alpha_j)
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
        """Return control operators for each channel.

        Without Stark shift: ``[X_0, Y_0, X_1, Y_1, ...]`` (I/Q per qubit).
        With Stark shift:    ``[X_0, Y_0, Z_0, X_1, Y_1, Z_1, ...]``.
        """
        ops = []
        for i in range(self.n_qubits):
            ops.append(self._X[i])
            ops.append(self._Y[i])
            if self.stark_shift_coeffs is not None:
                ops.append(self._Z[i])
        return ops

    def control_amplitudes(
        self,
        cx: torch.Tensor,
        cy: torch.Tensor,
        rabi_freq: torch.Tensor,
        n_h0: int,
    ) -> torch.Tensor:
        r"""Map I/Q waveforms to control amplitudes.

        Without Stark shift, produces ``(n_pulse, n_batch, 2*n_qubits)``
        with channels ``[I_0*Ω_0, Q_0*Ω_0, I_1*Ω_1, Q_1*Ω_1, ...]``.

        With Stark shift enabled, produces ``(n_pulse, n_batch, 3*n_qubits)``
        with channels ``[I_0*Ω_0, Q_0*Ω_0, s_0*(I_0²+Q_0²)*Ω_0², ...]``.

        Args:
            cx: ``(n_pulse, n_qubits)`` — in-phase waveform.
            cy: ``(n_pulse, n_qubits)`` — quadrature waveform.
            rabi_freq: ``(n_rabi, n_qubits)`` — drive amplitude snapshots.
            n_h0: number of drift Hamiltonian snapshots.

        Returns:
            ``(n_pulse, n_batch, n_controls)`` where
            ``n_batch = n_rabi * n_h0``.
        """
        n_pulse = cx.shape[0]

        if self.stark_shift_coeffs is not None:
            # Build (n_pulse, 3*n_qubits) with [I, Q, s*(I²+Q²)] per qubit
            s = torch.as_tensor(
                self.stark_shift_coeffs, dtype=cx.dtype, device=cx.device
            )
            iq_power = cx**2 + cy**2  # (n_pulse, n_qubits)

            channels = []
            for i in range(self.n_qubits):
                channels.append(cx[:, i : i + 1])  # I_i
                channels.append(cy[:, i : i + 1])  # Q_i
                channels.append(s[i] * iq_power[:, i : i + 1])  # s_i*(I²+Q²)
            u = torch.cat(channels, dim=-1)  # (n_pulse, 3*n_qubits)

            # Build rabi scaling: [Ω_0, Ω_0, Ω_0², Ω_1, Ω_1, Ω_1², ...]
            rabi_cols = []
            for i in range(self.n_qubits):
                rabi_cols.append(rabi_freq[:, i : i + 1])  # X channel: Ω
                rabi_cols.append(rabi_freq[:, i : i + 1])  # Y channel: Ω
                rabi_cols.append(rabi_freq[:, i : i + 1] ** 2)  # Z channel: Ω²
            rabi_expanded = torch.cat(rabi_cols, dim=-1)  # (n_rabi, 3*n_qubits)
            rabi_batch = rabi_expanded.repeat(n_h0, 1)

            u = u.unsqueeze(1) * rabi_batch.unsqueeze(0)
            return u  # (n_pulse, n_rabi * n_h0, 3 * n_qubits)

        # Standard path (no Stark shift): [I_0, Q_0, I_1, Q_1, ...]
        u = torch.stack([cx, cy], dim=-1).reshape(n_pulse, 2 * self.n_qubits)

        # Drive amplitude scaling: [Omega_d_0, Omega_d_0, Omega_d_1, ...]
        rabi_expanded = rabi_freq.repeat_interleave(2, dim=-1)
        rabi_batch = rabi_expanded.repeat(n_h0, 1)

        u = u.unsqueeze(1) * rabi_batch.unsqueeze(0)

        return u  # (n_pulse, n_rabi * n_h0, 2 * n_qubits)

    @classmethod
    def from_config(cls, n_qubits: int, params: dict) -> SuperconductingQubitModel:
        """Construct from config dict, extracting superconducting-specific params.

        Frequency-domain parameters (``anharmonicities``, ``zz_crosstalk``)
        are specified in Hz in the configuration and converted to rad/s here.
        ``stark_shift_coeffs`` are dimensionless and used as-is.
        """
        coupling_type = params.get("coupling_type", "XY")
        anharmonicities = params.get("anharmonicities", None)
        if anharmonicities is not None:
            anharmonicities = 2 * np.pi * np.array(anharmonicities)

        zz_crosstalk = params.get("zz_crosstalk", None)
        if zz_crosstalk is not None:
            zz_crosstalk = 2 * np.pi * np.array(zz_crosstalk)

        stark_shift_coeffs = params.get("stark_shift_coeffs", None)
        if stark_shift_coeffs is not None:
            stark_shift_coeffs = np.array(stark_shift_coeffs)

        return cls(
            n_qubits,
            coupling_type=coupling_type,
            anharmonicities=anharmonicities,
            zz_crosstalk=zz_crosstalk,
            stark_shift_coeffs=stark_shift_coeffs,
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
