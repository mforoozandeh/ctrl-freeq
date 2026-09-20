import itertools
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


from ctrl_freeq.setup.hamiltonian_generation.hamiltonians import (
    createHcs,
    createHJ,
    _symmetrise_coupling,
)
from ctrl_freeq.setup.hamiltonian_generation import get_hamiltonian_class
from ctrl_freeq.setup.basis_generation.mat_x0_gen import (
    generate_mat_x0_from_basis,
    generate_mat_x0_from_fourier_basis,
    mat_with_amplitude_and_qr,
)
from ctrl_freeq.setup.operator_generation.generate_operators import (
    create_hamiltonian_basis,
    create_density_matrices,
    create_observable_operators,
)


from ctrl_freeq.make_pulse.waveform_gen_torch import WaveformSpec
from ctrl_freeq.utils.utility_functions import generate_instances

# Gate-name aliases mapped onto their canonical name.  Two configurations that
# name the same physical gate must select the same objective, so aliases are
# resolved before anything compares gate names.
_GATE_ALIASES = {
    "CX": "CNOT",
    "SQRTISWAP": "√iSWAP",
    "TOFFOLI": "Toff",
    "CCX": "Toff",
}


def canonical_gate_name(gate):
    """Return the canonical name for *gate*, resolving known aliases."""
    if not isinstance(gate, str):
        raise ValueError(f"Gate name must be a string, got {gate!r}.")
    return _GATE_ALIASES.get(gate.upper(), gate)


def pauli_string_basis(n_qubits):
    r"""Return the ``4**n`` unnormalised Pauli strings, identity first.

    They satisfy :math:`\mathrm{Tr}(P_j P_k) = d\,\delta_{jk}` with
    :math:`d = 2^n`, which is the normalisation the average-gate-fidelity
    channel formula assumes.  These are operators, not density matrices: they
    are traceless (except :math:`P_0 = I`) and generally indefinite, so they
    must never be fed to a state fidelity.
    """
    singles = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    basis = []
    for indices in itertools.product(range(4), repeat=n_qubits):
        op = np.array([[1.0 + 0j]])
        for i in indices:
            op = np.kron(op, singles[i])
        basis.append(op)
    return basis


def band_selective_profile(offsets, centre, bandwidth, order):
    r"""Target rotation-angle profile for ``band_selective`` coverage.

    .. math::

        p(\Delta) = \exp\!\left[-\ln 2
            \left(\frac{2\,|\Delta - \Delta_0|}{\text{bw}}\right)^{2p}\right]

    ``bandwidth`` is the **FWHM of this target rotation-angle profile**: the
    profile equals 1 at the centre and exactly 1/2 at ``centre ± bw/2``, for
    every order *p*.  It is *not* a guarantee that the achieved excitation
    curve of the optimised pulse has that FWHM.

    Args:
        offsets: offsets at which to evaluate the profile (rad/s).
        centre: band centre (rad/s).
        bandwidth: FWHM of the target profile (rad/s), strictly positive.
        order: super-Gaussian order *p*, a positive integer.
    """
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(
            f"band_selective bandwidth must be positive and finite, got {bandwidth!r}."
        )
    order_int = int(order)
    if order_int != order or order_int <= 0:
        raise ValueError(
            f"band_selective profile_order must be a positive integer, got {order!r}."
        )
    scaled = 2.0 * np.abs(np.asarray(offsets, dtype=float) - centre) / bandwidth
    return np.exp(-np.log(2.0) * scaled ** (2 * order_int))


@dataclass
class Initialise:
    def __init__(self, data):
        # Set by an optimizer whose solution vector is not in the configured
        # basis representation; see waveform_spec().
        self._waveform_spec = None
        self.qubits = data["qubits"]
        self.n_qubits = len(self.qubits)
        self.space = data["optimization"]["space"]
        self.coverage = data["parameters"]["coverage"]
        self.pulse_duration = data["parameters"]["pulse_duration"][0]
        self.np_pulse = data["parameters"]["point_in_pulse"][0]
        self.wf_type = data["parameters"]["wf_type"]
        self.wf_mode = data["parameters"]["wf_mode"]
        self.amplitude_envelope = data["parameters"]["amplitude_envelope"]
        self.amplitude_order = data["parameters"]["amplitude_order"]
        self.algorithm = data["optimization"]["algorithm"]
        self.op = create_hamiltonian_basis(self.n_qubits)
        self.max_iter = data["optimization"]["max_iter"]
        self.targ_fid = data["optimization"]["targ_fid"]
        self.n_para = data["parameters"]["n_para"]
        self.n_para_updated = self.update_n_para()
        self.coupling_type = (
            data["parameters"].get("coupling_type") if self.n_qubits > 1 else None
        )
        self.sw = 2 * np.pi * data["parameters"]["sw"]
        self.Delta = (
            2 * np.pi * np.array(data["parameters"]["Delta"])
            if "Delta" in data["parameters"]
            else np.zeros(self.n_qubits)
        )
        self.sigma_Delta = (
            2 * np.pi * np.array(data["parameters"]["sigma_Delta"])
            if "sigma_Delta" in data["parameters"]
            else np.zeros(self.n_qubits)
        )
        self.sigma_J = (
            2 * np.pi * data["parameters"]["sigma_J"]
            if self.n_qubits > 1
            and "sigma_J" in data["parameters"]
            and data["parameters"]["sigma_J"] is not None
            else None
        )
        self.Jmat = (
            2 * np.pi * np.array(data["parameters"]["J"])
            if "J" in data["parameters"]
            else np.zeros((self.n_qubits, self.n_qubits))
        )
        self.H0_snapshots = data["optimization"]["H0_snapshots"]
        self.Omega_R_max = 2 * np.pi * data["parameters"]["Omega_R_max"]
        self.sigma_Omega_R_max = 2 * np.pi * data["parameters"]["sigma_Omega_R_max"]
        self.Omega_R_snapshots = data["optimization"]["Omega_R_snapshots"]
        self.init_ax = data["initial_states"]
        self.targ_ax = data["target_states"]
        self.profile_order = data["parameters"]["profile_order"]
        self.ratio_factor = data["parameters"]["ratio_factor"]
        self.pulse_offset = 2 * np.pi * data["parameters"]["pulse_offset"]
        self.pulse_bandwidth = 2 * np.pi * data["parameters"]["pulse_bandwidth"]

        # Dissipation parameters
        self.dissipation_mode = data["optimization"].get(
            "dissipation_mode", "non-dissipative"
        )
        if self.dissipation_mode == "dissipative":
            # Force Liouville space for dissipative evolution
            self.space = "liouville"
            self.T1 = np.array(data["parameters"]["T1"], dtype=float)
            self.T2 = np.array(data["parameters"]["T2"], dtype=float)
            # Validate T1/T2 are positive and finite
            for i in range(self.n_qubits):
                if not (np.isfinite(self.T1[i]) and self.T1[i] > 0):
                    raise ValueError(
                        f"Qubit {i + 1}: T1 must be positive and finite, "
                        f"got {self.T1[i]}"
                    )
                if not (np.isfinite(self.T2[i]) and self.T2[i] > 0):
                    raise ValueError(
                        f"Qubit {i + 1}: T2 must be positive and finite, "
                        f"got {self.T2[i]}"
                    )
                # Physical constraint: T2 <= 2*T1
                if self.T2[i] > 2 * self.T1[i]:
                    raise ValueError(
                        f"Qubit {i + 1}: T2 ({self.T2[i]:.2e}) must be <= 2*T1 ({2 * self.T1[i]:.2e})"
                    )
            self.collapse_operators = self.build_collapse_operators()
        else:
            self.T1 = None
            self.T2 = None
            self.collapse_operators = None

        self.frq_band = self.get_frequency_band()
        self.t = self.generate_time_sequence()
        self.x0 = self.generate_initial_x0()
        self.mat, self.x0_updated = self.generate_matrices_from_basis()
        self.Omega_R = self.get_omega_1()
        self.offs = self.get_offset()
        self.Omega_instances = self.generate_Omega_instances()
        self.Jmat_instances = self.generate_Jmat_instances()
        self.excitation_profile = self.get_excitation_profile()
        self.x0_con = np.concatenate(self.x0_updated)
        self.obs_op = create_observable_operators(self.n_qubits)
        self.modulation_exponent = self.pulse_offset_exponent()

        # Calculate the target state
        if "initial_states" in data and "target_states" in data:
            self.init_ax = data["initial_states"]
            self.init = self.get_initial_state_from_ax()

            # ``band_selective`` produces a smooth rotation-angle profile, so
            # almost every drawn offset sits strictly between 0 and 1.  An Axis
            # or Gate target is all-or-nothing: there is no partial version of
            # "apply CNOT".  Silently handing such an offset the initial state
            # as its target requests the identity nearly everywhere, which is
            # not what the configuration asks for.
            if any(cov == "band_selective" for cov in self.coverage) and (
                "Axis" in data["target_states"] or "Gate" in data["target_states"]
            ):
                raise ValueError(
                    "band_selective coverage is not supported with Axis or Gate "
                    "targets: the smooth rotation-angle profile has no "
                    "all-or-nothing interpretation for a discrete target. "
                    "Use 'selective' coverage for hard in-band/out-of-band "
                    "targets, or a Phi/Beta (rotation-angle) target, which "
                    "scales smoothly with the profile."
                )

            if "Axis" in data["target_states"]:
                self.targ_ax = data["target_states"]["Axis"]
                self.targ = self.get_target_state_from_ax()
                self.initial = self.init
                self.target = self.targ
                self.initials, self.targets = self.compute_targets_targ()

            elif "Gate" in data["target_states"]:
                self.gate = data["target_states"]["Gate"]
                self.initial = self.init
                self.initials, self.targets = self.compute_targets_gate()

            elif "Phi" in data["target_states"] and "Beta" in data["target_states"]:
                self.axis = data["target_states"]["Phi"]
                self.beta = np.pi * data["target_states"]["Beta"] / 180
                self.initial = self.init
                self.betas = self.get_betas()
                self.u_tot = self.compute_u_tot_beta_axis()
                self.initials, self.targets = self.compute_targets_beta_axis()

        else:
            raise ValueError(
                "Input data is missing necessary initial or target state information."
            )

        # Build Hamiltonian model (if hamiltonian_type is specified in config)
        self.hamiltonian_type = data.get("hamiltonian_type", None)
        self.hamiltonian_model = self._build_hamiltonian_model(data)

        # Guard: dissipative mode is only supported for 2-level models.
        # Collapse operators are built from 2×2 Pauli matrices and the
        # liouville state-embedding path expects state vectors, not density
        # matrices.  Block the combination until model-aware dissipation is
        # implemented.
        if (
            self.hamiltonian_model is not None
            and self.hamiltonian_model.dim != 2**self.n_qubits
            and self.dissipation_mode == "dissipative"
        ):
            raise ValueError(
                f"Dissipative mode is not yet supported for models with "
                f"local dimension > 2 (model dim = {self.hamiltonian_model.dim}, "
                f"computational dim = {2**self.n_qubits}). "
                f"Collapse operators and Liouville-space embedding are "
                f"currently hard-coded for 2-level systems."
            )

        # Embed states/gates into model Hilbert space (e.g. 3-level Duffing)
        if self.hamiltonian_model is not None:
            self._embed_states_for_model()

        self.H0 = self.get_H0()

        # Optional runtime settings
        try:
            self.compute_resource = data.get("compute_resource", "cpu")
        except Exception:
            self.compute_resource = "cpu"
        try:
            self.cpu_cores = data.get("cpu_cores")
        except Exception:
            self.cpu_cores = None

    def _build_hamiltonian_model(self, data):
        """Build a HamiltonianModel instance from the config, or None for legacy path."""
        h_type = self.hamiltonian_type
        if h_type is None:
            return None

        params = data.get("parameters", {})
        cls = get_hamiltonian_class(h_type)
        return cls.from_config(self.n_qubits, params)

    def _embed_states_for_model(self):
        """Embed computational states/gates into the model's Hilbert space.

        For 2-level models (dim = 2^n) this is a no-op.  For models with
        larger local dimensions (e.g. DuffingTransmonModel with dim = 3^n),
        the ``embed_computational_state`` / ``embed_computational_gate``
        methods re-map the 2^n vectors/matrices into the full space.

        Idempotent: if states are already at the model dimension, this is
        a no-op.  Raises ``ValueError`` if states are at an incompatible
        dimension (e.g. embedded for a different non-2-level model).
        """
        model = self.hamiltonian_model
        d_comp = 2**self.n_qubits
        d_model = model.dim
        if d_model == d_comp:
            # Standard 2-level model — no embedding needed
            return

        def _check_and_embed_states(states, label):
            """Return embedded states, or the originals if already embedded.

            State *vectors* embed as ``V psi``; density matrices and Pauli
            strings embed as ``V rho V^dag``.  The two are distinguished by
            array rank, never by a trailing dimension: a ``(d, d)`` density
            matrix and a ``(d,)`` vector both end in ``d``, and applying the
            vector rule to a density matrix produces a ``(D, d)`` object that
            no longer matches the ``(D, D)`` drift matrices.
            """
            if states is None:
                return None
            first = np.asarray(states[0])
            ndim = first.ndim
            if ndim not in (1, 2):
                raise ValueError(
                    f"{label} must be state vectors (1-D) or operators (2-D), "
                    f"got rank {ndim}."
                )
            dim = first.shape[-1]
            if dim == d_model:
                return states  # already at target dimension
            if dim != d_comp:
                raise ValueError(
                    f"{label} have dimension {dim}, which is incompatible "
                    f"with both the computational dimension ({d_comp}) and "
                    f"the model dimension ({d_model}). Cannot re-embed — "
                    f"re-initialise from the config instead."
                )
            if ndim == 1:
                return [model.embed_computational_state(s) for s in states]
            if first.shape[0] != d_comp:
                raise ValueError(
                    f"{label} must be square in the computational dimension "
                    f"({d_comp}), got shape {first.shape}."
                )
            return [model.embed_computational_operator(np.asarray(s)) for s in states]

        # Embed initial and target state vectors (used by optimizer)
        if hasattr(self, "initials") and self.initials is not None:
            self.initials = np.array(
                _check_and_embed_states(self.initials, "Initial states")
            )
        if hasattr(self, "targets") and self.targets is not None:
            self.targets = np.array(
                _check_and_embed_states(self.targets, "Target states")
            )

        # Embed raw init states (used by plotter for evolution replay)
        if hasattr(self, "init") and self.init is not None:
            self.init = _check_and_embed_states(self.init, "Raw init states")

        # Embed observable operators into the full space so plots work.
        # Observables embed as V O V^dag — the gate embedding adds identity on
        # the leakage subspace, which would make a fully leaked state report
        # <Z> = +1 instead of 0 and corrupt <X> and <Y> the same way.
        if hasattr(self, "obs_op") and self.obs_op is not None:
            new_ops = {}
            for k, v in self.obs_op.items():
                if v.shape[0] == d_model:
                    new_ops[k] = v  # already embedded
                elif v.shape[0] == d_comp:
                    new_ops[k] = model.embed_computational_operator(v)
                else:
                    raise ValueError(
                        f"Observable operator '{k}' has dimension {v.shape[0]}, "
                        f"incompatible with model dimension {d_model}."
                    )
            self.obs_op = new_ops

    def __str__(self):
        return (
            f"Initialisation:\n"
            f"Coverage: {self.coverage}\n"
            f"Number of Qubits: {self.n_qubits}\n"
            f"Pulse Duration: {self.pulse_duration}\n"
            f"Number of Points in Pulse: {self.np_pulse}\n"
            f"WF Type: {self.wf_type}\n"
            f"WF Mode: {self.wf_mode}\n"
            f"Op: {self.op}\n"
            f"Max Iterations: {self.max_iter}\n"
            f"Target Fidelity: {self.targ_fid}\n"
            f"Number of Parameters: {self.n_para}\n"
            f"Algorithm: {self.algorithm}\n"
            f"Coupling Type: {self.coupling_type}\n"
            f"SW: {self.sw}\n"
            f"σ Delta: {self.sigma_Delta}\n"
            f"Delta: {self.Delta}\n"
            f"σ J: {self.sigma_J}\n"
            f"J Matrix: {self.Jmat}\n"
            f"H0 Snapshots: {self.H0_snapshots}\n"
            f"σ Ω_R max: {self.sigma_Omega_R_max}\n"
            f"Ω_R max: {self.Omega_R_max}\n"
            f"Ω_R Snapshots: {self.Omega_R_snapshots}\n"
            f"Ω_R: {self.Omega_R}\n"
            f"x0 Concatenated: {self.x0_con}\n"
        )

    @property
    def dt(self):
        """Propagation time step used by every evolution path: ``T / N``."""
        return self.pulse_duration / self.np_pulse

    def generate_time_sequence(self):
        """Waveform sample times: the midpoint of each propagation interval.

        The optimiser advances the state by ``dt = T / N`` for each of the
        ``N`` samples, so sample ``k`` represents the interval
        ``[k*dt, (k+1)*dt]`` and is evaluated at its midpoint
        ``(k + 1/2) * dt``.  The previous ``linspace(eps, T, N)`` grid was
        spaced by ``T / (N - 1)``, which made the carrier modulation advance
        faster than the propagation and pushed analysis past the intended
        pulse duration.
        """
        return (np.arange(self.np_pulse) + 0.5) * self.dt

    def state_boundary_times(self):
        """The ``N + 1`` times at which a propagated state exists: ``k * dt``.

        These are the times for stored trajectories (including the initial
        state at ``t = 0`` and the final state at exactly ``T``), and are
        distinct from the waveform sample times returned by
        :meth:`generate_time_sequence`.
        """
        return np.arange(self.np_pulse + 1) * self.dt

    def generate_initial_x0(self):
        self.x0 = []
        for i in range(self.n_qubits):
            self.x0.append(np.random.uniform(-1, 1, size=self.n_para[i]))
        return self.x0

    def basis_waveform_spec(self):
        """Return the configured orthonormal-basis representation.

        This is what a basis-path solution is expressed in, independent of
        which optimizer last ran on this object.
        """
        return WaveformSpec(
            n_para=tuple(self.n_para_updated),
            mat=tuple(self.mat),
            wf_mode=tuple(self.wf_mode),
        )

    def waveform_spec(self):
        """Return the representation of the most recent run's solution.

        Defaults to the configured basis.  An optimizer that uses a different
        parameterisation (e.g. the piecewise identity basis) records its own
        via ``_waveform_spec`` so that analysis reconstructs the waveform that
        optimiser evaluated.

        This tracks the *latest* run only.  When several optimizers share one
        parameters object, pass the owning run's spec to the analysis entry
        points rather than relying on this.
        """
        if self._waveform_spec is not None:
            return self._waveform_spec
        return self.basis_waveform_spec()

    def generate_matrices_from_basis(self):
        self.mat = []
        self.x0_updated = []
        for i in range(self.n_qubits):
            if self.wf_type[i] == "fou":
                mat_val, x0_updated_val = generate_mat_x0_from_fourier_basis(
                    self.x0[i], self.np_pulse, self.wf_mode[i]
                )
                mat_val, x0_updated_val = mat_with_amplitude_and_qr(
                    mat_val,
                    x0_updated_val,
                    self.wf_mode[i],
                    self.amplitude_envelope[i],
                    self.amplitude_order[i],
                )
                self.mat.append(mat_val)
                self.x0_updated.append(x0_updated_val)
            else:
                mat_val, x0_updated_val = generate_mat_x0_from_basis(
                    self.x0[i], self.np_pulse, self.wf_type[i], self.wf_mode[i]
                )
                mat_val, x0_updated_val = mat_with_amplitude_and_qr(
                    mat_val,
                    x0_updated_val,
                    self.wf_mode[i],
                    self.amplitude_envelope[i],
                    self.amplitude_order[i],
                )
                self.mat.append(mat_val)
                self.x0_updated.append(x0_updated_val)
        return self.mat, self.x0_updated

    def get_axis_vector(self):
        n_set = []
        axis_mapping = {
            "x": np.array([1, 0, 0], dtype=np.float64),
            "y": np.array([0, 1, 0], dtype=np.float64),
            "z": np.array([0, 0, 1], dtype=np.float64),
        }
        for ax in self.axis:
            n = []
            for i in range(self.n_qubits):
                n.append(axis_mapping.get(ax[i]))
            n_set.append(np.array(n, dtype=np.float64))
        return n_set

    @staticmethod
    def normalize_vector(vec):
        normalized_vec = []
        for v in vec:
            norm = np.linalg.norm(v)
            if norm != 0:
                normalized_vec.append(v / norm)
            else:
                normalized_vec.append(v)
        return normalized_vec

    def get_gate(self, gate):
        if self.n_qubits == 1:
            gate_mat = self.single_qubit_gate(gate)
        elif self.n_qubits == 2:
            gate_mat = self.two_qubit_gate(gate)
        elif self.n_qubits == 3:
            gate_mat = self.three_qubit_gate(gate)
        return gate_mat

    def single_qubit_gate(self, gate):
        """
        Generate a 2x2 matrix for a specified single-qubit gate ('X', 'Y', 'Z', 'H', 'S', 'T').

        :return: 2x2 matrix for the operation
        """

        if gate == "X":
            return np.array([[0, 1], [1, 0]])

        elif gate == "Y":
            return np.array([[0, -1j], [1j, 0]])

        elif gate == "Z":
            return np.array([[1, 0], [0, -1]])

        elif gate == "H":
            return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        elif gate == "S":
            return np.array([[1, 0], [0, 1j]])

        elif gate == "T":
            return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

        else:
            raise ValueError(
                "Unsupported operation. Please specify 'X', 'Y', 'Z', 'H', 'S', or 'T'."
            )

    def two_qubit_gate(self, gate):
        """
        Generate a 4x4 matrix for a specified two-qubit gate.

        Supported: 'CNOT'/'CX', 'CZ', 'SWAP', 'iSWAP', '√iSWAP', 'ECR'.

        :return: 4x4 matrix for the operation
        """

        if gate == "CNOT" or gate == "CX":
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        elif gate == "CZ":
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

        elif gate == "SWAP":
            return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

        elif gate == "iSWAP":
            return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])

        elif gate == "√iSWAP":
            s = 1 / np.sqrt(2)
            return np.array(
                [
                    [1, 0, 0, 0],
                    [0, s, 1j * s, 0],
                    [0, 1j * s, s, 0],
                    [0, 0, 0, 1],
                ]
            )

        elif gate == "ECR":
            s = 1 / np.sqrt(2)
            # Echoed Cross-Resonance: (1/√2)(IX - XY)
            return np.array(
                [
                    [0, 0, s, 1j * s],
                    [0, 0, 1j * s, s],
                    [s, -1j * s, 0, 0],
                    [-1j * s, s, 0, 0],
                ]
            )

        else:
            raise ValueError(
                "Unsupported operation. Please specify 'CNOT', 'CZ', 'SWAP', "
                "'iSWAP', '√iSWAP', or 'ECR'."
            )

    def three_qubit_gate(self, gate):
        if gate == "Toff":
            return np.array(
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                ]
            )

    def _coupling_is_deterministic(self):
        """True when the coupling part of the drift carries no uncertainty."""
        if self.n_qubits < 2:
            return True
        if self.sigma_J is None or self.sigma_J == 0:
            return True
        # Uncertainty is only applied to non-zero nominal couplings.
        return not np.any(np.asarray(self.Jmat) != 0)

    def _offsets_are_deterministic(self):
        """True when every qubit's frequency offset is a fixed single value."""
        return all(cov == "single" for cov in self.coverage) and all(
            sig == 0 for sig in self.sigma_Delta
        )

    def _sample_banded_offsets(self, om, sig, sw, fb, rf, in_band_normal):
        """Draw ``H0_snapshots`` offsets split between in-band and out-of-band.

        ``in_band_normal`` selects a normal draw about *om* (``selective``)
        rather than a uniform draw across the band (``band_selective``).

        The completed per-qubit sample array is shuffled before it is
        returned.  Joint snapshots are formed by pairing each qubit's samples
        by index, so leaving the arrays in ``[left | in-band | right]`` order
        would make every qubit in-band or out-of-band at the same index and
        the ensemble would never contain a mixed combination.
        """

        num_outside_band = int(np.round(self.H0_snapshots * rf))
        num_outside_band = min(max(num_outside_band, 0), self.H0_snapshots)
        num_in_band = self.H0_snapshots - num_outside_band

        # Split the out-of-band samples as evenly as the count allows and give
        # the remainder to one tail.  Rounding the count up to an even number
        # and then halving it for *both* tails drops a sample: with
        # ratio_factor 1, requests of 1/3/5 snapshots produced 0/2/4.
        num_left = num_outside_band // 2
        num_right = num_outside_band - num_left

        offs_in_left_band = np.random.uniform(om - sw / 2, fb[0], num_left)
        offs_in_right_band = np.random.uniform(fb[1], om + sw / 2, num_right)
        if in_band_normal:
            offs_in_middle_band = np.random.normal(om, sig, num_in_band)
        else:
            offs_in_middle_band = np.random.uniform(fb[0], fb[1], num_in_band)

        offset = np.concatenate(
            [offs_in_left_band, offs_in_middle_band, offs_in_right_band]
        )
        if offset.size != self.H0_snapshots:
            raise ValueError(
                f"Selective sampling produced {offset.size} offsets for "
                f"{self.H0_snapshots} requested snapshots."
            )
        np.random.shuffle(offset)
        return offset

    def get_offset(self):
        offset_results = []

        if self._offsets_are_deterministic():
            # The offsets themselves carry no uncertainty.  Collapse to a
            # single drift snapshot only when the *whole* drift is
            # deterministic; if the coupling is uncertain the requested
            # ensemble still needs H0_snapshots drift matrices, so the fixed
            # offsets are repeated to match it.
            n_snapshots = 1 if self._coupling_is_deterministic() else self.H0_snapshots
            for om in self.Delta:
                offset_results.append(np.full(n_snapshots, om, dtype=float))
            return offset_results

        for coverage, om, sig, sw, fb, rf in zip(
            self.coverage,
            self.Delta,
            self.sigma_Delta,
            self.sw,
            self.frq_band,
            self.ratio_factor,
        ):
            if coverage == "broadband":
                offset = np.random.uniform(om - sw / 2, om + sw / 2, self.H0_snapshots)

            elif coverage == "band_selective":
                offset = self._sample_banded_offsets(
                    om, sig, sw, fb, rf, in_band_normal=False
                )

            elif coverage == "single":
                offset = np.random.normal(om, sig, self.H0_snapshots)

            elif coverage == "selective":
                offset = self._sample_banded_offsets(
                    om, sig, sw, fb, rf, in_band_normal=True
                )

            else:
                raise ValueError(f"Unknown coverage type: {coverage}")

            offset_results.append(offset)

        return offset_results

    def get_frequency_band(self):
        frequency_band = []
        for Om, pbw in zip(self.Delta, self.pulse_bandwidth):
            # Calculating the lower and upper bounds of frequency from bandwidth and center bandwidth
            lower_bound = Om - pbw / 2
            upper_bound = Om + pbw / 2

            frequency_band.append([lower_bound, upper_bound])

        return frequency_band

    def get_omega_1(self):
        return generate_instances(
            self.Omega_R_max, self.sigma_Omega_R_max, self.Omega_R_snapshots
        )

    def get_H0(self):
        if self.hamiltonian_model is not None:
            return self._get_H0_from_model()

        # Legacy path (no hamiltonian_type specified)
        if self.n_qubits == 1:
            HCSs = []
            for Omega_instance in self.Omega_instances:
                HCS = createHcs(Omega_instance, self.op)
                HCSs.append(HCS)
            H0 = HCSs
        elif self.n_qubits > 1:
            HCSs = []
            for Omega_instance in self.Omega_instances:
                HCS = createHcs(Omega_instance, self.op)
                HCSs.append(HCS)

            HJs = []
            for Jmat_instance in self.Jmat_instances:
                HJ = createHJ(Jmat_instance, self.op, coupling_type=self.coupling_type)
                HJs.append(HJ)

            H0 = [HJ + HCS for HJ, HCS in zip(HJs, HCSs)]

        return self._stack_H0_rows(H0)

    def _stack_H0_rows(self, H0):
        """Repeat the drift ensemble once per objective row (row-major order).

        The resulting flat index is ``row * n_drift_snapshots + snapshot``,
        matching the ordering the initial/target arrays are built in.
        """
        n_snapshots = self.n_drift_snapshots()
        if len(H0) != n_snapshots:
            raise ValueError(
                f"Drift ensemble has {len(H0)} matrices but "
                f"{n_snapshots} drift snapshots were requested."
            )
        n_rows = getattr(self, "n_objective_rows", None)
        if n_rows is None:
            raise ValueError(
                "Objective rows are unknown; targets must be computed before H0."
            )
        H0_stacked = []
        for _ in range(n_rows):
            H0_stacked.extend(H0)
        return H0_stacked

    def _get_H0_from_model(self):
        """Build H0 using the HamiltonianModel abstraction."""
        model = self.hamiltonian_model
        coupling = self.Jmat_instances if self.n_qubits > 1 else None
        H0 = model.build_drift(
            frequency_instances=self.Omega_instances,
            coupling_instances=coupling,
        )

        return self._stack_H0_rows(H0)

    def create_state_vector_pure(self, ax):
        """
        Create the initial state for an n-qubit system based on a list of directions.
        Directions can be 'Z', '-Z', 'X', '-X', 'Y', '-Y'.
        """
        # Define the single-qubit states inside the function
        state_0 = np.array([1, 0], dtype=complex)  # |0>
        state_1 = np.array([0, 1], dtype=complex)  # |1>
        state_plus = (state_0 + state_1) / np.sqrt(2)  # |+>
        state_minus = (state_0 - state_1) / np.sqrt(2)  # |->
        state_i_plus = (state_0 + 1j * state_1) / np.sqrt(2)  # |i+>
        state_i_minus = (state_0 - 1j * state_1) / np.sqrt(2)  # |i->

        # Mapping of directions to states
        state_map = {
            "Z": state_0,
            "-Z": state_1,
            "X": state_plus,
            "-X": state_minus,
            "Y": state_i_plus,
            "-Y": state_i_minus,
        }

        # Initialize the state with the state of the first qubit
        pure_state = state_map[ax[0]]

        # Tensor product with each subsequent qubit's state
        for direction in ax[1:]:
            pure_state = np.kron(pure_state, state_map[direction])

        return pure_state

    def create_state_vector_mixed(self, ax):
        """
        Create the state for an n-qubit system based on a list of directions.
        Directions can be 'Z', '-Z', 'X', '-X', 'Y', '-Y'.
        """
        mixed_state = create_density_matrices(self.n_qubits, ax)

        return mixed_state

    def get_initial_state_from_ax(self):
        init_set = []
        for ax in self.init_ax:
            if self.space == "hilbert":
                init_set.append(self.create_state_vector_pure(ax))
            elif self.space == "liouville":
                init_set.append(self.create_state_vector_mixed(ax))
        return init_set

    def get_target_state_from_ax(self):
        targ_set = []
        for ax in self.targ_ax:
            if self.space == "hilbert":
                targ_set.append(self.create_state_vector_pure(ax))
            elif self.space == "liouville":
                targ_set.append(self.create_state_vector_mixed(ax))
        return targ_set

    def get_excitation_profile(self):
        profiles = []

        for cov, ord, om, pbw, offs in zip(
            self.coverage,
            self.profile_order,
            self.Delta,
            self.pulse_bandwidth,
            self.offs,
        ):
            offs = np.asarray(offs, dtype=float)
            if cov == "broadband":
                profile = np.ones(len(offs))
            elif cov == "single":
                profile = np.ones(len(offs))
            elif cov == "selective":
                profile = np.where(
                    (offs >= om - pbw / 2) & (offs <= om + pbw / 2), 1, 0
                )
            elif cov == "band_selective":
                profile = band_selective_profile(offs, om, pbw, ord)
            else:
                raise ValueError(f"Unknown coverage type: {cov}")

            profiles.append(profile)

        return profiles

    def n_drift_snapshots(self):
        """Number of drift (H0) snapshots in the requested ensemble."""
        return len(self.Omega_instances)

    def generate_Omega_instances(self):
        lengths = {len(o) for o in self.offs}
        if len(lengths) != 1:
            raise ValueError(
                f"Per-qubit offset sample arrays have inconsistent lengths "
                f"{sorted(len(o) for o in self.offs)}; they are paired by index "
                f"to form joint drift snapshots and must all match."
            )
        return [list(group) for group in zip(*self.offs)]

    def generate_Jmat_instances(self):
        """Draw one coupling matrix per drift snapshot.

        The nominal coupling matrix is normalised once (upper-triangular,
        lower-triangular and symmetric inputs all give the same result), then
        exactly one random value is drawn per *physical pair* and mirrored
        into both triangles.  Sampling each triangle independently would turn
        a symmetric nominal matrix into an asymmetric one and the downstream
        builders would reject it.
        """
        n_snapshots = self.n_drift_snapshots()
        if self.n_qubits < 2:
            return [
                np.zeros((self.n_qubits, self.n_qubits)) for _ in range(n_snapshots)
            ]

        J = _symmetrise_coupling(self.Jmat)
        sigma = self.sigma_J if self.sigma_J is not None else 0.0
        iu = np.triu_indices(self.n_qubits, k=1)
        nominal = J[iu]

        Jmat_instances = []
        for _ in range(n_snapshots):
            if sigma:
                drawn = np.where(nominal != 0, np.random.normal(nominal, sigma), 0.0)
            else:
                drawn = nominal
            instance = np.zeros_like(J)
            instance[iu] = drawn
            Jmat_instances.append(instance + instance.T)
        return Jmat_instances

    def get_Jmat(self):
        # Create a random n x n matrix
        random_matrix = 1e6 * np.random.rand(self.n_qubits, self.n_qubits)

        # Make the matrix symmetric
        symmetric_matrix = (random_matrix + random_matrix.T) / 2

        # Set the diagonal elements to zero
        np.fill_diagonal(symmetric_matrix, 0)

        return symmetric_matrix

    def update_parameters(self, new_np_pulse, new_targ_fid, new_x0):
        self.np_pulse = new_np_pulse
        self.targ_fid = new_targ_fid
        self.x0_con = new_x0
        self.t = self.generate_time_sequence()
        self.mat, _ = self.generate_matrices_from_basis()
        return self

    def update_n_para(self):
        n_para_updated = self.n_para.copy()
        for i in range(self.n_qubits):
            if self.wf_type[i] == "fou":
                # Fourier basis recalculates n_para internally
                if self.wf_mode[i] == "polar_phase":
                    # n = (n_para - 1) // 2, then n_para = 2*n + 1, then +1 for polar_phase
                    n = (self.n_para[i] - 1) // 2
                    n_para_updated[i] = 2 * n + 1 + 1
                else:
                    # n = (n_para - 2) // 4, then n_para = 4*n + 2
                    n = (self.n_para[i] - 2) // 4
                    n_para_updated[i] = 4 * n + 2
            elif self.wf_mode[i] == "polar_phase":
                # Non-Fourier basis with polar_phase just adds 1
                n_para_updated[i] += 1
        return n_para_updated

    # ------------------------------------------------------------------
    # Target construction
    #
    # Every objective array is built in one canonical ordering:
    #
    #     flat index = row * n_drift_snapshots + drift_snapshot
    #
    # where ``row`` indexes the objective's rows (configured initial states
    # for state transfer, computational basis columns or Pauli strings for a
    # gate objective).  ``get_H0`` stacks the drift snapshots row-major to
    # match, and ``h0_omega_1_iterator_torch`` then expands the Rabi snapshot
    # as the fastest-varying index.  Building any of these arrays in a
    # different order silently pairs each row with a subset of the ensemble.
    # ------------------------------------------------------------------

    def _qubit_in_band(self, qubit, snapshot):
        """True when *qubit* lies inside its selective band for *snapshot*."""
        return bool(self.excitation_profile[qubit][snapshot] == 1)

    def _snapshot_all_in_band(self, snapshot):
        """True when every qubit is inside its band for *snapshot*."""
        return all(self._qubit_in_band(q, snapshot) for q in range(self.n_qubits))

    def _state_from_ax(self, ax):
        """Build a state (vector or density matrix) from per-qubit axis labels."""
        if self.space == "hilbert":
            return self.create_state_vector_pure(ax)
        return self.create_state_vector_mixed(ax)

    def _finalise_objective_arrays(self, inits, targets, n_rows):
        """Validate and stack row-major ``(row, snapshot)`` objective arrays."""
        n_snapshots = self.n_drift_snapshots()
        expected = n_rows * n_snapshots
        if len(inits) != expected or len(targets) != expected:
            raise ValueError(
                f"Objective arrays have {len(inits)} initial and {len(targets)} "
                f"target entries; expected {expected} "
                f"({n_rows} rows x {n_snapshots} drift snapshots)."
            )
        self.n_objective_rows = n_rows
        inits = np.array(inits)
        targets = np.array(targets)
        if inits.shape != targets.shape:
            raise ValueError(
                f"Initial states {inits.shape} and targets {targets.shape} "
                f"must have the same shape."
            )
        return inits, targets

    def compute_targets_targ(self):
        """Axis targets, built as a product over qubits of per-qubit coverage.

        A qubit inside its selective band is driven to its target axis; a
        qubit outside it must be left where it started.  Inspecting only
        qubit 1's coverage applied (or withheld) the whole product target
        regardless of what the other qubits' bands were doing.
        """
        self.objective_mode = "state_transfer"
        n_snapshots = self.n_drift_snapshots()
        inits = []
        targets = []

        for r, init in enumerate(self.initial):
            for s in range(n_snapshots):
                ax = [
                    self.targ_ax[r][q]
                    if self._qubit_in_band(q, s)
                    else self.init_ax[r][q]
                    for q in range(self.n_qubits)
                ]
                inits.append(init)
                targets.append(self._state_from_ax(ax))

        return self._finalise_objective_arrays(inits, targets, len(self.initial))

    def compute_targets_gate(self):
        """Dispatch between the average gate objective and state transfer.

        When every configured initial state names the same canonical gate the
        request is for *that gate*, so the objective covers the whole
        computational subspace.  Scoring a gate only on the configured input
        states lets a pulse reach fidelity 1 while implementing a different
        operation on the states that were not scored.

        When the initial states name *different* gates the request is not a
        single coherent gate, so state-transfer scoring is retained.
        """
        gates = [canonical_gate_name(g) for g in self.gate]
        self.gate_canonical = gates
        if len(set(gates)) == 1:
            return self._compute_targets_gate_average(gates[0])
        return self._compute_targets_gate_state_transfer(gates)

    def _compute_targets_gate_average(self, gate_name):
        """Rows spanning the computational subspace for an average gate fidelity.

        In Hilbert space the rows are the ``d`` computational basis vectors,
        propagated as columns so that relative phases are preserved.  In
        Liouville space the rows are the ``d**2`` unnormalised Pauli strings
        (identity first), which the channel metric in
        :func:`~ctrl_freeq.ctrlfreeq.ctrl_freeq.fidelity_gate_liouville`
        contracts against ``G P_j G^dag``.
        """
        self.objective_mode = "gate"
        d = 2**self.n_qubits
        self.computational_dim = d
        self.gate_name = gate_name
        gate = np.asarray(self.get_gate(gate_name), dtype=complex)
        self.gate_matrix = gate
        identity = np.eye(d, dtype=complex)

        n_snapshots = self.n_drift_snapshots()
        inits = []
        targets = []

        if self.space == "hilbert":
            rows = [identity[:, r].copy() for r in range(d)]
            for row in rows:
                for s in range(n_snapshots):
                    g = gate if self._snapshot_all_in_band(s) else identity
                    inits.append(row)
                    targets.append(g @ row)
        else:
            rows = pauli_string_basis(self.n_qubits)
            for row in rows:
                for s in range(n_snapshots):
                    g = gate if self._snapshot_all_in_band(s) else identity
                    inits.append(row)
                    targets.append(g @ row @ g.conj().T)

        return self._finalise_objective_arrays(inits, targets, len(rows))

    def _compute_targets_gate_state_transfer(self, gates):
        """Per-initial-state gate targets scored as state transfer.

        This is *not* certification of a coherent conditional gate: it scores
        only the configured input states.
        """
        self.objective_mode = "state_transfer"
        d = 2**self.n_qubits
        identity = np.eye(d, dtype=complex)
        n_snapshots = self.n_drift_snapshots()
        inits = []
        targets = []

        for gate_name, init in zip(gates, self.initial):
            total_U = np.asarray(self.get_gate(gate_name), dtype=complex)
            for s in range(n_snapshots):
                u = total_U if self._snapshot_all_in_band(s) else identity
                inits.append(init)
                if self.space == "hilbert":
                    targets.append(u @ init)
                else:
                    targets.append(u @ init @ u.conj().T)

        return self._finalise_objective_arrays(inits, targets, len(self.initial))

    def get_betas(self):
        bs = []
        for beta in self.beta:
            bs_set = []
            for b, ex in zip(beta, self.excitation_profile):
                bs_set.append(b * ex)
            bs.append(bs_set)
        return bs

    def compute_u_tot_beta_axis(self):
        u_tot_for_offset_instances = []
        for offs in range(len(self.excitation_profile[0])):
            u_tot_for_init_instances = []
            for ax_sublist, bs_sublist in zip(self.axis, self.betas):
                u_tot = np.eye(2**self.n_qubits)
                for qubit_index, (ax_elem, bs_elem) in enumerate(
                    zip(ax_sublist, bs_sublist)
                ):
                    if ax_elem == "x":
                        uq = expm(-1j * bs_elem[offs] * self.op[f"X_{qubit_index + 1}"])
                    elif ax_elem == "-x":
                        uq = expm(1j * bs_elem[offs] * self.op[f"X_{qubit_index + 1}"])
                    elif ax_elem == "y":
                        uq = expm(-1j * bs_elem[offs] * self.op[f"Y_{qubit_index + 1}"])
                    elif ax_elem == "-y":
                        uq = expm(1j * bs_elem[offs] * self.op[f"Y_{qubit_index + 1}"])
                    elif ax_elem == "z":
                        uq = expm(-1j * bs_elem[offs] * self.op[f"Z_{qubit_index + 1}"])
                    elif ax_elem == "-z":
                        uq = expm(1j * bs_elem[offs] * self.op[f"Z_{qubit_index + 1}"])
                    u_tot = u_tot @ uq
                u_tot_for_init_instances.append(u_tot)
            u_tot_for_offset_instances.append(u_tot_for_init_instances)
        return u_tot_for_offset_instances

    def compute_targets_beta_axis(self):
        """Rotation-angle targets; smooth per-qubit coverage scaling is kept.

        ``self.u_tot`` is indexed ``[snapshot][row]``; the objective arrays are
        built row-major to match the drift stacking.
        """
        self.objective_mode = "state_transfer"
        n_snapshots = self.n_drift_snapshots()
        if len(self.u_tot) != n_snapshots:
            raise ValueError(
                f"Rotation targets were built for {len(self.u_tot)} snapshots "
                f"but the drift ensemble has {n_snapshots}."
            )

        inits = []
        targets = []
        for r, init in enumerate(self.initial):
            for s in range(n_snapshots):
                ut = self.u_tot[s][r]
                inits.append(init)
                if self.space == "hilbert":
                    targets.append(ut @ init)
                else:
                    targets.append(ut @ init @ ut.conj().T)

        return self._finalise_objective_arrays(inits, targets, len(self.initial))

    def build_collapse_operators(self):
        """
        Build Lindblad collapse operators for amplitude damping (T1) and
        pure dephasing (T2) for each qubit.

        For qubit i:
          - Amplitude damping: L1_i = sqrt(gamma1_i) * (sigma_minus_i ⊗ I_rest)
          - Pure dephasing:    L2_i = sqrt(gamma_phi_i / 2) * (sigma_z_i ⊗ I_rest)

        where gamma1 = 1/T1, gamma_phi = 1/T2 - 1/(2*T1).

        The dephasing operator is chosen so that the Lindblad dissipator
        D[L2]ρ = (gamma_phi/2)(σ_z ρ σ_z − ρ) yields off-diagonal decay
        dρ_01/dt = −gamma_phi · ρ_01, matching the physical T2 convention.

        Returns:
            np.ndarray of shape ``(n_ops, D, D)`` containing the collapse
            operators, or ``None`` if no dissipation channels are active.
        """
        identity = np.eye(2, dtype=complex)

        # sigma_minus = |0><1|
        sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
        # sigma_z
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

        def tensor_op(single_op, qubit_index):
            """Place single_op on qubit_index, identity on others."""
            ops = [identity] * self.n_qubits
            ops[qubit_index] = single_op
            result = ops[0]
            for op in ops[1:]:
                result = np.kron(result, op)
            return result

        collapse_ops = []
        for i in range(self.n_qubits):
            gamma1 = 1.0 / self.T1[i]
            gamma_phi = 1.0 / self.T2[i] - 1.0 / (2.0 * self.T1[i])

            # Amplitude damping operator
            if gamma1 > 0:
                L1 = np.sqrt(gamma1) * tensor_op(sigma_minus, i)
                collapse_ops.append(L1)

            # Pure dephasing operator:
            # L2 = sqrt(gamma_phi / 2) * sigma_z gives
            # D[L2]rho = (gamma_phi/2)(sigma_z rho sigma_z - rho),
            # i.e. off-diagonal decay rate = gamma_phi.
            if gamma_phi > 0:
                L2 = np.sqrt(gamma_phi / 2.0) * tensor_op(sigma_z, i)
                collapse_ops.append(L2)

        return np.array(collapse_ops) if collapse_ops else None

    def pulse_offset_exponent(self):
        exponent = []
        for pulse_offset in self.pulse_offset:
            e = np.exp(1j * pulse_offset * (self.t - self.pulse_duration / 2))
            exponent.append(e)
        exponent = np.array(exponent)
        return exponent.T
