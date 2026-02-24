from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ctrl_freeq.setup.hamiltonian_generation.hamiltonians import createHcs, createHJ
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
from ctrl_freeq.utils.utility_functions import generate_instances


@dataclass
class Initialise:
    def __init__(self, data):
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
            data["parameters"]["coupling_type"] if self.n_qubits > 1 else None
        )
        self.sw = 2 * np.pi * data["parameters"]["sw"]
        self.Delta = 2 * np.pi * data["parameters"]["Delta"]
        self.sigma_Delta = 2 * np.pi * data["parameters"]["sigma_Delta"]
        self.sigma_J = (
            2 * np.pi * data["parameters"]["sigma_J"] if self.n_qubits > 1 else None
        )
        self.Jmat = 2 * np.pi * data["parameters"]["J"]
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
            self.T1 = np.array(data["parameters"]["T1"])
            self.T2 = np.array(data["parameters"]["T2"])
            # Validate T2 <= 2*T1 (physical constraint)
            for i in range(self.n_qubits):
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

    def generate_time_sequence(self):
        return np.linspace(np.finfo(float).eps, self.pulse_duration, self.np_pulse)

    def generate_initial_x0(self):
        self.x0 = []
        for i in range(self.n_qubits):
            self.x0.append(np.random.uniform(-1, 1, size=self.n_para[i]))
        return self.x0

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
        Generate a 4x4 matrix for a specified two-qubit gate ('CNOT', 'CZ', 'SWAP').

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

        else:
            raise ValueError(
                "Unsupported operation. Please specify 'CNOT', 'CZ', 'SWAP', or 'iSWAP'."
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

    def get_offset(self):
        offset_results = []

        if all(cov == "single" for cov in self.coverage) and all(
            sig == 0 for sig in self.sigma_Delta
        ):
            for om in self.Delta:
                offset = [om]
                offset_results.append(offset)

        else:
            for coverage, om, sig, sw, fb, rf in zip(
                self.coverage,
                self.Delta,
                self.sigma_Delta,
                self.sw,
                self.frq_band,
                self.ratio_factor,
            ):
                if coverage == "broadband":
                    offset = np.random.uniform(
                        om - sw / 2, om + sw / 2, self.H0_snapshots
                    )

                elif coverage == "band_selective":
                    num_outside_band = np.round(self.H0_snapshots * rf)
                    if num_outside_band % 2 == 0:
                        num_outside_band = int(num_outside_band)
                    else:
                        num_outside_band = (
                            int(num_outside_band) + 1
                            if num_outside_band % 2 == 1
                            else int(num_outside_band) - 1
                        )

                    num_in_band = int(self.H0_snapshots - num_outside_band)

                    left_sw = [om - sw / 2, fb[0]]
                    right_sw = [fb[1], om + sw / 2]

                    offs_in_left_band = np.random.uniform(
                        left_sw[0], left_sw[1], int(num_outside_band / 2)
                    )
                    offs_in_right_band = np.random.uniform(
                        right_sw[0], right_sw[1], int(num_outside_band / 2)
                    )
                    offs_in_middle_band = np.random.uniform(fb[0], fb[1], num_in_band)

                    offset = np.concatenate(
                        [offs_in_left_band, offs_in_middle_band, offs_in_right_band]
                    )

                elif coverage == "single":
                    offset = np.random.normal(om, sig, self.H0_snapshots)

                elif coverage == "selective":
                    num_outside_band = np.round(self.H0_snapshots * rf)
                    if num_outside_band % 2 == 0:
                        num_outside_band = int(num_outside_band)
                    else:
                        num_outside_band = (
                            int(num_outside_band) + 1
                            if num_outside_band % 2 == 1
                            else int(num_outside_band) - 1
                        )

                    num_in_band = int(self.H0_snapshots - num_outside_band)

                    left_sw = [om - sw / 2, fb[0]]
                    right_sw = [fb[1], om + sw / 2]

                    offs_in_left_band = np.random.uniform(
                        left_sw[0], left_sw[1], int(num_outside_band / 2)
                    )
                    offs_in_right_band = np.random.uniform(
                        right_sw[0], right_sw[1], int(num_outside_band / 2)
                    )
                    offs_in_middle_band = np.random.normal(om, sig, num_in_band)

                    offset = np.concatenate(
                        [offs_in_left_band, offs_in_middle_band, offs_in_right_band]
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

        H0_stacked = []
        for _ in range(len(self.initial)):
            H0_stacked.extend(H0)

        return H0_stacked

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
            if cov == "broadband":
                profile = np.ones(self.H0_snapshots)
            elif cov == "single":
                profile = np.ones(len(self.offs[0]))
            elif cov == "selective":
                profile = np.where(
                    (offs >= om - pbw / 2) & (offs <= om + pbw / 2), 1, 0
                )
            elif cov == "band_selective":
                sigma = pbw / (2 * np.sqrt(2 * np.log(2)))
                profile = np.exp(-(((offs - om) / sigma) ** (2 * ord)))
            else:
                raise ValueError(f"Unknown coverage type: {cov}")

            profiles.append(profile)

        return profiles

    def generate_Omega_instances(self):
        return [list(group) for group in zip(*self.offs)]

    def generate_Jmat_instances(self):
        Jmat_instances = []

        for _ in range(self.H0_snapshots):
            Jmat_instance = np.array(
                [
                    [
                        np.random.normal(elm, self.sigma_J) if elm != 0 else 0
                        for elm in row
                    ]
                    for row in self.Jmat
                ]
            )
            Jmat_instances.append(Jmat_instance)
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

    def compute_targets_targ(self):
        targets = []
        inits = []

        M = len(self.excitation_profile[0])  # Number of offsets

        for offs in range(M):
            target_set = []
            init_set = []
            for target, init in zip(self.target, self.initial):
                if self.excitation_profile[0][offs] == 1:
                    target_set.append(target)
                else:
                    target_set.append(init)
                init_set.append(init)
            targets.append(target_set)
            inits.append(init_set)

        inits = np.array(inits)
        targets = np.array(targets)
        if self.space == "hilbert":
            (N, P, Q) = inits.shape
            inits = inits.reshape(N * P, Q)
            targets = targets.reshape(N * P, Q)
        elif self.space == "liouville":
            (N, P, Q, R) = inits.shape
            inits = inits.reshape(N * P, Q, R)
            targets = targets.reshape(N * P, Q, R)

        return inits, targets

    def compute_targets_gate(self):
        targets = []
        inits = []

        M = len(self.excitation_profile[0])  # Number of offsets

        for offs in range(M):
            target_set = []
            init_set = []
            for gate, init in zip(self.gate, self.initial):
                total_U = self.get_gate(gate)
                if self.space == "hilbert":
                    if self.excitation_profile[0][offs] == 1:
                        target_set.append(total_U @ init)
                    else:
                        target_set.append(init)
                elif self.space == "liouville":
                    if self.excitation_profile[0][offs] == 1:
                        target_set.append(total_U @ init @ total_U.conj().T)
                    else:
                        target_set.append(init)
                init_set.append(init)
            targets.append(target_set)
            inits.append(init_set)

        inits = np.array(inits)
        targets = np.array(targets)
        if self.space == "hilbert":
            (N, P, Q) = inits.shape
            inits = inits.reshape(N * P, Q)
            targets = targets.reshape(N * P, Q)
        elif self.space == "liouville":
            (N, P, Q, R) = inits.shape
            inits = inits.reshape(N * P, Q, R)
            targets = targets.reshape(N * P, Q, R)

        return inits, targets

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
        targets = []
        inits = []
        for uts in self.u_tot:
            target_set = []
            init_set = []
            for ut, init in zip(uts, self.initial):
                if self.space == "hilbert":
                    target_set.append(ut @ init)
                elif self.space == "liouville":
                    target_set.append(ut @ init @ ut.conj().T)
                init_set.append(init)
            targets.append(target_set)
            inits.append(init_set)

        inits = np.array(inits)
        targets = np.array(targets)
        if self.space == "hilbert":
            (N, P, Q) = inits.shape
            inits = inits.reshape(N * P, Q)
            targets = targets.reshape(N * P, Q)
        elif self.space == "liouville":
            (N, P, Q, R) = inits.shape
            inits = inits.reshape(N * P, Q, R)
            targets = targets.reshape(N * P, Q, R)

        return inits, targets

    def build_collapse_operators(self):
        """
        Build Lindblad collapse operators for amplitude damping (T1) and
        pure dephasing (T2) for each qubit.

        For qubit i:
          - Amplitude damping: L1_i = sqrt(gamma1_i) * (sigma_minus_i ⊗ I_rest)
          - Pure dephasing:    L2_i = sqrt(gamma_phi_i) * (sigma_z_i/2 ⊗ I_rest)

        where gamma1 = 1/T1, gamma_phi = 1/T2 - 1/(2*T1).

        Returns:
            list of tuples (L, L_dag, L_dag_L): each collapse operator with
            its pre-computed adjoint and L†L product.
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

            # Pure dephasing operator
            if gamma_phi > 0:
                L2 = np.sqrt(gamma_phi) * tensor_op(0.5 * sigma_z, i)
                collapse_ops.append(L2)

        return np.array(collapse_ops) if collapse_ops else None

    def pulse_offset_exponent(self):
        exponent = []
        for pulse_offset in self.pulse_offset:
            e = np.exp(1j * pulse_offset * (self.t - self.pulse_duration / 2))
            exponent.append(e)
        exponent = np.array(exponent)
        return exponent.T
