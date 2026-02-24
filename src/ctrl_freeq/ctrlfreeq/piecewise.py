import torch

from src.ctrl_freeq.conditions.stopping_conds import OptimizationInterrupted
from src.ctrl_freeq.optimizers.qiskit_optimizers import (
    run_qiskit_optimization,
    get_supported_qiskit_optimizers,
)
from src.ctrl_freeq.utils.colored_logging import setup_colored_logging
from src.ctrl_freeq.ctrlfreeq.ctrl_freeq import (
    pulse_hamiltonian,
    simulator_optimized,
    penalty,
    modulate_waveforms,
)
from src.ctrl_freeq.make_pulse.waveform_gen_torch import (
    waveform_gen_cart,
    waveform_gen_polar,
    waveform_gen_polar_phase,
)
from src.ctrl_freeq.make_pulse.waveform_gen_piecewise import (
    create_identity_basis_for_piecewise,
    get_piecewise_parameter_count,
)


class Piecewise:
    """
    Piecewise optimization variant of CtrlFreeQ.

    Unlike standard CtrlFreeQ which uses orthogonal basis functions to generate smooth
    waveforms from few parameters, piecewise optimization directly optimizes the value
    of each pulse segment, providing more local control but requiring more parameters.
    """

    def __init__(
        self,
        n_qubits,
        op,
        rabi_freq,
        n_pulse,
        n_h0,
        n_rabi,
        H0,
        dt,
        initial_state,
        target_state,
        wf_method,  # 'cart', 'polar', or 'polar_phase'
        u_fun,
        state_fun,
        fid_fun,
        targ_fid,
        me,
        amp_envelopes=None,
        fixed_amplitude=1.0,  # Only used for polar_phase method
        dtype=torch.float32,  # Add dtype parameter for consistency
        collapse_ops=None,
    ):
        self.n_qubits = n_qubits
        self.op = op
        self.rabi_freq = rabi_freq
        self.n_pulse = n_pulse
        self.n_h0 = n_h0
        self.n_rabi = n_rabi
        self.H0 = H0
        self.dt = dt
        self.initial_state = initial_state
        self.target_state = target_state
        self.wf_method = wf_method
        self.u_fun = u_fun
        self.state_fun = state_fun
        self.fid_fun = fid_fun
        self.iter = 0
        self.exit_val = targ_fid
        self.me = me
        self.fixed_amplitude = fixed_amplitude
        self.dtype = dtype
        self.collapse_ops = collapse_ops
        self.amp_envelopes = (
            amp_envelopes  # list of tensors [n_pulse,1] per qubit or None
        )

        # Set up waveform generation function based on method
        # Create identity basis matrices for piecewise optimization
        self.n_para_per_qubit = get_piecewise_parameter_count(n_pulse, wf_method)
        self.n_para = [self.n_para_per_qubit] * n_qubits

        # Create identity basis matrices for each qubit
        self.identity_basis = []
        for qi in range(n_qubits):
            if wf_method == "polar_phase":
                # For polar_phase, we need:
                # - mat[0]: shape (n_pulse, 1) for amplitude scaling (uses para[0])
                # - mat[1]: shape (n_pulse, n_pulse) for phase basis (uses para[1:])
                if self.amp_envelopes is not None and len(self.amp_envelopes) > qi:
                    amp_vec = self.amp_envelopes[qi]
                    # Ensure correct shape and dtype
                    if amp_vec.ndim == 1:
                        amp_vec = amp_vec.reshape(-1, 1)
                    amp_vec = amp_vec.to(dtype)
                else:
                    amp_vec = torch.ones(n_pulse, 1, dtype=dtype)

                basis = [
                    amp_vec,  # Amplitude envelope column used by waveform_gen_polar_phase
                    torch.eye(n_pulse, n_pulse, dtype=dtype),  # Phase basis
                ]
            else:
                # For cart and polar, we need two bases for cx/cy or amp/phase
                basis = create_identity_basis_for_piecewise(
                    n_pulse, n_pulse, dtype=dtype
                )
            self.identity_basis.append(basis)

        # Set up waveform generation function using existing CtrlFreeQ functions
        if wf_method == "cart":
            self.wf_fun = waveform_gen_cart
        elif wf_method == "polar":
            self.wf_fun = waveform_gen_polar
        elif wf_method == "polar_phase":
            self.wf_fun = waveform_gen_polar_phase
        else:
            raise ValueError(f"Unknown piecewise method: {wf_method}")

        self.fid = None
        self.pen = None
        self.fidelity_history = []  # Track fidelity per iteration

        # Flag for COBYLA early termination (avoids exception-based termination)
        self.early_termination_flag = False
        self.early_termination_solution = None

        # Initialize logger for optimization progress
        self.logger = setup_colored_logging(level="INFO")

    def objective_function(self, para):
        """Objective function for piecewise optimization."""
        # Split parameters for each qubit
        parameters = torch.split(para, self.n_para)

        # Generate pulse parameters using CtrlFreeQ waveform generators with identity basis
        amps, phis, cxs, cys = pulse_para_piecewise(
            self.n_qubits, parameters, self.identity_basis, self.wf_fun, self.me
        )

        # Calculate penalty for amplitude violations
        self.pen = penalty(amps)

        # Generate pulse Hamiltonian (reuse from CtrlFreeQ)
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

        # Simulate evolution (reuse optimized simulator from CtrlFreeQ)
        state = simulator_optimized(
            self.H0,
            Hp,
            self.dt,
            self.initial_state,
            self.u_fun,
            self.state_fun,
            collapse_ops=self.collapse_ops,
        )

        # Calculate fidelity and cost
        self.fid = self.fid_fun(state, self.target_state)
        self.cost = -self.fid + self.pen
        return self.cost

    def callback_function(self, para):
        """Callback function for monitoring optimization progress."""
        if self.iter == 0:
            self.logger.info("=" * 40)
            self.logger.info(f"PIECEWISE {self.wf_method.upper()} OPTIMIZATION")
            self.logger.info("=" * 40)
            self.logger.info(f"{'Iteration':<10} | {'Fidelity':<10} | {'Penalty':<10}")
            self.logger.info("=" * 40)

        self.iter += 1
        current_fidelity = -self.cost  # cost = -fidelity + penalty
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
        COBYLA-specific callback function for piecewise optimization that uses flag-based termination
        instead of raising exceptions to avoid COBYLA callback failures.
        """
        # Skip callback if early termination has already been triggered
        if self.early_termination_flag:
            return

        if self.iter == 0:
            self.logger.info("=" * 40)
            self.logger.info(f"PIECEWISE {self.wf_method.upper()} OPTIMIZATION")
            self.logger.info("=" * 40)
            self.logger.info(f"{'Iteration':<10} | {'Fidelity':<10} | {'Penalty':<10}")
            self.logger.info("=" * 40)

        self.iter += 1
        current_fidelity = -self.cost  # cost = -fidelity + penalty
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


def pulse_para_piecewise(n_qubits, parameters, identity_basis, wf_fun, me):
    """
    Generate pulse parameters using existing CtrlFreeQ waveform generators with identity basis.

    Args:
        n_qubits (int): Number of qubits.
        parameters (list): List of parameter tensors for each qubit.
        identity_basis (list): List of identity basis matrices for each qubit.
        wf_fun (callable): CtrlFreeQ waveform generation function.
        me (list): Modulation envelope functions.

    Returns:
        tuple: (amps, phis, cxs, cys) tensors of shape [n_pulse, n_qubits].
    """
    amps_list = []
    phis_list = []
    cxs_list = []
    cys_list = []

    for i in range(n_qubits):
        # Generate waveform for each qubit using CtrlFreeQ waveform generator with identity basis
        amp, phi, cx, cy = wf_fun(parameters[i], identity_basis[i])
        amps_list.append(amp)  # Each has shape [n_pulse, 1]
        phis_list.append(phi)
        cxs_list.append(cx)
        cys_list.append(cy)

    # Concatenate along dim=1 to get shape [n_pulse, n_qubits]
    amps = torch.cat(amps_list, dim=1)
    phis = torch.cat(phis_list, dim=1)
    cxs = torch.cat(cxs_list, dim=1)
    cys = torch.cat(cys_list, dim=1)

    # Apply modulation (reuse from CtrlFreeQ)
    cxs, cys = modulate_waveforms(cxs, cys, me)

    return amps, phis, cxs, cys


class PiecewiseAPI:
    """
    API for CtrlFreeQ Piecewise optimization that mirrors the CtrlFreeQAPI interface.
    """

    def __init__(self, config, method="cart"):
        """
        Initialize piecewise API from CtrlFreeQ configuration.

        Args:
            config: CtrlFreeQ API instance or configuration dict
            method: Piecewise method ('cart', 'polar', 'polar_phase')
        """
        from src.ctrl_freeq.api import CtrlFreeQAPI

        if isinstance(config, CtrlFreeQAPI):
            self.ctrlfreeq_api = config
            self.config = config.config
            self.parameters = config.parameters
        else:
            self.ctrlfreeq_api = CtrlFreeQAPI(config)
            self.config = self.ctrlfreeq_api.config
            self.parameters = self.ctrlfreeq_api.parameters

        self.method = method
        self._piecewise_instance = None

    def run_optimization(self):
        """Run piecewise optimization using CtrlFreeQ infrastructure."""
        from torchmin import minimize
        from src.ctrl_freeq.conditions.stopping_conds import OptimizationInterrupted
        from src.ctrl_freeq.setup.operator_generation.generate_operators import (
            create_hamiltonian_basis_torch,
        )
        from src.ctrl_freeq.setup.basis_generation.mat_x0_gen import (
            amplitude_envelope as np_amplitude_envelope,
        )
        from src.ctrl_freeq.ctrlfreeq.ctrl_freeq import (
            exp_mat_exact,
            exp_mat_torch,
            fidelity_hilbert,
            fidelity_liouville,
            state_hilbert,
            state_liouville,
            state_lindblad,
        )
        from src.ctrl_freeq.utils.conversion import array_to_tensor
        from src.ctrl_freeq.setup.iterator_generation.generate_iterator import (
            h0_omega_1_iterator_torch,
        )
        import torch
        import numpy as np

        # Use CtrlFreeQ's parameter processing
        p = self.parameters

        # Convert to tensors like CtrlFreeQ does
        rabi_freq = array_to_tensor(p.Omega_R)
        H0 = array_to_tensor(p.H0)
        initials = array_to_tensor(p.initials)
        targets = array_to_tensor(p.targets)
        dt = array_to_tensor(p.pulse_duration / p.np_pulse)
        me = array_to_tensor(p.modulation_exponent)

        n_h0 = H0.size(0)
        n_rabi = rabi_freq.size(0)

        H0, initials, targets = h0_omega_1_iterator_torch(H0, n_rabi, initials, targets)

        # Set up functions
        op = create_hamiltonian_basis_torch(p.n_qubits)
        u_fun = exp_mat_exact if p.n_qubits == 1 else exp_mat_torch

        dissipation_mode = getattr(p, "dissipation_mode", "non-dissipative")

        if dissipation_mode == "dissipative":
            fid_fun = fidelity_liouville
            state_fun = state_lindblad
            collapse_ops = array_to_tensor(p.collapse_operators)
        elif p.space == "hilbert":
            fid_fun = fidelity_hilbert
            state_fun = state_hilbert
            collapse_ops = None
        elif p.space == "liouville":
            fid_fun = fidelity_liouville
            state_fun = state_liouville
            collapse_ops = None

        # Create piecewise optimizer
        # Use float64 for qiskit-cobyla to avoid dtype mismatch, float32 for others
        dtype = torch.float64 if p.algorithm == "qiskit-cobyla" else torch.float32
        # Build amplitude envelopes (same as CtrlFreeQ) for polar_phase mode
        amp_env_tensors = None
        if self.method == "polar_phase":
            # p.amplitude_envelope: list of strings per qubit (e.g., 'gn', 'hs', 'quad')
            # p.amplitude_order: list of ints per qubit
            amp_env_tensors = []
            x = np.linspace(-1, 1, p.np_pulse)
            for qi in range(p.n_qubits):
                env_type = (
                    p.amplitude_envelope[qi]
                    if isinstance(p.amplitude_envelope, (list, tuple))
                    else p.amplitude_envelope
                )
                order = (
                    p.amplitude_order[qi]
                    if isinstance(p.amplitude_order, (list, tuple))
                    else p.amplitude_order
                )
                env_np = np_amplitude_envelope(
                    x, envelope=str(env_type), order=int(order)
                )
                env_t = torch.tensor(env_np, dtype=dtype).reshape(-1, 1)
                amp_env_tensors.append(env_t)

        self._piecewise_instance = Piecewise(
            n_qubits=p.n_qubits,
            op=op,
            rabi_freq=rabi_freq,
            n_pulse=p.np_pulse,
            n_h0=n_h0,
            n_rabi=n_rabi,
            H0=H0,
            dt=dt,
            initial_state=initials,
            target_state=targets,
            wf_method=self.method,
            u_fun=u_fun,
            state_fun=state_fun,
            fid_fun=fid_fun,
            targ_fid=p.targ_fid,
            me=me,
            amp_envelopes=amp_env_tensors,
            dtype=dtype,
            collapse_ops=collapse_ops,
        )

        # Initialize parameters with same dtype as piecewise instance
        total_params = sum(self._piecewise_instance.n_para)
        x0 = torch.randn(total_params, dtype=dtype, requires_grad=True) * 0.1

        # Run optimization based on algorithm
        algorithm = p.algorithm

        if algorithm in [
            "bfgs",
            "l-bfgs",
            "cg",
            "newton-cg",
            "newton-exact",
            "dogleg",
            "trust-ncg",
            "trust-krylov",
            "trust-exact",
        ]:
            # Gradient-based optimizers using torchmin
            try:
                soln = minimize(
                    self._piecewise_instance.objective_function,
                    x0,
                    method=algorithm,
                    callback=self._piecewise_instance.callback_function,
                    max_iter=p.max_iter,
                )
                return soln.x
            except OptimizationInterrupted as e:
                return e.solution

        else:
            # Get supported Qiskit optimizers for error message
            qiskit_optimizers = get_supported_qiskit_optimizers()
            supported_algorithms = [
                "bfgs",
                "l-bfgs",
                "cg",
                "newton-cg",
                "newton-exact",
                "dogleg",
                "trust-ncg",
                "trust-krylov",
                "trust-exact",
            ] + qiskit_optimizers

            # Check if it's a qiskit optimizer
            if algorithm in qiskit_optimizers:
                # Qiskit optimizers are derivative-free
                x0_no_grad = x0.detach().clone().requires_grad_(False)

                try:
                    sol = run_qiskit_optimization(
                        optimizer_name=algorithm,
                        objective_func=self._piecewise_instance.objective_function,
                        x0=x0_no_grad,
                        callback=self._piecewise_instance.callback_function,
                        max_iter=p.max_iter,
                    )
                    return sol

                except OptimizationInterrupted as e:
                    return e.solution
            else:
                raise ValueError(
                    f"Algorithm '{algorithm}' not supported. Supported algorithms: {', '.join(supported_algorithms)}"
                )

    def get_config_summary(self):
        """Get configuration summary with piecewise method info."""
        summary = self.ctrlfreeq_api.get_config_summary()
        summary += f"\nPiecewise method: {self.method}"
        if self._piecewise_instance:
            summary += f"\nPiecewise parameters: {sum(self._piecewise_instance.n_para)}"
        return summary

    def update_parameter(self, parameter_path: str, value):
        """Update parameter using underlying CtrlFreeQ API."""
        self.ctrlfreeq_api.update_parameter(parameter_path, value)
        self.parameters = self.ctrlfreeq_api.parameters


def create_piecewise_api(config=None, method="cart"):
    """
    Factory function to create a piecewise CtrlFreeQ API.

    Args:
        config: Configuration (if None, uses single_qubit_config)
        method (str): Piecewise method ('cart', 'polar', or 'polar_phase')

    Returns:
        PiecewiseAPI: Configured piecewise API.
    """
    if config is None:
        from src.ctrl_freeq.api import load_single_qubit_config

        config = load_single_qubit_config()

    return PiecewiseAPI(config, method)
