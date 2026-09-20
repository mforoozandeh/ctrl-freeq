import torch
from torchmin import minimize

from ctrl_freeq.conditions.stopping_conds import OptimizationInterrupted
from ctrl_freeq.optimizers.qiskit_optimizers import (
    run_qiskit_optimization,
    get_supported_qiskit_optimizers,
)
from ctrl_freeq.utils.colored_logging import setup_colored_logging
from ctrl_freeq.make_pulse.waveform_gen_torch import (
    waveform_gen_polar_phase,
    waveform_gen_polar,
    waveform_gen_cart,
)


from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
    fidelity_hilbert,
    exp_mat_exact,
    exp_mat_torch,
    fidelity_liouville,
    fidelity_gate_hilbert,
    fidelity_gate_liouville,
    state_hilbert,
    state_liouville,
    state_lindblad,
    CtrlFreeQ,
    amplitude_limit_report,
    format_amplitude_limit_report,
    _as_float,
)
from ctrl_freeq.setup.iterator_generation.generate_iterator import (
    h0_omega_1_iterator_torch,
)
from ctrl_freeq.setup.operator_generation.generate_operators import (
    create_hamiltonian_basis_torch,
)
from ctrl_freeq.utils.conversion import array_to_tensor
from ctrl_freeq.utils.utility_functions import (
    convert_attributes_to_numpy,
    set_cores,
)
from ctrl_freeq.utils.device import select_device, resolve_cpu_cores


def build_fidelity_function(p, space, device):
    """Select the objective metric for this configuration.

    A gate objective scores the whole computational subspace (average gate
    fidelity in Hilbert space, the Pauli-transfer channel metric in Liouville
    space).  Everything else is state transfer on the configured inputs.
    """
    if getattr(p, "objective_mode", "state_transfer") != "gate":
        return fidelity_hilbert if space == "hilbert" else fidelity_liouville

    d = int(p.computational_dim)
    n_rows = int(p.n_objective_rows)

    if space == "hilbert":
        model = getattr(p, "hamiltonian_model", None)
        projector = None
        if model is not None and model.dim != d:
            projector = array_to_tensor(model.computational_projector(), device=device)

        def fid_fun(states, targets):
            return fidelity_gate_hilbert(states, targets, n_rows, d, projector)

    else:

        def fid_fun(states, targets):
            return fidelity_gate_liouville(states, targets, n_rows, d)

    return fid_fun


def run_ctrl(p):
    convert_attributes_to_numpy(p)

    # Initialize logger for optimization process
    logger = setup_colored_logging(level="INFO")

    # Determine device and CPU cores
    compute_resource = getattr(p, "compute_resource", "cpu")
    cpu_cores_requested = getattr(p, "cpu_cores", None)
    device, backend = select_device(compute_resource)

    # Apply CPU threads policy
    if device.type == "cpu":
        cores = resolve_cpu_cores(cpu_cores_requested)
        set_cores(cores)
    else:
        logger.info("Using CUDA device; CPU threads setting is not applied.")

    n_para = p.n_para_updated
    n_qubits = p.n_qubits
    n_pulse = p.np_pulse
    space = p.space
    wf_mode = p.wf_mode
    algorithm = p.algorithm
    max_iter = p.max_iter
    targ_fid = p.targ_fid

    rabi_freq = array_to_tensor(p.Omega_R, device=device)
    H0 = array_to_tensor(p.H0, device=device)
    initials = array_to_tensor(p.initials, device=device)
    targets = array_to_tensor(p.targets, device=device)
    mat = array_to_tensor(p.mat, device=device)
    x0_con = array_to_tensor(p.x0_con, device=device)
    dt = array_to_tensor(p.pulse_duration / p.np_pulse, device=device)

    me = array_to_tensor(p.modulation_exponent, device=device)

    n_h0 = H0.size(0)
    n_rabi = rabi_freq.size(0)

    H0, initials, targets = h0_omega_1_iterator_torch(H0, n_rabi, initials, targets)

    # Select matrix exponential function based on Hilbert space dimension
    hamiltonian_model = getattr(p, "hamiltonian_model", None)
    if hamiltonian_model is not None:
        D = hamiltonian_model.dim
    else:
        D = 2**n_qubits

    if D == 2:
        u_fun = exp_mat_exact
    else:
        u_fun = exp_mat_torch

    dissipation_mode = getattr(p, "dissipation_mode", "non-dissipative")

    if dissipation_mode == "dissipative":
        state_fun = state_lindblad
        collapse_ops = array_to_tensor(p.collapse_operators, device=device)
    elif space == "hilbert":
        state_fun = state_hilbert
        collapse_ops = None
    elif space == "liouville":
        state_fun = state_liouville
        collapse_ops = None

    fid_fun = build_fidelity_function(p, space, device)

    wf_fun = []

    for mode in wf_mode:
        if mode == "polar_phase":
            wf_fun.append(waveform_gen_polar_phase)
        elif mode == "polar":
            wf_fun.append(waveform_gen_polar)
        elif mode == "cart":
            wf_fun.append(waveform_gen_cart)

    # Build control operators from model (generic path) or Pauli basis (legacy)
    if hamiltonian_model is not None:
        control_ops = hamiltonian_model.control_ops_tensor(device=device)
        op = None  # Not needed for generic path
    else:
        control_ops = None

        op = create_hamiltonian_basis_torch(n_qubits, device=device)

    # This run's solution is a vector of basis coefficients.  Clearing any
    # representation recorded by a previous optimizer (e.g. a piecewise run on
    # the same parameters object) keeps analysis in step with the solution
    # this call returns; ``waveform_spec()`` then falls back to the configured
    # basis.
    p._waveform_spec = None

    ctrlfreeq_instance = CtrlFreeQ(
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
        initials,
        targets,
        wf_fun,
        u_fun,
        state_fun,
        fid_fun,
        targ_fid,
        me,
        collapse_ops=collapse_ops,
        hamiltonian_model=hamiltonian_model,
        control_ops=control_ops,
    )

    x0 = x0_con.requires_grad_(True)

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
        try:
            soln = minimize(
                ctrlfreeq_instance.objective_function,
                x0,
                method=algorithm,
                callback=ctrlfreeq_instance.callback_function,
                max_iter=max_iter,
            )

            sol = soln.x

        except OptimizationInterrupted as e:
            logger.warning(str(e))
            sol = e.solution

    else:
        # Check if it's a qiskit optimizer
        qiskit_optimizers = get_supported_qiskit_optimizers()
        if algorithm in qiskit_optimizers:
            # Handle all qiskit optimizers
            x0_no_grad = x0.detach().clone().requires_grad_(False)

            try:
                sol = run_qiskit_optimization(
                    optimizer_name=algorithm,
                    objective_func=ctrlfreeq_instance.objective_function,
                    x0=x0_no_grad,
                    callback=ctrlfreeq_instance.callback_function,
                    max_iter=max_iter,
                )

            except OptimizationInterrupted as e:
                logger.warning(str(e))
                sol = e.solution

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

            raise ValueError(
                f"Algorithm '{algorithm}' not supported. Supported algorithms: {', '.join(supported_algorithms)}"
            )

    # Store optimization tracking information in parameters object.
    # The final metrics are evaluated on the solution that is actually
    # returned: the optimiser's last objective evaluation is generally a
    # rejected trial point, not the solution.
    sol_tensor = sol if isinstance(sol, torch.Tensor) else array_to_tensor(sol)
    with torch.no_grad():
        ctrlfreeq_instance.objective_function(sol_tensor.detach())

    p.iterations = ctrlfreeq_instance.iter
    p.fidelity_history = ctrlfreeq_instance.fidelity_history
    p.penalty_history = ctrlfreeq_instance.penalty_history
    p.score_history = ctrlfreeq_instance.score_history

    p.final_fidelity = _as_float(ctrlfreeq_instance.fid)
    p.final_penalty = _as_float(ctrlfreeq_instance.pen)
    p.final_score = p.final_fidelity - p.final_penalty

    p.amplitude_report = amplitude_limit_report(
        ctrlfreeq_instance.last_cx,
        ctrlfreeq_instance.last_cy,
        p.Omega_R_max,
        rabi_samples=rabi_freq,
    )
    report_text = format_amplitude_limit_report(p.amplitude_report)
    if any(not entry["within_limit"] for entry in p.amplitude_report):
        logger.warning("Amplitude limit exceeded:\n%s", report_text)
    else:
        logger.info("Amplitude limit check:\n%s", report_text)

    return sol
