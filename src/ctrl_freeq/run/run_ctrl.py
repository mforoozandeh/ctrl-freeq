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
    state_hilbert,
    state_liouville,
    state_lindblad,
    CtrlFreeQ,
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
    op = create_hamiltonian_basis_torch(n_qubits, device=device)
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

    if n_qubits == 1:
        u_fun = exp_mat_exact
    else:
        u_fun = exp_mat_torch

    dissipation_mode = getattr(p, "dissipation_mode", "non-dissipative")

    if dissipation_mode == "dissipative":
        fid_fun = fidelity_liouville
        state_fun = state_lindblad
        collapse_ops = array_to_tensor(p.collapse_operators, device=device)
    elif space == "hilbert":
        fid_fun = fidelity_hilbert
        state_fun = state_hilbert
        collapse_ops = None
    elif space == "liouville":
        fid_fun = fidelity_liouville
        state_fun = state_liouville
        collapse_ops = None

    wf_fun = []

    for mode in wf_mode:
        if mode == "polar_phase":
            wf_fun.append(waveform_gen_polar_phase)
        elif mode == "polar":
            wf_fun.append(waveform_gen_polar)
        elif mode == "cart":
            wf_fun.append(waveform_gen_cart)

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

    # Store optimization tracking information in parameters object
    p.iterations = ctrlfreeq_instance.iter
    p.fidelity_history = ctrlfreeq_instance.fidelity_history
    p.final_fidelity = (
        ctrlfreeq_instance.fid.item()
        if hasattr(ctrlfreeq_instance.fid, "item")
        else float(ctrlfreeq_instance.fid)
    )

    return sol
