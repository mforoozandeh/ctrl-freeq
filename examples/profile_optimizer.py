import torch
import time
import cProfile
import pstats
from io import StringIO

# Import the CtrlFreeQ components
from src.ctrl_freeq.ctrlfreeq.ctrl_freeq import (
    CtrlFreeQ,
    pulse_para,
    pulse_hamiltonian,
    simulator,
    fidelity_hilbert,
    exp_mat_torch,
    state_hilbert,
)
from src.ctrl_freeq.setup.operator_generation.generate_operators import (
    create_hamiltonian_basis_torch,
)
from src.ctrl_freeq.make_pulse.waveform_gen_torch import waveform_gen_polar_phase


def create_test_setup():
    """Create a tests setup for profiling"""
    n_qubits = 2
    n_pulse = 100
    n_h0 = 5
    n_rabi = 3

    # Create tests parameters
    op = create_hamiltonian_basis_torch(n_qubits)
    rabi_freq = torch.randn(n_rabi, n_qubits, dtype=torch.float64)
    H0 = torch.randn(n_h0 * n_rabi, 2**n_qubits, 2**n_qubits, dtype=torch.complex128)
    dt = torch.tensor(0.01, dtype=torch.float64)

    # Create initial and target states
    initial_state = torch.randn(n_h0 * n_rabi, 2**n_qubits, dtype=torch.complex128)
    target_state = torch.randn(n_h0 * n_rabi, 2**n_qubits, dtype=torch.complex128)

    # Normalize states
    initial_state = initial_state / torch.norm(initial_state, dim=1, keepdim=True)
    target_state = target_state / torch.norm(target_state, dim=1, keepdim=True)

    # Create waveform parameters
    n_para = [10, 10]  # Parameters for each qubit
    # For polar_phase: mat[i] should be a list of [mat_amp, mat_phase]
    # mat_amp: shape (n_pulse, 1) for amplitude, mat_phase: shape (n_pulse, n_para-1) for phase
    mat = []
    for i in range(n_qubits):
        mat_amp = torch.randn(n_pulse, 1)  # For amplitude calculation
        mat_phase = torch.randn(n_pulse, n_para[i] - 1)  # For phase calculation
        mat.append([mat_amp, mat_phase])
    wf_fun = [waveform_gen_polar_phase, waveform_gen_polar_phase]

    # Create modulation exponent
    me = torch.ones(n_pulse, n_qubits, dtype=torch.complex64)

    # Create tests parameters vector
    para = torch.randn(sum(n_para), requires_grad=True)

    return {
        "n_para": n_para,
        "n_qubits": n_qubits,
        "op": op,
        "rabi_freq": rabi_freq,
        "n_pulse": n_pulse,
        "n_h0": n_h0,
        "n_rabi": n_rabi,
        "mat": mat,
        "H0": H0,
        "dt": dt,
        "initial_state": initial_state,
        "target_state": target_state,
        "wf_fun": wf_fun,
        "me": me,
        "para": para,
    }


def profile_objective_function():
    """Profile the objective function to identify bottlenecks"""
    setup = create_test_setup()

    # Create CtrlFreeQ instance
    ctrlfreeq = CtrlFreeQ(
        setup["n_para"],
        setup["n_qubits"],
        setup["op"],
        setup["rabi_freq"],
        setup["n_pulse"],
        setup["n_h0"],
        setup["n_rabi"],
        setup["mat"],
        setup["H0"],
        setup["dt"],
        setup["initial_state"],
        setup["target_state"],
        setup["wf_fun"],
        exp_mat_torch,
        state_hilbert,
        fidelity_hilbert,
        0.99,
        setup["me"],
    )

    # Warm up
    for _ in range(5):
        ctrlfreeq.objective_function(setup["para"])

    # Profile the objective function
    pr = cProfile.Profile()
    pr.enable()

    # Run multiple iterations for better statistics
    start_time = time.time()
    for _ in range(50):
        _ = ctrlfreeq.objective_function(setup["para"])
    end_time = time.time()

    pr.disable()

    # Print timing results
    print(
        f"Average time per objective function call: {(end_time - start_time) / 50:.4f} seconds"
    )

    # Print profiling results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # Top 20 functions
    print("\nProfiling Results:")
    print(s.getvalue())


def profile_individual_functions():
    """Profile individual functions to identify specific bottlenecks"""
    setup = create_test_setup()

    # Test pulse_para function
    parameters = torch.split(setup["para"], list(setup["n_para"]))

    print("Profiling individual functions:")

    # Profile pulse_para
    start_time = time.time()
    for _ in range(100):
        amps, phis, cxs, cys = pulse_para(
            setup["n_qubits"], parameters, setup["mat"], setup["wf_fun"], setup["me"]
        )
    end_time = time.time()
    print(f"pulse_para: {(end_time - start_time) / 100:.4f} seconds per call")

    # Profile pulse_hamiltonian
    start_time = time.time()
    for _ in range(100):
        Hp = pulse_hamiltonian(
            cxs,
            cys,
            setup["rabi_freq"],
            setup["op"],
            setup["n_pulse"],
            setup["n_h0"],
            setup["n_rabi"],
            setup["n_qubits"],
        )
    end_time = time.time()
    print(f"pulse_hamiltonian: {(end_time - start_time) / 100:.4f} seconds per call")

    # Profile simulator
    start_time = time.time()
    for _ in range(100):
        state = simulator(
            setup["H0"],
            Hp,
            setup["dt"],
            setup["initial_state"],
            exp_mat_torch,
            state_hilbert,
        )
    end_time = time.time()
    print(f"simulator: {(end_time - start_time) / 100:.4f} seconds per call")

    # Profile fidelity calculation
    start_time = time.time()
    for _ in range(100):
        _ = fidelity_hilbert(state, setup["target_state"])
    end_time = time.time()
    print(f"fidelity_hilbert: {(end_time - start_time) / 100:.4f} seconds per call")


if __name__ == "__main__":
    print("Starting CtrlFreeQ optimizer profiling...")
    print("=" * 50)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    print(f"PyTorch version: {torch.__version__}")
    print("=" * 50)

    # Profile individual functions first
    profile_individual_functions()
    print("\n" + "=" * 50)

    # Profile the complete objective function
    profile_objective_function()
