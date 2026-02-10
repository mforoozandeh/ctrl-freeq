import torch
import time
import numpy as np
import unittest

# Import CtrlFreeQ components
from src.ctrl_freeq.ctrlfreeq.ctrl_freeq import (
    CtrlFreeQ,
    simulator,
    exp_mat_torch,
    state_hilbert,
    fidelity_hilbert,
)
from src.ctrl_freeq.setup.operator_generation.generate_operators import (
    create_hamiltonian_basis_torch,
)
from src.ctrl_freeq.make_pulse.waveform_gen_torch import waveform_gen_polar_phase


def create_complete_test_setup():
    """Create a complete tests setup for end-to-end benchmarking"""
    n_qubits = 2
    n_pulse = 100
    n_h0 = 5
    n_rabi = 3
    batch_size = n_h0 * n_rabi

    # Create tests parameters with consistent dtypes
    op = create_hamiltonian_basis_torch(n_qubits)
    rabi_freq = torch.randn(n_rabi, n_qubits, dtype=torch.float64)
    H0 = torch.randn(batch_size, 2**n_qubits, 2**n_qubits, dtype=torch.complex128)
    dt = torch.tensor(0.01, dtype=torch.float64)

    # Create initial and target states
    initial_state = torch.randn(batch_size, 2**n_qubits, dtype=torch.complex128)
    target_state = torch.randn(batch_size, 2**n_qubits, dtype=torch.complex128)

    # Normalize states
    initial_state = initial_state / torch.norm(initial_state, dim=1, keepdim=True)
    target_state = target_state / torch.norm(target_state, dim=1, keepdim=True)

    # Create waveform parameters
    n_para = [10, 10]  # Parameters for each qubit
    mat = []
    for i in range(n_qubits):
        mat_amp = torch.randn(n_pulse, 1, dtype=torch.float64)
        mat_phase = torch.randn(n_pulse, n_para[i] - 1, dtype=torch.float64)
        mat.append([mat_amp, mat_phase])
    wf_fun = [waveform_gen_polar_phase, waveform_gen_polar_phase]

    # Create modulation exponent
    me = torch.ones(n_pulse, n_qubits, dtype=torch.complex128)

    # Create tests parameters vector
    para = torch.randn(sum(n_para), requires_grad=True, dtype=torch.float64)

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


class CtrlFreeQ_Original(CtrlFreeQ):
    """CtrlFreeQ class using the original simulator for comparison with optimized simulator"""

    def objective_function(self, para):
        parameters = torch.split(para, list(self.n_para))
        from src.ctrl_freeq.ctrlfreeq.ctrl_freeq import (
            pulse_para,
            pulse_hamiltonian,
            penalty,
        )

        amps, phis, cxs, cys = pulse_para(
            self.n_qubits, parameters, self.mat, self.wf_fun, self.me
        )
        self.pen = penalty(amps)
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
        # Use original simulator (main difference from optimized version)
        state = simulator(
            self.H0, Hp, self.dt, self.initial_state, self.u_fun, self.state_fun
        )
        self.fid = self.fid_fun(state, self.target_state)
        self.cost = -self.fid + self.pen
        return self.cost


class TestCompleteOptimization(unittest.TestCase):
    """Test class for CtrlFreeQ simulator optimization benchmarks"""

    def test_complete_optimization_benchmark(self):
        """Test and benchmark the CtrlFreeQ simulator optimization (original vs optimized simulator)"""
        print("Creating complete optimization benchmark setup...")
        setup = create_complete_test_setup()

        print("Setup details:")
        print(f"  n_qubits: {setup['n_qubits']}")
        print(f"  n_pulse: {setup['n_pulse']}")
        print(f"  n_h0: {setup['n_h0']}")
        print(f"  n_rabi: {setup['n_rabi']}")
        print(f"  batch_size: {setup['n_h0'] * setup['n_rabi']}")
        print(f"  parameter vector size: {setup['para'].shape[0]}")

        # Create CtrlFreeQ instances
        ctrlfreeq_original = CtrlFreeQ_Original(
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

        ctrlfreeq_optimized = CtrlFreeQ(
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

        # Warm up both versions
        print("\nWarming up...")
        for _ in range(3):
            _ = ctrlfreeq_original.objective_function(setup["para"])
            _ = ctrlfreeq_optimized.objective_function(setup["para"])

        # Benchmark original CtrlFreeQ
        print("\nBenchmarking original CtrlFreeQ (with original simulator)...")
        n_runs = 20
        times_original = []

        for i in range(n_runs):
            start_time = time.time()
            cost_original = ctrlfreeq_original.objective_function(setup["para"])
            end_time = time.time()
            times_original.append(end_time - start_time)
            print(f"  Run {i + 1}: {times_original[-1]:.4f} seconds")

        # Benchmark optimized CtrlFreeQ
        print("\nBenchmarking optimized CtrlFreeQ (with optimized simulator)...")
        times_optimized = []

        for i in range(n_runs):
            start_time = time.time()
            cost_optimized = ctrlfreeq_optimized.objective_function(setup["para"])
            end_time = time.time()
            times_optimized.append(end_time - start_time)
            print(f"  Run {i + 1}: {times_optimized[-1]:.4f} seconds")

        # Check correctness
        print("\nChecking correctness...")
        cost_diff = abs(cost_original.item() - cost_optimized.item())
        print(f"Cost difference: {cost_diff:.2e}")

        # Use unittest assertions to validate correctness
        self.assertLess(
            cost_diff, 1e-6, f"Results differ significantly: {cost_diff:.2e}"
        )
        self.assertIsNotNone(cost_original, "Original cost should not be None")
        self.assertIsNotNone(cost_optimized, "Optimized cost should not be None")

        if cost_diff < 1e-10:
            print("✓ Results are numerically identical")
        elif cost_diff < 1e-6:
            print("✓ Results are very close (acceptable numerical difference)")
        else:
            print("⚠ Results differ significantly - check implementation")
            print(f"Original cost: {cost_original.item()}")
            print(f"Optimized cost: {cost_optimized.item()}")

        # Calculate statistics
        avg_original = np.mean(times_original)
        std_original = np.std(times_original)
        avg_optimized = np.mean(times_optimized)
        std_optimized = np.std(times_optimized)

        speedup = avg_original / avg_optimized

        # Additional unittest assertions to validate benchmark results
        self.assertGreater(avg_original, 0, "Average original time should be positive")
        self.assertGreater(
            avg_optimized, 0, "Average optimized time should be positive"
        )
        self.assertGreater(speedup, 0, "Speedup should be positive")
        self.assertEqual(
            len(times_original), n_runs, f"Should have {n_runs} original timing results"
        )
        self.assertEqual(
            len(times_optimized),
            n_runs,
            f"Should have {n_runs} optimized timing results",
        )

        print(f"\n{'=' * 60}")
        print("SIMULATOR OPTIMIZATION BENCHMARK RESULTS")
        print(f"{'=' * 60}")
        print("CtrlFreeQ with original simulator:")
        print(f"  Average time: {avg_original:.4f} ± {std_original:.4f} seconds")
        print("CtrlFreeQ with optimized simulator:")
        print(f"  Average time: {avg_optimized:.4f} ± {std_optimized:.4f} seconds")
        print(f"\nSimulator optimization speedup: {speedup:.2f}x")
        print(
            f"Time reduction from simulator optimization: {(1 - 1 / speedup) * 100:.1f}%"
        )

    def test_optimization_scaling(self):
        """Test how the optimization scales with different problem sizes"""
        print(f"\n{'=' * 60}")
        print("TESTING OPTIMIZATION SCALING")
        print(f"{'=' * 60}")

        test_configs = [
            (1, 50, 3, 2),  # Small: 1 qubit, 50 pulses
            (2, 100, 5, 3),  # Medium: 2 qubits, 100 pulses
            (2, 200, 5, 3),  # Large: 2 qubits, 200 pulses
        ]

        for n_qubits, n_pulse, n_h0, n_rabi in test_configs:
            print(
                f"\nTesting: {n_qubits} qubits, {n_pulse} pulses, {n_h0} H0, {n_rabi} rabi"
            )

            # Create custom setup for this configuration
            batch_size = n_h0 * n_rabi
            op = create_hamiltonian_basis_torch(n_qubits)
            rabi_freq = torch.randn(n_rabi, n_qubits, dtype=torch.float64)
            H0 = torch.randn(
                batch_size, 2**n_qubits, 2**n_qubits, dtype=torch.complex128
            )
            dt = torch.tensor(0.01, dtype=torch.float64)

            initial_state = torch.randn(batch_size, 2**n_qubits, dtype=torch.complex128)
            target_state = torch.randn(batch_size, 2**n_qubits, dtype=torch.complex128)
            initial_state = initial_state / torch.norm(
                initial_state, dim=1, keepdim=True
            )
            target_state = target_state / torch.norm(target_state, dim=1, keepdim=True)

            n_para = [10] * n_qubits
            mat = []
            for i in range(n_qubits):
                mat_amp = torch.randn(n_pulse, 1, dtype=torch.float64)
                mat_phase = torch.randn(n_pulse, n_para[i] - 1, dtype=torch.float64)
                mat.append([mat_amp, mat_phase])
            wf_fun = [waveform_gen_polar_phase] * n_qubits
            me = torch.ones(n_pulse, n_qubits, dtype=torch.complex128)
            para = torch.randn(sum(n_para), requires_grad=True, dtype=torch.float64)

            # Create instances
            ctrlfreeq_orig = CtrlFreeQ_Original(
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
                exp_mat_torch,
                state_hilbert,
                fidelity_hilbert,
                0.99,
                me,
            )

            ctrlfreeq_opt = CtrlFreeQ(
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
                exp_mat_torch,
                state_hilbert,
                fidelity_hilbert,
                0.99,
                me,
            )

            # Time both versions
            n_runs = 5

            # Original
            times_orig = []
            for _ in range(n_runs):
                start = time.time()
                _ = ctrlfreeq_orig.objective_function(para)
                times_orig.append(time.time() - start)

            # Optimized
            times_opt = []
            for _ in range(n_runs):
                start = time.time()
                _ = ctrlfreeq_opt.objective_function(para)
                times_opt.append(time.time() - start)

            avg_orig = np.mean(times_orig)
            avg_opt = np.mean(times_opt)
            speedup = avg_orig / avg_opt

            # Add unittest assertions to validate scaling results
            self.assertGreater(
                avg_orig,
                0,
                f"Average original time should be positive for config {n_qubits} qubits, {n_pulse} pulses",
            )
            self.assertGreater(
                avg_opt,
                0,
                f"Average optimized time should be positive for config {n_qubits} qubits, {n_pulse} pulses",
            )
            self.assertGreater(
                speedup,
                0,
                f"Speedup should be positive for config {n_qubits} qubits, {n_pulse} pulses",
            )
            self.assertEqual(
                len(times_orig), n_runs, f"Should have {n_runs} original timing results"
            )
            self.assertEqual(
                len(times_opt), n_runs, f"Should have {n_runs} optimized timing results"
            )

            print(
                f"  Original: {avg_orig:.4f}s, Optimized: {avg_opt:.4f}s, Speedup: {speedup:.2f}x"
            )


if __name__ == "__main__":
    print("CtrlFreeQ Simulator Optimization Benchmark")
    print("=" * 60)

    # Check PyTorch version and device
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("CUDA not available, using CPU")
    print()

    # Run the unittest tests
    unittest.main(verbosity=2)
