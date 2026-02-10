import os

# Enable MPS fallback before importing PyTorch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import time
import numpy as np

# Import both simulator versions
from src.ctrl_freeq.ctrlfreeq.ctrl_freeq import (
    simulator,
    simulator_optimized,
    exp_mat_torch,
    state_hilbert,
)


def create_benchmark_setup(device="cpu"):
    """Create a benchmark setup for testing simulator performance"""
    n_qubits = 2
    n_pulse = 100
    n_h0 = 100
    n_rabi = 1
    batch_size = n_h0 * n_rabi

    # MPS (Apple Silicon GPU) only supports float32, others can use float64
    if device == "mps":
        float_dtype = torch.float32
        complex_dtype = torch.complex64
    else:
        float_dtype = torch.float64
        complex_dtype = torch.complex128

    # Create tests parameters with device-appropriate dtypes
    _ = torch.randn(n_rabi, n_qubits, dtype=float_dtype, device=device)
    H0 = torch.randn(
        batch_size, 2**n_qubits, 2**n_qubits, dtype=complex_dtype, device=device
    )
    Hp = torch.randn(
        n_pulse,
        batch_size,
        2**n_qubits,
        2**n_qubits,
        dtype=complex_dtype,
        device=device,
    )
    dt = torch.tensor(0.01, dtype=float_dtype, device=device)

    # Create initial state
    initial_state = torch.randn(
        batch_size, 2**n_qubits, dtype=complex_dtype, device=device
    )

    # Normalize initial state (MPS doesn't support torch.norm with complex tensors)
    if device == "mps":
        # Manual normalization for MPS: sqrt(sum(|z|^2)) = sqrt(sum(real^2 + imag^2))
        norm_squared = torch.sum(
            initial_state.real**2 + initial_state.imag**2, dim=1, keepdim=True
        )
        norm = torch.sqrt(norm_squared)
        initial_state = initial_state / norm
    else:
        initial_state = initial_state / torch.norm(initial_state, dim=1, keepdim=True)

    return H0, Hp, dt, initial_state


def benchmark_simulators(device="cpu"):
    """Benchmark both simulator versions on specified device"""
    print(f"Creating benchmark setup on {device}...")
    H0, Hp, dt, initial_state = create_benchmark_setup(device)

    print("Setup details:")
    print(f"  Device: {device}")
    print(f"  H0 shape: {H0.shape}")
    print(f"  Hp shape: {Hp.shape}")
    print(f"  Initial state shape: {initial_state.shape}")
    print(f"  dt: {dt}")

    # Warm up both functions
    print("\nWarming up...")
    for _ in range(3):
        _ = simulator(H0, Hp, dt, initial_state, exp_mat_torch, state_hilbert)
        _ = simulator_optimized(H0, Hp, dt, initial_state, exp_mat_torch, state_hilbert)

    # Benchmark original simulator
    print("\nBenchmarking original simulator...")
    n_runs = 10
    times_original = []

    for i in range(n_runs):
        start_time = time.time()
        result_original = simulator(
            H0, Hp, dt, initial_state, exp_mat_torch, state_hilbert
        )
        end_time = time.time()
        times_original.append(end_time - start_time)
        print(f"  Run {i + 1}: {times_original[-1]:.4f} seconds")

    # Benchmark optimized simulator
    print("\nBenchmarking optimized simulator...")
    times_optimized = []

    for i in range(n_runs):
        start_time = time.time()
        result_optimized = simulator_optimized(
            H0, Hp, dt, initial_state, exp_mat_torch, state_hilbert
        )
        end_time = time.time()
        times_optimized.append(end_time - start_time)
        print(f"  Run {i + 1}: {times_optimized[-1]:.4f} seconds")

    # Check correctness
    print("\nChecking correctness...")
    max_diff = torch.max(torch.abs(result_original - result_optimized)).item()
    print(f"Maximum difference between results: {max_diff:.2e}")

    if max_diff < 1e-10:
        print("✓ Results are numerically identical")
    elif max_diff < 1e-6:
        print("✓ Results are very close (acceptable numerical difference)")
    else:
        print("⚠ Results differ significantly - check implementation")

    # Calculate statistics
    avg_original = np.mean(times_original)
    std_original = np.std(times_original)
    avg_optimized = np.mean(times_optimized)
    std_optimized = np.std(times_optimized)

    speedup = avg_original / avg_optimized

    print(f"\n{'=' * 60}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 60}")
    print("Original simulator:")
    print(f"  Average time: {avg_original:.4f} ± {std_original:.4f} seconds")
    print("Optimized simulator:")
    print(f"  Average time: {avg_optimized:.4f} ± {std_optimized:.4f} seconds")
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Time reduction: {(1 - 1 / speedup) * 100:.1f}%")

    return avg_original, avg_optimized, speedup


def profile_matrix_exp_calls(device="cpu"):
    """Profile the number of matrix_exp calls in both versions"""
    print(f"\n{'=' * 60}")
    print(f"PROFILING MATRIX EXPONENTIAL CALLS ON {device.upper()}")
    print(f"{'=' * 60}")

    H0, Hp, dt, initial_state = create_benchmark_setup(device)
    n_pulse, batch_size = Hp.shape[0], Hp.shape[1]

    print(f"Setup: {n_pulse} pulses, {batch_size} batch size")
    print(f"Original simulator: {n_pulse} separate matrix_exp calls")
    print(
        f"Optimized simulator: 1 batched matrix_exp call with {n_pulse * batch_size} matrices"
    )

    # Count actual calls by monkey-patching
    original_matrix_exp = torch.linalg.matrix_exp
    call_count = 0

    def counting_matrix_exp(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_matrix_exp(*args, **kwargs)

    # Test original simulator
    torch.linalg.matrix_exp = counting_matrix_exp
    call_count = 0
    _ = simulator(H0, Hp, dt, initial_state, exp_mat_torch, state_hilbert)
    original_calls = call_count

    # Test optimized simulator
    call_count = 0
    _ = simulator_optimized(H0, Hp, dt, initial_state, exp_mat_torch, state_hilbert)
    optimized_calls = call_count

    # Restore original function
    torch.linalg.matrix_exp = original_matrix_exp

    print("\nActual matrix_exp calls:")
    print(f"  Original simulator: {original_calls} calls")
    print(f"  Optimized simulator: {optimized_calls} calls")
    print(
        f"  Call reduction: {original_calls - optimized_calls} calls ({(1 - optimized_calls / original_calls) * 100:.1f}%)"
    )


def get_available_devices():
    """Get list of available devices for benchmarking"""
    devices = ["cpu"]
    device_info = {"cpu": "CPU"}

    # Check for CUDA
    if torch.cuda.is_available():
        devices.append("cuda")
        device_info["cuda"] = f"CUDA ({torch.cuda.get_device_name()})"

    # Check for MPS (Metal Performance Shaders) - Apple Silicon GPU
    if torch.backends.mps.is_available():
        devices.append("mps")
        device_info["mps"] = "MPS (Apple Silicon GPU)"

    return devices, device_info


def compare_devices_performance(results):
    """Compare performance across different devices"""
    print(f"\n{'=' * 80}")
    print("CROSS-DEVICE PERFORMANCE COMPARISON")
    print(f"{'=' * 80}")

    if len(results) < 2:
        print("Need at least 2 devices for comparison")
        return

    # Find best performing device for each simulator type
    best_original = min(results.items(), key=lambda x: x[1]["avg_original"])
    best_optimized = min(results.items(), key=lambda x: x[1]["avg_optimized"])

    print(f"Best Original Simulator Performance: {best_original[0].upper()}")
    print(f"  Time: {best_original[1]['avg_original']:.4f} seconds")

    print(f"Best Optimized Simulator Performance: {best_optimized[0].upper()}")
    print(f"  Time: {best_optimized[1]['avg_optimized']:.4f} seconds")

    # Compare GPU vs CPU if both available
    if "cpu" in results and ("cuda" in results or "mps" in results):
        gpu_device = "cuda" if "cuda" in results else "mps"
        cpu_orig = results["cpu"]["avg_original"]
        gpu_orig = results[gpu_device]["avg_original"]
        cpu_opt = results["cpu"]["avg_optimized"]
        gpu_opt = results[gpu_device]["avg_optimized"]

        print("\nGPU vs CPU Comparison:")
        print(
            f"  Original Simulator - {gpu_device.upper()} vs CPU: {cpu_orig / gpu_orig:.2f}x faster"
        )
        print(
            f"  Optimized Simulator - {gpu_device.upper()} vs CPU: {cpu_opt / gpu_opt:.2f}x faster"
        )


if __name__ == "__main__":
    print("CtrlFreeQ Simulator Optimization Benchmark with GPU Support")
    print("=" * 80)

    # Check PyTorch version and available devices
    print(f"PyTorch version: {torch.__version__}")

    devices, device_info = get_available_devices()
    print("\nAvailable devices:")
    for device in devices:
        print(f"  - {device.upper()}: {device_info[device]}")

    # Show MPS fallback warning if MPS is available
    if "mps" in devices:
        print("\n⚠️  MPS Fallback Enabled:")
        print("   The matrix exponential operation (linalg_matrix_exp) is not natively")
        print(
            "   supported on MPS. CPU fallback is enabled for unsupported operations."
        )
        print("   This may result in slower performance than pure CPU execution.")

    # Run benchmarks on all available devices
    results = {}

    for device in devices:
        print(f"\n{'=' * 80}")
        print(f"BENCHMARKING ON {device.upper()}")
        print(f"{'=' * 80}")

        try:
            avg_original, avg_optimized, speedup = benchmark_simulators(device)
            results[device] = {
                "avg_original": avg_original,
                "avg_optimized": avg_optimized,
                "speedup": speedup,
            }

            # Profile matrix exponential calls for this device
            profile_matrix_exp_calls(device)

        except Exception as e:
            print(f"Error benchmarking on {device}: {e}")
            continue

    # Compare performance across devices
    if len(results) > 1:
        compare_devices_performance(results)

    # Final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")

    for device, result in results.items():
        print(f"{device.upper()}:")
        print(f"  Speedup: {result['speedup']:.2f}x")
        print(f"  Time reduction: {(1 - 1 / result['speedup']) * 100:.1f}%")
        print(
            f"  Original: {result['avg_original']:.4f}s, Optimized: {result['avg_optimized']:.4f}s"
        )

    if results:
        best_device = min(results.items(), key=lambda x: x[1]["avg_optimized"])
        print(
            f"\nBest overall performance: {best_device[0].upper()} with {best_device[1]['avg_optimized']:.4f}s"
        )
