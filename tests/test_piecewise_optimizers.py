#!/usr/bin/env python3
"""
Test script to verify COBYLA and SPSA optimizers work with CtrlFreeQ Piecewise.
This script tests the new derivative-free optimizers by running piecewise optimizations.
"""

import pytest
from ctrl_freeq.ctrlfreeq.piecewise import create_piecewise_api


@pytest.mark.parametrize(
    "algorithm_name,method",
    [
        ("newton-cg", "cart"),
        ("qiskit-cobyla", "cart"),
        ("qiskit-spsa", "cart"),
        ("newton-cg", "polar"),
        ("qiskit-cobyla", "polar"),
        ("qiskit-spsa", "polar"),
    ],
)
def test_piecewise_optimizer(algorithm_name, method):
    """Test a specific optimizer with piecewise method."""
    print(f"\n{'=' * 60}")
    print(f"Testing {algorithm_name.upper()} optimizer with Piecewise {method.upper()}")
    print(f"{'=' * 60}")

    # Create piecewise API
    piecewise_api = create_piecewise_api(method=method)

    # Update to use the specific algorithm
    piecewise_api.update_parameter("optimization.algorithm", algorithm_name)

    # Reduce iterations for quick testing
    piecewise_api.update_parameter("optimization.max_iter", 30)
    piecewise_api.update_parameter("optimization.targ_fid", 0.8)

    print("Configuration summary:")
    print(piecewise_api.get_config_summary())

    # Run optimization
    print(f"\nRunning piecewise optimization with {algorithm_name}...")
    solution = piecewise_api.run_optimization()

    print(f"✅ Success! Solution shape: {solution.shape}")
    print(f"Final solution (first 10 parameters): {solution[:10].detach().numpy()}")

    # Access optimization tracking information
    if hasattr(piecewise_api.parameters, "final_fidelity"):
        print(f"Final fidelity: {piecewise_api.parameters.final_fidelity}")
    if hasattr(piecewise_api.parameters, "iterations"):
        print(f"Iterations completed: {piecewise_api.parameters.iterations}")

    # Assertions for pytest
    assert solution is not None
    assert solution.shape[0] > 0


def test_all_piecewise_optimizers():
    """Test all optimizers with piecewise methods."""
    optimizers_to_test = [
        "newton-cg",  # Reference gradient-based optimizer
        "qiskit-cobyla",  # Qiskit derivative-free optimizer
        "qiskit-spsa",  # Qiskit SPSA optimizer (replaces legacy SPSA)
    ]

    methods_to_test = ["cart", "polar"]  # Test different piecewise methods

    results = {}

    print("Testing CtrlFreeQ Piecewise with derivative-free optimizers")
    print("=" * 70)

    for method in methods_to_test:
        results[method] = {}
        for optimizer in optimizers_to_test:
            try:
                test_piecewise_optimizer(optimizer, method)
                results[method][optimizer] = True
            except Exception as e:
                results[method][optimizer] = False
                print(f"❌ {optimizer} with {method}: FAILED - {str(e)}")

    print(f"\n{'=' * 70}")
    print("SUMMARY:")
    print(f"{'=' * 70}")

    for method in methods_to_test:
        print(f"\nMethod: {method.upper()}")
        for optimizer, success in results[method].items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {optimizer:<15}: {status}")

    # Check if all tests passed and use assertion instead of return
    all_passed = all(all(results[method].values()) for method in methods_to_test)
    print(
        f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
    )

    # Use assertion instead of return for pytest compatibility
    assert all_passed, (
        f"Some optimizer tests failed: {[(method, optimizer) for method in methods_to_test for optimizer, success in results[method].items() if not success]}"
    )


if __name__ == "__main__":
    test_all_piecewise_optimizers()
