#!/usr/bin/env python3
"""
Test script to verify COBYLA and SPSA optimizers work with CtrlFreeQ.
This script tests the new derivative-free optimizers by running a simple single qubit optimization.
"""

import pytest
from ctrl_freeq.api import load_single_qubit_config


@pytest.mark.parametrize(
    "algorithm_name", ["newton-cg", "qiskit-cobyla", "qiskit-spsa"]
)
def test_optimizer(algorithm_name):
    """Test a specific optimizer with a simple single qubit problem."""
    print(f"\n{'=' * 50}")
    print(f"Testing {algorithm_name.upper()} optimizer")
    print(f"{'=' * 50}")

    # Load default single qubit configuration
    api = load_single_qubit_config()

    # Update to use the specific algorithm
    api.update_parameter("optimization.algorithm", algorithm_name)

    # Reduce iterations for quick testing
    api.update_parameter("optimization.max_iter", 50)
    api.update_parameter("optimization.targ_fid", 0.8)

    print("Configuration summary:")
    print(api.get_config_summary())

    # Run optimization
    print(f"\nRunning optimization with {algorithm_name}...")
    solution = api.run_optimization()

    print(f"✅ Success! Solution shape: {solution.shape}")
    print(f"Final solution (first 10 parameters): {solution[:10].detach().numpy()}")
    print(f"Final fidelity: {api.parameters.final_fidelity}")
    print(f"Iterations completed: {api.parameters.iterations}")

    # Assertions for pytest
    assert solution is not None
    assert solution.shape[0] > 0
    assert api.parameters.final_fidelity >= 0.0


def test_all_optimizers():
    """Test all optimizers including the new ones."""
    optimizers_to_test = [
        "newton-cg",  # Reference optimizer
        "qiskit-cobyla",  # Qiskit derivative-free optimizer
        "qiskit-spsa",  # New derivative-free optimizer
    ]

    results = {}

    print("Testing CtrlFreeQ with new derivative-free optimizers")
    print("=" * 60)

    for optimizer in optimizers_to_test:
        try:
            # Load default single qubit configuration
            api = load_single_qubit_config()
            api.update_parameter("optimization.algorithm", optimizer)
            api.update_parameter("optimization.max_iter", 50)
            api.update_parameter("optimization.targ_fid", 0.8)

            api.run_optimization()
            results[optimizer] = True
            print(f"✅ {optimizer}: PASSED")
        except Exception as e:
            results[optimizer] = False
            print(f"❌ {optimizer}: FAILED - {str(e)}")

    all_passed = all(results.values())
    assert all_passed, (
        f"Some optimizers failed: {[k for k, v in results.items() if not v]}"
    )


if __name__ == "__main__":
    test_all_optimizers()
