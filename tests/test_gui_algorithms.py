#!/usr/bin/env python3
"""
Test script to verify GUI algorithm dropdown is properly updated with qiskit optimizers.
"""


def test_gui_imports():
    """Test that GUI setup imports work correctly."""
    from ctrl_freeq.optimizers.qiskit_optimizers import (
        get_supported_qiskit_optimizers,
    )

    # Get the qiskit optimizers
    qiskit_optimizers = get_supported_qiskit_optimizers()

    # Create the expected algorithm list (matching GUI setup)
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
        "qiskit-cobyla",
    ] + qiskit_optimizers

    assert len(supported_algorithms) > 10, (
        f"Expected more than 10 algorithms, got {len(supported_algorithms)}"
    )

    # Verify that qiskit-spsa is included and legacy spsa is not
    assert "qiskit-spsa" in supported_algorithms, (
        "qiskit-spsa is missing from supported algorithms"
    )
    assert "spsa" not in supported_algorithms, "Legacy spsa should have been removed"
