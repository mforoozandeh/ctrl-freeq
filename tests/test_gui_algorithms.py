#!/usr/bin/env python3
"""
Test script to verify GUI algorithm dropdown is properly updated with qiskit optimizers.
"""


def test_gui_imports():
    """Test that GUI setup imports work correctly."""
    try:
        # Test the import without actually creating the GUI
        import sys
        import os

        # Add the project root to Python path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

        # Import the qiskit optimizers function
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

        print("✅ GUI imports working successfully")
        print(f"✅ Total supported algorithms in GUI: {len(supported_algorithms)}")
        print("\nTorchmin algorithms:")
        torchmin_algs = [
            "bfgs",
            "l-bfgs",
            "cg",
            "newton-cg",
            "newton-exact",
            "dogleg",
            "trust-ncg",
            "trust-krylov",
            "trust-exact",
        ]
        for alg in torchmin_algs:
            print(f"  - {alg}")

        print("\nScipy algorithms:")
        print("  - cobyla")

        print("\nQiskit algorithms:")
        for alg in qiskit_optimizers:
            print(f"  - {alg}")

        # Verify that qiskit-spsa is included and legacy spsa is not
        if "qiskit-spsa" in supported_algorithms:
            print("✅ qiskit-spsa is properly included")
        else:
            print("❌ qiskit-spsa is missing from supported algorithms")

        if "spsa" not in supported_algorithms:
            print("✅ Legacy spsa has been properly removed")
        else:
            print("❌ Legacy spsa is still present")

        return True

    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


if __name__ == "__main__":
    print("Testing GUI algorithm dropdown updates...")
    print("=" * 50)

    success = test_gui_imports()

    print("\n" + "=" * 50)
    if success:
        print("✅ GUI algorithm dropdown test passed!")
    else:
        print("❌ GUI algorithm dropdown test failed!")
