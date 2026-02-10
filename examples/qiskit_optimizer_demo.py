"""
Demo script showing how to use Qiskit optimizers with CtrlFreeQ.

This script demonstrates the integration of Qiskit's SPSA optimizer
with the CtrlFreeQ quantum control framework.

Note: Requires compatible versions of qiskit and qiskit-algorithms packages.
"""

from src.ctrl_freeq.api import load_single_qubit_config


def demo_qiskit_spsa_optimizer():
    """
    Demonstrate using Qiskit SPSA optimizer for quantum control optimization.

    This example shows how to:
    1. Load a standard single qubit configuration
    2. Configure the system to use Qiskit's SPSA optimizer
    3. Run the optimization
    4. Analyze the results
    """

    print("=== Qiskit SPSA Optimizer Demo ===")
    print()

    # Load default single qubit configuration
    print("1. Loading single qubit configuration...")
    api = load_single_qubit_config()

    # Configure for Qiskit SPSA optimizer
    print("2. Configuring Qiskit SPSA optimizer...")
    api.update_parameter("optimization.algorithm", "qiskit-spsa")
    api.update_parameter("optimization.max_iter", 100)
    api.update_parameter("optimization.targ_fid", 0.95)

    # Display configuration summary
    # Display configuration summary
    print("3. Configuration summary:")
    print(api.get_config_summary())
    print()

    try:
        # Run optimization
        print("4. Running optimization with Qiskit SPSA...")
        result = api.run_optimization()

        # Display results
        print("5. Results:")
        print(f"   Final fidelity: {api.parameters.final_fidelity:.6f}")
        print(f"   Iterations completed: {api.parameters.iterations}")
        print(f"   Optimization successful: {api.parameters.final_fidelity > 0.9}")

        # Show fidelity evolution
        if (
            hasattr(api.parameters, "fidelity_history")
            and api.parameters.fidelity_history
        ):
            print(f"   Initial fidelity: {api.parameters.fidelity_history[0]:.6f}")
            print(f"   Final fidelity: {api.parameters.fidelity_history[-1]:.6f}")
            improvement = (
                api.parameters.fidelity_history[-1] - api.parameters.fidelity_history[0]
            )
            print(f"   Improvement: {improvement:.6f}")

        print("\n✅ Qiskit SPSA optimization completed successfully!")

        return result

    except ImportError as e:
        print("❌ Import error - Qiskit optimizers not available:")
        print(f"   {str(e)}")
        print("\n📝 To use Qiskit optimizers:")
        print("   1. Install compatible versions: pip install qiskit qiskit-algorithms")
        print("   2. Ensure version compatibility between qiskit and qiskit-algorithms")
        print("   3. Re-run this demo")
        return None

    except Exception as e:
        print(f"❌ Optimization failed: {str(e)}")
        return None


def compare_optimizers():
    """
    Compare Qiskit SPSA with built-in SPSA optimizer.

    This demonstrates that both optimizers can be used interchangeably
    with the same API.
    """

    print("\n=== Optimizer Comparison Demo ===")
    print()

    optimizers_to_test = [("spsa", "Built-in SPSA"), ("qiskit-spsa", "Qiskit SPSA")]

    results = {}

    for optimizer_key, optimizer_name in optimizers_to_test:
        print(f"Testing {optimizer_name}...")

        try:
            # Load configuration
            api = load_single_qubit_config()
            api.update_parameter("optimization.algorithm", optimizer_key)
            api.update_parameter("optimization.max_iter", 50)
            api.update_parameter("optimization.targ_fid", 0.9)

            # Run optimization
            result = api.run_optimization()

            results[optimizer_name] = {
                "final_fidelity": api.parameters.final_fidelity,
                "iterations": api.parameters.iterations,
                "success": True,
            }

            print(
                f"   ✅ {optimizer_name}: Fidelity = {api.parameters.final_fidelity:.4f}"
            )

        except Exception as e:
            results[optimizer_name] = {"error": str(e), "success": False}
            print(f"   ❌ {optimizer_name}: Failed - {str(e)}")

    # Summary
    print("\n📊 Comparison Summary:")
    for optimizer_name, result in results.items():
        if result["success"]:
            print(
                f"   {optimizer_name:<15}: Fidelity = {result['final_fidelity']:.4f}, "
                f"Iterations = {result['iterations']}"
            )
        else:
            print(f"   {optimizer_name:<15}: Failed")

    return results


def main():
    """
    Main demo function showcasing Qiskit optimizer integration.
    """

    print("🔬 CtrlFreeQ Qiskit Optimizer Integration Demo")
    print("=" * 50)

    # Demo 1: Basic Qiskit SPSA usage
    demo_qiskit_spsa_optimizer()

    # Demo 2: Optimizer comparison
    compare_optimizers()

    print("\n" + "=" * 50)
    print("🎉 Demo completed!")
    print("\n📖 Usage Notes:")
    print("   • Use 'qiskit-spsa' as the algorithm parameter")
    print("   • All existing CtrlFreeQ features work with Qiskit optimizers")
    print("   • Qiskit optimizers are derivative-free like built-in SPSA")
    print("   • Compatible with all CtrlFreeQ problem types (single/multi-qubit, etc.)")


if __name__ == "__main__":
    main()
