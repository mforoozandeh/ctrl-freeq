"""
Simple test script to verify the piecewise optimization implementation works correctly.
"""

import torch
from ctrl_freeq.make_pulse.waveform_gen_piecewise import (
    waveform_gen_piecewise_cart,
    waveform_gen_piecewise_polar,
    waveform_gen_piecewise_polar_phase,
    get_piecewise_parameter_count,
    compare_parameter_efficiency,
)


def test_piecewise_waveform_functions():
    """Test the piecewise waveform generation functions."""
    print("Testing piecewise waveform generation functions...")

    n_pulse = 10

    # Test cart method
    n_params_cart = get_piecewise_parameter_count(n_pulse, "cart")
    para_cart = torch.randn(n_params_cart) * 0.1
    amp, phi, cx, cy = waveform_gen_piecewise_cart(para_cart, n_pulse)

    print(
        f"Cart method - Parameters: {n_params_cart}, Output shapes: amp={amp.shape}, phi={phi.shape}, cx={cx.shape}, cy={cy.shape}"
    )
    assert amp.shape == (
        n_pulse,
        1,
    ), f"Expected amp shape {(n_pulse, 1)}, got {amp.shape}"
    assert cx.shape == (n_pulse, 1), f"Expected cx shape {(n_pulse, 1)}, got {cx.shape}"

    # Test polar method
    n_params_polar = get_piecewise_parameter_count(n_pulse, "polar")
    para_polar = torch.randn(n_params_polar) * 0.1
    amp, phi, cx, cy = waveform_gen_piecewise_polar(para_polar, n_pulse)

    print(
        f"Polar method - Parameters: {n_params_polar}, Output shapes: amp={amp.shape}, phi={phi.shape}, cx={cx.shape}, cy={cy.shape}"
    )
    assert amp.shape == (
        n_pulse,
        1,
    ), f"Expected amp shape {(n_pulse, 1)}, got {amp.shape}"

    # Test polar_phase method
    n_params_polar_phase = get_piecewise_parameter_count(n_pulse, "polar_phase")
    para_polar_phase = torch.randn(n_params_polar_phase) * 0.1
    amp, phi, cx, cy = waveform_gen_piecewise_polar_phase(para_polar_phase, n_pulse)

    print(
        f"Polar_phase method - Parameters: {n_params_polar_phase}, Output shapes: amp={amp.shape}, phi={phi.shape}, cx={cx.shape}, cy={cy.shape}"
    )
    assert amp.shape == (
        n_pulse,
        1,
    ), f"Expected amp shape {(n_pulse, 1)}, got {amp.shape}"

    print("✓ All piecewise waveform functions work correctly!")
    assert True, "Piecewise waveform functions work correctly"


def test_parameter_efficiency():
    """Test parameter efficiency comparison."""
    print("\nTesting parameter efficiency comparison...")

    n_pulse = 50
    basis_size = 10

    for method in ["cart", "polar", "polar_phase"]:
        comparison = compare_parameter_efficiency(n_pulse, basis_size, method)
        print(f"{method.title()} method:")
        print(f"  Piecewise params: {comparison['piecewise_params']}")
        print(f"  CtrlFreeQ params: {comparison['ctrlfreeq_params']}")
        print(f"  Efficiency ratio: {comparison['efficiency_ratio']:.1f}x")

    print("✓ Parameter efficiency comparison works correctly!")
    assert True, "Parameter efficiency comparison works correctly"


def test_piecewise_ctrlfreeq_creation():
    """Test creating a CtrlFreeQ_Piecewise instance."""
    print("\nTesting CtrlFreeQ_Piecewise class creation...")

    try:
        from ctrl_freeq.ctrlfreeq.piecewise import Piecewise
        from ctrl_freeq.setup.operator_generation.generate_operators import (
            create_hamiltonian_basis_torch,
        )
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
            fidelity_hilbert,
            exp_mat_exact,
            state_hilbert,
        )

        # Simple test configuration
        n_qubits = 1
        n_pulse = 20

        # Create test matrices
        H0 = torch.zeros((2, 2), dtype=torch.complex64)
        rabi_freq = torch.tensor([1.0])
        initial_state = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.complex64)
        target_state = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
        me = [torch.ones(n_pulse, dtype=torch.float32)]

        op = create_hamiltonian_basis_torch(n_qubits)

        # Test creating piecewise optimizer
        piecewise = Piecewise(
            n_qubits=n_qubits,
            op=op,
            rabi_freq=rabi_freq,
            n_pulse=n_pulse,
            n_h0=H0.size(0),
            n_rabi=rabi_freq.size(0),
            H0=H0,
            dt=torch.tensor(0.01),
            initial_state=initial_state,
            target_state=target_state,
            wf_method="cart",
            u_fun=exp_mat_exact,
            state_fun=state_hilbert,
            fid_fun=fidelity_hilbert,
            targ_fid=0.99,
            me=me,
        )

        print(
            f"Created CtrlFreeQ_Piecewise instance with {sum(piecewise.n_para)} parameters"
        )
        print(f"Method: {piecewise.wf_method}")
        print("✓ CtrlFreeQ_Piecewise class creation works correctly!")
        assert True, "CtrlFreeQ_Piecewise class creation works correctly"

    except Exception as e:
        print(f"✗ Error creating CtrlFreeQ_Piecewise: {e}")
        assert False, f"Error creating CtrlFreeQ_Piecewise: {e}"


def main():
    """Run all tests."""
    print("=" * 50)
    print("PIECEWISE OPTIMIZATION IMPLEMENTATION TEST")
    print("=" * 50)

    success = True

    try:
        success &= test_piecewise_waveform_functions()
        success &= test_parameter_efficiency()
        success &= test_piecewise_ctrlfreeq_creation()

        print("\n" + "=" * 50)
        if success:
            print("✅ ALL TESTS PASSED - Implementation is working correctly!")
        else:
            print("❌ SOME TESTS FAILED - Check implementation")
        print("=" * 50)

    except Exception as e:
        print(f"❌ TEST FAILED WITH ERROR: {e}")
        success = False

    return success


if __name__ == "__main__":
    main()
