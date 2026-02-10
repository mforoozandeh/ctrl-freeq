#!/usr/bin/env python3
"""
Test script to verify CtrlFreeQ optimizer now uses logger instead of print statements.
"""

import sys
import torch

sys.path.append("/")

from ctrl_freeq.ctrlfreeq.ctrl_freeq import CtrlFreeQ
from ctrl_freeq.conditions.stopping_conds import OptimizationInterrupted
from ctrl_freeq.utils.colored_logging import setup_colored_logging


def test_callback_logging():
    """Test that CtrlFreeQ callback function uses logger instead of print"""
    print("Testing CtrlFreeQ callback function logging...")
    print("=" * 60)

    # Create a minimal CtrlFreeQ instance for testing
    n_para = [10]
    n_qubits = 1
    op = (
        torch.eye(2).unsqueeze(0).repeat(3, 1, 1, 1)
    )  # Mock operator tensor (2x2 for 1 qubit)
    rabi_freq = torch.tensor([[1.0]])
    n_pulse = 50
    n_h0 = 1
    n_rabi = 1
    mat = [torch.eye(10)]
    H0 = torch.eye(2).unsqueeze(0)
    dt = torch.tensor(0.1)
    initial_state = torch.eye(2).unsqueeze(0)
    target_state = torch.eye(2).unsqueeze(0)

    # Mock waveform function
    def mock_wf_fun(params, mat):
        return (
            torch.ones(n_pulse, 1),
            torch.zeros(n_pulse, 1),
            torch.ones(n_pulse, 1),
            torch.zeros(n_pulse, 1),
        )

    wf_fun = [mock_wf_fun]

    # Mock functions
    def mock_u_fun(H, dt):
        # H has shape (batch_size, D, D), return same shape
        batch_size, D, _ = H.shape
        return torch.eye(D).unsqueeze(0).expand(batch_size, -1, -1)

    def mock_state_fun(U, state):
        return state

    def mock_fid_fun(state1, state2):
        return torch.tensor(0.85)

    targ_fid = 0.99
    me = torch.tensor([1.0])

    # Create CTRL instance
    ctrl = CtrlFreeQ(
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
        mock_u_fun,
        mock_state_fun,
        mock_fid_fun,
        targ_fid,
        me,
    )

    # Test callback function with mock parameters
    para = torch.randn(10, requires_grad=True)

    # First call objective function to set cost
    ctrl.objective_function(para)

    # Test callback function (should use logger, not print)
    print("Calling callback function - should see colored logger output:")
    try:
        for i in range(3):
            ctrl.callback_function(para)
            # Modify cost to simulate different iterations
            ctrl.cost = ctrl.cost - 0.01
    except OptimizationInterrupted as e:
        print(f"Caught expected OptimizationInterrupted: {e}")

    print("=" * 60)


def test_optimization_interrupted_logging():
    """Test OptimizationInterrupted exception logging"""
    print("\nTesting OptimizationInterrupted logging...")
    print("=" * 60)

    logger = setup_colored_logging(level="INFO")

    try:
        raise OptimizationInterrupted("Test optimization reached target fidelity", None)
    except OptimizationInterrupted as e:
        logger.warning(str(e))
        print("Exception logged successfully with WARNING level")

    print("=" * 60)


if __name__ == "__main__":
    test_callback_logging()
    test_optimization_interrupted_logging()
    print("\nCtrlFreeQ logging test completed!")
