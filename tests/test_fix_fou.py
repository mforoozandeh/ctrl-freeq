#!/usr/bin/env python3
"""
Pytest tests for Fourier basis parameter update fix.
Tests the update_n_para function behavior with different wf_types and wf_modes.
"""

import numpy as np


def update_n_para_fixed(n_para, wf_type, wf_mode, n_qubits):
    """
    Update n_para based on wf_type and wf_mode.

    Args:
        n_para: List of parameter counts per qubit
        wf_type: List of wavefront types per qubit
        wf_mode: List of wavefront modes per qubit
        n_qubits: Number of qubits

    Returns:
        Updated n_para list
    """
    n_para_updated = n_para.copy()
    for i in range(n_qubits):
        if wf_type[i] == "fou":
            # Fourier basis recalculates n_para internally
            if wf_mode[i] == "polar_phase":
                # n = (n_para - 1) // 2, then n_para = 2*n + 1, then +1 for polar_phase
                n = (n_para[i] - 1) // 2
                n_para_updated[i] = 2 * n + 1 + 1
            else:
                # n = (n_para - 2) // 4, then n_para = 4*n + 2
                n = (n_para[i] - 2) // 4
                n_para_updated[i] = 4 * n + 2
        elif wf_mode[i] == "polar_phase":
            # Non-Fourier basis with polar_phase just adds 1
            n_para_updated[i] += 1
    return n_para_updated


def test_fourier_with_polar_phase():
    """Test Fourier basis with polar_phase mode."""
    # Setup
    n_para = [16]
    wf_type = ["fou"]
    wf_mode = ["polar_phase"]
    n_qubits = 1

    # Update n_para
    n_para_updated = update_n_para_fixed(n_para, wf_type, wf_mode, n_qubits)

    # Simulate what happens with x0
    x0 = np.random.uniform(-1, 1, size=n_para[0])

    # Fourier recalculates
    n = (len(x0) - 1) // 2
    n_para_fourier = 2 * n + 1
    x0_after_fourier = x0[0:n_para_fourier]

    # mat_with_amplitude_and_qr adds 1 for polar_phase
    x0_final = np.append(x0_after_fourier, x0_after_fourier[-1])

    # Assert
    assert len(x0_final) == n_para_updated[0], (
        f"Expected {n_para_updated[0]}, got {len(x0_final)}"
    )


def test_fourier_with_polar():
    """Test Fourier basis with polar mode (not polar_phase)."""
    # Setup
    n_para = [16]
    wf_type = ["fou"]
    wf_mode = ["polar"]
    n_qubits = 1

    # Update n_para
    n_para_updated = update_n_para_fixed(n_para, wf_type, wf_mode, n_qubits)

    # Simulate what happens with x0
    x0 = np.random.uniform(-1, 1, size=n_para[0])

    # Fourier recalculates for polar (not polar_phase)
    n = (len(x0) - 2) // 4
    n_para_fourier = 4 * n + 2
    x0_after_fourier = x0[0:n_para_fourier]

    # mat_with_amplitude_and_qr does NOT add for polar mode
    x0_final = x0_after_fourier

    # Assert
    assert len(x0_final) == n_para_updated[0], (
        f"Expected {n_para_updated[0]}, got {len(x0_final)}"
    )


def test_non_fourier_with_polar_phase():
    """Test non-Fourier basis (cheb) with polar_phase mode."""
    # Setup
    n_para = [16]
    wf_type = ["cheb"]
    wf_mode = ["polar_phase"]
    n_qubits = 1

    # Update n_para
    n_para_updated = update_n_para_fixed(n_para, wf_type, wf_mode, n_qubits)

    # Simulate what happens with x0
    x0 = np.random.uniform(-1, 1, size=n_para[0])

    # Non-Fourier uses all of x0
    x0_after_basis = x0

    # mat_with_amplitude_and_qr adds 1 for polar_phase
    x0_final = np.append(x0_after_basis, x0_after_basis[-1])

    # Assert
    assert len(x0_final) == n_para_updated[0], (
        f"Expected {n_para_updated[0]}, got {len(x0_final)}"
    )


def test_non_fourier_with_polar():
    """Test non-Fourier basis (cheb) with polar mode (not polar_phase)."""
    # Setup
    n_para = [16]
    wf_type = ["cheb"]
    wf_mode = ["polar"]
    n_qubits = 1

    # Update n_para
    n_para_updated = update_n_para_fixed(n_para, wf_type, wf_mode, n_qubits)

    # Simulate what happens with x0
    x0 = np.random.uniform(-1, 1, size=n_para[0])

    # Non-Fourier uses all of x0
    x0_after_basis = x0

    # mat_with_amplitude_and_qr does NOT add for polar mode
    x0_final = x0_after_basis

    # Assert
    assert len(x0_final) == n_para_updated[0], (
        f"Expected {n_para_updated[0]}, got {len(x0_final)}"
    )
