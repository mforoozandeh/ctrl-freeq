"""Tests for the Hamiltonian model abstraction layer.

Tests cover:
- SpinChainModel: drift, control ops, control amplitudes, equivalence with legacy
- SuperconductingQubitModel: drift (qubit frequencies + coupling), control ops, ZZ coupling
- pulse_hamiltonian_generic: equivalence with legacy pulse_hamiltonian
"""

import numpy as np
import pytest
import torch

from ctrl_freeq.setup.hamiltonian_generation.base import HamiltonianModel
from ctrl_freeq.setup.hamiltonian_generation.spin_chain import SpinChainModel
from ctrl_freeq.setup.hamiltonian_generation.superconducting import (
    SuperconductingQubitModel,
)
from ctrl_freeq.setup.hamiltonian_generation.hamiltonians import createHcs, createHJ
from ctrl_freeq.setup.operator_generation.generate_operators import (
    create_hamiltonian_basis,
    create_hamiltonian_basis_torch,
)
from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
    pulse_hamiltonian,
    pulse_hamiltonian_generic,
)


# ---------------------------------------------------------------------------
# SpinChainModel tests
# ---------------------------------------------------------------------------


class TestSpinChainModel:
    def test_is_hamiltonian_model(self):
        model = SpinChainModel(2, coupling_type="XY")
        assert isinstance(model, HamiltonianModel)

    def test_dim_single_qubit(self):
        model = SpinChainModel(1)
        assert model.dim == 2

    def test_dim_two_qubits(self):
        model = SpinChainModel(2)
        assert model.dim == 4

    def test_n_controls(self):
        model = SpinChainModel(3)
        assert model.n_controls == 6  # X, Y per qubit

    def test_build_drift_single_qubit(self):
        model = SpinChainModel(1)
        Delta = np.array([2 * np.pi * 1e7])
        H0_list = model.build_drift(Delta_instances=[Delta])
        assert len(H0_list) == 1
        assert H0_list[0].shape == (2, 2)

    def test_build_drift_matches_legacy(self):
        """Verify SpinChainModel.build_drift matches createHcs + createHJ."""
        n_qubits = 2
        model = SpinChainModel(n_qubits, coupling_type="XY")
        op = create_hamiltonian_basis(n_qubits)

        Delta = np.array([2 * np.pi * 1e7, 2 * np.pi * 2e7])
        J = np.array([[0.0, 2 * np.pi * 1e6], [0.0, 0.0]])

        # Model path
        H0_model = model.build_drift(Delta_instances=[Delta], J_instances=[J])

        # Legacy path
        Hcs = createHcs(Delta, op)
        HJ = createHJ(J, op, coupling_type="XY")
        H0_legacy = HJ + Hcs

        np.testing.assert_allclose(H0_model[0], H0_legacy, atol=1e-12)

    def test_build_control_ops_count(self):
        model = SpinChainModel(2)
        ops = model.build_control_ops()
        assert len(ops) == 4  # X_0, Y_0, X_1, Y_1

    def test_build_control_ops_hermitian(self):
        model = SpinChainModel(2)
        ops = model.build_control_ops()
        for op in ops:
            np.testing.assert_allclose(op, op.conj().T, atol=1e-12)

    def test_control_ops_tensor(self):
        model = SpinChainModel(2)
        t = model.control_ops_tensor()
        assert t.shape == (4, 4, 4)
        assert t.dtype == torch.complex128

    def test_control_amplitudes_shape(self):
        model = SpinChainModel(2)
        n_pulse, n_rabi, n_h0 = 50, 3, 10
        cx = torch.randn(n_pulse, 2)
        cy = torch.randn(n_pulse, 2)
        rabi = torch.rand(n_rabi, 2) * 1e7
        u = model.control_amplitudes(cx, cy, rabi, n_h0)
        assert u.shape == (n_pulse, n_rabi * n_h0, 4)

    def test_control_amplitudes_interleaving(self):
        """Verify [cx_0, cy_0, cx_1, cy_1] interleaving."""
        model = SpinChainModel(2)
        cx = torch.tensor([[1.0, 3.0]])
        cy = torch.tensor([[2.0, 4.0]])
        rabi = torch.tensor([[1.0, 1.0]])  # Unit Rabi so we see raw values
        u = model.control_amplitudes(cx, cy, rabi, n_h0=1)
        expected = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        torch.testing.assert_close(u, expected)


class TestPulseHamiltonianGenericEquivalence:
    """Verify pulse_hamiltonian_generic matches the legacy pulse_hamiltonian."""

    @pytest.mark.parametrize("n_qubits", [1, 2, 3])
    def test_equivalence(self, n_qubits):
        n_pulse, n_h0, n_rabi = 20, 5, 2
        D = 2**n_qubits

        cx = torch.randn(n_pulse, n_qubits, dtype=torch.float64)
        cy = torch.randn(n_pulse, n_qubits, dtype=torch.float64)
        rabi = torch.rand(n_rabi, n_qubits, dtype=torch.float64) * 1e7

        # Legacy
        op_tensor = create_hamiltonian_basis_torch(n_qubits)
        Hp_legacy = pulse_hamiltonian(
            cx, cy, rabi, op_tensor, n_pulse, n_h0, n_rabi, n_qubits
        )

        # Generic via SpinChainModel
        model = SpinChainModel(n_qubits)
        ctrl_ops = model.control_ops_tensor()
        u = model.control_amplitudes(cx, cy, rabi, n_h0)
        Hp_generic = pulse_hamiltonian_generic(u, ctrl_ops)

        assert Hp_legacy.shape == Hp_generic.shape == (n_pulse, n_rabi * n_h0, D, D)
        torch.testing.assert_close(Hp_generic, Hp_legacy, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# SuperconductingQubitModel tests
# ---------------------------------------------------------------------------


class TestSuperconductingQubitModel:
    def test_is_hamiltonian_model(self):
        model = SuperconductingQubitModel(2)
        assert isinstance(model, HamiltonianModel)

    def test_dim(self):
        model = SuperconductingQubitModel(3)
        assert model.dim == 8

    def test_n_controls(self):
        model = SuperconductingQubitModel(2)
        assert model.n_controls == 4

    def test_build_drift_single_qubit(self):
        model = SuperconductingQubitModel(1)
        omega = np.array([2 * np.pi * 5e9])
        H0 = model.build_drift(omega_instances=[omega])
        assert len(H0) == 1
        assert H0[0].shape == (2, 2)
        np.testing.assert_allclose(H0[0], H0[0].conj().T, atol=1e-12)

    def test_build_drift_two_qubit_xy(self):
        model = SuperconductingQubitModel(2, coupling_type="XY")
        omega = np.array([2 * np.pi * 5e9, 2 * np.pi * 5.2e9])
        g = np.array([[0.0, 2 * np.pi * 5e6], [0.0, 0.0]])
        H0 = model.build_drift(omega_instances=[omega], g_instances=[g])
        assert H0[0].shape == (4, 4)
        np.testing.assert_allclose(H0[0], H0[0].conj().T, atol=1e-12)

    def test_build_drift_zz_coupling(self):
        model = SuperconductingQubitModel(
            2,
            coupling_type="XY+ZZ",
            anharmonicities=np.array([2 * np.pi * -330e6, 2 * np.pi * -330e6]),
        )
        omega = np.array([2 * np.pi * 5e9, 2 * np.pi * 5.2e9])
        g = np.array([[0.0, 2 * np.pi * 5e6], [0.0, 0.0]])
        H0 = model.build_drift(omega_instances=[omega], g_instances=[g])
        assert H0[0].shape == (4, 4)
        np.testing.assert_allclose(H0[0], H0[0].conj().T, atol=1e-12)
        # ZZ should add additional coupling
        model_xy_only = SuperconductingQubitModel(2, coupling_type="XY")
        H0_xy = model_xy_only.build_drift(omega_instances=[omega], g_instances=[g])
        # Hamiltonians should differ due to ZZ term
        assert not np.allclose(H0[0], H0_xy[0])

    def test_control_ops_hermitian(self):
        model = SuperconductingQubitModel(2)
        ops = model.build_control_ops()
        assert len(ops) == 4
        for op in ops:
            np.testing.assert_allclose(op, op.conj().T, atol=1e-12)

    def test_control_amplitudes_shape(self):
        model = SuperconductingQubitModel(2)
        n_pulse, n_rabi, n_h0 = 50, 2, 5
        cx = torch.randn(n_pulse, 2)
        cy = torch.randn(n_pulse, 2)
        rabi = torch.rand(n_rabi, 2) * 1e7
        u = model.control_amplitudes(cx, cy, rabi, n_h0)
        assert u.shape == (n_pulse, n_rabi * n_h0, 4)

    def test_control_amplitudes_matches_spin_chain(self):
        """SC qubits and spin chains have identical control structures (I/Q = X/Y)."""
        n_qubits = 2
        sc_model = SuperconductingQubitModel(n_qubits)
        spin_model = SpinChainModel(n_qubits)

        cx = torch.randn(20, n_qubits)
        cy = torch.randn(20, n_qubits)
        rabi = torch.rand(3, n_qubits) * 1e7

        u_sc = sc_model.control_amplitudes(cx, cy, rabi, n_h0=5)
        u_spin = spin_model.control_amplitudes(cx, cy, rabi, n_h0=5)

        torch.testing.assert_close(u_sc, u_spin)


# ---------------------------------------------------------------------------
# Generic pulse Hamiltonian tests
# ---------------------------------------------------------------------------


class TestPulseHamiltonianGeneric:
    def test_output_shape(self):
        n_pulse, n_batch, n_ctrl, D = 10, 6, 4, 4
        amplitudes = torch.randn(n_pulse, n_batch, n_ctrl, dtype=torch.complex128)
        ctrl_ops = torch.randn(n_ctrl, D, D, dtype=torch.complex128)
        Hp = pulse_hamiltonian_generic(amplitudes, ctrl_ops)
        assert Hp.shape == (n_pulse, n_batch, D, D)

    def test_zero_amplitudes_give_zero_hamiltonian(self):
        n_pulse, n_batch, n_ctrl, D = 10, 6, 4, 4
        amplitudes = torch.zeros(n_pulse, n_batch, n_ctrl, dtype=torch.complex128)
        ctrl_ops = torch.randn(n_ctrl, D, D, dtype=torch.complex128)
        Hp = pulse_hamiltonian_generic(amplitudes, ctrl_ops)
        torch.testing.assert_close(Hp, torch.zeros_like(Hp))

    def test_single_control(self):
        """With one control, Hp = u(t) * H_ctrl."""
        D = 2
        ctrl_op = torch.eye(D, dtype=torch.complex128)
        ctrl_ops = ctrl_op.unsqueeze(0)  # (1, D, D)
        amplitudes = torch.tensor(
            [[[3.0]], [[5.0]]], dtype=torch.complex128
        )  # (2, 1, 1)
        Hp = pulse_hamiltonian_generic(amplitudes, ctrl_ops)
        expected_0 = 3.0 * torch.eye(D, dtype=torch.complex128)
        expected_1 = 5.0 * torch.eye(D, dtype=torch.complex128)
        torch.testing.assert_close(Hp[0, 0], expected_0)
        torch.testing.assert_close(Hp[1, 0], expected_1)


# ---------------------------------------------------------------------------
# Base class tests
# ---------------------------------------------------------------------------


class TestHamiltonianModelABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            HamiltonianModel()

    def test_embed_operator(self):
        """Test the static _embed_operator helper."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        embedded = HamiltonianModel._embed_operator(X, 0, 2)
        assert embedded.shape == (4, 4)
        # Should be X ⊗ I
        I2 = np.eye(2, dtype=complex)
        expected = np.kron(X, I2)
        np.testing.assert_allclose(embedded, expected, atol=1e-12)

    def test_embed_operator_second_qubit(self):
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        embedded = HamiltonianModel._embed_operator(Z, 1, 2)
        I2 = np.eye(2, dtype=complex)
        expected = np.kron(I2, Z)
        np.testing.assert_allclose(embedded, expected, atol=1e-12)
