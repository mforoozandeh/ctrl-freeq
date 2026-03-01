"""Tests for the Hamiltonian model abstraction layer.

Tests cover:
- SpinChainModel: drift, control ops, control amplitudes, equivalence with legacy
- SuperconductingQubitModel: drift (qubit frequencies + coupling), control ops, ZZ coupling
- pulse_hamiltonian_generic: equivalence with legacy pulse_hamiltonian
- Plugin registry: registration, lookup, custom models, from_config, default_config
"""

import numpy as np
import pytest
import torch

from ctrl_freeq.setup.hamiltonian_generation.base import (
    HamiltonianModel,
    _HAMILTONIAN_REGISTRY,
    get_hamiltonian_class,
    list_hamiltonians,
    register_hamiltonian,
)
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
        H0_list = model.build_drift(frequency_instances=[Delta])
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
        H0_model = model.build_drift(
            frequency_instances=[Delta], coupling_instances=[J]
        )

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
        H0 = model.build_drift(frequency_instances=[omega])
        assert len(H0) == 1
        assert H0[0].shape == (2, 2)
        np.testing.assert_allclose(H0[0], H0[0].conj().T, atol=1e-12)

    def test_build_drift_two_qubit_xy(self):
        model = SuperconductingQubitModel(2, coupling_type="XY")
        omega = np.array([2 * np.pi * 5e9, 2 * np.pi * 5.2e9])
        g = np.array([[0.0, 2 * np.pi * 5e6], [0.0, 0.0]])
        H0 = model.build_drift(frequency_instances=[omega], coupling_instances=[g])
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
        H0 = model.build_drift(frequency_instances=[omega], coupling_instances=[g])
        assert H0[0].shape == (4, 4)
        np.testing.assert_allclose(H0[0], H0[0].conj().T, atol=1e-12)
        # ZZ should add additional coupling
        model_xy_only = SuperconductingQubitModel(2, coupling_type="XY")
        H0_xy = model_xy_only.build_drift(
            frequency_instances=[omega], coupling_instances=[g]
        )
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


# ---------------------------------------------------------------------------
# Plugin registry tests
# ---------------------------------------------------------------------------


class TestHamiltonianRegistry:
    """Tests for the registry / decorator infrastructure."""

    def test_builtins_registered(self):
        """SpinChainModel and SuperconductingQubitModel should be auto-registered."""
        assert get_hamiltonian_class("spin_chain") is SpinChainModel
        assert get_hamiltonian_class("superconducting") is SuperconductingQubitModel

    def test_list_hamiltonians(self):
        names = list_hamiltonians()
        assert "spin_chain" in names
        assert "superconducting" in names

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError, match="Unknown hamiltonian_type"):
            get_hamiltonian_class("nonexistent")

    def test_unknown_name_shows_available(self):
        with pytest.raises(KeyError, match="spin_chain"):
            get_hamiltonian_class("nonexistent")

    def test_register_custom_model(self):
        """Register a minimal custom model and look it up."""

        @register_hamiltonian("_test_custom")
        class _TestModel(HamiltonianModel):
            def __init__(self, n_qubits):
                self._n = n_qubits

            @property
            def dim(self):
                return 2**self._n

            @property
            def n_controls(self):
                return 2 * self._n

            def build_drift(self, frequency_instances, coupling_instances=None, **kw):
                return [np.diag(f) for f in frequency_instances]

            def build_control_ops(self):
                return [np.eye(self.dim)] * self.n_controls

            def control_amplitudes(self, cx, cy, rabi_freq, n_h0):
                n_pulse = cx.shape[0]
                u = torch.stack([cx, cy], dim=-1).reshape(n_pulse, 2 * self._n)
                rabi_expanded = rabi_freq.repeat_interleave(2, dim=-1)
                rabi_batch = rabi_expanded.repeat(n_h0, 1)
                return u.unsqueeze(1) * rabi_batch.unsqueeze(0)

            @classmethod
            def from_config(cls, n_qubits, params):
                return cls(n_qubits)

        try:
            assert get_hamiltonian_class("_test_custom") is _TestModel
            assert "_test_custom" in list_hamiltonians()

            # Verify from_config works
            model = _TestModel.from_config(2, {})
            assert model.dim == 4

            # Verify build_drift works with standardized signature
            freq = [np.array([1.0, 2.0])]
            H0 = model.build_drift(frequency_instances=freq)
            assert len(H0) == 1
            np.testing.assert_allclose(H0[0], np.diag([1.0, 2.0]))
        finally:
            _HAMILTONIAN_REGISTRY.pop("_test_custom", None)

    def test_duplicate_registration_raises(self):
        @register_hamiltonian("_test_dup")
        class _Dup1(HamiltonianModel):
            @property
            def dim(self):
                return 2

            @property
            def n_controls(self):
                return 2

            def build_drift(self, frequency_instances, coupling_instances=None, **kw):
                return []

            def build_control_ops(self):
                return []

            def control_amplitudes(self, cx, cy, rabi_freq, n_h0):
                return cx

            @classmethod
            def from_config(cls, n_qubits, params):
                return cls()

        try:
            with pytest.raises(ValueError, match="already registered"):

                @register_hamiltonian("_test_dup")
                class _Dup2(_Dup1):
                    pass
        finally:
            _HAMILTONIAN_REGISTRY.pop("_test_dup", None)

    def test_non_subclass_registration_raises(self):
        with pytest.raises(TypeError, match="must be a HamiltonianModel subclass"):

            @register_hamiltonian("_test_bad")
            class _NotAModel:
                pass


# ---------------------------------------------------------------------------
# from_config tests
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_spin_chain_from_config(self):
        params = {"coupling_type": "XY"}
        model = SpinChainModel.from_config(2, params)
        assert isinstance(model, SpinChainModel)
        assert model.n_qubits == 2
        assert model.coupling_type == "XY"

    def test_spin_chain_from_config_defaults(self):
        model = SpinChainModel.from_config(3, {})
        assert model.coupling_type == "XY"
        assert model.n_qubits == 3

    def test_superconducting_from_config(self):
        params = {
            "coupling_type": "XY+ZZ",
            "anharmonicities": [-330e6, -330e6],
        }
        model = SuperconductingQubitModel.from_config(2, params)
        assert isinstance(model, SuperconductingQubitModel)
        assert model.coupling_type == "XY+ZZ"
        assert model.anharmonicities is not None
        # Should be scaled by 2*pi
        expected = 2 * np.pi * np.array([-330e6, -330e6])
        np.testing.assert_allclose(model.anharmonicities, expected)

    def test_superconducting_from_config_defaults(self):
        model = SuperconductingQubitModel.from_config(2, {})
        assert model.coupling_type == "XY"
        assert model.anharmonicities is None

    def test_from_config_via_registry(self):
        """End-to-end: look up by name, then construct via from_config."""
        cls = get_hamiltonian_class("spin_chain")
        model = cls.from_config(2, {"coupling_type": "XY"})
        assert isinstance(model, SpinChainModel)
        assert model.dim == 4


# ---------------------------------------------------------------------------
# default_config tests
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_spin_chain_default_config_single_qubit(self):
        config = SpinChainModel.default_config(1)
        assert config["hamiltonian_type"] == "spin_chain"
        assert len(config["qubits"]) == 1
        assert "parameters" in config
        assert "optimization" in config
        assert "initial_states" in config
        assert "target_states" in config
        # Single qubit → Axis target
        assert "Axis" in config["target_states"]

    def test_spin_chain_default_config_two_qubit(self):
        config = SpinChainModel.default_config(2)
        assert len(config["qubits"]) == 2
        # Two qubit → Gate target (CNOT)
        assert "Gate" in config["target_states"]
        assert config["target_states"]["Gate"] == ["CNOT"]

    def test_superconducting_default_config_two_qubit(self):
        config = SuperconductingQubitModel.default_config(2)
        assert config["hamiltonian_type"] == "superconducting"
        assert len(config["qubits"]) == 2
        # Two qubit → Gate target (iSWAP)
        assert "Gate" in config["target_states"]
        assert config["target_states"]["Gate"] == ["iSWAP"]

    def test_default_config_is_runnable(self):
        """Verify default_config produces a config that CtrlFreeQAPI can accept."""
        from ctrl_freeq.api import CtrlFreeQAPI

        config = SpinChainModel.default_config(1)
        # Should not raise — just verify it initializes
        api = CtrlFreeQAPI(config)
        assert api.parameters is not None
        assert api.parameters.n_qubits == 1


# ---------------------------------------------------------------------------
# Direct model injection tests
# ---------------------------------------------------------------------------


class TestDirectModelInjection:
    def test_api_accepts_hamiltonian_model(self):
        """CtrlFreeQAPI should accept a pre-built model via hamiltonian_model param."""
        from ctrl_freeq.api import CtrlFreeQAPI

        config = SpinChainModel.default_config(1)
        model = SpinChainModel(1, coupling_type="XY")
        api = CtrlFreeQAPI(config, hamiltonian_model=model)
        assert api.parameters.hamiltonian_model is model

    def test_injected_model_overrides_config(self):
        """An injected model should override whatever hamiltonian_type is in config."""
        from ctrl_freeq.api import CtrlFreeQAPI

        # Config says spin_chain, but we inject superconducting
        config = SpinChainModel.default_config(1)
        sc_model = SuperconductingQubitModel(1, coupling_type="XY")
        api = CtrlFreeQAPI(config, hamiltonian_model=sc_model)
        assert isinstance(api.parameters.hamiltonian_model, SuperconductingQubitModel)
