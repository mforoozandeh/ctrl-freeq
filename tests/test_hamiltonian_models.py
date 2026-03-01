"""Tests for the Hamiltonian model abstraction layer.

Tests cover:
- SpinChainModel: drift, control ops, control amplitudes, equivalence with legacy
- SuperconductingQubitModel: drift (qubit frequencies + coupling), control ops, ZZ coupling
- SuperconductingQubitModel: calibrated ZZ (zz_crosstalk), AC Stark shift
- DuffingTransmonModel: 3-level operators, embedding, leakage
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
from ctrl_freeq.setup.hamiltonian_generation.duffing_transmon import (
    DuffingTransmonModel,
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


# ---------------------------------------------------------------------------
# Tier 2A — Calibrated ZZ (zz_crosstalk) tests
# ---------------------------------------------------------------------------


class TestCalibratedZZ:
    def test_zz_crosstalk_used_directly(self):
        """When zz_crosstalk is provided, it should be used instead of the formula."""
        zz_cal = np.array([[0.0, 1e6], [0.0, 0.0]])
        model = SuperconductingQubitModel(2, coupling_type="XY+ZZ", zz_crosstalk=zz_cal)
        omega = np.array([2 * np.pi * 5e9, 2 * np.pi * 5.2e9])
        g = np.array([[0.0, 2 * np.pi * 5e6], [0.0, 0.0]])
        H0 = model.build_drift(frequency_instances=[omega], coupling_instances=[g])
        assert H0[0].shape == (4, 4)
        np.testing.assert_allclose(H0[0], H0[0].conj().T, atol=1e-12)

    def test_zz_crosstalk_differs_from_formula(self):
        """Calibrated ZZ should produce different H0 than the formula."""
        alpha = np.array([2 * np.pi * -330e6, 2 * np.pi * -330e6])
        omega = np.array([2 * np.pi * 5e9, 2 * np.pi * 5.2e9])
        g = np.array([[0.0, 2 * np.pi * 5e6], [0.0, 0.0]])

        model_formula = SuperconductingQubitModel(
            2, coupling_type="XY+ZZ", anharmonicities=alpha
        )
        H0_formula = model_formula.build_drift(
            frequency_instances=[omega], coupling_instances=[g]
        )

        # Use a very different calibrated value
        zz_cal = np.array([[0.0, 1e8], [0.0, 0.0]])
        model_cal = SuperconductingQubitModel(
            2, coupling_type="XY+ZZ", zz_crosstalk=zz_cal
        )
        H0_cal = model_cal.build_drift(
            frequency_instances=[omega], coupling_instances=[g]
        )

        assert not np.allclose(H0_formula[0], H0_cal[0])

    def test_zz_instances_overrides_zz_crosstalk(self):
        """Runtime zz_instances should override calibrated zz_crosstalk."""
        zz_cal = np.array([[0.0, 1e6], [0.0, 0.0]])
        zz_runtime = np.array([[0.0, 5e8], [0.0, 0.0]])

        model = SuperconductingQubitModel(2, coupling_type="XY+ZZ", zz_crosstalk=zz_cal)
        omega = np.array([2 * np.pi * 5e9, 2 * np.pi * 5.2e9])
        g = np.array([[0.0, 2 * np.pi * 5e6], [0.0, 0.0]])

        H0_cal = model.build_drift(frequency_instances=[omega], coupling_instances=[g])
        H0_runtime = model.build_drift(
            frequency_instances=[omega],
            coupling_instances=[g],
            zz_instances=[zz_runtime],
        )

        assert not np.allclose(H0_cal[0], H0_runtime[0])

    def test_from_config_extracts_zz_crosstalk(self):
        """from_config should extract and 2π-scale zz_crosstalk."""
        params = {
            "coupling_type": "XY+ZZ",
            "zz_crosstalk": [[0.0, 1e6], [0.0, 0.0]],
        }
        model = SuperconductingQubitModel.from_config(2, params)
        assert model.zz_crosstalk is not None
        expected = 2 * np.pi * np.array([[0.0, 1e6], [0.0, 0.0]])
        np.testing.assert_allclose(model.zz_crosstalk, expected)

    def test_backward_compat_no_zz_crosstalk(self):
        """Omitting zz_crosstalk should preserve existing formula behaviour."""
        alpha = np.array([2 * np.pi * -330e6, 2 * np.pi * -330e6])
        omega = np.array([2 * np.pi * 5e9, 2 * np.pi * 5.2e9])
        g = np.array([[0.0, 2 * np.pi * 5e6], [0.0, 0.0]])

        model_old = SuperconductingQubitModel(
            2, coupling_type="XY+ZZ", anharmonicities=alpha
        )
        model_new = SuperconductingQubitModel(
            2,
            coupling_type="XY+ZZ",
            anharmonicities=alpha,
            zz_crosstalk=None,
        )

        H0_old = model_old.build_drift(
            frequency_instances=[omega], coupling_instances=[g]
        )
        H0_new = model_new.build_drift(
            frequency_instances=[omega], coupling_instances=[g]
        )
        np.testing.assert_allclose(H0_old[0], H0_new[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Tier 2B — AC Stark shift tests
# ---------------------------------------------------------------------------


class TestACStarkShift:
    def test_n_controls_without_stark(self):
        model = SuperconductingQubitModel(2)
        assert model.n_controls == 4

    def test_n_controls_with_stark(self):
        model = SuperconductingQubitModel(2, stark_shift_coeffs=[0.01, 0.01])
        assert model.n_controls == 6  # 3 per qubit

    def test_build_control_ops_without_stark(self):
        model = SuperconductingQubitModel(2)
        ops = model.build_control_ops()
        assert len(ops) == 4  # X_0, Y_0, X_1, Y_1

    def test_build_control_ops_with_stark(self):
        model = SuperconductingQubitModel(2, stark_shift_coeffs=[0.01, 0.01])
        ops = model.build_control_ops()
        assert len(ops) == 6  # X_0, Y_0, Z_0, X_1, Y_1, Z_1
        for op in ops:
            np.testing.assert_allclose(op, op.conj().T, atol=1e-12)

    def test_control_amplitudes_shape_with_stark(self):
        model = SuperconductingQubitModel(2, stark_shift_coeffs=[0.01, 0.02])
        n_pulse, n_rabi, n_h0 = 50, 2, 5
        cx = torch.randn(n_pulse, 2)
        cy = torch.randn(n_pulse, 2)
        rabi = torch.rand(n_rabi, 2) * 1e7
        u = model.control_amplitudes(cx, cy, rabi, n_h0)
        assert u.shape == (n_pulse, n_rabi * n_h0, 6)

    def test_z_amplitude_is_quadratic(self):
        """Z channel amplitude should be s*(I²+Q²)*Ω²."""
        s = np.array([0.5, 0.3])
        model = SuperconductingQubitModel(2, stark_shift_coeffs=s)

        cx = torch.tensor([[1.0, 2.0]])
        cy = torch.tensor([[3.0, 4.0]])
        rabi = torch.tensor([[10.0, 10.0]])

        u = model.control_amplitudes(cx, cy, rabi, n_h0=1)
        # u has shape (1, 1, 6): [I_0*Ω, Q_0*Ω, s_0*(I²+Q²)*Ω², I_1*Ω, Q_1*Ω, s_1*(I²+Q²)*Ω²]

        # Qubit 0: s=0.5, I=1, Q=3, Ω=10  → Z = 0.5*(1+9)*100 = 500
        expected_z0 = s[0] * (1.0**2 + 3.0**2) * 10.0**2
        assert abs(u[0, 0, 2].item() - expected_z0) < 1e-6

        # Qubit 1: s=0.3, I=2, Q=4, Ω=10  → Z = 0.3*(4+16)*100 = 600
        expected_z1 = s[1] * (2.0**2 + 4.0**2) * 10.0**2
        assert abs(u[0, 0, 5].item() - expected_z1) < 1e-6

    def test_stark_compatible_with_pulse_hamiltonian_generic(self):
        """Stark-shift control amplitudes should work with einsum pipeline."""
        model = SuperconductingQubitModel(2, stark_shift_coeffs=[0.01, 0.01])
        ctrl_ops = model.control_ops_tensor()
        assert ctrl_ops.shape == (6, 4, 4)

        cx = torch.randn(20, 2, dtype=torch.float64)
        cy = torch.randn(20, 2, dtype=torch.float64)
        rabi = torch.rand(2, 2, dtype=torch.float64) * 1e7
        u = model.control_amplitudes(cx, cy, rabi, n_h0=3)

        Hp = pulse_hamiltonian_generic(u, ctrl_ops)
        assert Hp.shape == (20, 6, 4, 4)

    def test_from_config_extracts_stark_shift_coeffs(self):
        params = {"stark_shift_coeffs": [0.01, 0.02]}
        model = SuperconductingQubitModel.from_config(2, params)
        assert model.stark_shift_coeffs is not None
        np.testing.assert_allclose(model.stark_shift_coeffs, [0.01, 0.02])

    def test_backward_compat_no_stark(self):
        """Without Stark shift, behaviour should be identical to before."""
        model_no_stark = SuperconductingQubitModel(2)
        model_none = SuperconductingQubitModel(2, stark_shift_coeffs=None)

        cx = torch.randn(20, 2)
        cy = torch.randn(20, 2)
        rabi = torch.rand(3, 2) * 1e7

        u1 = model_no_stark.control_amplitudes(cx, cy, rabi, n_h0=5)
        u2 = model_none.control_amplitudes(cx, cy, rabi, n_h0=5)
        torch.testing.assert_close(u1, u2)


# ---------------------------------------------------------------------------
# Tier 3 — DuffingTransmonModel tests
# ---------------------------------------------------------------------------


class TestDuffingTransmonModel:
    def test_registered_in_registry(self):
        assert get_hamiltonian_class("duffing_transmon") is DuffingTransmonModel
        assert "duffing_transmon" in list_hamiltonians()

    def test_is_hamiltonian_model(self):
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        assert isinstance(model, HamiltonianModel)

    def test_dim_single_qubit(self):
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        assert model.dim == 3

    def test_dim_two_qubits(self):
        model = DuffingTransmonModel(2, anharmonicities=[2 * np.pi * -330e6] * 2)
        assert model.dim == 9

    def test_n_controls(self):
        model = DuffingTransmonModel(2, anharmonicities=[2 * np.pi * -330e6] * 2)
        assert model.n_controls == 4

    def test_anharmonicities_required(self):
        with pytest.raises(ValueError, match="requires anharmonicities"):
            DuffingTransmonModel(1, anharmonicities=None)

    def test_build_drift_hermitian(self):
        model = DuffingTransmonModel(
            2, anharmonicities=np.array([2 * np.pi * -330e6, 2 * np.pi * -330e6])
        )
        omega = np.array([2 * np.pi * 5e9, 2 * np.pi * 5.2e9])
        g = np.array([[0.0, 2 * np.pi * 5e6], [0.0, 0.0]])
        H0 = model.build_drift(frequency_instances=[omega], coupling_instances=[g])
        assert H0[0].shape == (9, 9)
        np.testing.assert_allclose(H0[0], H0[0].conj().T, atol=1e-12)

    def test_anharmonicity_structure_single_qubit(self):
        """Single-qubit drift diagonal should have correct level structure."""
        alpha = 2 * np.pi * -330e6
        omega = np.array([2 * np.pi * 5e9])
        model = DuffingTransmonModel(1, anharmonicities=[alpha])
        H0 = model.build_drift(frequency_instances=[omega])
        H = H0[0]

        # For single qubit: H = omega * n_hat + (alpha/2) * n_hat*(n_hat - I)
        # n_hat = diag(0, 1, 2)
        # n_hat*(n_hat-I) = diag(0, 0, 2)
        # H = diag(0, omega, 2*omega) + diag(0, 0, alpha)
        #   = diag(0, omega, 2*omega + alpha)
        expected_diag = np.array([0, omega[0], 2 * omega[0] + alpha])
        np.testing.assert_allclose(np.diag(H).real, expected_diag, rtol=1e-10)

    def test_build_control_ops_count_and_hermiticity(self):
        model = DuffingTransmonModel(2, anharmonicities=[2 * np.pi * -330e6] * 2)
        ops = model.build_control_ops()
        assert len(ops) == 4
        for op in ops:
            assert op.shape == (9, 9)
            np.testing.assert_allclose(op, op.conj().T, atol=1e-12)

    def test_control_amplitudes_shape(self):
        model = DuffingTransmonModel(2, anharmonicities=[2 * np.pi * -330e6] * 2)
        n_pulse, n_rabi, n_h0 = 50, 2, 5
        cx = torch.randn(n_pulse, 2)
        cy = torch.randn(n_pulse, 2)
        rabi = torch.rand(n_rabi, 2) * 1e7
        u = model.control_amplitudes(cx, cy, rabi, n_h0)
        assert u.shape == (n_pulse, n_rabi * n_h0, 4)

    def test_control_amplitudes_matches_superconducting(self):
        """Duffing control amplitudes should match superconducting (same structure)."""
        n_qubits = 2
        alpha = [2 * np.pi * -330e6] * n_qubits
        duff = DuffingTransmonModel(n_qubits, anharmonicities=alpha)
        sc = SuperconductingQubitModel(n_qubits)

        cx = torch.randn(20, n_qubits)
        cy = torch.randn(20, n_qubits)
        rabi = torch.rand(3, n_qubits) * 1e7

        u_duff = duff.control_amplitudes(cx, cy, rabi, n_h0=5)
        u_sc = sc.control_amplitudes(cx, cy, rabi, n_h0=5)

        torch.testing.assert_close(u_duff, u_sc)


class TestDuffingEmbedding:
    def test_embed_computational_state_ground(self):
        """Embedding |0> (2-level) should give |0> (3-level)."""
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        state_2 = np.array([1, 0], dtype=complex)
        state_3 = model.embed_computational_state(state_2)
        assert state_3.shape == (3,)
        np.testing.assert_allclose(state_3, [1, 0, 0])

    def test_embed_computational_state_excited(self):
        """Embedding |1> (2-level) should give |1> (3-level)."""
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        state_2 = np.array([0, 1], dtype=complex)
        state_3 = model.embed_computational_state(state_2)
        np.testing.assert_allclose(state_3, [0, 1, 0])

    def test_embed_two_qubit_state(self):
        """Embedding |01> (4-dim) should give correct 9-dim state."""
        model = DuffingTransmonModel(2, anharmonicities=[2 * np.pi * -330e6] * 2)
        # |01> = [0, 1, 0, 0] in 2^2 = 4 dimensional space
        state_4 = np.array([0, 1, 0, 0], dtype=complex)
        state_9 = model.embed_computational_state(state_4)
        assert state_9.shape == (9,)
        # |01> in 3^2 = 9 dim:
        # qubit 0 = |1>, qubit 1 = |0>
        # index = 1*3^0 + 0*3^1 = 1
        expected = np.zeros(9, dtype=complex)
        expected[1] = 1.0
        np.testing.assert_allclose(state_9, expected)

    def test_embed_computational_gate_x(self):
        """Embedding Pauli X should flip |0>↔|1> and leave |2> unchanged."""
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        X_embed = model.embed_computational_gate(X)
        assert X_embed.shape == (3, 3)

        # Apply to |0> → should give |1>
        result = X_embed @ np.array([1, 0, 0], dtype=complex)
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-12)

        # Apply to |2> → should give |2> (identity on leakage subspace)
        result = X_embed @ np.array([0, 0, 1], dtype=complex)
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-12)

    def test_embed_gate_is_unitary(self):
        """Embedded gate should be unitary."""
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        H_embed = model.embed_computational_gate(H)
        np.testing.assert_allclose(H_embed @ H_embed.conj().T, np.eye(3), atol=1e-12)


class TestDuffingLeakage:
    def test_leakage_zero_for_computational_state(self):
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        state = np.array([0, 1, 0], dtype=complex)  # |1> — computational
        assert abs(model.leakage(state)) < 1e-12

    def test_leakage_one_for_pure_leakage_state(self):
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        state = np.array([0, 0, 1], dtype=complex)  # |2> — leakage
        assert abs(model.leakage(state) - 1.0) < 1e-12

    def test_leakage_partial(self):
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        # 50% in |1>, 50% in |2>
        state = np.array([0, 1, 1], dtype=complex) / np.sqrt(2)
        assert abs(model.leakage(state) - 0.5) < 1e-12

    def test_leakage_with_torch_tensor(self):
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        state = torch.tensor([0, 0, 1], dtype=torch.complex128)
        assert abs(model.leakage(state) - 1.0) < 1e-12

    def test_leakage_two_qubit(self):
        """Leakage for a 2-qubit state with some |2> population."""
        model = DuffingTransmonModel(2, anharmonicities=[2 * np.pi * -330e6] * 2)
        # |00> in 3^2 space = index 0 — computational
        state = np.zeros(9, dtype=complex)
        state[0] = 1.0
        assert abs(model.leakage(state)) < 1e-12

        # |20> in 3^2 space = index 0*3^0 + 2*3^1 = 6 — leakage
        state2 = np.zeros(9, dtype=complex)
        state2[6] = 1.0
        assert abs(model.leakage(state2) - 1.0) < 1e-12


class TestDuffingPulseHamiltonianGenericCompat:
    def test_einsum_compatible(self):
        """pulse_hamiltonian_generic should work with Duffing model."""
        model = DuffingTransmonModel(1, anharmonicities=[2 * np.pi * -330e6])
        ctrl_ops = model.control_ops_tensor()
        assert ctrl_ops.shape == (2, 3, 3)

        cx = torch.randn(10, 1, dtype=torch.float64)
        cy = torch.randn(10, 1, dtype=torch.float64)
        rabi = torch.rand(2, 1, dtype=torch.float64) * 1e7
        u = model.control_amplitudes(cx, cy, rabi, n_h0=3)

        Hp = pulse_hamiltonian_generic(u, ctrl_ops)
        assert Hp.shape == (10, 6, 3, 3)


class TestDuffingFromConfig:
    def test_from_config_with_anharmonicities(self):
        params = {
            "coupling_type": "XY",
            "anharmonicities": [-330e6, -330e6],
        }
        model = DuffingTransmonModel.from_config(2, params)
        assert model.dim == 9
        expected = 2 * np.pi * np.array([-330e6, -330e6])
        np.testing.assert_allclose(model.anharmonicities, expected)

    def test_from_config_without_anharmonicities_raises(self):
        with pytest.raises(ValueError, match="requires anharmonicities"):
            DuffingTransmonModel.from_config(2, {})

    def test_default_config_round_trip(self):
        config = DuffingTransmonModel.default_config(2)
        assert config["hamiltonian_type"] == "duffing_transmon"
        assert "anharmonicities" in config["parameters"]

        # Should be constructible via from_config
        cls = get_hamiltonian_class(config["hamiltonian_type"])
        model = cls.from_config(len(config["qubits"]), config["parameters"])
        assert model.dim == 9
