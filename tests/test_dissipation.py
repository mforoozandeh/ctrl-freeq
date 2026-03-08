"""Tests for Lindblad dissipation: collapse operators, validation, and rates.

Covers:
- Collapse operator construction (shape, L†L hermiticity, trace-preserving)
- Dephasing rate correctness (off-diagonal decay matches 1/T2)
- T1/T2 input validation (positive, finite, T2 <= 2*T1)
- Dissipative + non-2-level model guard
- Lindblad dissipator properties
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ctrl_freeq.setup.initialise_gui import Initialise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_dissipative_config():
    """Load the bundled single-qubit dissipative config as a dict."""
    config_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "ctrl_freeq"
        / "data"
        / "json_input"
        / "single_qubit_dissipative.json"
    )
    with open(config_path) as f:
        return json.load(f)


def _make_config(T1, T2, n_qubits=1):
    """Build a minimal dissipative config dict with given T1/T2 arrays."""
    cfg = _load_dissipative_config()
    cfg["parameters"]["T1"] = list(T1)
    cfg["parameters"]["T2"] = list(T2)

    if n_qubits > 1:
        cfg["qubits"] = [f"q{i + 1}" for i in range(n_qubits)]
        cfg["parameters"]["Delta"] = [10e6 * (i + 1) for i in range(n_qubits)]
        for key in [
            "sigma_Delta",
            "Omega_R_max",
            "pulse_duration",
            "point_in_pulse",
            "wf_type",
            "wf_mode",
            "amplitude_envelope",
            "amplitude_order",
            "coverage",
            "sw",
            "pulse_offset",
            "pulse_bandwidth",
            "ratio_factor",
            "sigma_Omega_R_max",
            "profile_order",
            "n_para",
        ]:
            val = cfg["parameters"][key]
            cfg["parameters"][key] = val * n_qubits if isinstance(val, list) else val
        J = [[0.0] * n_qubits for _ in range(n_qubits)]
        for i in range(n_qubits - 1):
            J[i][i + 1] = 16.67e6
        cfg["parameters"]["J"] = J
        cfg["parameters"]["coupling_type"] = "XY"
        cfg["initial_states"] = [["Z"] + ["-Z"] * (n_qubits - 1)]
        cfg["target_states"] = {"Axis": [["-Z"] + ["Z"] * (n_qubits - 1)]}

    # Convert lists to numpy arrays as Initialise expects
    cfg["parameters"] = {
        k: np.array(v) if isinstance(v, list) else v
        for k, v in cfg["parameters"].items()
    }
    cfg["target_states"] = {
        k: np.array(v) if isinstance(v, list) else v
        for k, v in cfg["target_states"].items()
    }

    return cfg


# ---------------------------------------------------------------------------
# Collapse-operator construction
# ---------------------------------------------------------------------------


class TestCollapseOperatorConstruction:
    """Tests for build_collapse_operators shape and algebraic properties."""

    def test_single_qubit_shape(self):
        """Single qubit with T1 and T2 should produce 2 collapse ops of shape (2,2)."""
        cfg = _make_config(T1=[1e-3], T2=[5e-4])
        p = Initialise(cfg)
        ops = p.collapse_operators
        assert ops is not None
        assert ops.shape[0] == 2  # L1 (damping) + L2 (dephasing)
        assert ops.shape[1:] == (2, 2)

    def test_two_qubit_shape(self):
        """Two qubits should produce 4 collapse ops of shape (4,4)."""
        cfg = _make_config(T1=[1e-3, 2e-3], T2=[5e-4, 1e-3], n_qubits=2)
        p = Initialise(cfg)
        ops = p.collapse_operators
        assert ops is not None
        assert ops.shape[0] == 4  # 2 ops per qubit
        assert ops.shape[1:] == (4, 4)

    def test_pure_t1_only(self):
        """When T2 = 2*T1 (no pure dephasing), only amplitude damping ops remain."""
        cfg = _make_config(T1=[1e-3], T2=[2e-3])
        p = Initialise(cfg)
        ops = p.collapse_operators
        assert ops is not None
        assert ops.shape[0] == 1  # only L1

    def test_ldag_l_hermitian(self):
        """L†L must be Hermitian for every collapse operator."""
        cfg = _make_config(T1=[1e-3], T2=[5e-4])
        p = Initialise(cfg)
        for L in p.collapse_operators:
            LdL = L.conj().T @ L
            np.testing.assert_allclose(LdL, LdL.conj().T, atol=1e-15)

    def test_trace_preserving(self):
        """sum_k L_k†L_k should be diagonal (necessary for trace preservation)."""
        cfg = _make_config(T1=[1e-3], T2=[5e-4])
        p = Initialise(cfg)
        total = sum(L.conj().T @ L for L in p.collapse_operators)
        # Off-diagonal elements should be zero
        D = total.shape[0]
        for i in range(D):
            for j in range(D):
                if i != j:
                    assert abs(total[i, j]) < 1e-15


# ---------------------------------------------------------------------------
# Dephasing rate correctness
# ---------------------------------------------------------------------------


class TestDephasingRate:
    """Verify that the dephasing operator produces the correct decay rate."""

    def test_single_qubit_offdiag_decay_rate(self):
        """The dissipator should decay rho_01 at rate gamma_phi = 1/T2 - 1/(2*T1).

        We construct rho = |+><+| and compute D[L2](rho)_01.  The rate of
        change of the off-diagonal should be -gamma_phi * rho_01.
        """
        T1, T2 = 1e-3, 5e-4
        gamma1 = 1.0 / T1
        gamma_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
        total_offdiag_rate = gamma1 / 2.0 + gamma_phi  # T1 also decoheres

        cfg = _make_config(T1=[T1], T2=[T2])
        p = Initialise(cfg)

        # |+> state density matrix
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(plus, plus.conj())
        rho_01 = rho[0, 1]

        # Compute full dissipator D(rho)
        D_rho = np.zeros_like(rho)
        for L in p.collapse_operators:
            Ld = L.conj().T
            D_rho += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)

        # Off-diagonal decay rate: d(rho_01)/dt = -total_rate * rho_01
        expected_drho_01 = -total_offdiag_rate * rho_01
        np.testing.assert_allclose(D_rho[0, 1], expected_drho_01, rtol=1e-12)

    def test_pure_dephasing_only(self):
        """When T1 -> inf (set very large), only dephasing should contribute.

        Gamma_phi = 1/T2 - 1/(2*T1) ≈ 1/T2 for large T1.
        """
        T1 = 1e6  # effectively infinite
        T2 = 1e-3
        gamma_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
        gamma1 = 1.0 / T1

        cfg = _make_config(T1=[T1], T2=[T2])
        p = Initialise(cfg)

        # |+> state
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
        rho = np.outer(plus, plus.conj())

        D_rho = np.zeros_like(rho)
        for L in p.collapse_operators:
            Ld = L.conj().T
            D_rho += L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)

        # The off-diagonal decay should be dominated by gamma_phi
        total_rate = gamma1 / 2.0 + gamma_phi
        expected = -total_rate * rho[0, 1]
        np.testing.assert_allclose(D_rho[0, 1], expected, rtol=1e-6)

    def test_two_qubit_per_qubit_rates(self):
        """Each qubit's dephasing operator should act locally."""
        T1 = [1e-3, 2e-3]
        T2 = [5e-4, 1e-3]
        cfg = _make_config(T1=T1, T2=T2, n_qubits=2)
        p = Initialise(cfg)
        ops = p.collapse_operators
        assert ops is not None
        # Should have 4 ops (2 per qubit)
        assert ops.shape[0] == 4
        # Each L†L should be a valid positive semidefinite operator
        for L in ops:
            LdL = L.conj().T @ L
            eigvals = np.linalg.eigvalsh(LdL)
            assert np.all(eigvals >= -1e-15)


# ---------------------------------------------------------------------------
# T1/T2 input validation
# ---------------------------------------------------------------------------


class TestT1T2Validation:
    """build_collapse_operators should reject invalid T1/T2 inputs."""

    def test_t2_exceeds_2t1(self):
        with pytest.raises(ValueError, match="must be <= 2\\*T1"):
            _make_config(T1=[1e-3], T2=[3e-3])
            Initialise(_make_config(T1=[1e-3], T2=[3e-3]))

    def test_t1_zero(self):
        with pytest.raises(ValueError, match="T1 must be positive and finite"):
            Initialise(_make_config(T1=[0.0], T2=[0.0]))

    def test_t2_zero(self):
        with pytest.raises(ValueError, match="T2 must be positive and finite"):
            Initialise(_make_config(T1=[1e-3], T2=[0.0]))

    def test_t1_negative(self):
        with pytest.raises(ValueError, match="T1 must be positive and finite"):
            Initialise(_make_config(T1=[-1e-3], T2=[5e-4]))

    def test_t2_negative(self):
        with pytest.raises(ValueError, match="T2 must be positive and finite"):
            Initialise(_make_config(T1=[1e-3], T2=[-5e-4]))

    def test_t1_inf(self):
        with pytest.raises(ValueError, match="T1 must be positive and finite"):
            Initialise(_make_config(T1=[np.inf], T2=[1e-3]))

    def test_t2_inf(self):
        with pytest.raises(ValueError, match="T2 must be positive and finite"):
            Initialise(_make_config(T1=[1e-3], T2=[np.inf]))

    def test_t1_nan(self):
        with pytest.raises(ValueError, match="T1 must be positive and finite"):
            Initialise(_make_config(T1=[np.nan], T2=[1e-3]))


# ---------------------------------------------------------------------------
# Dissipative + non-2-level model guard
# ---------------------------------------------------------------------------


class TestDissipativeModelGuard:
    """Dissipative mode must reject non-2-level Hamiltonian models."""

    def test_duffing_dissipative_raises(self):
        """Duffing (3-level) + dissipative should fail with a clear error."""
        cfg = _make_config(T1=[1e-3], T2=[5e-4])
        cfg["hamiltonian_type"] = "duffing_transmon"
        cfg["parameters"]["anharmonicities"] = np.array([-330e6])
        with pytest.raises(ValueError, match="not yet supported.*local dimension > 2"):
            Initialise(cfg)

    def test_superconducting_dissipative_allowed(self):
        """Superconducting (2-level) + dissipative should work."""
        cfg = _make_config(T1=[1e-3], T2=[5e-4])
        cfg["hamiltonian_type"] = "superconducting"
        p = Initialise(cfg)
        assert p.collapse_operators is not None


# ---------------------------------------------------------------------------
# Lindblad dissipator tensor operation
# ---------------------------------------------------------------------------


class TestLindbladDissipator:
    """Test the core lindblad_dissipator function."""

    def test_dissipator_trace_zero(self):
        """Tr(D[L](rho)) = 0 for any valid rho and L (trace preservation)."""
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import lindblad_dissipator

        cfg = _make_config(T1=[1e-3], T2=[5e-4])
        p = Initialise(cfg)

        # Random valid density matrix
        psi = np.array([0.6, 0.8j], dtype=complex)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())

        collapse_t = torch.as_tensor(p.collapse_operators, dtype=torch.complex128)
        rho_t = torch.as_tensor(rho, dtype=torch.complex128).unsqueeze(0)

        D_rho = lindblad_dissipator(rho_t, collapse_t)
        trace = torch.diagonal(D_rho, dim1=-2, dim2=-1).sum(dim=-1)
        assert torch.allclose(trace, torch.zeros_like(trace), atol=1e-14)

    def test_dissipator_preserves_hermiticity(self):
        """D[L](rho) should be Hermitian if rho is Hermitian."""
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import lindblad_dissipator

        cfg = _make_config(T1=[1e-3], T2=[5e-4])
        p = Initialise(cfg)

        psi = np.array([0.6, 0.8j], dtype=complex)
        psi /= np.linalg.norm(psi)
        rho = np.outer(psi, psi.conj())

        collapse_t = torch.as_tensor(p.collapse_operators, dtype=torch.complex128)
        rho_t = torch.as_tensor(rho, dtype=torch.complex128).unsqueeze(0)

        D_rho = lindblad_dissipator(rho_t, collapse_t)
        D_np = D_rho.squeeze(0).numpy()
        np.testing.assert_allclose(D_np, D_np.conj().T, atol=1e-14)
