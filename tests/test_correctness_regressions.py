"""Regression tests for the correctness fixes listed in the changelog.

Each class pins down one behaviour: the objective metrics, dissipative
propagation, Hamiltonian conventions, ensemble construction, analysis replay
and reporting.  Scientific assertions use float64/complex128 and independent
analytical or independently propagated references rather than the
implementation under test.
"""

import copy

import numpy as np
import pytest
import torch

from ctrl_freeq.api import CtrlFreeQAPI

TWO_PI = 2 * np.pi

# --- single-qubit stabilizer states: an exact 2-design for d = 2 -----------
_STABILIZER_1Q = [
    np.array([1, 0], dtype=complex),
    np.array([0, 1], dtype=complex),
    np.array([1, 1], dtype=complex) / np.sqrt(2),
    np.array([1, -1], dtype=complex) / np.sqrt(2),
    np.array([1, 1j], dtype=complex) / np.sqrt(2),
    np.array([1, -1j], dtype=complex) / np.sqrt(2),
]


def make_config(
    n_qubits=1,
    initial_states=None,
    target_states=None,
    coverage=None,
    n_points=8,
    n_para=4,
    h0_snapshots=1,
    rabi_snapshots=1,
    sigma_delta=None,
    sigma_omega=None,
    sigma_J=0.0,
    J=None,
    delta=None,
    space="hilbert",
    dissipation=None,
    hamiltonian_type=None,
    extra_parameters=None,
    algorithm="l-bfgs",
    max_iter=2,
    wf_type="cheb",
    wf_mode="cart",
    pulse_bandwidth=None,
    ratio_factor=None,
    profile_order=None,
    pulse_duration=2e-7,
    omega_r_max=4e7,
    sw=5e6,
):
    """Build a minimal but complete configuration dict."""
    if delta is None:
        delta = [1e7] * n_qubits
    if J is None:
        J = [[0.0] * n_qubits for _ in range(n_qubits)]
        for i in range(n_qubits - 1):
            J[i][i + 1] = 1.0e7
    cfg = {
        "qubits": [f"q{i + 1}" for i in range(n_qubits)],
        "parameters": {
            "Delta": list(delta),
            "sigma_Delta": list(sigma_delta or [0.0] * n_qubits),
            "Omega_R_max": [omega_r_max] * n_qubits,
            "sigma_Omega_R_max": list(sigma_omega or [0.0] * n_qubits),
            "pulse_duration": [pulse_duration] * n_qubits,
            "point_in_pulse": [n_points] * n_qubits,
            "wf_type": [wf_type] * n_qubits,
            "wf_mode": [wf_mode] * n_qubits,
            "amplitude_envelope": ["gn"] * n_qubits,
            "amplitude_order": [1] * n_qubits,
            "coverage": list(coverage or ["single"] * n_qubits),
            "sw": [sw] * n_qubits,
            "pulse_offset": [0.0] * n_qubits,
            "pulse_bandwidth": list(pulse_bandwidth or [5e5] * n_qubits),
            "ratio_factor": list(ratio_factor or [0.5] * n_qubits),
            "profile_order": list(profile_order or [2] * n_qubits),
            "n_para": [n_para] * n_qubits,
            "J": J,
            "sigma_J": sigma_J,
            "coupling_type": "XY",
        },
        "initial_states": initial_states or [["Z"] * n_qubits],
        "target_states": target_states or {"Axis": [["-Z"] * n_qubits]},
        "optimization": {
            "space": space,
            "H0_snapshots": h0_snapshots,
            "Omega_R_snapshots": rabi_snapshots,
            "algorithm": algorithm,
            "max_iter": max_iter,
            "targ_fid": 0.999,
        },
    }
    if dissipation is not None:
        cfg["optimization"]["dissipation_mode"] = "dissipative"
        cfg["parameters"]["T1"] = list(dissipation["T1"])
        cfg["parameters"]["T2"] = list(dissipation["T2"])
    if hamiltonian_type is not None:
        cfg["hamiltonian_type"] = hamiltonian_type
    if extra_parameters:
        cfg["parameters"].update(extra_parameters)
    return cfg


def params_from(cfg):
    return CtrlFreeQAPI(copy.deepcopy(cfg)).parameters


# ======================================================================
# Gate targets must be scored over the whole computational subspace
# ======================================================================


class TestGateObjective:
    def test_identity_evolution_vs_z_target_is_one_third(self):
        """A pulse that does nothing scores 1/3 against Z, not 1."""
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import fidelity_gate_hilbert

        basis = torch.eye(2, dtype=torch.complex128)
        states = basis.T.reshape(2, 2)  # identity evolution: |r> stays |r>
        gate_z = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128)
        targets = torch.stack([gate_z @ basis[:, r] for r in range(2)])

        fidelity = fidelity_gate_hilbert(states, targets, n_rows=2, d=2)
        assert fidelity.item() == pytest.approx(1 / 3, abs=1e-12)

    def test_state_transfer_on_one_input_would_have_scored_one(self):
        """The old behaviour: |0> alone cannot distinguish I from Z."""
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import fidelity_hilbert

        ket0 = torch.tensor([[1.0 + 0j, 0.0 + 0j]], dtype=torch.complex128)
        assert fidelity_hilbert(ket0, ket0).item() == pytest.approx(1.0)

    @pytest.mark.parametrize("gate_name", ["CNOT", "CX"])
    def test_identity_evolution_vs_cnot_is_zero_point_four(self, gate_name):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import fidelity_gate_hilbert

        cfg = make_config(
            n_qubits=2,
            initial_states=[["Z", "-Z"]],
            target_states={"Gate": [gate_name]},
        )
        p = params_from(cfg)
        assert p.objective_mode == "gate"
        assert p.gate_name == "CNOT"

        states = torch.as_tensor(np.asarray(p.initials), dtype=torch.complex128)
        targets = torch.as_tensor(np.asarray(p.targets), dtype=torch.complex128)
        fidelity = fidelity_gate_hilbert(
            states, targets, n_rows=p.n_objective_rows, d=p.computational_dim
        )
        assert fidelity.item() == pytest.approx(0.4, abs=1e-12)

    def test_cnot_and_cx_configs_are_identical(self):
        """CX is an alias of CNOT and must produce the same objective."""
        cnot = params_from(
            make_config(
                n_qubits=2,
                initial_states=[["Z", "-Z"]],
                target_states={"Gate": ["CNOT"]},
            )
        )
        cx = params_from(
            make_config(
                n_qubits=2,
                initial_states=[["Z", "-Z"]],
                target_states={"Gate": ["CX"]},
            )
        )
        np.testing.assert_allclose(cnot.initials, cx.initials)
        np.testing.assert_allclose(cnot.targets, cx.targets)

    def test_gate_objective_does_not_depend_on_configured_inputs(self):
        """Plot inputs must not change the gate objective."""
        a = params_from(
            make_config(
                n_qubits=2,
                initial_states=[["Z", "Z"]],
                target_states={"Gate": ["CNOT"]},
            )
        )
        b = params_from(
            make_config(
                n_qubits=2,
                initial_states=[["X", "-Y"], ["Z", "Z"]],
                target_states={"Gate": ["CNOT", "CNOT"]},
            )
        )
        np.testing.assert_allclose(a.initials, b.initials)
        np.testing.assert_allclose(a.targets, b.targets)
        assert a.n_objective_rows == b.n_objective_rows == 4

    def test_different_gates_keep_state_transfer(self):
        p = params_from(
            make_config(
                n_qubits=2,
                initial_states=[["Z", "-Z"], ["X", "Z"]],
                target_states={"Gate": ["CNOT", "SWAP"]},
            )
        )
        assert p.objective_mode == "state_transfer"
        assert p.n_objective_rows == 2
        assert np.asarray(p.initials).shape[0] == 2 * p.n_drift_snapshots()


def _two_design_average_gate_fidelity(channel, gate, projector):
    """Independent reference: average over an exact single-qubit 2-design.

    ``channel`` maps a ``(D, D)`` matrix to a ``(D, D)`` matrix; ``projector``
    is the ``(D, 2)`` isometry ``V``.  Leaked population simply does not come
    back through ``V``, so nothing is renormalised away.
    """
    total = 0.0
    for psi in _STABILIZER_1Q:
        embedded = projector @ psi
        rho_out = channel(np.outer(embedded, embedded.conj()))
        back = projector.conj().T @ rho_out @ projector
        target = gate @ psi
        total += np.real(target.conj() @ back @ target)
    return total / len(_STABILIZER_1Q)


class TestChannelGateFidelity:
    def test_total_leak_channel_scores_zero(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import fidelity_gate_liouville
        from ctrl_freeq.setup.initialise_gui import pauli_string_basis

        V = np.zeros((3, 2), dtype=complex)
        V[0, 0] = V[1, 1] = 1.0
        leak = np.zeros((3, 3), dtype=complex)
        leak[2, 2] = 1.0

        paulis = pauli_string_basis(1)
        states = torch.stack(
            [torch.as_tensor(np.trace(V @ P @ V.conj().T) * leak) for P in paulis]
        )
        targets = torch.stack([torch.as_tensor(V @ P @ V.conj().T) for P in paulis])
        fidelity = fidelity_gate_liouville(states, targets, n_rows=4, d=2)
        assert fidelity.item() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("q", [0.0, 0.2, 0.6, 1.0])
    def test_leaking_channel_matches_two_design_reference(self, q):
        """A leak is penalised, and matches an independent 2-design average."""
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import fidelity_gate_liouville
        from ctrl_freeq.setup.initialise_gui import pauli_string_basis

        V = np.zeros((3, 2), dtype=complex)
        V[0, 0] = V[1, 1] = 1.0
        K0 = np.diag([1.0, np.sqrt(1 - q), 1.0]).astype(complex)
        K1 = np.zeros((3, 3), dtype=complex)
        K1[2, 1] = np.sqrt(q)

        def channel(rho):
            return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

        gate = np.eye(2, dtype=complex)
        paulis = pauli_string_basis(1)
        states = torch.stack(
            [torch.as_tensor(channel(V @ P @ V.conj().T)) for P in paulis]
        )
        targets = torch.stack(
            [
                torch.as_tensor(V @ (gate @ P @ gate.conj().T) @ V.conj().T)
                for P in paulis
            ]
        )

        got = fidelity_gate_liouville(states, targets, n_rows=4, d=2).item()
        expected = _two_design_average_gate_fidelity(channel, gate, V)
        assert got == pytest.approx(expected, abs=1e-12)
        if q > 0:
            assert got < 1.0

    def test_identity_gate_under_t1_t2_matches_analytic(self):
        """F = 1/2 + (2 exp(-T/T2) + exp(-T/T1)) / 6."""
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
            dissipative_channel,
            apply_channel,
            fidelity_gate_liouville,
        )
        from ctrl_freeq.setup.initialise_gui import pauli_string_basis

        T1, T2, T = 1.0, 0.7, 0.45
        gamma1 = 1.0 / T1
        gamma_phi = 1.0 / T2 - 1.0 / (2.0 * T1)

        collapse = torch.as_tensor(
            np.array(
                [
                    np.sqrt(gamma1) * np.array([[0, 1], [0, 0]], dtype=complex),
                    np.sqrt(gamma_phi / 2.0)
                    * np.array([[1, 0], [0, -1]], dtype=complex),
                ]
            ),
            dtype=torch.complex128,
        )
        channel = dissipative_channel(collapse, T)

        paulis = [torch.as_tensor(P) for P in pauli_string_basis(1)]
        initial = torch.stack(paulis)
        final = apply_channel(channel, initial)
        targets = torch.stack(paulis)

        got = fidelity_gate_liouville(final, targets, n_rows=4, d=2).item()
        expected = 0.5 + (2 * np.exp(-T / T2) + np.exp(-T / T1)) / 6
        assert got == pytest.approx(expected, abs=1e-10)


# ======================================================================
# Dissipative propagation must be a CPTP channel, not an Euler step
# ======================================================================


def _amplitude_damping_ops(T1, T2=None):
    ops = [np.sqrt(1.0 / T1) * np.array([[0, 1], [0, 0]], dtype=complex)]
    if T2 is not None:
        gamma_phi = 1.0 / T2 - 1.0 / (2.0 * T1)
        ops.append(
            np.sqrt(gamma_phi / 2.0) * np.array([[1, 0], [0, -1]], dtype=complex)
        )
    return torch.as_tensor(np.array(ops), dtype=torch.complex128)


class TestDissipativeChannel:
    def test_large_step_stays_physical(self):
        """T1=1, no drive, |1>, dt=2: ground population 1-exp(-2), not 2."""
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import state_lindblad

        collapse = _amplitude_damping_ops(1.0)
        rho = torch.tensor([[[0, 0], [0, 1]]], dtype=torch.complex128)
        identity = torch.eye(2, dtype=torch.complex128).unsqueeze(0)

        out = state_lindblad(identity, rho, 2.0, collapse)
        populations = torch.diagonal(out[0]).real

        assert populations[0].item() == pytest.approx(1 - np.exp(-2.0), abs=1e-12)
        assert populations[1].item() == pytest.approx(np.exp(-2.0), abs=1e-12)

    def test_channel_is_positive_trace_preserving_hermitian(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import state_lindblad

        collapse = _amplitude_damping_ops(1.0, 0.8)
        identity = torch.eye(2, dtype=torch.complex128).unsqueeze(0)
        rho = torch.tensor(
            [[[0.3, 0.4 + 0.1j], [0.4 - 0.1j, 0.7]]], dtype=torch.complex128
        )
        for dt in (0.1, 1.0, 5.0):
            out = state_lindblad(identity, rho, dt, collapse)[0]
            assert torch.trace(out).real.item() == pytest.approx(1.0, abs=1e-12)
            assert torch.allclose(out, out.conj().T, atol=1e-12)
            assert torch.linalg.eigvalsh(out).min().item() > -1e-12

    def test_zero_step_is_identity(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import state_lindblad

        collapse = _amplitude_damping_ops(1.0)
        identity = torch.eye(2, dtype=torch.complex128).unsqueeze(0)
        rho = torch.tensor([[[0.4, 0.2], [0.2, 0.6]]], dtype=torch.complex128)
        assert torch.allclose(state_lindblad(identity, rho, 0.0, collapse), rho)

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_invalid_step_is_rejected(self, bad):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import dissipative_channel

        with pytest.raises(ValueError):
            dissipative_channel(_amplitude_damping_ops(1.0), bad)

    def test_second_order_convergence_with_noncommuting_drive(self):
        """Strang splitting is second order: error falls ~4x per refinement."""
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
            exp_mat_exact,
            simulator_optimized,
            state_lindblad,
            dissipator_superoperator,
        )

        T = 0.9
        collapse = _amplitude_damping_ops(1.3, 1.0)
        # X drive does not commute with amplitude damping.
        H = torch.tensor([[[0.0, 0.7], [0.7, 0.0]]], dtype=torch.complex128)
        rho0 = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.complex128)

        # Independent reference: exp(T * L_total) with the full Liouvillian.
        eye = torch.eye(2, dtype=torch.complex128)
        commutator = -1j * (
            torch.kron(H[0], eye) - torch.kron(eye, H[0].transpose(-2, -1).contiguous())
        )
        L_total = commutator + dissipator_superoperator(collapse)
        exact = (torch.linalg.matrix_exp(L_total * T) @ rho0[0].reshape(4)).reshape(
            2, 2
        )

        errors = []
        for n_steps in (10, 20, 40):
            dt = T / n_steps
            Hp = torch.zeros(n_steps, 1, 2, 2, dtype=torch.complex128)
            out = simulator_optimized(
                H, Hp, dt, rho0, exp_mat_exact, state_lindblad, collapse_ops=collapse
            )
            errors.append((out[0] - exact).abs().max().item())

        assert errors[0] > errors[1] > errors[2]
        assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.25)
        assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.25)

    def test_numpy_path_uses_the_same_channel(self):
        from ctrl_freeq.evolution.time_evolution import (
            apply_multi_pulse_multi_qubits_lindblad,
        )

        T, n_steps = 0.6, 16
        collapse = np.array(
            [np.sqrt(1.0 / 1.1) * np.array([[0, 1], [0, 0]], dtype=complex)]
        )
        zeros = np.zeros(n_steps)
        Ix = 0.5 * np.array([[0, 1], [1, 0]], dtype=complex)
        Iy = 0.5 * np.array([[0, -1j], [1j, 0]], dtype=complex)
        rho0 = np.array([[0, 0], [0, 1]], dtype=complex)

        out = apply_multi_pulse_multi_qubits_lindblad(
            np.zeros((2, 2), dtype=complex),
            [(zeros, zeros, Ix, Iy)],
            T,
            [1.0],
            rho0,
            collapse,
        )
        assert np.real(out[1, 1]) == pytest.approx(np.exp(-T / 1.1), abs=1e-12)
        assert np.real(np.trace(out)) == pytest.approx(1.0, abs=1e-12)


class TestDissipativeGradients:
    def test_channel_keeps_a_tensor_duration_in_the_graph(self):
        """exp(dt*D) must stay differentiable in the duration itself."""
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import apply_channel, dissipative_channel

        collapse = _amplitude_damping_ops(1.0)
        rho = torch.tensor([[[0, 0], [0, 1]]], dtype=torch.complex128)
        dt = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)

        excited = apply_channel(dissipative_channel(collapse, dt), rho)[0, 1, 1].real
        assert excited.item() == pytest.approx(np.exp(-0.4), abs=1e-12)

        excited.backward()
        assert dt.grad is not None
        # d/dt exp(-t/T1) = -exp(-t/T1) with T1 = 1
        assert dt.grad.item() == pytest.approx(-np.exp(-0.4), abs=1e-9)

    def test_channel_still_accepts_a_plain_float_duration(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import apply_channel, dissipative_channel

        collapse = _amplitude_damping_ops(1.0)
        rho = torch.tensor([[[0, 0], [0, 1]]], dtype=torch.complex128)
        excited = apply_channel(dissipative_channel(collapse, 0.4), rho)[0, 1, 1].real
        assert excited.item() == pytest.approx(np.exp(-0.4), abs=1e-12)

    def test_dissipative_objective_gradient_matches_finite_differences(self):
        p = params_from(
            make_config(
                space="liouville",
                dissipation={"T1": [1e-6], "T2": [8e-7]},
                n_points=6,
                n_para=4,
            )
        )
        cost, grad = _objective_and_grad(p)
        numeric = _finite_difference_grad(p, cost)
        np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-7)


def _build_instance(p):
    """Build the CtrlFreeQ instance exactly as run_ctrl does."""
    from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
        CtrlFreeQ,
        exp_mat_exact,
        exp_mat_torch,
        state_hilbert,
        state_liouville,
        state_lindblad,
    )
    from ctrl_freeq.run.run_ctrl import build_fidelity_function
    from ctrl_freeq.setup.iterator_generation.generate_iterator import (
        h0_omega_1_iterator_torch,
    )
    from ctrl_freeq.setup.operator_generation.generate_operators import (
        create_hamiltonian_basis_torch,
    )
    from ctrl_freeq.make_pulse.waveform_gen_torch import (
        waveform_gen_cart,
        waveform_gen_polar,
        waveform_gen_polar_phase,
    )
    from ctrl_freeq.utils.conversion import array_to_tensor

    rabi = array_to_tensor(np.asarray(p.Omega_R))
    H0 = array_to_tensor(np.asarray(p.H0))
    initials = array_to_tensor(np.asarray(p.initials))
    targets = array_to_tensor(np.asarray(p.targets))
    n_h0, n_rabi = H0.size(0), rabi.size(0)
    H0, initials, targets = h0_omega_1_iterator_torch(H0, n_rabi, initials, targets)

    model = getattr(p, "hamiltonian_model", None)
    D = model.dim if model is not None else 2**p.n_qubits
    u_fun = exp_mat_exact if D == 2 else exp_mat_torch

    if getattr(p, "dissipation_mode", "non-dissipative") == "dissipative":
        state_fun = state_lindblad
        collapse = array_to_tensor(np.asarray(p.collapse_operators))
    elif p.space == "hilbert":
        state_fun, collapse = state_hilbert, None
    else:
        state_fun, collapse = state_liouville, None

    modes = {
        "cart": waveform_gen_cart,
        "polar": waveform_gen_polar,
        "polar_phase": waveform_gen_polar_phase,
    }
    wf_fun = [modes[m] for m in p.wf_mode]

    return CtrlFreeQ(
        p.n_para_updated,
        p.n_qubits,
        None if model is not None else create_hamiltonian_basis_torch(p.n_qubits),
        rabi,
        p.np_pulse,
        n_h0,
        n_rabi,
        array_to_tensor(np.asarray(p.mat)),
        H0,
        array_to_tensor(p.pulse_duration / p.np_pulse),
        initials,
        targets,
        wf_fun,
        u_fun,
        state_fun,
        build_fidelity_function(p, p.space, torch.device("cpu")),
        p.targ_fid,
        array_to_tensor(np.asarray(p.modulation_exponent)),
        collapse_ops=collapse,
        hamiltonian_model=model,
        control_ops=None if model is None else model.control_ops_tensor(),
    )


def _objective_and_grad(p, x=None):
    instance = _build_instance(p)
    if x is None:
        x = torch.as_tensor(np.asarray(p.x0_con), dtype=torch.float64)
    x = x.detach().clone().requires_grad_(True)
    cost = instance.objective_function(x)
    cost.backward()
    return instance.objective_function, x.grad.detach().numpy().copy()


def _finite_difference_grad(p, cost_fn, eps=1e-6):
    x0 = torch.as_tensor(np.asarray(p.x0_con), dtype=torch.float64)
    grad = np.zeros(x0.shape[0])
    with torch.no_grad():
        for i in range(x0.shape[0]):
            plus, minus = x0.clone(), x0.clone()
            plus[i] += eps
            minus[i] -= eps
            grad[i] = (cost_fn(plus).item() - cost_fn(minus).item()) / (2 * eps)
    return grad


# ======================================================================
# Two-level and Duffing transmons must share a convention
# ======================================================================


def _projected_duffing(detunings, coupling, anharmonicities):
    from ctrl_freeq.setup.hamiltonian_generation.duffing_transmon import (
        DuffingTransmonModel,
    )

    n = len(detunings)
    model = DuffingTransmonModel(n, anharmonicities=anharmonicities)
    H = model.build_drift([np.asarray(detunings)], [np.asarray(coupling)])[0]
    V = model.computational_projector()
    return V.conj().T @ H @ V


def _two_level(detunings, coupling, coupling_type="XY"):
    from ctrl_freeq.setup.hamiltonian_generation.superconducting import (
        SuperconductingQubitModel,
    )

    model = SuperconductingQubitModel(len(detunings), coupling_type=coupling_type)
    return model.build_drift([np.asarray(detunings)], [np.asarray(coupling)])[0]


def _differs_only_by_scalar_identity(a, b, atol=1e-6):
    diff = a - b
    scalar = diff[0, 0]
    return np.max(np.abs(diff - scalar * np.eye(diff.shape[0]))) < atol


class TestTransmonConventions:
    def test_exchange_matrix_element_matches_projected_duffing(self):
        g = np.array([[0.0, 3.0], [0.0, 0.0]])
        det = np.zeros(2)
        alpha = [-TWO_PI * 3e8, -TWO_PI * 3e8]
        two_level = _two_level(det, g)
        projected = _projected_duffing(det, g, alpha)
        # |01> <-> |10> matrix element
        assert two_level[1, 2] == pytest.approx(projected[1, 2], abs=1e-12)
        assert two_level[1, 2] == pytest.approx(3.0, abs=1e-12)

    def test_detuning_only_agrees_up_to_scalar_identity(self):
        det = np.array([TWO_PI * 1e7, -TWO_PI * 4e6])
        zero = np.zeros((2, 2))
        alpha = [-TWO_PI * 3e8, -TWO_PI * 2.8e8]
        assert _differs_only_by_scalar_identity(
            _projected_duffing(det, zero, alpha), _two_level(det, zero)
        )

    def test_detuning_and_exchange_agree_up_to_scalar_identity(self):
        det = np.array([TWO_PI * 1e7, -TWO_PI * 4e6])
        g = np.array([[0.0, TWO_PI * 5e6], [0.0, 0.0]])
        alpha = [-TWO_PI * 3e8, -TWO_PI * 2.8e8]
        assert _differs_only_by_scalar_identity(
            _projected_duffing(det, g, alpha), _two_level(det, g)
        )

    def test_stark_channel_matches_a_frequency_shift(self):
        """A positive s raises the qubit frequency, so it enters as -s(...) Z."""
        from ctrl_freeq.setup.hamiltonian_generation.superconducting import (
            SuperconductingQubitModel,
        )

        s = 0.4
        in_phase, quad, omega = 0.3, 0.5, 2.0
        model = SuperconductingQubitModel(1, stark_shift_coeffs=[s])
        u = model.control_amplitudes(
            torch.tensor([[in_phase]], dtype=torch.float64),
            torch.tensor([[quad]], dtype=torch.float64),
            torch.tensor([[omega]], dtype=torch.float64),
            n_h0=1,
        )
        shift = s * (in_phase**2 + quad**2) * omega**2
        assert u[0, 0, 2].item() == pytest.approx(-shift, abs=1e-12)

        # The resulting term must equal the drift built from that same shift.
        ctrl_ops = model.build_control_ops()
        stark_term = u[0, 0, 2].item() * ctrl_ops[2]
        drift_from_shift = model.build_drift([np.array([shift])], None)[0]
        np.testing.assert_allclose(stark_term, drift_from_shift, atol=1e-12)


# ======================================================================
# Perturbative static ZZ: sign, detuning dependence, validity guard
# ======================================================================


def _exact_zeta(detunings, alpha, g):
    """zeta = E11 - (E10 + E01) + E00 from the Duffing spectrum.

    Excitation number is conserved, so the levels are identified block by
    block; the single-excitation pair is used as a sum because the two levels
    hybridise when the detuning vanishes.
    """
    from ctrl_freeq.setup.hamiltonian_generation.duffing_transmon import (
        DuffingTransmonModel,
    )

    model = DuffingTransmonModel(2, anharmonicities=alpha)
    H = model.build_drift([np.asarray(detunings)], [np.array([[0.0, g], [0.0, 0.0]])])[
        0
    ]

    def idx(i, j):
        return i * 3 + j

    E00 = H[idx(0, 0), idx(0, 0)].real
    one = [idx(0, 1), idx(1, 0)]
    sum_E1 = np.linalg.eigvalsh(H[np.ix_(one, one)]).sum()
    two = [idx(1, 1), idx(2, 0), idx(0, 2)]
    w, v = np.linalg.eigh(H[np.ix_(two, two)])
    E11 = w[np.argmax(np.abs(v[0, :]) ** 2)]
    return E11 - sum_E1 + E00


class TestPerturbativeZZ:
    @pytest.mark.parametrize(
        "detuning_mhz, g_mhz, tol",
        [(500, 5, 0.01), (800, 3, 0.01), (1200, 8, 0.01)],
    )
    def test_matches_exact_spectrum_when_well_separated(self, detuning_mhz, g_mhz, tol):
        from ctrl_freeq.setup.hamiltonian_generation.superconducting import (
            SuperconductingQubitModel,
        )

        alpha = [-TWO_PI * 3e8, -TWO_PI * 3e8]
        det = np.array([TWO_PI * detuning_mhz * 1e6, 0.0])
        g = TWO_PI * g_mhz * 1e6
        model = SuperconductingQubitModel(
            2, coupling_type="XY+ZZ", anharmonicities=alpha
        )
        estimate = model.perturbative_zz(det, np.array([[0.0, g], [0.0, 0.0]]))[0, 1]
        exact = _exact_zeta(det, alpha, g)
        assert estimate == pytest.approx(exact, rel=tol)
        # The old formula had the opposite sign for negative anharmonicity.
        assert np.sign(estimate) == np.sign(exact)

    def test_equal_detuning_is_accurate_to_about_two_percent(self):
        from ctrl_freeq.setup.hamiltonian_generation.superconducting import (
            SuperconductingQubitModel,
        )

        alpha = [-TWO_PI * 3e8, -TWO_PI * 3e8]
        det = np.zeros(2)
        g = TWO_PI * 21e6
        model = SuperconductingQubitModel(2, coupling_type="ZZ", anharmonicities=alpha)
        estimate = model.perturbative_zz(det, np.array([[0.0, g], [0.0, 0.0]]))[0, 1]
        exact = _exact_zeta(det, alpha, g)
        rel_error = abs(estimate - exact) / abs(exact)
        # mixing is just under the 0.1 heuristic, yet the error is ~1.9%:
        # 0.1 is a small-mixing heuristic, not a 1% error bound.
        assert np.sqrt(2) * g / (TWO_PI * 3e8) < 0.1
        assert rel_error == pytest.approx(0.0192, abs=0.002)

    @pytest.mark.parametrize("detuning_mhz", [285, 200])
    def test_rejects_near_the_avoided_crossings(self, detuning_mhz):
        from ctrl_freeq.setup.hamiltonian_generation.superconducting import (
            SuperconductingQubitModel,
        )

        alpha = [-TWO_PI * 3e8, -TWO_PI * 3e8]
        det = np.array([TWO_PI * detuning_mhz * 1e6, 0.0])
        g = TWO_PI * 1e7
        model = SuperconductingQubitModel(2, coupling_type="ZZ", anharmonicities=alpha)
        with pytest.raises(ValueError, match="mixing"):
            model.perturbative_zz(det, np.array([[0.0, g], [0.0, 0.0]]))

    def test_calibrated_zz_bypasses_the_estimate(self):
        from ctrl_freeq.setup.hamiltonian_generation.superconducting import (
            SuperconductingQubitModel,
        )

        alpha = [-TWO_PI * 3e8, -TWO_PI * 3e8]
        det = np.array([TWO_PI * 2e8, 0.0])  # would be rejected by the guard
        g = TWO_PI * 1e7
        calibrated = np.array([[0.0, TWO_PI * 1e5], [0.0, 0.0]])
        model = SuperconductingQubitModel(
            2,
            coupling_type="ZZ",
            anharmonicities=alpha,
            zz_crosstalk=calibrated,
        )
        H = model.build_drift([det], [np.array([[0.0, g], [0.0, 0.0]])])[0]
        # |11> diagonal picks up zeta/4 with Z = sigma_z / 2
        assert np.isfinite(H).all()

    def test_estimate_uses_the_snapshot_detuning(self):
        from ctrl_freeq.setup.hamiltonian_generation.superconducting import (
            SuperconductingQubitModel,
        )

        alpha = [-TWO_PI * 3e8, -TWO_PI * 3e8]
        g = np.array([[0.0, TWO_PI * 5e6], [0.0, 0.0]])
        model = SuperconductingQubitModel(2, coupling_type="ZZ", anharmonicities=alpha)
        a = model.perturbative_zz(np.array([TWO_PI * 5e8, 0.0]), g)[0, 1]
        b = model.perturbative_zz(np.array([TWO_PI * 8e8, 0.0]), g)[0, 1]
        assert a != b


# ======================================================================
# Coupling uncertainty alone must not collapse the drift ensemble
# ======================================================================


class TestDriftEnsemble:
    @pytest.mark.parametrize("hamiltonian_type", [None, "superconducting"])
    def test_coupling_uncertainty_keeps_all_snapshots(self, hamiltonian_type):
        p = params_from(
            make_config(
                n_qubits=2,
                h0_snapshots=10,
                sigma_J=1e6,
                hamiltonian_type=hamiltonian_type,
                initial_states=[["Z", "-Z"]],
                target_states={"Axis": [["-Z", "Z"]]},
            )
        )
        assert p.n_drift_snapshots() == 10
        assert len(p.Jmat_instances) == 10
        assert len(p.H0) == 10 * p.n_objective_rows

        couplings = [inst[0, 1] for inst in p.Jmat_instances]
        assert all(c != 0 for c in couplings)
        assert len(set(couplings)) == 10

        drifts = np.asarray(p.H0)[:10]
        assert len({tuple(np.round(d.ravel(), 12)) for d in drifts}) == 10

    def test_fully_deterministic_drift_collapses_to_one_snapshot(self):
        p = params_from(
            make_config(
                n_qubits=2,
                h0_snapshots=10,
                sigma_J=0.0,
                initial_states=[["Z", "-Z"]],
                target_states={"Axis": [["-Z", "Z"]]},
            )
        )
        assert p.n_drift_snapshots() == 1
        assert len(p.Jmat_instances) == 1

    def test_offset_uncertainty_alone_still_expands(self):
        p = params_from(make_config(n_qubits=1, h0_snapshots=7, sigma_delta=[1e5]))
        assert p.n_drift_snapshots() == 7

    def test_mismatched_offset_lengths_are_rejected(self):
        p = params_from(make_config(n_qubits=2, h0_snapshots=4))
        p.offs = [np.zeros(4), np.zeros(3)]
        with pytest.raises(ValueError, match="inconsistent lengths"):
            p.generate_Omega_instances()

    def test_drift_ensemble_size_is_validated(self):
        p = params_from(make_config(n_qubits=1, h0_snapshots=3, sigma_delta=[1e5]))
        with pytest.raises(ValueError, match="drift snapshots were requested"):
            p._stack_H0_rows([np.eye(2)] * 2)


# ======================================================================
# Coverage masks are per qubit; band_selective rejects discrete targets
# ======================================================================


def _force_profiles(p, profiles):
    """Pin per-qubit coverage masks for a fixed number of drift snapshots."""
    p.excitation_profile = [np.asarray(prof) for prof in profiles]
    return p


class TestCoverageMasks:
    def _two_qubit_selective(self):
        cfg = make_config(
            n_qubits=2,
            h0_snapshots=4,
            coverage=["selective", "selective"],
            sigma_delta=[1e5, 1e5],
            initial_states=[["Z", "Z"]],
            target_states={"Axis": [["-Z", "-Z"]]},
        )
        p = params_from(cfg)
        assert p.n_drift_snapshots() == 4
        # snapshots: (in,in), (in,out), (out,in), (out,out)
        return _force_profiles(p, [[1, 1, 0, 0], [1, 0, 1, 0]])

    def test_axis_targets_are_a_product_over_qubits(self):
        p = self._two_qubit_selective()
        inits, targets = p.compute_targets_targ()
        assert targets.shape == (4, 4)

        expected_axes = [
            ["-Z", "-Z"],  # both in band
            ["-Z", "Z"],  # qubit 2 out of band keeps its initial axis
            ["Z", "-Z"],  # qubit 1 out of band
            ["Z", "Z"],  # neither in band -> identity
        ]
        for s, axes in enumerate(expected_axes):
            np.testing.assert_allclose(
                targets[s], p.create_state_vector_pure(axes), atol=1e-12
            )

    def test_gate_targets_need_every_qubit_in_band(self):
        cfg = make_config(
            n_qubits=2,
            h0_snapshots=4,
            coverage=["selective", "selective"],
            sigma_delta=[1e5, 1e5],
            initial_states=[["Z", "-Z"]],
            target_states={"Gate": ["CNOT"]},
        )
        p = params_from(cfg)
        _force_profiles(p, [[1, 1, 0, 0], [1, 0, 1, 0]])
        inits, targets = p.compute_targets_gate()

        gate = p.get_gate("CNOT")
        identity = np.eye(4, dtype=complex)
        n_snapshots = 4
        for r in range(4):
            row = inits[r * n_snapshots]
            for s in range(n_snapshots):
                expected_u = gate if s == 0 else identity
                np.testing.assert_allclose(
                    targets[r * n_snapshots + s], expected_u @ row, atol=1e-12
                )

    @pytest.mark.parametrize(
        "target_states",
        [{"Axis": [["-Z"]]}, {"Gate": ["X"]}],
    )
    def test_band_selective_rejects_discrete_targets(self, target_states):
        cfg = make_config(
            coverage=["band_selective"],
            h0_snapshots=4,
            target_states=target_states,
        )
        with pytest.raises(ValueError, match="band_selective coverage is not"):
            params_from(cfg)

    def test_band_selective_still_allows_rotation_targets(self):
        cfg = make_config(
            coverage=["band_selective"],
            h0_snapshots=4,
            target_states={"Phi": [["x"]], "Beta": [[180.0]]},
        )
        p = params_from(cfg)
        assert p.objective_mode == "state_transfer"
        assert np.asarray(p.targets).shape[0] == p.n_drift_snapshots()


# ======================================================================
# Observables and leakage outside the computational subspace
# ======================================================================


def _duffing_model(n_qubits=1):
    from ctrl_freeq.setup.hamiltonian_generation.duffing_transmon import (
        DuffingTransmonModel,
    )

    return DuffingTransmonModel(n_qubits, anharmonicities=[-TWO_PI * 3e8] * n_qubits)


class TestObservableEmbedding:
    @pytest.mark.parametrize("name", ["X", "Y", "Z"])
    def test_leaked_state_reports_zero(self, name):
        paulis = {
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
        }
        model = _duffing_model()
        embedded = model.embed_computational_operator(paulis[name])
        ket2 = np.array([0, 0, 1], dtype=complex)
        assert np.real(np.vdot(ket2, embedded @ ket2)) == pytest.approx(0.0)

    def test_gate_embedding_would_have_reported_plus_one(self):
        model = _duffing_model()
        gate_embedded = model.embed_computational_gate(
            np.array([[1, 0], [0, -1]], dtype=complex)
        )
        ket2 = np.array([0, 0, 1], dtype=complex)
        assert np.real(np.vdot(ket2, gate_embedded @ ket2)) == pytest.approx(1.0)

    def test_computational_states_keep_their_values(self):
        model = _duffing_model()
        z = model.embed_computational_operator(
            np.array([[1, 0], [0, -1]], dtype=complex)
        )
        for k, expected in ((0, 1.0), (1, -1.0)):
            v = np.zeros(3, dtype=complex)
            v[k] = 1.0
            assert np.real(np.vdot(v, z @ v)) == pytest.approx(expected)

    def test_configured_observables_use_the_operator_embedding(self):
        p = params_from(
            make_config(
                hamiltonian_type="duffing_transmon",
                extra_parameters={"anharmonicities": [-3e8]},
            )
        )
        ket2 = np.array([0, 0, 1], dtype=complex)
        for key in ("X_1", "Y_1", "Z_1"):
            value = np.real(np.vdot(ket2, p.obs_op[key] @ ket2))
            assert value == pytest.approx(0.0, abs=1e-12)


class TestLeakageTrajectories:
    def test_leaked_and_computational_states(self):
        model = _duffing_model()
        assert model.computational_leakage(
            np.array([0, 0, 1], dtype=complex)
        ) == pytest.approx(1.0)
        assert model.computational_leakage(
            np.array([1, 0, 0], dtype=complex)
        ) == pytest.approx(0.0)
        assert model.computational_leakage(
            np.array([0, 1, 0], dtype=complex)
        ) == pytest.approx(0.0)

    def test_two_leaked_qubits_total_one_not_two(self):
        model = _duffing_model(2)
        state = np.zeros(9, dtype=complex)
        state[3 * 2 + 2] = 1.0  # |22>
        assert model.computational_leakage(state) == pytest.approx(1.0)

    def test_density_matrix_leakage(self):
        model = _duffing_model()
        rho = np.diag([0.25, 0.25, 0.5]).astype(complex)
        assert model.computational_leakage(rho) == pytest.approx(0.5)

    def test_leakage_series_matches_population_in_level_two(self):
        """A driven qutrit's leakage equals |psi_2|^2 at every boundary."""
        from ctrl_freeq.visualisation.plotter import compute_and_store_evolution

        cfg = make_config(
            hamiltonian_type="duffing_transmon",
            extra_parameters={"anharmonicities": [-3e8]},
            n_points=12,
            n_para=4,
        )
        p = params_from(cfg)
        x = torch.full((sum(p.n_para_updated),), 0.6, dtype=torch.float64)
        _, _, history, _, leakage = compute_and_store_evolution(x, p, p.init[0])

        nominal = leakage["nominal"]
        assert nominal.shape == (p.np_pulse + 1,)
        # history is the sampled ensemble; with a single snapshot it is the
        # same trajectory, so |psi_2|^2 must equal the reported leakage.
        pop2 = np.abs(history[2, :, 0]) ** 2
        np.testing.assert_allclose(leakage["snapshots"][:, 0], pop2, atol=1e-10)
        assert nominal[0] == pytest.approx(0.0, abs=1e-12)
        assert nominal.max() > 0.0

    def test_visibility_rule(self):
        from ctrl_freeq.visualisation.plotter import _leakage_is_visible

        visible = {
            "nominal": np.array([0.0, 0.0036]),
            "snapshots": np.array([[0.0], [5.11e-11]]),
        }
        assert _leakage_is_visible(visible)
        hidden = {"nominal": np.zeros(2), "snapshots": np.zeros((2, 1))}
        assert not _leakage_is_visible(hidden)


# ======================================================================
# Selective sampling must not correlate qubits
# ======================================================================


class TestSamplingCorrelation:
    @pytest.mark.parametrize("coverage", ["selective", "band_selective"])
    def test_mixed_band_combinations_appear(self, coverage):
        np.random.seed(20260920)
        target_states = (
            {"Axis": [["-Z", "-Z"]]}
            if coverage == "selective"
            else {"Phi": [["x", "x"]], "Beta": [[180.0, 180.0]]}
        )
        p = params_from(
            make_config(
                n_qubits=2,
                h0_snapshots=100,
                coverage=[coverage] * 2,
                sigma_delta=[1e5, 1e5],
                target_states=target_states,
            )
        )

        in_band = []
        for q in range(2):
            low, high = p.frq_band[q]
            offs = np.asarray(p.offs[q])
            in_band.append((offs >= low) & (offs <= high))

        mixed = np.sum(in_band[0] != in_band[1])
        assert mixed > 0, "no mixed in-band/out-of-band snapshot was produced"

    @pytest.mark.parametrize("requested", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("ratio_factor", [1.0, 0.5, 0.0])
    @pytest.mark.parametrize("coverage", ["selective", "band_selective"])
    def test_every_requested_snapshot_is_produced(
        self, coverage, ratio_factor, requested
    ):
        """Splitting the out-of-band count over two tails must not drop one."""
        np.random.seed(1)
        target_states = (
            {"Axis": [["-Z"]]}
            if coverage == "selective"
            else {"Phi": [["x"]], "Beta": [[180.0]]}
        )
        p = params_from(
            make_config(
                h0_snapshots=requested,
                coverage=[coverage],
                sigma_delta=[1e5],
                ratio_factor=[ratio_factor],
                target_states=target_states,
            )
        )
        assert len(p.offs[0]) == requested
        assert p.n_drift_snapshots() == requested
        assert len(p.excitation_profile[0]) == requested
        assert len(p.H0) == requested * p.n_objective_rows

    @pytest.mark.parametrize("coverage", ["selective", "band_selective"])
    def test_per_qubit_sample_counts_are_preserved(self, coverage):
        np.random.seed(7)
        target_states = (
            {"Axis": [["-Z", "-Z"]]}
            if coverage == "selective"
            else {"Phi": [["x", "x"]], "Beta": [[180.0, 180.0]]}
        )
        p = params_from(
            make_config(
                n_qubits=2,
                h0_snapshots=100,
                coverage=[coverage] * 2,
                sigma_delta=[1e5, 1e5],
                ratio_factor=[0.5, 0.5],
                target_states=target_states,
            )
        )
        for q in range(2):
            offs = np.asarray(p.offs[q])
            assert offs.shape == (100,)
            low, high = p.frq_band[q]
            # ratio_factor 0.5 puts half the samples outside the band
            assert np.sum((offs >= low) & (offs <= high)) == 50
            centre = p.Delta[q]
            half_sw = p.sw[q] / 2
            assert offs.min() >= centre - half_sw - 1e-9
            assert offs.max() <= centre + half_sw + 1e-9


# ======================================================================
# Sampling times, propagation steps and plotted times must agree
# ======================================================================


class TestTiming:
    def test_waveform_times_are_interval_midpoints(self):
        p = params_from(make_config(n_points=8, pulse_duration=2e-7))
        dt = p.pulse_duration / p.np_pulse
        np.testing.assert_allclose(p.t, (np.arange(8) + 0.5) * dt, rtol=0, atol=0)
        np.testing.assert_allclose(np.diff(p.t), dt)

    def test_state_boundaries_end_exactly_at_T(self):
        p = params_from(make_config(n_points=8, pulse_duration=2e-7))
        boundaries = p.state_boundary_times()
        assert boundaries.shape == (p.np_pulse + 1,)
        assert boundaries[0] == 0.0
        assert boundaries[-1] == pytest.approx(p.pulse_duration, rel=1e-15)
        assert p.np_pulse * p.dt == pytest.approx(p.pulse_duration, rel=1e-15)

    def test_modulation_advances_by_exactly_offset_times_dt(self):
        offset_hz = 3e6
        p = params_from(
            make_config(
                n_points=10,
                extra_parameters={"pulse_offset": [offset_hz]},
            )
        )
        me = np.asarray(p.modulation_exponent)[:, 0]
        expected = np.exp(1j * TWO_PI * offset_hz * p.dt)
        ratios = me[1:] / me[:-1]
        np.testing.assert_allclose(ratios, expected, rtol=1e-12)

    def test_history_starts_at_the_initial_state_and_has_N_plus_1_entries(self):
        from ctrl_freeq.visualisation.plotter import compute_and_store_evolution

        p = params_from(make_config(n_points=9, n_para=4))
        x = torch.full((sum(p.n_para_updated),), 0.2, dtype=torch.float64)
        _, _, history, history_mean, _ = compute_and_store_evolution(x, p, p.init[0])

        assert history.shape[1] == p.np_pulse + 1
        assert history_mean.shape[1] == p.np_pulse + 1
        np.testing.assert_allclose(history[:, 0, 0], np.asarray(p.init[0]), atol=1e-14)
        np.testing.assert_allclose(
            history_mean[:, 0], np.asarray(p.init[0]), atol=1e-14
        )


# ======================================================================
# QR must not silently complete a rank-deficient basis
# ======================================================================


class TestRankDeficientBasis:
    def test_eight_chirps_on_ten_points_are_rejected(self):
        from ctrl_freeq.setup.basis_generation.mat_x0_gen import (
            chirp_matrix,
            mat_with_amplitude_and_qr,
        )

        mat = chirp_matrix(np.linspace(-1, 1, 10), 8)
        assert np.linalg.matrix_rank(mat) < 8
        with pytest.raises(ValueError, match="rank deficient"):
            mat_with_amplitude_and_qr(mat, np.zeros(16), "cart", "gn", 1)

    def test_duplicated_columns_are_rejected(self):
        from ctrl_freeq.setup.basis_generation.mat_x0_gen import (
            mat_with_amplitude_and_qr,
        )

        x = np.linspace(-1, 1, 20)
        mat = np.column_stack([np.ones_like(x), x, x])
        with pytest.raises(ValueError, match="rank deficient"):
            mat_with_amplitude_and_qr(mat, np.zeros(6), "cart", "gn", 1)

    def test_more_columns_than_samples_are_rejected(self):
        from ctrl_freeq.setup.basis_generation.mat_x0_gen import (
            chebyshev_matrix,
            mat_with_amplitude_and_qr,
        )

        mat = chebyshev_matrix(np.linspace(-1, 1, 4), 6)
        with pytest.raises(ValueError, match="sample points"):
            mat_with_amplitude_and_qr(mat, np.zeros(12), "cart", "gn", 1)

    def test_well_conditioned_bases_are_accepted(self):
        from ctrl_freeq.setup.basis_generation.mat_x0_gen import (
            chebyshev_matrix,
            chirp_matrix,
            mat_with_amplitude_and_qr,
        )

        for mat in (
            chebyshev_matrix(np.linspace(-1, 1, 50), 8),
            chirp_matrix(np.linspace(-1, 1, 50), 8),
        ):
            assert np.linalg.matrix_rank(mat) == 8
            out, _ = mat_with_amplitude_and_qr(mat, np.zeros(16), "cart", "gn", 1)
            assert out.shape == (2, 50, 8)

    def test_polar_phase_checks_only_the_factorised_matrix(self):
        from ctrl_freeq.setup.basis_generation.mat_x0_gen import (
            chebyshev_matrix,
            mat_with_amplitude_and_qr,
        )

        mat = chebyshev_matrix(np.linspace(-1, 1, 30), 5)
        out, x0 = mat_with_amplitude_and_qr(mat, np.zeros(5), "polar_phase", "gn", 1)
        assert out.shape == (2, 30, 5)
        assert x0.shape == (6,)


# ======================================================================
# Constant envelopes must not divide by zero
# ======================================================================


class TestConstantEnvelopes:
    @pytest.mark.parametrize("grid", [np.array([-1.0, 1.0]), np.array([0.0])])
    def test_two_point_envelopes_are_finite_and_flat(self, grid):
        from ctrl_freeq.setup.basis_generation.mat_x0_gen import (
            g_envelope,
            hs_envelope,
        )

        for envelope in (g_envelope(grid), hs_envelope(grid)):
            assert np.all(np.isfinite(envelope))
            np.testing.assert_allclose(envelope, np.ones_like(grid))

    def test_longer_envelopes_keep_their_shape(self):
        from ctrl_freeq.setup.basis_generation.mat_x0_gen import (
            g_envelope,
            hs_envelope,
        )

        x = np.linspace(-1, 1, 51)
        for envelope in (g_envelope(x), hs_envelope(x)):
            assert envelope.argmax() == 25
            assert envelope[0] < envelope[25]
            assert envelope[-1] < envelope[25]
            np.testing.assert_allclose(envelope, envelope[::-1], atol=1e-12)

    @pytest.mark.parametrize("envelope", ["gn", "hs", "quad"])
    def test_two_point_pulse_passes_through_setup(self, envelope):
        cfg = make_config(
            n_points=2,
            n_para=2,
            extra_parameters={"amplitude_envelope": [envelope]},
        )
        p = params_from(cfg)
        assert np.all(np.isfinite(np.asarray(p.mat)))

    def test_invalid_envelope_inputs_are_rejected(self):
        from ctrl_freeq.setup.basis_generation.mat_x0_gen import (
            amplitude_envelope,
            g_envelope,
        )

        with pytest.raises(ValueError):
            g_envelope(np.array([0.0, np.nan]))
        with pytest.raises(ValueError):
            g_envelope(np.linspace(-1, 1, 5), n=0)
        with pytest.raises(ValueError, match="Unknown amplitude envelope"):
            amplitude_envelope(np.linspace(-1, 1, 5), envelope="nope")


# ======================================================================
# band_selective bandwidth is the FWHM of the target profile
# ======================================================================


class TestBandSelectiveWidth:
    @pytest.mark.parametrize("order", [1, 2, 3, 7])
    def test_centre_is_one_and_both_edges_are_one_half(self, order):
        from ctrl_freeq.setup.initialise_gui import band_selective_profile

        centre, bw = 5.0, 4.0
        assert band_selective_profile(
            np.array([centre]), centre, bw, order
        ) == pytest.approx(1.0)
        edges = band_selective_profile(
            np.array([centre - bw / 2, centre + bw / 2]), centre, bw, order
        )
        np.testing.assert_allclose(edges, [0.5, 0.5], rtol=1e-12)

    def test_symmetric_and_decaying_outside_the_band(self):
        from ctrl_freeq.setup.initialise_gui import band_selective_profile

        offsets = np.linspace(-10, 10, 101)
        profile = band_selective_profile(offsets, 0.0, 4.0, 2)
        np.testing.assert_allclose(profile, profile[::-1], atol=1e-12)
        assert profile[offsets > 2.0].max() < 0.5
        assert np.all(np.diff(profile[offsets >= 0.0]) <= 1e-15)

    @pytest.mark.parametrize("bandwidth", [0.0, -1.0, np.nan, np.inf])
    def test_invalid_bandwidth_is_rejected(self, bandwidth):
        from ctrl_freeq.setup.initialise_gui import band_selective_profile

        with pytest.raises(ValueError, match="bandwidth"):
            band_selective_profile(np.array([0.0]), 0.0, bandwidth, 2)

    @pytest.mark.parametrize("order", [0, -1, 1.5])
    def test_invalid_order_is_rejected(self, order):
        from ctrl_freeq.setup.initialise_gui import band_selective_profile

        with pytest.raises(ValueError, match="profile_order"):
            band_selective_profile(np.array([0.0]), 0.0, 4.0, order)


# ======================================================================
# Coupling normalisation and noise across matrix representations
# ======================================================================


UPPER_J = [[0.0, 1.6e7], [0.0, 0.0]]
LOWER_J = [[0.0, 0.0], [1.6e7, 0.0]]
SYMMETRIC_J = [[0.0, 1.6e7], [1.6e7, 0.0]]


class TestCouplingRepresentations:
    @pytest.mark.parametrize("hamiltonian_type", [None, "superconducting"])
    def test_upper_lower_and_symmetric_give_the_same_drift(self, hamiltonian_type):
        drifts = []
        for J in (UPPER_J, LOWER_J, SYMMETRIC_J):
            p = params_from(
                make_config(
                    n_qubits=2,
                    J=J,
                    sigma_J=0.0,
                    hamiltonian_type=hamiltonian_type,
                    initial_states=[["Z", "-Z"]],
                    target_states={"Axis": [["-Z", "Z"]]},
                )
            )
            drifts.append(np.asarray(p.H0)[0])
        np.testing.assert_allclose(drifts[0], drifts[1], atol=1e-9)
        np.testing.assert_allclose(drifts[0], drifts[2], atol=1e-9)

    def test_duffing_accepts_a_lower_triangular_matrix(self):
        from ctrl_freeq.setup.hamiltonian_generation.duffing_transmon import (
            DuffingTransmonModel,
        )

        model = DuffingTransmonModel(2, anharmonicities=[-TWO_PI * 3e8] * 2)
        det = np.array([TWO_PI * 1e7, 0.0])
        upper = model.build_drift([det], [np.array(UPPER_J)])[0]
        lower = model.build_drift([det], [np.array(LOWER_J)])[0]
        np.testing.assert_allclose(upper, lower, atol=1e-9)

    def test_symmetric_matrix_with_noise_stays_symmetric(self):
        np.random.seed(11)
        p = params_from(
            make_config(
                n_qubits=2,
                J=SYMMETRIC_J,
                sigma_J=1e6,
                h0_snapshots=6,
                initial_states=[["Z", "-Z"]],
                target_states={"Axis": [["-Z", "Z"]]},
            )
        )
        assert len(p.Jmat_instances) == 6
        for instance in p.Jmat_instances:
            np.testing.assert_allclose(instance, instance.T, atol=0)
            # exactly one draw per pair, mirrored
            assert instance[0, 1] == instance[1, 0]
        values = [inst[0, 1] for inst in p.Jmat_instances]
        assert len(set(values)) == 6

    def test_zero_nominal_couplings_stay_zero(self):
        np.random.seed(3)
        p = params_from(
            make_config(
                n_qubits=3,
                J=[[0.0, 1.6e7, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                sigma_J=1e6,
                h0_snapshots=5,
                initial_states=[["Z", "-Z", "Z"]],
                target_states={"Axis": [["-Z", "Z", "-Z"]]},
            )
        )
        for instance in p.Jmat_instances:
            assert instance[0, 2] == 0.0
            assert instance[1, 2] == 0.0
            assert instance[0, 1] != 0.0

    def test_plotter_uses_the_same_normalisation(self):
        from ctrl_freeq.visualisation.plotter import get_Jmat_for_plotter

        np.random.seed(5)
        p = params_from(
            make_config(
                n_qubits=2,
                J=SYMMETRIC_J,
                sigma_J=1e6,
                initial_states=[["Z", "-Z"]],
                target_states={"Axis": [["-Z", "Z"]]},
            )
        )
        instances = get_Jmat_for_plotter(p, 4)
        assert len(instances) == 4
        for instance in instances:
            np.testing.assert_allclose(instance, instance.T, atol=0)

    def test_conflicting_triangles_are_rejected(self):
        from ctrl_freeq.setup.hamiltonian_generation.hamiltonians import (
            _symmetrise_coupling,
        )

        with pytest.raises(ValueError, match="asymmetric"):
            _symmetrise_coupling(np.array([[0.0, 1.0], [2.0, 0.0]]))


# ======================================================================
# One canonical (row, drift snapshot, Rabi snapshot) ordering
# ======================================================================


def _expanded_batch(p):
    from ctrl_freeq.setup.iterator_generation.generate_iterator import (
        h0_omega_1_iterator_torch,
    )
    from ctrl_freeq.utils.conversion import array_to_tensor

    H0 = array_to_tensor(np.asarray(p.H0))
    initials = array_to_tensor(np.asarray(p.initials))
    targets = array_to_tensor(np.asarray(p.targets))
    return h0_omega_1_iterator_torch(H0, len(p.Omega_R), initials, targets)


class TestBatchAlignment:
    def _config(self, **overrides):
        cfg = make_config(
            n_qubits=1,
            initial_states=[["Z"], ["X"]],
            target_states={"Axis": [["-Z"], ["Y"]]},
            h0_snapshots=2,
            rabi_snapshots=2,
            sigma_delta=[2e6],
            sigma_omega=[2e6],
            n_points=6,
            n_para=4,
        )
        cfg["parameters"].update(overrides)
        return cfg

    def test_every_row_sees_every_drift_and_rabi_snapshot(self):
        np.random.seed(2026)
        p = params_from(self._config())
        S, M, R = p.n_drift_snapshots(), len(p.Omega_R), p.n_objective_rows
        assert (S, M, R) == (2, 2, 2)

        drifts = np.asarray(p.H0)[:S]
        assert not np.allclose(drifts[0], drifts[1])

        H0_b, init_b, targ_b = _expanded_batch(p)
        assert H0_b.shape[0] == R * S * M

        seen = set()
        for r in range(R):
            for s in range(S):
                for m in range(M):
                    i = (r * S + s) * M + m
                    np.testing.assert_allclose(
                        init_b[i].numpy(), np.asarray(p.initial[r]), atol=1e-14
                    )
                    np.testing.assert_allclose(H0_b[i].numpy(), drifts[s], atol=1e-14)
                    np.testing.assert_allclose(
                        targ_b[i].numpy(),
                        np.asarray(p.targets[r * S + s]),
                        atol=1e-14,
                    )
                    seen.add((r, s, m))
        assert len(seen) == R * S * M

    def test_each_initial_state_is_paired_with_both_drifts(self):
        """The old ordering gave state 0 only drift a and state 1 only drift b."""
        np.random.seed(99)
        p = params_from(self._config())
        S = p.n_drift_snapshots()
        drifts = np.asarray(p.H0)[:S]
        H0_b, init_b, _ = _expanded_batch(p)

        for r in range(p.n_objective_rows):
            matched = set()
            for i in range(H0_b.shape[0]):
                if np.allclose(init_b[i].numpy(), np.asarray(p.initial[r])):
                    for s in range(S):
                        if np.allclose(H0_b[i].numpy(), drifts[s]):
                            matched.add(s)
            assert matched == set(range(S))

    def test_vectorised_objective_matches_an_explicit_loop(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
            exp_mat_exact,
            pulse_para,
            state_hilbert,
        )
        from ctrl_freeq.make_pulse.waveform_gen_torch import waveform_gen_cart
        from ctrl_freeq.utils.conversion import array_to_tensor

        np.random.seed(4242)
        p = params_from(self._config())
        S, M, R = p.n_drift_snapshots(), len(p.Omega_R), p.n_objective_rows

        x = torch.as_tensor(np.asarray(p.x0_con), dtype=torch.float64)
        instance = _build_instance(p)
        with torch.no_grad():
            vectorised = instance.objective_function(x)
        fid_vectorised = instance.fid.item()

        # Independent reference: propagate every (row, drift, rabi) tuple alone.
        params = torch.split(x, list(p.n_para_updated))
        _, cx, cy = pulse_para(
            p.n_qubits,
            params,
            array_to_tensor(np.asarray(p.mat)),
            [waveform_gen_cart],
            array_to_tensor(np.asarray(p.modulation_exponent)),
        )
        op = array_to_tensor(np.asarray(p.op["X_1"]))
        opy = array_to_tensor(np.asarray(p.op["Y_1"]))
        drifts = array_to_tensor(np.asarray(p.H0)[:S])
        rabi = np.asarray(p.Omega_R)
        dt = float(p.pulse_duration) / p.np_pulse

        total = 0.0
        for r in range(R):
            for s in range(S):
                for m in range(M):
                    state = array_to_tensor(np.asarray(p.initial[r])).reshape(1, -1)
                    for k in range(p.np_pulse):
                        H = drifts[s].unsqueeze(0) + rabi[m, 0] * (
                            cx[k, 0] * op + cy[k, 0] * opy
                        ).unsqueeze(0)
                        state = state_hilbert(exp_mat_exact(H, dt), state)
                    target = array_to_tensor(np.asarray(p.targets[r * S + s]))
                    total += abs(torch.vdot(target, state[0])) ** 2
        reference = (total / (R * S * M)).item()

        assert fid_vectorised == pytest.approx(reference, rel=1e-10)
        assert vectorised.item() == pytest.approx(
            -reference + instance.pen.item(), rel=1e-10
        )

    def test_gradients_match_finite_differences(self):
        np.random.seed(17)
        p = params_from(self._config())
        cost_fn, grad = _objective_and_grad(p)
        numeric = _finite_difference_grad(p, cost_fn)
        np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-8)

    @pytest.mark.parametrize("hamiltonian_type", [None, "superconducting"])
    def test_model_and_legacy_paths_use_the_same_ordering(self, hamiltonian_type):
        np.random.seed(5)
        cfg = make_config(
            n_qubits=2,
            initial_states=[["Z", "-Z"], ["X", "Z"]],
            target_states={"Axis": [["-Z", "Z"], ["Y", "Z"]]},
            h0_snapshots=3,
            sigma_delta=[1e6, 1e6],
            hamiltonian_type=hamiltonian_type,
        )
        p = params_from(cfg)
        S, R = p.n_drift_snapshots(), p.n_objective_rows
        assert (S, R) == (3, 2)
        assert len(p.H0) == S * R
        for r in range(R):
            for s in range(S):
                np.testing.assert_allclose(
                    np.asarray(p.initials[r * S + s]),
                    np.asarray(p.initial[r]),
                    atol=1e-14,
                )
                np.testing.assert_allclose(
                    np.asarray(p.H0[r * S + s]), np.asarray(p.H0[s]), atol=1e-14
                )


# ======================================================================
# Analysis must replay the physics the optimizer ran
# ======================================================================


def _objective_final_states(p, x):
    """Final states from the optimizer's own simulator, for comparison."""
    from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
        pulse_hamiltonian,
        pulse_hamiltonian_generic,
        pulse_para,
        simulator_optimized,
    )
    from ctrl_freeq.make_pulse.waveform_gen_torch import waveform_gen_cart
    from ctrl_freeq.setup.operator_generation.generate_operators import (
        create_hamiltonian_basis_torch,
    )
    from ctrl_freeq.utils.conversion import array_to_tensor
    from ctrl_freeq.visualisation.plotter import _plotter_evolution_functions

    params = torch.split(x.detach(), list(p.n_para_updated))
    _, cx, cy = pulse_para(
        p.n_qubits,
        params,
        array_to_tensor(np.asarray(p.mat)),
        [waveform_gen_cart] * p.n_qubits,
        array_to_tensor(np.asarray(p.modulation_exponent)),
    )
    rabi = array_to_tensor(np.asarray(p.Omega_R))
    n_h0 = len(p.H0)
    model = getattr(p, "hamiltonian_model", None)
    if model is not None:
        Hp = pulse_hamiltonian_generic(
            model.control_amplitudes(cx, cy, rabi, n_h0), model.control_ops_tensor()
        )
    else:
        Hp = pulse_hamiltonian(
            cx,
            cy,
            rabi,
            create_hamiltonian_basis_torch(p.n_qubits),
            p.np_pulse,
            n_h0,
            rabi.shape[0],
            p.n_qubits,
        )
    H0_b, init_b, _ = _expanded_batch(p)
    u_fun, state_fun, collapse = _plotter_evolution_functions(p)
    return simulator_optimized(
        H0_b,
        Hp,
        array_to_tensor(p.pulse_duration / p.np_pulse),
        init_b,
        u_fun,
        state_fun,
        collapse_ops=collapse,
    )


class TestAnalysisReplay:
    @pytest.mark.parametrize(
        "cfg_kwargs",
        [
            dict(),
            dict(space="liouville", dissipation={"T1": [1e-6], "T2": [8e-7]}),
        ],
        ids=["unitary", "dissipative"],
    )
    def test_plot_final_state_matches_the_objective(self, cfg_kwargs):
        from ctrl_freeq.visualisation.plotter import compute_and_store_evolution

        np.random.seed(31)
        cfg = make_config(
            n_points=10,
            n_para=4,
            h0_snapshots=2,
            rabi_snapshots=2,
            sigma_delta=[1e6],
            sigma_omega=[1e6],
            **cfg_kwargs,
        )
        p = params_from(cfg)
        x = torch.as_tensor(np.asarray(p.x0_con), dtype=torch.float64)

        objective = _objective_final_states(p, x)
        _, _, history, _, _ = compute_and_store_evolution(x, p, p.init[0])

        S, M = p.n_drift_snapshots(), len(p.Omega_R)
        for s in range(S):
            for m in range(M):
                expected = objective[(0 * S + s) * M + m].detach().numpy()
                if p.space == "hilbert":
                    got = history[:, -1, s * M + m]
                else:
                    got = history[:, :, -1, s * M + m]
                np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_zero_drive_dissipative_replay_decays(self):
        """T = T1, no drive, |1>: excited population exp(-1), not 1."""
        from ctrl_freeq.visualisation.plotter import compute_and_store_evolution

        T1 = 1e-6
        p = params_from(
            make_config(
                space="liouville",
                dissipation={"T1": [T1], "T2": [T1]},
                pulse_duration=T1,
                n_points=40,
                n_para=4,
                initial_states=[["-Z"]],
                target_states={"Axis": [["Z"]]},
            )
        )
        x = torch.zeros(sum(p.n_para_updated), dtype=torch.float64)
        _, _, history, history_mean, _ = compute_and_store_evolution(x, p, p.init[0])

        excited = np.real(history_mean[1, 1, -1])
        assert excited == pytest.approx(np.exp(-1.0), abs=1e-6)

    def test_stark_channels_reach_the_right_qubits(self):
        """Pair-indexed controls drove qubit 2 with Z0 and X1."""
        from ctrl_freeq.visualisation.plotter import (
            _plotter_pulse_hamiltonian,
            _plotter_rabi,
        )
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import pulse_hamiltonian_generic

        p = params_from(
            make_config(
                n_qubits=2,
                hamiltonian_type="superconducting",
                extra_parameters={"stark_shift_coeffs": [0.3, 0.7]},
                initial_states=[["Z", "-Z"]],
                target_states={"Axis": [["-Z", "Z"]]},
                n_points=6,
            )
        )
        model = p.hamiltonian_model
        assert model.n_controls == 6  # [X0, Y0, Z0, X1, Y1, Z1]

        torch.manual_seed(0)
        cx = torch.randn(6, 2, dtype=torch.float64)
        cy = torch.randn(6, 2, dtype=torch.float64)
        rabi = _plotter_rabi(p, nominal=True)

        got = _plotter_pulse_hamiltonian(p, cx, cy, rabi, n_h0=1)
        expected = pulse_hamiltonian_generic(
            model.control_amplitudes(cx, cy, rabi, 1), model.control_ops_tensor()
        )
        torch.testing.assert_close(got, expected)

        # The pair-indexed version would have used ops[2] (= Z0) and ops[3]
        # (= X1) for qubit 2 and dropped the Stark power terms entirely.
        ops = model.build_control_ops()
        naive = torch.zeros_like(got)
        for i in range(2):
            Ix = torch.as_tensor(ops[2 * i], dtype=torch.complex128)
            Iy = torch.as_tensor(ops[2 * i + 1], dtype=torch.complex128)
            naive += rabi[0, i] * (
                cx[:, i].to(torch.complex128).reshape(-1, 1, 1, 1) * Ix
                + cy[:, i].to(torch.complex128).reshape(-1, 1, 1, 1) * Iy
            )
        assert not torch.allclose(got, naive)

    def test_excitation_profile_sweep_uses_the_shared_kernel(self):
        from ctrl_freeq.visualisation.plotter import (
            get_final_rho_for_excitation_profile,
            get_H0_for_plotter,
            _plotter_propagate,
            _plotter_rabi,
            _plotter_waveforms,
        )

        p = params_from(make_config(n_points=8, n_para=4))
        x = torch.as_tensor(np.asarray(p.x0_con), dtype=torch.float64)

        states = get_final_rho_for_excitation_profile(x, p, p.init[0], 5)
        assert len(states) == 5

        _, cx, cy = _plotter_waveforms(x, p)
        reference = _plotter_propagate(
            p,
            get_H0_for_plotter(p, 5),
            _plotter_rabi(p, nominal=True),
            cx,
            cy,
            p.init[0],
            record=False,
        )
        for i, state in enumerate(states):
            np.testing.assert_allclose(state, reference[i].numpy(), atol=1e-14)

    def test_nominal_and_sampled_provenance_are_distinct(self):
        from ctrl_freeq.visualisation.plotter import compute_and_store_evolution

        np.random.seed(64)
        p = params_from(
            make_config(
                n_points=8,
                n_para=4,
                h0_snapshots=3,
                rabi_snapshots=2,
                sigma_delta=[5e6],
                sigma_omega=[5e6],
            )
        )
        x = torch.as_tensor(np.asarray(p.x0_con), dtype=torch.float64)
        _, _, history, history_mean, _ = compute_and_store_evolution(x, p, p.init[0])

        # sampled ensemble covers drift x Rabi
        assert history.shape[-1] == p.n_drift_snapshots() * len(p.Omega_R)
        # the nominal trajectory is not simply one of the samples
        assert not any(
            np.allclose(history_mean[:, -1], history[:, -1, j])
            for j in range(history.shape[-1])
        )


# ======================================================================
# Duffing density matrices embed as V rho V^dag
# ======================================================================


class TestDuffingLiouville:
    @pytest.mark.parametrize("n_qubits", [1, 2])
    def test_embedded_densities_are_square_and_physical(self, n_qubits):
        p = params_from(
            make_config(
                n_qubits=n_qubits,
                space="liouville",
                hamiltonian_type="duffing_transmon",
                extra_parameters={"anharmonicities": [-3e8] * n_qubits},
                initial_states=[["Z"] * n_qubits],
                target_states={"Axis": [["-Z"] * n_qubits]},
                n_points=6,
            )
        )
        dim = 3**n_qubits
        initials = np.asarray(p.initials)
        targets = np.asarray(p.targets)
        assert initials.shape[1:] == (dim, dim)
        assert targets.shape[1:] == (dim, dim)
        assert np.asarray(p.H0).shape[1:] == (dim, dim)

        for rho in list(initials) + list(targets):
            assert np.trace(rho).real == pytest.approx(1.0, abs=1e-12)
            np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
            assert np.linalg.eigvalsh(rho).min() > -1e-12

    def test_hilbert_and_liouville_agree_on_populations(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
            pulse_hamiltonian_generic,
            pulse_para,
            simulator_optimized,
            exp_mat_torch,
            state_hilbert,
            state_liouville,
        )
        from ctrl_freeq.make_pulse.waveform_gen_torch import waveform_gen_cart
        from ctrl_freeq.utils.conversion import array_to_tensor

        common = dict(
            hamiltonian_type="duffing_transmon",
            extra_parameters={"anharmonicities": [-3e8]},
            n_points=8,
            n_para=4,
        )
        p_h = params_from(make_config(space="hilbert", **common))
        p_l = params_from(make_config(space="liouville", **common))

        x = torch.as_tensor(np.asarray(p_h.x0_con), dtype=torch.float64)

        def propagate(p, state_fun):
            params = torch.split(x, list(p.n_para_updated))
            _, cx, cy = pulse_para(
                1,
                params,
                array_to_tensor(np.asarray(p.mat)),
                [waveform_gen_cart],
                array_to_tensor(np.asarray(p.modulation_exponent)),
            )
            model = p.hamiltonian_model
            rabi = array_to_tensor(np.asarray(p.Omega_R))
            Hp = pulse_hamiltonian_generic(
                model.control_amplitudes(cx, cy, rabi, len(p.H0)),
                model.control_ops_tensor(),
            )
            return simulator_optimized(
                array_to_tensor(np.asarray(p.H0)),
                Hp,
                array_to_tensor(p.pulse_duration / p.np_pulse),
                array_to_tensor(np.asarray(p.initials)),
                exp_mat_torch,
                state_fun,
            )

        psi = propagate(p_h, state_hilbert)[0].detach().numpy()
        rho = propagate(p_l, state_liouville)[0].detach().numpy()
        np.testing.assert_allclose(np.abs(psi) ** 2, np.real(np.diag(rho)), atol=1e-10)

    def test_dissipative_duffing_is_still_rejected(self):
        cfg = make_config(
            space="liouville",
            hamiltonian_type="duffing_transmon",
            extra_parameters={"anharmonicities": [-3e8]},
            dissipation={"T1": [1e-6], "T2": [8e-7]},
        )
        with pytest.raises(ValueError, match="Dissipative mode is not yet supported"):
            params_from(cfg)


# ======================================================================
# Differentiable Uhlmann fidelity for pure targets
# ======================================================================


class TestDifferentiableFidelity:
    def test_value_and_gradient_for_a_mixed_state(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import fidelity_liouville

        rho = torch.tensor(
            [[[0.3, 0.0], [0.0, 0.7]]], dtype=torch.complex128, requires_grad=True
        )
        sigma = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.complex128)

        value = fidelity_liouville(rho, sigma)
        assert value.item() == pytest.approx(0.3, abs=1e-12)

        value.backward()
        assert torch.isfinite(rho.grad).all()
        # d/d rho_00 Re Tr(rho sigma) = 1
        assert rho.grad[0, 0, 0].real.item() == pytest.approx(1.0, abs=1e-12)

    def test_degenerate_and_pure_states(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import fidelity_liouville

        sigma = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]], dtype=torch.complex128)
        maximally_mixed = torch.tensor(
            [[[0.5, 0.0], [0.0, 0.5]]], dtype=torch.complex128
        )
        assert fidelity_liouville(maximally_mixed, sigma).item() == pytest.approx(0.5)
        assert fidelity_liouville(sigma, sigma).item() == pytest.approx(1.0)

    def test_non_pure_targets_are_rejected(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import fidelity_liouville

        rho = torch.tensor([[[0.3, 0.0], [0.0, 0.7]]], dtype=torch.complex128)
        mixed_target = torch.tensor([[[0.5, 0.0], [0.0, 0.5]]], dtype=torch.complex128)
        with pytest.raises(ValueError, match="pure"):
            fidelity_liouville(rho, mixed_target)

    @pytest.mark.parametrize(
        "cfg_kwargs",
        [
            dict(space="liouville"),
            dict(space="liouville", dissipation={"T1": [1e-6], "T2": [8e-7]}),
        ],
        ids=["unitary", "dissipative"],
    )
    def test_coefficient_gradients_match_finite_differences(self, cfg_kwargs):
        p = params_from(make_config(n_points=6, n_para=4, **cfg_kwargs))
        cost_fn, grad = _objective_and_grad(p)
        numeric = _finite_difference_grad(p, cost_fn)
        np.testing.assert_allclose(grad, numeric, rtol=1e-5, atol=1e-8)

    def test_piecewise_gradients_are_finite_in_liouville_space(self):
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        cfg = make_config(space="liouville", n_points=6, max_iter=2)
        api = PiecewiseAPI(copy.deepcopy(cfg), method="cart")
        solution = api.run_optimization()
        assert torch.isfinite(solution).all()
        assert np.isfinite(api.parameters.final_fidelity)


# ======================================================================
# Fidelity, penalty and penalized score are separate quantities
# ======================================================================


def _make_reporter(cls, fid, pen):
    """Build a bare optimizer instance carrying only the reporting state."""
    from ctrl_freeq.utils.colored_logging import setup_colored_logging

    reporter = object.__new__(cls)
    reporter.iter = 0
    reporter.fid = torch.tensor(fid, dtype=torch.float64)
    reporter.pen = torch.tensor(pen, dtype=torch.float64)
    reporter.cost = -reporter.fid + reporter.pen
    reporter.fidelity_history = []
    reporter.penalty_history = []
    reporter.score_history = []
    reporter.exit_val = 2.0
    reporter.early_termination_flag = False
    reporter.early_termination_solution = None
    reporter.wf_method = "cart"
    reporter.logger = setup_colored_logging(level="ERROR")
    return reporter


class TestProgressReporting:
    @pytest.mark.parametrize("cls_name", ["CtrlFreeQ", "Piecewise"])
    def test_fidelity_penalty_and_score_are_recorded_separately(self, cls_name):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import CtrlFreeQ
        from ctrl_freeq.ctrlfreeq.piecewise import Piecewise

        cls = {"CtrlFreeQ": CtrlFreeQ, "Piecewise": Piecewise}[cls_name]
        reporter = _make_reporter(cls, 0.97, 0.09)
        fidelity, penalty_value, score = reporter._record_iteration()

        assert fidelity == pytest.approx(0.97)
        assert penalty_value == pytest.approx(0.09)
        assert score == pytest.approx(0.88)
        assert reporter.fidelity_history == [pytest.approx(0.97)]
        assert reporter.penalty_history == [pytest.approx(0.09)]
        assert reporter.score_history == [pytest.approx(0.88)]

    @pytest.mark.parametrize("cls_name", ["CtrlFreeQ", "Piecewise"])
    def test_cobyla_callback_records_the_same_quantities(self, cls_name):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import CtrlFreeQ
        from ctrl_freeq.ctrlfreeq.piecewise import Piecewise

        cls = {"CtrlFreeQ": CtrlFreeQ, "Piecewise": Piecewise}[cls_name]
        reporter = _make_reporter(cls, 0.97, 0.09)
        reporter.callback_function_cobyla(torch.zeros(1))
        assert reporter.fidelity_history == [pytest.approx(0.97)]
        assert reporter.penalty_history == [pytest.approx(0.09)]
        assert reporter.score_history == [pytest.approx(0.88)]
        assert not reporter.early_termination_flag

    def test_final_metrics_describe_the_returned_solution(self):
        cfg = make_config(n_points=8, n_para=4, max_iter=3)
        api = CtrlFreeQAPI(copy.deepcopy(cfg))
        solution = api.run_optimization()
        p = api.parameters

        instance = _build_instance(p)
        with torch.no_grad():
            instance.objective_function(solution.detach())

        assert p.final_fidelity == pytest.approx(instance.fid.item(), rel=1e-10)
        assert p.final_penalty == pytest.approx(instance.pen.item(), rel=1e-10)
        assert p.final_score == pytest.approx(
            p.final_fidelity - p.final_penalty, rel=1e-12
        )
        assert len(p.fidelity_history) == len(p.penalty_history)
        assert len(p.fidelity_history) == len(p.score_history)
        for f, pen, score in zip(
            p.fidelity_history, p.penalty_history, p.score_history
        ):
            assert score == pytest.approx(f - pen, abs=1e-12)


# ======================================================================
# Piecewise optimisation must honour the Hamiltonian model
# ======================================================================


class TestPiecewiseModels:
    def test_one_qutrit_piecewise_runs(self):
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        cfg = make_config(
            hamiltonian_type="duffing_transmon",
            extra_parameters={"anharmonicities": [-3e8]},
            n_points=6,
            max_iter=2,
        )
        api = PiecewiseAPI(copy.deepcopy(cfg), method="cart")
        solution = api.run_optimization()
        assert torch.isfinite(solution).all()
        assert api.parameters.hamiltonian_model.dim == 3

    def test_two_qutrit_piecewise_runs(self):
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        cfg = make_config(
            n_qubits=2,
            hamiltonian_type="duffing_transmon",
            extra_parameters={"anharmonicities": [-3e8, -3e8]},
            initial_states=[["Z", "-Z"]],
            target_states={"Axis": [["-Z", "Z"]]},
            n_points=5,
            max_iter=2,
        )
        api = PiecewiseAPI(copy.deepcopy(cfg), method="cart")
        solution = api.run_optimization()
        assert torch.isfinite(solution).all()
        assert api.parameters.hamiltonian_model.dim == 9

    def test_stark_controls_agree_with_the_basis_path(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import pulse_hamiltonian_generic
        from ctrl_freeq.ctrlfreeq.piecewise import Piecewise
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
            exp_mat_torch,
            state_hilbert,
            fidelity_hilbert,
        )
        from ctrl_freeq.utils.conversion import array_to_tensor

        p = params_from(
            make_config(
                n_qubits=2,
                hamiltonian_type="superconducting",
                extra_parameters={"stark_shift_coeffs": [0.3, 0.7]},
                initial_states=[["Z", "-Z"]],
                target_states={"Axis": [["-Z", "Z"]]},
                n_points=5,
            )
        )
        model = p.hamiltonian_model
        rabi = array_to_tensor(np.asarray(p.Omega_R))
        instance = Piecewise(
            n_qubits=2,
            op=None,
            rabi_freq=rabi,
            n_pulse=p.np_pulse,
            n_h0=len(p.H0),
            n_rabi=rabi.shape[0],
            H0=array_to_tensor(np.asarray(p.H0)),
            dt=array_to_tensor(p.pulse_duration / p.np_pulse),
            initial_state=array_to_tensor(np.asarray(p.initials)),
            target_state=array_to_tensor(np.asarray(p.targets)),
            wf_method="cart",
            u_fun=exp_mat_torch,
            state_fun=state_hilbert,
            fid_fun=fidelity_hilbert,
            targ_fid=0.999,
            me=array_to_tensor(np.asarray(p.modulation_exponent)),
            dtype=torch.float64,
            hamiltonian_model=model,
            control_ops=model.control_ops_tensor(),
        )
        assert instance.hamiltonian_model is model

        torch.manual_seed(1)
        cx = torch.randn(p.np_pulse, 2, dtype=torch.float64)
        cy = torch.randn(p.np_pulse, 2, dtype=torch.float64)
        expected = pulse_hamiltonian_generic(
            model.control_amplitudes(cx, cy, rabi, len(p.H0)),
            model.control_ops_tensor(),
        )
        got = pulse_hamiltonian_generic(
            instance.hamiltonian_model.control_amplitudes(cx, cy, rabi, len(p.H0)),
            instance.control_ops,
        )
        torch.testing.assert_close(got, expected)

    def test_propagator_is_chosen_by_matrix_dimension(self):
        """A 3-level model must not use the analytical 2x2 exponential."""
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        cfg = make_config(
            hamiltonian_type="duffing_transmon",
            extra_parameters={"anharmonicities": [-3e8]},
            n_points=5,
            max_iter=1,
        )
        api = PiecewiseAPI(copy.deepcopy(cfg), method="cart")
        api.run_optimization()
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import exp_mat_torch

        assert api._piecewise_instance.u_fun is exp_mat_torch

    def test_legacy_spin_chain_piecewise_still_uses_the_pauli_path(self):
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        cfg = make_config(n_points=5, max_iter=1)
        api = PiecewiseAPI(copy.deepcopy(cfg), method="cart")
        api.run_optimization()
        assert api._piecewise_instance.hamiltonian_model is None
        assert api._piecewise_instance.op is not None

    def test_gate_metric_and_dissipation_in_the_piecewise_path(self):
        """Piecewise must use the same objective selection as the basis path."""
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        gate_cfg = make_config(
            n_qubits=2,
            initial_states=[["Z", "-Z"]],
            target_states={"Gate": ["CNOT"]},
            n_points=5,
            max_iter=2,
        )
        api = PiecewiseAPI(copy.deepcopy(gate_cfg), method="cart")
        api.run_optimization()
        assert api.parameters.objective_mode == "gate"
        assert 0.0 <= api.parameters.final_fidelity <= 1.0

        dissipative_cfg = make_config(
            space="liouville",
            dissipation={"T1": [1e-6], "T2": [8e-7]},
            n_points=5,
            max_iter=2,
        )
        api = PiecewiseAPI(copy.deepcopy(dissipative_cfg), method="cart")
        solution = api.run_optimization()
        assert torch.isfinite(solution).all()
        assert 0.0 <= api.parameters.final_fidelity <= 1.0

    def test_dtype_is_consistent_across_the_piecewise_pipeline(self):
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        cfg = make_config(
            hamiltonian_type="superconducting",
            n_points=5,
            max_iter=1,
        )
        api = PiecewiseAPI(copy.deepcopy(cfg), method="cart")
        api.run_optimization()
        instance = api._piecewise_instance
        assert instance.control_ops.dtype == torch.complex128
        assert instance.H0.dtype == torch.complex128


# ======================================================================
# Amplitude-limit semantics and violations are explicit
# ======================================================================


class TestAmplitudeReport:
    def test_peak_uses_magnitude_not_signed_amplitude(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import amplitude_limit_report

        # A polar amplitude that dips negative: the magnitude peak is 0.8.
        amp = torch.tensor([[0.2], [-0.8], [0.5]], dtype=torch.float64)
        phase = torch.zeros_like(amp)
        cx = amp * torch.cos(phase)
        cy = amp * torch.sin(phase)
        report = amplitude_limit_report(cx, cy, [1.0e8])
        assert report[0]["normalized_peak"] == pytest.approx(0.8)
        assert report[0]["within_limit"]

    def test_multiple_qubits_are_reported_separately(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import amplitude_limit_report

        cx = torch.tensor([[0.3, 1.2], [0.1, 0.4]], dtype=torch.float64)
        cy = torch.zeros_like(cx)
        report = amplitude_limit_report(cx, cy, [1.0e8, 2.0e8])
        assert len(report) == 2
        assert report[0]["within_limit"]
        assert not report[1]["within_limit"]
        assert report[1]["excess"] == pytest.approx(0.2)
        assert report[1]["nominal_peak_rad_per_s"] == pytest.approx(1.2 * 2.0e8)

    def test_exactly_one_is_within_the_limit(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import amplitude_limit_report

        cx = torch.tensor([[1.0]], dtype=torch.float64)
        cy = torch.tensor([[0.0]], dtype=torch.float64)
        report = amplitude_limit_report(cx, cy, [1.0])
        assert report[0]["normalized_peak"] == pytest.approx(1.0)
        assert report[0]["within_limit"]
        assert report[0]["excess"] == 0.0

    def test_negative_rabi_gains_use_their_magnitude(self):
        """A negative sampled gain is a phase reversal, not a smaller drive."""
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import amplitude_limit_report

        cx = torch.tensor([[1.0]], dtype=torch.float64)
        cy = torch.zeros_like(cx)
        gains = torch.tensor([[-2.0], [1.0]], dtype=torch.float64)
        report = amplitude_limit_report(cx, cy, [1.0], rabi_samples=gains)
        assert report[0]["sampled_peak_rad_per_s"] == (
            pytest.approx(1.0),
            pytest.approx(2.0),
        )

    def test_sampled_rabi_gains_are_reported_separately(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import amplitude_limit_report

        cx = torch.tensor([[0.5]], dtype=torch.float64)
        cy = torch.zeros_like(cx)
        gains = torch.tensor([[0.9e8], [1.1e8]], dtype=torch.float64)
        report = amplitude_limit_report(cx, cy, [1.0e8], rabi_samples=gains)
        low, high = report[0]["sampled_peak_rad_per_s"]
        assert low == pytest.approx(0.5 * 0.9e8)
        assert high == pytest.approx(0.5 * 1.1e8)
        assert report[0]["nominal_peak_rad_per_s"] == pytest.approx(0.5 * 1.0e8)

    def test_formatted_report_states_the_semantics(self):
        from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
            amplitude_limit_report,
            format_amplitude_limit_report,
        )

        cx = torch.tensor([[1.5]], dtype=torch.float64)
        cy = torch.zeros_like(cx)
        text = format_amplitude_limit_report(amplitude_limit_report(cx, cy, [1.0e8]))
        assert "SOFT penalty" in text
        assert "not clipped or constrained" in text
        assert "EXCEEDS LIMIT" in text
        assert "interpolated continuous waveform" in text

    def test_run_ctrl_attaches_the_report(self):
        cfg = make_config(n_points=8, n_para=4, max_iter=2)
        api = CtrlFreeQAPI(copy.deepcopy(cfg))
        api.run_optimization()
        report = api.parameters.amplitude_report
        assert len(report) == 1
        assert set(report[0]) >= {
            "normalized_peak",
            "excess",
            "within_limit",
            "nominal_peak_rad_per_s",
        }


# ======================================================================
# Second derivatives after the objective / propagator changes
# ======================================================================


class TestHessianVectorProducts:
    @pytest.mark.parametrize(
        "cfg_kwargs",
        [
            dict(),
            dict(space="liouville"),
            dict(space="liouville", dissipation={"T1": [1e-6], "T2": [8e-7]}),
        ],
        ids=["hilbert", "liouville", "dissipative"],
    )
    def test_hvp_matches_a_finite_difference_of_the_gradient(self, cfg_kwargs):
        p = params_from(make_config(n_points=6, n_para=4, **cfg_kwargs))
        instance = _build_instance(p)
        x0 = torch.as_tensor(np.asarray(p.x0_con), dtype=torch.float64)

        torch.manual_seed(0)
        direction = torch.randn_like(x0)
        direction = direction / direction.norm()

        def grad_at(x):
            x = x.detach().clone().requires_grad_(True)
            (g,) = torch.autograd.grad(instance.objective_function(x), x)
            return g.detach()

        x = x0.detach().clone().requires_grad_(True)
        (g,) = torch.autograd.grad(instance.objective_function(x), x, create_graph=True)
        hvp = torch.autograd.grad(g @ direction, x)[0].detach().numpy()

        eps = 1e-6
        numeric = (
            (grad_at(x0 + eps * direction) - grad_at(x0 - eps * direction)) / (2 * eps)
        ).numpy()
        np.testing.assert_allclose(hvp, numeric, rtol=1e-4, atol=1e-6)

    def test_gate_objective_hvp_is_finite(self):
        p = params_from(
            make_config(
                n_qubits=2,
                initial_states=[["Z", "-Z"]],
                target_states={"Gate": ["CNOT"]},
                n_points=5,
                n_para=4,
            )
        )
        instance = _build_instance(p)
        x = torch.as_tensor(np.asarray(p.x0_con), dtype=torch.float64).requires_grad_(
            True
        )
        (g,) = torch.autograd.grad(instance.objective_function(x), x, create_graph=True)
        torch.manual_seed(1)
        direction = torch.randn_like(x)
        hvp = torch.autograd.grad(g @ direction, x)[0]
        assert torch.isfinite(hvp).all()


# ======================================================================
# Optimisation and analysis share the waveform representation
# ======================================================================


class TestWaveformRepresentation:
    def test_basis_solutions_use_the_configured_basis_by_default(self):
        from ctrl_freeq.make_pulse.waveform_gen_torch import waveform_gen_cart

        p = params_from(make_config(n_points=8, n_para=4))
        spec = p.waveform_spec()
        assert tuple(spec.n_para) == tuple(p.n_para_updated)
        assert tuple(spec.wf_mode) == tuple(p.wf_mode)
        assert spec.functions == [waveform_gen_cart]
        np.testing.assert_allclose(np.asarray(spec.mat), np.asarray(p.mat))

    @pytest.mark.parametrize("method", ["cart", "polar", "polar_phase"])
    @pytest.mark.parametrize("n_points", [2, 8])
    def test_piecewise_replay_matches_the_optimizer(self, method, n_points):
        """Replay must use the piecewise layout, not the configured basis.

        With ``n_points=2`` the piecewise and basis parameter counts happen to
        coincide, so the mismatch was silent; with ``n_points=8`` they differ
        and splitting the solution raised a split-size error.
        """
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI
        from ctrl_freeq.visualisation.plotter import compute_and_store_evolution

        cfg = make_config(n_points=n_points, n_para=4, max_iter=3)
        api = PiecewiseAPI(copy.deepcopy(cfg), method=method)
        solution = api.run_optimization()
        p = api.parameters

        _, _, history, _, _ = compute_and_store_evolution(solution, p, p.init[0])
        final = history[:, -1, 0]
        target = np.asarray(p.targets[0])
        replay_fidelity = abs(np.vdot(target, final)) ** 2

        assert replay_fidelity == pytest.approx(p.final_fidelity, abs=1e-9)

    @pytest.mark.parametrize("method", ["cart", "polar", "polar_phase"])
    def test_piecewise_spec_describes_the_identity_basis(self, method):
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        cfg = make_config(n_points=6, n_para=4, max_iter=1)
        api = PiecewiseAPI(copy.deepcopy(cfg), method=method)
        solution = api.run_optimization()
        p = api.parameters

        spec = p.waveform_spec()
        instance = api._piecewise_instance
        assert tuple(spec.n_para) == tuple(instance.n_para)
        assert sum(spec.n_para) == solution.shape[0]
        assert tuple(spec.wf_mode) == (method,) * p.n_qubits
        for per_qubit, basis in zip(spec.mat, instance.identity_basis):
            for got, expected in zip(per_qubit, basis):
                torch.testing.assert_close(got, expected)

    def test_piecewise_waveforms_match_the_optimizer_waveforms(self):
        """The replayed I/Q samples are the ones the objective was built from."""
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI, pulse_para_piecewise
        from ctrl_freeq.visualisation.plotter import _plotter_waveforms

        cfg = make_config(n_points=8, n_para=4, max_iter=2)
        api = PiecewiseAPI(copy.deepcopy(cfg), method="cart")
        solution = api.run_optimization()
        instance = api._piecewise_instance

        _, _, cx_opt, cy_opt = pulse_para_piecewise(
            instance.n_qubits,
            torch.split(solution.detach(), instance.n_para),
            instance.identity_basis,
            instance.wf_fun,
            instance.me,
        )
        _, cx_plot, cy_plot = _plotter_waveforms(solution, api.parameters)

        torch.testing.assert_close(cx_plot.to(cx_opt.dtype), cx_opt, atol=1e-12, rtol=0)
        torch.testing.assert_close(cy_plot.to(cy_opt.dtype), cy_opt, atol=1e-12, rtol=0)

    def _replay_fidelity(self, solution, p, waveform_spec=None):
        from ctrl_freeq.visualisation.plotter import compute_and_store_evolution

        _, _, history, _, _ = compute_and_store_evolution(
            solution, p, p.init[0], waveform_spec
        )
        target = np.asarray(p.targets[0])
        return abs(np.vdot(target, history[:, -1, 0])) ** 2

    def test_basis_run_after_a_piecewise_run_replays_correctly(self):
        """A piecewise representation must not leak into a later basis run."""
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        cfg = make_config(n_points=8, n_para=4, max_iter=3)
        api = CtrlFreeQAPI(copy.deepcopy(cfg))

        PiecewiseAPI(api).run_optimization()
        assert sum(api.parameters.waveform_spec().n_para) == 16

        solution = api.run_optimization()
        p = api.parameters
        assert solution.numel() == 4
        assert sum(p.waveform_spec().n_para) == 4
        assert self._replay_fidelity(solution, p) == pytest.approx(
            p.final_fidelity, abs=1e-9
        )

    def test_piecewise_run_after_a_basis_run_replays_correctly(self):
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        cfg = make_config(n_points=8, n_para=4, max_iter=3)
        api = CtrlFreeQAPI(copy.deepcopy(cfg))
        api.run_optimization()

        piecewise = PiecewiseAPI(api)
        solution = piecewise.run_optimization()
        p = piecewise.parameters
        assert solution.numel() == 16
        assert sum(p.waveform_spec().n_para) == 16
        assert self._replay_fidelity(solution, p) == pytest.approx(
            p.final_fidelity, abs=1e-9
        )

    def test_stale_solution_is_rejected_with_a_clear_message(self):
        """Analysing an older solution must not silently use the wrong basis."""
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI
        from ctrl_freeq.visualisation.plotter import compute_and_store_evolution

        cfg = make_config(n_points=8, n_para=4, max_iter=2)
        api = CtrlFreeQAPI(copy.deepcopy(cfg))
        basis_solution = api.run_optimization()
        PiecewiseAPI(api).run_optimization()

        with pytest.raises(ValueError, match="different optimizer"):
            compute_and_store_evolution(
                basis_solution, api.parameters, api.parameters.init[0]
            )

    def test_equal_count_representations_need_an_explicit_spec(self):
        """Two live solutions with the same parameter count are ambiguous.

        At eight points a piecewise cart solution has 16 parameters and a basis
        one has 4, so the guard catches a mix-up.  At two points both have 4:
        the counts cannot distinguish them, implicit replay silently uses the
        latest run's representation, and the owning run's spec has to be passed
        explicitly.
        """
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI
        from ctrl_freeq.visualisation.plotter import _plotter_waveforms

        cfg = make_config(n_points=2, n_para=4, max_iter=2)
        api = CtrlFreeQAPI(copy.deepcopy(cfg))

        basis_solution = api.run_optimization()
        basis_spec = api.waveform_spec()
        # Captured before the next run: the metrics live on the shared
        # parameters object too, so the piecewise run overwrites them.
        basis_fidelity = api.parameters.final_fidelity

        piecewise = PiecewiseAPI(api)
        piecewise_solution = piecewise.run_optimization()
        piecewise_spec = piecewise.waveform_spec()

        assert basis_solution.numel() == piecewise_solution.numel() == 4
        assert sum(basis_spec.n_para) == sum(piecewise_spec.n_para)

        p = api.parameters
        # The guard cannot fire: the counts agree.
        _, cx_implicit, _ = _plotter_waveforms(basis_solution, p)
        _, cx_explicit, _ = _plotter_waveforms(basis_solution, p, basis_spec)
        _, cx_piecewise, _ = _plotter_waveforms(basis_solution, p, piecewise_spec)

        # Implicit replay used the latest run's (piecewise) representation.
        torch.testing.assert_close(cx_implicit, cx_piecewise)
        assert not torch.allclose(cx_explicit, cx_implicit)

        # Each solution replays correctly when given its own spec.
        assert self._replay_fidelity(basis_solution, p, basis_spec) == pytest.approx(
            basis_fidelity, abs=1e-9
        )
        assert self._replay_fidelity(
            piecewise_solution, p, piecewise_spec
        ) == pytest.approx(piecewise.parameters.final_fidelity, abs=1e-9)

    def test_basis_spec_survives_a_later_piecewise_run(self):
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        cfg = make_config(n_points=8, n_para=4, max_iter=2)
        api = CtrlFreeQAPI(copy.deepcopy(cfg))
        basis_spec = api.waveform_spec()
        PiecewiseAPI(api).run_optimization()
        assert api.waveform_spec() == basis_spec
        assert sum(api.parameters.waveform_spec().n_para) == 16

    def test_piecewise_api_spec_requires_a_run(self):
        from ctrl_freeq.ctrlfreeq.piecewise import PiecewiseAPI

        piecewise = PiecewiseAPI(copy.deepcopy(make_config(n_points=4)))
        with pytest.raises(RuntimeError, match="Run the optimization"):
            piecewise.waveform_spec()

    def test_unknown_waveform_mode_is_rejected(self):
        from ctrl_freeq.make_pulse.waveform_gen_torch import waveform_function

        with pytest.raises(ValueError, match="Unknown waveform mode"):
            waveform_function("nope")
