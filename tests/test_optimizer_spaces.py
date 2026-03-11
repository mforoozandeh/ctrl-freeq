"""Comprehensive optimizer x space tests.

Single-qubit inversion (|+Z> -> |-Z>) run with every supported algorithm in
both Hilbert and Liouville (pure + dissipative) spaces.  The intent is to
catch regressions where a code change breaks one evolution mode but not the
other, and to verify that every algorithm can run to completion without
errors in all three modes.

The tests use a small number of iterations so the suite stays fast, but
enough that gradient-based methods should show meaningful fidelity
improvement.  Derivative-free methods (COBYLA, Nelder-Mead, SPSA, etc) may
not converge in so few evaluations; for those we only assert the run
completes and returns a positive fidelity.
"""

import pytest
import torch

from ctrl_freeq.api import (
    load_single_qubit_config,
    load_single_qubit_dissipative_config,
)

# ---------------------------------------------------------------------------
# Algorithm lists
# ---------------------------------------------------------------------------
TORCHMIN_ALGORITHMS = [
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

QISKIT_ALGORITHMS = [
    "qiskit-cobyla",
    "qiskit-spsa",
    "qiskit-nelder-mead",
    "qiskit-powell",
    "qiskit-slsqp",
    "qiskit-l-bfgs-b",
    "qiskit-tnc",
    "qiskit-gsls",
    "qiskit-adam",
]

ALL_ALGORITHMS = TORCHMIN_ALGORITHMS + QISKIT_ALGORITHMS

# Second-order methods that can raise ``torch._C._LinAlgError`` at certain
# starting points (e.g. non-positive-definite Hessian, singular Cholesky) --
# a known numerical limitation, not a code bug.
HESSIAN_SENSITIVE = {
    "newton-cg",
    "newton-exact",
    "dogleg",
    "trust-ncg",
    "trust-krylov",
    "trust-exact",
}

# Algorithms expected to show measurable improvement in <= MAX_ITER
# gradient-based iterations.  Excludes only trust-region PD methods that
# almost always skip (dogleg etc).  newton-cg/exact are kept because they
# usually converge; the rare LinAlgError is handled by the skip guard.
# Qiskit wrappers use finite-difference gradients (no autograd), so they
# converge too slowly for this test.
GRADIENT_ALGORITHMS = [
    "bfgs",
    "l-bfgs",
    "cg",
    "newton-cg",
    "newton-exact",
]

MAX_ITER = 20

HILBERT_MIN_FID = 0.80
LIOUVILLE_MIN_FID = 0.70
DISSIPATIVE_MIN_FID = 0.50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_optimization(api, algorithm, max_iter=MAX_ITER):
    """Configure and run an optimization, returning (solution, final_fidelity).

    Trust-region methods that require positive-definite Hessians may raise
    ``LinAlgError`` on certain starting points.  When this happens the test
    is skipped (known numerical limitation of the algorithm).
    """
    api.update_parameter("optimization.algorithm", algorithm)
    api.update_parameter("optimization.max_iter", max_iter)
    api.update_parameter("optimization.targ_fid", 0.99)
    try:
        solution = api.run_optimization()
    except torch._C._LinAlgError as exc:
        if algorithm in HESSIAN_SENSITIVE:
            pytest.skip(
                f"{algorithm}: LinAlgError (non-PD Hessian at starting point): {exc}"
            )
        raise
    return solution, api.parameters.final_fidelity


# ---------------------------------------------------------------------------
# Hilbert space
# ---------------------------------------------------------------------------
class TestHilbertSpace:
    """Single-qubit inversion in Hilbert space with every algorithm."""

    @pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
    def test_algorithm_runs(self, algorithm):
        api = load_single_qubit_config()
        api.update_parameter("optimization.space", "hilbert")
        solution, fidelity = _run_optimization(api, algorithm)

        assert solution is not None
        assert solution.shape[0] > 0
        assert fidelity > 0.0, f"{algorithm} fidelity {fidelity}"

    @pytest.mark.parametrize("algorithm", GRADIENT_ALGORITHMS)
    def test_gradient_algorithm_converges(self, algorithm):
        api = load_single_qubit_config()
        api.update_parameter("optimization.space", "hilbert")
        _, fidelity = _run_optimization(api, algorithm)

        assert fidelity >= HILBERT_MIN_FID, (
            f"{algorithm} Hilbert fidelity {fidelity:.4f} < {HILBERT_MIN_FID}"
        )


# ---------------------------------------------------------------------------
# Liouville space (pure, no dissipation)
# ---------------------------------------------------------------------------
class TestLiouvilleSpace:
    """Single-qubit inversion in Liouville space (density matrices, no dissipation)."""

    @pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
    def test_algorithm_runs(self, algorithm):
        api = load_single_qubit_config()
        api.update_parameter("optimization.space", "liouville")
        solution, fidelity = _run_optimization(api, algorithm)

        assert solution is not None
        assert solution.shape[0] > 0
        assert fidelity > 0.0, f"{algorithm} fidelity {fidelity}"

    @pytest.mark.parametrize("algorithm", GRADIENT_ALGORITHMS)
    def test_gradient_algorithm_converges(self, algorithm):
        api = load_single_qubit_config()
        api.update_parameter("optimization.space", "liouville")
        _, fidelity = _run_optimization(api, algorithm)

        assert fidelity >= LIOUVILLE_MIN_FID, (
            f"{algorithm} Liouville fidelity {fidelity:.4f} < {LIOUVILLE_MIN_FID}"
        )


# ---------------------------------------------------------------------------
# Dissipative (Lindblad)
# ---------------------------------------------------------------------------
class TestDissipativeSpace:
    """Single-qubit inversion with Lindblad dissipation (T1/T2 decoherence)."""

    @pytest.mark.parametrize("algorithm", ALL_ALGORITHMS)
    def test_algorithm_runs(self, algorithm):
        api = load_single_qubit_dissipative_config()
        solution, fidelity = _run_optimization(api, algorithm)

        assert solution is not None
        assert solution.shape[0] > 0
        assert fidelity > 0.0, f"{algorithm} fidelity {fidelity}"

    @pytest.mark.parametrize("algorithm", GRADIENT_ALGORITHMS)
    def test_gradient_algorithm_converges(self, algorithm):
        api = load_single_qubit_dissipative_config()
        _, fidelity = _run_optimization(api, algorithm)

        assert fidelity >= DISSIPATIVE_MIN_FID, (
            f"{algorithm} dissipative fidelity {fidelity:.4f} < {DISSIPATIVE_MIN_FID}"
        )


# ---------------------------------------------------------------------------
# Cross-space consistency
# ---------------------------------------------------------------------------
class TestCrossSpaceConsistency:
    """Verify Hilbert and Liouville produce consistent results."""

    @pytest.mark.parametrize("algorithm", ["bfgs", "l-bfgs"])
    def test_hilbert_liouville_fidelity_agreement(self, algorithm):
        """Both spaces solve the same problem -- both should reach high fidelity."""
        api_h = load_single_qubit_config()
        api_h.update_parameter("optimization.space", "hilbert")
        _, fid_h = _run_optimization(api_h, algorithm, max_iter=20)

        api_l = load_single_qubit_config()
        api_l.update_parameter("optimization.space", "liouville")
        _, fid_l = _run_optimization(api_l, algorithm, max_iter=20)

        assert fid_h > 0.90, f"Hilbert fidelity too low: {fid_h:.4f}"
        assert fid_l > 0.90, f"Liouville fidelity too low: {fid_l:.4f}"

    def test_dissipative_bounded_below_pure(self):
        """Decoherence should not improve fidelity over the pure case."""
        api_pure = load_single_qubit_config()
        api_pure.update_parameter("optimization.space", "liouville")
        _, fid_pure = _run_optimization(api_pure, "l-bfgs", max_iter=20)

        api_diss = load_single_qubit_dissipative_config()
        _, fid_diss = _run_optimization(api_diss, "l-bfgs", max_iter=20)

        assert fid_diss <= fid_pure + 0.05, (
            f"Dissipative {fid_diss:.4f} > pure {fid_pure:.4f} + tolerance"
        )
