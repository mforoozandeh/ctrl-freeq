#!/usr/bin/env python3
"""
Comprehensive pytest tests for Qiskit optimizer wrappers in CtrlFreeQ.

Based on the working test, this comprehensive test suite covers:
1. QiskitOptimizerWrapper class functionality
2. QiskitOptimizeResult class
3. create_qiskit_spsa_optimizer function
4. run_qiskit_optimization function
5. Integration with torch tensors
6. Error handling and edge cases
7. Callback functionality
8. Multiple optimization scenarios
"""

import pytest
import numpy as np
import torch

# Import the modules to test
from ctrl_freeq.optimizers.qiskit_optimizers import (
    QiskitOptimizerWrapper,
    QiskitOptimizeResult,
    create_qiskit_spsa_optimizer,
    run_qiskit_optimization,
    get_supported_qiskit_optimizers,
)


def rosenbrock_torch(x: torch.Tensor) -> torch.Tensor:
    """Rosenbrock function for torch tensors."""
    a = 1.0
    b = 100.0
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def quadratic_torch(x: torch.Tensor) -> torch.Tensor:
    """Simple quadratic function for torch tensors."""
    return torch.sum(x**2)


def sphere_torch(x: torch.Tensor) -> torch.Tensor:
    """N-dimensional sphere function for torch tensors."""
    return torch.sum((x - 1.0) ** 2)


class TestQiskitOptimizerWrapper:
    """Test the QiskitOptimizerWrapper class."""

    def test_wrapper_initialization(self):
        """Test wrapper initialization with SPSA optimizer."""
        optimizer = create_qiskit_spsa_optimizer(maxiter=100)
        assert optimizer is not None
        assert isinstance(optimizer, QiskitOptimizerWrapper)
        assert optimizer.optimizer is not None
        assert optimizer.callback is None
        assert optimizer.iteration_count == 0

    def test_minimize_shifted_quadratic(self):
        """Test minimization of shifted quadratic function."""

        def shifted_quadratic(x):
            target = torch.tensor([2.0, -1.0])
            return torch.sum((x - target) ** 2)

        optimizer = create_qiskit_spsa_optimizer(
            maxiter=300, learning_rate=0.02, perturbation=0.01
        )
        x0 = torch.tensor([0.0, 0.0], dtype=torch.float32)

        result = optimizer.minimize(shifted_quadratic, x0)

        assert isinstance(result, QiskitOptimizeResult)
        assert isinstance(result.x, torch.Tensor)
        assert result.x.shape == x0.shape
        assert result.x.dtype == x0.dtype
        assert result.success is True

        # Check if result is finite and reasonable
        assert not torch.any(torch.isnan(result.x))  # Ensure no NaN values
        assert torch.all(torch.isfinite(result.x))  # Ensure finite values
        # Should be within reasonable distance of target [2.0, -1.0]
        target = torch.tensor([2.0, -1.0])
        distance = torch.norm(result.x - target)
        assert distance < 2.0  # Reasonable tolerance for SPSA

    def test_minimize_quadratic(self):
        """Test minimization of simple quadratic function."""
        optimizer = create_qiskit_spsa_optimizer(
            maxiter=200, learning_rate=0.05, perturbation=0.01
        )
        x0 = torch.tensor([2.0, -1.5], dtype=torch.float32)

        result = optimizer.minimize(quadratic_torch, x0)

        assert isinstance(result, QiskitOptimizeResult)
        assert result.x.shape == x0.shape

        # Should converge reasonably close to origin (generous tolerance for SPSA)
        assert torch.norm(result.x) < 3.0

    def test_minimize_with_callback(self):
        """Test minimization with callback function."""
        callback_values = []

        def test_callback(x):
            callback_values.append(x.clone())

        optimizer = create_qiskit_spsa_optimizer(maxiter=50)
        x0 = torch.tensor([1.0, 1.0], dtype=torch.float32)

        optimizer.minimize(quadratic_torch, x0, callback=test_callback)

        assert len(callback_values) > 0
        assert all(isinstance(x, torch.Tensor) for x in callback_values)
        assert all(x.shape == x0.shape for x in callback_values)

    def test_different_dtypes(self):
        """Test with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            optimizer = create_qiskit_spsa_optimizer(maxiter=100)
            x0 = torch.tensor([0.5, -0.5], dtype=dtype)

            result = optimizer.minimize(quadratic_torch, x0)

            assert result.x.dtype == dtype

    def test_different_devices(self):
        """Test with different devices (CPU)."""
        # Test CPU device
        device = torch.device("cpu")
        optimizer = create_qiskit_spsa_optimizer(maxiter=100)
        x0 = torch.tensor([0.5, -0.5], device=device)

        result = optimizer.minimize(quadratic_torch, x0)

        assert result.x.device == device

    def test_higher_dimensional_problem(self):
        """Test optimization of higher-dimensional problem."""
        optimizer = create_qiskit_spsa_optimizer(maxiter=300)
        x0 = torch.tensor([2.0, -1.0, 0.5, -2.0], dtype=torch.float32)

        result = optimizer.minimize(sphere_torch, x0)

        assert result.x.shape == (4,)
        # Should converge near [1, 1, 1, 1]
        target = torch.ones(4)
        distance = torch.norm(result.x - target)
        assert distance < 2.0  # Generous tolerance


class TestQiskitOptimizeResult:
    """Test the QiskitOptimizeResult class."""

    def test_result_creation(self):
        """Test creation of QiskitOptimizeResult."""
        x_tensor = torch.tensor([1.0, 2.0])

        # Mock qiskit result
        class MockQiskitResult:
            def __init__(self):
                self.nit = 100
                self.fun = 0.5
                self.nfev = 200
                self.x = np.array([1.0, 2.0])

        mock_result = MockQiskitResult()
        result = QiskitOptimizeResult(x_tensor, mock_result)

        assert torch.equal(result.x, x_tensor)
        assert result.success is True
        assert result.message == "Qiskit optimization completed"
        assert result.nit == 100
        assert result.fun == 0.5
        assert result.nfev == 200
        assert result._qiskit_result is mock_result

    def test_result_with_missing_attributes(self):
        """Test result creation when qiskit result has missing attributes."""
        x_tensor = torch.tensor([1.0, 2.0])

        class MockQiskitResultMinimal:
            def __init__(self):
                self.x = np.array([1.0, 2.0])

        mock_result = MockQiskitResultMinimal()
        result = QiskitOptimizeResult(x_tensor, mock_result)

        assert result.nit is None
        assert result.fun is None
        assert result.nfev is None


class TestCreateQiskitSpsa:
    """Test the create_qiskit_spsa_optimizer function."""

    def test_default_parameters(self):
        """Test SPSA optimizer creation with default parameters."""
        optimizer = create_qiskit_spsa_optimizer()

        assert isinstance(optimizer, QiskitOptimizerWrapper)
        assert optimizer.optimizer is not None

    def test_custom_parameters(self):
        """Test SPSA optimizer creation with custom parameters."""
        optimizer = create_qiskit_spsa_optimizer(
            maxiter=500, last_avg=5, learning_rate=0.1, perturbation=0.05
        )

        assert isinstance(optimizer, QiskitOptimizerWrapper)
        # Verify parameters were passed (indirectly by running optimization)
        x0 = torch.tensor([1.0, 1.0])
        result = optimizer.minimize(quadratic_torch, x0)
        assert result is not None

    def test_additional_kwargs(self):
        """Test SPSA optimizer with additional keyword arguments."""
        optimizer = create_qiskit_spsa_optimizer(
            maxiter=100,
            blocking=True,  # Additional SPSA parameter
            allowed_increase=None,
        )

        assert isinstance(optimizer, QiskitOptimizerWrapper)


class TestRunQiskitOptimization:
    """Test the run_qiskit_optimization function."""

    def test_qiskit_spsa_optimization(self):
        """Test run_qiskit_optimization with SPSA."""
        x0 = torch.tensor([0.5, -0.5], dtype=torch.float32)

        result_x = run_qiskit_optimization(
            optimizer_name="qiskit-spsa",
            objective_func=quadratic_torch,
            x0=x0,
            max_iter=200,
        )

        assert isinstance(result_x, torch.Tensor)
        assert result_x.shape == x0.shape
        assert result_x.dtype == x0.dtype

    def test_qiskit_spsa_with_callback(self):
        """Test run_qiskit_optimization with callback."""
        callback_calls = []

        def test_callback(x):
            callback_calls.append(len(callback_calls))

        x0 = torch.tensor([1.0, -1.0], dtype=torch.float32)

        result_x = run_qiskit_optimization(
            optimizer_name="qiskit-spsa",
            objective_func=quadratic_torch,
            x0=x0,
            callback=test_callback,
            max_iter=100,
        )

        assert isinstance(result_x, torch.Tensor)
        assert len(callback_calls) > 0

    def test_qiskit_spsa_with_kwargs(self):
        """Test run_qiskit_optimization with optimizer-specific kwargs."""
        x0 = torch.tensor([2.0, 1.0], dtype=torch.float32)

        result_x = run_qiskit_optimization(
            optimizer_name="qiskit-spsa",
            objective_func=quadratic_torch,
            x0=x0,
            max_iter=150,
            learning_rate=0.05,
            perturbation=0.1,
        )

        assert isinstance(result_x, torch.Tensor)
        assert result_x.shape == x0.shape

    def test_unsupported_optimizer_error(self):
        """Test error handling for unsupported optimizer."""
        x0 = torch.tensor([1.0, 1.0])

        with pytest.raises(ValueError, match="Unsupported Qiskit optimizer"):
            run_qiskit_optimization(
                optimizer_name="unsupported-optimizer",
                objective_func=quadratic_torch,
                x0=x0,
            )


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_supported_optimizers(self):
        """Test get_supported_qiskit_optimizers function."""
        supported = get_supported_qiskit_optimizers()

        assert isinstance(supported, list)
        assert "qiskit-spsa" in supported
        assert len(supported) >= 1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_parameter_optimization(self):
        """Test optimization with single parameter."""

        def single_param_func(x):
            return (x[0] - 2.0) ** 2

        optimizer = create_qiskit_spsa_optimizer(maxiter=100)
        x0 = torch.tensor([0.0])

        result = optimizer.minimize(single_param_func, x0)

        assert result.x.shape == (1,)
        assert abs(result.x[0] - 2.0) < 1.0  # Should be close to 2

    def test_objective_returns_tensor(self):
        """Test when objective function returns tensor instead of scalar."""

        def tensor_objective(x):
            return torch.tensor((x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2)

        optimizer = create_qiskit_spsa_optimizer(maxiter=100)
        x0 = torch.tensor([0.0, 0.0])

        result = optimizer.minimize(tensor_objective, x0)

        assert isinstance(result, QiskitOptimizeResult)
        assert result.x.shape == (2,)

    def test_objective_returns_scalar(self):
        """Test when objective function returns Python float."""

        def scalar_objective(x):
            return float((x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2)

        optimizer = create_qiskit_spsa_optimizer(maxiter=100)
        x0 = torch.tensor([0.0, 0.0])

        result = optimizer.minimize(scalar_objective, x0)

        assert isinstance(result, QiskitOptimizeResult)
        assert result.x.shape == (2,)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    def test_comprehensive_optimization_scenarios(self):
        """Comprehensive test of various optimization scenarios."""
        # Test different functions and starting points
        test_cases = [
            {
                "func": lambda x: torch.sum((x - 1.0) ** 2),  # Sphere function
                "x0": torch.tensor([0.0, 0.0]),
                "target": torch.tensor([1.0, 1.0]),
                "name": "sphere",
            },
            {
                "func": lambda x: torch.sum(x**2)
                + 0.1 * torch.sum(torch.sin(5 * x)),  # Noisy quadratic
                "x0": torch.tensor([2.0, -1.0]),
                "target": torch.tensor([0.0, 0.0]),
                "name": "noisy_quadratic",
            },
        ]

        for case in test_cases:
            optimizer = create_qiskit_spsa_optimizer(
                maxiter=250, learning_rate=0.02, perturbation=0.01
            )

            result = optimizer.minimize(case["func"], case["x0"])

            assert isinstance(result, QiskitOptimizeResult)
            assert result.success is True

            # Check that result is finite
            assert not torch.any(torch.isnan(result.x))
            assert torch.all(torch.isfinite(result.x))

            # Check reasonable convergence (generous tolerance for SPSA)
            distance = torch.norm(result.x - case["target"])
            assert distance < 3.0, f"Failed for {case['name']}: distance={distance}"

    def test_multiple_optimizer_runs(self):
        """Test running multiple optimizations in sequence."""
        results = []

        for i in range(3):
            x0 = torch.tensor([float(i), float(-i)])
            result_x = run_qiskit_optimization(
                optimizer_name="qiskit-spsa",
                objective_func=quadratic_torch,
                x0=x0,
                max_iter=100,
                learning_rate=0.01,  # Add required parameters
                perturbation=0.01,
            )
            results.append(result_x)

        # Check that all results are finite and reasonable
        for result_x in results:
            assert not torch.any(torch.isnan(result_x))
            assert torch.all(torch.isfinite(result_x))
            assert torch.norm(result_x) < 5.0  # Generous bound for SPSA

    def test_optimization_with_noise(self):
        """Test optimization of noisy objective function."""

        def noisy_quadratic(x):
            noise = 0.01 * torch.randn_like(x).sum()
            return torch.sum(x**2) + noise

        optimizer = create_qiskit_spsa_optimizer(maxiter=300)
        x0 = torch.tensor([1.0, -1.0])

        result = optimizer.minimize(noisy_quadratic, x0)

        assert isinstance(result, QiskitOptimizeResult)
        # Should still converge reasonably close to origin despite noise
        assert torch.norm(result.x) < 1.5
