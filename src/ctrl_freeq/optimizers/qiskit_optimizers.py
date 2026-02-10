"""
Qiskit optimizer wrapper module for CtrlFreeQ.

This module provides wrappers for Qiskit optimizers to integrate them
with the existing CtrlFreeQ optimization framework.
"""

import torch
import numpy as np
from typing import Callable, Optional


from qiskit_algorithms.optimizers import (
    ADAM,
    COBYLA,
    GSLS,
    L_BFGS_B,
    NELDER_MEAD,
    P_BFGS,
    POWELL,
    SLSQP,
    SPSA as QiskitSPSA,
    TNC,
)
from qiskit_algorithms.optimizers import OptimizerResult


class QiskitOptimizerWrapper:
    """Base wrapper class for Qiskit optimizers."""

    def __init__(self, optimizer_class, **kwargs):
        self.optimizer = optimizer_class(**kwargs)
        self.callback = None
        self.iteration_count = 0
        self.max_iter = kwargs.get("maxiter", 1000)  # Store max_iter from kwargs

    def minimize(
        self,
        objective_func: Callable,
        x0: torch.Tensor,
        callback: Optional[Callable] = None,
        **kwargs,
    ) -> "QiskitOptimizeResult":
        """
        Minimize the objective function using the wrapped Qiskit optimizer.

        Args:
            objective_func: The objective function to minimize
            x0: Initial parameter vector (torch tensor)
            callback: Optional callback function
            **kwargs: Additional arguments

        Returns:
            QiskitOptimizeResult: Result object containing optimized parameters
        """
        self.callback = callback
        self.iteration_count = 0
        self.last_x = None  # Track last evaluated parameters

        # Convert torch tensor to numpy for Qiskit
        x0_numpy = x0.detach().cpu().numpy().astype(np.float64)

        # Wrap the objective function to handle torch/numpy conversion
        def objective_wrapper(x):
            # Store current parameters
            self.last_x = x.copy()

            # Check iteration limit before processing
            if self.iteration_count >= self.max_iter:
                raise StopIteration(f"Maximum iteration limit {self.max_iter} reached")

            x_tensor = torch.tensor(
                x, dtype=x0.dtype, device=x0.device, requires_grad=False
            )
            result = objective_func(x_tensor)

            # Handle callback
            if self.callback is not None:
                self.callback(x_tensor)

            self.iteration_count += 1

            # Return scalar numpy value
            if hasattr(result, "detach"):
                return result.detach().cpu().numpy().item()
            else:
                return float(result)

        # Run optimization with iteration limit handling
        try:
            result = self.optimizer.minimize(objective_wrapper, x0_numpy)
            # Convert result back to torch tensor
            x_optimized = torch.tensor(result.x, dtype=x0.dtype, device=x0.device)
            return QiskitOptimizeResult(x_optimized, result)
        except StopIteration as e:
            # Max iteration reached - create a mock result with current best estimate
            # Use the last parameters evaluated if available, otherwise fallback to x0_numpy
            print(f"Optimization stopped early: {e}")

            # Use last evaluated parameters if available, otherwise initial parameters
            final_params = self.last_x if self.last_x is not None else x0_numpy

            # Create a mock OptimizerResult for consistency
            class MockOptimizerResult:
                def __init__(self, x, iterations):
                    self.x = x
                    self.fun = None  # We don't have final objective value
                    self.nit = iterations
                    self.nfev = iterations
                    self.success = True

            mock_result = MockOptimizerResult(final_params, self.iteration_count)
            x_optimized = torch.tensor(final_params, dtype=x0.dtype, device=x0.device)
            return QiskitOptimizeResult(x_optimized, mock_result)


class QiskitOptimizeResult:
    """Result object for Qiskit optimizers, compatible with existing CtrlFreeQ interface."""

    def __init__(self, x: torch.Tensor, qiskit_result: OptimizerResult):
        self.x = x
        self.success = True
        self.message = "Qiskit optimization completed"
        self.nit = qiskit_result.nit if hasattr(qiskit_result, "nit") else None
        self.fun = qiskit_result.fun if hasattr(qiskit_result, "fun") else None
        self.nfev = qiskit_result.nfev if hasattr(qiskit_result, "nfev") else None
        self._qiskit_result = qiskit_result


def create_qiskit_spsa_optimizer(
    maxiter: int = 1000,
    last_avg: int = 1,
    learning_rate: Optional[float] = None,
    perturbation: Optional[float] = None,
    **kwargs,
) -> QiskitOptimizerWrapper:
    """
    Create a Qiskit SPSA optimizer wrapper.

    Args:
        maxiter: Maximum number of iterations
        last_avg: Number of last iterations to average for final result
        learning_rate: Learning rate parameter (a in SPSA literature)
        perturbation: Perturbation parameter (c in SPSA literature)
        **kwargs: Additional arguments passed to Qiskit SPSA optimizer

    Returns:
        QiskitOptimizerWrapper: Wrapped Qiskit SPSA optimizer
    """

    # Set up default parameters for SPSA
    spsa_kwargs = {
        "maxiter": maxiter,
        "last_avg": last_avg,
    }

    # Add optional parameters if provided
    if learning_rate is not None:
        spsa_kwargs["learning_rate"] = learning_rate
    if perturbation is not None:
        spsa_kwargs["perturbation"] = perturbation

    # Add any additional kwargs
    spsa_kwargs.update(kwargs)

    return QiskitOptimizerWrapper(QiskitSPSA, **spsa_kwargs)


def create_qiskit_nelder_mead_optimizer(
    maxiter: int = 1000, **kwargs
) -> QiskitOptimizerWrapper:
    """Create a Qiskit Nelder-Mead optimizer wrapper."""
    nelder_mead_kwargs = {"maxiter": maxiter}
    nelder_mead_kwargs.update(kwargs)
    return QiskitOptimizerWrapper(NELDER_MEAD, **nelder_mead_kwargs)


def create_qiskit_powell_optimizer(
    maxiter: int = 1000, **kwargs
) -> QiskitOptimizerWrapper:
    """Create a Qiskit Powell optimizer wrapper."""
    powell_kwargs = {"maxiter": maxiter}
    powell_kwargs.update(kwargs)
    return QiskitOptimizerWrapper(POWELL, **powell_kwargs)


def create_qiskit_slsqp_optimizer(
    maxiter: int = 1000, **kwargs
) -> QiskitOptimizerWrapper:
    """Create a Qiskit SLSQP optimizer wrapper."""
    slsqp_kwargs = {"maxiter": maxiter}
    slsqp_kwargs.update(kwargs)
    return QiskitOptimizerWrapper(SLSQP, **slsqp_kwargs)


def create_qiskit_l_bfgs_b_optimizer(
    maxiter: int = 1000, **kwargs
) -> QiskitOptimizerWrapper:
    """Create a Qiskit L-BFGS-B optimizer wrapper."""
    l_bfgs_b_kwargs = {"maxiter": maxiter}
    l_bfgs_b_kwargs.update(kwargs)
    return QiskitOptimizerWrapper(L_BFGS_B, **l_bfgs_b_kwargs)


def create_qiskit_tnc_optimizer(
    maxiter: int = 1000, **kwargs
) -> QiskitOptimizerWrapper:
    """Create a Qiskit TNC optimizer wrapper."""
    tnc_kwargs = {"maxiter": maxiter}
    tnc_kwargs.update(kwargs)
    return QiskitOptimizerWrapper(TNC, **tnc_kwargs)


def create_qiskit_p_bfgs_optimizer(
    maxiter: int = 1000, **kwargs
) -> QiskitOptimizerWrapper:
    """Create a Qiskit P-BFGS optimizer wrapper."""
    # P-BFGS doesn't support maxiter parameter, so we exclude it
    p_bfgs_kwargs = {}
    p_bfgs_kwargs.update(kwargs)
    return QiskitOptimizerWrapper(P_BFGS, **p_bfgs_kwargs)


def create_qiskit_cobyla_optimizer(
    maxiter: int = 1000, **kwargs
) -> QiskitOptimizerWrapper:
    """Create a Qiskit COBYLA optimizer wrapper."""
    cobyla_kwargs = {"maxiter": maxiter}
    cobyla_kwargs.update(kwargs)
    return QiskitOptimizerWrapper(COBYLA, **cobyla_kwargs)


def create_qiskit_gsls_optimizer(
    maxiter: int = 1000, **kwargs
) -> QiskitOptimizerWrapper:
    """Create a Qiskit GSLS optimizer wrapper."""
    gsls_kwargs = {"maxiter": maxiter}
    gsls_kwargs.update(kwargs)
    return QiskitOptimizerWrapper(GSLS, **gsls_kwargs)


def create_qiskit_adam_optimizer(
    maxiter: int = 1000, **kwargs
) -> QiskitOptimizerWrapper:
    """Create a Qiskit ADAM optimizer wrapper."""
    adam_kwargs = {"maxiter": maxiter}
    adam_kwargs.update(kwargs)
    return QiskitOptimizerWrapper(ADAM, **adam_kwargs)


# Function to integrate with existing run_ctrlfreeq function
def run_qiskit_optimization(
    optimizer_name: str,
    objective_func: Callable,
    x0: torch.Tensor,
    callback: Optional[Callable] = None,
    max_iter: int = 1000,
    **optimizer_kwargs,
) -> torch.Tensor:
    """
    Run optimization using Qiskit optimizers.

    Args:
        optimizer_name: Name of the Qiskit optimizer ('qiskit-spsa', etc.)
        objective_func: The objective function to minimize
        x0: Initial parameter vector
        callback: Optional callback function
        max_iter: Maximum number of iterations
        **optimizer_kwargs: Additional optimizer-specific parameters

    Returns:
        torch.Tensor: Optimized parameters
    """
    optimizer_map = {
        "qiskit-spsa": create_qiskit_spsa_optimizer,
        "qiskit-nelder-mead": create_qiskit_nelder_mead_optimizer,
        "qiskit-powell": create_qiskit_powell_optimizer,
        "qiskit-slsqp": create_qiskit_slsqp_optimizer,
        "qiskit-l-bfgs-b": create_qiskit_l_bfgs_b_optimizer,
        "qiskit-tnc": create_qiskit_tnc_optimizer,
        "qiskit-p-bfgs": create_qiskit_p_bfgs_optimizer,
        "qiskit-cobyla": create_qiskit_cobyla_optimizer,
        "qiskit-gsls": create_qiskit_gsls_optimizer,
        "qiskit-adam": create_qiskit_adam_optimizer,
    }

    if optimizer_name in optimizer_map:
        optimizer_func = optimizer_map[optimizer_name]
        optimizer = optimizer_func(maxiter=max_iter, **optimizer_kwargs)
        result = optimizer.minimize(objective_func, x0, callback=callback)
        return result.x
    else:
        raise ValueError(f"Unsupported Qiskit optimizer: {optimizer_name}")


def get_supported_qiskit_optimizers():
    """Get list of supported Qiskit optimizers."""

    return [
        "qiskit-spsa",
        "qiskit-nelder-mead",
        "qiskit-powell",
        "qiskit-slsqp",
        "qiskit-l-bfgs-b",
        "qiskit-tnc",
        "qiskit-p-bfgs",
        "qiskit-cobyla",
        "qiskit-gsls",
        "qiskit-adam",
    ]
