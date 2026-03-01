from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    pass

# ======================================================================
# Hamiltonian model registry
# ======================================================================

_HAMILTONIAN_REGISTRY: dict[str, type[HamiltonianModel]] = {}


def register_hamiltonian(name: str):
    """Decorator to register a :class:`HamiltonianModel` subclass by name.

    Usage::

        @register_hamiltonian("trapped_ion")
        class TrappedIonModel(HamiltonianModel):
            ...

    The model can then be looked up via :func:`get_hamiltonian_class`.
    """

    def decorator(cls: type[HamiltonianModel]) -> type[HamiltonianModel]:
        if not (isinstance(cls, type) and issubclass(cls, HamiltonianModel)):
            raise TypeError(
                f"Cannot register {cls!r}: must be a HamiltonianModel subclass"
            )
        if name in _HAMILTONIAN_REGISTRY:
            raise ValueError(
                f"Hamiltonian name '{name}' is already registered to "
                f"{_HAMILTONIAN_REGISTRY[name].__name__}"
            )
        _HAMILTONIAN_REGISTRY[name] = cls
        return cls

    return decorator


def get_hamiltonian_class(name: str) -> type[HamiltonianModel]:
    """Look up a registered :class:`HamiltonianModel` class by name.

    Raises :class:`KeyError` with a helpful message listing available names.
    """
    if name not in _HAMILTONIAN_REGISTRY:
        available = ", ".join(sorted(_HAMILTONIAN_REGISTRY.keys()))
        raise KeyError(
            f"Unknown hamiltonian_type '{name}'. "
            f"Available: {available or '(none registered)'}"
        )
    return _HAMILTONIAN_REGISTRY[name]


def list_hamiltonians() -> list[str]:
    """Return the names of all registered Hamiltonian models."""
    return sorted(_HAMILTONIAN_REGISTRY.keys())


# ======================================================================
# Abstract base class
# ======================================================================


class HamiltonianModel(ABC):
    """Abstract base class for Hamiltonian models in optimal control.

    Any physical platform (spin chain, superconducting qubit, trapped ion, …)
    implements this interface.  The optimizer pipeline only depends on this
    abstraction via the standard bilinear control formulation:

        H(t) = H_drift + sum_k  u_k(t) * H_ctrl_k

    where u_k(t) are scalar control amplitudes and H_ctrl_k are fixed operators.

    Subclasses must implement:
        - build_drift()        : construct H0 matrix/matrices
        - build_control_ops()  : return the fixed control operators
        - control_amplitudes() : map waveform outputs to u_k(t) per batch element
        - from_config()        : construct model instance from config dict
        - dim (property)       : Hilbert space dimension
        - n_controls (property): number of independent control channels

    Optionally:
        - default_config()     : return a complete runnable config dict
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_drift(
        self,
        frequency_instances: list[np.ndarray],
        coupling_instances: list[np.ndarray] | None = None,
        **kwargs: Any,
    ) -> list[np.ndarray]:
        """Build drift Hamiltonian H0 instances.

        Args:
            frequency_instances: list of length ``n_snapshots``, each element
                is an array of per-qubit frequency-like parameters (chemical
                shifts for spin chains, qubit frequencies for transmons, etc.).
            coupling_instances: list of length ``n_snapshots``, each element
                is a coupling matrix (J-couplings, capacitive couplings, etc.).
                ``None`` for single-qubit or uncoupled systems.
            **kwargs: Model-specific optional overrides.

        Returns:
            List of ``(D, D)`` complex numpy arrays, one per snapshot.
        """
        ...

    @abstractmethod
    def build_control_ops(self) -> list[np.ndarray]:
        """Return the fixed control operators H_ctrl_k.

        The pulse Hamiltonian is:  Hp(t) = sum_k  u_k(t) * H_ctrl_k

        Returns a list of ``n_controls`` numpy arrays, each of shape ``(D, D)``.
        """
        ...

    @abstractmethod
    def control_amplitudes(
        self,
        cx: torch.Tensor,
        cy: torch.Tensor,
        rabi_freq: torch.Tensor,
        n_h0: int,
    ) -> torch.Tensor:
        """Map waveform outputs (cx, cy) and platform parameters to u_k(t).

        This is where the platform-specific amplitude mapping happens.  For
        example a spin chain interleaves cx/cy and scales by Rabi frequency,
        while a superconducting qubit model maps them to I/Q drive channels.

        Args:
            cx: ``(n_pulse, n_qubits)`` – first waveform quadrature.
            cy: ``(n_pulse, n_qubits)`` – second waveform quadrature.
            rabi_freq: ``(n_rabi, n_qubits)`` – Rabi frequency snapshots.
            n_h0: number of drift Hamiltonian snapshots.

        Returns:
            ``(n_pulse, n_batch, n_controls)`` tensor of control amplitudes,
            where ``n_batch = n_rabi * n_h0``.
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Hilbert space dimension D."""
        ...

    @property
    @abstractmethod
    def n_controls(self) -> int:
        """Number of independent control channels."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, n_qubits: int, params: dict) -> HamiltonianModel:
        """Construct a model instance from a configuration dictionary.

        Each subclass extracts its own specific parameters (coupling_type,
        anharmonicities, etc.) from *params*.

        Args:
            n_qubits: Number of qubits.
            params: The ``"parameters"`` section of the config dict.

        Returns:
            A fully constructed model instance.
        """
        ...

    @classmethod
    def default_config(cls, n_qubits: int) -> dict:
        """Return a complete, runnable default configuration for this model.

        The returned dict can be passed directly to
        :class:`~ctrl_freeq.api.CtrlFreeQAPI` to run an optimisation
        out of the box.  Subclasses should override this to provide
        sensible defaults for their platform.
        """
        raise NotImplementedError(f"{cls.__name__} does not provide a default_config")

    # ------------------------------------------------------------------
    # Convenience helpers (shared by all models)
    # ------------------------------------------------------------------

    def control_ops_tensor(
        self,
        dtype: torch.dtype = torch.complex128,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """Return control operators as a stacked torch tensor.

        Shape: ``(n_controls, D, D)``.
        """
        ops = self.build_control_ops()
        return torch.stack(
            [torch.as_tensor(op, dtype=dtype, device=device) for op in ops]
        )

    @staticmethod
    def _embed_operator(op: np.ndarray, qubit_idx: int, n_qubits: int) -> np.ndarray:
        """Embed a single-site operator into the full Hilbert space via tensor product."""
        I2 = np.eye(op.shape[0], dtype=complex)
        result = np.eye(1, dtype=complex)
        for i in range(n_qubits):
            result = np.kron(result, op if i == qubit_idx else I2)
        return result
