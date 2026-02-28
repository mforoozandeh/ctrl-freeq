from abc import ABC, abstractmethod

import numpy as np
import torch


class HamiltonianModel(ABC):
    """Abstract base class for Hamiltonian models in optimal control.

    Any physical platform (spin chain, superconducting qubit, etc.)
    implements this interface.  The optimizer pipeline only depends on this
    abstraction via the standard bilinear control formulation:

        H(t) = H_drift + sum_k  u_k(t) * H_ctrl_k

    where u_k(t) are scalar control amplitudes and H_ctrl_k are fixed operators.

    Subclasses must implement:
        - build_drift()        : construct H0 matrix/matrices
        - build_control_ops()  : return the fixed control operators
        - control_amplitudes() : map waveform outputs to u_k(t) per batch element
        - dim (property)       : Hilbert space dimension
        - n_controls (property): number of independent control channels
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_drift(self, **params) -> list[np.ndarray]:
        """Build drift Hamiltonian H0 instances.

        Returns a list of ``(D, D)`` complex numpy arrays, one per snapshot.
        The number of snapshots depends on the robustness sampling strategy
        (e.g. offset/J uncertainty for spin chains, or frequency uncertainty
        for superconducting qubits).
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
