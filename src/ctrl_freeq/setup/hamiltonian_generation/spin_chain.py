import numpy as np
import torch

from ctrl_freeq.setup.hamiltonian_generation.base import HamiltonianModel
from ctrl_freeq.setup.hamiltonian_generation.hamiltonians import createHcs, createHJ
from ctrl_freeq.setup.operator_generation.generate_operators import (
    create_hamiltonian_basis,
)


class SpinChainModel(HamiltonianModel):
    """Spin-chain Hamiltonian (NMR / spin-qubit systems).

    Drift Hamiltonian:
        H0 = sum_i Delta_i * Z_i  +  sum_{i<j} J_{ij} * coupling_term_{ij}

    Control Hamiltonian:
        Hp(t) = sum_i [ cx_i(t) * X_i + cy_i(t) * Y_i ] * Omega_R_i

    This wraps the existing ``createHcs`` / ``createHJ`` functions and
    implements the ``HamiltonianModel`` interface so the optimizer pipeline
    is Hamiltonian-agnostic.
    """

    def __init__(self, n_qubits: int, coupling_type: str = "XY"):
        self.n_qubits = n_qubits
        self.coupling_type = coupling_type
        self._op = create_hamiltonian_basis(n_qubits)

    @property
    def dim(self) -> int:
        return 2**self.n_qubits

    @property
    def n_controls(self) -> int:
        return 2 * self.n_qubits  # X and Y per qubit

    def build_drift(
        self,
        Delta_instances: list[np.ndarray],
        J_instances: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Build drift Hamiltonian snapshots.

        Args:
            Delta_instances: list of length ``n_h0``, each element is an array
                of chemical shift values (one per qubit).
            J_instances: list of length ``n_h0``, each element is a coupling
                matrix.  ``None`` for single-qubit systems.

        Returns:
            List of ``(D, D)`` complex numpy arrays.
        """
        H0_list = []
        if self.n_qubits == 1 or J_instances is None:
            for Delta in Delta_instances:
                H0_list.append(createHcs(Delta, self._op))
        else:
            for Delta, J in zip(Delta_instances, J_instances):
                Hcs = createHcs(Delta, self._op)
                HJ = createHJ(J, self._op, coupling_type=self.coupling_type)
                H0_list.append(HJ + Hcs)
        return H0_list

    def build_control_ops(self) -> list[np.ndarray]:
        """Return ``[X_0, Y_0, X_1, Y_1, ...]``."""
        ops = []
        for i in range(self.n_qubits):
            ops.append(self._op[f"X_{i + 1}"])
            ops.append(self._op[f"Y_{i + 1}"])
        return ops

    def control_amplitudes(
        self,
        cx: torch.Tensor,
        cy: torch.Tensor,
        rabi_freq: torch.Tensor,
        n_h0: int,
    ) -> torch.Tensor:
        """Interleave cx/cy and scale by Rabi frequency.

        Args:
            cx: ``(n_pulse, n_qubits)``
            cy: ``(n_pulse, n_qubits)``
            rabi_freq: ``(n_rabi, n_qubits)``
            n_h0: number of drift Hamiltonian snapshots.

        Returns:
            ``(n_pulse, n_rabi * n_h0, 2 * n_qubits)`` tensor.
        """
        n_pulse = cx.shape[0]

        # Interleave: [cx_0, cy_0, cx_1, cy_1, ...]  shape (n_pulse, 2*N)
        u = torch.stack([cx, cy], dim=-1).reshape(n_pulse, 2 * self.n_qubits)

        # Rabi scaling: [Omega_0, Omega_0, Omega_1, Omega_1, ...]
        rabi_expanded = rabi_freq.repeat_interleave(2, dim=-1)  # (n_rabi, 2*N)

        # Tile for n_h0 along batch dim: (n_rabi*n_h0, 2*N)
        rabi_batch = rabi_expanded.repeat(n_h0, 1)

        # Broadcast multiply: (n_pulse, 1, 2N) * (1, n_rabi*n_h0, 2N)
        u = u.unsqueeze(1) * rabi_batch.unsqueeze(0)

        return u  # (n_pulse, n_rabi * n_h0, 2 * n_qubits)
