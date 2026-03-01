from ctrl_freeq.setup.hamiltonian_generation.base import (
    HamiltonianModel,
    get_hamiltonian_class,
    list_hamiltonians,
    register_hamiltonian,
)

# These imports trigger the @register_hamiltonian decorators
from ctrl_freeq.setup.hamiltonian_generation.spin_chain import SpinChainModel
from ctrl_freeq.setup.hamiltonian_generation.superconducting import (
    SuperconductingQubitModel,
)
from ctrl_freeq.setup.hamiltonian_generation.duffing_transmon import (
    DuffingTransmonModel,
)

from ctrl_freeq.setup.hamiltonian_generation.hamiltonians import (
    createHcs,
    createHJ,
    create_H_total,
)

__all__ = [
    "HamiltonianModel",
    "SpinChainModel",
    "SuperconductingQubitModel",
    "DuffingTransmonModel",
    "register_hamiltonian",
    "get_hamiltonian_class",
    "list_hamiltonians",
    "createHcs",
    "createHJ",
    "create_H_total",
]
