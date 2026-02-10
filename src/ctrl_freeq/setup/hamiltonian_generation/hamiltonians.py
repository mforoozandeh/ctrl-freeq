import numpy as np


def createHcs(cs_vals, Op):
    """
    Generate the sub-Hamiltonian for chemical shifts.

    Args:
    - cs_vals (numpy.ndarray or float): The chemical shift values for each spin.
    - Op (dict): The dictionary of spin operators.

    Returns:
    - numpy.ndarray: The chemical shift sub-Hamiltonian.
    """
    cs_vals = np.atleast_1d(cs_vals)
    nspins = len(cs_vals)
    Hcs = np.zeros_like(Op["X_1"], dtype=complex)

    for n in range(nspins):
        Hcs += cs_vals[n] * Op[f"Z_{n + 1}"]

    return Hcs


def createHJ(Jmat, Op, coupling_type="z"):
    """
    Generate the sub-Hamiltonian containing only the interaction part
    for either weak or strong coupling conditions.

    Args:
    - Jmat (numpy.ndarray): The interaction matrix.
    - Op (dict): The dictionary of spin operators.
    - coupling_type (str): The type of coupling, either 'hetero' or 'homo'.

    Returns:
    - numpy.ndarray: The interaction sub-Hamiltonian.
    """

    nspins = Jmat.shape[0]
    HJ = np.zeros_like(Op["X_1"], dtype=complex)

    if coupling_type == "Z":
        for n in range(nspins):
            for k in range(n):
                if n != k:
                    HJ += Jmat[n, k] * Op[f"Z_{k + 1}"] @ Op[f"Z_{n + 1}"]

    elif coupling_type == "XYZ":
        for n in range(nspins):
            for k in range(n):
                if n != k:
                    HJ += Jmat[n, k] * (
                        Op[f"X_{k + 1}"] @ Op[f"X_{n + 1}"]
                        + Op[f"Y_{k + 1}"] @ Op[f"Y_{n + 1}"]
                        + Op[f"Z_{k + 1}"] @ Op[f"Z_{n + 1}"]
                    )
    elif coupling_type == "XY":
        for n in range(nspins):
            for k in range(n):
                if n != k:
                    HJ += Jmat[n, k] * (
                        Op[f"X_{k + 1}"] @ Op[f"X_{n + 1}"]
                        + Op[f"Y_{k + 1}"] @ Op[f"Y_{n + 1}"]
                    )
    else:
        raise ValueError("Coupling type must be either 'Z', 'XYZ', or 'XY'.")
    return HJ


def create_H_total(p):
    if p.n_qubits == 1:
        return createHcs(p.Delta, p.op)

    elif p.n_qubits > 1:
        HCS = createHcs(p.Delta, p.op)

        HJ = createHJ(p.Jmat, p.op, coupling_type=p.coupling_type)

        return HJ + HCS
