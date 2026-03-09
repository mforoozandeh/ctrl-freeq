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


def _symmetrise_coupling(Jmat):
    """Return the upper-triangular coupling values from *Jmat*.

    Accepts any of the three common conventions:
      - upper-triangular only  (``J[i,j]`` with ``i < j``)
      - lower-triangular only  (``J[j,i]`` with ``j > i``)
      - full symmetric matrix

    When both ``J[i,j]`` and ``J[j,i]`` are non-zero *and* differ, the
    matrix is ambiguous and a ``ValueError`` is raised.  Otherwise the
    non-zero half is mirrored so that callers can always read ``J[i,j]``
    with ``i < j``.
    """
    Jmat = np.asarray(Jmat, dtype=float)
    upper = np.triu(Jmat, k=1)
    lower = np.tril(Jmat, k=-1)

    upper_nonzero = upper != 0
    lower_t = lower.T  # transpose so indices align with upper

    # Both triangles populated at the same (i,j) pair but disagree?
    conflict = upper_nonzero & (lower_t != 0) & ~np.isclose(upper, lower_t)
    if conflict.any():
        idx = tuple(np.argwhere(conflict)[0])
        raise ValueError(
            f"Coupling matrix is asymmetric: J[{idx[0]},{idx[1]}]="
            f"{upper[idx]} != J[{idx[1]},{idx[0]}]={lower_t[idx]}.  "
            f"Provide a symmetric matrix or populate one triangle only."
        )

    # Merge: take whichever triangle is non-zero (or both if equal)
    merged = np.where(upper_nonzero, upper, lower_t)
    return merged + merged.T  # full symmetric matrix


def createHJ(Jmat, Op, coupling_type="Z"):
    """
    Generate the sub-Hamiltonian containing only the interaction part
    for either weak or strong coupling conditions.

    The coupling matrix is symmetrised before use so that upper-triangular,
    lower-triangular, and fully symmetric inputs all produce the same result.

    Args:
    - Jmat (numpy.ndarray): The interaction matrix (symmetric or triangular).
    - Op (dict): The dictionary of spin operators.
    - coupling_type (str): The type of coupling: ``'Z'``, ``'XY'``, or ``'XYZ'``.

    Returns:
    - numpy.ndarray: The interaction sub-Hamiltonian.
    """
    # Normalise legacy lowercase values; fall back to default if None
    coupling_type = (coupling_type or "Z").upper()

    Jmat = _symmetrise_coupling(Jmat)
    nspins = Jmat.shape[0]
    HJ = np.zeros_like(Op["X_1"], dtype=complex)

    if coupling_type == "Z":
        for i in range(nspins):
            for j in range(i + 1, nspins):
                HJ += Jmat[i, j] * Op[f"Z_{i + 1}"] @ Op[f"Z_{j + 1}"]

    elif coupling_type == "XYZ":
        for i in range(nspins):
            for j in range(i + 1, nspins):
                HJ += Jmat[i, j] * (
                    Op[f"X_{i + 1}"] @ Op[f"X_{j + 1}"]
                    + Op[f"Y_{i + 1}"] @ Op[f"Y_{j + 1}"]
                    + Op[f"Z_{i + 1}"] @ Op[f"Z_{j + 1}"]
                )
    elif coupling_type == "XY":
        for i in range(nspins):
            for j in range(i + 1, nspins):
                HJ += Jmat[i, j] * (
                    Op[f"X_{i + 1}"] @ Op[f"X_{j + 1}"]
                    + Op[f"Y_{i + 1}"] @ Op[f"Y_{j + 1}"]
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
