import numpy as np
from scipy.linalg import expm


def dissipator_superoperator(collapse_operators):
    r"""Lindblad dissipator as a superoperator on the row-major vec of rho.

    ``vec(A rho B) = (A kron B^T) vec(rho)`` for the row-major vectorisation
    used by ``rho.reshape(-1)``, hence

    .. math::

        S = \sum_k L_k \otimes L_k^{*}
            - \tfrac12 (L_k^\dagger L_k)\otimes I
            - \tfrac12 I \otimes (L_k^\dagger L_k)^{T}
    """
    collapse_operators = np.asarray(collapse_operators)
    D = collapse_operators.shape[-1]
    eye = np.eye(D, dtype=complex)
    S = np.zeros((D * D, D * D), dtype=complex)
    for L in collapse_operators:
        L_dag_L = L.conj().T @ L
        S += (
            np.kron(L, L.conj())
            - 0.5 * np.kron(L_dag_L, eye)
            - 0.5 * np.kron(eye, L_dag_L.T)
        )
    return S


def dissipative_channel(collapse_operators, dt):
    """Exact CPTP dissipative channel ``exp(dt * D)``.

    The Euler step ``rho + dt * D[rho]`` is only an approximation to this and
    leaves the physical state space for step sizes comparable to T1/T2.
    """
    if not np.isfinite(dt):
        raise ValueError(f"Channel duration must be finite, got {dt!r}.")
    if dt < 0:
        raise ValueError(f"Channel duration must be non-negative, got {dt!r}.")
    return expm(dissipator_superoperator(collapse_operators) * dt)


def apply_channel(channel, rho):
    """Apply a row-major vec superoperator to a single density matrix."""
    D = rho.shape[-1]
    return (channel @ rho.reshape(D * D)).reshape(D, D)


def apply_multi_pulse_multi_qubits_hilbert(
    H_0, pulse_params, duration, peak_amplitudes, rho_0
):
    """
    Apply multiple pulses to the system and compute the final density matrix.

    Args:
    - H_0 (np.array): Initial Hamiltonian.
    - rho_0 (np.array): Initial density matrix.
    - pulse_params (list of tuples): Each tuple contains (f, g, Ix, Iy) for a specific spin channel.
    - duration (float): Total time of the pulse.
    - peak_amplitudes (list of floats): List of amplitudes corresponding to each pulse.

    Returns:
    - rho_final (np.array): Final density matrix after pulse.
    """

    rho_t = rho_0
    dt = duration / len(pulse_params[0][0])

    H_t = H_0.copy()
    for i in range(len(pulse_params[0][0])):  # Assumes all f have the same length
        added_term = np.zeros_like(H_0)  # To store the added term for this iteration

        for (f, g, Ix, Iy), amplitude in zip(pulse_params, peak_amplitudes):
            H1_t = amplitude * (f[i] * Ix + g[i] * Iy)
            H_t += H1_t
            added_term += H1_t

        U_t = expm(-1j * H_t * dt)  # Time evolution operator for this slice
        rho_t = U_t @ rho_t

        # Reset H_t for the next iteration
        H_t -= added_term

    rho_final = rho_t
    return rho_final


def apply_multi_pulse_multi_qubits_liouville(
    H_0, pulse_params, duration, peak_amplitudes, rho_0
):
    """
    Apply multiple pulses to the system and compute the final density matrix.

    Args:
    - H_0 (np.array): Initial Hamiltonian.
    - rho_0 (np.array): Initial density matrix.
    - pulse_params (list of tuples): Each tuple contains (f, g, Ix, Iy) for a specific spin channel.
    - duration (float): Total time of the pulse.
    - peak_amplitudes (list of floats): List of amplitudes corresponding to each pulse.

    Returns:
    - rho_final (np.array): Final density matrix after pulse.
    """

    rho_t = rho_0
    dt = duration / len(pulse_params[0][0])

    H_t = H_0.copy()
    for i in range(len(pulse_params[0][0])):  # Assumes all f have the same length
        added_term = np.zeros_like(H_0)  # To store the added term for this iteration

        for (f, g, Ix, Iy), amplitude in zip(pulse_params, peak_amplitudes):
            H1_t = amplitude * (f[i] * Ix + g[i] * Iy)
            H_t += H1_t
            added_term += H1_t

        U_t = expm(-1j * H_t * dt)  # Time evolution operator for this slice
        rho_t = U_t @ rho_t @ U_t.conj().T

        # Reset H_t for the next iteration
        H_t -= added_term

    rho_final = rho_t
    return rho_final


def apply_multi_pulse_multi_qubits_lindblad(
    H_0, pulse_params, duration, peak_amplitudes, rho_0, collapse_operators
):
    """

    Apply multiple pulses with Lindblad dissipation and compute the final density matrix.

    Uses Strang splitting: half dissipative channel, unitary step, half
    dissipative channel.  The channels are exact matrix exponentials of the
    dissipator, so every intermediate state stays a physical density matrix
    and the splitting is second-order accurate in ``dt``.

    Args:
    - H_0 (np.array): Initial Hamiltonian.
    - rho_0 (np.array): Initial density matrix.
    - pulse_params (list of tuples): Each tuple contains (f, g, Ix, Iy) for a specific spin channel.
    - duration (float): Total time of the pulse.
    - peak_amplitudes (list of floats): List of amplitudes corresponding to each pulse.
    - collapse_operators (np.array): Array of shape (n_ops, D, D) containing Lindblad collapse operators.

    Returns:
    - rho_final (np.array): Final density matrix after pulse.
    """

    rho_t = rho_0
    dt = duration / len(pulse_params[0][0])

    half = (
        dissipative_channel(collapse_operators, dt / 2)
        if collapse_operators is not None
        else None
    )

    H_t = H_0.copy()
    for i in range(len(pulse_params[0][0])):
        added_term = np.zeros_like(H_0)

        for (f, g, Ix, Iy), amplitude in zip(pulse_params, peak_amplitudes):
            H1_t = amplitude * (f[i] * Ix + g[i] * Iy)
            H_t += H1_t
            added_term += H1_t

        U_t = expm(-1j * H_t * dt)
        if half is not None:
            rho_t = apply_channel(half, rho_t)
            rho_t = U_t @ rho_t @ U_t.conj().T
            rho_t = apply_channel(half, rho_t)
        else:
            rho_t = U_t @ rho_t @ U_t.conj().T

        H_t -= added_term

    rho_final = rho_t
    return rho_final
