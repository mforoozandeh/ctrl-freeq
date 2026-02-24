import numpy as np
from scipy.linalg import expm


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

    Uses Euler splitting: unitary step followed by dissipative Lindblad step.

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

    H_t = H_0.copy()
    for i in range(len(pulse_params[0][0])):
        added_term = np.zeros_like(H_0)

        for (f, g, Ix, Iy), amplitude in zip(pulse_params, peak_amplitudes):
            H1_t = amplitude * (f[i] * Ix + g[i] * Iy)
            H_t += H1_t
            added_term += H1_t

        # Unitary step
        U_t = expm(-1j * H_t * dt)
        rho_t = U_t @ rho_t @ U_t.conj().T

        # Lindblad dissipator step (Euler)
        if collapse_operators is not None:
            dissipator = np.zeros_like(rho_t)
            for L in collapse_operators:
                L_dag = L.conj().T
                L_dag_L = L_dag @ L
                dissipator += L @ rho_t @ L_dag - 0.5 * (
                    L_dag_L @ rho_t + rho_t @ L_dag_L
                )
            rho_t = rho_t + dt * dissipator

        H_t -= added_term

    rho_final = rho_t
    return rho_final
