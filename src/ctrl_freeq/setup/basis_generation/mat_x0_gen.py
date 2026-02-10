import sys

import numpy as np
from numpy.polynomial import hermite_e


def hs_envelope(x, beta=10.6 / 2, n=1, eps=sys.float_info.epsilon):
    """
    Normalized envelope function to scale between eps and 1.

    :param x: Time array.
    :param beta: Scaling factor, typically 10.6 / duration.
    :param n: Power factor, typically 1.
    :param eps: A small number to ensure the minimum value at extremities.
    :return: Normalized envelope values scaled between eps and 1.
    """
    # Compute the original envelope function
    argument = beta * (x**n)
    envelope = 2 / (np.exp(argument) + np.exp(-argument))

    # Normalize envelope to have its maximum at 1
    envelope_norm = (envelope - np.min(envelope)) / (
        np.max(envelope) - np.min(envelope)
    )

    # Scale between eps and 1
    envelope_scaled = envelope_norm * (1 - eps) + eps

    return envelope_scaled


def g_envelope(x, n=1, sigma=1 / 4, eps=sys.float_info.epsilon):
    """
    Generates a scaled Gaussian envelope with configurable sharpness and width.

    This function calculates a Gaussian-like envelope where the width (sigma) is adjusted
    so that the value of the function at x=1 remains constant across different values of n.
    The resulting envelope is then normalized to the range [eps, 1] to avoid zero values.

    Parameters:
    x (numpy.ndarray): Input array of x values where the envelope is evaluated.
    n (int, optional): The power to which the Gaussian exponent is raised, affecting the sharpness of the peak.
        Default is 1, which results in a standard Gaussian function, >1 results in a super-Gaussian.
    sigma (float, optional): Initial standard deviation of the Gaussian function.
        Default is 0.25, which influences the initial calculation of g_ext.
    eps (float, optional): A small number added to the normalized envelope to avoid zero values.
        Default is the smallest positive float such that 1.0 + eps != 1.0.

    Returns:
    numpy.ndarray: The scaled Gaussian envelope, normalized to the range [eps, 1].

    Examples:
    >>> x = np.linspace(-2, 2, 500)
    >>> envelope = g_envelope(x, n=2, sigma=0.25)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, envelope)
    >>> plt.show()
    """
    g_ext = np.exp(-(((1**2) / (2 * sigma**2)) ** 1))
    sigma_updated = np.sqrt(1 / (2 * ((-np.log(g_ext)) ** (1 / n))))
    envelope = np.exp(-((x**2 / (2 * sigma_updated**2)) ** n))

    envelope_norm = (envelope - np.min(envelope)) / (
        np.max(envelope) - np.min(envelope)
    )

    envelope_scaled = (envelope_norm * (1 - eps)) + eps

    return envelope_scaled


def chebyshev_matrix(x, n):
    basis = np.empty((len(x), n))

    if n > 0:
        basis[:, 0] = 1
    if n > 1:
        basis[:, 1] = x

    for i in range(2, n):
        basis[:, i] = 2 * x * basis[:, i - 1] - basis[:, i - 2]

    return basis


def legendre_matrix(x, n):
    basis = np.empty((len(x), n))

    if n > 0:
        basis[:, 0] = 1
    if n > 1:
        basis[:, 1] = x

    for i in range(2, n):
        basis[:, i] = (
            (2 * i - 1) * x * basis[:, i - 1] - (i - 1) * basis[:, i - 2]
        ) / i

    return basis


def gegenbauer_matrix(x, n, lamb=0.5):
    basis = np.empty((len(x), n))

    if n > 0:
        basis[:, 0] = 1
    if n > 1:
        basis[:, 1] = 2 * lamb * x

    for i in range(2, n):
        basis[:, i] = (
            (2 * (i + lamb - 1) * x * basis[:, i - 1])
            - ((i + 2 * lamb - 2) * basis[:, i - 2])
        ) / i

    return basis


def poly_matrix(x, n):
    basis = np.empty((len(x), n))

    for i in range(n):
        basis[:, i] = x**i

    return basis


def chirp_matrix(x, n):
    basis = np.empty((len(x), n))

    for i in range(n):
        phi = 2 * np.pi * i * (0.5 * x) ** 2
        basis[:, i] = np.cos(phi)

    return basis


def hermite_matrix(x, n):
    basis = np.empty((len(x), n))

    for i in range(n):
        basis[:, i] = hermite_e.hermeval(x, [0] * i + [1])

    return basis


def random_matrix(x, n):
    basis = np.empty((len(x), n))

    for i in range(n):
        basis[:, i] = np.random.rand(len(x))

    return basis


def fourier_matrix(x, n):
    basis = np.empty((len(x), 2 * n + 1))
    # Zero frequency component (constant)
    basis[:, 0] = 1

    for k in range(1, n + 1):
        basis[:, 2 * k - 1] = np.cos(2 * np.pi * k * x)  # Cosine components
        basis[:, 2 * k] = np.sin(2 * np.pi * k * x)  # Sine components

    return basis


def generate_mat_x0_from_basis(para, ntp, wf_type, wf_mode):
    if wf_mode == "polar_phase":
        n_para = len(para)
    else:
        n_para = len(para) // 2

    time_array = np.linspace(-1, 1, ntp)

    wf_type_to_func = {
        "cheb": chebyshev_matrix,
        "leg": legendre_matrix,
        "poly": poly_matrix,
        "chirp": chirp_matrix,
        "hermite": hermite_matrix,
        "gegen": gegenbauer_matrix,
        "random": random_matrix,
    }

    mat = wf_type_to_func[wf_type](time_array, n_para)

    if wf_mode == "polar_phase":
        c = para[0:n_para]
        x0 = c
    else:
        c1 = para[0:n_para]
        c2 = para[n_para : 2 * n_para]
        x0 = np.append(c1, c2)

    return mat, x0


def generate_mat_x0_from_fourier_basis(para, ntp, wf_mode):
    if wf_mode == "polar_phase":
        n = (len(para) - 1) // 2
        n_para = 2 * n + 1
    else:
        n = (len(para) - 2) // 4
        n_para = 4 * n + 2

    time_array = np.linspace(-1, 1, ntp)
    mat = fourier_matrix(time_array, n)

    if wf_mode == "polar_phase":
        c = para[0:n_para]
        x0 = c
    else:
        c1 = para[0 : int(n_para / 2)]
        c2 = para[int(n_para / 2) : n_para]
        x0 = np.append(c1, c2)

    return mat, x0


def amplitude_envelope(x, envelope="gn", order=1):
    if envelope == "quad":
        return 1 - x**2 + sys.float_info.epsilon
    elif envelope == "gn":
        return g_envelope(x, n=order)
    elif envelope == "hs":
        return hs_envelope(x, n=order)


def mat_with_amplitude_and_qr(mat, x0, wf_mode, envelope, order):
    x = np.linspace(-1, 1, mat.shape[0])
    amp = amplitude_envelope(x, envelope=envelope, order=order)
    amp = amp.reshape(-1, 1)

    mat_new = np.empty((2, mat.shape[0], mat.shape[1]))

    if wf_mode == "polar_phase":
        x0_new = np.append(x0, x0[-1])
        mat_new[0] = np.tile(amp, len(x0))
        mat_new[1], _ = np.linalg.qr(mat)

    elif wf_mode == "polar":
        x0_new = x0
        mat_new[0], _ = np.linalg.qr(mat * amp)
        mat_new[1], _ = np.linalg.qr(mat)

    elif wf_mode == "cart":
        x0_new = x0
        mat_new[0], _ = np.linalg.qr(mat * amp)
        mat_new[1], _ = np.linalg.qr(mat * amp)

    return mat_new, x0_new
