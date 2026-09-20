import sys

import numpy as np
from numpy.polynomial import hermite_e


def _validate_envelope_grid(x, n):
    """Validate the sample grid and power factor shared by all envelopes."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(
            f"Envelope grid must be a non-empty 1-D array, got shape {x.shape}."
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("Envelope grid must contain only finite values.")
    if np.ndim(n) != 0 or not np.isfinite(n) or n <= 0:
        raise ValueError(f"Envelope order must be positive and finite, got {n!r}.")
    return x


def _normalise_envelope(envelope, eps):
    """Scale *envelope* onto ``[eps, 1]``.

    When the envelope is constant over the grid (``max == min``) the usual
    ``(e - min) / (max - min)`` normalisation is a 0/0 division.  A constant
    envelope carries no shape information, so the agreed behaviour is a flat
    unit envelope.  This happens for grids with one or two points, where every
    sample sits at the same ``|x|`` and therefore has the same envelope value.
    """
    lo = np.min(envelope)
    hi = np.max(envelope)
    span = hi - lo
    if span == 0:
        return np.ones_like(envelope)
    envelope_norm = (envelope - lo) / span
    return envelope_norm * (1 - eps) + eps


def hs_envelope(x, beta=10.6 / 2, n=1, eps=sys.float_info.epsilon):
    """
    Normalized envelope function to scale between eps and 1.

    A constant envelope (e.g. a one- or two-point grid) is returned flat at 1
    instead of producing ``NaN`` from a zero-width normalisation range.

    :param x: Time array.
    :param beta: Scaling factor, typically 10.6 / duration.
    :param n: Power factor, typically 1.
    :param eps: A small number to ensure the minimum value at extremities.
    :return: Normalized envelope values scaled between eps and 1.
    """
    x = _validate_envelope_grid(x, n)

    # Compute the original envelope function
    argument = beta * (x**n)
    envelope = 2 / (np.exp(argument) + np.exp(-argument))

    return _normalise_envelope(envelope, eps)


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
    x = _validate_envelope_grid(x, n)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Envelope sigma must be positive and finite, got {sigma!r}.")

    g_ext = np.exp(-(((1**2) / (2 * sigma**2)) ** 1))
    sigma_updated = np.sqrt(1 / (2 * ((-np.log(g_ext)) ** (1 / n))))
    envelope = np.exp(-((x**2 / (2 * sigma_updated**2)) ** n))

    return _normalise_envelope(envelope, eps)


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
        x = _validate_envelope_grid(x, order)
        return _normalise_envelope(1 - x**2, sys.float_info.epsilon)
    elif envelope == "gn":
        return g_envelope(x, n=order)
    elif envelope == "hs":
        return hs_envelope(x, n=order)
    raise ValueError(
        f"Unknown amplitude envelope {envelope!r}. Choose 'quad', 'gn', or 'hs'."
    )


def _qr_checked(matrix, label):
    """Orthonormalise *matrix* by QR, rejecting rank-deficient inputs.

    ``np.linalg.qr`` happily returns a full set of orthonormal columns for a
    rank-deficient matrix: the columns spanning the null directions are chosen
    arbitrarily.  Using them silently adds waveform directions that the
    requested basis never contained, so a numerically rank-deficient (or
    over-determined) request is rejected instead.

    The tolerance is NumPy's default SVD tolerance from
    :func:`numpy.linalg.matrix_rank`.
    """
    n_rows, n_cols = matrix.shape
    if n_cols > n_rows:
        raise ValueError(
            f"{label}: cannot build {n_cols} independent basis functions from "
            f"{n_rows} sample points. Increase point_in_pulse or reduce n_para."
        )
    rank = np.linalg.matrix_rank(matrix)
    if rank < n_cols:
        raise ValueError(
            f"{label}: the requested basis is numerically rank deficient "
            f"(rank {rank} < {n_cols} columns on {n_rows} sample points). "
            f"QR would silently complete it with directions outside the "
            f"requested basis. Increase point_in_pulse, reduce n_para, or "
            f"choose a better-conditioned wf_type."
        )
    q, _ = np.linalg.qr(matrix)
    return q


def mat_with_amplitude_and_qr(mat, x0, wf_mode, envelope, order):
    x = np.linspace(-1, 1, mat.shape[0])
    amp = amplitude_envelope(x, envelope=envelope, order=order)
    amp = amp.reshape(-1, 1)

    mat_new = np.empty((2, mat.shape[0], mat.shape[1]))

    if wf_mode == "polar_phase":
        x0_new = np.append(x0, x0[-1])
        mat_new[0] = np.tile(amp, len(x0))
        mat_new[1] = _qr_checked(mat, "phase basis")

    elif wf_mode == "polar":
        x0_new = x0
        mat_new[0] = _qr_checked(mat * amp, "amplitude-weighted basis")
        mat_new[1] = _qr_checked(mat, "phase basis")

    elif wf_mode == "cart":
        x0_new = x0
        mat_new[0] = _qr_checked(mat * amp, "amplitude-weighted basis (I)")
        mat_new[1] = _qr_checked(mat * amp, "amplitude-weighted basis (Q)")

    else:
        raise ValueError(
            f"Unknown waveform mode {wf_mode!r}. Choose 'cart', 'polar', or "
            f"'polar_phase'."
        )

    return mat_new, x0_new
