import torch


def apply_pulse_single_qubit_hilbert_torch(
    H_0, pulse_params, duration, peak_amplitudes, rho_0
):
    f, g, Ix, Iy = pulse_params
    amplitude = peak_amplitudes
    dt = duration / len(f)
    h3 = 2 * H_0[:, 0, 0]

    rho_t = rho_0.unsqueeze(1)  # Add a dimension for batch processing
    for i in range(len(f)):
        h1 = amplitude * f[i]
        h2 = amplitude * g[i]
        h_magnitude = 0.5 * torch.sqrt(h1**2 + h2**2 + h3**2)

        U_t = torch.cos(dt * h_magnitude).unsqueeze(-1).unsqueeze(-1) * torch.eye(
            2, dtype=torch.complex64
        ).unsqueeze(0) - 1j * torch.sin(dt * h_magnitude).unsqueeze(-1).unsqueeze(
            -1
        ) / h_magnitude.unsqueeze(-1).unsqueeze(-1) * (
            h1.unsqueeze(-1).unsqueeze(-1) * Ix
            + h2.unsqueeze(-1).unsqueeze(-1) * Iy
            + H_0
        )

        rho_t = torch.matmul(U_t, rho_t)

    rho_final = rho_t.squeeze(1)  # Remove the added dimension
    return rho_final
