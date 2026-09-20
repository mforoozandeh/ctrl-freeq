import matplotlib.pyplot as plt
import numpy as np


from ctrl_freeq.setup.hamiltonian_generation.hamiltonians import (
    create_H_total,
    createHcs,
    createHJ,
    _symmetrise_coupling,
)


from ctrl_freeq.ctrlfreeq.ctrl_freeq import (
    exp_mat_exact,
    exp_mat_torch,
    pulse_para,
    pulse_hamiltonian,
    pulse_hamiltonian_generic,
    simulate_trajectory,
    state_hilbert,
    state_lindblad,
    state_liouville,
)
from ctrl_freeq.setup.operator_generation.generate_operators import (
    create_hamiltonian_basis_torch,
)
import torch

import plotly.graph_objects as go

from ctrl_freeq.visualisation.plot_settings import plot_style, plotly_style


# from tikzplotlib import save as tikz_save


def _waveform_sample_times(T, n_pulse):
    """Midpoint sample times ``(k + 1/2) T / N`` of the propagation intervals.

    Matches ``Initialise.generate_time_sequence``; state boundaries live at
    ``k T / N`` instead and are returned by ``Initialise.state_boundary_times``.
    """
    return (np.arange(n_pulse) + 0.5) * (T / n_pulse)


@plot_style()
def plot_pulses_iq(cxs, cys, T, plot_step=True, plot_line=True):
    """
    :param cxs: List of real parts of complex numbers for multiple signals.
    :param cys: List of imaginary parts of complex numbers for multiple signals.
    :param T: Total duration.
    :param plot_step: Boolean to control plotting of step graphs.
    :param plot_line: Boolean to control plotting of line graphs.

    :return: The figure object containing the plot.
    """
    # Waveform samples are the midpoints of the N propagation intervals.
    t = _waveform_sample_times(T, len(cxs[0])) * 1e9  # Convert to ns

    # Create a subplot for each pair of cx and cy
    fig, axes = plt.subplots(len(cxs), 1, sharex="all")

    if isinstance(axes, plt.Axes):
        axes = [axes]

    # Loop over each cx and cy pair and create a subplot
    for i, (cx, cy) in enumerate(zip(cxs, cys)):
        ax = axes[i]

        # Plot step if requested
        if plot_step:
            ax.step(t, cx, linewidth=1, where="mid", label=r"$I$")
            ax.step(t, cy, linewidth=1, where="mid", label=r"$Q$")

        # Plot line if requested
        if plot_line:
            ax.plot(t, cx, "--", linewidth=0.5, alpha=0.5)
            ax.plot(t, cy, "--", linewidth=0.5, alpha=0.5)

        # Add legend if there is something to plot
        if plot_step or plot_line:
            if i == 0:
                ax.legend(loc="upper center", ncol=2)

        # Set y-label
        ax.set_ylabel(f"Qubit {i + 1}")

        # # Set the same y-axis limits for all subplots
        # ax.set_ylim(global_ymin, global_ymax)

        # Manage x-axis labels and tick labels
        if i != len(cxs) - 1:
            ax.set_xlabel("")  # Remove x-label
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0)  # Remove x-ticks
        else:
            ax.set_xlabel("Time / ns")  # Keep x-label

    return fig


@plot_style()
def plot_pulses_amp_phi(cxs, cys, T, plot_step=True, plot_line=True):
    """
    :param cxs: List of real parts of complex numbers for multiple signals.
    :param cys: List of imaginary parts of complex numbers for multiple signals.
    :param T: Total duration.
    :param plot_step: Boolean to control plotting of step graphs.
    :param plot_line: Boolean to control plotting of line graphs.
    :return: The figure object containing the plot.
    """

    # Waveform samples are the midpoints of the N propagation intervals.
    t = _waveform_sample_times(T, len(cxs[0])) * 1e9  # Convert to ns

    global_ymax_phi = 1.05 * np.pi
    global_ymin_phi = -global_ymax_phi

    # Create a subplot for each pair of cx and cy
    fig, axes = plt.subplots(len(cxs), 1, sharex="all")

    if isinstance(axes, plt.Axes):
        axes = [axes]

    # Loop over each cx and cy pair and create a subplot
    for i, (cx, cy) in enumerate(zip(cxs, cys)):
        ax = axes[i]

        # Compute amplitude and phase
        amp = np.sqrt(cx**2 + cy**2)
        phi = np.arctan2(cy, cx)

        # Create a second y-axis for phase
        ax2 = ax.twinx()

        # Plot step if requested
        if plot_step:
            ax.step(t, amp, linewidth=1, where="mid", label=r"$Amplitude$")
            ax2.step(t, phi, linewidth=1, where="mid", label=r"$Phase$", color="orange")

        # Plot line if requested
        if plot_line:
            ax.plot(t, amp, "--", linewidth=0.5, alpha=0.5)
            ax2.plot(t, phi, "--", linewidth=0.5, alpha=0.5, color="orange")

        # Add legend if there is something to plot
        if plot_step or plot_line:
            if i == 0:
                ax.legend(loc="upper left")
                ax2.legend(loc="upper right")

        # Set y-labels
        ax.set_ylabel(f"Qubit {i + 1} Amplitude")
        ax2.set_ylabel(f"Qubit {i + 1} Phase")

        # Set the same y-axis limits for all subplots
        # ax.set_ylim(global_ymin_amp, global_ymax_amp)
        ax2.set_ylim(global_ymin_phi, global_ymax_phi)

        # Manage x-axis labels and tick labels
        if i != len(cxs) - 1:
            ax.set_xlabel("")  # Remove x-label
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0)  # Remove x-ticks
        else:
            ax.set_xlabel("Time / ns")  # Keep x-label

    return fig


@plot_style()
def plot_history_with_mean_lu_liouville(
    history,
    history_mean,
    time_vector,
):
    """
    Plot a grid of line plots with mean and standard deviation bands for history data.

    Parameters:
    history (np.ndarray): The history data with shape (n, n, k, l).
    history_mean (np.ndarray): The mean history data with shape (n, n, k).
    time_vector (np.ndarray): The time vector with shape (k,).

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """

    time_vector = time_vector * 1e9  # Convert to ns

    # Determine the size of the grid
    n, _, k, num_samples = history.shape

    # Determine global y-axis limits based on max and min values
    global_ymax = 1.05 * np.max(np.abs(history))
    global_ymin = -global_ymax

    fig, axes = plt.subplots(nrows=n, ncols=n, squeeze=False)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            # Plot each individual history in pale color
            for m in range(num_samples):
                ax.plot(
                    time_vector,
                    history[i, j, :, m].real,
                    color="lightblue",
                    linewidth=0.2,
                )
                ax.plot(
                    time_vector,
                    history[i, j, :, m].imag,
                    color="lightcoral",
                    linewidth=0.2,
                )

            # Plot the solid mean line for real and imaginary components
            ax.plot(
                time_vector,
                history_mean[i, j, :].real,
                color="blue",
                linewidth=1,
            )
            ax.plot(
                time_vector,
                history_mean[i, j, :].imag,
                color="red",
                linewidth=1,
            )

            # Set the same y-axis limits for all subplots
            ax.set_ylim(global_ymin, global_ymax)

            # Only show y-axis labels on the leftmost subplots
            if j > 0:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", which="both", length=0)

            # Only show x-axis labels on the bottom subplots
            if i == n - 1:
                ax.set_xlabel("Time / ns")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
                ax.tick_params(axis="x", which="both", length=0)

            # Add the LaTeX-formatted rho element in the corner of each subplot
            # Adjust the position using transform=ax.transAxes
            ax.text(
                0.05,
                0.95,  # Position: (x, y) in axes coordinates
                rf"$\rho_{{{i + 1}{j + 1}}}$",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="left",
            )

    return fig


@plot_style()
def plot_history_with_mean_lu_hilbert(
    history,
    history_mean,
    time_vector,
):
    """
    Plot line plots with mean and standard deviation bands for history data in Hilbert space.

    Parameters:
    history (np.ndarray): The history data with shape (n, k, l).
    history_mean (np.ndarray): The mean history data with shape (n, k).
    time_vector (np.ndarray): The time vector with shape (k,).

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """

    time_vector = time_vector * 1e9  # Convert to ns

    n, k, num_samples = history.shape  # n is the dimension of the Hilbert space

    # Determine global y-axis limits based on max and min values
    global_ymax = 1.05 * np.max(np.abs(history))
    global_ymin = -global_ymax

    # Create a grid of subplots without specifying figsize
    fig, axes = plt.subplots(nrows=n, ncols=1, squeeze=False)

    for i in range(n):
        ax = axes[i, 0]

        # Plot each individual history in pale color
        for j in range(num_samples):
            ax.plot(time_vector, history[i, :, j].real, linewidth=0.2)
            ax.plot(time_vector, history[i, :, j].imag, linewidth=0.2)

        # Plot the solid mean line for real and imaginary components
        ax.plot(
            time_vector,
            history_mean[i, :].real,
            linewidth=1,
            label=r"$Re$",
        )
        ax.plot(
            time_vector,
            history_mean[i, :].imag,
            linewidth=1,
            label=r"$Im$",
        )

        # Set the same y-axis limits for all subplots
        ax.set_ylim(global_ymin, global_ymax)

        # Set y-label
        ax.set_ylabel(r"$\psi_{%d}$" % (i + 1))

        # Manage x-axis labels and tick labels
        if i != n - 1:
            ax.set_xlabel("")  # Remove x-label
            ax.set_xticklabels([])
            ax.tick_params(axis="x", which="both", length=0)  # Remove x-ticks
        else:
            ax.set_xlabel("Time / ns")  # Keep x-label

        if i == 0:
            ax.legend(loc="upper center", ncol=2)

    return fig


@plot_style()
def plot_observable_dynamics_liouville(
    history, history_mean, time_vector, ops, n_qubits
):
    """
    Plot for observable dynamics in Liouville space.

    Parameters:
    - history (np.ndarray): The history data with shape (4, 4, k, l).
    - history_mean (np.ndarray): The mean history data with shape (4, 4, k).
    - time_vector (np.ndarray): The time vector with shape (k,).
    - ops (dict): Dictionary of operators.
    - n_qubits (int): Number of qubits.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """

    time_vector = time_vector * 1e9  # Convert to ns

    # Initialize the plot with 3 rows (for X, Y, Z) and columns equal to the number of qubits
    # Removed 'figsize' since it's handled by the decorator
    fig, axs = plt.subplots(3, max(n_qubits, 1), sharex="col")

    # Ensure axs is always a 2D array
    if n_qubits == 1:
        axs = axs.reshape(3, 1)  # Convert to a 2D array with a single column

    # Define the labels for each row
    row_labels = ["X", "Y", "Z"]

    # First pass: Determine the global y-axis limits
    global_max = 0  # Initialize to zero

    for qubit in range(n_qubits):
        for axis in row_labels:
            operator_key = f"{axis}_{qubit + 1}"  # Construct the key for the operator
            op = ops[operator_key]

            # Calculate observable values
            observable_values = np.array(
                [
                    calculate_observable(
                        history[:, :, i, j], op, n_qubits, space_type="liouville"
                    )
                    for i in range(history.shape[2])
                    for j in range(history.shape[3])
                ]
            ).reshape(history.shape[2], history.shape[3])

            # Update global_max if current data has a larger absolute value
            current_max = np.max(np.abs(observable_values))
            if current_max > global_max:
                global_max = current_max

    # Set global y-axis limits with a 5% margin
    global_ymax = 1.05 * global_max
    global_ymin = -global_ymax

    # Second pass: Plot the data with determined y-axis limits
    for qubit in range(n_qubits):
        for axis_index, axis in enumerate(row_labels):
            operator_key = f"{axis}_{qubit + 1}"  # Construct the key for the operator
            op = ops[operator_key]

            # Calculate observable values
            observable_values = np.array(
                [
                    calculate_observable(
                        history[:, :, i, j], op, n_qubits, space_type="liouville"
                    )
                    for i in range(history.shape[2])
                    for j in range(history.shape[3])
                ]
            ).reshape(history.shape[2], history.shape[3])

            # Select the appropriate axis
            ax = axs[axis_index, qubit]

            # Plot each individual observable history in pale color
            for j in range(history.shape[3]):
                ax.plot(
                    time_vector,
                    observable_values[:, j],
                    color="lightblue",
                    linewidth=0.2,
                )

            # Calculate mean observable
            observable_mean = np.array(
                [
                    calculate_observable(
                        history_mean[:, :, i], op, n_qubits, space_type="liouville"
                    )
                    for i in range(history_mean.shape[2])
                ]
            )  # Shape: (k,)

            # Plotting the mean observable in bold color
            ax.plot(
                time_vector,
                observable_mean,
                color="blue",
                linewidth=1,
            )

            # Set y-axis limits
            ax.set_ylim(global_ymin, global_ymax)

            # Set y-labels only on the first column
            if qubit == 0:
                ax.set_ylabel(axis)
            else:
                ax.set_ylabel("")  # Hide y-labels for other columns

            # Remove y-axis tick labels for columns except the first one
            if qubit != 0:
                ax.set_yticklabels([])  # Remove y-axis tick labels
                ax.set_yticklabels([])  # Remove y-axis tick labels
                ax.tick_params(axis="y", which="both", length=0)  # Remove y-ticks

            # Set x-labels only on the bottom row
            if axis_index == 2:
                ax.set_xlabel("Time / ns")
                ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
            else:
                ax.set_xlabel("")  # Hide x-labels for other rows
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # Set column titles for each qubit
    for qubit in range(n_qubits):
        ax_top = axs[0, qubit]
        ax_top.set_title(f"Qubit {qubit + 1}", fontsize=10, pad=10)

    return fig


@plot_style()
def plot_observable_dynamics_hilbert(history, history_mean, time_vector, ops, n_qubits):
    """
    Plot for observable dynamics in Hilbert space.

    Parameters:
    - history (np.ndarray): The history data with shape (4, k, l).
    - history_mean (np.ndarray): The mean history data with shape (4, k).
    - time_vector (np.ndarray): The time vector with shape (k,).
    - ops (dict): Dictionary of operators.
    - n_qubits (int): Number of qubits.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """

    time_vector = time_vector * 1e9  # Convert to ns

    # Initialize the plot with 3 rows (for X, Y, Z) and columns equal to the number of qubits
    fig, axs = plt.subplots(3, max(n_qubits, 1), sharex="col")

    # Ensure axs is always a 2D array
    if n_qubits == 1:
        axs = axs.reshape(3, 1)  # Convert to a 2D array with a single column

    # Define the labels for each row
    row_labels = ["X", "Y", "Z"]

    # First pass: Determine the global y-axis limits
    global_max = 0  # Initialize to zero

    for qubit in range(n_qubits):
        for axis in row_labels:
            operator_key = f"{axis}_{qubit + 1}"  # Construct the key for the operator
            op = ops[operator_key]

            # Calculate observable values
            observable_values = np.array(
                [
                    calculate_observable(
                        history[:, i, j], op, n_qubits, space_type="hilbert"
                    )
                    for i in range(history.shape[1])
                    for j in range(history.shape[2])
                ]
            ).reshape(history.shape[1], history.shape[2])

            # Update global_max if current data has a larger absolute value
            current_max = np.max(np.abs(observable_values))
            if current_max > global_max:
                global_max = current_max

    # Set global y-axis limits with a 5% margin
    global_ymax = 1.05 * global_max
    global_ymin = -global_ymax

    # Second pass: Plot the data with determined y-axis limits
    for qubit in range(n_qubits):
        for axis_index, axis in enumerate(row_labels):
            operator_key = f"{axis}_{qubit + 1}"  # Construct the key for the operator
            op = ops[operator_key]

            # Calculate observable values
            observable_values = np.array(
                [
                    calculate_observable(
                        history[:, i, j], op, n_qubits, space_type="hilbert"
                    )
                    for i in range(history.shape[1])
                    for j in range(history.shape[2])
                ]
            ).reshape(history.shape[1], history.shape[2])

            # Select the appropriate axis
            ax = axs[axis_index, qubit]

            # Plot each individual observable history in pale color
            for j in range(history.shape[2]):
                ax.plot(
                    time_vector,
                    observable_values[:, j],
                    color="lightblue",
                    linewidth=0.2,
                )

            # Calculate mean observable
            observable_mean = np.array(
                [
                    calculate_observable(
                        history_mean[:, i], op, n_qubits, space_type="hilbert"
                    )
                    for i in range(history_mean.shape[1])
                ]
            )  # Shape: (k,)

            # Plotting the mean observable in bold color
            ax.plot(
                time_vector,
                observable_mean,
                color="blue",
                linewidth=1,
            )

            # Set y-axis limits
            ax.set_ylim(global_ymin, global_ymax)

            # Set y-labels only on the first column
            if qubit == 0:
                ax.set_ylabel(axis)
            else:
                ax.set_ylabel("")  # Hide y-labels for other columns
                ax.set_yticklabels([])  # Remove y-axis tick labels
                ax.tick_params(axis="y", which="both", length=0)  # Remove y-ticks

            # Set x-labels only on the bottom row
            if axis_index == 2:
                ax.set_xlabel("Time / ns")
                ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
            else:
                ax.set_xlabel("")  # Hide x-labels for other rows
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # Set column titles for each qubit
    for qubit in range(n_qubits):
        ax_top = axs[0, qubit]
        ax_top.set_title(f"Qubit {qubit + 1}", fontsize=10, pad=10)

    return fig


@plot_style()
def plot_excitation_profiles(x, p, rho_0, num_points, waveform_spec=None):
    """
    Plot excitation profiles in a 3 by n_qubits subplot layout.

    Parameters:
    - x: Parameters for the waveform generation.
    - p: Parameters object containing various settings.
    - rho_0: Initial density matrix.
    - num_points: Number of points for the frequency range.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Obtain the final density matrices for the range of frequencies

    rho_end = get_final_rho_for_excitation_profile(
        x, p, rho_0, num_points, waveform_spec
    )

    # Prepare the frequency ranges for each qubit
    frequencies = []
    for i in range(p.n_qubits):
        sw = p.sw[i] / 1e6
        om = p.Delta[i] / 1e6
        frequencies.append(
            linear_distribution(om / (2 * np.pi), 1.5 * sw / (2 * np.pi), num_points)
        )

    # Initialize the plot with 3 rows (for X, Y, Z) and columns equal to the number of qubits
    fig, axs = plt.subplots(3, max(p.n_qubits, 1), sharex="col")

    # Ensure axs is always a 2D array
    if p.n_qubits == 1:
        axs = axs.reshape(3, 1)  # Convert to a 2D array with a single column

    # Define the labels for each row
    row_labels = ["X", "Y", "Z"]

    # Determine global y-axis limits based on max and min values
    global_ymax = 1.05
    global_ymin = -1.05

    # Plotting for each qubit
    for qubit in range(p.n_qubits):
        qubit_index = qubit + 1  # Adjust index for naming convention in 'op'
        excitation_profile = {"X": [], "Y": [], "Z": []}

        # Calculate the observables for this qubit
        for state in rho_end:
            for axis in ["X", "Y", "Z"]:
                operator_key = f"{axis}_{qubit_index}"
                value = calculate_observable(
                    state, p.obs_op[operator_key], p.n_qubits, p.space
                )
                excitation_profile[axis].append(value)

        # Plot the X, Y, and Z components using the specific frequency range for this qubit
        for axis_index, axis in enumerate(row_labels):
            ax = axs[axis_index, qubit]
            ax.plot(
                frequencies[qubit],
                excitation_profile[axis],
                linewidth=1,
                label=f"{axis}",
            )

            # Set y-axis limits
            ax.set_ylim(global_ymin, global_ymax)

            # Set y-labels only on the first column
            if qubit == 0:
                ax.set_ylabel(axis)
            else:
                ax.set_ylabel("")  # Hide y-labels for other columns
                ax.set_yticklabels([])  # Remove y-axis tick labels
                ax.tick_params(axis="y", which="both", length=0)  # Remove y-ticks

            # Set x-labels only on the bottom row
            if axis_index == 2:
                ax.set_xlabel(r"$f$ / MHz")
                ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)
            else:
                ax.set_xlabel("")  # Hide x-labels for other rows
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # Set column titles for each qubit
    for qubit in range(p.n_qubits):
        ax_top = axs[0, qubit]
        ax_top.set_title(f"Qubit {qubit + 1}", fontsize=10, pad=10)

    return fig


@plotly_style(show=True)
def plot_bloch_sphere_dynamics_hilbert(
    history,
    history_mean,
    ops,
    n_qubits,
    show=True,
):
    """
    Plot the dynamics of qubit states on a Bloch sphere using Plotly for interactivity.

    Parameters:
    history (np.ndarray): The history data with shape (n, k, l).
    history_mean (np.ndarray): The mean history data with shape (n, k).
    ops (dict): Dictionary of operators.
    n_qubits (int): Number of qubits.
    show (bool): Whether to display the plot (default: True).
    """
    fig = go.Figure()

    # Draw the Bloch sphere surface
    theta = np.linspace(0.0, np.pi, 50)
    phi = np.linspace(0.0, 2 * np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            opacity=0.1,
            showscale=False,
            hoverinfo="skip",
        )
    )

    # Get the operators for this qubit
    X_op = ops["X_1"]
    Y_op = ops["Y_1"]
    Z_op = ops["Z_1"]

    num_samples = history.shape[2]

    # Plot individual trajectories
    for sample_idx in range(num_samples):
        bloch_vectors = []

        for t_idx in range(history.shape[1]):
            state_vector = history[:, t_idx, sample_idx]

            X_exp = calculate_observable(
                state_vector, X_op, n_qubits, space_type="hilbert"
            )
            Y_exp = calculate_observable(
                state_vector, Y_op, n_qubits, space_type="hilbert"
            )
            Z_exp = calculate_observable(
                state_vector, Z_op, n_qubits, space_type="hilbert"
            )

            bloch_vectors.append([X_exp, Y_exp, Z_exp])

        bloch_vectors = np.array(bloch_vectors)

        fig.add_trace(
            go.Scatter3d(
                x=bloch_vectors[:, 0],
                y=bloch_vectors[:, 1],
                z=bloch_vectors[:, 2],
                mode="lines",
                line=dict(color="lightblue", width=1),
                opacity=0.5,
                showlegend=False,
            )
        )

    # Plot the mean trajectory
    bloch_vectors_mean = []

    for t_idx in range(history_mean.shape[1]):
        state_vector = history_mean[:, t_idx]

        X_exp = calculate_observable(state_vector, X_op, n_qubits, space_type="hilbert")
        Y_exp = calculate_observable(state_vector, Y_op, n_qubits, space_type="hilbert")
        Z_exp = calculate_observable(state_vector, Z_op, n_qubits, space_type="hilbert")

        bloch_vectors_mean.append([X_exp, Y_exp, Z_exp])

    bloch_vectors_mean = np.array(bloch_vectors_mean)

    fig.add_trace(
        go.Scatter3d(
            x=bloch_vectors_mean[:, 0],
            y=bloch_vectors_mean[:, 1],
            z=bloch_vectors_mean[:, 2],
            mode="lines",
            line=dict(color="blue", width=4),
            name="Mean Trajectory",
        )
    )

    # Starting point
    fig.add_trace(
        go.Scatter3d(
            x=[bloch_vectors_mean[0, 0]],
            y=[bloch_vectors_mean[0, 1]],
            z=[bloch_vectors_mean[0, 2]],
            mode="markers",
            marker=dict(color="green", size=5),
            name="Start (Mean)",
        )
    )

    # Ending point
    fig.add_trace(
        go.Scatter3d(
            x=[bloch_vectors_mean[-1, 0]],
            y=[bloch_vectors_mean[-1, 1]],
            z=[bloch_vectors_mean[-1, 2]],
            mode="markers",
            marker=dict(color="red", size=5),
            name="End (Mean)",
        )
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
            xaxis=dict(range=[-1, 1], autorange=False),
            yaxis=dict(range=[-1, 1], autorange=False),
            zaxis=dict(range=[-1, 1], autorange=False),
            camera=dict(
                eye=dict(
                    x=1.5, y=1.5, z=1.5
                ),  # Adjust eye position for better centering
                center=dict(x=0, y=0, z=0),  # Ensure camera centers on origin
                up=dict(x=0, y=0, z=1),  # Define the up direction
            ),
        ),
        showlegend=True,
        legend=dict(
            x=0.5,  # Horizontal position (0: left, 1: right)
            y=0.95,  # Vertical position (0: bottom, 1: top)
            xanchor="center",  # Anchors the legend's x position ('left', 'center', 'right')
            yanchor="top",  # Anchors the legend's y position ('top', 'middle', 'bottom')
            bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent white background
            bordercolor="black",  # Black border color
            borderwidth=1,  # Border width in pixels
            font=dict(
                size=10,  # Font size of legend text
                color="black",  # Font color of legend text
            ),
        ),
    )
    return fig


@plotly_style
def plot_bloch_sphere_dynamics_liouville(
    history,
    history_mean,
    ops,
    n_qubits,
):
    """
    Plot the dynamics of qubit states on a Bloch sphere using Plotly for interactivity,
    adapted for states represented in Liouville space.

    Parameters:
    history (np.ndarray): The history data with shape (4, 4, k, l).
    history_mean (np.ndarray): The mean history data with shape (4, 4, k).
    ops (dict): Dictionary of operators.
    n_qubits (int): Number of qubits.
    show (bool): Whether to display the plot (default: True).
    """
    fig = go.Figure()

    # Draw the Bloch sphere surface
    theta = np.linspace(0.0, np.pi, 50)
    phi = np.linspace(0.0, 2 * np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            opacity=0.1,
            showscale=False,
            hoverinfo="skip",
        )
    )

    # Get the operators for this qubit
    X_op = ops["X_1"]
    Y_op = ops["Y_1"]
    Z_op = ops["Z_1"]

    num_samples = history.shape[3]  # Updated index for Liouville space

    # Plot individual trajectories
    for sample_idx in range(num_samples):
        bloch_vectors = []

        for t_idx in range(history.shape[2]):  # Updated index for Liouville space
            # Extract the density matrix at this time step for this sample
            density_matrix = history[:, :, t_idx, sample_idx]

            # Calculate the expectation values
            X_exp = calculate_observable(
                density_matrix, X_op, n_qubits, space_type="liouville"
            )
            Y_exp = calculate_observable(
                density_matrix, Y_op, n_qubits, space_type="liouville"
            )
            Z_exp = calculate_observable(
                density_matrix, Z_op, n_qubits, space_type="liouville"
            )

            bloch_vectors.append([X_exp, Y_exp, Z_exp])

        bloch_vectors = np.array(bloch_vectors)

        fig.add_trace(
            go.Scatter3d(
                x=bloch_vectors[:, 0],
                y=bloch_vectors[:, 1],
                z=bloch_vectors[:, 2],
                mode="lines",
                line=dict(color="lightblue", width=1),
                opacity=0.5,
                showlegend=False,
            )
        )

    # Plot the mean trajectory
    bloch_vectors_mean = []

    for t_idx in range(history_mean.shape[2]):  # Updated index for Liouville space
        # Extract the mean density matrix at this time step
        density_matrix_mean = history_mean[:, :, t_idx]

        # Calculate the expectation values
        X_exp = calculate_observable(
            density_matrix_mean, X_op, n_qubits, space_type="liouville"
        )
        Y_exp = calculate_observable(
            density_matrix_mean, Y_op, n_qubits, space_type="liouville"
        )
        Z_exp = calculate_observable(
            density_matrix_mean, Z_op, n_qubits, space_type="liouville"
        )

        bloch_vectors_mean.append([X_exp, Y_exp, Z_exp])

    bloch_vectors_mean = np.array(bloch_vectors_mean)

    fig.add_trace(
        go.Scatter3d(
            x=bloch_vectors_mean[:, 0],
            y=bloch_vectors_mean[:, 1],
            z=bloch_vectors_mean[:, 2],
            mode="lines",
            line=dict(color="blue", width=4),
            name="Mean Trajectory",
        )
    )

    # Starting point
    fig.add_trace(
        go.Scatter3d(
            x=[bloch_vectors_mean[0, 0]],
            y=[bloch_vectors_mean[0, 1]],
            z=[bloch_vectors_mean[0, 2]],
            mode="markers",
            marker=dict(color="green", size=5),
            name="Start (Mean)",
        )
    )

    # Ending point
    fig.add_trace(
        go.Scatter3d(
            x=[bloch_vectors_mean[-1, 0]],
            y=[bloch_vectors_mean[-1, 1]],
            z=[bloch_vectors_mean[-1, 2]],
            mode="markers",
            marker=dict(color="red", size=5),
            name="End (Mean)",
        )
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube",
            xaxis=dict(range=[-1, 1], autorange=False),
            yaxis=dict(range=[-1, 1], autorange=False),
            zaxis=dict(range=[-1, 1], autorange=False),
            camera=dict(
                eye=dict(
                    x=1.5, y=1.5, z=1.5
                ),  # Adjust eye position for better centering
                center=dict(x=0, y=0, z=0),  # Ensure camera centers on origin
                up=dict(x=0, y=0, z=1),  # Define the up direction
            ),
        ),
        showlegend=True,
        legend=dict(
            x=0.5,  # Horizontal position (0: left, 1: right)
            y=0.95,  # Vertical position (0: bottom, 1: top)
            xanchor="center",  # Anchors the legend's x position ('left', 'center', 'right')
            yanchor="top",  # Anchors the legend's y position ('top', 'middle', 'bottom')
            bgcolor="rgba(255, 255, 255, 0.5)",  # Semi-transparent white background
            bordercolor="black",  # Black border color
            borderwidth=1,  # Border width in pixels
            font=dict(
                size=10,  # Font size of legend text
                color="black",  # Font color of legend text
            ),
        ),
    )
    return fig


def process_and_plot(x, p, save_plots=False, show_plots=False, waveform_spec=None):
    """
    Process the optimization solution and generate plots.

    Args:
        x: The optimization solution
        p: The parameters object
        save_plots: Whether to save the plots to files (default: False)
        show_plots: Whether to display the plots (default: False)

    Returns:
        all_waveforms: Dictionary containing the waveforms for all initial states
    """
    from datetime import datetime

    # Create plots directory if saving plots
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent
    plots_dir = project_root / "results" / "plots"
    if save_plots and not plots_dir.exists():
        plots_dir.mkdir(parents=True)

    plots_dir = str(plots_dir)

    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # List to store all figures
    figures = []

    # Dictionary to store all waveforms
    all_waveforms = {"timestamp": timestamp, "waveforms": []}

    # Generate waveforms once (they are identical for all initial states)
    # Use the first initial state as reference since waveforms don't depend on initial state
    first_rho_0 = p.init[0]

    cxs, cys, _, _, _ = compute_and_store_evolution(x, p, first_rho_0, waveform_spec)

    # State trajectories are recorded at the N+1 propagation boundaries.
    boundary_times = p.state_boundary_times()

    # Waveforms are always identical since there is one set of parameters that generate them
    waveforms_identical = True

    # Plot waveforms only once if they are identical
    if waveforms_identical:
        # Plot pulses IQ once
        fig_iq = plot_pulses_iq(
            cxs,
            cys,
            p.pulse_duration,
            additional_settings={"figsize": (3.35, 2.5 * p.n_qubits)},
        )
        figures.append(fig_iq)
        if show_plots:
            plt.show()
        if save_plots:
            plt.figure(fig_iq.number)
            plt.savefig(
                f"{plots_dir}/pulses_iq_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

        # Plot pulses amplitude and phase once
        fig_amp_phi = plot_pulses_amp_phi(
            cxs,
            cys,
            p.pulse_duration,
            additional_settings={"figsize": (3.35, 2.5 * p.n_qubits)},
        )
        figures.append(fig_amp_phi)
        if show_plots:
            plt.show()
        if save_plots:
            plt.figure(fig_amp_phi.number)
            plt.savefig(
                f"{plots_dir}/pulses_amp_phi_{timestamp}.png",
                dpi=300,
                bbox_inches="tight",
            )

    for rho_idx, rho_0 in enumerate(p.init):
        # Compute evolution for this specific initial state (needed for other plots)

        _, _, history, history_mean, leakage = compute_and_store_evolution(
            x, p, rho_0, waveform_spec
        )

        if _leakage_is_visible(leakage):
            fig_leak = plot_leakage(
                leakage,
                boundary_times,
                additional_settings={"figsize": (3.35, 2.5)},
            )
            figures.append(fig_leak)
            if show_plots:
                plt.show()
            if save_plots:
                plt.figure(fig_leak.number)
                plt.savefig(
                    f"{plots_dir}/leakage_{timestamp}_rho{rho_idx}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

        # Store the waveforms for this initial state
        all_waveforms["waveforms"].append({"rho_idx": rho_idx, "cxs": cxs, "cys": cys})

        if p.space == "liouville":
            # Plot history with mean (liouville)
            fig_history = plot_history_with_mean_lu_liouville(
                history=history,
                history_mean=history_mean,
                time_vector=boundary_times,
                additional_settings={"figsize": (3.35 * p.n_qubits, 3.35 * p.n_qubits)},
            )
            figures.append(fig_history)
            if show_plots:
                plt.show()
            if save_plots:
                plt.figure(fig_history.number)
                plt.savefig(
                    f"{plots_dir}/history_liouville_{timestamp}_rho{rho_idx}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

            # Plot observable dynamics (liouville)
            fig_obs = plot_observable_dynamics_liouville(
                history,
                history_mean,
                boundary_times,
                p.obs_op,
                p.n_qubits,
            )
            figures.append(fig_obs)
            if show_plots:
                plt.show()
            if save_plots:
                plt.figure(fig_obs.number)
                plt.savefig(
                    f"{plots_dir}/observable_dynamics_liouville_{timestamp}_rho{rho_idx}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

        elif p.space == "hilbert":
            # Plot history with mean (hilbert)
            fig_history = plot_history_with_mean_lu_hilbert(
                history=history,
                history_mean=history_mean,
                time_vector=boundary_times,
                additional_settings={"figsize": (3.35, 3.35 * p.n_qubits)},
            )
            figures.append(fig_history)
            if show_plots:
                plt.show()
            if save_plots:
                plt.figure(fig_history.number)
                plt.savefig(
                    f"{plots_dir}/history_hilbert_{timestamp}_rho{rho_idx}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

            # Plot observable dynamics (hilbert)
            fig_obs = plot_observable_dynamics_hilbert(
                history,
                history_mean,
                boundary_times,
                p.obs_op,
                p.n_qubits,
            )
            figures.append(fig_obs)
            if show_plots:
                plt.show()
            if save_plots:
                plt.figure(fig_obs.number)
                plt.savefig(
                    f"{plots_dir}/observable_dynamics_hilbert_{timestamp}_rho{rho_idx}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

        if p.n_qubits == 1:
            if p.space == "liouville":
                # Plot Bloch sphere dynamics (liouville)
                fig_bloch = plot_bloch_sphere_dynamics_liouville(
                    history,
                    history_mean,
                    p.obs_op,
                    p.n_qubits,
                    show=show_plots,
                )
                figures.append(fig_bloch)
                if save_plots:
                    # Save Plotly figure as HTML
                    fig_bloch.write_html(
                        f"{plots_dir}/bloch_sphere_liouville_{timestamp}_rho{rho_idx}.html"
                    )

            elif p.space == "hilbert":
                # Plot Bloch sphere dynamics (hilbert)
                fig_bloch = plot_bloch_sphere_dynamics_hilbert(
                    history,
                    history_mean,
                    p.obs_op,
                    p.n_qubits,
                    show=show_plots,
                )
                figures.append(fig_bloch)
                if save_plots:
                    # Save Plotly figure as HTML
                    fig_bloch.write_html(
                        f"{plots_dir}/bloch_sphere_hilbert_{timestamp}_rho{rho_idx}.html"
                    )

        # Plot excitation profiles

        fig_excitation = plot_excitation_profiles(
            x,
            p,
            rho_0,
            1000,
            waveform_spec,
        )
        figures.append(fig_excitation)
        if show_plots:
            plt.show()
        if save_plots:
            plt.figure(fig_excitation.number)
            plt.savefig(
                f"{plots_dir}/excitation_profiles_{timestamp}_rho{rho_idx}.png",
                dpi=300,
                bbox_inches="tight",
            )

    plt.close("all")

    # Return the waveforms and figures
    return all_waveforms, figures


def _get_mean_H0(p):
    """Build the mean drift Hamiltonian, using the model when available.

    This replaces ``create_H_total(p)`` for Hamiltonian-aware plotting:
    when ``p.hamiltonian_model`` is set (e.g. for superconducting qubits),
    the model's ``build_drift`` is used instead of the legacy spin-chain
    functions.
    """
    model = getattr(p, "hamiltonian_model", None)
    if model is not None:
        freq_mean = [np.array(p.Delta)]
        coupling_mean = [p.Jmat] if p.n_qubits > 1 else None
        return model.build_drift(
            frequency_instances=freq_mean, coupling_instances=coupling_mean
        )[0]

    # Legacy fallback
    return create_H_total(p)


def get_H0_for_plotter(p, num_points):
    Omegas = []
    for i in range(p.n_qubits):
        Omegas.append(linear_distribution(p.Delta[i], 1.5 * p.sw[i], num_points))

    Om = [list(group) for group in zip(*Omegas)]

    # Model-aware path: use the Hamiltonian model when available
    model = getattr(p, "hamiltonian_model", None)
    if model is not None:
        Om_arrays = [np.array(om) for om in Om]
        coupling = get_Jmat_for_plotter(p, num_points) if p.n_qubits > 1 else None
        return model.build_drift(
            frequency_instances=Om_arrays, coupling_instances=coupling
        )

    # Legacy path (no hamiltonian_type specified)
    if p.n_qubits == 1:
        HCSs = []

        for om in Om:
            HCS = createHcs(om, p.op)
            HCSs.append(HCS)
        H0 = HCSs
    elif p.n_qubits > 1:
        HCSs = []
        for om in Om:
            HCS = createHcs(om, p.op)
            HCSs.append(HCS)

        HJs = []
        Jmat_instances = get_Jmat_for_plotter(p, num_points)
        for Jmat_instance in Jmat_instances:
            HJ = createHJ(Jmat_instance, p.op, coupling_type=p.coupling_type)
            HJs.append(HJ)

        H0 = [HJ + HCS for HJ, HCS in zip(HJs, HCSs)]
    return H0


def get_Jmat_for_plotter(p, num_points):
    """Coupling draws for analysis, using the same normalisation as setup.

    One random value per *physical pair*, mirrored into both triangles, so a
    symmetric nominal matrix stays symmetric and the Hamiltonian builders
    accept it.
    """
    n_qubits = p.n_qubits
    if n_qubits < 2:
        return [np.zeros((n_qubits, n_qubits)) for _ in range(num_points)]

    J = _symmetrise_coupling(p.Jmat)
    sigma = p.sigma_J if p.sigma_J is not None else 0.0
    iu = np.triu_indices(n_qubits, k=1)
    nominal = J[iu]

    Jmat_instances = []
    for _ in range(num_points):
        drawn = (
            np.where(nominal != 0, np.random.normal(nominal, sigma), 0.0)
            if sigma
            else nominal
        )
        instance = np.zeros_like(J)
        instance[iu] = drawn
        Jmat_instances.append(instance + instance.T)
    return Jmat_instances


def linear_distribution(center, band, num_points):
    # Calculate the step size
    step = band / (num_points - 1)

    # Calculate the starting point
    start = center - (band / 2)

    # Generate the points
    return [start + step * i for i in range(num_points)]


def get_final_rho_for_excitation_profile(x, p, rho_0, num_points, waveform_spec=None):
    """Final states over a frequency sweep, replaying the optimizer's physics.

    The sweep is a *nominal* trajectory: one drift per swept offset at the
    nominal Rabi amplitude, not a sampled ensemble.

    Args:
        waveform_spec: representation *x* was optimised in; defaults to the
            one recorded by the most recent run on *p*.
    """
    _amps, cxs, cys = _plotter_waveforms(x, p, waveform_spec)
    H0 = get_H0_for_plotter(p, num_points)
    final = _plotter_propagate(
        p, H0, _plotter_rabi(p, nominal=True), cxs, cys, rho_0, record=False
    )
    return [state.detach().cpu().numpy() for state in final]


def calculate_observable(state, operator, n_qubits, space_type="hilbert"):
    """
    Calculate the observable value for a given operator in Hilbert or Liouville space.

    Parameters:
    state (np.ndarray): The state of the system (wave function or density matrix).
    operator (np.ndarray): The operator representing the observable.
    space_type (str): 'liouville' for density matrix, 'hilbert' for wave function.

    Returns:
    float: The expectation value of the observable.
    """

    if space_type == "hilbert":
        # In Hilbert space, the state is a wave function |ψ⟩
        return np.real(np.vdot(state, operator @ state))

    elif space_type == "liouville":
        # In Liouville space, the state is a density matrix ρ
        return np.real(np.trace(state @ operator))

    else:
        raise ValueError("Invalid space type. Choose 'liouville' or 'hilbert'.")


def _leakage_is_visible(leakage):
    """True when any leakage trace is non-zero.

    Both the nominal trajectory and the snapshot maxima are inspected: a run
    whose sampled snapshots barely leak can still leak measurably at the
    nominal operating point, and hiding the plot would conceal that.  Only
    two identically zero traces stay hidden.
    """
    nominal = np.asarray(leakage["nominal"])
    snapshots = np.asarray(leakage["snapshots"])
    peak = 0.0
    if nominal.size:
        peak = max(peak, float(np.max(np.abs(nominal))))
    if snapshots.size:
        peak = max(peak, float(np.max(np.abs(snapshots))))
    return peak > 0.0


@plot_style()
def plot_leakage(leakage, time_vector, label_scale=1e9):
    r"""Plot total leakage ``L(t) = 1 - Tr(Pi_comp rho(t))`` over the pulse.

    Shows the nominal trajectory and the min/max envelope over the sampled
    ensemble.  ``L`` is a single total for the whole register, not a sum of
    per-qubit leakages.
    """
    t = np.asarray(time_vector) * label_scale
    nominal = np.asarray(leakage["nominal"]).real
    snapshots = np.asarray(leakage["snapshots"]).real

    fig, ax = plt.subplots()
    if snapshots.size:
        ax.fill_between(
            t,
            snapshots.min(axis=1),
            snapshots.max(axis=1),
            alpha=0.25,
            linewidth=0,
            label="snapshot min/max",
        )
    ax.plot(t, nominal, linewidth=1, label="nominal")
    ax.set_xlabel(r"$t$ (ns)")
    ax.set_ylabel(r"leakage $1-\mathrm{Tr}(\Pi_{\mathrm{comp}}\rho)$")
    ax.legend(frameon=False)
    return fig


# ======================================================================
# Shared analysis replay
#
# Analysis must replay the *same* physics the optimizer ran: the same control
# operator / amplitude mapping (including extra channels such as AC Stark),
# the same propagator, and the same dissipative channel.  These helpers call
# the optimizer's own kernels rather than reimplementing them, so the two
# cannot drift apart again.
# ======================================================================


def _plotter_waveforms(x, p, waveform_spec=None):
    """Regenerate the optimizer's modulated I/Q waveforms from a solution.

    The parameter counts, basis matrices and modes come from the solution's own
    :class:`~ctrl_freeq.make_pulse.waveform_gen_torch.WaveformSpec`, not from
    the configured basis attributes.  A piecewise solution is one value per
    pulse segment against an identity basis; splitting it with the basis
    parameter counts either raises a split-size error or, when the counts
    coincide, silently reconstructs a different waveform.

    Args:
        waveform_spec: the representation *x* was optimised in.  When omitted,
            the one recorded on *p* by the most recent run is used, which is
            only correct for that run's solution.  Pass it explicitly to
            analyse a solution from an earlier run, or whenever more than one
            optimizer shares a parameters object.

    Returns ``(amps, cxs, cys)`` torch tensors of shape
    ``(n_pulse, n_qubits)``.
    """

    x = x.detach().clone()
    spec = waveform_spec if waveform_spec is not None else p.waveform_spec()

    expected = int(sum(spec.n_para))
    if x.numel() != expected:
        raise ValueError(
            f"Solution has {x.numel()} parameters but the waveform "
            f"representation expects {expected}. The solution came from a "
            f"different optimizer than the one that produced this "
            f"representation; pass that run's waveform_spec explicitly, or "
            f"analyse each solution right after the run that produced it. "
            f"Representations with equal parameter counts cannot be told "
            f"apart here."
        )

    parameters = torch.split(x, list(spec.n_para))

    mats = []
    for per_qubit in spec.mat:
        mat_i = []
        for m in per_qubit:
            if isinstance(m, torch.Tensor):
                mat_i.append(m.detach().to(dtype=x.dtype))
            else:
                mat_i.append(torch.as_tensor(np.asarray(m), dtype=x.dtype))
        mats.append(mat_i)

    me = torch.as_tensor(np.asarray(p.modulation_exponent))
    return pulse_para(p.n_qubits, parameters, mats, spec.functions, me)


def _plotter_pulse_hamiltonian(p, cxs, cys, rabi_freq, n_h0):
    """Build Hp with the model's full control mapping (or the legacy path).

    Indexing the control operators in pairs assumes exactly two channels per
    qubit, which is wrong as soon as a model adds a channel (e.g. the Stark
    Z channel makes the operator list ``[X0, Y0, Z0, X1, Y1, Z1]``, so qubit 1
    would be driven with ``Z0`` and ``X1``).  Going through
    ``control_amplitudes`` keeps every channel, including the quadratic Stark
    power term.
    """
    n_pulse = cxs.shape[0]
    n_rabi = rabi_freq.shape[0]
    model = getattr(p, "hamiltonian_model", None)
    if model is not None:
        u = model.control_amplitudes(cxs, cys, rabi_freq, n_h0)
        return pulse_hamiltonian_generic(u, model.control_ops_tensor())

    op = create_hamiltonian_basis_torch(p.n_qubits)
    return pulse_hamiltonian(cxs, cys, rabi_freq, op, n_pulse, n_h0, n_rabi, p.n_qubits)


def _plotter_evolution_functions(p):
    """Return ``(u_fun, state_fun, collapse_ops)`` matching the optimizer."""
    model = getattr(p, "hamiltonian_model", None)
    D = model.dim if model is not None else 2**p.n_qubits
    u_fun = exp_mat_exact if D == 2 else exp_mat_torch

    dissipation_mode = getattr(p, "dissipation_mode", "non-dissipative")
    collapse_operators = getattr(p, "collapse_operators", None)
    if dissipation_mode == "dissipative" and collapse_operators is not None:
        return (
            u_fun,
            state_lindblad,
            torch.as_tensor(np.asarray(collapse_operators), dtype=torch.complex128),
        )
    if p.space == "hilbert":
        return u_fun, state_hilbert, None
    return u_fun, state_liouville, None


def _as_state_batch(state, batch, space):
    """Broadcast one initial state across a batch of ensemble members."""
    tensor = torch.as_tensor(np.asarray(state), dtype=torch.complex128)
    if space == "hilbert":
        return tensor.reshape(1, -1).expand(batch, -1).contiguous()
    return tensor.reshape(1, *tensor.shape).expand(batch, -1, -1).contiguous()


def _plotter_propagate(p, H0_list, rabi_freq, cxs, cys, state, record=False):
    """Propagate *state* over the ensemble ``H0_list x rabi_freq``.

    Returns the final states ``(batch, ...)`` when ``record`` is false, or the
    full ``(n_pulse + 1, batch, ...)`` trajectory when it is true.
    """
    H0 = torch.as_tensor(np.asarray(H0_list), dtype=torch.complex128)
    n_h0, D, _ = H0.shape
    n_rabi = rabi_freq.shape[0]

    # Batch layout must match the optimizer's: index = h0 * n_rabi + rabi.
    H0_batch = H0.unsqueeze(1).expand(n_h0, n_rabi, D, D).reshape(n_h0 * n_rabi, D, D)
    Hp = _plotter_pulse_hamiltonian(p, cxs, cys, rabi_freq, n_h0)

    u_fun, state_fun, collapse_ops = _plotter_evolution_functions(p)

    # float64: torch.as_tensor on a Python float defaults to float32, which
    # would make the replay's time step differ from the optimizer's.
    dt = torch.as_tensor(float(p.pulse_duration) / int(p.np_pulse), dtype=torch.float64)
    initial = _as_state_batch(state, n_h0 * n_rabi, p.space)

    trajectory = simulate_trajectory(
        H0_batch, Hp, dt, initial, u_fun, state_fun, collapse_ops=collapse_ops
    )
    return trajectory if record else trajectory[-1]


def _plotter_rabi(p, nominal=False):
    """Rabi snapshots for analysis: the full ensemble, or the nominal value."""
    if nominal:
        return torch.as_tensor(np.asarray(p.Omega_R_max, dtype=float)).reshape(
            1, p.n_qubits
        )
    return torch.as_tensor(np.asarray(p.Omega_R, dtype=float)).reshape(-1, p.n_qubits)


def _leakage_series(p, trajectory):
    """Total leakage ``1 - Tr(Pi_comp rho)`` at every recorded boundary.

    One total per register: per-qubit leakages must not be summed, which would
    count a state with two leaked qubits twice.  Returns ``(n_times, batch)``.
    """
    model = getattr(p, "hamiltonian_model", None)
    if model is None or model.dim == 2**p.n_qubits:
        return np.zeros(trajectory.shape[:2])

    V = torch.as_tensor(model.computational_projector(), dtype=trajectory.dtype)
    if p.space == "hilbert":
        comp = torch.einsum("tbi,ia->tba", trajectory, V.conj())
        pop = (comp.conj() * comp).real.sum(-1)
    else:
        pop = torch.einsum("ai,tbij,ja->tb", V.conj().T, trajectory, V).real
    return (1.0 - pop).detach().cpu().numpy()


def compute_and_store_evolution(x, p, rho_0, waveform_spec=None):
    """Replay the optimised pulse and record the full state trajectory.

    The replay uses the optimizer's own control mapping, propagator and
    dissipative channel, so a dissipative run decays in the plots exactly as
    it did in the objective instead of evolving unitarily.

    Two provenances are returned:

    * ``history_mean`` — the *nominal* trajectory: mean drift, nominal Rabi
      amplitude.  This is one trajectory, not an average over the ensemble.
    * ``history`` — the *sampled* ensemble: every drift snapshot crossed with
      every Rabi snapshot, i.e. the same batch the objective was evaluated on.

    Both carry ``n_pulse + 1`` entries, one per state boundary ``k * dt``
    starting with the initial state; plot them against
    ``p.state_boundary_times()``.


    Args:
        waveform_spec: representation *x* was optimised in; defaults to the
            one recorded by the most recent run on *p*.

    Returns:
        ``(cxs, cys, history, history_mean, leakage)`` where ``leakage`` is a
        dict with the nominal and per-snapshot total-leakage series.
    """
    _amps, cxs_t, cys_t = _plotter_waveforms(x, p, waveform_spec)
    cxs = [cxs_t[:, i].detach().cpu().numpy() for i in range(p.n_qubits)]
    cys = [cys_t[:, i].detach().cpu().numpy() for i in range(p.n_qubits)]

    # Sampled ensemble: the distinct drift snapshots (p.H0 repeats them once
    # per objective row) crossed with the Rabi snapshots.
    n_snapshots = p.n_drift_snapshots()
    H0_ensemble = np.asarray(p.H0)[:n_snapshots]
    traj = _plotter_propagate(
        p, H0_ensemble, _plotter_rabi(p), cxs_t, cys_t, rho_0, record=True
    )

    # Nominal trajectory: mean drift, nominal Rabi amplitude.
    traj_mean = _plotter_propagate(
        p,
        [_get_mean_H0(p)],
        _plotter_rabi(p, nominal=True),
        cxs_t,
        cys_t,
        rho_0,
        record=True,
    )

    leakage = {
        "nominal": _leakage_series(p, traj_mean)[:, 0],
        "snapshots": _leakage_series(p, traj),
    }

    traj_np = traj.detach().cpu().numpy()
    traj_mean_np = traj_mean.detach().cpu().numpy()[:, 0]

    if p.space == "hilbert":
        # (T, instances, D) -> (D, T, instances);  (T, D) -> (D, T)
        history = np.transpose(traj_np, (2, 0, 1))
        history_mean = np.transpose(traj_mean_np, (1, 0))
    else:
        # (T, instances, D, D) -> (D, D, T, instances)
        history = np.transpose(traj_np, (2, 3, 0, 1))
        history_mean = np.transpose(traj_mean_np, (1, 2, 0))

    return cxs, cys, history, history_mean, leakage
