import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from ctrl_freeq.evolution.time_evolution import (
    apply_multi_pulse_multi_qubits_hilbert,
    apply_multi_pulse_multi_qubits_liouville,
    apply_multi_pulse_multi_qubits_lindblad,
)
from ctrl_freeq.setup.hamiltonian_generation.hamiltonians import (
    create_H_total,
    createHcs,
    createHJ,
)
from ctrl_freeq.make_pulse.waveform_gen_torch import (
    waveform_gen_cart,
    waveform_gen_polar,
    waveform_gen_polar_phase,
)
import torch

import plotly.graph_objects as go

from ctrl_freeq.visualisation.plot_settings import plot_style, plotly_style


# from tikzplotlib import save as tikz_save


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
    t = np.linspace(np.finfo(float).eps, T, len(cxs[0]))
    t = t * 1e9  # Convert to ns

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
    t = np.linspace(np.finfo(float).eps, T, len(cxs[0]))
    t = t * 1e9  # Convert to ns

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
def plot_excitation_profiles(x, p, rho_0, num_points):
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
    rho_end = get_final_rho_for_excitation_profile(x, p, rho_0, num_points)

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


def process_and_plot(x, p, save_plots=False, show_plots=False):
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
    cxs, cys, _, _ = compute_and_store_evolution(x, p, first_rho_0)

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
        _, _, history, history_mean = compute_and_store_evolution(x, p, rho_0)

        # Store the waveforms for this initial state
        all_waveforms["waveforms"].append({"rho_idx": rho_idx, "cxs": cxs, "cys": cys})

        if p.space == "liouville":
            # Plot history with mean (liouville)
            fig_history = plot_history_with_mean_lu_liouville(
                history=history,
                history_mean=history_mean,
                time_vector=p.t,
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
                p.t,
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
                time_vector=p.t,
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
                p.t,
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
    Jmat_instances = []
    sigma = p.sigma_J if p.sigma_J is not None else 0

    for _ in range(num_points):
        Jmat_instance = np.where(
            p.Jmat != 0,
            np.random.normal(p.Jmat, sigma),
            0.0,
        )
        Jmat_instances.append(Jmat_instance)
    return Jmat_instances


def linear_distribution(center, band, num_points):
    # Calculate the step size
    step = band / (num_points - 1)

    # Calculate the starting point
    start = center - (band / 2)

    # Generate the points
    return [start + step * i for i in range(num_points)]


def get_final_rho_for_excitation_profile(x, p, rho_0, num_points):
    duration = p.pulse_duration
    wf_mode = p.wf_mode
    op = p.op
    mat = p.mat
    n_qubits = p.n_qubits
    n_para = p.n_para_updated
    peak_amplitudes = p.Omega_R_max
    space = p.space

    # Sample list of parameters
    parameters = torch.split(x.detach().clone(), list(n_para))

    # Lists to store the results
    amps = []
    phis = []
    cxs = []
    cys = []
    pulse_params = []

    wf_fun = []

    for mode in wf_mode:
        if mode == "polar_phase":
            wf_fun.append(waveform_gen_polar_phase)
        elif mode == "polar":
            wf_fun.append(waveform_gen_polar)
        elif mode == "cart":
            wf_fun.append(waveform_gen_cart)

    # Loop through each parameter and call waveform_gen
    for i in range(n_qubits):
        # Ensure mats have same dtype as parameters x
        mat_i = []
        for m in mat[i]:
            if isinstance(m, np.ndarray):
                mat_i.append(torch.tensor(m, dtype=x.dtype))
            else:
                mat_i.append(m.to(dtype=x.dtype))
        amp, phi, cx, cy = wf_fun[i](parameters[i], mat_i)
        # Convert PyTorch tensors to NumPy arrays
        amp_np = amp.detach().cpu().numpy()
        phi_np = phi.detach().cpu().numpy()
        cx_np = cx.detach().cpu().numpy()
        cy_np = cy.detach().cpu().numpy()
        c = (cx_np + 1j * cy_np).reshape(-1)  # Reshape to (100,)
        modulation_exponent = p.modulation_exponent[:, i].reshape(
            -1
        )  # Reshape to (100,)
        modulated_waveform = np.multiply(c, modulation_exponent)
        cx = modulated_waveform.real
        cy = modulated_waveform.imag
        pulse_params.append((cx, cy, op[f"X_{i + 1}"], op[f"Y_{i + 1}"]))
        amps.append(amp_np)
        phis.append(phi_np)
        cxs.append(cx)
        cys.append(cy)
    H0 = get_H0_for_plotter(p, num_points)

    dissipation_mode = getattr(p, "dissipation_mode", "non-dissipative")
    collapse_operators = getattr(p, "collapse_operators", None)

    rho_end = []
    for h0 in H0:
        if dissipation_mode == "dissipative" and collapse_operators is not None:
            rho_end.append(
                apply_multi_pulse_multi_qubits_lindblad(
                    h0,
                    pulse_params,
                    duration,
                    peak_amplitudes,
                    rho_0,
                    collapse_operators,
                )
            )
        elif space == "hilbert":
            rho_end.append(
                apply_multi_pulse_multi_qubits_hilbert(
                    h0, pulse_params, duration, peak_amplitudes, rho_0
                )
            )
        elif space == "liouville":
            rho_end.append(
                apply_multi_pulse_multi_qubits_liouville(
                    h0, pulse_params, duration, peak_amplitudes, rho_0
                )
            )
        else:
            raise ValueError("Invalid space type. Choose 'liouville' or 'hilbert'.")
    return rho_end


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


def compute_and_store_evolution(x, p, rho_0):
    """
    Visualize the time evolution of each element of the density matrix during pulse application in a compact manner.

    Args:
    - H_0 (np.array): Initial Hamiltonian.
    - rho_0 (np.array): Initial density matrix.
    - pulse_params (list of tuples): Each tuple contains (f, g, Ix, Iy) for a specific spin channel.
    - dt (float): time step.
    """

    H_0 = p.H0
    t = p.t
    wf_mode = p.wf_mode
    op = p.op
    mat = p.mat
    n_qubits = p.n_qubits
    n_para = p.n_para_updated
    peak_amplitudes = p.Omega_R_max
    H0_mean = _get_mean_H0(p)

    dt = t[1] - t[0]

    wf_fun = []

    for mode in wf_mode:
        if mode == "polar_phase":
            wf_fun.append(waveform_gen_polar_phase)
        elif mode == "polar":
            wf_fun.append(waveform_gen_polar)
        elif mode == "cart":
            wf_fun.append(waveform_gen_cart)

    # Sample list of parameters
    parameters = torch.split(x.detach().clone(), list(n_para))

    # Lists to store the results
    amps = []
    phis = []
    cxs = []
    cys = []
    pulse_params = []

    # Loop through each parameter and call waveform_gen
    for i in range(n_qubits):
        # Ensure the waveform‐gen mats have the same dtype as x
        mat_i = []
        for m in mat[i]:
            if isinstance(m, np.ndarray):
                mat_i.append(torch.tensor(m, dtype=x.dtype))
            else:
                mat_i.append(m.to(dtype=x.dtype))
        amp, phi, cx, cy = wf_fun[i](parameters[i], mat_i)
        # Convert PyTorch tensors to NumPy arrays
        amp_np = amp.detach().cpu().numpy()
        phi_np = phi.detach().cpu().numpy()
        cx_np = cx.detach().cpu().numpy()
        cy_np = cy.detach().cpu().numpy()
        c = (cx_np + 1j * cy_np).reshape(-1)  # Reshape to (100,)
        modulation_exponent = p.modulation_exponent[:, i].reshape(
            -1
        )  # Reshape to (100,)
        modulated_waveform = np.multiply(c, modulation_exponent)
        cx = modulated_waveform.real
        cy = modulated_waveform.imag
        pulse_params.append((cx, cy, op[f"X_{i + 1}"], op[f"Y_{i + 1}"]))
        amps.append(amp_np)
        phis.append(phi_np)
        cxs.append(cx)
        cys.append(cy)

    n = 2**n_qubits
    instances = len(H_0)
    time_steps = len(pulse_params[0][0])
    if p.space == "hilbert":
        history = np.zeros((n, time_steps, instances), dtype=complex)
        history_mean = np.zeros((n, time_steps), dtype=complex)
    elif p.space == "liouville":
        history = np.zeros((n, n, time_steps, instances), dtype=complex)
        history_mean = np.zeros((n, n, time_steps), dtype=complex)

    rho_t_mean = rho_0
    H_t_mean = H0_mean.copy()
    for i in range(len(pulse_params[0][0])):  # Assumes all f have the same length
        H1_t = np.zeros_like(H0_mean)  # To store the added term for this iteration

        for (f, g, Ix, Iy), amplitude in zip(pulse_params, peak_amplitudes):
            H1_t += amplitude * (f[i] * Ix + g[i] * Iy)

        H_t_curr = H_t_mean + H1_t

        U_t = expm(-1j * H_t_curr * dt)  # Time evolution operator for this slice

        if p.space == "hilbert":
            rho_t_mean = U_t @ rho_t_mean
            history_mean[:, i] = rho_t_mean
        elif p.space == "liouville":
            rho_t_mean = U_t @ rho_t_mean @ U_t.conj().T
            history_mean[:, :, i] = rho_t_mean

    H_t = H_0.copy()
    for j in range(len(H_0)):
        rho_t = rho_0
        for i in range(len(pulse_params[0][0])):  # Assumes all f have the same length
            H1_t = np.zeros_like(H0_mean)  # To store the added term for this iteration

            for (f, g, Ix, Iy), amplitude in zip(pulse_params, peak_amplitudes):
                H1_t += amplitude * (f[i] * Ix + g[i] * Iy)

            H_t_curr = H_t[j] + H1_t
            U_t = expm(-1j * H_t_curr * dt)  # Time evolution operator for this slice

            if p.space == "hilbert":
                rho_t = U_t @ rho_t
                history[:, i, j] = rho_t
            elif p.space == "liouville":
                rho_t = U_t @ rho_t @ U_t.conj().T
                history[:, :, i, j] = rho_t

    return cxs, cys, history, history_mean
