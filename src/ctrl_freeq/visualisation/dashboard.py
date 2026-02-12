"""
Dashboard generation using Panel for combining all analysis figures.

This module creates an interactive HTML dashboard that displays all generated
figures (both Matplotlib and Plotly) in a presentable two-column card grid.
"""

import panel as pn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Any


def create_dashboard(figures: List[Any], timestamp: str, parameters=None) -> str:
    """
    Create an interactive dashboard from a list of figures.

    Args:
        figures: List of matplotlib.figure.Figure and/or plotly.graph_objects.Figure
        timestamp: Timestamp string for the dashboard filename
        parameters: Optional Initialise object containing simulation parameters

    Returns:
        str: Path to the saved dashboard HTML file
    """
    # Initialize Panel with Plotly extension and dark theme
    pn.extension("plotly", notifications=True, sizing_mode="stretch_width")

    def get_figure_title(index: int, fig: Any, total_figures: int) -> str:
        """
        Determine the descriptive title for a figure based on its position and type.

        Figure order from process_and_plot():
        1. Pulse Waveforms - IQ Plot
        2. Pulse Waveforms - Amplitude/Phase Plot
        3+. State Evolution - History
        4+. Observable Dynamics
        5+. Bloch Sphere (if single qubit, Plotly)
        6+. Excitation Profiles
        """
        # First two figures are always waveforms
        if index == 0:
            return "Pulse Waveforms: IQ Components"
        elif index == 1:
            return "Pulse Waveforms: Amplitude & Phase"

        # Remaining figures depend on initial states
        # Pattern repeats: History, Observable, [Bloch Sphere if single qubit], Excitation
        remaining_idx = index - 2

        # Detect Bloch sphere (it's the only Plotly figure after waveforms)
        is_plotly = hasattr(fig, "to_plotly_json")

        if is_plotly:
            return "Bloch Sphere"

        # For matplotlib figures after the first two, determine type by position pattern
        # The pattern is: History -> Observable -> Excitation (or History -> Observable -> Bloch -> Excitation)
        # We can use modulo to determine position in the pattern

        # Check if we have Bloch spheres by looking for Plotly figures
        has_bloch = any(
            hasattr(figures[i], "to_plotly_json") for i in range(2, len(figures))
        )

        if has_bloch:
            # Pattern: History, Observable, Bloch, Excitation (length 4)
            pattern_length = 4
            pos_in_pattern = remaining_idx % pattern_length

            if pos_in_pattern == 0:
                return "State Time Evolution"
            elif pos_in_pattern == 1:
                return "Observable Dynamics"
            # pos 2 is Bloch (Plotly, handled above)
            else:  # pos_in_pattern == 3
                return "Excitation Profiles (Frequency Response)"
        else:
            # Pattern: History, Observable, Excitation (length 3)
            pattern_length = 3
            pos_in_pattern = remaining_idx % pattern_length

            if pos_in_pattern == 0:
                return "State Time Evolution"
            elif pos_in_pattern == 1:
                return "Observable Dynamics"
            else:  # pos_in_pattern == 2
                return "Excitation Profiles (Frequency Response)"

    # Create cards for each figure
    cards = []
    for i, fig in enumerate(figures):
        # Determine figure type and create appropriate pane
        if hasattr(fig, "to_plotly_json"):  # Plotly figure
            pane = pn.pane.Plotly(
                fig,
                config={"displaylogo": False, "responsive": True},
                sizing_mode="stretch_width",
            )
        else:  # Matplotlib figure
            pane = pn.pane.Matplotlib(fig, tight=True, sizing_mode="stretch_width")

        # Get descriptive title for this figure
        title = get_figure_title(i, fig, len(figures))

        # Create a card with the figure
        card = pn.Card(
            pane,
            title=title,
            collapsed=False,
            header_background="#1f77b4",
            styles={"borderRadius": "8px", "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"},
        )
        cards.append(card)

    # Create a two-column grid layout
    grid = pn.GridBox(
        *cards, ncols=2, sizing_mode="stretch_width", styles={"gap": "16px"}
    )

    # Create sidebar with information
    if parameters is not None:
        # Build general parameter list
        param_lines = [
            f"**Coverage:** {parameters.coverage}",
            f"**Number of Qubits:** {parameters.n_qubits}",
            f"**Pulse Duration:** {parameters.pulse_duration}",
            f"**Points in Pulse:** {parameters.np_pulse}",
            f"**Waveform Type:** {parameters.wf_type}",
            f"**Waveform Mode:** {parameters.wf_mode}",
            f"**Number of Parameters:** {parameters.n_para}",
        ]

        # Add coupling type if multi-qubit
        if parameters.n_qubits > 1 and hasattr(parameters, "coupling_type"):
            param_lines.append(f"**Coupling Type:** {parameters.coupling_type}")

        # Add frequency/amplitude parameters (convert from rad/s to Hz)
        # Handle array parameters (per-qubit) by taking first element
        sw_val = (
            parameters.sw[0]
            if hasattr(parameters.sw, "__len__") and not isinstance(parameters.sw, str)
            else parameters.sw
        )
        omega_val = (
            parameters.Delta[0]
            if hasattr(parameters.Delta, "__len__")
            and not isinstance(parameters.Delta, str)
            else parameters.Delta
        )
        sigma_omega_val = (
            parameters.sigma_Delta[0]
            if hasattr(parameters.sigma_Delta, "__len__")
            and not isinstance(parameters.sigma_Delta, str)
            else parameters.sigma_Delta
        )

        param_lines.extend(
            [
                f"**SW (Hz):** {float(sw_val / (2 * np.pi)):.2f}",
                f"**Δ (Hz):** {float(omega_val / (2 * np.pi)):.2f}",
                f"**σ Δ (Hz):** {float(sigma_omega_val / (2 * np.pi)):.2f}",
            ]
        )

        # Add J coupling if multi-qubit
        if (
            parameters.n_qubits > 1
            and hasattr(parameters, "sigma_J")
            and parameters.sigma_J is not None
        ):
            sigma_j_val = (
                parameters.sigma_J
                if np.isscalar(parameters.sigma_J)
                else parameters.sigma_J
            )
            param_lines.append(f"**σ J (Hz):** {float(sigma_j_val / (2 * np.pi)):.2f}")

        # Handle array parameters for Omega_R_max and sigma_Omega_R_max
        omega_1_max_val = (
            parameters.Omega_R_max[0]
            if hasattr(parameters.Omega_R_max, "__len__")
            and not isinstance(parameters.Omega_R_max, str)
            else parameters.Omega_R_max
        )
        sigma_omega_1_max_val = (
            parameters.sigma_Omega_R_max[0]
            if hasattr(parameters.sigma_Omega_R_max, "__len__")
            and not isinstance(parameters.sigma_Omega_R_max, str)
            else parameters.sigma_Omega_R_max
        )

        param_lines.extend(
            [
                f"**Ω_R max (Hz):** {float(omega_1_max_val / (2 * np.pi)):.2f}",
                f"**σ Ω_R max (Hz):** {float(sigma_omega_1_max_val / (2 * np.pi)):.2f}",
            ]
        )

        # Add initial and target state information if available
        if hasattr(parameters, "init_ax"):
            param_lines.append(f"**Initial State:** {parameters.init_ax}")
        if hasattr(parameters, "targ_ax"):
            param_lines.append(f"**Target State:** {parameters.targ_ax}")
        elif hasattr(parameters, "gate"):
            param_lines.append(f"**Target State:** {parameters.gate}")
        elif hasattr(parameters, "axis") and hasattr(parameters, "beta"):
            param_lines.append(
                f"**Target State:** Phi={parameters.axis}, Beta={parameters.beta}"
            )

        params_markdown = "\n\n".join(param_lines)

        # Build optimization parameter list
        opt_param_lines = [
            f"**Algorithm:** {parameters.algorithm}",
            f"**Max Iterations:** {parameters.max_iter}",
            f"**Target Fidelity:** {parameters.targ_fid}",
            f"**H0 Snapshots:** {parameters.H0_snapshots}",
            f"**Ω_R Snapshots:** {parameters.Omega_R_snapshots}",
        ]

        opt_params_markdown = "\n\n".join(opt_param_lines)

        sidebar_content = pn.pane.Markdown(f"""
### Dashboard Information

**Generated:** {timestamp}

**Total Figures:** {len(figures)}

---

### Parameters

{params_markdown}

---

### Optimization Parameters

{opt_params_markdown}
""")
    else:
        # Fallback to basic information if no parameters provided
        sidebar_content = pn.pane.Markdown(f"""
### Dashboard Information

**Generated:** {timestamp}

**Total Figures:** {len(figures)}
""")

    # Create the main template with dark theme
    template = pn.template.FastListTemplate(
        title="ctrl-freeq Analysis Dashboard",
        theme="dark",
        sidebar=[sidebar_content],
        main=[grid],
        header_background="#2c3e50",
        accent_base_color="#3498db",
        header_color="#ecf0f1",
    )

    # Save the dashboard to an HTML file
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent.parent
    results_dir = project_root / "results" / "dashboards"
    results_dir.mkdir(parents=True, exist_ok=True)

    dashboard_path = str(results_dir / f"dashboard_{timestamp}.html")
    # Use CDN resources to avoid embedding ~5-10 MB of JS/CSS inline.
    # Dashboards require an internet connection to load Panel/Bokeh/Plotly
    # libraries, but the file size drops from ~8-50 MB to ~1-2 MB.
    template.save(dashboard_path, resources="cdn")

    # Close matplotlib figures to free memory now that they're saved to disk
    for fig in figures:
        if not hasattr(fig, "to_plotly_json"):  # matplotlib figure
            plt.close(fig)

    return dashboard_path


def create_demo_dashboard():
    """
    Create a demo dashboard with sample plots for testing.
    This is useful for development and testing purposes.
    """
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go
    import numpy as np
    from datetime import datetime

    # Create some sample Matplotlib figures
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x), label="sin(x)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Sample Matplotlib Plot")
    ax1.legend()
    ax1.grid(True)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(x, np.cos(x), label="cos(x)", color="orange")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Another Matplotlib Plot")
    ax2.legend()
    ax2.grid(True)

    # Create a sample Plotly figure
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=x, y=np.tan(x), mode="lines", name="tan(x)"))
    fig3.update_layout(
        title="Sample Plotly Plot",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_dark",
    )

    # Create dashboard
    figures = [fig1, fig2, fig3]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dashboard_path = create_dashboard(figures, timestamp)
    print(f"Demo dashboard created: {dashboard_path}")

    return dashboard_path


if __name__ == "__main__":
    # Create a demo dashboard when run directly
    create_demo_dashboard()
