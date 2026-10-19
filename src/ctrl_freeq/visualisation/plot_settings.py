import matplotlib.pyplot as plt
import matplotlib
from functools import wraps
import shutil
import sys


def _latex_available():
    """Check whether a LaTeX installation is available on the system."""
    return shutil.which("latex") is not None


# Base plot settings
BASE_PLOT_SETTINGS = {
    "text.usetex": _latex_available(),  # Use LaTeX for text rendering if available
    "font.family": "serif",  # Use serif fonts
    "font.serif": ["Times"],  # Use Times font
    "axes.labelsize": 10,  # Axis label font size
    "font.size": 10,  # Base font size
    "legend.fontsize": 9,  # Legend font size
    "xtick.labelsize": 9,  # X-tick labels font size
    "ytick.labelsize": 9,  # Y-tick labels font size
    "figure.dpi": 300,  # Dots per inch for raster formats
    "savefig.dpi": 300,  # Dots per inch for saving figures
    "pdf.fonttype": 42,  # Embed fonts in PDF
    "ps.fonttype": 42,  # Embed fonts in EPS
    "savefig.format": "pdf",  # Default format to save figures
    # Grid settings
    "axes.grid": True,  # Enable grid
    "grid.color": "lightgray",  # Grid line color
    "grid.linestyle": "--",  # Grid line style
    "grid.linewidth": 0.5,  # Grid line width
    "grid.alpha": 0.7,  # Grid transparency
    "axes.edgecolor": "black",  # Axes edge color (optional)
    "axes.facecolor": "white",  # Axes face color (optional)
    "xtick.major.size": 5,  # Major tick size
    "ytick.major.size": 5,  # Major tick size
    "xtick.minor.size": 3,  # Minor tick size
    "ytick.minor.size": 3,  # Minor tick size
}


def plot_style(save_path=None, show=True):
    """
    Decorator to apply plotting styles, set figure size from additional_settings if provided,
    and optionally save or show the plot.

    Parameters:
    - save_path: File path to save the plot. If None, the plot is not saved.
    - show: Boolean indicating whether to display the plot.

    Returns:
    - The figure object created by the decorated function.
    """

    def decorator(plot_func):
        @wraps(plot_func)
        def wrapper(*args, **kwargs):
            # Extract additional_settings from kwargs, default to empty dict
            additional_settings = kwargs.pop("additional_settings", {})

            # Extract figsize from additional_settings if present
            figsize = additional_settings.pop("figsize", (3.35, 3.35))

            # Combine base settings with any additional settings
            settings = BASE_PLOT_SETTINGS.copy()
            if additional_settings:
                settings.update(additional_settings)

            # Inject 'figure.figsize' into settings
            settings["figure.figsize"] = figsize

            with plt.rc_context(settings):
                # Execute the plotting function and capture the return value
                fig = plot_func(*args, **kwargs)

                plt.tight_layout()

                if save_path:
                    plt.savefig(save_path, bbox_inches="tight")

                if show:
                    # Check if the current backend is interactive before showing
                    if matplotlib.is_interactive():
                        plt.show()

                # Don't close the figure here, as we need to return it
                # plt.close()

                return fig

        return wrapper

    return decorator


def _is_running_from_gui():
    """
    Detect if the code is running from the CtrlFreeQ GUI.

    Returns:
        bool: True if running from GUI, False otherwise
    """
    # Simple approach: check if we're in a non-TTY environment
    # This will catch GUI applications that redirect stdout
    if not sys.stdout.isatty():
        return True

    # Check if any of the GUI modules are in the call stack
    import inspect

    # Get the current call stack
    stack = inspect.stack()

    # Look for GUI-related modules in the call stack
    gui_indicators = [
        "gui_setup.py",
        "initialise_gui.py",
        "tkinter",
        "run_optimisation_and_plot",
    ]

    for frame in stack:
        filename = frame.filename
        function_name = frame.function

        # Check if any GUI indicators are in the filename or function name
        for indicator in gui_indicators:
            if indicator in filename or indicator in function_name:
                return True

    # Check if tkinter is available and imported
    try:
        import tkinter

        # Check if tkinter root window exists
        if hasattr(tkinter, "_default_root") and tkinter._default_root is not None:
            return True
    except ImportError:
        pass

    return False
