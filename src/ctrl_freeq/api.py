"""
API - A simple interface for running quantum control optimizations.

This module provides a high-level API for loading configuration files and running
CtrlFreeQ optimizations with minimal setup required.
"""

import json
from pathlib import Path
from typing import Union, Dict, Any

from ctrl_freeq.setup.initialise_gui import Initialise
from ctrl_freeq.run.run_ctrl import run_ctrl


def _resolve_config_path(config: Union[str, Path]) -> Path:
    """Resolve a configuration file path robustly.

    Supports several aliases used across notebooks/tests:
    - Absolute or relative paths that already exist are returned as-is
    - Strings starting with "src/utils/json_input/…" are mapped to
      the packaged data/json_input folder
    - Default packaged location under src/ctrl_freeq/data/json_input

    Raises FileNotFoundError if nothing resolves.
    """
    p = Path(config)
    if p.exists():
        return p

    # Map "src/utils/json_input/..." (legacy path) to packaged data/json_input
    try_paths = []
    if isinstance(config, str) and config.startswith("src/utils/json_input/"):
        try_paths.append(
            Path(__file__).parent / "data" / "json_input" / Path(config).name
        )

    # Packaged default location: src/ctrl_freeq/data/json_input
    if isinstance(config, (str, Path)):
        try_paths.append(
            Path(__file__).parent / "data" / "json_input" / Path(config).name
        )

    for tp in try_paths:
        if tp.exists():
            return tp

    raise FileNotFoundError(f"Config file not found for: {config}")


class CtrlFreeQAPI:
    """
    High-level API for CtrlFreeQ quantum control optimization.

    This class provides a simple interface to load configuration files
    and run CtrlFreeQ optimizations.
    """

    def __init__(
        self, config: Union[str, Path, Dict[str, Any]], hamiltonian_model=None
    ):
        """
        Initialize the CtrlFreeQ API with a configuration.

        Args:
            config: Either a path to a JSON configuration file, or a dictionary
                   containing the configuration parameters.
            hamiltonian_model: Optional pre-built HamiltonianModel instance.
                   If provided, overrides the ``hamiltonian_type`` in the config.
                   Useful for custom/user-defined models that are not registered
                   in the model registry.
        """
        if isinstance(config, (str, Path)):
            self.config_path = _resolve_config_path(config)
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        elif isinstance(config, dict):
            self.config = config
            self.config_path = None
        else:
            raise ValueError("Config must be a file path or dictionary")

        # Preprocess the configuration for compatibility with Initialise class
        self.processed_config = self._preprocess_config(self.config)

        # Initialize the parameter object
        self.parameters = Initialise(self.processed_config)

        # Override model if user provided one directly
        if hamiltonian_model is not None:
            self.parameters.hamiltonian_model = hamiltonian_model
            self.parameters.H0 = self.parameters.get_H0()

    def _preprocess_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the configuration to ensure compatibility with Initialise class.

        The Initialise class expects numpy arrays for numeric parameters.
        This method converts lists to numpy arrays like the GUI does.

        Args:
            config: Raw configuration dictionary

        Returns:
            Processed configuration dictionary
        """
        import copy
        import numpy as np

        processed = copy.deepcopy(config)

        # Convert numeric lists to numpy arrays using the same logic as GUI
        def convert_to_array(data):
            """Convert numeric lists to numpy arrays."""

            def is_numeric(val):
                return isinstance(val, (int, float, complex))

            if isinstance(data, list):
                # Check for matrix (list of lists)
                if all(isinstance(i, list) for i in data):
                    if all(is_numeric(j) for i in data for j in i):
                        return np.array(data)
                    else:
                        return data
                # Check for simple list
                elif all(is_numeric(i) for i in data):
                    return np.array(data)
                else:
                    return data
            else:
                return data

        # Convert parameters to arrays
        processed["parameters"] = {
            key: convert_to_array(value)
            for key, value in processed["parameters"].items()
        }

        # Convert target states to arrays
        processed["target_states"] = {
            key: convert_to_array(value)
            for key, value in processed["target_states"].items()
        }

        # Defaults for runtime options
        if "compute_resource" not in processed:
            processed["compute_resource"] = "cpu"
        # cpu_cores is optional and passed-through if present

        return processed

    def run_optimization(self) -> Any:
        """
        Run the CtrlFreeQ optimization with the loaded configuration.

        Returns:
            The optimization solution from CtrlFreeQ.
        """
        return run_ctrl(self.parameters)

    def get_config_summary(self) -> str:
        """
        Get a summary of the current configuration.

        Returns:
            A string summary of the configuration parameters.
        """
        summary = []
        summary.append(f"Number of qubits: {len(self.config['qubits'])}")
        summary.append(f"Optimization space: {self.config['optimization']['space']}")
        summary.append(f"Algorithm: {self.config['optimization']['algorithm']}")
        summary.append(f"Max iterations: {self.config['optimization']['max_iter']}")
        summary.append(f"Target fidelity: {self.config['optimization']['targ_fid']}")

        # Dissipation mode
        dissipation_mode = self.config["optimization"].get(
            "dissipation_mode", "non-dissipative"
        )
        summary.append(f"Dissipation: {dissipation_mode}")
        if dissipation_mode == "dissipative":
            T1 = self.config["parameters"].get("T1", [])
            T2 = self.config["parameters"].get("T2", [])
            for i, (t1, t2) in enumerate(zip(T1, T2)):
                summary.append(f"  Qubit {i + 1}: T1={t1:.2e} s, T2={t2:.2e} s")

        # Initial states
        init_states = self.config["initial_states"]
        summary.append(f"Initial states: {init_states}")

        # Target states
        target_states = self.config["target_states"]
        for key, value in target_states.items():
            if value and any(
                v is not None for v in (value if isinstance(value, list) else [value])
            ):
                summary.append(f"Target {key}: {value}")

        return "\n".join(summary)

    def update_parameter(self, parameter_path: str, value: Any):
        """
        Update a specific parameter in the configuration.

        Args:
            parameter_path: Dot-separated path to the parameter (e.g., "optimization.max_iter")
            value: New value for the parameter
        """
        keys = parameter_path.split(".")
        config_ref = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_ref:
                raise KeyError(f"Parameter path '{parameter_path}' not found")
            config_ref = config_ref[key]

        # Set the final value
        final_key = keys[-1]
        if final_key not in config_ref:
            raise KeyError(f"Parameter path '{parameter_path}' not found")

        config_ref[final_key] = value

        # Reprocess the configuration and reinitialize parameters
        self.processed_config = self._preprocess_config(self.config)
        self.parameters = Initialise(self.processed_config)


def load_config(config_path: Union[str, Path]) -> CtrlFreeQAPI:
    """
    Load a configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        A CtrlFreeQAPI instance with the loaded configuration
    """
    return CtrlFreeQAPI(config_path)


def run_from_config(
    config: Union[str, Path, Dict[str, Any]], hamiltonian_model=None
) -> Any:
    """
    Run an optimization directly from a configuration.

    Args:
        config: Either a path to a JSON configuration file, or a dictionary
               containing the configuration parameters.
        hamiltonian_model: Optional pre-built HamiltonianModel instance.

    Returns:
        The optimization solution.
    """
    api = CtrlFreeQAPI(config, hamiltonian_model=hamiltonian_model)
    return api.run_optimization()


# Convenience functions for loading the provided example configurations
def load_single_qubit_config() -> CtrlFreeQAPI:
    """Load the single qubit example configuration."""
    return CtrlFreeQAPI("single_qubit_parameters.json")


def load_single_qubit_multiple_config() -> CtrlFreeQAPI:
    """Load the single qubit multiple initial/target states example configuration."""
    return CtrlFreeQAPI("single_qubit_parameters_multiple_initial_targ.json")


def load_two_qubit_config() -> CtrlFreeQAPI:
    """Load the two qubit example configuration."""
    return CtrlFreeQAPI("two_qubit_parameters.json")


def load_two_qubit_multiple_config() -> CtrlFreeQAPI:
    """Load the two qubit multiple initial/target states example configuration."""
    return CtrlFreeQAPI("two_qubit_parameters_multiple_initial_targ.json")


def load_single_qubit_polar_phase_config() -> CtrlFreeQAPI:
    """Load the single qubit polar phase example configuration."""
    return CtrlFreeQAPI("single_qubit_parameters_polar_phase.json")


def load_two_qubit_polar_phase_config() -> CtrlFreeQAPI:
    """Load the two qubit polar phase example configuration."""
    return CtrlFreeQAPI("two_qubit_parameters_polar_phase.json")


def load_four_qubit_polar_phase_config() -> CtrlFreeQAPI:
    """Load the four qubit polar phase example configuration."""
    return CtrlFreeQAPI("four_qubit_parameters_polar_phase.json")


def load_single_qubit_dissipative_config() -> CtrlFreeQAPI:
    """Load the single qubit dissipative (Lindblad) example configuration."""
    return CtrlFreeQAPI("single_qubit_dissipative.json")
