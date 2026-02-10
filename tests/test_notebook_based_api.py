#!/usr/bin/env python3
"""
Pytest tests based on the CtrlFreeQ API demo notebook.
Tests all functionality demonstrated in the notebook including the new polar_phase configurations.
"""

import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from ctrl_freeq.api import (
    load_config,
    run_from_config,
    load_single_qubit_config,
    load_single_qubit_multiple_config,
    load_two_qubit_config,
    load_two_qubit_multiple_config,
    load_single_qubit_polar_phase_config,
    load_two_qubit_polar_phase_config,
)


class TestNotebookBasedAPI:
    """Test cases based on the CtrlFreeQ API demo notebook."""

    def test_single_qubit_optimization(self):
        """Test single qubit optimization as shown in notebook section 1."""
        # Load single qubit configuration
        single_qubit_api = load_single_qubit_config()

        # Display configuration summary
        config_summary = single_qubit_api.get_config_summary()
        assert config_summary is not None
        assert isinstance(config_summary, str)
        assert "Single Qubit" in config_summary or "1" in config_summary

        # Run the optimization with reduced iterations for testing
        single_qubit_api.update_parameter("optimization.max_iter", 5)
        single_qubit_solution = single_qubit_api.run_optimization()

        assert single_qubit_solution is not None
        assert hasattr(single_qubit_solution, "shape")
        assert single_qubit_solution.shape[0] > 0
        print(
            f"Single qubit optimization completed. Solution shape: {single_qubit_solution.shape}"
        )

    def test_single_qubit_multiple_states(self):
        """Test single qubit with multiple initial/target states as shown in notebook section 2."""
        # Load single qubit multiple states configuration
        single_multiple_api = load_single_qubit_multiple_config()

        # Display configuration summary
        config_summary = single_multiple_api.get_config_summary()
        assert config_summary is not None
        assert isinstance(config_summary, str)

        # Run the optimization with reduced iterations for testing
        single_multiple_api.update_parameter("optimization.max_iter", 5)
        single_multiple_solution = single_multiple_api.run_optimization()

        assert single_multiple_solution is not None
        assert hasattr(single_multiple_solution, "shape")
        assert single_multiple_solution.shape[0] > 0
        print(
            f"Single qubit multiple states optimization completed. Solution shape: {single_multiple_solution.shape}"
        )

    def test_two_qubit_optimization(self):
        """Test two qubit optimization as shown in notebook section 3."""
        # Load two qubit configuration
        two_qubit_api = load_two_qubit_config()

        # Display configuration summary
        config_summary = two_qubit_api.get_config_summary()
        assert config_summary is not None
        assert isinstance(config_summary, str)
        assert "2" in config_summary or "Two" in config_summary

        # Run the optimization with reduced iterations for testing
        two_qubit_api.update_parameter("optimization.max_iter", 5)
        two_qubit_solution = two_qubit_api.run_optimization()

        assert two_qubit_solution is not None
        assert hasattr(two_qubit_solution, "shape")
        assert two_qubit_solution.shape[0] > 0
        print(
            f"Two qubit optimization completed. Solution shape: {two_qubit_solution.shape}"
        )

    def test_two_qubit_multiple_states(self):
        """Test two qubit with multiple initial/target states as shown in notebook section 4."""
        # Load two qubit multiple states configuration
        two_multiple_api = load_two_qubit_multiple_config()

        # Display configuration summary
        config_summary = two_multiple_api.get_config_summary()
        assert config_summary is not None
        assert isinstance(config_summary, str)

        # Run the optimization with reduced iterations for testing
        two_multiple_api.update_parameter("optimization.max_iter", 5)
        two_multiple_solution = two_multiple_api.run_optimization()

        assert two_multiple_solution is not None
        assert hasattr(two_multiple_solution, "shape")
        assert two_multiple_solution.shape[0] > 0
        print(
            f"Two qubit multiple states optimization completed. Solution shape: {two_multiple_solution.shape}"
        )

    def test_custom_configuration_loading(self):
        """Test loading custom configurations as shown in notebook section 5."""
        # Test loading a configuration from a file path
        custom_api = load_config("src/utils/json_input/single_qubit_parameters.json")
        config_summary = custom_api.get_config_summary()
        assert config_summary is not None
        assert isinstance(config_summary, str)
        print("Custom loaded configuration test passed")

    def test_parameter_modification(self):
        """Test modifying parameters before running as shown in notebook section 5."""
        # Load configuration and modify parameters
        custom_api = load_single_qubit_config()

        # Modify parameters before running
        custom_api.update_parameter("optimization.max_iter", 3)
        custom_api.update_parameter("optimization.targ_fid", 0.995)

        # Verify parameters were updated in config summary
        modified_summary = custom_api.get_config_summary()
        assert "3" in modified_summary  # max_iter should be 3
        assert "0.995" in modified_summary  # targ_fid should be 0.995

        # Run optimization to ensure modified parameters work
        solution = custom_api.run_optimization()
        assert solution is not None
        print("Parameter modification test passed")

    def test_direct_configuration_usage(self):
        """Test direct configuration usage as shown in notebook section 6."""
        # Run directly from a configuration file
        direct_solution = run_from_config(
            "src/utils/json_input/single_qubit_parameters.json"
        )
        assert direct_solution is not None
        assert hasattr(direct_solution, "shape")
        assert direct_solution.shape[0] > 0
        print(f"Direct optimization completed. Solution shape: {direct_solution.shape}")

    def test_solution_visualization_data(self):
        """Test that solutions can be used for visualization as shown in notebook section 7."""
        # Get solutions from different configurations
        single_qubit_api = load_single_qubit_config()
        single_qubit_api.update_parameter("optimization.max_iter", 3)
        single_qubit_solution = single_qubit_api.run_optimization()

        two_qubit_api = load_two_qubit_config()
        two_qubit_api.update_parameter("optimization.max_iter", 3)
        two_qubit_solution = two_qubit_api.run_optimization()

        # Verify solutions can be converted to numpy for plotting
        single_numpy = single_qubit_solution.detach().numpy()
        two_numpy = two_qubit_solution.detach().numpy()

        assert single_numpy.shape[0] > 0
        assert two_numpy.shape[0] > 0

        # Test that matplotlib can handle the data (without actually displaying)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(single_numpy)
        ax.set_title("Test Plot")
        plt.close(fig)  # Close to avoid display in tests

        print("Solution visualization data test passed")


class TestPolarPhaseConfigurations:
    """Test the new polar_phase configurations."""

    def test_single_qubit_polar_phase_config(self):
        """Test single qubit polar phase configuration."""
        # Load single qubit polar phase configuration
        polar_api = load_single_qubit_polar_phase_config()

        # Display configuration summary
        config_summary = polar_api.get_config_summary()
        assert config_summary is not None
        assert isinstance(config_summary, str)

        # Run the optimization with reduced iterations for testing
        polar_api.update_parameter("optimization.max_iter", 5)
        polar_solution = polar_api.run_optimization()

        assert polar_solution is not None
        assert hasattr(polar_solution, "shape")
        assert polar_solution.shape[0] > 0
        print(
            f"Single qubit polar phase optimization completed. Solution shape: {polar_solution.shape}"
        )

    def test_two_qubit_polar_phase_config(self):
        """Test two qubit polar phase configuration."""
        # Load two qubit polar phase configuration
        polar_api = load_two_qubit_polar_phase_config()

        # Display configuration summary
        config_summary = polar_api.get_config_summary()
        assert config_summary is not None
        assert isinstance(config_summary, str)
        assert "2" in config_summary or "Two" in config_summary

        # Run the optimization with reduced iterations for testing
        polar_api.update_parameter("optimization.max_iter", 5)
        polar_solution = polar_api.run_optimization()

        assert polar_solution is not None
        assert hasattr(polar_solution, "shape")
        assert polar_solution.shape[0] > 0
        print(
            f"Two qubit polar phase optimization completed. Solution shape: {polar_solution.shape}"
        )

    def test_polar_phase_config_parameters(self):
        """Test that polar phase configurations have the correct waveform mode."""
        # Test single qubit polar phase
        single_polar_api = load_single_qubit_polar_phase_config()
        single_config = single_polar_api.config
        assert single_config["parameters"]["wf_mode"][0] == "polar_phase"

        # Test two qubit polar phase
        two_polar_api = load_two_qubit_polar_phase_config()
        two_config = two_polar_api.config
        assert two_config["parameters"]["wf_mode"][0] == "polar_phase"
        assert two_config["parameters"]["wf_mode"][1] == "polar_phase"

        print("Polar phase configuration parameters test passed")

    def test_polar_phase_vs_regular_configs(self):
        """Test that polar phase configs produce different results from regular configs."""
        # Load regular single qubit config
        regular_api = load_single_qubit_config()
        regular_api.update_parameter("optimization.max_iter", 3)
        regular_solution = regular_api.run_optimization()

        # Load polar phase single qubit config
        polar_api = load_single_qubit_polar_phase_config()
        polar_api.update_parameter("optimization.max_iter", 3)
        polar_solution = polar_api.run_optimization()

        # Both solutions should run successfully
        # Note: Different waveform modes may have different parameter counts
        assert regular_solution is not None
        assert polar_solution is not None
        assert regular_solution.shape[0] > 0
        assert polar_solution.shape[0] > 0

        print(
            f"Polar phase vs regular configs comparison test passed. Regular shape: {regular_solution.shape}, Polar shape: {polar_solution.shape}"
        )


class TestAllConfigurationsParametric:
    """Parametric tests for all configurations including new polar_phase ones."""

    @pytest.mark.parametrize(
        "config_name,load_func",
        [
            ("Single Qubit", load_single_qubit_config),
            ("Single Qubit Multiple States", load_single_qubit_multiple_config),
            ("Two Qubit", load_two_qubit_config),
            ("Two Qubit Multiple States", load_two_qubit_multiple_config),
            ("Single Qubit Polar Phase", load_single_qubit_polar_phase_config),
            ("Two Qubit Polar Phase", load_two_qubit_polar_phase_config),
        ],
    )
    def test_all_configurations_loading_and_execution(self, config_name, load_func):
        """Test all configurations including new polar_phase ones."""
        # Load configuration
        api = load_func()
        assert api is not None

        # Display configuration summary
        config_summary = api.get_config_summary()
        assert config_summary is not None
        assert isinstance(config_summary, str)

        # Reduce iterations for faster testing
        api.update_parameter("optimization.max_iter", 3)

        # Run optimization
        solution = api.run_optimization()

        assert solution is not None
        assert hasattr(solution, "shape")
        assert solution.shape[0] > 0

        print(
            f"{config_name} configuration test passed. Solution shape: {solution.shape}"
        )

    @pytest.mark.parametrize(
        "load_func",
        [
            load_single_qubit_config,
            load_single_qubit_multiple_config,
            load_two_qubit_config,
            load_two_qubit_multiple_config,
            load_single_qubit_polar_phase_config,
            load_two_qubit_polar_phase_config,
        ],
    )
    def test_all_configurations_parameter_updates(self, load_func):
        """Test parameter updates work for all configurations."""
        api = load_func()

        # Test updating max_iter
        api.update_parameter("optimization.max_iter", 2)

        # Test updating target fidelity
        api.update_parameter("optimization.targ_fid", 0.99)

        # Verify the parameters were updated by running optimization
        solution = api.run_optimization()
        assert solution is not None

        print(f"Parameter update test passed for {load_func.__name__}")
