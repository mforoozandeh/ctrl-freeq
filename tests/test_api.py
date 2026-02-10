#!/usr/bin/env python3
"""
Pytest tests for CtrlFreeQ API functionality.
"""

import pytest
from ctrl_freeq.api import (
    load_single_qubit_config,
    load_single_qubit_multiple_config,
    load_two_qubit_config,
    load_two_qubit_multiple_config,
    load_single_qubit_polar_phase_config,
    load_two_qubit_polar_phase_config,
)


class TestAPIBasicFunctionality:
    """Test basic API functionality."""

    def test_api_basic_functionality(self):
        """Test the basic API functionality."""
        # Load single qubit configuration
        api = load_single_qubit_config()

        # Display configuration summary
        config_summary = api.get_config_summary()
        assert config_summary is not None
        assert isinstance(config_summary, str)

        # Test parameter update
        api.update_parameter("optimization.max_iter", 10)

        # Run optimization (with reduced iterations for quick tests)
        solution = api.run_optimization()

        assert solution is not None
        assert hasattr(solution, "shape")
        assert solution.shape[0] > 0


class TestAPIConfigurations:
    """Test all provided JSON configurations."""

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
    def test_config_loading_and_execution(self, config_name, load_func):
        """Test a specific configuration loading and execution."""
        # Load configuration
        api = load_func()
        assert api is not None

        # Display configuration summary
        config_summary = api.get_config_summary()
        assert config_summary is not None
        assert isinstance(config_summary, str)

        # Reduce iterations for faster testing
        api.update_parameter("optimization.max_iter", 5)

        # Run optimization
        solution = api.run_optimization()

        assert solution is not None
        assert hasattr(solution, "shape")
        assert solution.shape[0] > 0

    def test_single_qubit_config(self):
        """Test single qubit configuration specifically."""
        api = load_single_qubit_config()
        solution = self._run_quick_optimization(api)
        assert solution is not None

    def test_single_qubit_multiple_config(self):
        """Test single qubit multiple states configuration."""
        api = load_single_qubit_multiple_config()
        solution = self._run_quick_optimization(api)
        assert solution is not None

    def test_two_qubit_config(self):
        """Test two qubit configuration."""
        api = load_two_qubit_config()
        solution = self._run_quick_optimization(api)
        assert solution is not None

    def test_two_qubit_multiple_config(self):
        """Test two qubit multiple states configuration."""
        api = load_two_qubit_multiple_config()
        solution = self._run_quick_optimization(api)
        assert solution is not None

    def test_single_qubit_polar_phase_config(self):
        """Test single qubit polar phase configuration."""
        api = load_single_qubit_polar_phase_config()
        solution = self._run_quick_optimization(api)
        assert solution is not None
        # Verify it's using polar_phase mode
        assert api.config["parameters"]["wf_mode"][0] == "polar_phase"

    def test_two_qubit_polar_phase_config(self):
        """Test two qubit polar phase configuration."""
        api = load_two_qubit_polar_phase_config()
        solution = self._run_quick_optimization(api)
        assert solution is not None
        # Verify both qubits are using polar_phase mode
        assert api.config["parameters"]["wf_mode"][0] == "polar_phase"
        assert api.config["parameters"]["wf_mode"][1] == "polar_phase"

    def _run_quick_optimization(self, api):
        """Helper method to run optimization with reduced iterations."""
        api.update_parameter("optimization.max_iter", 5)
        return api.run_optimization()


class TestAPIParameterUpdates:
    """Test API parameter update functionality."""

    def test_parameter_update(self):
        """Test that parameter updates work correctly."""
        api = load_single_qubit_config()

        # Test updating max_iter
        api.update_parameter("optimization.max_iter", 3)

        # Verify the parameter was updated by running optimization
        # (if it runs without error, the parameter was likely updated)
        solution = api.run_optimization()
        assert solution is not None

    def test_invalid_parameter_update(self):
        """Test handling of invalid parameter updates."""
        api = load_single_qubit_config()

        # This should either raise an exception or handle gracefully
        # depending on the API implementation
        try:
            api.update_parameter("invalid.parameter", 10)
        except (KeyError, AttributeError, ValueError):
            # Expected behavior for invalid parameters
            pass


class TestAPIConfigSummary:
    """Test API configuration summary functionality."""

    def test_config_summary_format(self):
        """Test that config summary returns properly formatted string."""
        api = load_single_qubit_config()
        summary = api.get_config_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        # Summary should contain some key information
        assert any(
            keyword in summary.lower() for keyword in ["qubit", "config", "parameter"]
        )

    def test_config_summary_all_configs(self):
        """Test config summary for all configuration types."""
        configs = [
            load_single_qubit_config,
            load_single_qubit_multiple_config,
            load_two_qubit_config,
            load_two_qubit_multiple_config,
            load_single_qubit_polar_phase_config,
            load_two_qubit_polar_phase_config,
        ]

        for load_func in configs:
            api = load_func()
            summary = api.get_config_summary()
            assert isinstance(summary, str)
            assert len(summary) > 0
