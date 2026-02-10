#!/usr/bin/env python3
"""
Pytest tests to verify key parts of the demo notebook work correctly.
"""

import pytest


def test_notebook_imports():
    """Test that all imports from the notebook work."""
    import importlib.util

    # Test basic imports that should be available
    try:
        # Test src.api availability
        api_spec = importlib.util.find_spec("ctrl_freeq.api")
        assert api_spec is not None, "src.api module not found"

        # Test matplotlib.pyplot availability
        plt_spec = importlib.util.find_spec("matplotlib.pyplot")
        assert plt_spec is not None, "matplotlib.pyplot module not found"

        # If we get here, imports are available
        assert True
    except Exception as e:
        pytest.fail(f"Import availability test failed: {e}")


def test_notebook_basic_usage():
    """Test basic notebook usage patterns."""
    from ctrl_freeq.api import load_single_qubit_config

    # Load configuration
    api = load_single_qubit_config()
    assert api is not None

    # Get summary
    summary = api.get_config_summary()
    assert summary is not None
    assert isinstance(summary, str)

    # Update parameter
    api.update_parameter("optimization.max_iter", 3)

    # Run optimization
    solution = api.run_optimization()
    assert solution is not None
    assert hasattr(solution, "shape")
    assert solution.shape[0] > 0


def test_notebook_direct_config():
    """Test direct configuration usage from notebook."""
    from ctrl_freeq.api import run_from_config
    import os

    # Check if config file exists
    config_path = "src/utils/json_input/single_qubit_parameters.json"
    if not os.path.exists(config_path):
        pytest.skip(f"Config file not found: {config_path}")

    # Run directly from config
    solution = run_from_config(config_path)
    assert solution is not None
    assert hasattr(solution, "shape")
    assert solution.shape[0] > 0


def test_notebook_visualization_prep():
    """Test that solutions can be prepared for visualization."""
    from ctrl_freeq.api import load_single_qubit_config
    import matplotlib.pyplot as plt

    # Get a solution
    api = load_single_qubit_config()
    api.update_parameter("optimization.max_iter", 2)
    solution = api.run_optimization()

    # Test that we can convert to numpy for plotting
    solution_numpy = solution.detach().numpy()
    assert solution_numpy is not None
    assert hasattr(solution_numpy, "shape")

    # Test basic plotting setup (without actually showing plot)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(solution_numpy)
    ax.set_title("Test Plot")
    plt.close(fig)  # Close to avoid display

    # If we get here, visualization prep worked
    assert True
