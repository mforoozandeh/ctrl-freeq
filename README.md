# ctrl-freeq

Ctrl-freeq is a numerical framework for the design of quantum gates and pulses via optimal control theory, implemented in Python with PyTorch. The control pulse optimization problem is formulated and solved using automatic differentiation, and the resulting software provides both a programmatic API and a graphical user interface for the configuration and execution of optimization workflows.

- **PyPI package**: `ctrl-freeq`
- **Import path**: `ctrl_freeq`
- **Minimum Python**: 3.13
- **CLI entry point**: `freeq-gui`

## Features

The framework supports the definition and solution of gate optimization problems through JSON configuration files. A high-level Python API is provided for loading configurations, executing the optimizer, and post-processing results, while a Tkinter-based GUI (launched via `freeq-gui`) offers an interactive alternative for users who prefer a graphical workflow. Optimization results may be visualized through interactive Plotly dashboards exported as standalone HTML files. CPU thread management is handled automatically, and optional GPU acceleration is available via CUDA.

## Installation

Ctrl-freeq depends on NumPy, PyTorch, Qiskit, pytorch-minimize, Plotly, and other scientific packages.

```bash
# Install from source
uv sync

# Or with development tools
uv sync --group dev
```

> **Note:** Dependency groups (`dev`, `docs`, `tools`) use [PEP 735](https://peps.python.org/pep-0735/) and require `uv` — they are not pip extras.

For pip users:

```bash
pip install .
```

A detailed treatment of all installation methods is provided in the [Installation](https://ctrl-freeq.readthedocs.io/) documentation.

## Quick Start

### Launch the GUI

```bash
freeq-gui
```

Execution of this command launches the Tkinter-based graphical interface defined in `ctrl_freeq/cli.py`.

### Use the API

```python
from ctrl_freeq.api import CtrlFreeQAPI, run_from_config, load_single_qubit_config

# Load a built-in example configuration
config = load_single_qubit_config()

# Option A: one-shot helper
result = run_from_config(config)

# Option B: create an API instance to inspect and edit parameters
api = CtrlFreeQAPI(config)
print(api.get_config_summary())
api.update_parameter("optimization.max_iter", 500)
result = api.run_optimization()
print(f"Final fidelity: {api.parameters.final_fidelity:.6f}")
```

## Project Structure

```
src/ctrl_freeq/       Core package
  api.py              High-level API
  cli.py              CLI entry point (freeq-gui)
  setup/              GUI and configuration setup
  run/                Optimization engine
  utils/              Plotting, dashboards, helpers
examples/             Example configs and notebooks
tests/                Test suite
pyproject.toml        Build and packaging configuration
```

## Development

Development tools are organized into dependency groups: `dev` (pytest, jupyter), `docs` (mkdocs), and `tools` (ruff, pre-commit).

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest -q

# Lint
uv run ruff check .
```

## Compute Resource Selection (CPU/GPU)

Ctrl-freeq supports execution on both CPU and GPU (CUDA) hardware:

- **`cpu`** (default) — PyTorch threads are limited to `max(1, os.cpu_count() - 1)`. This default may be overridden with the `cpu_cores` field.
- **`gpu`** — CUDA is used if available; the framework falls back to CPU with a warning if CUDA is not present.

### API Configuration

```python
config = {
    "compute_resource": "gpu",   # or "cpu" (default)
    "cpu_cores": 8,              # optional, CPU only
    ...
}
```

### GUI

The compute resource may be selected from the **Compute Resource** dropdown in the Optimization Parameters section of the graphical interface.

> **Note:** Only CUDA is supported for GPU acceleration. MPS is not currently supported.

A demonstration notebook is provided at `examples/api_gpu_cpu_demo.ipynb`, illustrating CPU vs CUDA selection and comparative timing.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If this software is used in academic work, please cite it using the metadata provided in [CITATION.cff](CITATION.cff).
