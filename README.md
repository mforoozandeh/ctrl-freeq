# ctrl-freeq

Gate design with optimal control for quantum circuits, implemented in Python with PyTorch. This repository provides both a programmatic API and a GUI for configuring and running ctrl-freeq optimization workflows.

- **PyPI package**: `ctrl-freeq`
- **Import path**: `ctrl_freeq`
- **Minimum Python**: 3.13
- **CLI entry point**: `freeq-gui`

## Features

- Define and run gate optimization problems via JSON configuration.
- Programmatic API for loading configs, running the optimizer, and post-processing results.
- GUI launcher for interactive runs (via `freeq-gui`).
- Interactive Plotly dashboards for result visualization.
- CPU thread management and optional GPU (CUDA) acceleration.

## Installation

ctrl-freeq depends on NumPy, PyTorch, Qiskit, pytorch-minimize, Plotly, and other scientific packages.

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

See [Installation](https://ctrl-freeq.readthedocs.io/) for full details.

## Quick Start

### Launch the GUI

```bash
freeq-gui
```

This launches the Tkinter-based GUI defined in `ctrl_freeq/cli.py`.

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

Development tools are in the `dev` dependency group. Additional groups: `docs` (mkdocs), `tools` (ruff, pre-commit).

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest -q

# Lint
uv run ruff check .
```

## Compute Resource Selection (CPU/GPU)

ctrl-freeq supports CPU and GPU (CUDA) execution:

- **`cpu`** (default) — Limits PyTorch threads to `max(1, os.cpu_count() - 1)`. Override with `cpu_cores`.
- **`gpu`** — Uses CUDA if available; falls back to CPU with a warning if not.

### API Configuration

```python
config = {
    "compute_resource": "gpu",   # or "cpu" (default)
    "cpu_cores": 8,              # optional, CPU only
    ...
}
```

### GUI

Use the **Compute Resource** dropdown in the Optimization Parameters section.

> **Note:** Only CUDA is supported for GPU. MPS is not currently supported.

A demo notebook is provided at `examples/api_gpu_cpu_demo.ipynb` showing CPU vs CUDA selection and timing.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in academic work, please cite it appropriately. A CITATION.cff can be added upon request.
