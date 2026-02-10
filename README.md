# CtrlFreeQ

Gate design with optimal control for quantum circuits, implemented in Python with PyTorch. This repository provides both a programmatic API and a simple GUI entry point to run CtrlFreeQ optimization workflows.

- Project name (distribution on PyPI/TestPyPI): `ctrl-freeq`
- Python module path: `src`
- Minimum Python: 3.13
- CLI entry point: `freeq-gui`

## Features
- Define and run gate optimization problems via JSON/YAML-like configuration.
- Programmatic API for loading configs, running the optimizer, and updating parameters.
- GUI launcher for interactive runs (via `freeq-gui`).

## Installation
CtrlFreeQ depends on NumPy, SciPy, PyTorch, Qiskit, Plotly, and a few other scientific packages. Install from source:

```bash
# from the repository root
pip install .
```

Or using a development install:

```bash
pip install -e .[dev]
```

Note: Python 3.13 or newer is required.

## Quick start

### Launch the GUI
After installation, run:

```bash
freeq-gui
```

This launches the CtrlFreeQ GUI (Tkinter-based) defined in `src/cli.py` and `src/setup/gui_setup.py`.

### Use the API
You can load a configuration and run optimization directly from Python using the high-level API:

```python
from src.ctrl_freeq.api import CtrlFreeQAPI, run_from_config, load_single_qubit_config

# Load a built-in example configuration
config = load_single_qubit_config()

# Option A: one-shot helper
result = run_from_config(config)

# Option B: create an API instance to inspect and edit parameters
api = CtrlFreeQAPI(config)
print(api.get_config_summary())
api.update_parameter("control.amplitude", 0.5)
run_result = api.run_optimization()
```

## Project structure (high level)
- `src/` – Core package sources
  - `api.py` – High-level API for loading configs and running optimization
  - `cli.py` – CLI entry point that launches the GUI (`freeq-gui`)
  - Additional modules under `setup/`, `run/`, `utils/`, etc.
- `pyproject.toml` – Build and packaging configuration
- `RELEASE.md` – Release notes and guidelines

## Development
Recommended tools are listed under the `dev` dependency group in `pyproject.toml` (pytest, jupyter, tbump, etc.). Typical workflow:

```bash
# Run tests
pytest -q

# Lint (if configured)
ruff .
```

## License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Citation
If you use this project in academic work, please cite it appropriately. A CITATION.cff can be added upon request.


## Compute resource selection (CPU/GPU)
- New option compute_resource is available in both API and GUI.
- Values: "cpu" (default) or "gpu" (CUDA only).
- If "gpu" is selected but CUDA is not available (torch.cuda.is_available() is False), the run automatically falls back to CPU and logs a visible warning.
- On CPU runs, the number of PyTorch threads is limited to max(1, os.cpu_count() - 1) by default. You may optionally provide cpu_cores to override and it will be clamped to [1, os.cpu_count()].

Examples
- API: include at top-level of your config dict:
  {
    "compute_resource": "gpu",  # or "cpu" (default)
    "cpu_cores": 8,              # optional
    ...
  }
  If absent, compute_resource defaults to "cpu".
- GUI: Use the "Compute Resource" dropdown in the Optimization section to choose cpu or gpu.

Notes
- Only CUDA is supported for GPU. We do not check or use MPS.
- If CUDA is unavailable, a warning is printed and the run continues on CPU automatically.

A Colab-ready demo notebook/script is provided at notebooks/api_gpu_cpu_demo.py showing CPU vs CUDA selection and timing. You can open it in Colab directly or run it locally as a script/notebook (it uses #%% cells).
