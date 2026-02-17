# Installation

In this section, the installation procedure for ctrl-freeq is described. The package may be obtained either from PyPI, where both stable and pre-release versions are published, or directly from the GitHub repository.

## Requirements

- **Python 3.11–3.13** (required)
- A working installation of `pip` or `uv`

### Core Dependencies

Ctrl-freeq automatically installs the following dependencies:

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computing |
| `matplotlib` | Plotting and visualization |
| `torch` (PyTorch) | Tensor operations and GPU acceleration |
| `pytorch-minimize` | Gradient-based optimization on PyTorch tensors |
| `scipy` | Scientific computing |
| `qiskit` | Quantum computing framework |
| `qiskit-algorithms` | Optimization algorithms |
| `plotly` | Interactive plots |
| `kaleido` | Static figure export |
| `panel` | Dashboard generation |

!!! note
    It should be noted that the PyPI distribution name is `ctrl-freeq`, which differs from the Python import path `ctrl_freeq`.

---

## Install from PyPI

### Stable Releases

The latest stable version may be installed via pip:

```bash
pip install ctrl-freeq
```

### Beta (Pre-release) Versions

Pre-releases follow standard Python packaging conventions (PEP 440) and are **not** installed by default.

The latest available pre-release may be installed as follows:

```bash
pip install --pre ctrl-freeq
```

A specific beta version may also be requested:

```bash
pip install ctrl-freeq==0.1.0b4
```

!!! tip "Finding Available Versions"
    All available versions may be viewed on [PyPI](https://pypi.org/project/ctrl-freeq/#history).

---

## Install from GitHub

### Install the Latest Commit from `main`

```bash
pip install "ctrl-freeq @ git+https://github.com/mforoozandeh/ctrl-freeq.git"
```

### Install a Specific Branch or Tag

```bash
# Install from a specific branch
pip install "ctrl-freeq @ git+https://github.com/mforoozandeh/ctrl-freeq.git@branch-name"

# Install from a specific tag
pip install "ctrl-freeq @ git+https://github.com/mforoozandeh/ctrl-freeq.git@v0.1.0"
```

### Editable/Development Install

For development or contributing purposes, the package may be installed in editable mode:

```bash
git clone https://github.com/mforoozandeh/ctrl-freeq.git
cd ctrl-freeq

# Using uv (recommended) — installs with all dependency group
uv sync --all-groups

# Or using pip — installs the package in editable mode (without dev dependencies)
pip install -e .
```

!!! note "Dependency Groups"
    Development dependencies (`pytest`, `jupyter`, etc.) are defined as [dependency groups](https://peps.python.org/pep-0735/) in `pyproject.toml`. These are installed via `uv sync --group dev`. It should be noted that the `pip install -e .[dev]` syntax does not support dependency groups.

---

## Using uv (Recommended Package Manager)

As an alternative to pip, the [uv](https://github.com/astral-sh/uv) package manager may be employed, which offers certain advantages in dependency resolution and environment management:

```bash
# Install from PyPI
uv pip install ctrl-freeq

# Install pre-release
uv pip install --prerelease=allow ctrl-freeq

# Development install with dependency groups
git clone https://github.com/mforoozandeh/ctrl-freeq.git
cd ctrl-freeq
uv sync --all-groups
```

---

## Verify the Installation

### Launch the GUI

```bash
freeq-gui
```

If the command is found and the graphical interface window opens successfully, the installation is complete.

### Verify in Python

```python
from ctrl_freeq.api import CtrlFreeQAPI, load_single_qubit_config

# Load a built-in example configuration
api = load_single_qubit_config()
print(api.get_config_summary())
```

### Check Version

```python
import ctrl_freeq
print(ctrl_freeq.__version__)
```

---

## Upgrading

### Upgrade to Latest Stable

```bash
pip install --upgrade ctrl-freeq
```

### Upgrade to Latest Pre-release

```bash
pip install --upgrade --pre ctrl-freeq
```

---

## GPU Support (Optional)

Ctrl-freeq supports GPU acceleration via the CUDA backend of PyTorch. To enable GPU support, the following steps should be followed:

1. Ensure that a CUDA-compatible NVIDIA GPU is available
2. Install PyTorch with CUDA support:

```bash
# Example for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

3. Verify CUDA availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
```

A detailed treatment of GPU configuration is provided in [Compute (CPU/GPU)](optimization/compute.md).

---

## Troubleshooting

### Common Issues

#### `freeq-gui` command not found

It should be verified that the package is installed and that the Python scripts directory is included in the system PATH:

```bash
# Check if installed
pip show ctrl-freeq

# Try running with python -m
python -m ctrl_freeq.cli
```

#### Import errors

All dependencies may be reinstalled by upgrading the package:

```bash
pip install --upgrade ctrl-freeq
```

#### Tkinter not available

The GUI requires Tkinter, which may need to be installed separately on certain systems:

=== "Ubuntu/Debian"
    ```bash
    sudo apt-get install python3-tk
    ```

=== "Fedora"
    ```bash
    sudo dnf install python3-tkinter
    ```

=== "macOS"
    Tkinter is included with the Python distribution from python.org. If Homebrew is used:
    ```bash
    brew install python-tk@3.11  # adjust version to match your Python
    ```

=== "Windows"
    Tkinter is included with the standard Python installer.

#### CUDA/GPU issues

If GPU mode fails, the following steps may be taken:

1. Verify the PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check CUDA driver compatibility with the installed PyTorch version
3. Fall back to CPU mode by setting `compute_resource: "cpu"` in the configuration

---

## Uninstalling

```bash
pip uninstall ctrl-freeq
```

---

## Next Steps

- [API Reference](api.md) — Use ctrl-freeq programmatically
- [GUI Guide](gui.md) — Configure and run optimization interactively
- [Optimization Parameters](optimization/parameters.md) — Understand configuration options
