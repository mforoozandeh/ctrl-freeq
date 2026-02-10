# Installation

This page covers how to install ctrl-freeq from **PyPI** (stable and beta releases) or directly from **GitHub**.

## Requirements

- **Python 3.13+** (required)
- A working installation of `pip` or `uv`

### Core Dependencies

ctrl-freeq automatically installs the following dependencies:

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
| `tqdm` | Progress bars |

!!! note
    The PyPI distribution name is `ctrl-freeq`.

---

## Install from PyPI

### Stable Releases

Install the latest stable version:

```bash
pip install ctrl-freeq
```

### Beta (Pre-release) Versions

Pre-releases follow standard Python packaging (PEP 440). They are **not** installed by default.

Install the latest available pre-release:

```bash
pip install --pre ctrl-freeq
```

Or install a specific beta version:

```bash
pip install ctrl-freeq==0.1.0b4
```

!!! tip "Finding Available Versions"
    View all available versions on [PyPI](https://pypi.org/project/ctrl-freeq/#history).

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

For development or contributing:

```bash
git clone https://github.com/mforoozandeh/ctrl-freeq.git
cd ctrl-freeq

# Using uv (recommended) — installs with dev dependency group
uv sync --group dev

# Or using pip — installs the package in editable mode (without dev dependencies)
pip install -e .
```

!!! note "Dependency Groups"
    Development dependencies (`pytest`, `jupyter`, etc.) are defined as [dependency groups](https://peps.python.org/pep-0735/) in `pyproject.toml`. Use `uv sync --group dev` to install them. The `pip install -e .[dev]` syntax does not support dependency groups.

---

## Using uv (Recommended Package Manager)

If you prefer using [uv](https://github.com/astral-sh/uv):

```bash
# Install from PyPI
uv pip install ctrl-freeq

# Install pre-release
uv pip install --prerelease=allow ctrl-freeq

# Development install with dependency groups
git clone https://github.com/mforoozandeh/ctrl-freeq.git
cd ctrl-freeq
uv sync --group dev --group docs --group tools
```

---

## Verify the Installation

### Launch the GUI

```bash
freeq-gui
```

If the command is found and the window opens, the installation is complete.

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

ctrl-freeq supports GPU acceleration via PyTorch CUDA. To enable GPU support:

1. Ensure you have a CUDA-compatible NVIDIA GPU
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

See [Compute (CPU/GPU)](optimization/compute.md) for configuration details.

---

## Troubleshooting

### Common Issues

#### `freeq-gui` command not found

Ensure the package is installed and your Python scripts directory is in your PATH:

```bash
# Check if installed
pip show ctrl-freeq

# Try running with python -m
python -m ctrl_freeq.cli
```

#### Import errors

Verify all dependencies are installed:

```bash
pip install --upgrade ctrl-freeq
```

#### Tkinter not available

The GUI requires Tkinter. On some systems, you may need to install it separately:

=== "Ubuntu/Debian"
    ```bash
    sudo apt-get install python3-tk
    ```

=== "Fedora"
    ```bash
    sudo dnf install python3-tkinter
    ```

=== "macOS"
    Tkinter is included with Python from python.org. If using Homebrew:
    ```bash
    brew install python-tk@3.13
    ```

=== "Windows"
    Tkinter is included with the standard Python installer.

#### CUDA/GPU issues

If GPU mode fails:

1. Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
2. Check CUDA driver compatibility with your PyTorch version
3. Fall back to CPU mode by setting `compute_resource: "cpu"` in your configuration

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
