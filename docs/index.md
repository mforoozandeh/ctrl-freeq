# ctrl-freeq

Gate design with optimal control for quantum circuits, implemented in Python with PyTorch.

ctrl-freeq finds optimized control pulses for quantum gates by solving the optimal control problem numerically. It supports single- and multi-qubit systems with configurable Hamiltonians, coupling types, and robustness to parameter uncertainties.

## Key Features

- **Multiple optimization algorithms** — 9 gradient-based methods via pytorch-minimize and 10 Qiskit optimizers
- **Flexible quantum systems** — Single-qubit through multi-qubit with configurable coupling (Z, XY, XYZ)
- **Robust pulse design** — Account for detuning and Rabi frequency uncertainties
- **Waveform basis functions** — Chebyshev, Fourier, and polynomial bases in Cartesian or polar modes
- **GPU acceleration** — Optional CUDA support via PyTorch with automatic CPU fallback
- **Interactive dashboards** — Combined Matplotlib/Plotly analysis exported as standalone HTML
- **Two interfaces** — Programmatic Python API and Tkinter-based GUI (`freeq-gui`)

## Quick Example

```python
from ctrl_freeq.api import load_single_qubit_config

# Load a built-in single-qubit configuration
api = load_single_qubit_config()

# Run optimization
solution = api.run_optimization()

# Check results
print(f"Final fidelity: {api.parameters.final_fidelity}")
```

## Start Here

- Install ctrl-freeq: [`Installation`](installation.md)
- Run from Python: [`API`](api.md)
- Configure runs interactively: [`GUI`](gui.md)
- Learn about algorithms, parameters, and CPU/GPU support: [`Optimization`](optimization/index.md)
- View and export analysis dashboards: [`Dashboard`](dashboard.md)

## License

Apache License 2.0. See [`LICENSE`](https://github.com/mforoozandeh/ctrl-freeq/blob/main/LICENSE).
