# ctrl-freeq

Ctrl-freeq is a numerical framework for the design of quantum gates and pulses via optimal control theory. The control pulse optimization problem is formulated and solved using automatic differentiation, as provided by PyTorch, and the resulting software supports single- and multi-qubit systems with configurable Hamiltonians, inter-qubit coupling types, and robustness to parameter uncertainties.

## Key Features

The framework provides access to a comprehensive suite of optimization algorithms, comprising nine gradient-based methods available through the pytorch-minimize library and ten additional optimizers from the Qiskit ecosystem. Quantum systems of varying complexity are supported, ranging from single-qubit configurations to multi-qubit architectures with configurable inter-qubit coupling of the Ising (Z), exchange (XY), or Heisenberg (XYZ) type.

Robustness to experimental imperfections is incorporated through the specification of detuning and Rabi frequency uncertainties, allowing the optimizer to find pulses that are tolerant to parameter variations. Open quantum systems subject to decoherence may be modelled via the Lindblad master equation, with support for amplitude damping (T1) and pure dephasing (T2) relaxation channels on a per-qubit basis. The waveform parameterization is flexible, supporting Chebyshev, Fourier, and polynomial basis functions in Cartesian or polar modes.

GPU acceleration is available via the CUDA backend of PyTorch, with automatic fallback to CPU execution when CUDA is not present. Interactive dashboards, combining Matplotlib and Plotly visualizations, are generated as standalone HTML files for analysis and sharing. Two complementary interfaces are provided: a programmatic Python API and a Tkinter-based graphical user interface, accessible via the `freeq-gui` command.

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
