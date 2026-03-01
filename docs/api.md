# API Reference

A high-level Python API is provided for the programmatic execution of quantum control optimizations. The principal interface is the `CtrlFreeQAPI` class, defined in the `ctrl_freeq.api` module, which encapsulates the complete workflow from configuration loading through optimization to result retrieval.

---

## Quick Start

```python
from ctrl_freeq.api import CtrlFreeQAPI, load_single_qubit_config

# Load a built-in example configuration
api = load_single_qubit_config()

# View configuration summary
print(api.get_config_summary())

# Run optimization (returns optimized pulse parameters as a torch.Tensor)
solution = api.run_optimization()

# Access results stored on the parameters object
print(f"Final fidelity: {api.parameters.final_fidelity}")
print(f"Iterations: {api.parameters.iterations}")
```

---

## CtrlFreeQAPI Class

The `CtrlFreeQAPI` class serves as the primary interface for interacting with the ctrl-freeq optimization framework.

### Constructor

```python
CtrlFreeQAPI(config: Union[str, Path, Dict[str, Any]], hamiltonian_model=None)
```

An API instance may be created from a path to a JSON configuration file (provided as a string or `Path` object) or from a dictionary containing the configuration parameters directly. An optional `hamiltonian_model` argument allows a pre-built `HamiltonianModel` instance to be injected directly, bypassing the registry lookup.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str`, `Path`, or `dict` | Configuration source |
| `hamiltonian_model` | `HamiltonianModel` or `None` | Optional pre-built model instance (overrides `hamiltonian_type` in config) |

**Examples:**

=== "From JSON file"
    ```python
    from ctrl_freeq.api import CtrlFreeQAPI

    api = CtrlFreeQAPI("path/to/config.json")
    ```

=== "From dictionary"
    ```python
    from ctrl_freeq.api import CtrlFreeQAPI

    config = {
        "qubits": ["q1"],
        "hamiltonian_type": "spin_chain",  # or "superconducting"
        "compute_resource": "cpu",
        "parameters": {
            "Delta": [0.0],
            "Omega_R_max": [10000.0],
            "pulse_duration": [0.001],
            "point_in_pulse": [100],
            # ... other parameters
        },
        "initial_states": [["Z"]],
        "target_states": {"Axis": [["-Z"]]},
        "optimization": {
            "space": "hilbert",
            "algorithm": "bfgs",
            "max_iter": 100,
            "targ_fid": 0.99
        }
    }

    api = CtrlFreeQAPI(config)
    ```

=== "From built-in config"
    ```python
    from ctrl_freeq.api import load_single_qubit_config

    api = load_single_qubit_config()
    ```

---

### Methods

#### `run_optimization()`

Executes the quantum control optimization with the loaded configuration.

```python
solution = api.run_optimization()
```

**Returns:** `torch.Tensor` — The optimized pulse parameters.

Upon completion of the optimization, the following attributes are stored on `api.parameters`:

| Attribute | Type | Description |
|-----------|------|-------------|
| `final_fidelity` | `float` | Final achieved fidelity |
| `iterations` | `int` | Number of iterations performed |
| `fidelity_history` | `list[float]` | Fidelity at each iteration |

**Example:**

```python
from ctrl_freeq.api import load_single_qubit_config

api = load_single_qubit_config()
solution = api.run_optimization()

# Access results from the parameters object
print(f"Final fidelity: {api.parameters.final_fidelity}")
print(f"Iterations: {api.parameters.iterations}")
print(f"Fidelity history: {api.parameters.fidelity_history}")
```

---

#### `get_config_summary()`

Returns a human-readable summary of the current configuration.

```python
summary = api.get_config_summary()
```

**Returns:** `str` — A formatted string containing the number of qubits, optimization space (Hilbert/Liouville), algorithm name, maximum iterations, target fidelity, and the initial and target states.

**Example:**

```python
from ctrl_freeq.api import load_two_qubit_config

api = load_two_qubit_config()
print(api.get_config_summary())
```

**Output:**
```
Number of qubits: 2
Optimization space: hilbert
Algorithm: newton-cg
Max iterations: 1000
Target fidelity: 0.999
Initial states: [['Z', '-Z']]
Target Gate: ['CNOT']
```

---

#### `update_parameter(parameter_path, value)`

Updates a specific parameter in the configuration and reinitializes the internal state accordingly.

```python
api.update_parameter(parameter_path: str, value: Any)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `parameter_path` | `str` | Dot-separated path to the parameter |
| `value` | `Any` | New value for the parameter |

**Raises:** `KeyError` if the parameter path is not found.

**Example:**

```python
from ctrl_freeq.api import load_single_qubit_config

api = load_single_qubit_config()

# Update optimization settings
api.update_parameter("optimization.max_iter", 200)
api.update_parameter("optimization.algorithm", "qiskit-cobyla")
api.update_parameter("optimization.targ_fid", 0.999)

# Update pulse parameters
api.update_parameter("parameters.pulse_duration", [0.002])

# Run with updated configuration
solution = api.run_optimization()
```

**Common parameter paths:**

| Path | Description |
|------|-------------|
| `hamiltonian_type` | `spin_chain` or `superconducting` |
| `optimization.algorithm` | Optimization algorithm |
| `optimization.max_iter` | Maximum iterations |
| `optimization.targ_fid` | Target fidelity |
| `optimization.space` | `hilbert` or `liouville` |
| `optimization.dissipation_mode` | `non-dissipative` or `dissipative` |
| `parameters.Delta` | Detuning / qubit frequency (Hz) |
| `parameters.Omega_R_max` | Maximum Rabi frequency / drive amplitude (Hz) |
| `parameters.pulse_duration` | Pulse duration (seconds) |
| `parameters.point_in_pulse` | Discretization points |
| `parameters.T1` | Amplitude damping time (seconds) |
| `parameters.T2` | Pure dephasing time (seconds) |
| `compute_resource` | `cpu` or `gpu` |

---

### Properties

#### `config`

The raw configuration dictionary.

```python
raw_config = api.config
```

#### `config_path`

The path to the configuration file, if the instance was created from a file; otherwise `None`.

```python
path = api.config_path
```

#### `parameters`

The initialized parameter object used internally by the optimization engine.

```python
params = api.parameters
```

---

## Helper Functions

### `run_from_config(config, hamiltonian_model=None)`

A convenience function that creates a `CtrlFreeQAPI` instance and executes the optimization in a single call.

```python
from ctrl_freeq.api import run_from_config

solution = run_from_config("path/to/config.json")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str`, `Path`, or `dict` | Configuration source |
| `hamiltonian_model` | `HamiltonianModel` or `None` | Optional pre-built model instance |

**Returns:** `torch.Tensor` — The optimized pulse parameters.

**Example with dictionary:**

```python
from ctrl_freeq.api import run_from_config

solution = run_from_config({
    "qubits": ["q1"],
    "compute_resource": "cpu",
    "parameters": { ... },
    "initial_states": [["Z"]],
    "target_states": {"Axis": [["-Z"]]},
    "optimization": {
        "space": "hilbert",
        "algorithm": "bfgs",
        "max_iter": 100,
        "targ_fid": 0.99
    }
})
```

---

### `load_config(config_path)`

Loads a JSON configuration file and returns a `CtrlFreeQAPI` instance.

```python
from ctrl_freeq.api import load_config

api = load_config("path/to/config.json")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config_path` | `str` or `Path` | Path to JSON configuration file |

**Returns:** `CtrlFreeQAPI` instance.

---

## Built-in Example Configurations

Several pre-configured examples are bundled with the package for common use cases. These correspond to JSON files located under `src/ctrl_freeq/data/json_input/`.

### Single-Qubit Configurations

#### `load_single_qubit_config()`

A basic single-qubit optimization configuration.

```python
from ctrl_freeq.api import load_single_qubit_config

api = load_single_qubit_config()
solution = api.run_optimization()
```

#### `load_single_qubit_multiple_config()`

A single-qubit configuration with multiple initial/target state pairs for universal rotation design.

```python
from ctrl_freeq.api import load_single_qubit_multiple_config

api = load_single_qubit_multiple_config()
```

#### `load_single_qubit_polar_phase_config()`

A single-qubit configuration using the polar/phase waveform mode.

```python
from ctrl_freeq.api import load_single_qubit_polar_phase_config

api = load_single_qubit_polar_phase_config()
```

#### `load_single_qubit_dissipative_config()`

A single-qubit configuration with dissipative (Lindblad) dynamics enabled. The evolution includes amplitude damping (T1 = 1 ms) and pure dephasing (T2 = 500 μs), and the optimization is performed in Liouville (density matrix) space.

```python
from ctrl_freeq.api import load_single_qubit_dissipative_config

api = load_single_qubit_dissipative_config()
```

### Two-Qubit Configurations

#### `load_two_qubit_config()`

A basic two-qubit optimization configuration with inter-qubit coupling.

```python
from ctrl_freeq.api import load_two_qubit_config

api = load_two_qubit_config()
```

#### `load_two_qubit_multiple_config()`

A two-qubit configuration with multiple initial/target state pairs.

```python
from ctrl_freeq.api import load_two_qubit_multiple_config

api = load_two_qubit_multiple_config()
```

#### `load_two_qubit_polar_phase_config()`

A two-qubit configuration using the polar/phase waveform mode.

```python
from ctrl_freeq.api import load_two_qubit_polar_phase_config

api = load_two_qubit_polar_phase_config()
```

### Multi-Qubit Configurations

#### `load_four_qubit_polar_phase_config()`

A four-qubit optimization configuration using the polar/phase waveform mode.

```python
from ctrl_freeq.api import load_four_qubit_polar_phase_config

api = load_four_qubit_polar_phase_config()
```

---

## Complete Workflow Example

The following example illustrates a complete optimization workflow, from configuration loading through parameter customization, optimization, and result visualization:

```python
from ctrl_freeq.api import CtrlFreeQAPI, load_single_qubit_config
from ctrl_freeq.visualisation.plotter import process_and_plot
from ctrl_freeq.visualisation.dashboard import create_dashboard
from datetime import datetime

# 1. Load configuration
api = load_single_qubit_config()

# 2. Customize parameters
api.update_parameter("optimization.algorithm", "qiskit-cobyla")
api.update_parameter("optimization.max_iter", 150)
api.update_parameter("optimization.targ_fid", 0.995)

# 3. Review configuration
print(api.get_config_summary())

# 4. Run optimization
solution = api.run_optimization()

# 5. Check results
print(f"Final fidelity: {api.parameters.final_fidelity}")
print(f"Iterations: {api.parameters.iterations}")

# 6. Generate plots
# process_and_plot takes the solution tensor and the parameters object
waveforms, figures = process_and_plot(solution, api.parameters, save_plots=True)

# 7. Create dashboard (pass parameters for a richer sidebar)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dashboard_path = create_dashboard(figures, timestamp, parameters=api.parameters)
print(f"Dashboard saved to: {dashboard_path}")
```

---

## Dissipative (Lindblad) Workflow Example

The following example demonstrates a complete optimization workflow for an open quantum system subject to amplitude damping and pure dephasing. The Lindblad master equation is employed to model the dissipative channels, and the optimization is performed in Liouville (density matrix) space.

=== "From built-in config"
    ```python
    from ctrl_freeq.api import load_single_qubit_dissipative_config
    from ctrl_freeq.visualisation.plotter import process_and_plot

    # Load the pre-configured dissipative example
    api = load_single_qubit_dissipative_config()

    # Review configuration (includes T1/T2 information)
    print(api.get_config_summary())

    # Run optimization
    solution = api.run_optimization()

    # Check results
    print(f"Final fidelity: {api.parameters.final_fidelity}")
    print(f"Iterations: {api.parameters.iterations}")

    # Generate plots
    waveforms = process_and_plot(solution, api.parameters, show_plots=True)
    ```

=== "From dictionary"
    ```python
    from ctrl_freeq.api import CtrlFreeQAPI
    from ctrl_freeq.visualisation.plotter import process_and_plot

    config = {
        "qubits": ["q1"],
        "compute_resource": "cpu",
        "parameters": {
            "Delta": [10000000.0],
            "Omega_R_max": [40000000.0],
            "pulse_duration": [2e-07],
            "point_in_pulse": [100],
            "wf_type": ["cheb"],
            "wf_mode": ["cart"],
            "amplitude_envelope": ["gn"],
            "amplitude_order": [1],
            "coverage": ["broadband"],
            "sw": [5000000.0],
            "pulse_offset": [0.0],
            "pulse_bandwidth": [500000.0],
            "ratio_factor": [0.5],
            "sigma_Delta": [0.0],
            "sigma_Omega_R_max": [0.0],
            "profile_order": [2],
            "n_para": [16],
            "J": [[0.0]],
            "T1": [0.001],           # 1 ms amplitude damping
            "T2": [0.0005]           # 500 μs pure dephasing
        },
        "initial_states": [["Z"]],
        "target_states": {"Axis": [["-Z"]]},
        "optimization": {
            "space": "liouville",
            "dissipation_mode": "dissipative",
            "H0_snapshots": 100,
            "Omega_R_snapshots": 1,
            "algorithm": "qiskit-cobyla",
            "max_iter": 1000,
            "targ_fid": 0.99
        }
    }

    api = CtrlFreeQAPI(config)
    solution = api.run_optimization()
    print(f"Final fidelity: {api.parameters.final_fidelity}")
    ```

!!! note "Dissipation Requirements"
    When `dissipation_mode` is set to `"dissipative"`, the following conditions must be satisfied:

    - The optimization `space` must be `"liouville"` (density matrix mode)
    - Per-qubit `T1` and `T2` values must be provided in the `parameters` section
    - The physical constraint \(T_2 \leq 2\,T_1\) must hold for each qubit

    For a detailed description of the dissipation parameters, see [Optimization → Parameters → Dissipation Parameters](optimization/parameters.md#dissipation-parameters).

---

## Hamiltonian Models

Ctrl-freeq provides a platform-agnostic Hamiltonian model abstraction for defining quantum control problems. All models implement the `HamiltonianModel` abstract base class, which follows the standard bilinear control formulation:

$$
H(t) = H_\text{drift} + \sum_k u_k(t) \, H_{\text{ctrl},k}
$$

The framework uses a **plugin architecture** based on a model registry. New Hamiltonian types can be added by subclassing `HamiltonianModel` and decorating with `@register_hamiltonian` — no framework files need to be modified.

### HamiltonianModel (Abstract Base Class)

```python
from ctrl_freeq.setup.hamiltonian_generation import HamiltonianModel
```

All Hamiltonian models implement the following interface:

| Method / Property | Description |
|-------------------|-------------|
| `build_drift(frequency_instances, coupling_instances=None, **kwargs)` | Construct drift Hamiltonian \(H_0\) snapshots (list of \((D, D)\) numpy arrays) |
| `build_control_ops()` | Return fixed control operators \(H_{\text{ctrl},k}\) (list of \((D, D)\) numpy arrays) |
| `control_amplitudes(cx, cy, rabi_freq, n_h0)` | Map waveform outputs to control amplitudes \(u_k(t)\) |
| `from_config(n_qubits, params)` (classmethod) | Construct a model instance from a configuration dictionary |
| `default_config(n_qubits)` (classmethod) | Return a complete runnable configuration dictionary |
| `dim` (property) | Hilbert space dimension \(D = 2^{n_\text{qubits}}\) |
| `n_controls` (property) | Number of independent control channels (\(2 \times n_\text{qubits}\)) |
| `control_ops_tensor()` | Control operators as a stacked torch tensor \((n_\text{controls}, D, D)\) |

The `build_drift` method uses a **standardized signature** across all models: `frequency_instances` contains per-qubit frequency arrays (detunings, qubit frequencies, etc.), and `coupling_instances` contains coupling matrices. Model-specific parameters are passed via `**kwargs`.

### Model Registry

The registry provides automatic discovery and lookup of Hamiltonian models:

```python
from ctrl_freeq.setup.hamiltonian_generation import (
    register_hamiltonian,
    get_hamiltonian_class,
    list_hamiltonians,
)

# List all registered models
print(list_hamiltonians())  # ['spin_chain', 'superconducting']

# Look up a model class by name
cls = get_hamiltonian_class("spin_chain")
model = cls.from_config(n_qubits=2, params={"coupling_type": "XY"})
```

| Function | Description |
|----------|-------------|
| `register_hamiltonian(name)` | Decorator that registers a `HamiltonianModel` subclass under the given name |
| `get_hamiltonian_class(name)` | Look up and return the model class for a given name |
| `list_hamiltonians()` | Return a sorted list of all registered model names |

### SpinChainModel

Models spin-chain qubits with detuning-based drift and configurable inter-qubit coupling (Ising, XY, Heisenberg).

```python
from ctrl_freeq.setup.hamiltonian_generation import SpinChainModel

model = SpinChainModel(n_qubits=2, coupling_type="XY")

# Build drift Hamiltonian
import numpy as np
Delta = np.array([2 * np.pi * 1e7, 2 * np.pi * 2e7])
J = np.array([[0.0, 2 * np.pi * 1e6], [0.0, 0.0]])
H0_list = model.build_drift(frequency_instances=[Delta], coupling_instances=[J])

# Get control operators
ctrl_ops = model.build_control_ops()  # [X_0, Y_0, X_1, Y_1]
```

| Parameter | Description |
|-----------|-------------|
| `n_qubits` | Number of spin-1/2 qubits |
| `coupling_type` | `"Z"` (Ising), `"XY"` (exchange), or `"XYZ"` (Heisenberg) |

### SuperconductingQubitModel

Models fixed-frequency transmon qubits with capacitive coupling, supporting exchange (XY), static ZZ, and combined (XY+ZZ) coupling.

```python
from ctrl_freeq.setup.hamiltonian_generation import SuperconductingQubitModel

model = SuperconductingQubitModel(
    n_qubits=2,
    coupling_type="XY+ZZ",
    anharmonicities=np.array([2 * np.pi * -330e6, 2 * np.pi * -330e6]),
)

# Build drift Hamiltonian
omega = np.array([2 * np.pi * 5e9, 2 * np.pi * 5.2e9])
g = np.array([[0.0, 2 * np.pi * 5e6], [0.0, 0.0]])
H0_list = model.build_drift(frequency_instances=[omega], coupling_instances=[g])
```

| Parameter | Description |
|-----------|-------------|
| `n_qubits` | Number of transmon qubits |
| `coupling_type` | `"XY"` (exchange), `"ZZ"` (static ZZ), or `"XY+ZZ"` (both) |
| `anharmonicities` | Per-qubit anharmonicities (rad/s), used for ZZ computation |

### Using Models in Configuration

To use a Hamiltonian model, set the `hamiltonian_type` field in the configuration:

```python
config = {
    "qubits": ["q1", "q2"],
    "hamiltonian_type": "superconducting",  # or "spin_chain"
    "compute_resource": "cpu",
    "parameters": { ... },
    "initial_states": [["Z", "Z"]],
    "target_states": {"Gate": ["iSWAP"]},
    "optimization": { ... }
}
```

### Default Configurations

Each model provides a `default_config` classmethod that returns a complete, ready-to-run configuration dictionary with sensible physical defaults. This is useful for quick experiments and will also serve as the basis for GUI form population.

```python
from ctrl_freeq.setup.hamiltonian_generation import SpinChainModel, SuperconductingQubitModel
from ctrl_freeq.api import CtrlFreeQAPI

# Get a complete spin-chain config for 2 qubits (CNOT gate, XY coupling)
config = SpinChainModel.default_config(n_qubits=2)
api = CtrlFreeQAPI(config)

# Get a complete superconducting config for 2 qubits (iSWAP gate, XY coupling)
config = SuperconductingQubitModel.default_config(n_qubits=2)
api = CtrlFreeQAPI(config)
```

### Adding Custom Hamiltonian Models

A user who wants to add a new Hamiltonian needs to do just **one thing**: write a Python class that subclasses `HamiltonianModel` and decorate it with `@register_hamiltonian("name")`. No framework files need to be touched.

#### Step 1 — Define the model class

Implement six required members (`build_drift`, `build_control_ops`, `control_amplitudes`, `from_config`, `dim`, `n_controls`) and optionally `default_config`:

```python
from ctrl_freeq.setup.hamiltonian_generation import HamiltonianModel, register_hamiltonian
import numpy as np

@register_hamiltonian("trapped_ion")
class TrappedIonModel(HamiltonianModel):
    def __init__(self, n_qubits, laser_wavelength=729e-9):
        self.n_qubits = n_qubits
        self.laser_wavelength = laser_wavelength
        # ... pre-compute operators, Lamb-Dicke params, etc.

    @property
    def dim(self):
        return 2 ** self.n_qubits

    @property
    def n_controls(self):
        return 2 * self.n_qubits  # I, Q per qubit

    def build_drift(self, frequency_instances, coupling_instances=None, **kwargs):
        # frequency_instances: list of arrays with per-qubit trap frequencies
        # coupling_instances: list of coupling matrices (Mølmer-Sørensen, etc.)
        # Return: list of (D, D) numpy arrays — one per snapshot
        ...

    def build_control_ops(self):
        # Return list of (D, D) numpy arrays: [X_0, Y_0, X_1, Y_1, ...]
        ...

    def control_amplitudes(self, cx, cy, rabi_freq, n_h0):
        # Map waveform (cx, cy) and Rabi frequency to control amplitudes u_k(t)
        ...

    @classmethod
    def from_config(cls, n_qubits, params):
        # Extract model-specific constructor args from the config dict
        return cls(n_qubits, laser_wavelength=params.get("laser_wavelength", 729e-9))

    @classmethod
    def default_config(cls, n_qubits):
        # Optional: return a complete, ready-to-run config with sensible defaults
        return {
            "hamiltonian_type": "trapped_ion",
            "qubits": [f"q{i+1}" for i in range(n_qubits)],
            "parameters": { ... },
            "initial_states": [["Z", "Z"]],
            "target_states": {"Gate": ["XX"]},
            "optimization": {"algorithm": "l-bfgs", "max_iter": 300, "targ_fid": 0.999, ...},
        }
```

#### Step 2 — Make sure the class is imported

The `@register_hamiltonian` decorator fires at import time, so the module just needs to be imported before the config is loaded. Define it in the same script that runs the optimization, or import it from your own package.

#### Step 3 — Use it

Two workflows are available:

=== "Via registry (main path)"
    Reference the model by name in any configuration dictionary or JSON file:

    ```python
    from ctrl_freeq.api import CtrlFreeQAPI

    # Use default_config for zero-effort setup
    config = TrappedIonModel.default_config(n_qubits=2)
    # or: config = {"hamiltonian_type": "trapped_ion", ...}

    api = CtrlFreeQAPI(config)
    solution = api.run_optimization()
    ```

=== "Via direct injection (quick experiments)"
    For one-off experiments, inject a model instance directly without registration:

    ```python
    model = TrappedIonModel(n_qubits=2, laser_wavelength=729e-9)
    api = CtrlFreeQAPI(config_dict, hamiltonian_model=model)
    solution = api.run_optimization()
    ```

    The injected model overrides any `hamiltonian_type` specified in the configuration.

!!! tip "What you get for free"
    Everything else in the pipeline works automatically — waveform parameterisation, gradient computation (autograd), fidelity evaluation, plotting, and dashboards — because it all goes through the generic `pulse_hamiltonian_generic` path using `build_control_ops` and `control_amplitudes`. The user only defines the *physics*; the *optimisation machinery* is inherited.

---

## Configuration Schema

For a detailed treatment of all configuration parameters, the reader is referred to:

- [Optimization Parameters](optimization/parameters.md) — Full parameter reference
- [Algorithms](optimization/algorithms.md) — Available optimization algorithms
- [Compute (CPU/GPU)](optimization/compute.md) — CPU and GPU configuration

---

## Error Handling

```python
from ctrl_freeq.api import CtrlFreeQAPI

try:
    api = CtrlFreeQAPI("nonexistent.json")
except FileNotFoundError as e:
    print(f"Configuration file not found: {e}")

try:
    api.update_parameter("invalid.path", 100)
except KeyError as e:
    print(f"Invalid parameter path: {e}")
```

---

## Next Steps

- [GUI Guide](gui.md) — Configure and run optimization interactively
- [Dashboard](dashboard.md) — Visualize and export results
- [Optimization Parameters](optimization/parameters.md) — Detailed parameter reference
