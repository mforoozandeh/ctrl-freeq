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
CtrlFreeQAPI(config: Union[str, Path, Dict[str, Any]])
```

An API instance may be created from a path to a JSON configuration file (provided as a string or `Path` object) or from a dictionary containing the configuration parameters directly.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str`, `Path`, or `dict` | Configuration source |

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
| `optimization.algorithm` | Optimization algorithm |
| `optimization.max_iter` | Maximum iterations |
| `optimization.targ_fid` | Target fidelity |
| `optimization.space` | `hilbert` or `liouville` |
| `parameters.Delta` | Detuning (Hz) |
| `parameters.Omega_R_max` | Maximum Rabi frequency (Hz) |
| `parameters.pulse_duration` | Pulse duration (seconds) |
| `parameters.point_in_pulse` | Discretization points |
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

### `run_from_config(config)`

A convenience function that creates a `CtrlFreeQAPI` instance and executes the optimization in a single call.

```python
from ctrl_freeq.api import run_from_config

solution = run_from_config("path/to/config.json")
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | `str`, `Path`, or `dict` | Configuration source |

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
