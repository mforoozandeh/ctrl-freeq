# Compute (CPU/GPU)

Ctrl-freeq supports the execution of optimization routines on both CPU and GPU hardware. The computational backend is selected through the `compute_resource` configuration field, which determines whether tensor operations are performed on the host processor or offloaded to a CUDA-compatible graphics processing unit.

---

## Configuration

The compute resource is specified as a top-level field in the configuration:

```json
{
  "compute_resource": "cpu"
}
```

Valid values:

- `"cpu"` (default)
- `"gpu"` (CUDA)

---

## CPU Configuration

When the optimization is executed on CPU, thread allocation is managed automatically by the framework. By default, the number of PyTorch threads is set to one fewer than the total number of available CPU cores, thereby reserving one core for system responsiveness. This default behaviour may be overridden by specifying the `cpu_cores` field in the configuration, in which case the requested value is clamped to the interval [1, total_cores] to prevent invalid settings.

```json
{
  "compute_resource": "cpu",
  "cpu_cores": 4
}
```

---

## GPU Configuration

GPU acceleration requires a CUDA-enabled PyTorch installation and a compatible NVIDIA graphics processing unit.

### Setup

1. Ensure that a CUDA-compatible NVIDIA GPU is available
2. Install PyTorch with CUDA support (see [Installation](../installation.md#gpu-support-optional))
3. Set `compute_resource` to `"gpu"` in the configuration

### Automatic Fallback

If GPU execution is requested but CUDA is not available on the host system, ctrl-freeq automatically falls back to CPU execution and emits a warning:

```
WARNING: CUDA not available; falling back to CPU. To use GPU, run on a CUDA-enabled environment.
```

This fallback mechanism ensures that optimization runs will always execute, regardless of the available hardware.

### Verifying GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

---

## GUI

The compute resource may also be selected from the **Optimization Parameters > Compute Resource** dropdown in the graphical interface.

---

## API Usage

```python
from ctrl_freeq.api import load_single_qubit_config

api = load_single_qubit_config()

# Switch to GPU
api.update_parameter("compute_resource", "gpu")

# Run optimization
solution = api.run_optimization()
```

---

## Next Steps

- [Algorithms](algorithms.md) — Algorithm selection guide
- [Parameters](parameters.md) — Full configuration reference
- [Installation](../installation.md#gpu-support-optional) — GPU setup instructions
