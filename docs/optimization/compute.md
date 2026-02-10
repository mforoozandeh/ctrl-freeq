# Compute (CPU/GPU)

ctrl-freeq supports running optimization on **CPU** or **GPU** via the `compute_resource` configuration key.

---

## Configuration

Top-level config field:

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

When running on CPU, ctrl-freeq automatically manages thread allocation:

- **Default:** Uses `total_cores - 1` threads (leaves one core free for system responsiveness)
- **Custom:** You can set the `cpu_cores` field in your configuration to specify the number of threads

```json
{
  "compute_resource": "cpu",
  "cpu_cores": 4
}
```

The requested value is clamped to `[1, total_cores]` to prevent invalid settings.

---

## GPU Configuration

GPU support requires a CUDA-enabled PyTorch installation.

### Setup

1. Ensure you have a CUDA-compatible NVIDIA GPU
2. Install PyTorch with CUDA support (see [Installation](../installation.md#gpu-support-optional))
3. Set `compute_resource` to `"gpu"` in your configuration

### Automatic Fallback

If `"gpu"` is requested but CUDA is not available, ctrl-freeq **automatically falls back to CPU** and logs a warning:

```
WARNING: CUDA not available; falling back to CPU. To use GPU, run on a CUDA-enabled environment.
```

This ensures your optimization runs will always execute, even without a GPU.

### Verifying GPU Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

---

## GUI

Use **Optimization Parameters > Compute Resource** to select `cpu` or `gpu`.

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
