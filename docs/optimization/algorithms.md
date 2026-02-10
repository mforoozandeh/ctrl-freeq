# Optimization Algorithms

ctrl-freeq supports multiple optimization algorithms through the `algorithm` configuration field.

---

## Algorithm Families

There are two families of algorithms:

1. **Built-in algorithms** — Gradient-based methods via [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) (runs on PyTorch tensors, supports automatic differentiation)
2. **Qiskit optimizers** — Wrapped Qiskit algorithms (prefixed with `qiskit-`)

---

## Built-in Algorithms

| Algorithm | Description |
|-----------|-------------|
| `bfgs` | Broyden–Fletcher–Goldfarb–Shanno (quasi-Newton) |
| `l-bfgs` | Limited-memory BFGS (lower memory usage) |
| `cg` | Conjugate gradient |
| `newton-cg` | Newton conjugate gradient |
| `newton-exact` | Newton with exact Hessian |
| `dogleg` | Dogleg trust-region |
| `trust-ncg` | Trust-region Newton-CG |
| `trust-krylov` | Trust-region with Krylov subspace |
| `trust-exact` | Trust-region with exact Hessian |

---

## Qiskit Algorithms

Qiskit optimizers are wrapped in `ctrl_freeq/optimizers/qiskit_optimizers.py`.

| Algorithm | Description |
|-----------|-------------|
| `qiskit-cobyla` | Constrained Optimization BY Linear Approximation |
| `qiskit-nelder-mead` | Nelder-Mead simplex (derivative-free) |
| `qiskit-powell` | Powell's method (derivative-free) |
| `qiskit-slsqp` | Sequential Least Squares Programming |
| `qiskit-spsa` | Simultaneous Perturbation Stochastic Approximation |
| `qiskit-adam` | Adam optimizer |
| `qiskit-l-bfgs-b` | L-BFGS-B with bounds |
| `qiskit-tnc` | Truncated Newton |
| `qiskit-p-bfgs` | Parallel L-BFGS-B |
| `qiskit-gsls` | Gaussian Smoothed Line Search |

!!! note
    The exact set available depends on `get_supported_qiskit_optimizers()` in your installed version.

---

## Choosing an Algorithm

!!! tip "General-Purpose"
    **`bfgs`** or **`l-bfgs`** — Good default choices for most problems. Use `l-bfgs` for larger parameter spaces to reduce memory usage.

!!! tip "Smooth Problems"
    **`newton-cg`** — Often achieves faster convergence for smooth, well-behaved cost landscapes.

!!! tip "Constrained Problems"
    **`qiskit-cobyla`** — Robust derivative-free method, good for constrained optimization.

!!! tip "Noisy Landscapes"
    **`qiskit-adam`** or **`qiskit-spsa`** — Useful for noisy or non-convex optimization landscapes.

!!! tip "High-Precision"
    **`trust-exact`** or **`newton-exact`** — When you need high precision and can afford exact Hessian computation.

---

## Convergence Parameters

These parameters control optimization termination:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_iter` | Maximum optimization iterations | 1000 |
| `targ_fid` | Target fidelity threshold (stops when reached) | 0.999 |

---

## Next Steps

- [Parameters](parameters.md) — Full configuration reference
- [Compute (CPU/GPU)](compute.md) — Hardware acceleration options
