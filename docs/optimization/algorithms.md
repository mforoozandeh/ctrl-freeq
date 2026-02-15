# Optimization Algorithms

Ctrl-freeq provides access to multiple optimization algorithms through the `algorithm` configuration field. These algorithms are drawn from two distinct families, each offering complementary capabilities for the solution of quantum control problems.

---

## Algorithm Families

The available algorithms are organized into two families. The first comprises gradient-based methods implemented via the [pytorch-minimize](https://github.com/rfeinman/pytorch-minimize) library, which operates directly on PyTorch tensors and supports automatic differentiation for the computation of gradients and, where applicable, Hessians. The second family consists of wrapped Qiskit optimizers, identified by the `qiskit-` prefix, which include both gradient-based and derivative-free methods.

---

## Built-in Algorithms

| Algorithm | Description |
|-----------|-------------|
| `bfgs` | Broyden-Fletcher-Goldfarb-Shanno (quasi-Newton) |
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

The Qiskit optimizers are wrapped in `ctrl_freeq/optimizers/qiskit_optimizers.py`.

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
    The exact set of available Qiskit optimizers is determined by the `get_supported_qiskit_optimizers()` function and may vary with the installed version.

---

## Choosing an Algorithm

The selection of an appropriate optimization algorithm depends on the characteristics of the problem at hand, the dimensionality of the parameter space, and the available computational resources.

For general-purpose applications, the BFGS or L-BFGS algorithms provide reliable quasi-Newton methods that are well-suited to the majority of pulse design problems; the latter is particularly advantageous when the parameter space is large, owing to its reduced memory requirements. For smooth, well-behaved cost landscapes, the Newton-CG method often exhibits faster convergence due to its use of curvature information.

In cases where derivative-free optimization is preferred or the problem involves constraints, COBYLA (`qiskit-cobyla`) offers a robust alternative that does not require gradient computation. For noisy or non-convex optimization landscapes, stochastic methods such as Adam (`qiskit-adam`) or SPSA (`qiskit-spsa`) may prove more effective, as they are designed to navigate irregular objective function surfaces.

When high numerical precision is required and the computational cost of exact Hessian evaluation is acceptable, the `trust-exact` or `newton-exact` algorithms should be considered, as they exploit full second-order information to achieve rapid convergence in the neighbourhood of a minimum.

---

## Convergence Parameters

The following parameters control the termination criteria of the optimization:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_iter` | Maximum optimization iterations | 1000 |
| `targ_fid` | Target fidelity threshold (stops when reached) | 0.999 |

---

## Next Steps

- [Parameters](parameters.md) — Full configuration reference
- [Compute (CPU/GPU)](compute.md) — Hardware acceleration options
