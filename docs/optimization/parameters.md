# Optimization Parameters

Ctrl-freeq employs a JSON-based configuration schema that is shared between the programmatic API and the graphical user interface. This section provides a complete reference for all configurable parameters, organized by category.

---

## Configuration Schema

The top-level structure of a configuration is as follows:

```jsonc
{
  "qubits": ["q1", "q2"],
  "compute_resource": "cpu",
  "parameters": { ... },
  "initial_states": [["Z", "Z"]],
  "target_states": { ... },
  "optimization": { ... }
}
```

| Field | Description |
|-------|-------------|
| `qubits` | List of qubit identifiers for display and indexing |
| `compute_resource` | `"cpu"` or `"gpu"` (see [Compute](compute.md)) |

---

## System Parameters

The system parameters define the physical properties of the quantum system under consideration. All values are specified as **per-qubit arrays**, with one entry for each qubit in the configuration.

| Key | GUI Label | Description | Default | Units |
|-----|-----------|-------------|---------|-------|
| `Delta` | Δ (Hz) | Detuning frequency | 10 MHz | Hz |
| `sigma_Delta` | σΔ (Hz) | Detuning uncertainty (for robustness) | 0 | Hz |
| `Omega_R_max` | Ω_R Max (Hz) | Maximum Rabi frequency | 40 MHz | Hz |
| `sigma_Omega_R_max` | σΩ_R Max (Hz) | Rabi frequency uncertainty | 0 | Hz |

---

## Pulse Parameters

The pulse parameters control the temporal characteristics of the control waveform, including its duration, discretization, and frequency properties. As with the system parameters, all values are **per-qubit arrays**.

| Key | GUI Label | Description | Default | Units |
|-----|-----------|-------------|---------|-------|
| `pulse_duration` | Pulse Duration (sec) | Total pulse length | 200 ns | seconds |
| `point_in_pulse` | Point in Pulse | Time discretization points | 100 | — |
| `sw` | Sweep Rate (Hz) | Frequency sweep rate | 5 MHz | Hz |
| `pulse_offset` | Pulse Offset (Hz) | Frequency offset | 0 | Hz |
| `pulse_bandwidth` | Pulse Bandwidth (Hz) | Pulse frequency bandwidth | 500 kHz | Hz |

---

## Waveform Settings

The waveform settings determine the basis functions and envelope used to parameterize the control pulse. The choice of basis and mode has a significant influence on the smoothness, bandwidth, and convergence properties of the optimized waveform. All values are **per-qubit arrays**.

| Key | GUI Label | Options | Default | Description |
|-----|-----------|---------|---------|-------------|
| `wf_type` | Waveform Type | See below | `cheb` | Basis function type |
| `wf_mode` | Waveform Mode | `cart`, `polar`, `polar_phase` | `cart` | Cartesian or polar representation |
| `amplitude_envelope` | Amplitude Envelope | `gn`, `rect`, `sinc` | `gn` | Pulse envelope shape |
| `amplitude_order` | Amplitude Order | Integer | 1 | Envelope order parameter |
| `coverage` | Coverage | `single`, `broadband`, `selective`, `band_selective` | `single` | Frequency coverage mode |
| `profile_order` | Profile Order | Integer | 2 | Supergaussian order for selectivity profile (1 = Gaussian, higher values → more rectangular) |
| `n_para` | Number of Parameters | Integer | 16 | Optimization parameters per qubit |

!!! info "Waveform Types"
    Several families of orthogonal basis functions are available for the parameterization of the control waveform:

    - **cheb** — Chebyshev polynomial basis (recommended for most applications due to favourable convergence properties)
    - **fou** — Fourier series basis
    - **poly** — Standard polynomial basis
    - **leg** — Legendre polynomial basis
    - **hermite** — Hermite polynomial basis
    - **gegen** — Gegenbauer polynomial basis
    - **chirp** — Chirp basis

!!! info "Waveform Modes"
    The waveform representation may be specified in one of three modes:

    - **cart** — Cartesian mode, in which the in-phase (I) and quadrature (Q) components are optimized independently
    - **polar** — Polar mode, in which the amplitude and phase are optimized directly
    - **polar_phase** — Polar phase mode, in which only the phase is optimized while the amplitude profile remains fixed

---

## Dissipation Parameters

For the simulation of open quantum systems, ctrl-freeq supports dissipative dynamics via the Lindblad master equation. When dissipation is enabled, the evolution includes amplitude damping (characterised by the longitudinal relaxation time T1) and pure dephasing (characterised by the transverse relaxation time T2) for each qubit. All values are **per-qubit arrays**, with one entry for each qubit in the configuration.

| Key | GUI Label | Description | Default | Units |
|-----|-----------|-------------|---------|-------|
| `T1` | T1 (s) | Amplitude damping time constant (energy relaxation) | — | seconds |
| `T2` | T2 (s) | Pure dephasing time constant (phase decoherence) | — | seconds |

The Lindblad collapse operators constructed from these parameters are:

- **Amplitude damping:** $L_1 = \sqrt{1/T_1}\, \sigma^-$ — models energy relaxation to the ground state
- **Pure dephasing:** $L_2 = \sqrt{1/T_2 - 1/(2T_1)}\, \sigma_z/2$ — models loss of phase coherence without energy exchange

For multi-qubit systems, the collapse operators are extended to the full Hilbert space via tensor products with identity operators on the remaining qubits.

!!! warning "Physical Constraint"
    The physical constraint $T_2 \leq 2 T_1$ is enforced automatically. Configurations that violate this bound will raise a validation error, as such values are unphysical (the pure dephasing rate cannot be negative).

!!! note "Liouville Space Requirement"
    Dissipative simulations require the optimization space to be set to `liouville` (density matrix mode). When `dissipation_mode` is set to `"dissipative"`, the space is automatically forced to `liouville` in the GUI.

**Example:**

```jsonc
{
  "parameters": {
    // ... other parameters ...
    "T1": [0.001],       // 1 ms amplitude damping
    "T2": [0.0005]       // 500 μs pure dephasing (T2 ≤ 2·T1)
  },
  "optimization": {
    "space": "liouville",
    "dissipation_mode": "dissipative",
    // ...
  }
}
```

---

## Multi-Qubit Coupling

For systems comprising two or more qubits, the inter-qubit coupling parameters must be specified. These fields are used when `len(qubits) > 1`:

| Key | GUI Label | Description                      | Default |
|-----|-----------|----------------------------------|---------|
| `J` | J_ij | Coupling matrix (N×N), symmetric | ~16.7 MHz |
| `coupling_type` | Coupling Type | `Z`, `XY`, or `XYZ`              | `XY` |
| `sigma_J` | σ J | Coupling strength uncertainty    | 0 |

!!! info "Coupling Types"
    Three coupling models are supported, corresponding to common interaction Hamiltonians encountered in quantum information processing:

    - **Z** — Ising-type ZZ coupling only
    - **XY** — XX + YY exchange coupling (default)
    - **XYZ** — Full Heisenberg coupling (XX + YY + ZZ)

---

## Initial States

The initial state of each qubit is specified as a list of Bloch sphere axis labels. For multi-qubit systems, each entry in the outer list corresponds to a distinct initial configuration of the entire register.

```jsonc
// Single qubit
{ "initial_states": [["Z"]] }

// Two qubits
{ "initial_states": [["Z", "Z"]] }
```

---

## Target States

The desired final state or gate may be specified using one of three methods, each suited to different use cases.

### Axis Targets

The target state is specified as a Bloch sphere axis direction.

```jsonc
{ "target_states": { "Axis": [["-Z"]] } }
```

| Value | Description |
|-------|-------------|
| `Z` | Ground state (\|0⟩) |
| `-Z` | Excited state (\|1⟩) |
| `X`, `-X` | Superposition along ±X |
| `Y`, `-Y` | Superposition along ±Y |

### Gate Targets

The target is specified as a quantum gate, which implicitly defines the required unitary transformation.

```jsonc
{ "target_states": { "Gate": ["X", "H", "CNOT"] } }
```

**Single-qubit:** `X`, `Y`, `Z`, `H`, `S`, `T`

**Two-qubit:** `CNOT`, `CZ`, `SWAP`, `iSWAP`

**Three-qubit:** `Toff` (Toffoli)

### Phi/Beta Targets

The target rotation is specified in terms of the rotation axis and angle.

```jsonc
{ "target_states": { "Phi": ["x"], "Beta": ["180"] } }
```

| Parameter | Description | Range |
|-----------|-------------|-------|
| Phi (φ) | Rotation axis in XY plane | `x`, `y`, or angle in degrees |
| Beta (β) | Rotation angle | 0° – 360° |

---

## Multi-State Optimization

Ctrl-freeq supports the simultaneous optimization of pulses for multiple initial/target state pairs. This capability is essential for the design of universal rotation pulses (in the single-qubit case) and conditional gates such as CNOT or SWAP (in the multi-qubit case). The optimizer seeks a single pulse that achieves all specified state transformations concurrently.

### Single-Qubit Example: Universal Rotation

To design a pulse that performs a universal rotation, multiple initial-to-target state mappings are specified:

```jsonc
{
  "initial_states": [["X"], ["Y"], ["Z"]],
  "target_states": {
    "Axis": [["X"], ["Z"], ["-Y"]]
  }
}
```

The optimization then seeks a single pulse that simultaneously achieves the following transformations:

| Initial | Target | Transformation |
|---------|--------|----------------|
| X | X | X axis preserved |
| Y | Z | Y → Z rotation |
| Z | -Y | Z → -Y rotation |

### Multi-Qubit Example: Conditional Gates

For two-qubit systems, multiple state pairs may be specified to design conditional gates:

```jsonc
{
  "initial_states": [["Z", "-Z"], ["X", "Y"]],
  "target_states": {
    "Gate": ["CNOT", "SWAP"]
  }
}
```

!!! tip "State Pair Matching"
    It should be noted that the number of initial states must equal the number of target states. Each initial state at index `i` is mapped to the target state at the corresponding index.

### GUI Usage

In the graphical interface, multiple states are entered as comma-separated values:

- **Initial States**: `X, Y, Z`
- **Target Axis**: `X, Z, -Y`

For two-qubit systems, semicolons are used to separate qubits within each state pair.

### API Usage

```python
from ctrl_freeq.api import load_single_qubit_multiple_config

# Load pre-configured multi-state example
api = load_single_qubit_multiple_config()
solution = api.run_optimization()
```

Available multi-state configurations:

| Function | Description |
|----------|-------------|
| `load_single_qubit_multiple_config()` | Single qubit with X,Y,Z → X,Z,-Y |
| `load_two_qubit_multiple_config()` | Two qubits with multiple gate targets |

---

## Optimization Settings

The following fields control the behaviour of the optimization procedure:

| Key | GUI Label | Description | Default |
|-----|-----------|-------------|---------|
| `space` | Space | `hilbert` (pure states) or `liouville` (density matrices) | `hilbert` |
| `dissipation_mode` | Dissipation | `non-dissipative` or `dissipative` (Lindblad master equation) | `non-dissipative` |
| `H0_snapshots` | H₀ Snapshots | Time steps for drift Hamiltonian | 100 |
| `Omega_R_snapshots` | Ω_R Snapshots | Time steps for control Hamiltonian | 1 |
| `algorithm` | Algorithm | Optimization algorithm (see [Algorithms](algorithms.md)) | varies by config |
| `max_iter` | Max Iterations | Maximum optimization iterations | 1000 |
| `targ_fid` | Target Fidelity | Stop when fidelity reaches this value | 0.999 |

---

## Next Steps

- [Algorithms](algorithms.md) — Algorithm selection guide
- [Compute (CPU/GPU)](compute.md) — Hardware acceleration options
- [GUI Guide](../gui.md) — Interactive configuration
