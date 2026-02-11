# Optimization Parameters

ctrl-freeq uses a JSON configuration with a consistent schema across API and GUI.

---

## Configuration Schema

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

Physical parameters of the quantum system. All are **per-qubit arrays**.

| Key | GUI Label | Description | Default | Units |
|-----|-----------|-------------|---------|-------|
| `Delta` | Δ (Hz) | Detuning frequency | 10 MHz | Hz |
| `sigma_Delta` | σΔ (Hz) | Detuning uncertainty (for robustness) | 0 | Hz |
| `Omega_R_max` | Ω_R Max (Hz) | Maximum Rabi frequency | 40 MHz | Hz |
| `sigma_Omega_R_max` | σΩ_R Max (Hz) | Rabi frequency uncertainty | 0 | Hz |

---

## Pulse Parameters

Parameters controlling pulse shape and timing. All are **per-qubit arrays**.

| Key | GUI Label | Description | Default | Units |
|-----|-----------|-------------|---------|-------|
| `pulse_duration` | Pulse Duration (sec) | Total pulse length | 200 ns | seconds |
| `point_in_pulse` | Point in Pulse | Time discretization points | 100 | — |
| `sw` | Sweep Rate (Hz) | Frequency sweep rate | 5 MHz | Hz |
| `pulse_offset` | Pulse Offset (Hz) | Frequency offset | 0 | Hz |
| `pulse_bandwidth` | Pulse Bandwidth (Hz) | Pulse frequency bandwidth | 500 kHz | Hz |

---

## Waveform Settings

Parameters controlling waveform basis and envelope. All are **per-qubit arrays**.

| Key | GUI Label | Options | Default | Description |
|-----|-----------|---------|---------|-------------|
| `wf_type` | Waveform Type | See below | `cheb` | Basis function type |
| `wf_mode` | Waveform Mode | `cart`, `polar`, `polar_phase` | `cart` | Cartesian or polar representation |
| `amplitude_envelope` | Amplitude Envelope | `gn`, `rect`, `sinc` | `gn` | Pulse envelope shape |
| `amplitude_order` | Amplitude Order | Integer | 1 | Envelope order parameter |
| `coverage` | Coverage | `single`, `broadband` | `single` | Frequency coverage mode |
| `profile_order` | Profile Order | Integer | 2 | Supergaussian order for selectivity profile (1 = Gaussian, higher values → more rectangular) |
| `n_para` | Number of Parameters | Integer | 16 | Optimization parameters per qubit |

!!! info "Waveform Types"
    - **cheb** — Chebyshev polynomial basis (recommended for most cases)
    - **fou** — Fourier series basis
    - **poly** — Standard polynomial basis
    - **leg** — Legendre polynomial basis
    - **hermite** — Hermite polynomial basis
    - **gegen** — Gegenbauer polynomial basis
    - **chirp** — Chirp basis

!!! info "Waveform Modes"
    - **cart** — Cartesian mode: optimizes I (in-phase) and Q (quadrature) components
    - **polar** — Polar mode: optimizes amplitude and phase directly
    - **polar_phase** — Polar phase mode: optimizes phase with fixed amplitude profile

---

## Multi-Qubit Coupling

These fields are used when `len(qubits) > 1`:

| Key | GUI Label | Description                      | Default |
|-----|-----------|----------------------------------|---------|
| `J` | J_ij | Coupling matrix (N×N), symmetric | ~16.7 MHz |
| `coupling_type` | Coupling Type | `Z`, `XY`, or `XYZ`              | `XY` |
| `sigma_J` | σ J | Coupling strength uncertainty    | 0 |

!!! info "Coupling Types"
    - **Z** — Ising-type ZZ coupling only
    - **XY** — XX + YY exchange coupling (default)
    - **XYZ** — Full Heisenberg coupling (XX + YY + ZZ)

---

## Initial States

List of initial state specifications per qubit.

```jsonc
// Single qubit
{ "initial_states": [["Z"]] }

// Two qubits
{ "initial_states": [["Z", "Z"]] }
```

---

## Target States

Three methods for specifying target states:

### Axis Targets

Specify target as Bloch sphere axis direction.

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

Specify a target quantum gate.

```jsonc
{ "target_states": { "Gate": ["X", "H", "CNOT"] } }
```

**Single-qubit:** `X`, `Y`, `Z`, `H`, `S`, `T`

**Two-qubit:** `CNOT`, `CZ`, `SWAP`, `iSWAP`

**Three-qubit:** `Toff` (Toffoli)

### Phi/Beta Targets

Specify rotation angles.

```jsonc
{ "target_states": { "Phi": ["x"], "Beta": ["180"] } }
```

| Parameter | Description | Range |
|-----------|-------------|-------|
| Phi (φ) | Rotation axis in XY plane | `x`, `y`, or angle in degrees |
| Beta (β) | Rotation angle | 0° – 360° |

---

## Multi-State Optimization

ctrl-freeq supports optimizing pulses for **multiple initial/target state pairs simultaneously**. This is essential for designing:

- **Universal rotation pulses** (single qubit)
- **Conditional gates** like CNOT, SWAP (multi-qubit)

The optimizer finds a single pulse that achieves all specified state transformations.

### Single-Qubit Example: Universal Rotation

To design a pulse that performs a universal rotation, specify multiple initial→target mappings:

```jsonc
{
  "initial_states": [["X"], ["Y"], ["Z"]],
  "target_states": {
    "Axis": [["X"], ["Z"], ["-Y"]]
  }
}
```

This optimizes a pulse that simultaneously achieves:

| Initial | Target | Transformation |
|---------|--------|----------------|
| X | X | X axis preserved |
| Y | Z | Y → Z rotation |
| Z | -Y | Z → -Y rotation |

### Multi-Qubit Example: Conditional Gates

For two-qubit systems, use multiple state pairs to design conditional gates:

```jsonc
{
  "initial_states": [["Z", "-Z"], ["X", "Y"]],
  "target_states": {
    "Gate": ["CNOT", "SWAP"]
  }
}
```

!!! tip "State Pair Matching"
    The number of initial states must match the number of target states. Each initial state at index `i` maps to the target state at index `i`.

### GUI Usage

In the GUI, enter comma-separated values for multiple states:

- **Initial States**: `X, Y, Z`
- **Target Axis**: `X, Z, -Y`

For two-qubit systems, use semicolons to separate qubits within each state pair.

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

| Key | GUI Label | Description | Default |
|-----|-----------|-------------|---------|
| `space` | Space | `hilbert` (pure states) or `liouville` (density matrices) | `hilbert` |
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
