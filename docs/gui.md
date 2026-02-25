# GUI Guide

In addition to the programmatic API, ctrl-freeq provides a graphical user interface, implemented using the Tkinter framework, which allows the configuration and execution of quantum control optimizations without the need for writing Python code. The interface exposes the full set of configurable parameters and provides real-time feedback during the optimization process.

---

## Launching the GUI

The GUI may be launched from the command line after installation:

```bash
freeq-gui
```

Alternatively, it may be invoked as a Python module:

```bash
python -m ctrl_freeq.cli
```

!!! tip "First Launch"
    Upon first launch, the GUI opens with a single qubit configured. Additional qubits may be added using the **Add Qubit** button for multi-qubit optimizations.

---

## GUI Layout

The interface is organized into five principal sections:

| Section | Purpose |
|---------|---------|
| **Qubit Configuration** | Per-qubit pulse and system parameters |
| **Target Specification** | Definition of the target quantum state or gate |
| **Optimization Settings** | Algorithm, iterations, and compute options |
| **Coupling Controls** | Inter-qubit coupling (multi-qubit only) |
| **Action Buttons** | Run, Save, and Reset controls |

### Single-Qubit Configuration

<figure markdown="span">
  ![Single-qubit GUI](assets/gui_single_qubit.png){ width="600" }
  <figcaption>The GUI configured for a single-qubit Hadamard gate optimization.</figcaption>
</figure>

### Two-Qubit Configuration

<figure markdown="span">
  ![Two-qubit GUI](assets/gui_two_qubit.png){ width="700" }
  <figcaption>The GUI configured for a two-qubit CNOT gate optimization with XY coupling.</figcaption>
</figure>

!!! note "Coupling Controls"
    The **Coupling Type** and **Coupling Constants** section appears on the right-hand side of the interface only when two or more qubits are configured.

---

## Configuring Parameters

Each qubit is assigned its own configuration panel. The parameters within each panel are organized into three categories:

- **System Parameters** — Detuning (Δ), Rabi frequency (Ω_R), and their associated uncertainties
- **Pulse Parameters** — Duration, discretization, sweep rate, and bandwidth
- **Waveform Settings** — Basis type, mode, envelope, and expansion order

A detailed description of all parameters, including their defaults and units, is provided in [Optimization → Parameters](optimization/parameters.md).

---

## Target Specification

The **Target States Method** dropdown determines the method by which the desired final state is specified:

| Method | Description |
|--------|-------------|
| **Axis** | Bloch sphere direction (`Z`, `-Z`, `X`, `-X`, `Y`, `-Y`) |
| **Gate** | Quantum gate (`X`, `H`, `CNOT`, `CZ`, etc.) |
| **Phi/Beta** | Rotation angles (axis φ and angle β) |

**Example — π-pulse (inversion):**

- Initial State: `Z`
- Target Method: `Axis`
- Target Axis: `-Z`

A complete description of all target specification options is provided in [Optimization → Parameters → Target States](optimization/parameters.md#target-states).

!!! info "Multi-State Optimization"
    For the design of universal rotations or conditional gates, multiple initial/target state pairs may be specified. Comma-separated values are entered in the respective fields (e.g., Initial: `X, Y, Z`, Target: `X, Z, -Y`). A detailed treatment of this capability is provided in [Multi-State Optimization](optimization/parameters.md#multi-state-optimization).

---

## Optimization Settings

| Setting | Description |
|---------|-------------|
| **Space** | `hilbert` (pure states) or `liouville` (density matrices) |
| **Dissipation** | `non-dissipative` (default) or `dissipative` (Lindblad master equation) |
| **Algorithm** | Optimization algorithm |
| **Max Iterations** | Maximum optimization iterations |
| **Target Fidelity** | Stop when fidelity reaches this value |
| **Compute Resource** | `cpu` or `gpu` |

For guidance on algorithm selection, the reader is referred to [Optimization → Algorithms](optimization/algorithms.md).

For GPU setup instructions, see [Optimization → Compute](optimization/compute.md).

---

## Dissipation Controls

When the **Dissipation** dropdown is set to `dissipative`, the interface enables the modelling of open quantum systems via the Lindblad master equation. The following behaviour is activated:

- The **Space** setting is automatically forced to `liouville` (density matrix mode)
- Per-qubit **T1** and **T2** entry fields are revealed in the qubit configuration panel
- The optimizer employs an Euler operator-splitting scheme, applying a unitary step followed by a Lindblad dissipative step at each time increment

| Control | Description | Units |
|---------|-------------|-------|
| **T1** | Amplitude damping time constant (energy relaxation) | seconds |
| **T2** | Pure dephasing time constant (phase decoherence) | seconds |

!!! warning "Physical Constraint"
    The values entered must satisfy the physical constraint $T_2 \leq 2 T_1$ for each qubit. Configurations that violate this bound will produce a validation error.

!!! note "Liouville Space"
    Selecting `dissipative` mode automatically sets the optimization space to `liouville`. This is required because dissipative dynamics operate on density matrices rather than pure state vectors. The space selection cannot be changed back to `hilbert` while `dissipative` mode is active.

---

## Coupling Controls (Multi-Qubit)

When two or more qubits are configured, a coupling section becomes available:

| Control | Description |
|---------|-------------|
| **J_ij** | Coupling strength between qubit pairs |
| **Coupling Type** | `Z` (Ising), `XY` (exchange), or `XYZ` (Heisenberg) |
| **σ J** | Coupling uncertainty |

For a detailed description of the coupling parameters, see [Optimization → Parameters → Multi-Qubit Coupling](optimization/parameters.md#multi-qubit-coupling).

---

## Action Buttons

### Run

Initiates the optimization with the current settings. Progress information is displayed in the console, and the resulting plots are presented upon completion.

### Save

Saves the current configuration to a JSON file. The exported file may subsequently be loaded in the GUI, used with the Python API, or shared with collaborators.

### Save Results

Upon completion of an optimization run, this button performs the following actions:

1. All generated plots are saved to `results/plots/`
2. An interactive dashboard HTML file is created in `results/dashboards/`

A detailed description of the dashboard is provided in [Dashboard](dashboard.md).

### Set Defaults

Resets all parameters to their default values.

---

## Workflow Examples

### Example 1: Single-Qubit π-Pulse

1. Launch the GUI: `freeq-gui`
2. Set **Initial State**: `Z`
3. Set **Target States Method**: `Axis`
4. Set **Target Axis**: `-Z`
5. Set **Algorithm**: `bfgs`
6. Click **Run**

### Example 2: Two-Qubit CNOT Gate

1. Launch the GUI: `freeq-gui`
2. Click **Add Qubit** to add a second qubit
3. Set **Target States Method**: `Gate`
4. Set **Gate**: `CNOT`
5. Configure coupling in the **Coupling** section
6. Click **Run**

### Example 3: Robust Pulse Design

1. Configure the basic pulse parameters
2. Set **σΔ (Hz)**: `100000` (100 kHz uncertainty)
3. Set **σΩ_R Max (Hz)**: `50000` (50 kHz uncertainty)
4. Set **Target Fidelity**: `0.9999`
5. Click **Run**

The optimizer will seek pulses that are robust to the specified parameter uncertainties.

### Example 4: Dissipative π-Pulse (Open Quantum System)

1. Launch the GUI: `freeq-gui`
2. Set **Initial State**: `Z`
3. Set **Target States Method**: `Axis`
4. Set **Target Axis**: `-Z`
5. Set **Dissipation**: `dissipative`
6. Enter **T1**: `0.001` (1 ms)
7. Enter **T2**: `0.0005` (500 μs)
8. Set **Algorithm**: `qiskit-cobyla`
9. Click **Run**

The optimizer will design a pulse that achieves the desired inversion while accounting for amplitude damping and dephasing during the pulse.

### Example 5: Universal Rotation Pulse

1. Launch the GUI: `freeq-gui`
2. Set **Initial State**: `X, Y, Z`
3. Set **Target States Method**: `Axis`
4. Set **Target Axis**: `X, Z, -Y`
5. Set **Algorithm**: `qiskit-cobyla` (recommended for multi-state optimization)
6. Click **Run**

This procedure designs a single pulse that performs a universal rotation, mapping all three Bloch sphere axes to their specified targets simultaneously. A detailed treatment of multi-state optimization is provided in [Multi-State Optimization](optimization/parameters.md#multi-state-optimization).

---

## Tips

!!! tip "Recommended Initial Approach"
    It is generally advisable to begin with the default parameter values and a straightforward target configuration (such as a π-pulse) in order to verify that the setup is functioning correctly before proceeding to more complex optimization problems.

!!! tip "Iteration Count"
    For initial testing, a moderate number of iterations (100–500) is typically sufficient. For production-quality results, the iteration count should be increased to 1000 or more.

!!! warning "Memory Usage"
    Multi-qubit systems with a large number of time discretization points can require significant memory. System resources should be monitored when working with large problem sizes.

---

## Troubleshooting

### GUI does not launch

```bash
# Verify that Tkinter is available
python -c "import tkinter; print('Tkinter OK')"

# Attempt to run directly
python -m ctrl_freeq.cli
```

### Optimization does not converge

- Increase the **Max Iterations** setting
- Consider using a different **Algorithm** (see [Algorithms](optimization/algorithms.md#choosing-an-algorithm))
- Reduce the **Target Fidelity** initially to verify convergence behaviour
- Ensure that parameter values are physically reasonable

### Plots do not appear

- Verify that the matplotlib backend is correctly configured
- Check the console for error messages
- As an alternative, save the results rather than viewing them interactively

---

## Next Steps

- [API Reference](api.md) — Use ctrl-freeq programmatically
- [Dashboard](dashboard.md) — Visualize and export results
- [Optimization Parameters](optimization/parameters.md) — Detailed parameter reference
- [Algorithms](optimization/algorithms.md) — Algorithm details and selection guide
