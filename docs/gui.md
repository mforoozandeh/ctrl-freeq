# GUI Guide

ctrl-freeq includes a Tkinter-based graphical user interface for configuring and running quantum control optimizations without writing code.

---

## Launching the GUI

After installation, launch the GUI from the command line:

```bash
freeq-gui
```

Alternatively, run it as a Python module:

```bash
python -m ctrl_freeq.cli
```

!!! tip "First Launch"
    On first launch, the GUI opens with a single qubit configured. Use the **Add Qubit** button to add more qubits for multi-qubit optimizations.

---

## GUI Layout

The GUI is organized into five sections:

| Section | Purpose |
|---------|---------|
| **Qubit Configuration** | Per-qubit pulse and system parameters |
| **Target Specification** | Define the target quantum state or gate |
| **Optimization Settings** | Algorithm, iterations, and compute options |
| **Coupling Controls** | Inter-qubit coupling (multi-qubit only) |
| **Action Buttons** | Run, Save, and Reset controls |

### Single-Qubit Configuration

<figure markdown="span">
  ![Single-qubit GUI](assets/gui_single_qubit.png){ width="600" }
  <figcaption>GUI configured for a single-qubit Hadamard gate optimization.</figcaption>
</figure>

### Two-Qubit Configuration

<figure markdown="span">
  ![Two-qubit GUI](assets/gui_two_qubit.png){ width="700" }
  <figcaption>GUI configured for a two-qubit CNOT gate optimization with XY coupling.</figcaption>
</figure>

!!! note "Coupling Controls"
    The **Coupling Type** and **Coupling Constants** section on the right only appears when two or more qubits are configured.

---

## Configuring Parameters

Each qubit has its own configuration panel. Parameters are organized into:

- **System Parameters** — Detuning (Δ), Rabi frequency (Ω_R), and uncertainties
- **Pulse Parameters** — Duration, discretization, sweep rate, bandwidth
- **Waveform Settings** — Basis type, mode, envelope, and expansion order

For detailed parameter descriptions, defaults, and units, see [Optimization → Parameters](optimization/parameters.md).

---

## Target Specification

The **Target States Method** dropdown determines how you specify the desired final state:

| Method | Description |
|--------|-------------|
| **Axis** | Bloch sphere direction (`Z`, `-Z`, `X`, `-X`, `Y`, `-Y`) |
| **Gate** | Quantum gate (`X`, `H`, `CNOT`, `CZ`, etc.) |
| **Phi/Beta** | Rotation angles (axis φ and angle β) |

**Example — π-pulse (inversion):**

- Initial State: `Z`
- Target Method: `Axis`
- Target Axis: `-Z`

For complete target specification options, see [Optimization → Parameters → Target States](optimization/parameters.md#target-states).

!!! info "Multi-State Optimization"
    To design universal rotations or conditional gates, you can specify multiple initial/target state pairs. Enter comma-separated values (e.g., Initial: `X, Y, Z`, Target: `X, Z, -Y`). See [Multi-State Optimization](optimization/parameters.md#multi-state-optimization) for details.

---

## Optimization Settings

| Setting | Description |
|---------|-------------|
| **Space** | `hilbert` (pure states) or `liouville` (density matrices) |
| **Algorithm** | Optimization algorithm |
| **Max Iterations** | Maximum optimization iterations |
| **Target Fidelity** | Stop when fidelity reaches this value |
| **Compute Resource** | `cpu` or `gpu` |

For algorithm selection guidance, see [Optimization → Algorithms](optimization/algorithms.md).

For GPU setup, see [Optimization → Compute](optimization/compute.md).

---

## Coupling Controls (Multi-Qubit)

When using two or more qubits, a coupling section appears:

| Control | Description |
|---------|-------------|
| **J_ij** | Coupling strength between qubit pairs |
| **Coupling Type** | `Z` (Ising), `XY` (exchange), or `XYZ` (Heisenberg) |
| **σ J** | Coupling uncertainty |

For coupling parameter details, see [Optimization → Parameters → Multi-Qubit Coupling](optimization/parameters.md#multi-qubit-coupling).

---

## Action Buttons

### Run

Executes the optimization with current settings. Progress is displayed in the console, and results are shown in plot windows.

### Save

Saves the current configuration to a JSON file. The file can be:

- Loaded later in the GUI
- Used with the Python API
- Shared with collaborators

### Save Results

After running an optimization:

1. Saves all generated plots to `results/plots/`
2. Creates an interactive dashboard HTML file in `results/dashboards/`

See [Dashboard](dashboard.md) for dashboard details.

### Set Defaults

Resets all parameters to their default values.

---

## Workflow Examples

### Example 1: Single-Qubit π-Pulse

1. Launch GUI: `freeq-gui`
2. Set **Initial State**: `Z`
3. Set **Target States Method**: `Axis`
4. Set **Target Axis**: `-Z`
5. Set **Algorithm**: `bfgs`
6. Click **Run**

### Example 2: Two-Qubit CNOT Gate

1. Launch GUI: `freeq-gui`
2. Click **Add Qubit** to add a second qubit
3. Set **Target States Method**: `Gate`
4. Set **Gate**: `CNOT`
5. Configure coupling in the **Coupling** section
6. Click **Run**

### Example 3: Robust Pulse Design

1. Configure basic pulse parameters
2. Set **σΔ (Hz)**: `100000` (100 kHz uncertainty)
3. Set **σΩ_R Max (Hz)**: `50000` (50 kHz uncertainty)
4. Set **Target Fidelity**: `0.9999`
5. Click **Run**

The optimizer will find pulses robust to the specified parameter uncertainties.

### Example 4: Universal Rotation Pulse

1. Launch GUI: `freeq-gui`
2. Set **Initial State**: `X, Y, Z`
3. Set **Target States Method**: `Axis`
4. Set **Target Axis**: `X, Z, -Y`
5. Set **Algorithm**: `qiskit-cobyla` (recommended for multi-state)
6. Click **Run**

This designs a single pulse that performs a universal rotation, mapping all three Bloch sphere axes to their specified targets. See [Multi-State Optimization](optimization/parameters.md#multi-state-optimization) for more details.

---

## Tips

!!! tip "Start Simple"
    Begin with default parameters and a simple target (like a π-pulse) to verify your setup works before attempting complex optimizations.

!!! tip "Iteration Count"
    Start with fewer iterations (100-500) for quick tests. Increase to 1000+ for production runs.

!!! warning "Memory Usage"
    Multi-qubit systems with many time points can require significant memory. Monitor system resources for large problems.

---

## Troubleshooting

### GUI doesn't launch

```bash
# Check Tkinter is available
python -c "import tkinter; print('Tkinter OK')"

# Try running directly
python -m ctrl_freeq.cli
```

### Optimization doesn't converge

- Increase **Max Iterations**
- Try a different **Algorithm** (see [Algorithms](optimization/algorithms.md#choosing-an-algorithm))
- Reduce **Target Fidelity** initially
- Check parameter values are physically reasonable

### Plots don't appear

- Ensure matplotlib backend is configured
- Check console for error messages
- Try saving results instead of viewing interactively

---

## Next Steps

- [API Reference](api.md) — Use ctrl-freeq programmatically
- [Dashboard](dashboard.md) — Visualize and export results
- [Optimization Parameters](optimization/parameters.md) — Detailed parameter reference
- [Algorithms](optimization/algorithms.md) — Algorithm details and selection guide
