# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] — 2026-03-09

### Added

- Hamiltonian model abstraction layer (`HamiltonianModel` ABC) enabling platform-agnostic pulse optimization via the standard bilinear control formulation H(t) = H_drift + Σ_k u_k(t) · H_ctrl_k.
- Plugin architecture with model registry: `@register_hamiltonian("name")` decorator, `get_hamiltonian_class()` lookup, and `list_hamiltonians()` discovery, allowing new Hamiltonian types to be added without modifying any framework files.
- `from_config(n_qubits, params)` classmethod on all models for registry-driven construction from configuration dictionaries.
- `default_config(n_qubits)` classmethod on all models returning a complete, ready-to-run configuration with sensible physical defaults (GUI-ready templates).
- Direct model injection via `CtrlFreeQAPI(config, hamiltonian_model=model)` for quick experiments with custom or unregistered models.
- `SpinChainModel` — wraps the existing spin-chain drift and coupling Hamiltonians (Ising, XY, Heisenberg) behind the new model interface.
- `SuperconductingQubitModel` — transmon qubit Hamiltonian with qubit-frequency drift, capacitive coupling (XY, ZZ, XY+ZZ), and anharmonicity-derived static ZZ shifts.
- Calibrated ZZ parameter (`zz_crosstalk`) for `SuperconductingQubitModel`: accepts a calibrated static ZZ coupling matrix that overrides the perturbative formula, with a clear priority chain (runtime `zz_instances` > calibrated `zz_crosstalk` > perturbative formula > zero).
- AC Stark shift (`stark_shift_coeffs`) for `SuperconductingQubitModel`: adds per-qubit drive-dependent Z control channels modelling the light shift H_Stark = Σ_i s_i/2 (I²+Q²) Ω_d² σ_z.
- `DuffingTransmonModel` — 3-level (Duffing oscillator) transmon Hamiltonian with dim = 3^n_qubits, enabling leakage detection to the |2⟩ state. Registered as `"duffing_transmon"` in the plugin registry.
- `embed_computational_state()` and `embed_computational_gate()` methods on `HamiltonianModel` ABC for mapping 2^n states/gates into higher-dimensional model spaces (identity for standard qubit models, active embedding for 3-level models).
- `leakage()` method on `DuffingTransmonModel` to compute population outside the computational subspace.
- Automatic state/gate embedding in `initialise_gui.py` for models with dim > 2^n (e.g. Duffing transmon).
- `pulse_hamiltonian_generic()` — model-agnostic pulse Hamiltonian construction via `einsum`, replacing the spin-chain-specific implementation for new model paths.
- GUI Hamiltonian Type selector dropdown (Spin Chain / Superconducting) that dynamically relabels fields (Δ↔ω, J↔g) and reconfigures coupling controls.
- Gate dropdown (Combobox) in the GUI, replacing free-text entry, with platform-aware gate lists and defaults (CNOT for Spin Chain, iSWAP for Superconducting).
- Two new two-qubit gates: √iSWAP and ECR (echoed cross-resonance).
- Superconducting coupling controls in the GUI: coupling types XY, ZZ, XY+ZZ; per-qubit anharmonicity entry.
- MathJax rendering for all mathematical notation in the documentation (`pymdownx.arithmatex` with MathJax 3).
- Documentation: step-by-step guide for adding custom Hamiltonian models (define, import, use), with registry and direct-injection workflows.
- API demo notebook section 8d: `default_config` usage and direct model injection examples.
- API demo notebook section 9: Duffing transmon demos — single-qubit inversion (9a), leakage measurement (9b), two-qubit iSWAP (9c), custom anharmonicities (9d), and DRAG vs ctrl-freeq comparison (9e).
- `tests/test_dissipation.py` — 20 tests covering collapse-operator construction, dephasing-rate correctness, T1/T2 input validation, dissipative + non-2-level model guard, and Lindblad dissipator algebraic properties (trace-zero, hermiticity preservation). This path previously had zero test coverage.
- `tests/test_optimizer_spaces.py` — comprehensive test of all 19 supported optimizers (9 torchmin + 10 qiskit) across Hilbert, Liouville, and dissipative spaces, plus cross-space consistency checks.
- Coupling-matrix indexing and symmetrisation tests in `test_hamiltonian_models.py`: verifies upper/lower/symmetric inputs produce identical Hamiltonians, asymmetric matrices are rejected, and all coupling types (Z, XY, XYZ) produce non-zero output with bundled configs.

### Changed

- Standardized `build_drift` signature across all models: `frequency_instances`, `coupling_instances` (replaces model-specific parameter names).
- `initialise_gui.py` and `plotter.py` now use the registry for model construction and generic `build_drift` calls, eliminating all isinstance dispatch.
- `SuperconductingQubitModel` docstrings now explicitly document the rotating-frame convention, sign/scaling conventions (rad/s, spin-½ Paulis), and the distinction between exchange coupling (g) and static ZZ (ζ).
- Default two-qubit gate is now Hamiltonian-type-aware: CNOT for spin chains, iSWAP for superconducting qubits.
- Gate entry fields in the GUI are now dropdown menus (Combobox) with available gates filtered by qubit count and Hamiltonian type.
- Plotter functions (`compute_and_store_evolution`, `get_final_rho_for_excitation_profile`) are now model-aware: they use `HamiltonianModel.build_control_ops()` and `model.dim` when available, falling back to legacy Pauli operators for standard qubit models.
- State embedding (`_embed_states_for_model`) now also embeds raw initial states (`self.init`) and observable operators (`self.obs_op`) into the model's Hilbert space, ensuring all six plot types (IQ, amplitude/phase, history, observables, Bloch sphere, excitation profile) work correctly with higher-dimensional models like `DuffingTransmonModel`.
- Backward compatibility is maintained: configurations without `hamiltonian_type` continue to use the legacy spin-chain code path. All new parameters (`zz_crosstalk`, `stark_shift_coeffs`) default to `None` and are fully backward-compatible.
- API reference: constructor and `run_from_config` signatures now document the `hamiltonian_model` parameter for direct model injection.
- Parameter reference: `hamiltonian_type` field now links to the model registry documentation and notes support for custom registered models.

### Fixed

- Fixed pure-dephasing rate being off by 2×. Dephasing collapse operator changed from `√γ_φ · σ_z/2` to `√(γ_φ/2) · σ_z` so the Lindblad dissipator yields the correct off-diagonal decay rate `dρ₀₁/dt = −γ_φ · ρ₀₁` matching the physical T₂ convention.
- Fixed model injection via `CtrlFreeQAPI` leaving state dimensions inconsistent with the injected model. The override path now calls `_embed_states_for_model()` after replacement, with an idempotency guard that skips re-embedding when states are already at the target dimension and raises `ValueError` for incompatible dimensions.
- Fixed ZZ coupling terms being silently skipped unless `coupling_instances` was provided. Calibrated `zz_crosstalk` and runtime `zz_instances` now work independently of exchange-coupling matrices. The perturbative formula still requires `coupling_instances` (as it uses g_{ij}).
- Fixed invalid `T1`/`T2` inputs (zero, negative, infinity, NaN) producing `NaN`/`inf` collapse operators instead of failing. Validation now enforces positive finite values before computing decay rates.
- Fixed dissipative mode silently producing invalid results with non-2-level Hamiltonian models (e.g. `DuffingTransmonModel`). The combination now raises `ValueError` at init time, since collapse operators and Liouville-space embedding are hard-coded for 2-level systems.
- Fixed `DuffingTransmonModel` silently accepting unsupported `coupling_type` values. The constructor now raises `ValueError` for anything other than `"XY"`.
- Fixed inconsistent snapshot semantics across Hamiltonian models. `SpinChainModel.build_drift` now uses repeat-last-element semantics for mismatched `coupling_instances` length, matching `SuperconductingQubitModel` and `DuffingTransmonModel`.
- Fixed `createHJ` defaulting to lowercase `"z"` which silently failed the uppercase comparison and returned a zero matrix. Input is now normalised via `.upper()`, `None` falls back to `"Z"`, and the docstring lists the actual accepted values.
- Fixed `build_collapse_operators` docstring claiming it returns tuples `(L, L_dag, L_dag_L)` when it actually returns `np.ndarray` of shape `(n_ops, D, D)` or `None`.
- Fixed `exp_mat_exact` producing `NaN` for zero-magnitude Hamiltonians (e.g. identity-only drift). The `sin(x)/x` term is now computed via `torch.sinc`, which is numerically safe at `x = 0`.
- Fixed Lindblad dissipator recomputing `L†` and `L†L` at every time step inside the propagation loop. These products are now precomputed once before the loop and passed through.
- Fixed `pulse_para` concatenating an unused `phis` tensor into its return value. The function now returns `(amps, cxs, cys)`, avoiding a redundant allocation on every optimizer iteration.
- Fixed brittle absolute imports in `piecewise.py` (`from src.ctrl_freeq.…`) that break when the package is installed normally. All imports now use the package-relative form `from ctrl_freeq.…`.
- Fixed `createHJ` reading the lower triangle of the coupling matrix (`J[n,k]` with `k < n`) while all configs, the GUI, and the new Hamiltonian models populate the upper triangle (`J[i,j]` with `i < j`). This caused spin-chain inter-qubit coupling to be silently zero for every bundled and GUI-generated two-qubit configuration. The function now symmetrises the input so that upper-triangular, lower-triangular, and fully symmetric matrices all produce the correct Hamiltonian. Asymmetric matrices with conflicting entries raise `ValueError`.

## [0.2.0] — 2026-02-25

### Added

- Support for dissipative evolution in the optimization process with Lindblad master equation.
- Documentation of Lindblad master equation support for dissipative open quantum systems (T1/T2 relaxation channels, GUI controls, API workflow, configuration schema).

### Fixed

- Fixed a bug in the calculation of density matrix.
- Fixed a normalisation bug in the plotter for observable dynamics visualisation in the Liouville space.

## [0.1.1] — 2026-02-17

### Changed

- Broadened Python version support from 3.13-only to 3.11–3.13.
- CI test matrix now covers Python 3.11, 3.12, and 3.13.
- GitHub Actions updated to actions/checkout v4 and actions/setup-python v5.

### Removed

- Hard dependency on a system LaTeX installation; text rendering now falls back gracefully when LaTeX is unavailable.
- Unused `tqdm` dependency.

### Fixed

- Corrected an invalid `tqdm` version pin that referenced the non-existent 3.x series.
- Resolved a broken relative link to the example dashboard in the documentation.

## [0.1.0] — 2026-02-17

Initial public release.

### Added

- Quantum gate and pulse optimization via automatic differentiation (PyTorch).
- Programmatic Python API for loading configurations, executing optimizations, and post-processing results.
- Tkinter-based graphical user interface, launched via the `freeq-gui` CLI entry point.
- Interactive Plotly dashboards exported as standalone HTML files.
- Multiple optimizer families: BFGS, L-BFGS, CG, Newton-CG, Newton-Exact, Dogleg, Trust-NCG, Trust-Krylov, Trust-Exact, and Qiskit-provided algorithms.
- Polynomial-based and piecewise-constant waveform parameterisations.
- Selective and band-selective coverage modes.
- CPU thread management and optional GPU acceleration via CUDA.
- Comprehensive documentation built with MkDocs Material.
- CI/CD pipelines: GitHub Actions for testing, PyPI publishing, and Docker/GHCR image builds.
