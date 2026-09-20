# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.4.0] — 2026-09-21

Several fixes change numerical results on purpose; see **Breaking behaviour**.

### Breaking behaviour

- **Gate targets are now scored as an average gate fidelity over the whole
  computational subspace.** When every configured initial state names the same
  canonical gate (aliases resolved first, e.g. `CX` → `CNOT`), the objective
  propagates the computational basis as columns and scores
  `F_avg = (Tr(M†M) + |Tr M|²) / (d(d+1))` with `M = G†V†UV`; in Liouville and
  dissipative modes it uses the Pauli-transfer channel metric
  `F_avg = (d·x₀ + Σⱼ xⱼ) / (d²(d+1))`, which also counts population lost from
  the computational subspace. Previously only the configured input states were
  scored, so a pulse could report fidelity 1 while implementing a different
  operation. Reported fidelities for gate configurations will drop accordingly.
  Configurations that name *different* gates per initial state keep
  state-transfer scoring; that score is not certification of a coherent gate.
  The selected mode is exposed as `parameters.objective_mode`
  (`"gate"` / `"state_transfer"`), with `computational_dim`, `n_objective_rows`,
  `gate_name` and `gate_matrix` alongside it.
- **Dissipative propagation now uses the exact channel `exp(dt·D)` with Strang
  splitting** (half-channel, unitary, half-channel) instead of an Euler step.
  The Euler step left the physical state space for step sizes comparable to
  T1/T2 (a T1=1, dt=2 step returned populations `[2, -1]`); the channel is CPTP
  for every step size and the splitting is second-order accurate. The same
  semantics now apply in the differentiable, NumPy, piecewise and analysis
  paths.
- **The two-level `SuperconductingQubitModel` drift now follows the Duffing
  convention**: `-δᵢ Zᵢ + 2gᵢⱼ(XᵢXⱼ + YᵢYⱼ)`, fixed by requiring it to equal the
  projection of `DuffingTransmonModel`'s drift onto the computational subspace.
  The AC Stark channel amplitude is correspondingly `-sᵢ(I²+Q²)Ωᵢ²`. The
  separate spin-chain/NMR convention (`createHcs`/`createHJ`) is unchanged.
- **`band_selective` bandwidth is the FWHM of the target rotation-angle
  profile**: `exp[-ln2 (2|Δ-Δ₀|/bw)^(2p)]`, equal to 1/2 at both band edges for
  every order. This is not a guarantee about the achieved excitation curve.
- **Waveform samples are the midpoints of the propagation intervals**
  (`(k+½)T/N`), and stored trajectories are the `N+1` state boundaries `kT/N`
  including the initial state. The previous `linspace(eps, T, N)` grid was
  spaced `T/(N-1)`, so carrier modulation ran fast and analysis propagated past
  the intended duration.
- **`compute_and_store_evolution` now returns five values**
  `(cxs, cys, history, history_mean, leakage)`, with `N+1` time entries.
  Plot these against `Initialise.state_boundary_times()`.
- **Progress histories now report the physical fidelity.** `fidelity_history`
  previously stored `fidelity − penalty`. Fidelity, penalty and the penalized
  score are now three separate series (`fidelity_history`, `penalty_history`,
  `score_history`), with `final_fidelity`, `final_penalty` and `final_score`
  evaluated on the solution that is actually returned. The stopping criterion
  remains the penalized score and is now named as such.
- **Rank-deficient waveform bases are rejected** instead of being silently
  completed by QR with directions outside the requested basis (e.g. 8 chirp
  columns on 10 sample points has numerical rank 5).
- **`band_selective` coverage with `Axis` or `Gate` targets is rejected.**
  The smooth profile has no all-or-nothing interpretation for a discrete
  target; use `selective` coverage or a `Phi`/`Beta` target.

### Fixed

- Gate objectives no longer score only the configured input states, and gate
  aliases are canonicalised before the objective is chosen.
- Euler relaxation replaced by an exact CPTP dissipative channel; channel
  durations are validated as finite and non-negative.
- Two-level and Duffing transmons now agree on detuning sign, exchange scale
  and Stark sign.
- The perturbative static ZZ estimate is
  `ζ = 2g²(αᵢ+αⱼ)/((Δ+αᵢ)(Δ-αⱼ))` with `Δ = δᵢ-δⱼ`, evaluated per snapshot, and
  is refused near the |11⟩↔|20⟩/|02⟩ avoided crossings (mixing
  `√2|g|/min(|Δ+αᵢ|,|Δ-αⱼ|) ≤ 0.1`, a small-mixing heuristic, not a 1% error
  bound). The previous estimate had the wrong sign, ignored the detuning and
  had no validity guard. Calibrated ZZ still takes priority.
- Coupling uncertainty alone no longer collapses the drift ensemble to a
  single snapshot; deterministic offsets are repeated to match it. Previously
  ten requested snapshots produced one frequency snapshot and ten coupling
  draws, and only one pair was used.
- Selective coverage masks are evaluated per qubit instead of only inspecting
  qubit 1: axis targets are built as a product over qubits, and a gate is
  applied only when every participating qubit is in its band.
- Observables and density matrices embed as `V O V†`, separate from the gate
  embedding `V G V† + (I − VV†)`. A fully leaked |2⟩ now reports 0 on the
  computational Pauli observables instead of `⟨Z⟩ = +1`.
- Each qubit's selective samples are shuffled before joint snapshots are
  formed, so the ensemble contains mixed in-band/out-of-band combinations.
  Pairing the concatenated `[left | in-band | right]` arrays by index made
  every qubit in-band or out-of-band at the same index.
- Waveform sample times, propagation step and plotted state times agree.
- Numerical rank and dimensions are checked for every matrix that is actually
  factorised, including the envelope-weighted ones.
- Constant envelopes (one- or two-point grids) return a flat unit envelope
  instead of `NaN` from a zero-width normalisation range; envelope inputs are
  validated.
- `band_selective` width uses the correct FWHM conversion; bandwidth and order
  are validated.
- Coupling matrices are normalised once per physical pair, sampled once per
  pair and mirrored, so a symmetric nominal matrix with `sigma_J > 0` stays
  symmetric instead of becoming asymmetric and being rejected downstream.
  Superconducting and Duffing models accept upper-triangular, lower-triangular
  and symmetric inputs, matching the legacy normaliser and the plotter.
- Initial-state, target and drift arrays share one canonical ordering
  (`row × drift snapshot`, with the Rabi snapshot expanded last), with explicit
  length validation. The two sides previously used different flattening orders,
  so each initial state saw only a subset of the drift ensemble.
- Analysis replays the optimizer's physics: the model's complete control
  operator/amplitude mapping (including extra channels such as AC Stark), the
  same propagator and the same dissipative channel. Dynamics replay previously
  evolved unitarily even in dissipative mode, and indexed control operators in
  pairs, so a Stark-enabled two-qubit model drove qubit 2 with `Z₀` and `X₁`.
  Nominal and sampled trajectories are now distinguished, and the sampled band
  covers drift × Rabi.
- Duffing states embed by rank: vectors as `Vψ`, density matrices and Pauli
  strings as `VρV†`. A Liouville-space Duffing run previously built `(1, 3, 2)`
  states against `(1, 3, 3)` drift matrices. Dissipative Duffing remains
  explicitly unsupported.
- The Uhlmann fidelity against a pure target is evaluated in closed form as
  `Re Tr(ρσ)`, which is differentiable; target purity is validated. The
  eigendecomposition-based version raised a complex eigenvector-phase error in
  its backward pass.
- Fidelity, penalty and penalized score are reported separately in both
  optimizer paths, and final metrics describe the returned solution rather than
  the optimiser's last trial point.
- Piecewise optimisation honours the Hamiltonian model's control mapping and
  dimension, and selects the propagator by actual matrix dimension. A one-qutrit
  drift previously raised a 2-versus-3 dimension error, and superconducting
  Stark channels were omitted.

- Analysis replay used a float32 time step, so it propagated with
  `dt = 9.99999993922529e-09` instead of `1e-8`.
- Analysis now reconstructs waveforms with the representation the solution was
  optimised in, via the new `WaveformSpec`. It previously always used the
  configured basis parameter counts and matrices, so a piecewise solution
  either raised a split-size error (when the counts differed) or silently
  reconstructed a different waveform (when they happened to coincide) — an
  eight-point piecewise solution crashed, and a two-point one replayed at
  fidelity 0.0132 against an optimised 1.0000.
- The dissipative channel keeps a tensor duration in the autograd graph.
  Substituting the validated Python scalar detached `exp(dt·D)` from the pulse
  duration, so duration gradients were silently zero. Gradients with respect to
  the waveform coefficients were unaffected.
- Selective and band-selective sampling produce exactly the requested number of
  snapshots. Rounding the out-of-band count up to an even number and then
  halving it for both tails dropped a sample: with `ratio_factor = 1`, requests
  of 1/3/5 snapshots produced 0/2/4, and a one-snapshot request produced an
  empty ensemble.
- The amplitude report derives the sampled physical peak range from `|gain|`.
  A negative sampled Rabi gain is a phase reversal, not a negative drive
  magnitude, so signed extrema reported `(-2, 1)` for gains `[-2, 1]` and
  understated the largest drive.
- A piecewise waveform representation no longer leaks into a later basis run on
  the same parameters object. Each optimizer now declares its own
  representation, so `PiecewiseAPI(api).run_optimization()` followed by
  `api.run_optimization()` replays the basis solution correctly instead of
  splitting its 4 coefficients with the piecewise layout's 16. Analysing a
  solution whose parameter count disagrees with the recorded representation
  raises a named error instead of reconstructing the wrong waveform; see
  **Known limitations** for what that check cannot catch.
- Piecewise `polar_phase` optimisation no longer raises `TypeError` when
  selecting the per-qubit amplitude envelope order from a preprocessed
  configuration.

### Added

- `WaveformSpec` in `make_pulse.waveform_gen_torch`, describing the parameter
  counts, basis matrices and modes a solution vector is expressed in, together
  with `waveform_function()` / `waveform_functions()`. Analysis reads it via
  `Initialise.waveform_spec()`, which returns the most recent run's
  representation (the configured basis by default);
  `Initialise.basis_waveform_spec()` always returns the configured basis, and
  `Piecewise.waveform_spec()` the piecewise identity basis.
- Optional `waveform_spec=` argument on `process_and_plot()`,
  `compute_and_store_evolution()`, `get_final_rho_for_excitation_profile()` and
  `plot_excitation_profiles()`, so a solution can be analysed with its own
  representation instead of whichever run finished last. Backwards compatible:
  omitting it keeps the previous implicit behaviour.
- `CtrlFreeQAPI.waveform_spec()` and `PiecewiseAPI.waveform_spec()` as the
  handles to pass to those entry points. The former stays valid after another
  optimizer has run on the same parameters object.
- `Initialise.dt`, `Initialise.state_boundary_times()` and
  `Initialise.n_drift_snapshots()`.
- `canonical_gate_name()`, `pauli_string_basis()` and
  `band_selective_profile()` in `setup.initialise_gui`.
- `fidelity_gate_hilbert()`, `fidelity_gate_liouville()`,
  `dissipator_superoperator()`, `dissipative_channel()`, `apply_channel()` and
  `simulate_trajectory()` in `ctrlfreeq.ctrl_freeq`; `dissipator_superoperator`,
  `dissipative_channel` and `apply_channel` also in `evolution.time_evolution`.
- `HamiltonianModel.embed_computational_operator()`,
  `HamiltonianModel.computational_projector()` and
  `HamiltonianModel.computational_leakage()`.
- `SuperconductingQubitModel.perturbative_zz()`.
- Total-leakage trajectories `L(t) = 1 − Tr(Π_comp ρ(t))` at the `N+1` state
  boundaries, returned by `compute_and_store_evolution` and drawn by the new
  `plot_leakage()` (nominal trace plus snapshot min/max envelope). The total is
  per register; per-qubit leakages are not summed.
- `amplitude_limit_report()` / `format_amplitude_limit_report()` and
  `parameters.amplitude_report`: per-qubit sampled peak `hypot(I, Q)` normalised
  to `Omega_R_max`, the amount above 1, and the nominal versus sampled physical
  peak (the sampled range uses gain magnitudes). The amplitude limit remains a
  **soft penalty** — the returned waveform is not clipped or constrained to it,
  and a sampled peak is not a bound on an independently interpolated continuous
  waveform. Hard hardware enforcement is deferred, not implemented.

- Documentation: a new **Objectives and Fidelity** page defining the cost
  function, the two objective modes, the average-gate-fidelity and channel
  metrics, the coverage interaction and the reported quantities; an
  **Analysis and Plotting** section in the API reference covering the time
  grids, leakage series and waveform representation. The dissipative
  splitting, AC Stark, perturbative ZZ and two-level drift conventions were
  corrected to match the implementation, along with the `targ_fid`,
  `pulse_bandwidth` and `amplitude_envelope` descriptions.
- 196 regression tests covering the fixes above, using independent analytical
  references (exact Duffing spectra, an exact single-qubit 2-design,
  full-Liouvillian matrix exponentials, explicit nested-loop propagation) and
  finite-difference checks of both gradients and Hessian-vector products in
  float64/complex128.

### Known limitations

- Implicit analysis replay uses the representation of the **most recent run**
  on a parameters object. When two optimizers share one object and both
  solutions are still in play, pass the owning run's representation to the
  analysis entry points — `CtrlFreeQAPI.waveform_spec()` stays valid across
  later runs, and `PiecewiseAPI.waveform_spec()` returns its own. The
  parameter-count check only catches representations of *different* sizes;
  equal-count representations (for example a two-point basis solution and a
  two-point piecewise cart solution, both 4 parameters) are indistinguishable
  and will replay with the latest run's representation. Optimisation metrics
  such as `final_fidelity` live on the same shared object and are likewise
  overwritten by a later run.
- RNG seeds are not yet exposed or recorded, and optimised pulses are not
  validated on held-out drift/Rabi draws or denser offset/time grids. Reported
  fidelities are training-batch fidelities.
- `stark_shift_coeffs` is documented as dimensionless but multiplies `Ω²`, so
  its effective units are inverse frequency; sensible values are far below 1.

## [0.3.0] — 2026-04-01

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
