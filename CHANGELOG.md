# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.3.0] — 2026-02-27

### Added

- Hamiltonian model abstraction layer (`HamiltonianModel` ABC) enabling platform-agnostic pulse optimization via the standard bilinear control formulation H(t) = H_drift + Σ_k u_k(t) · H_ctrl_k.
- `SpinChainModel` — wraps the existing spin-chain drift and coupling Hamiltonians (Ising, XY, Heisenberg) behind the new model interface.
- `SuperconductingQubitModel` — transmon qubit Hamiltonian with qubit-frequency drift, capacitive coupling (XY, ZZ, XY+ZZ), and anharmonicity-derived static ZZ shifts.
- `pulse_hamiltonian_generic()` — model-agnostic pulse Hamiltonian construction via `einsum`, replacing the spin-chain-specific implementation for new model paths.
- GUI Hamiltonian Type selector dropdown (Spin Chain / Superconducting) that dynamically relabels fields (Δ↔ω, J↔g) and reconfigures coupling controls.
- Gate dropdown (Combobox) in the GUI, replacing free-text entry, with platform-aware gate lists and defaults (CNOT for Spin Chain, iSWAP for Superconducting).
- Two new two-qubit gates: √iSWAP and ECR (echoed cross-resonance).
- Superconducting coupling controls in the GUI: coupling types XY, ZZ, XY+ZZ; per-qubit anharmonicity entry.

### Changed

- Default two-qubit gate is now Hamiltonian-type-aware: CNOT for spin chains, iSWAP for superconducting qubits.
- Gate entry fields in the GUI are now dropdown menus (Combobox) with available gates filtered by qubit count and Hamiltonian type.
- Backward compatibility is maintained: configurations without `hamiltonian_type` continue to use the legacy spin-chain code path.

### Removed

- `NeutralAtomModel` (Rydberg) — removed due to semantic mismatches between the |g⟩/|r⟩ basis and the Pauli-based state/gate/target infrastructure.

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
