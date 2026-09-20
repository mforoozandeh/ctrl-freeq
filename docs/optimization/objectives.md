# Objectives and Fidelity

This page defines what ctrl-freeq actually optimises, and which number is reported
where. The objective is selected automatically from the target type, so the same
configuration always produces the same metric.

## The cost function

Every optimizer minimises

\[
C = -F + P
\]

where \(F\) is the physical fidelity defined below and \(P\) is the amplitude
penalty. \(-C\) is the **penalized score**, not the fidelity; the two differ
whenever the pulse exceeds its amplitude limit. See
[Reported quantities](#reported-quantities).

## Objective modes

| Target type | Objective | `objective_mode` |
|---|---|---|
| `Axis` | State transfer on the configured initial states | `state_transfer` |
| `Phi` / `Beta` | State transfer, with smooth per-qubit coverage scaling | `state_transfer` |
| `Gate`, all initial states naming the **same** gate | Average gate fidelity over the whole computational subspace | `gate` |
| `Gate`, initial states naming **different** gates | State transfer on the configured initial states | `state_transfer` |

The selected mode is exposed on the parameters object:

```python
api.parameters.objective_mode      # "gate" or "state_transfer"
api.parameters.n_objective_rows    # rows scored per drift snapshot
api.parameters.computational_dim   # d = 2**n_qubits, gate mode only
api.parameters.gate_name           # canonical gate name, gate mode only
```

### State transfer

Each configured initial state is propagated and compared with its target:

\[
F = \bigl\langle\, |\langle \psi_{\text{target}} | \psi_{\text{final}}\rangle|^2 \,\bigr\rangle_{\text{ensemble}}
\]

In Liouville space this is the Uhlmann fidelity against a pure target, which
reduces exactly to \(\mathrm{Re}\,\mathrm{Tr}(\rho\sigma)\). Mixed targets are
rejected.

### Average gate fidelity

Gate names are canonicalised before the objective is chosen, so `CX` and `CNOT`
select the same objective and produce identical initial/target arrays.

When every configured initial state names the same canonical gate, the request
is for *that gate*, so the objective covers the whole computational subspace
rather than the configured inputs. The computational basis is propagated as
**columns**, preserving relative phases. With \(V\) the isometry embedding the
\(d = 2^n\) computational states in the model Hilbert space, \(U\) the
full-space evolution and \(G\) the target gate, \(M = G^\dagger V^\dagger U V\)
and

\[
F_{\text{avg}} = \frac{\mathrm{Tr}(M^\dagger M) + |\mathrm{Tr}\,M|^2}{d\,(d+1)}
\]

In Liouville and dissipative modes the metric is the Pauli-transfer channel
form. With unnormalised Pauli strings \(P_j\) (\(\mathrm{Tr}(P_jP_k) = d\,\delta_{jk}\),
\(P_0 = I\)) and \(x_j = \mathrm{Tr}\bigl[(G P_j G^\dagger)\,\mathcal{E}_{\text{comp}}(P_j)\bigr]\),

\[
F_{\text{avg}} = \frac{d\,x_0 + \sum_j x_j}{d^2 (d+1)}
\]

The \(d\,x_0\) term replaces the usual constant \(d^2\), so population lost from
the computational subspace lowers the score. Leakage is **not** renormalised
away.

!!! warning "Gate fidelities are lower than state-transfer fidelities"
    Scoring only the configured input states lets a pulse reach fidelity 1
    while implementing a different operation on the states that were not
    scored. A configuration that reported a high number under state-transfer
    scoring will report a lower, and more meaningful, number here.

    As a reference point: an identity evolution scores \(1/3\) against a
    single-qubit `Z` target, and \(0.4\) against `CNOT`.

!!! note "Different gates per initial state"
    Naming a different gate for each initial state is not a request for one
    coherent gate, so state-transfer scoring is retained. That score is **not**
    certification of a coherent conditional gate — it says only that those
    particular inputs reach those particular outputs.

## Coverage interaction

Selective coverage masks are evaluated per qubit:

- **Axis targets** are built as a product over qubits — a qubit inside its band
  is driven to its target axis, a qubit outside it stays where it started.
- **Gate targets** apply the full gate only when *every* participating qubit is
  in its band, and the identity otherwise.
- **Phi/Beta targets** keep smooth per-qubit scaling with the coverage profile.

`band_selective` coverage is rejected with `Axis` or `Gate` targets: its smooth
rotation-angle profile has no all-or-nothing interpretation for a discrete
target. Use `selective` coverage, or a `Phi`/`Beta` target.

## Reported quantities

| Attribute | Meaning |
|---|---|
| `final_fidelity` / `fidelity_history` | Physical fidelity \(F\) |
| `final_penalty` / `penalty_history` | Amplitude penalty \(P\) |
| `final_score` / `score_history` | Penalized score \(F - P\) |

The final metrics are evaluated on the solution that is returned, not on the
optimizer's last trial point.

`targ_fid` stops the optimisation when the **penalized score** reaches it, so a
solution that only reaches the target by exceeding the amplitude limit does not
stop early.

## Amplitude limit

The amplitude limit is a **soft penalty**. Exceeding it costs objective value,
but nothing constrains the returned waveform to stay within it. After a run,
`parameters.amplitude_report` gives, per qubit, the sampled peak
\(\max_t \sqrt{I^2+Q^2}\) normalised to `Omega_R_max`, the amount above 1, and
the nominal versus sampled physical peak.

Sampled peaks are not automatically bounds on an independently interpolated
continuous waveform, which can overshoot between samples.
