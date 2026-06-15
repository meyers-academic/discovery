# Metamatrix development notes

Working/architecture notes for the **metamatrix refactor** — the migration of
Discovery's kernel/likelihood machinery off the legacy closure-based `matrix.py`
onto the graph-based `metamath.py` (DSL: `metamatrix.py`). These are
*development records*, not user/Sphinx docs (hence `dev_architecture/`, not
`docs/`). They capture the design rationale, the phased execution plan, parity
evidence, and per-feature design notes.

## The refactor in one paragraph

`matrix.py` enumerated the Woodbury kernel `Σ = N + F P F^T` by hand across a
combinatorial explosion of const/var variant classes. The metamatrix rewrite
expresses the kernel math symbolically as a graph and lets graph-folding handle
const-vs-var, eliminating the variant explosion. The work proceeded in phases
behind a runtime switch (`ds.config(kernels='matrix'|'metamath')`) with a
parity test suite (`tests/metamatrix/`) certifying the two paths agree before
anything is deleted. As of this checkpoint, Phases 0–4 are done: the metamath
path is at full parity with the `matrix.py` oracle; Phase 5 (deleting
`matrix.py` / `likelihood.py`) is pending external testing.

## Files

| File | What it is | Status |
|---|---|---|
| `metamatrix_architecture.md` | The foundational design doc. The end-state vision, *why* the refactor exists, the graph primitives, method contracts under metamatrix, house rules for writing `metamath.py`, porting guidance (DON'T transliterate const/var branching; DO express math symbolically), a decentering composition example, the high-level migration plan, and an earlier project-status/handoff section. Referenced from `metamath.py`'s module docstring. | reference |
| `mixed_phi_compoundgp_plan.md` | Focused design note for one hard sub-problem: the mixed-Phi `CompoundGP` (a per-pulsar-diagonal commongp combined with a dense, e.g. HD, globalgp) and the external-prior / ExtSignal path. Goal, how it fits, plan of attack, explicit out-of-scope. Marked "completed". | done |
| `exit_plan.md` | The phased execution plan to retire `matrix.py` and `likelihood.py`. Evaluates the original strategy suggestions, then lays out Phase 0 (shared `utils.py` substrate) → 1 (factory + signals migration + measurement-noise collapse) → 2 (`likelihood_metamath` off `matrix`) → 3 (parity coverage gate) → 4 (close carry-overs, no deletion) → 5 (deletion). Living document; phase outcomes recorded inline. | living plan |
| `phase3_coverage.md` | The parity coverage sweep: a table mapping every kernel constructor that production code / example notebooks actually emit to the test model-builder that exercises it (run through the `matrix` / `mh_patched` / `mh_native` routes). Records gaps found and how each was closed (Phases 3–4), plus the `cw_extsignal` integration check. | evidence |
| `branch_diff_vs_upstream_metamatrix.md` | A snapshot comparison of this branch against upstream `nanograv/discovery@metamatrix` (commits ahead, diffstat, themed summary). Written early to scope what this fork carries beyond upstream. May go stale as the branch evolves. | snapshot |

## How they relate

`metamatrix_architecture.md` is the *why* and the conventions; `exit_plan.md` is
the *how* (and tracks progress); `phase3_coverage.md` is the *proof* that the
swap is safe; `mixed_phi_compoundgp_plan.md` is a deep-dive on one tricky piece;
`branch_diff_vs_upstream_metamatrix.md` situates the work against upstream.

Authoritative current state lives in `exit_plan.md` (phase status) and
`phase3_coverage.md` (parity table).
