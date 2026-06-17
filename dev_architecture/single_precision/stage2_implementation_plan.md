# Stage 2 implementation plan — f64 pins

> **STATUS: DONE (2026-06-17).** Both 2a and 2b landed and committed; the
> plan below was followed as written, with one refinement: `ld` is left in the
> working dtype (not pinned) — pinning it would drag `lS`'s Cholesky into f64.
> See `README.md` "Stage 2 (DONE)" for the as-built description and the Piece-2
> limitation. Kept as the rationale/record.

Plan for stage 2 of the single-precision graph work in metamath. Stage 1 (landed)
casts the whole graph to float32 — no exceptions. Stage 2 lets us mark a few
specific calculations to stay in float64 while everything else stays float32.

Read `README.md` (this folder) "Graph precision: implementation" and
`HANDOFF_graph_precision.md` first. This doc is the build/test breakdown.

## The plain-English idea

Some numbers in the likelihood need to be computed in float64 even when we're
running everything else in float32, because in float32 they lose too many digits.
The main one is `ytNmy` (= yᵀ N⁻¹ y), plus two log-determinants `lN` (white noise)
and `lP` (prior).

A number can't just be "float64" or "float32" by itself, because the *same*
number feeds two different places that want different precision. Example: the
white-noise matrix `N` feeds both `ytNmy` (want float64) and `FtNmF` (want float32,
it's the big expensive one). So precision has to be decided **per connection
(edge)**, not per number (value): `N` is stored once in float64, and when the
float32 calculation reads it, it reads a float32 copy.

So the mechanism is:
1. Mark ("pin") the few nodes that must be float64.
2. Walk backwards from each pinned node and mark everything it depends on as
   float64 too (so the pinned number is built entirely in float64).
3. Everything else is float32.
4. When a node reads one of its inputs, convert that input to *the reading node's*
   precision. That's how `N` can hand float64 to `ytNmy` and float32 to `FtNmF`.

### What the pins do and don't do (important, confirm with Patrick before 2b)

The pins make sure `ytNmy`, `lN`, `lP` are **computed accurately** (in float64).
Once computed, when they get combined into the final `logL` (which is float32),
they get converted down to float32. So the pins protect *building* those numbers,
not *keeping them float64 all the way to the end*. Keeping the final combination
in float64 is a separate, harder problem (Piece 2, reference+delta) and is NOT
part of stage 2.

Consequence: `ld = lN + lP + lS` stays float32. We do NOT pin `ld`, because `ld`
depends on `lS`, and pinning `ld` would drag `lS` into float64 — and `lS` being
float64 means the small Cholesky runs in float64, which we don't want. So:
**pin `ytNmy`, `lN`, `lP`; leave `lS` and `ld` float32.**

## 2a — build the machinery, no pins yet

Goal: build the dtype-map + convert-on-read mechanism with the pin set EMPTY.
With no pins, the result must be numerically identical to stage 1, and the
metamatrix parity tests (118) + single-precision tests (74) must stay green.
Ship this before adding any pins.

`metamatrix.py`:
- Add `pin: bool = False` to the `Node` dataclass. Default off = no change to
  the float64 default path. Survives folding (non-folded nodes are reused as-is)
  and pruning.
- `GraphBuilder.pin_f64(sym)`: set `self.graph[sym.name].pin = True`, return `sym`.
- `_dtype_map(graph, working)`: returns name -> dtype.
  - If not single precision, or no pinned nodes: everything float64 (so
    convert-on-read does nothing).
  - Seed float64 from pinned nodes; walk backwards over `node.inputs` to mark all
    ancestors float64 (reuse the prune_graph BFS pattern). Everything else =
    working dtype.
  - Computed once, inside `build_callable_from_graph` (i.e. at `func()`).
- In `build_callable_from_graph`: delete the stage-1 leaf cast (`_cast`). Instead,
  when evaluating each node, convert each input to that node's mapped dtype:
  `args = [_cast_to(env[inp], dtype_map[name]) for inp in node.inputs]`. Only
  convert floating arrays (leave integer indices etc. alone).

Tests (`tests/single_precision/test_graph_precision.py`):
- Existing blanket-f32 tests must still pass unchanged (2a == stage 1 with no pins).
- New unit test on `_dtype_map`: empty pins -> all working; a tiny hand-built
  graph with one pinned node -> pin + its ancestors float64, the rest working.
- Gate to ship 2a: `pytest tests/metamatrix -q` green.

## 2b — add the pins + tests

`metamath.py`, in `woodbury` (these are graph-build calls, not materialization,
so the house rule holds):
- pin the `g.dot(y, Nmy)` inside `logp` (that's `ytNmy`).
- pin `lN` and `lP` where they land.
- do NOT pin `ld` or `lS`.

Tests:
- White-box: build `full_rn` under working=float32; assert the dtype map marks
  `ytNmy`/`lN`/`lP` and their ancestors (`N`, `y`, `P`) float64, while `FtNmF`,
  the cho_factor, and `lS` stay float32.
- Fixed white noise: `ytNmy` folds to a constant — check it's float64-valued and
  `logL` still matches stage 1.
- Sampled white noise (live): hand-fill noise params; check the live `ytNmy`/`lN`
  path is float64 while the Cholesky is float32.
- Finiteness/accuracy unchanged vs stage 1.
- Perf smoke (informational, not asserted): time sampled-WN logL+grad with/without
  pins; expect no regression.

## Norms
- Default backend is `matrix`; graph tests must set `ds.config(kernels='metamath')`
  and reset.
- `utils.config(working=)` = precision; `ds.config(kernels=)` = which backend. Two
  different functions.
- Push only to `origin HEAD:metamatrix-meyers`. No commit/push without go-ahead.
- This doc and the HANDOFF/checkpoint docs are not committed.
