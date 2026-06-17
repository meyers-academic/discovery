# Future planned development — single precision

Ideas agreed on but deliberately deferred. Not started. Each entry: what, why,
and a rough sketch of how, so a future session can pick it up without re-deriving.

## 1. Configurable pin set

**What.** Today the float64 pins are hardcoded in `metamath.woodbury`
(`g.pin_f64(ytNmy)`, `g.pin_f64(lN)`, `g.pin_f64(lP)`). The set of pinned
quantities is fixed in the kernel math. Make *which* quantities are pinned a
config choice, without editing the kernel math.

**Why.**
- The design (`README.md`, stage 2) left an explicit open question: which of
  `{ytNmy, lN, lP, lS, ld}` should be pins. Confirming that wants experiments on
  different datasets, not code edits.
- Lets us run accuracy/perf sweeps: is `lP` ever load-bearing? does pinning `lS`
  (and paying for a float64 Cholesky) ever actually help on a big array?

**How (sketch).** Move from a boolean to a *label*, and split intent (in the
math) from policy (in config):
- `Node.pin` becomes `Optional[str]` (a label) instead of `bool`.
- `GraphBuilder.pin_f64(sym, key="ytNmy")` tags a node with its label. Still pure
  graph intent — the kernel math advertises the *candidates*; it does not decide
  which are active. House rule (methods return graphs) preserved.
- The policy lives in config and is read at materialization (`func()` boundary),
  in `_dtype_map`:
  ```python
  active = utils.active_pins()          # default: all known labels
  pinned = [name for name, node in graph.items()
            if getattr(node, "pin", None) in active]
  ```
- `utils.config(pins="all" | None | {"ytNmy", "lN"})` selects the set. Default
  "all" = exactly today's behavior, so it's a pure superset and the existing
  tests stay green.
- Add a test: a disabled pin really drops that node to the working dtype (the
  dtype map no longer marks it / its cone float64).

**Cost.** ~3 small touchpoints (`Node`, `_dtype_map`, `pin_f64` signature) + the
config knob + one test. Default-preserving.

**Pairs with** the graph visualization (below): once pins have labels, the
visualization can show *which* pin each float64 region traces back to.

## (add future items below as they come up)
