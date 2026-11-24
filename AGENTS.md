# Notes for coding agents

This file is for AI coding assistants (and humans) working on semforge. Read README.md first for what the
library does.

## Commands

```bash
pip install -e '.[test,bench,audit]'
python -m pytest -q tests                          # full suite, a few minutes; keep it green
python -m pytest -q tests/test_construct.py        # construction, analysis, checker
python -m semforge.construct.check <logic>         # validate a logic after touching it (soundness, requiredness, rendering)
bash scripts/reproduce_audits.sh quick             # end-to-end audits at small n
```

## Where things are

- `semforge/construct/signature.py`: the DSL (sorts, symbols, rule schemas, semantics, English templates).
- `semforge/construct/packs/<logic>.py`: one module per logic, registered in `packs/__init__.py`.
  A new logic starts from `packs/order.py` (small) or `packs/grid.py` (background facts).
- `semforge/construct/calculus.py`: closure and `Analysis` (mandatory steps and premises, topology, proofs).
- `semforge/construct/build.py`, `synth.py`, `atlas.py`: ProofPlant (backward construction, certification,
  shortcut atlas and nogood learning).
- `semforge/certcheck.py`: the independent certificate checker.
- `benchmarks/`: dataset audits and construction benchmarks. `experiments/`: the paper's training runs.

## Invariants (do not break)

1. **The checker stays independent.** `semforge/certcheck.py` must not import from `semforge.construct`
   or share helpers with it. A bug in the generator must not be able to make a false certificate pass.
2. **Labels come from the solver, structure from the calculus.** Never derive a label from the calculus
   or from construction intent. A query the solver cannot settle raises `Undecided` and the problem is
   dropped, never guessed.
3. **Every output is certified in full.** Atlas pruning is an optimization; it may reject valid
   placements but must never be the reason an instance is accepted.
4. **Rules are sound.** A rule added to a logic must pass `python -m semforge.construct.check <logic>`
   (proved for every instance by z3, or reported as not provable).
5. **Builds are deterministic** for a given seed and code version. Do not introduce unseeded randomness or
   iteration over unordered sets where order affects output.
6. **Claims are relative to a calculus.** Mandatory steps, depth and topology are defined relative to a
   logic's rules; do not describe them as absolute properties in docs or output.

## Conventions

- Python 3.10+, standard library plus z3 for the core; heavier dependencies (datasets, torch) only in
  optional extras and imported lazily.
- Add or update a test with every behavioural change; prefer small instances that run in seconds.
- Keep docstrings concrete: what is computed and relative to what.
