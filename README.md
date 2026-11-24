# semforge

Logic datasets usually state how hard a problem is by the length of the proof they planted: "this problem
takes 5 steps". But a planted proof is only one proof. Take the premises *a, a→b, b→c, c→d, d→e, e→f*
and the goal *f*. Planted as five modus ponens steps, it looks like a 5-step problem. Allow the rule that
chains two implications, and the same problem has 42 different minimal proofs, none of whose steps is
needed by all of them. A model can solve it without ever making any particular step.

semforge measures what a problem actually *requires*, and builds problems where what is planted is what is
required, with certificates anyone can check.

**What you can do with it**

- **Audit a dataset.** For each problem, find the steps every proof must make (*mandatory steps*), the
  shortest proof, how many minimal proofs there are, and which premises are indispensable. We have run
  this on ProofWriter, FLD×2, ZebraLogic, PlanBench, FOLIO, ProverQA, LogicGraph and a state-tracking
  dataset; some are as deep as they claim, others much less.
- **Build certified problems.** Ask for a structure (a label, the rules to use, a proof shape, a number of
  proofs, distractors) and get English problems whose every proof has it. The construction algorithm,
  *ProofPlant*, avoids shortcuts using an atlas of known shortcut patterns proved once per logic.
- **Shape training data.** Control the mix of depths, proof shapes and problem types in a training set,
  instead of inheriting whatever a generator happens to produce.
- **Declare your own logic.** A logic is a short declaration: symbols, inference rules, a semantics that a
  solver decides, and English templates. Fifteen come with the library, from propositional, first-order
  and modal logic to grid puzzles, planning and state tracking. Each rule is proved sound automatically.
- **Check without trusting us.** Every problem comes with a certificate (its proof, its mandatory steps,
  the label as an SMT query, and finite countermodels for "neither" answers) that a separate checker,
  sharing no code with the generator, verifies.

semforge accompanies the paper *Certify What You Plant: Shortcut-Free Proof Depth in Logical Reasoning
Data* (arXiv link to come). Version 0.1.0 is a research release: the entry points below are stable,
internals may still change.

## Install

```bash
pip install -e .                          # core: needs only z3-solver
pip install -e '.[test,bench,audit]'      # tests, benchmarks, dataset audits (Hugging Face datasets)
python -m pytest -q tests                 # a few minutes
```

## Quickstart

Validate a logic: every rule proved sound for all instances, how often each rule can be required, and
whether rendering is ambiguous.

```bash
python -m semforge.construct.check propositional
```

Construct a certified problem and look at what it requires:

```python
import random
from semforge.domains.constructed import ConstructedDomain

d = ConstructedDomain("monadic", strategy="shape")
rng = random.Random(0)
out = []
while not out:                 # an attempt can fail (unplaceable target); try again
    out = d.sample(rng, 3)
inst, label = out[0]
premises, hypothesis = d.pack.render(inst.premises, inst.hypothesis, rng)
print(premises, "\n->", hypothesis, f"({label})")
print(inst.analysis.required_rules)   # mandatory steps by rule
```

Audit a dataset: `bash scripts/reproduce_audits.sh quick` downloads the public datasets and reruns every
audit at small sample sizes; drop `quick` for the paper's sizes.

## Layout

```
semforge/construct/   logics and construction
  signature.py          sorts, symbols (z3 semantics, English templates), rule DSL, background theory
  calculus.py           closure and analysis: mandatory steps and premises, topology, minimal proofs
  build.py, synth.py    backward construction of derivations; target -> certified problem (ProofPlant)
  atlas.py              shortcut atlas: proved nogoods, lifting, nogood certificates
  archive.py            archives over certified proof shapes (coverage, target distributions)
  oracle.py             z3 oracle compiled from a signature
  check.py              logic validation CLI
  packs/                one module per logic; the registry is packs/__init__.py
semforge/certcheck.py   independent certificate checker (shares no code with construct/)
semforge/audit.py       calculus-free audit: minimal premise sets
semforge/domains/       dataset adapters (FLD×2, ZebraLogic, FOLIO, constructed data)
semforge/tasks/         proofs and plans as answers, with independent checking
benchmarks/             dataset audits and construction benchmarks
experiments/            training experiments of the paper
scripts/                reproduction entry points
```

A new logic is a signature plus rules, about 50 lines; see `semforge/construct/packs/order.py` for a small
one and `packs/grid.py` for one with background facts. Validate it with
`python -m semforge.construct.check <name>`.

## What the guarantees cover

Labels are exact relative to the logic's solver. Structure is exact relative to the logic's calculus over
a bounded universe of formulas; calculi are ground and forward (no hypothetical steps). Adding rules can
only make steps avoidable, never required, so a step found avoidable stays avoidable under any richer
calculus. Difficulty for models is measured, never certified.

## Reproducing the paper

See [docs/reproduce.md](docs/reproduce.md).

## Citation

```bibtex
@misc{juran2026certify,
  title  = {Certify What You Plant: Shortcut-Free Proof Depth in Logical Reasoning Data},
  author = {Juran, Baptiste and Kachler, Dimitri and Sileo, Damien},
  year   = {2026},
  note   = {arXiv preprint, to appear}
}
```

## License

Apache License 2.0 (see `LICENSE`).
