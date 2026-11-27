# ⚒️ semforge

**Generate and audit logical reasoning data with certified proof structure.**

Paper: [Certify What You Plant: Shortcut-Free Proof Depth in Logical Reasoning Data](https://arxiv.org/abs/2611.33303) (arXiv:2611.33303).

Ask for problems whose every proof makes exactly *k* inference steps you choose, in one of 15 logics, and
get English text plus a certificate that an independent checker verifies. Or audit an existing dataset:
how many of its claimed reasoning steps are actually needed.

```bash
pip install git+https://github.com/sileod/semforge     # needs only z3-solver
```

## Quickstart

```python
import semforge as sf

p = sf.generate("propositional", mandatory_steps=2, label="entailment", seed=3, distractors=0)
print(p.text)
print(p.mandatory_rules, p.proofs)
print(p.verify())        # [] = every claim re-checked by the independent checker
```

```
Tara drinks tea.
If Emma is happy, then Priya plays chess.
If Tara drinks tea, then Priya does not play chess.
Hypothesis: Emma is not happy.
Label: entailment
{'modus_ponens': 1, 'modus_tollens': 1} 1
[]
```

Every proof of the hypothesis must make both of those steps: none can be skipped by another route. That is
what `mandatory_steps` guarantees, relative to the logic's inference rules.

To make sure problems exercise particular rules, pass `require_rules`: every problem then has a
mandatory step of one of them, so no proof avoids it (`sf.rules("modal")` lists a logic's rules):

```python
modal = {"necessity_elimination", "necessity_refutation", "impossibility_elimination", "possibility_introduction"}
p = sf.generate("modal", mandatory_steps=3, require_rules=modal)   # needs modal reasoning, certified
```

A dataset is the same call with `n`. Leave `label` out to mix entailment, contradiction and neutral:

```python
import json
problems = sf.generate("monadic", mandatory_steps=4, n=1000, seed=0)
with open("monadic.jsonl", "w") as f:
    for i, p in enumerate(problems):
        f.write(json.dumps({"id": i, **p.to_dict()}) + "\n")
```

Anyone can then re-check every certificate without trusting the generator:

```bash
python -m semforge.certcheck monadic.jsonl       # -> 0 certificate(s) failed
```

## Logics

`sf.LOGICS` lists them. One generated example each, two mandatory steps, no distractors (seed 3):

| logic | example (premises / hypothesis, all entailed) | mandatory steps |
|---|---|---|
| `propositional` | Tara drinks tea. If Emma is happy, then Priya plays chess. If Tara drinks tea, then Priya does not play chess. ⟹ *Emma is not happy.* | modus ponens, modus tollens |
| `monadic` (first-order, one-place predicates) | All felpuses are lorpuses. Either Bea is a rompus or every belpus is a felpus. Bea is not a rompus. ⟹ *All belpuses are lorpuses.* | syllogism, disjunctive syllogism |
| `modal` (K; also `modal_s4`, `modal_nested`, `modal_hybrid`) | Both in every situation reachable from Quarry, the archive is sealed and in Calder, the bell rings. Dunmore is reachable from Quarry. ⟹ *In Dunmore, the archive is sealed.* | ∧-elimination, □-elimination |
| `order` | (transitivity, irreflexivity) Both Quin is taller than Mo and Cy is taller than Eli. Ada is shorter than Mo. ⟹ *Quin is taller than Ada.* | ∧-elimination, chain |
| `arithmetic` | Lea scored 6. Dee's score is 7. Zed scored 1 more than Dee. ⟹ *Zed scored 2 more than Lea.* | chain, difference |
| `kinship` | (rules for parents, siblings, ancestors...) facts about a family ⟹ *Tom is an aunt or uncle of Xia.* | sibling, uncle |
| `planning` | a robot, rooms, doors and a sequence of moves ⟹ *At step 4, the robot is in the garden.* | blocked move ×2 |
| `grid` (zebra puzzles) | five houses, people, colours and clues ⟹ *David is not in the fourth house.* | category, distance |
| `csp` | Either Alice's house is purple or Mateo's house is white. Mateo and Lena live in houses of different colours. Alice does not live in the purple house. ⟹ *Lena's house is not white.* | disjunctive syllogism, inequality |

Also `monadic_syllogistic` (instantiations and syllogisms only) and `boxes` (state tracking). Most logics
produce a problem in well under a second; `grid` and `boxes` take seconds to a minute, and `arithmetic`
goes up to about 5 mandatory steps.

## Audit a dataset

Two levels, with different requirements.

**Premise level: no calculus, any dataset.** Give the premises and an entailment oracle (anything that says
whether a subset of premises still yields the label, e.g. a z3 call on the dataset's formulas). semforge
enumerates the minimal premise sets:

```python
import z3
a, b, c = z3.Bools("a b c")
premises = [a, z3.Implies(a, b), z3.Implies(b, c), z3.Implies(a, c), c]

def entails(subset):
    s = z3.Solver(); s.add(*subset, z3.Not(c))
    return s.check() == z3.unsat

sf.audit_premises(premises, entails)
# {'supports': 3, 'min_support': 1, 'indispensable': [], 'never_needed': 0.0, ...}
```

Three different premise sets justify the label, the smallest has one premise, and no premise is needed by
all of them.

**Proof level: steps, relative to a logic's rules.** Write the problem in one of the logics' formulas:

```python
premises = ["a", ("->", "a", "b"), ("->", "b", "c"), ("->", "c", "d"), ("->", "d", "e"), ("->", "e", "f")]
sf.analyze("propositional", premises, "f")
# {'derivable': 'entailment', 'proofs': 42, 'mandatory_steps': 0, 'depth': 3, ...}
```

To check an advertised operator, name its rules: `families={"disjunction": {"disjunctive_syllogism",
"constructive_dilemma"}}` adds whether every proof needs one of them, even when no single step is
mandatory. Over a dataset, `sf.summarize` aggregates the per-problem results (and, given the dataset's
labels, how often they disagree with the derived label):

```python
results = [sf.analyze("propositional", p, h, families=fams) for p, h in problems]
sf.summarize(results, labels=gold)
# {'derivable': ..., 'mandatory_steps': ..., 'no_mandatory_step': ..., 'rule_in_every_proof': {...},
#  'family_in_every_proof': {'disjunction': 0.64}, 'label_disagreement': ...}
```

`benchmarks/` holds the full audits of ProofWriter, FLD×2, ZebraLogic, PlanBench, FOLIO, ProverQA,
LogicGraph and a state-tracking dataset; `bash scripts/reproduce_audits.sh quick` reruns them at small
sizes.

## Why semforge?

Logic datasets usually state how hard a problem is by the length of the proof they planted: "this problem
takes 5 steps". But a planted proof is only one proof. The audit above is a 5-step chain *a, a→b, …, e→f ⊢
f*: allow the rule that chains two implications, and it has 42 minimal proofs, none of whose steps is
needed by all of them. A model can solve it without making any particular step. Across published datasets,
some are as deep as they claim and others much less.

semforge measures what a problem *requires*, and builds problems where what is planted is required. The
construction algorithm, *ProofPlant*, avoids shortcuts using an atlas of shortcut patterns proved once per
logic, and certifies every output in full.

## What the guarantees cover

- **Labels** are exact relative to the logic's solver (z3).
- **Structure** (mandatory steps, proofs, depth) is exact relative to the logic's calculus over a bounded
  universe of formulas; calculi are ground and forward (no hypothetical steps). Adding rules can only make
  steps avoidable, never required, so a step found avoidable stays avoidable under any richer calculus.
- **Certificates** hold the proof, the mandatory steps and premises, the label as SMT queries and finite
  countermodels for neutral labels; `semforge/certcheck.py` shares no code with the generator.
- **Difficulty for models** is measured, never certified.

## Going further

- **Finer targets** (proof shapes, several proofs, distractor types, deliberately bypassable steps):
  `semforge.construct.target.Target` and `semforge.construct.synth.Synthesizer`, which `sf.generate` wraps.
- **Your own logic**: a signature (symbols with z3 semantics and English templates) plus rules, about 50
  lines. Start from `semforge/construct/packs/order.py` (small) or `packs/grid.py` (background facts), and
  validate it with `python -m semforge.construct.check <name>`: every rule is proved sound by z3.
- **Training data mixtures** and the paper's experiments: `experiments/`, and
  [docs/reproduce.md](docs/reproduce.md).

```
semforge/api.py         generate, analyze, summarize, audit_premises, rules
semforge/construct/     logics (packs/), calculus and analysis, construction (ProofPlant), shortcut atlas
semforge/certcheck.py   independent certificate checker
semforge/audit.py       premise-level audits of datasets
semforge/domains/       dataset adapters (FLD×2, ZebraLogic, FOLIO, constructed data)
benchmarks/             dataset audits and construction benchmarks
experiments/            training experiments of the paper
```

Development: `pip install -e '.[test]' && python -m pytest -q tests` (a few minutes).

## Citation

semforge accompanies *Certify What You Plant: Shortcut-Free Proof Depth in Logical Reasoning Data*
([arXiv:2611.33303](https://arxiv.org/abs/2611.33303)). Version 0.1.0 is a research release: `generate`, `analyze`, `summarize`, `audit_premises` and the
certificate format are stable, internals may change.

```bibtex
@misc{juran2026certify,
  title  = {Certify What You Plant: Shortcut-Free Proof Depth in Logical Reasoning Data},
  author = {Juran, Baptiste and Kachler, Dimitri and Sileo, Damien},
  year   = {2026},
  eprint = {2611.33303},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url    = {https://arxiv.org/abs/2611.33303}
}
```

## License

Apache License 2.0 (see `LICENSE`).
