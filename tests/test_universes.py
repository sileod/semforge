"""Invariants that make the certificates relative to nothing unstated.

The analysis universe is built from what a reader sees (public premises and hypothesis), while
plan sampling and pack validation admit the planted intermediates too, and a neutral instance's
structural claim is analyzed after restoring its withheld premises. These tests pin the
equivalences those choices rely on, so a pack or a rule shape that breaks one fails here instead
of silently changing what a certificate means. Also pinned: the independent checker imports
nothing of the generator, which is what its verdicts are worth.
"""
import ast
import random
from pathlib import Path

import pytest

from semforge.construct.build import Fresh, derive
from semforge.construct.calculus import Universe, analyze
from semforge.construct.packs import get
from semforge.construct.synth import Synthesizer
from semforge.construct.target import Target

PACKS = ["propositional", "monadic"]
CERTCHECK = Path(__file__).parent.parent / "semforge" / "certcheck.py"


def planted_and_public(pack, universe_kind=Universe):
    """Derivations with a plan the planted-universe check accepts: its intermediates should stay
    inside the public universe, so both analyses certify the same structure."""
    rng, syn = random.Random(1), Synthesizer(pack)
    out = []
    for _ in range(6):
        plan = syn.sample_plan(rng, 5)
        d = None
        for _ in range(6):
            d = syn._derive(tuple(plan), rng, Fresh())
            if d is not None:
                break
        if d is None:
            continue
        want = pack.calculus.kinds(plan)
        u_pub = universe_kind([*d.premises, d.conclusion], pack.negate, pack.closed)
        u_planted = universe_kind([*d.premises, *d.intermediates, d.conclusion], pack.negate, pack.closed)
        a_pub = analyze(pack.calculus, d.premises, d.conclusion, u_pub)
        a_planted = analyze(pack.calculus, d.premises, d.conclusion, u_planted)
        out.append((a_pub, a_planted, want))
    return out


@pytest.mark.parametrize("name", PACKS)
def test_planned_steps_are_required_in_the_public_universe(name):
    pack = get(name)
    for a_pub, a_planted, want in planted_and_public(pack):
        assert a_pub.requires(want) is a_planted.requires(want)


@pytest.mark.parametrize("name", PACKS)
def test_public_and_planted_universes_certify_the_same_structure(name):
    pack = get(name)
    for a_pub, a_planted, _ in planted_and_public(pack):
        assert a_pub.mandatory_steps == a_planted.mandatory_steps
        assert a_pub.topology == a_planted.topology
        assert a_pub.depth == a_planted.depth
        assert a_pub.premise_supports == a_planted.premise_supports


class NestedUniverse(Universe):
    """Closed constructors one level deeper than the default (a nested implication or symbol
    application as an operand): the certified structure of a constructed instance should not
    depend on this bound, because rule heads cannot reach outside the published box."""

    def _built(self, f, levels=2):
        if not (isinstance(f, tuple) and f[0] in self.closed):
            return False
        return all(x in self.literals or self._built(x, levels - 1) for x in f[1:])

    def __contains__(self, f):
        return f in self.literals or self._built(f) or self._built(self.negate(f))


@pytest.mark.parametrize("name", PACKS)
def test_certified_structure_is_insensitive_to_universe_nesting(name):
    pack = get(name)
    for a_pub, _, want in planted_and_public(pack, NestedUniverse):
        a_1 = analyze(pack.calculus, a_pub.premises, a_pub.goal,
                      Universe([*a_pub.premises, a_pub.goal], pack.negate, pack.closed))
        assert a_1.mandatory_steps == a_pub.mandatory_steps
        assert a_1.topology == a_pub.topology
        assert a_1.requires(want) == a_pub.requires(want)


@pytest.mark.parametrize("name", PACKS)
def test_counterfactual_certificate_uses_the_restore_universe(name):
    """A neutral instance's structural claim is analyzed with its withheld premises restored, but
    over the *public* universe of the instance it is. The two must agree for the claim to mean
    what its docstring says."""
    pack = get(name)
    rng, syn = random.Random(3), Synthesizer(pack)
    seen = 0
    while seen < 6:
        plan = syn.sample_plan(rng, 4)
        inst = syn.realize(Target(label="neutral", rules=plan, min_depth=2), rng)
        if inst is None or inst.label != "neutral":
            continue
        seen += 1
        restored = [*inst.premises, *inst.withheld, inst.hypothesis]
        a_wide = analyze(pack.calculus, [*inst.premises, *inst.withheld], inst.conclusion,
                         Universe(restored, pack.negate, pack.closed))
        assert inst.analysis.mandatory_steps == a_wide.mandatory_steps
        assert inst.analysis.topology == a_wide.topology
        assert inst.analysis.depth == a_wide.depth


def test_certcheck_imports_nothing_of_the_generator():
    """The checker's independence is evidential, not conventional: it must not import from
    `semforge.construct` (or any semforge module), so a generator bug cannot make a false
    certificate pass its own checker."""
    tree = ast.parse(CERTCHECK.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.startswith("semforge") for a in node.names), ast.dump(node)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or "").startswith("semforge"), ast.dump(node)


def test_all_skeletons_count_matches_the_exact_recurrences():
    from semforge.construct.archive import _exact, all_skeletons, shape_sizes
    for spine in range(1, 8):
        for joins in range(0, 3):
            for s in shape_sizes(spine, joins):
                trees = all_skeletons(spine, joins, s)
                assert len(trees) == _exact(spine, joins, s)
                assert len(set(trees)) == len(trees)  # each listed once


def test_retired_cells_report_why_they_are_unfillable():
    from collections import Counter
    from semforge.construct.archive import DistributionArchive
    from semforge.construct.packs import get

    # grid: nested joins cannot be derived in any way (no rule assignment fills any tree of the
    # cell), while one join is feasible: retired there, but for cost, not structure. Monadic fills any tree.
    weights = {("entailment", 3, 2): 1.0, ("entailment", 3, 1): 1.0}
    expected = {("entailment", 3, 2): "shape", ("entailment", 3, 1): ""}
    for name in ("grid", "monadic"):
        pack = get(name)
        archive = DistributionArchive(weights, calculus=pack.calculus)
        for cell in weights:
            archive.last = cell
            for _ in range(archive.retire):
                archive.add(None)
        rows = archive.report()
        assert all(r["retired"] for r in rows)
        reasons = {(r["label"], r["spine"], r["joins"]): r["reason"] for r in rows}
        for cell, want in expected.items():
            assert reasons[cell] == (want if name == "grid" else ""), (name, cell, reasons[cell])
