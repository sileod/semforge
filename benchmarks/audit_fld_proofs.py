"""FLD×2 at the proof level: certify its rows in a shared calculus and compare proof structure with ours.

FLD formulas are parsed into semforge terms. Universal facts are instantiated over the row's constants
(one premise per instance) and existentials kept opaque. Every row, FLD's and ours, is then re-certified
in one calculus: the propositional pack's rules plus FLD's simple ones (right conjunction elimination,
disjunction introduction). Its mandatory steps give the same proof-level measures for both:
- derivable rate;
- mandatory steps and spine;
- E/C topology cells (label × spine × joins);
- distinct mandatory structures and kind sets, at an equal count and at an equal mandatory-step
  distribution.

The calculus has no hypothetical rules (FLD's assumption + discharge, reductio), so some FLD rows are not
derivable here; they are counted, not dropped silently. Grounding universals makes instantiation
implicit, a step this calculus does not count; FLD counts it in `steps`, so min_size is compared with
the claim minus the gold proof's instantiation steps (`planted`), the raw comparison kept apart.

  python -m benchmarks.audit_fld_proofs --n 400 --out results/audit/fld_proofs.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict

from semforge.audit import summarize_profiles
from semforge.construct.archive import structure, topology_niche
from semforge.construct.calculus import Calculus, Universe, analyze
from semforge.construct.packs.propositional import RULES, negate
from semforge.construct.signature import rule
from semforge.domains.constructed import ConstructedDomain
from semforge.domains.fld import TOKEN, from_row, load

CALCULUS = Calculus(RULES + [rule("and_elimination_right", "(and A B) |- B"),
                             rule("or_introduction_left", "A |- or A B"),
                             rule("or_introduction_right", "B |- or A B")])
# FLD×2's own rules among ours: its basic rules, plus the derived rules its gold proofs use as single steps
# (hypothetical syllogism, constructive dilemma, modus tollens; counted over the train split in
# audit_fixed/fld_own_rules.py on branch audit_fixed_results). Its gold proofs also use contraposition, which
# is not in the shared calculus; adding it removes more mandatory steps (Proposition 1).
FLD_RULES = [r for r in RULES if r.name in ("modus_ponens", "and_elimination", "and_introduction",
                                             "hypothetical_syllogism", "modus_tollens", "constructive_dilemma")] + [
    rule("and_elimination_right", "(and A B) |- B"), rule("or_introduction_left", "A |- or A B"),
    rule("or_introduction_right", "B |- or A B")]
CLOSED = ("->",)  # as the propositional pack: conjunctions and disjunctions must occur in the inputs


class Parser:
    """FLD notation to terms: imp := or ('->' imp)? ; or := and ('v' and)* ; and := un ('&' un)* ;
    un := '¬' un | '(x):' imp | '(Ex):' imp | '(' imp ')' | atom. Binary connectives nest to the right."""

    def __init__(self, text):
        self.toks, pos, text = [], 0, text.strip()
        while pos < len(text):
            m = TOKEN.match(text, pos)
            if not m:
                raise ValueError(text)
            self.toks.append(m.group(1))
            pos = m.end()
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self):
        self.i += 1
        return self.toks[self.i - 1]

    def parse(self):
        e = self.imp()
        assert self.peek() is None, self.toks
        return e

    def imp(self):
        a = self.or_()
        if self.peek() == "->":
            self.take()
            return ("->", a, self.imp())
        return a

    def _chain(self, op, sub, tok):
        xs = [sub()]
        while self.peek() == tok:
            self.take()
            xs.append(sub())
        out = xs[-1]
        for x in reversed(xs[:-1]):
            out = (op, x, out)
        return out

    def or_(self):
        return self._chain("or", self.and_, "v")

    def and_(self):
        return self._chain("and", self.un, "&")

    def un(self):
        t = self.peek()
        if t == "¬":
            self.take()
            return ("not", self.un())
        if t in ("(x):", "(Ex):"):
            self.take()
            return ("forall" if t == "(x):" else "exists", self.imp())
        if t == "(":
            self.take()
            e = self.imp()
            self.take()
            return e
        return self.atom(self.take())

    @staticmethod
    def atom(t):
        m = re.fullmatch(r"\{([A-Za-z]+)\}(?:\{([a-z]+)\}|(x))?", t)
        pred, const, var = m.groups()
        return ("@", pred) if var else (f"{pred}({const})" if const else pred)


def constants(f, out):
    if isinstance(f, str):
        m = re.fullmatch(r"[A-Za-z]+\((\w+)\)", f)
        if m:
            out.add(m.group(1))
    elif f[0] != "@":
        for x in f[1:]:
            constants(x, out)
    return out


def instantiate(f, c):
    if isinstance(f, str):
        return f
    if f[0] == "@":
        return f"{f[1]}({c})"
    return (f[0], *(instantiate(x, c) for x in f[1:]))


def ground(f, consts) -> list:
    """Premises for one FLD formula: universals instantiated over the constants, existentials opaque."""
    if isinstance(f, tuple) and f[0] == "forall":
        return [instantiate(f[1], c) for c in sorted(consts)] or [f"∀{hash(repr(f)) % 10**6}"]
    if isinstance(f, tuple) and f[0] == "exists":
        return [f"∃{hash(repr(f)) % 10**6}"]
    return [f]


class _Row:  # shaped like an Instance for topology_niche / structure
    def __init__(self, label, analysis):
        self.label, self.analysis = label, analysis


def universe(formulas):
    """Subformulas and their negations, as the packs' universe, plus an explicit negation of each, so that
    double negations of subformulas (which our own rows reach through derivation nodes) are available to
    every source alike."""
    u = Universe(formulas, negate, CLOSED)
    u.literals |= {("not", f) for f in u.literals}
    return u


def certify(premises, hypothesis, label, calculus=None):
    goal = hypothesis if label == "entailment" else negate(hypothesis)
    a = analyze(calculus or CALCULUS, premises, goal, universe([*premises, hypothesis, goal]))
    return _Row(label, a) if a.derivable else None


def fld_rows(n, rng, depth=None):
    data, out, seen = load(), [], Counter()
    while len(out) < n:
        i = rng.randrange(len(data))
        inst = from_row(data[i], i)
        if inst.gold == "neutral" or (depth is not None and inst.steps != depth):
            continue
        try:
            facts = [Parser(f).parse() for _, f in inst.facts if f != "fake_formula"]
            hyp = Parser(inst.hyp_formula).parse()
        except (ValueError, AttributeError, AssertionError):
            seen["unparsed"] += 1
            continue
        consts = set()
        for f in [*facts, hyp]:
            constants(f, consts)
        if isinstance(hyp, tuple) and hyp[0] in ("forall", "exists"):
            seen["quantified_hypothesis"] += 1
            continue
        premises = [g for f in facts for g in ground(f, consts)]
        r = certify(premises, hyp, inst.gold)
        seen["derivable" if r else "not_derivable"] += 1
        if r:
            r.claimed_steps, r.hypothesis = inst.steps, hyp
            r.claimed_inferences = None if inst.steps is None else inst.steps - inst.instantiations
            out.append(r)
    return out, seen


def ours(strategy, n, rng):
    d, out, seen = ConstructedDomain("propositional", strategy=strategy), [], Counter()
    while len(out) < n:
        got = [x for x in d.sample(rng, rng.randint(0, 5)) if x[1] != "neutral"]
        if not got:
            continue
        inst, lab = got[0]
        r = certify(list(inst.premises), inst.hypothesis, lab)  # re-certified in the shared calculus
        seen["derivable" if r else "not_derivable"] += 1
        if r:
            out.append(r)
    return out, seen


def measures(rows):
    cells = {topology_niche(r) for r in rows}
    return {"n": len(rows), "mandatory_steps": round(sum(len(r.analysis.mandatory_steps) for r in rows) / len(rows), 2),
            "spine": round(sum(r.analysis.topology["spine"] for r in rows) / len(rows), 2),
            "ec_cells": len(cells), "structures": len({structure(r.analysis) for r in rows}),
            "kind_sets": len({frozenset(r.analysis.mandatory_kinds) for r in rows}),
            "kind_multisets": len({tuple(sorted(r.analysis.mandatory_kinds.items())) for r in rows}),
            "zero_mandatory": round(sum(not r.analysis.mandatory_steps for r in rows) / len(rows), 3),
            "profile": summarize_profiles([r.analysis.profile for r in rows], planted=_claims(rows)),
            **({"min_size_below_claimed_steps": round(sum(r.analysis.profile["min_size"] < r.claimed_steps for r in rows)
                                                      / len(rows), 3)} if _claims(rows) else {})}


def _claims(rows):
    """FLD's claimed proof length without its instantiation steps, which grounding makes free here: the
    claim comparable with min_size. None for sources that claim nothing."""
    claims = [getattr(r, "claimed_inferences", None) for r in rows]
    return claims if all(c is not None for c in claims) else None


def size_matched(groups, rng):
    by = {s: defaultdict(list) for s in groups}
    for s, rows in groups.items():
        for r in rows:
            by[s][min(len(r.analysis.mandatory_steps), 20)].append(r)
    common = {k: min(len(by[s][k]) for s in groups) for k in set().union(*map(set, by.values()))}
    return {s: measures([r for k, m in sorted(common.items()) if m for r in rng.sample(by[s][k], m)]) for s in groups}


def depth_audit(args, keys=("min_size", "mandatory", "zero_mandatory", "forced_share", "min_size_below_planted")):
    """A larger audit of one claimed depth: the means of Table 1 with 95% bootstrap intervals over rows."""
    rows, seen = fld_rows(args.n, random.Random(args.seed), args.depth)
    summary = lambda rs: summarize_profiles([r.analysis.profile for r in rs], planted=_claims(rs))
    rng, boot = random.Random(args.seed), []
    for _ in range(1000):
        boot.append(summary([rng.choice(rows) for _ in rows]))
    interval = lambda k: [round(sorted(b[k] for b in boot)[q], 3) for q in (25, 974)]
    report = {"depth": args.depth, "coverage": dict(seen), "summary": summary(rows),
              "interval95": {k: interval(k) for k in keys}}
    print(json.dumps(report, indent=1))
    json.dump(report, open(args.out, "w"), indent=1)


def fld_only_audit(args):
    """The same FLD rows in FLD's own rules (FLD_RULES) and in the shared calculus, paired on the rows FLD's
    rules derive: how much of the shared calculus's verdict comes from rules FLD lacks."""
    own = Calculus(FLD_RULES)
    rows, seen = fld_rows(args.n, random.Random(args.seed), args.depth)
    paired, lost = [], 0
    for r in rows:
        o = certify(list(r.analysis.premises), r.hypothesis, r.label, own)
        if o is None:
            lost += 1
            continue
        o.claimed_steps, o.claimed_inferences = r.claimed_steps, r.claimed_inferences
        paired.append((r, o))
    summary = lambda rs: summarize_profiles([x.analysis.profile for x in rs], planted=_claims(rs))
    report = {"depth": args.depth, "coverage": dict(seen), "shared_calculus_rows": len(rows),
              "not_derivable_in_fld_rules": lost,
              "shared_calculus_all": summary(rows),
              "paired_shared": summary([r for r, _ in paired]), "paired_fld_rules": summary([o for _, o in paired])}
    print(json.dumps(report, indent=1))
    json.dump(report, open(args.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, help="FLD rows claimed at this depth only, with bootstrap intervals")
    ap.add_argument("--fld-rules", action="store_true", help="also certify in FLD's own rules, paired")
    args = ap.parse_args()
    if args.fld_rules:
        return fld_only_audit(args)
    if args.depth is not None:
        return depth_audit(args)
    groups, seen = {}, {}
    groups["fld"], seen["fld"] = fld_rows(args.n, random.Random(args.seed))
    for s in ("forward", "shape"):
        groups[f"ours_{s}"], seen[f"ours_{s}"] = ours(s, args.n, random.Random(args.seed))
    report = {"coverage": {k: dict(v) for k, v in seen.items()},
              "equal_n": {k: measures(v) for k, v in groups.items()},
              "size_matched": size_matched(groups, random.Random(args.seed)),
              "fld_steps_vs_mandatory": {s: round(sum(len(r.analysis.mandatory_steps) for r in groups["fld"] if r.claimed_steps == s)
                                                  / max(1, sum(r.claimed_steps == s for r in groups["fld"])), 2)
                                         for s in sorted({r.claimed_steps for r in groups["fld"] if r.claimed_steps is not None})},
              "fld_by_claimed_depth": {s: summarize_profiles([r.analysis.profile for r in rs], planted=_claims(rs))
                                       for s in sorted({r.claimed_steps for r in groups["fld"] if r.claimed_steps is not None})
                                       for rs in [[r for r in groups["fld"] if r.claimed_steps == s]]}}
    print(json.dumps(report, indent=1))
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
