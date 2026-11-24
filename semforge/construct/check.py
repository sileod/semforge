"""Validate a pack before generating from it: `python -m semforge.construct.check [pack ...]`.

Per rule, over random single-rule derivations:
  sorts       the rule's metavariables have one sort each (no "X is a thing here and a class there")
  sound       the oracle accepts premises |- conclusion on every derivation (else the rule is wrong, or
              the oracle is: either way labels and structure would disagree)
  required    how often the rule is required when it was the one planned; 0 means a redundant calculus
              hides it (another rule always gives an alternative proof), so targets naming it cannot be met
  x2          how often a plan using it twice certifies two required uses (a lower number means
              other rules shortcut chains of it, e.g. Barbara replacing a second instantiation)
  bypass      how often its conclusion still follows from its premises without it: the rules that make
              it redundant are listed
Pack-wide: rendering ambiguity (two different formulas with the same sentence under one lexicon), and
surface forms that are not equivalent to the formula they render.
Exit status 1 on an unsound rule, a sort clash or an ambiguous sentence; redundancy is reported, since
it is a fact about the calculus (e.g. Barbara lets two instantiations be skipped), not an error.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict

from .build import Fresh, derive
from .calculus import Calculus, Universe, analyze


def check_rule(pack, name, n, rng):
    calc, rule = pack.calculus, pack.calculus.rules[name]
    out = dict(rule=name, sorts="ok", n=0, unsound=[], required=0, bypass=0, bypassed_by=Counter(), x2=0, n2=0)
    try:
        pack.signature.infer([*rule.premises, rule.conclusion])
    except (ValueError, KeyError) as e:
        out["sorts"] = str(e)
    others = Calculus([r for r in calc.rules.values() if r.family != rule.family], pack.signature)
    formulas = []
    for i in range(n):
        d = derive(calc, [name], rng, Fresh(f"{name}{i}_"))  # distinct constants across derivations
        if d is None:
            continue
        out["n"] += 1
        formulas += [*d.premises, *d.intermediates, d.conclusion]
        if not pack.entails(d.premises, d.conclusion):
            out["unsound"].append((d.premises, d.conclusion))
            continue
        universe = Universe([*d.premises, *d.intermediates, d.conclusion], pack.negate, pack.closed)
        out["required"] += analyze(calc, d.premises, d.conclusion, universe).required_rules[rule.family] > 0
        alt = analyze(others, d.premises, d.conclusion, universe)
        if alt.derivable:
            out["bypass"] += 1
            out["bypassed_by"].update(set(alt.required_rules) or {"?"})
        d = derive(calc, [name, name], rng, Fresh(f"{name}{i}x_"))
        if d is not None and pack.entails(d.premises, d.conclusion):
            universe = Universe([*d.premises, *d.intermediates, d.conclusion], pack.negate, pack.closed)
            out["n2"] += 1
            out["x2"] += analyze(calc, d.premises, d.conclusion, universe).required_rules[rule.family] >= 2
    return out, formulas


def prove_rule(pack, rule) -> str:
    """Soundness of the rule schema itself, for every instance: premises ⊨ conclusion with each
    metavariable an unconstrained symbol of its sort (formula metavariables as atoms, which is sound for
    logics closed under uniform substitution; entity metavariables without unique names, since two may
    denote one object). "proved", "refuted" (some instance is unsound), "unknown" (z3 timeout), or
    "n/a: ..." when the semantics cannot take symbolic arguments (e.g. computed from constant names)."""
    import z3
    from .oracle import query
    from .terms import is_var

    def plain(x):  # metavariable ?A -> constant mv_A
        if is_var(x):
            return "mv_" + x[1:]
        return tuple(plain(a) if i else a for i, a in enumerate(x)) if isinstance(x, tuple) else x
    try:
        s = query(pack.signature, [plain(p) for p in rule.premises], plain(rule.conclusion), unique_names=False)
    except Exception as e:  # noqa: BLE001 -- semantics that need concrete constants
        return f"n/a: {type(e).__name__}"
    s.set("timeout", 10_000)
    r = s.check()
    return "proved" if r == z3.unsat else "refuted" if r == z3.sat else "unknown"


def interpreted_disagreements(pack, rng, cap=2000) -> list:
    """Facts on which an interpreted symbol's `holds` and its z3 semantics disagree: every argument tuple
    when there are at most `cap`, otherwise a sample of `cap`. Labels come from z3 and structure from
    `holds`, so the two must define one relation."""
    from itertools import product
    import z3
    from .oracle import query
    out = []
    for name, (holds, domains) in pack.calculus.background_set.interpreted.items():
        tuples = list(product(*domains))
        for args in (tuples if len(tuples) <= cap else rng.sample(tuples, cap)):
            f = (name, *args)
            if query(pack.signature, [], f if holds(*args) else pack.negate(f)).check() != z3.unsat:
                out.append(f)
    return out


def ambiguities(pack, formulas, rng, tries=3):
    """Sentences that realize two different formulas under one shared lexicon."""
    from .realize import Realizer
    real = Realizer(pack.signature)
    bg = pack.signature.background_heads
    formulas = sorted({f for f in formulas if not (isinstance(f, tuple) and f[0] in bg)}, key=repr)  # never rendered
    while True:
        try:
            lex = real.lexicon(formulas, rng)
            break
        except ValueError:  # more constants than a sort's lexicon: check fewer formulas
            formulas = formulas[:len(formulas) // 2]
    seen = defaultdict(set)
    for f in formulas:
        for _ in range(tries):
            seen[real.sentence(f, lex, rng)].add(f)
    return {s: fs for s, fs in seen.items() if len(fs) > 1}


def unfaithful_surfaces(pack, formulas):
    """Formulas whose rendered form for some role is not equivalent to them under the pack's oracle."""
    bad = []
    for f in sorted(set(formulas), key=repr):
        for role in ("premise", "hypothesis"):
            g = pack.surface(f, role)
            if g != f and not (pack.entails([f], g) and pack.entails([g], f)):
                bad.append((role, f, g))
    return bad


def _pct(a, b):
    return f"{a / b:.0%}" if b else "-"


def check(pack, n=20, seed=0, out=sys.stdout):
    """True when the pack passes. A refuted rule schema fails it; schemas z3 could not settle ("unknown",
    "n/a") are listed but do not fail it, since sampled instances still check each rule."""
    rng, ok, pool, unproved = random.Random(seed), True, [], {}
    print(f"pack {pack.name}: {len(pack.calculus.rules)} rules, {len(pack.signature.symbols)} symbols, "
          f"{len(pack.signature.sorts)} sorts", file=out)
    print(f"  {'rule':28} {'sorts':6} {'sound':>7} {'schema':>9} {'required':>9} {'x2':>5} {'bypass':>7}  bypassed by", file=out)
    for name in sorted(pack.calculus.rules):
        r, formulas = check_rule(pack, name, n, rng)
        pool += formulas
        k = max(r["n"], 1)
        schema = prove_rule(pack, pack.calculus.rules[name])
        ok &= r["sorts"] == "ok" and not r["unsound"] and r["n"] > 0 and schema != "refuted"
        unproved.update({name: schema} if schema not in ("proved", "refuted") else {})
        print(f"  {name:28} {'ok' if r['sorts'] == 'ok' else 'CLASH':6} "
              f"{r['n'] - len(r['unsound'])}/{r['n']:<4} {schema[:9]:>9} {r['required'] / k:>8.0%} {_pct(r['x2'], r['n2']):>5} {r['bypass'] / k:>7.0%}  "
              f"{', '.join(sorted(r['bypassed_by'])) or '-'}", file=out)
        if schema == "refuted":
            print("    UNSOUND SCHEMA: some instance of the rule is invalid", file=out)
        if r["sorts"] != "ok":
            print(f"    sort clash: {r['sorts']}", file=out)
        for p, c in r["unsound"][:2]:
            print(f"    UNSOUND: {p} |- {c}", file=out)
        if r["n"] == 0:
            print("    could not derive any instance", file=out)
        elif r["required"] == 0:  # not a failure: a composite rule kept on purpose is a one-rule shortcut
            print(f"    NEVER REQUIRED: a composite of {', '.join(sorted(r['bypassed_by']))}", file=out)
    amb = {}
    for _ in range(5):
        amb.update(ambiguities(pack, rng.sample(pool, min(len(pool), 12)), rng))  # within lexicon sizes
    print(f"  rendering: {len(amb)} ambiguous sentences", file=out)
    for s, fs in list(amb.items())[:3]:
        print(f"    {s!r} <- {sorted(fs, key=repr)}", file=out)
    hyps = pool + [pack.negate(f) for f in pool]  # hypotheses are also negated for contradictions
    bad = unfaithful_surfaces(pack, hyps)
    print(f"  surface forms: {sum(pack.surface(f, 'hypothesis') != f for f in set(hyps))} rewritten, "
          f"{len(bad)} not equivalent", file=out)
    for role, f, g in bad[:3]:
        print(f"    {role}: {f} -> {g}", file=out)
    if unproved:
        print(f"  schemas not proved for every instance: {unproved}", file=out)
    wrong = interpreted_disagreements(pack, rng)
    if pack.calculus.background_set.interpreted:
        print(f"  interpreted symbols: {len(wrong)} facts where holds and z3 disagree", file=out)
    for f in wrong[:3]:
        print(f"    {f}", file=out)
    return ok and not amb and not bad and not wrong


def main(argv=None):
    from .packs import NAMES, get
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("packs", nargs="*", default=list(NAMES))
    ap.add_argument("-n", type=int, default=20, help="derivations per rule")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    ok = all([check(get(p), args.n, args.seed) for p in args.packs])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
