"""LogicGraph (EMNLP 2026) at the premise level: is each instance's "exhaustive set of minimal proofs" exhaustive?

LogicGraph instances are propositional: a knowledge base (rules and facts, `formal`), a target G, and a
list of solutions whose steps cite knowledge-base premises by id (intermediate conclusions are
`Derived`). The audit compares, per instance:
- the distinct premise sets the listed solutions use (several proofs can share one);
- every minimal premise support of G, enumerated by MARCO over the knowledge base with not-G
  (SAT, pysat, on clauses from z3):
  semantic, independent of any calculus;
and reports listed sets that entail G, listed sets that are minimal, minimal supports no solution uses
(missed paths, at the premise level), and premises every support needs (mandatory). Proofs differing
only in steps over the same premises are one support here, so the premise level counts at most as many
paths as LogicGraph's step level. Enumeration is capped (`--cap` supports,
`--budget-s` seconds); an incomplete instance is reported as such.

  python -m benchmarks.audit_logicgraph --out results/audit/logicgraph.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import threading
from collections import Counter

import z3

from benchmarks.audit_proverqa import Parser, Unparsed


class Inconsistent(ValueError):
    pass


def parse(text, env):
    return Parser(text, env).parse()


def atoms(f):
    out, todo = set(), [f]
    while todo:
        e = todo.pop()
        if z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            out.add(str(e))
        todo.extend(e.children())
    return out


def relevant(premises, goal):
    """Premises in the connected component of the goal's atoms (premises linked by shared atoms). For a
    consistent knowledge base, no premise outside it is in a minimal support: the rest is satisfiable
    on its own atoms, so removing it keeps the support unsatisfiable with not-G."""
    reach, keep, grew = atoms(goal), set(), True
    while grew:
        grew = False
        for i, f in premises.items():
            if i not in keep and atoms(f) & reach:
                keep.add(i)
                reach |= atoms(f)
                grew = True
    return {i: premises[i] for i in keep}


def minimal_supports_sat(premises, goal, cap, budget_s=None):
    """minimal_supports on a SAT solver (propositional premises): Tseitin clauses from z3, MARCO with
    assumptions, unsat cores to shrink and seeds biased toward selecting every premise. Stops after
    `budget_s` seconds (then incomplete). (supports, complete)"""
    from pysat.solvers import Solver
    ids = sorted(premises)
    sel = {i: z3.Bool(f"sel_{i}") for i in ids}
    g = z3.Goal()
    g.add(z3.Not(goal), *(z3.Implies(sel[i], premises[i]) for i in ids))
    var = {}
    lit = lambda e: -lit(e.arg(0)) if z3.is_not(e) else var.setdefault(str(e), len(var) + 1)
    clauses = []
    for sub in z3.Tactic("tseitin-cnf")(g):
        for c in sub:
            if z3.is_false(c):
                clauses.append([])
                continue
            clauses.append([lit(x) for x in (c.children() if z3.is_or(c) else [c])])
    sv = {i: var.setdefault(str(sel[i]), len(var) + 1) for i in ids}
    back = {v: i for i, v in sv.items()}
    check, seeds = Solver(name="g4", bootstrap_with=clauses), Solver(name="g4")
    seeds.add_clause([sv[i] for i in ids] + [-sv[i] for i in ids])  # declare the selector variables
    seeds.set_phases([sv[i] for i in ids])
    timer = threading.Timer(budget_s, lambda: (check.interrupt(), seeds.interrupt())) if budget_s else None
    timer and timer.start()

    def solve(solver, assumptions=()):
        r = solver.solve_limited(assumptions=assumptions, expect_interrupt=True)
        if r is None:
            raise TimeoutError
        return r

    out, complete = [], False
    try:
        while len(out) < cap:
            if not solve(seeds):
                complete = True
                break
            m = set(x for x in seeds.get_model() if x > 0)
            chosen = [sv[i] for i in ids if sv[i] in m]
            if solve(check, chosen):  # not a support: grow to a maximal non-support, block its subsets
                grown = set(chosen)
                for i in ids:
                    if sv[i] not in grown and solve(check, list(grown | {sv[i]})):
                        grown.add(sv[i])
                seeds.add_clause([sv[i] for i in ids if sv[i] not in grown])
            else:  # a support: shrink through cores to a minimal one, block its supersets
                core = set(check.get_core() or chosen)
                for x in list(core):
                    if x in core and not solve(check, list(core - {x})):
                        core = set(check.get_core() or (core - {x}))
                out.append(frozenset(back[x] for x in core))
                seeds.add_clause([-x for x in core])
    except TimeoutError:
        pass
    finally:
        timer and timer.cancel()
    return out, complete


def minimal_supports(premises, goal, cap, timeout_ms):
    """All minimal subsets of `premises` (dict id -> z3 formula) entailing `goal`, by MARCO; (supports, complete)."""
    ids = sorted(premises)
    sel = {i: z3.Bool(f"sel_{i}") for i in ids}
    check = z3.Solver()
    check.set("timeout", timeout_ms)
    check.add(z3.Not(goal), *(z3.Implies(sel[i], premises[i]) for i in ids))
    seeds = z3.Solver()  # subsets not yet covered: not a superset of a found support, not a subset of a non-support
    out = []
    while len(out) < cap and seeds.check() == z3.sat:
        m = seeds.model()
        chosen = [i for i in ids if not z3.is_false(m.eval(sel[i], model_completion=True))]
        r = check.check(*(sel[i] for i in chosen))
        if r == z3.unknown:
            raise TimeoutError
        if r == z3.sat:  # not a support: grow it to a maximal non-support, then block its subsets
            grown = set(chosen)
            for i in ids:
                if i not in grown and check.check(*(sel[j] for j in grown | {i})) == z3.sat:
                    grown.add(i)
            seeds.add(z3.Or([sel[i] for i in ids if i not in grown]))
        else:  # a support: shrink it to a minimal one, then block its supersets
            core = set(chosen)
            for i in list(core):
                if check.check(*(sel[j] for j in core - {i})) == z3.unsat:
                    core.discard(i)
            out.append(frozenset(core))
            seeds.add(z3.Or([z3.Not(sel[i]) for i in core]))
    return out, seeds.check() == z3.unsat


def audit(directory, cap, timeout_ms, budget_s=None):
    c = json.load(open(f"{directory}/complete.json"))
    s = json.load(open(f"{directory}/solutions.json"))
    env = {"sort": z3.DeclareSort("U"), "preds": {}, "consts": {}}
    kb = {p["id"]: parse(p["formal"], env) for p in c["knowledge_base"]["rules"] + c["knowledge_base"]["facts"]}
    goal = parse(c["target"]["formal"], env)
    listed = set()
    for sol in s["solutions"]:
        listed.add(frozenset(p["id"] for st in sol["steps"] for p in st["premises_used"]
                             if isinstance(p, dict) and p.get("id") in kb))
    if z3.Solver().check(*kb.values()) == z3.unsat:
        raise Inconsistent
    supports, complete = minimal_supports_sat(relevant(kb, goal), goal, cap, budget_s)
    entails = lambda ids: z3.Solver().check(z3.Not(goal), *(kb[i] for i in ids)) == z3.unsat
    minimal = lambda ids: entails(ids) and not any(entails(ids - {i}) for i in ids)
    mandatory = frozenset.intersection(*supports) if supports else frozenset()
    return {"instance": os.path.relpath(directory, os.path.dirname(os.path.dirname(directory))),
            "tier": directory.rstrip("/").split("/")[-2], "premises": len(kb), "relevant": len(relevant(kb, goal)),
            "solutions": len(s["solutions"]), "listed_sets": len(listed),
            "listed_entail": sum(map(entails, listed)), "listed_minimal": sum(map(minimal, listed)),
            "supports": len(supports), "complete": complete,
            "missed": len(set(supports) - listed), "mandatory": len(mandatory),
            "steps": [len(sol["steps"]) for sol in s["solutions"]]}


def _safe(directory, cap, timeout_ms, budget_s):
    """audit(), or the reason the instance is dropped."""
    if not os.path.exists(f"{directory}/solutions.json"):
        return "no_solutions_file"
    try:
        return audit(directory, cap, timeout_ms, budget_s)
    except Unparsed:
        return "unparsed"
    except TimeoutError:
        return "z3_timeout"
    except Inconsistent:
        return "inconsistent_kb"


def summarize(rows):
    mean = lambda k: round(sum(r[k] for r in rows) / len(rows), 2) if rows else None
    share = lambda f: round(sum(map(f, rows)) / len(rows), 3) if rows else None
    return {"instances": len(rows), "mean_premises": mean("premises"), "mean_solutions": mean("solutions"),
            "mean_listed_sets": mean("listed_sets"), "mean_supports": mean("supports"),
            "listed_sets_entail": round(sum(r["listed_entail"] for r in rows) / max(1, sum(r["listed_sets"] for r in rows)), 3),
            "listed_sets_minimal": round(sum(r["listed_minimal"] for r in rows) / max(1, sum(r["listed_sets"] for r in rows)), 3),
            "instances_missing_a_support": share(lambda r: r["missed"] > 0),
            "mean_missed": mean("missed"), "instances_with_mandatory": share(lambda r: r["mandatory"] > 0),
            "enumeration_complete": share(lambda r: r["complete"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="LogicGraph snapshot (default: the Hub cache)")
    ap.add_argument("--cap", type=int, default=500, help="most supports enumerated per instance")
    ap.add_argument("--timeout-ms", type=int, default=10000)
    ap.add_argument("--budget-s", type=float, default=120, help="enumeration time per instance")
    ap.add_argument("--n", type=int, default=0, help="instances per tier (0: all)")
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--out", default="results/audit/logicgraph.json")
    args = ap.parse_args()
    if args.root is None:
        from huggingface_hub import snapshot_download
        args.root = snapshot_download("kkkarry/LogicGraph", repo_type="dataset",
                                      allow_patterns=["*/complete.json", "*/solutions.json"], max_workers=2)
    out = {"source": "kkkarry/LogicGraph", "tiers": {}, "rows": []}
    for tier in sorted({d.split("/")[-3] for d in glob.glob(f"{args.root}/*/*/complete.json")}):
        dirs = sorted(os.path.dirname(f) for f in glob.glob(f"{args.root}/{tier}/*/complete.json"))
        from concurrent.futures import ProcessPoolExecutor
        dirs = dirs[:args.n or None]
        with ProcessPoolExecutor(args.jobs) as ex:
            results = list(ex.map(_safe, dirs, [args.cap] * len(dirs), [args.timeout_ms] * len(dirs), [args.budget_s] * len(dirs)))
        rows = [r for r in results if isinstance(r, dict)]
        dropped = Counter(r for r in results if isinstance(r, str))
        out["tiers"][tier] = {**summarize(rows), "dropped": dict(dropped)}
        out["rows"] += rows
        print(tier, json.dumps(out["tiers"][tier]), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
