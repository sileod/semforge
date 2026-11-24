"""ProofWriter at the proof level: does claimed depth (QDep) mean mandatory steps?

  python -m benchmarks.audit_proofwriter --data <proofwriter>/OWA/depth-5/meta-test.jsonl --per-depth 200

Uses ProofWriter's own logical forms: signed literals ("cow" "is" "big" "-") and rules over one variable
("something" / "someone"). Under the open-world reading a negative literal is a fact of its own, so each
theory is a ground Horn program over signed literals. Rules are grounded over the theory's entities and
chased exactly. For a True answer the goal is the question's literal; for False, its sign flipped.
Per question:
  - mandatory steps: ground rule applications (rule, binding) in every derivation, by removal;
  - mandatory facts: theory facts in every derivation, by removal;
  - whether the chase reproduces ProofWriter's label (a check of the parse);
  - the proof profile (semforge.audit.PROFILE): shallowest depth, fewest steps and number of minimal
    proofs, from a provenance chase over the ground steps (exact unless a support front hits its cap).
Candidates are the steps and facts of one derivation, so each needs one extra chase.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict

from semforge.audit import summarize_profiles
from semforge.core.provenance import provenance_chase

VARS = {"something", "someone"}


def literal(rep):
    """(subject, relation, object, sign); some OWA rules write a negated literal with "~" instead of "-"."""
    f = tuple(re.findall(r'"([^"]*)"', rep))
    return f[:3] + ("-" if f[3] == "~" else f[3],)


def theory(row):
    facts = {literal(t["representation"]) for t in row["triples"].values()}
    rules = []
    for name, r in row["rules"].items():
        body, head = r["representation"].split("->")
        rules.append((name, [literal(x) for x in re.findall(r"\([^()]*\)", body)], literal(head)))
    entities = sorted({a for f in facts for a in (f[0], f[2])} | {a for _, b, h in rules for f in b + [h]
                                                                  for a in (f[0], f[2])} - VARS)
    return facts, rules, entities


def ground(rules, entities):
    """[(step id, body, head)]; a step id is (rule name, binding)."""
    out = []
    for name, body, head in rules:
        uses = any(a in VARS for f in body + [head] for a in (f[0], f[2]))
        for e in (entities if uses else [None]):
            sub = lambda f: tuple(e if a in VARS else a for a in f)
            out.append(((name, e), [sub(f) for f in body], sub(head)))
    return out


def chase(facts, steps, banned_step=None, banned_fact=None):
    """Closure and, per derived literal, the first step deriving it (earliest round)."""
    have = set(facts) - {banned_fact}
    first, by_body = {}, defaultdict(list)
    for s in steps:
        if s[0] != banned_step:
            for f in s[1]:
                by_body[f].append(s)
    delta = set(have)
    while delta:
        new = {}
        for f in delta:
            for sid, body, head in by_body[f]:
                if head not in have and head not in new and all(b in have for b in body):
                    new[head] = (sid, body)
        have |= set(new)
        first.update(new)
        delta = set(new)
    return have, first


def derivation(goal, first, facts):
    """Steps and facts of one derivation of goal."""
    steps, used, stack, seen = [], set(), [goal], set()
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        if f in facts:
            used.add(f)
            continue
        sid, body = first[f]
        steps.append((sid, body, f))
        stack.extend(body)
    return steps, used


def profile(facts, steps, goal, mandatory, cap=256):
    """Minimal proofs as sets of ground steps and facts; sizes count steps only."""
    prov = provenance_chase(facts, steps, lambda s, have: [(s[2], s[1])] if all(b in have for b in s[1]) else [],
                            lambda x: x if len(x) == 4 else None, cap,  # facts are 4-tuples, steps 3-tuples
                            step_id=lambda s, head, parents: ("step", s[0]))
    supports, exact = prov.minimal_supports(goal, allow_incomplete=True), prov.is_exact(goal)
    steps_of = lambda sup: sum(isinstance(x, tuple) and x[0] == "step" for x in sup)
    facts_of = {frozenset(x for x in sup if not (isinstance(x, tuple) and x[0] == "step")) for sup in supports}
    return {"min_depth": prov.min_depth(goal) if exact else min(prov.supports[goal].values()),
            "min_size": min(map(steps_of, supports)), "mandatory": mandatory, "spine": None, "joins": None,
            "proofs": len(supports), "supports": sum(not any(t < f for t in facts_of) for f in facts_of),
            "exact": exact}


def audit(row, q):
    facts, rules, entities = theory(row)
    steps = ground(rules, entities)
    goal = literal(q["representation"])
    if q["answer"] is False:
        goal = goal[:3] + ("-" if goal[3] == "+" else "+",)
    have, first = chase(facts, steps)
    if goal not in have:
        return {"error": "not derivable"}
    proof, used = derivation(goal, first, facts)
    mandatory = [s for s in proof if goal not in chase(facts, steps, banned_step=s[0])[0]]
    mandatory_facts = [f for f in used if goal not in chase(facts, steps, banned_fact=f)[0]]
    shape = lambda s: f"{len(s[1])}:{''.join(sorted(f[3] for f in s[1]))}>{s[2][3]}"
    return {"qdep": q["QDep"], "answer": q["answer"], "facts": len(facts), "rules": len(rules),
            "steps": len(proof), "mandatory": len(mandatory), "mandatory_facts": len(mandatory_facts),
            "used_facts": len(used), "kinds": sorted(Counter(map(shape, mandatory)).items()),
            "rule_names": sorted(s[0][0] for s in mandatory),
            "profile": profile(facts, steps, goal, len(mandatory))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--per-depth", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    items = [(r, q) for line in open(args.data) for r in [json.loads(line)] for q in r["questions"].values()
             if q["answer"] in (True, False)]
    random.Random(args.seed).shuffle(items)
    by_dep, t = defaultdict(list), time.time()
    for r, q in items:
        if len(by_dep[q["QDep"]]) < args.per_depth:
            by_dep[q["QDep"]].append(audit(r, q))
    out = {}
    for d, rows in sorted(by_dep.items()):
        ok = [x for x in rows if "error" not in x]
        out[d] = {"n": len(rows), "errors": len(rows) - len(ok),
                  "claimed_depth": d, "proof_steps": round(statistics.mean(x["steps"] for x in ok), 2),
                  "mandatory_steps": round(statistics.mean(x["mandatory"] for x in ok), 2),
                  "zero_mandatory": round(sum(x["mandatory"] == 0 for x in ok) / len(ok), 3),
                  "mandatory_below_claim": round(sum(x["mandatory"] < d for x in ok) / len(ok), 3),
                  "mandatory_facts": round(statistics.mean(x["mandatory_facts"] for x in ok), 2),
                  "distractor_share": round(statistics.mean(1 - x["used_facts"] / x["facts"] for x in ok), 3),
                  "kind_sets": len({json.dumps(x["kinds"]) for x in ok}),
                  "profile": summarize_profiles([x["profile"] for x in ok], planted=[d] * len(ok))}
        print(d, out[d], flush=True)
    allrows = [x for rows in by_dep.values() for x in rows if "error" not in x]
    summary = {"n": len(allrows), "mandatory_steps": round(statistics.mean(x["mandatory"] for x in allrows), 2),
               "claimed_depth": round(statistics.mean(x["qdep"] for x in allrows), 2),
               "zero_mandatory": round(sum(x["mandatory"] == 0 for x in allrows) / len(allrows), 3),
               "profile": summarize_profiles([x["profile"] for x in allrows], planted=[x["qdep"] for x in allrows]),
               "seconds": round(time.time() - t, 1)}
    print(summary)
    if args.out:
        json.dump({"by_depth": out, "summary": summary, "rows": [x for rows in by_dep.values() for x in rows]},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
