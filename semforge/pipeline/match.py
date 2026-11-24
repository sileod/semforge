"""Source arms from different generators, matched on certified realized difficulty: the causal comparison.

  python -m semforge.pipeline.match --rows forward=results/patterns/constructed_monadic_forward/rows.jsonl \
      shape=results/patterns/constructed_monadic_shape/rows.jsonl --out results/transfer/data_causal

Every generator's rows come out of the same pipeline (same pack, realizer, labels, post-hoc certificate),
so arms differ only in how structure was chosen. Per generator g:
  g_matched  E/C rows equalized across generators within every stratum of *certified* structure:
             (label, mandatory rule applications, spine, premises // 3), from the post-hoc certificate.
             Equal realized difficulty and input length; what differs is whether the planted steps are necessary, and which
             proof shapes and kinds are present.
  g_natural  E/C rows drawn from the generator's own distribution, equal in number only.
All arms share the same neutral rows (from --neutral_from) and have the same size.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

from .common import certified, read_jsonl, required_size, write_jsonl


def stratum(r, cap_size=14, cap_spine=8):
    """Certified difficulty, plus input length (premises, in buckets of 3) so that distractor load
    is matched too."""
    c = r["certified"]
    return r["label"], min(required_size(r), cap_size), min(c["spine"], cap_spine), min(r["n_premises"] // 3, 6)


def usable(r):
    return r["label"] != "neutral" and certified(r) and r.get("certified") and r["variant"] == "natural"


def stats(rows):
    ec = [r for r in rows if r["label"] != "neutral"]
    mean = lambda f: round(sum(map(f, ec)) / max(len(ec), 1), 2)
    return {"n": len(rows), "ec": len(ec), "mandatory": mean(required_size), "spine": mean(lambda r: r["certified"]["spine"]),
            "joins": mean(lambda r: r["certified"]["joins"]), "proofs": mean(lambda r: r["n_justifications"]),
            "premises": mean(lambda r: r["n_premises"]),
            "patterns": len({tuple(map(tuple, r["signature"])) for r in ec}),
            "kind_sets": len({frozenset(k for k, _ in r["signature"]) for r in ec})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True, help="name=path/to/rows.jsonl, one per generator")
    ap.add_argument("--n_ec", type=int, default=1600)
    ap.add_argument("--n_neutral", type=int, default=800)
    ap.add_argument("--neutral_from", default=None, help="generator whose neutral rows all arms share (default: first with any)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    rows = {}
    for spec in args.rows:
        name, path = spec.split("=", 1)
        seen, rows[name] = set(), []
        for r in read_jsonl(path):
            key = (r["premises"], r["hypothesis"])
            if key not in seen:
                seen.add(key)
                rows[name].append(r)
    ec = {g: [r for r in rs if usable(r)] for g, rs in rows.items()}
    neutral_from = args.neutral_from or next(g for g, rs in rows.items() if any(r["label"] == "neutral" for r in rs))
    neutrals = [r for r in rows[neutral_from] if r["label"] == "neutral"]
    neutrals = rng.sample(neutrals, min(args.n_neutral, len(neutrals)))

    by = {g: collections.defaultdict(list) for g in ec}
    for g, rs in ec.items():
        for r in rs:
            by[g][stratum(r)].append(r)
    strata = sorted(set.intersection(*(set(b) for b in by.values())), key=repr)
    quota = {s: min(len(by[g][s]) for g in by) for s in strata}
    total = sum(quota.values())
    if total > args.n_ec:  # scale every stratum down alike
        quota = {s: q * args.n_ec // total for s, q in quota.items()}
    n_ec = sum(quota.values())

    os.makedirs(args.out, exist_ok=True)
    report = {"neutral_from": neutral_from, "strata": len(strata), "n_ec": n_ec, "arms": {}, "pools": {}}
    for g in ec:
        report["pools"][g] = stats(ec[g])
        matched = [r for s in strata for r in rng.sample(by[g][s], quota[s])]
        natural = rng.sample(ec[g], min(n_ec, len(ec[g])))
        for arm, sel in ((f"{g}_matched", matched), (f"{g}_natural", natural)):
            out = sel + neutrals
            rng.shuffle(out)
            write_jsonl(f"{args.out}/src_{arm}.jsonl", out)
            report["arms"][arm] = stats(out)
    json.dump(report, open(f"{args.out}/match.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
