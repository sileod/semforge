"""Audit comparison at an equal size distribution: sources are subsampled so that every smallest-
justification size (premises the label needs, capped at 10) occurs equally often in each, then distinct
required-kind multisets are counted. Reads the per-instance rows `python -m semforge.audit` writes
(in $AUDIT_ROOT, default results/audit).

  python -m benchmarks.audit_matched fld constructed_monadic_forward constructed_monadic_shape
"""
import collections
import json
import os
import random
import sys


def rows(name, root=os.environ.get("AUDIT_ROOT", "results/audit")):
    return [r for r in json.load(open(f"{root}/{name}_rows.json")) if "error" not in r and r["complete"]]


def matched(names, seed=0):
    R = {s: rows(s) for s in names}
    by = {s: collections.defaultdict(list) for s in names}
    for s in names:
        for r in R[s]:
            by[s][min(r["min_support"], 10)].append(r)
    common = {k: min(len(by[s][k]) for s in names) for k in set().union(*map(set, by.values()))}
    rng, out = random.Random(seed), {}
    for s in names:
        sub = [r for k, m in sorted(common.items()) if m for r in rng.sample(by[s][k], m)]
        out[s] = {"n": len(sub), "min_support": round(sum(r["min_support"] for r in sub) / max(len(sub), 1), 2),
                  "distinct_required_kinds": len({json.dumps(r["required"]) for r in sub}),
                  "unique_justification": round(sum(r["supports"] == 1 for r in sub) / max(len(sub), 1), 2)}
    return out


if __name__ == "__main__":
    print(json.dumps(matched(sys.argv[1:]), indent=1))
