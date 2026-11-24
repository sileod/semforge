"""Compare dependency-pattern diversity across sources at matched sample size.

Each source is subsampled to n natural, certified E/C rows `reps` times; metrics are reported as mean
(std). Kinds can be mapped to a shared vocabulary first: ZebraLogic templates to CSP operator names, FLD
skeletons to rule shapes.

  python -m semforge.pipeline.compare --n 1000 --out results/patterns/compare.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics as st

from semforge.domains.fld import rule_shape
from semforge.domains.zebralogic import TEMPLATE_OPERATOR

from .common import certified, entropy, pattern_key, read_jsonl, required_size


def natural(path, keep=lambda r: True, kind_map=None):
    out = []
    for r in read_jsonl(path):
        if r["variant"] != "natural" or not certified(r) or not keep(r):
            continue
        if kind_map:
            c = collections.Counter()
            for k, v in r["signature"]:
                c[kind_map(k)] += v
            r["signature"] = sorted(map(list, c.items()))
        out.append(r)
    return out


def metrics(rs):
    pk = collections.Counter(map(pattern_key, rs))
    kinds = collections.Counter(k for r in rs for k, _ in r["signature"])
    n = len(rs)
    return {"patterns": len(pk), "entropy_bits": entropy(pk), "top_share": pk.most_common(1)[0][1] / n,
            "kinds_used": len(kinds), "kind_entropy_bits": entropy(kinds),
            "multi_justification": sum(r["n_justifications"] > 1 for r in rs) / n,
            "search_share": sum(r["rung"] == "search" for r in rs) / n,
            "mean_required": sum(map(required_size, rs)) / n,
            "mean_premises": sum(r["n_premises"] for r in rs) / n,
            # justification structure: smallest support, and the share of it every justification shares
            "min_support": sum(r["support_sizes"][0] for r in rs) / n,
            "core_share": sum(r["core_size"] / r["support_sizes"][0] for r in rs) / n,
            "structure_entropy_bits": entropy(collections.Counter(
                (tuple(r["support_sizes"][:4]), r["core_size"], r["rung"]) for r in rs))}


def sources(root):
    z, c, f = (f"{root}/{d}/rows.jsonl" for d in ("zebralogic", "csp", "fld"))
    large = lambda r: int(r["family"].split("_")[1].split("x")[0]) >= 5
    out = {"zebralogic (templates)": natural(z),
           "zebralogic (operators)": natural(z, kind_map=TEMPLATE_OPERATOR.get),
           "zebralogic 5x5+ (operators)": natural(z, large, kind_map=TEMPLATE_OPERATOR.get),
           "csp assignment": natural(c, lambda r: r["family"] == "assignment"),
           "csp all families": natural(c),
           "logic": natural(f"{root}/logic/rows.jsonl")}
    if os.path.exists(f):
        out["FLD×2 (skeletons)"] = natural(f)
        out["FLD×2 (rule shapes)"] = natural(f, kind_map=rule_shape)
    return out


def compare(srcs, n, reps, seed=0):
    rng, out = random.Random(seed), {}
    for name, rs in srcs.items():
        if len(rs) < n:
            print(f"skip {name}: {len(rs)} rows")
            continue
        ms = [metrics(rng.sample(rs, n)) for _ in range(reps)]
        out[name] = {"pool": len(rs), **{k: (round(st.mean(m[k] for m in ms), 3), round(st.pstdev(m[k] for m in ms), 3))
                                         for k in ms[0]}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--root", default="results/patterns")
    ap.add_argument("--out", default="results/patterns/compare.json")
    args = ap.parse_args(argv)
    out = compare(sources(args.root), args.n, args.reps)
    json.dump(out, open(args.out, "w"), indent=1)
    keys = ["pool", "patterns", "entropy_bits", "top_share", "kinds_used", "multi_justification", "search_share",
            "mean_required", "mean_premises", "min_support", "core_share", "structure_entropy_bits"]
    print("source".ljust(30) + "".join(k[:12].rjust(13) for k in keys))
    for name, m in out.items():
        print(name.ljust(30) + "".join((str(m[k]) if k == "pool" else f"{m[k][0]:.3g}").rjust(13) for k in keys))


if __name__ == "__main__":
    main()
