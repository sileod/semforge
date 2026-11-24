"""Generate certified entailment rows with dependency patterns, for any domain, in parallel.

  python -m semforge.pipeline.rows --domain csp --levels 0 2 --per_level 2000 --out results/patterns/csp

writes rows.jsonl (one row per instance; certified variants are extra rows, variant="enforced" or
"expanded") and audit.json (pattern coverage per family, natural vs variants).
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import random

from semforge.core import patterns as P
from semforge.domains import base

from .common import certified, entropy, pattern_key, required_size, write_jsonl


def row(domain, inst, label, pattern, level, variant, world):
    prem, hyp = domain.render(inst)
    p = pattern
    return {"domain": domain.name, "family": domain.family(inst), "world": world, "level": level, "label": label,
            "premises": prem, "hypothesis": hyp, "variant": variant,
            "signature": [list(x) for x in p.signature] if p else None, "rung": p.rung if p else None,
            "n_justifications": len(p.justifications) if p else None,
            "complete": p.complete if p else None, "n_premises": domain.n_premises(inst),
            # justification structure: sizes of the minimal supports and of their common core
            "support_sizes": sorted(map(len, p.justifications)) if p else None,
            "core_size": len(frozenset.intersection(*map(frozenset, p.justifications))) if p else None,
            **domain.annotations(inst, p)}


def generate(domain, level, seed, n, variants=True):
    """Rows from n sample calls. Each call is one world: its questions and their variants share an id."""
    rng, seen, out = random.Random(seed), collections.Counter(), []
    for call in range(n):
        for inst, label in domain.sample(rng, level):
            world = domain.world(inst) or f"{domain.name}:{seed}:{call}"
            p = P.analyze(domain, inst, label)
            if label != "neutral" and p is None:
                continue
            out.append(row(domain, inst, label, p, level, "natural", world))
            if p is None or not variants:
                continue
            out += [row(domain, e, label, q, level, "enforced", world)
                    for e, q in P.enforced_variants(domain, inst, label, p)]
            e = inst
            for _ in range(2):
                e = domain.expand(e, label, rng, seen) or e
            if e is not inst and domain.label(e) == label:
                q = P.analyze(domain, e, label)
                if q is not None:
                    out.append(row(domain, e, label, q, level, "expanded", world))
    return out


def _job(args):
    name, kw, level, seed, n, variants = args
    return generate(base.get(name, **kw), level, seed, n, variants)


def audit(rows):
    out = {}
    for fam in sorted({r["family"] for r in rows}):
        ec = [r for r in rows if r["family"] == fam and certified(r)]
        nat = [r for r in ec if r["variant"] == "natural"]
        if not nat:
            continue
        cn = collections.Counter(map(pattern_key, nat))
        cv = collections.Counter(pattern_key(r) for r in ec if r["variant"] != "natural")
        out[fam] = {"natural": len(nat), "variants": sum(cv.values()), "patterns": len(cn),
                    "new_patterns_from_variants": len(set(cv) - set(cn)),
                    "entropy_bits": round(entropy(cn), 2), "top_share": round(cn.most_common(1)[0][1] / len(nat), 3),
                    "multi_justification": round(sum(r["n_justifications"] > 1 for r in nat) / len(nat), 3),
                    "search_share": round(sum(r["rung"] == "search" for r in nat) / len(nat), 3),
                    "mean_required": round(sum(map(required_size, nat)) / len(nat), 2),
                    "top_patterns": [[" ".join(f"{k}x{v}" for k, v in s) + (f" [{rg}]" if rg else ""), c]
                                     for (s, rg), c in cn.most_common(4)]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=base.NAMES, required=True)
    ap.add_argument("--families", nargs="*", default=None, help="csp only")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 2])
    ap.add_argument("--per_level", type=int, default=2000, help="sample calls per level")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--no_variants", action="store_true")
    ap.add_argument("--workers", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    kw = {"families": args.families} if args.families else {}
    base.get(args.domain, **kw)  # warm dataset caches once; forked workers inherit them
    jobs = [(args.domain, kw, lv, args.seed * 1_000_003 + lv * 10_007 + i, args.chunk, not args.no_variants)
            for lv in args.levels for i in range(args.per_level // args.chunk)]
    rows = []
    with mp.Pool(args.workers) as pool:
        for part in pool.imap_unordered(_job, jobs):
            rows += part
    write_jsonl(f"{args.out}/rows.jsonl", rows)
    a = audit(rows)
    json.dump(a, open(f"{args.out}/audit.json", "w"), indent=1)
    print(json.dumps(a, indent=1))


if __name__ == "__main__":
    main()
