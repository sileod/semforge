"""Paired per-seed deltas of a transfer run, as reported in the paper.

  python scripts/paired_deltas.py results/transfer/run9_deberta

Per target: each arm's mean gain over the target-only baseline, and paired differences (mean, sample
standard deviation, seeds with a positive difference) for necessary - bypass and constructed - FLD×2.
"""
import csv
import statistics as st
import sys
from collections import defaultdict

PAIRS = [("src_necessary", "src_bypass"), ("src_constructed_monadic_diverse_all", "src_fld_diverse_all")]


def main(run):
    acc, base = defaultdict(dict), {}
    for r in csv.DictReader(open(f"{run}/results.csv")):
        acc[r["target"], r["source"]][r["seed"]] = float(r["treatment"])
        base[r["target"], r["seed"]] = float(r["baseline"])
    for tgt in dict.fromkeys(t for t, _ in acc):
        gains = {s: 100 * st.mean(v - base[tgt, k] for k, v in acc[tgt, s].items())
                 for t, s in acc if t == tgt}
        parts = [f"{s.replace('src_', '')} {g:+.1f}" for s, g in gains.items()]
        for a, b in PAIRS:
            d = [100 * (acc[tgt, a][k] - acc[tgt, b][k]) for k in acc[tgt, a]]
            parts.append(f"{a.replace('src_', '')}-{b.replace('src_', '')} {st.mean(d):+.1f}±{st.stdev(d):.1f} "
                         f"({sum(x > 0 for x in d)}/{len(d)})")
        print(f"{tgt}: " + " | ".join(parts))


if __name__ == "__main__":
    main(sys.argv[1])
