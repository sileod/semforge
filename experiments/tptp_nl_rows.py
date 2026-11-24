"""Our TPTP/QMLTP NLI benchmarks (~/libs/tptp_nl, Vampire-certified labels) as test rows in the LLM
experiments' format.

- test_tptp: tptp_nli_llm (Sonnet English of TPTP problems; base, negated and premise-dropped variants:
  E/C/N), every row whose premises and hypothesis fit `--max-chars`.
- test_qmltp: qmltp_nli (first-order modal logic, K/D/T/S4/S5 x domain conditions, rule-based English;
  the logic's English description precedes the premises), `--qmltp` rows sampled evenly over labels.

`mandatory` is None (no certified step count of ours). Rows longer than `--max-chars` are dropped (the
LLM runs truncate at 768 tokens), and the counts are printed.

  python -m experiments.tptp_nl_rows ~/libs/tptp_nl/build/bench --out results/llm_mix/data
"""
from __future__ import annotations

import argparse
import collections
import json
import random

from experiments.llm_common import FORMAT


def load(path, max_chars, context=lambda r: ""):
    out, long = [], 0
    for line in open(path):
        r = json.loads(line)
        premises = "\n".join(filter(None, [context(r), *r["premises"]]))
        if len(premises) + len(r["hypothesis"]) > max_chars:
            long += 1
            continue
        out.append({"gold": r["label"], "mandatory": None, "id": r["id"], "problem": r["problem"],
                    "variant": r["variant"], "system": (r.get("logic") or {}).get("system"),
                    "text": FORMAT.format(premises=premises, hypothesis=r["hypothesis"])})
    return out, long


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", help="tptp_nl's build/bench directory")
    ap.add_argument("--out", default="results/llm_mix/data")
    ap.add_argument("--max-chars", type=int, default=2200)
    ap.add_argument("--qmltp", type=int, default=900)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tptp, long_t = load(f"{args.bench}/tptp_nli_llm.jsonl", args.max_chars)
    qmltp, long_q = load(f"{args.bench}/qmltp_nli.jsonl", args.max_chars, lambda r: r["logic_text"])
    by_label, rng = collections.defaultdict(list), random.Random(args.seed)
    for r in qmltp:
        by_label[r["gold"]].append(r)
    qmltp = [r for rs in by_label.values() for r in rng.sample(rs, min(len(rs), args.qmltp // 3))]
    rng.shuffle(qmltp)
    for name, rows, long in (("tptp", tptp, long_t), ("qmltp", qmltp, long_q)):
        with open(f"{args.out}/test_{name}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        print(name, len(rows), dict(collections.Counter(r["gold"] for r in rows)), "dropped as too long", long)


if __name__ == "__main__":
    main()
