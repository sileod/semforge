"""ModalBench's natural-language track (Hasan et al., ICLR 2026; github.com/mujtabahasan/modalbench, MIT)
as test rows in the LLM experiments' format, to score trained models zero-shot on modal model checking.

Each item states a full Kripke model (rooms or jurisdictions, who sees or influences whom, the facts in
each) and asks whether a formula holds at one world. The model description and the operator glosses are
the premises, the question as a statement is the hypothesis; True is entailment and False contradiction
(the model is complete, so there is no neutral row). `mandatory` is None (no certified depth); `system`,
`tier` and `modal_depth` are kept for analysis.

With `--dev`, a fixed third of the items (by a hash of the id) goes to `--dev-out` for selecting training
distributions, and `--out` keeps the other two thirds; `is_dev` decides the split for earlier full-set runs.

  python -m experiments.modalbench_rows ~/data/modalbench/repo/data/bench_final.json \
      --out results/llm_mix/data/test_modalbench.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re

from experiments.llm_common import FORMAT


def row(item):
    text = item["nl_description"]
    body, _, rest = text.partition("\nQuestion: ")
    question, gloss = rest.split("\n")[:2]
    hypothesis = re.fullmatch(r"Is it true that (.+)\?", question).group(1)
    premises = f"{body.strip()}\n{gloss.strip()}"
    return {"gold": "entailment" if item["ground_truth"] else "contradiction", "mandatory": None,
            "id": item["id"], "system": item["system"], "tier": item["tier"], "modal_depth": item["modal_depth"],
            "text": FORMAT.format(premises=premises, hypothesis=hypothesis[0].upper() + hypothesis[1:] + ".")}


def is_dev(item_id) -> bool:
    return int(hashlib.md5(str(item_id).encode()).hexdigest(), 16) % 3 == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench")
    ap.add_argument("--out", default="results/llm_mix/data/test_modalbench.jsonl")
    ap.add_argument("--dev", action="store_true", help="split off a dev third (see the docstring)")
    ap.add_argument("--dev-out", default="results/llm_mix/data/val_modalbench.jsonl")
    args = ap.parse_args()
    rows = [row(x) for x in json.load(open(args.bench)) if x["presentation"] == "nl"]
    splits = {args.out: [r for r in rows if not (args.dev and is_dev(r["id"]))]}
    if args.dev:
        splits[args.dev_out] = [r for r in rows if is_dev(r["id"])]
    for path, rs in splits.items():
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rs)
        print(path, len(rs), dict(collections.Counter(r["gold"] for r in rs)))


if __name__ == "__main__":
    main()
