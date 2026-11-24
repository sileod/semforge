"""Training rows from existing synthetic logic datasets, for a matched-size comparison with ours.

Each source gives `--n` training rows, a third per label, in the format of every other arm
(experiments.llm_common.FORMAT), and, for sources with no test file yet, a held-out test set:
  xfld          FLD×2 train split, rendered as test_fld is (semforge.domains.fld; PROVED/DISPROVED/UNKNOWN)
  xproverqa     ProverQA train (5000 rows); test_proverqa from its dev set (easy/medium/hard, 200 each)
  xproofwriter  ProofWriter OWA depth-5 train, at most one question per theory and label; test_proofwriter
                from its test split (600 rows, a third per label)
Contexts keep their own wording; ProverQA and ProofWriter put one sentence per line, as our rows do.
Arms are then trained like any other (scripts/queue_packmix.sh with these arm names); ours at the same
size is the dev-selected 3000-row arm (pmdk200).

  python -m experiments.external_rows --out results/llm_mix/data_packmix
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re

from experiments.llm_common import FORMAT, LABELS

PROVERQA = "opendatalab/ProverQA"
PROOFWRITER = os.path.expanduser("~/data/proofwriter-dataset-V2020.12.3/OWA/depth-5")


def lines(context):
    return "\n".join(s.strip() for s in re.split(r"(?<=\.)\s+", context.strip()) if s.strip())


def row(premises, hypothesis, gold, **meta):
    return {"gold": gold, "mandatory": None, **meta, "text": FORMAT.format(premises=premises, hypothesis=hypothesis)}


def balanced(rows, n, rng):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["gold"]].append(r)
    out = [r for lab in LABELS for r in rng.sample(by[lab], min(len(by[lab]), n // 3))]
    rng.shuffle(out)
    return out


def fld(n, rng):
    from semforge.domains.fld import FLDDomain, from_row, load
    data, render, out = load("train"), FLDDomain(rows=[]).render, collections.defaultdict(list)
    order = list(range(len(data)))
    rng.shuffle(order)
    for i in order:
        inst = from_row(data[i], i)
        if len(out[inst.gold]) >= n // 3:
            if all(len(out[lab]) >= n // 3 for lab in LABELS):
                break
            continue
        premises, hypothesis = render(inst)
        out[inst.gold].append(row(premises, hypothesis, inst.gold, source="fld", claimed=inst.steps))
    return balanced([r for rs in out.values() for r in rs], n, rng)


def proverqa_row(context, statement, answer, **meta):
    return row(lines(context), statement, {"A": "entailment", "B": "contradiction", "C": "neutral"}[answer], **meta)


def proverqa(n, rng):
    from huggingface_hub import hf_hub_download
    get = lambda f: json.load(open(hf_hub_download(PROVERQA, f, repo_type="dataset")))
    train = []
    for x in get("train/provergen-5000.json"):
        m = re.search(r"Context:\n(.*?)\n\nQuestion:.*?true, false, or uncertain\?\s*(.*?)\n", x["instruction"], re.S)
        a = re.search(r'"answer":\s*"([ABC])"', x["output"])
        if m and a:
            train.append(proverqa_row(m.group(1), m.group(2).strip(), a.group(1), source="proverqa"))
    test = []
    for level in ("easy", "medium", "hard"):
        dev = get(f"dev/{level}.json")
        for x in rng.sample(dev, 200):
            statement = x["question"].split("uncertain?", 1)[1].strip()
            test.append(proverqa_row(x["context"], statement, x["answer"], source="proverqa", level=level))
    return balanced(train, n, rng), test


def proofwriter_rows(path, per_theory, rng):
    gold = {True: "entailment", False: "contradiction", "Unknown": "neutral"}
    out = []
    for line in open(path):
        t = json.loads(line)
        qs = collections.defaultdict(list)
        for q in t["questions"].values():
            qs[gold[q["answer"]]].append(q)
        for lab, cands in qs.items():  # at most `per_theory` questions per theory and label
            for q in rng.sample(cands, min(per_theory, len(cands))):
                out.append(row(lines(t["theory"]), q["question"], lab, source="proofwriter", qdep=q["QDep"]))
    return out


def proofwriter(n, rng):
    return (balanced(proofwriter_rows(f"{PROOFWRITER}/meta-train.jsonl", 1, rng), n, rng),
            balanced(proofwriter_rows(f"{PROOFWRITER}/meta-test.jsonl", 1, rng), 600, rng))


def write(path, rows):
    with open(path + ".tmp", "w") as f:  # renamed when complete: running jobs glob test_*.jsonl
        f.writelines(json.dumps(r) + "\n" for r in rows)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/llm_mix/data_packmix")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    stats = {}
    train = {"xfld": fld(args.n, random.Random(f"{args.seed}:fld"))}
    train["xproverqa"], test_pq = proverqa(args.n, random.Random(f"{args.seed}:proverqa"))
    train["xproofwriter"], test_pw = proofwriter(args.n, random.Random(f"{args.seed}:proofwriter"))
    for name, rows in [*((f"train_{k}", v) for k, v in train.items()), ("test_proverqa", test_pq),
                       ("test_proofwriter", test_pw)]:
        write(f"{args.out}/{name}.jsonl", rows)
        stats[name] = {"rows": len(rows), "labels": collections.Counter(r["gold"] for r in rows),
                       "mean_chars": round(sum(len(r["text"]) for r in rows) / len(rows))}
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
