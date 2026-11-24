"""Do LLM errors grow with certified depth on constructed rows?

The FLD×2 experiment (experiments/llm_realized_depth.py) found errors following wording, not proof
length. Constructed shape rows remove both confounds: their minimum proof is the planted one (every
step mandatory) and one renderer words every row. Premise count grows with depth, so it is a covariate.

  python -m experiments.llm_constructed_depth --pack monadic --n 600 --model albert/deepseek-v4-flash-0731
"""
import argparse
import collections
import json
import os
import random

from experiments.llm_realized_depth import KEYS, PROMPT, accuracy, parse
from semforge.domains.constructed import ConstructedDomain


def build(pack, n, seed):
    d, rng, rows = ConstructedDomain(pack, strategy="shape"), random.Random(seed), []
    while len(rows) < n:
        for inst, label in d.sample(rng, 0):
            premises, hypothesis = d.render(inst)
            a = inst.analysis
            rows.append({"i": len(rows), "gold": label, "mandatory": len(a.mandatory_steps) if a else None,
                         "spine": a.topology["spine"] if a else None, "premises": len(inst.premises),
                         "prompt": PROMPT.format(premises=premises, hypothesis=hypothesis)})
    return rows


def logistic(rows, boot=500, seed=0):
    """correct ~ mandatory steps + premise count (standardized), E/C rows, bootstrap 95% intervals."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    ok = [r for r in rows if r["pred"] is not None and r["gold"] != "neutral"]
    X = np.array([[r["mandatory"], r["premises"]] for r in ok], float)
    y = np.array([r["pred"] == r["gold"] for r in ok], int)
    if len(set(y)) < 2:  # no errors (or no successes): nothing to regress
        return None
    X = (X - X.mean(0)) / X.std(0)
    fit = lambda X, y: LogisticRegression(C=1e6).fit(X, y).coef_[0]
    rng, coefs = np.random.default_rng(seed), []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            coefs.append(fit(X[i], y[i]))
    lo, hi = np.percentile(coefs, [2.5, 97.5], axis=0) if coefs else ([None] * 2, [None] * 2)
    return {k: {"coef": round(float(c), 3), "ci95": [a and round(float(a), 3), b and round(float(b), 3)]}
            for k, c, a, b in zip(("mandatory", "premises"), fit(X, y), lo, hi)}


def summarize(rows):
    ec = [r for r in rows if r["gold"] != "neutral"]
    bucket = lambda m: "1-3" if m <= 3 else "4-7" if m <= 7 else "8-12" if m <= 12 else "13+"
    return {"all": accuracy(rows), "by_label": {k: accuracy([r for r in rows if r["gold"] == k])
                                               for k in ("entailment", "contradiction", "neutral")},
            "by_mandatory": {b: accuracy([r for r in ec if bucket(r["mandatory"]) == b]) for b in ("1-3", "4-7", "8-12", "13+")},
            "errors": dict(collections.Counter(f"{r['gold']}->{r['pred']}" for r in rows
                                               if r["pred"] is not None and r["pred"] != r["gold"])),
            "logistic": logistic(rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default="monadic")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="albert/deepseek-v4-flash-0731")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--rpm", type=int, default=45, help="per Albert key")
    ap.add_argument("--out", default="results/llm_depth")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows_path = f"{args.out}/constructed_{args.pack}_s{args.seed}_n{args.n}.jsonl"
    if os.path.exists(rows_path):
        rows = [json.loads(line) for line in open(rows_path)]
    else:
        rows = build(args.pack, args.n, args.seed)
        with open(rows_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    print(len(rows), "rows", collections.Counter(r["gold"] for r in rows), flush=True)

    import litlm
    albert = args.model.startswith("albert/")
    replies = litlm.complete([r["prompt"] for r in rows], model=args.model,
                             api_key_envs=KEYS if albert else None, per_key_rpm=args.rpm if albert else None,
                             num_retries=2, max_tokens=8192, timeout=300, caching=True,
                             reasoning_effort=args.reasoning or None)
    tag = f"constructed_{args.pack}_" + args.model.split("/")[-1] + (f"-{args.reasoning}" if args.reasoning else "")
    for r, t in zip(rows, replies):
        r["pred"], r["reply_tail"] = parse(t), str(t)[-300:]
    with open(f"{args.out}/{tag}.jsonl", "w") as f:
        f.writelines(json.dumps({k: v for k, v in r.items() if k != "prompt"}) + "\n" for r in rows)
    summary = {"model": args.model, "reasoning": args.reasoning, "rows": rows_path, **summarize(rows)}
    json.dump(summary, open(f"{args.out}/{tag}.summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
