"""Does LLM accuracy on FLD×2 follow the depth it claims, or the depth it realizes?

FLD×2 rows state a proof length (`steps`); in our shared calculus many rows have a shorter proof
(benchmarks/audit_fld_proofs.py). Same rows, same English, so the two depths can be compared without a
change of source: within each claimed length, is a row with a short minimum proof easier than one
whose minimum proof is as long as claimed?

  python -m experiments.llm_realized_depth --per-steps 120 --model albert/deepseek-v4-flash-0731

Rows: FLD×2 test, entailment and contradiction rows derivable in the calculus, up to --per-steps per
claimed length. Unparsed replies are missing, not wrong, and counted apart.

Two row features confound the comparison and are recorded: FLD counts instantiating a universal fact
as a proof step (our ground calculus does not), so a genuine shortcut is a minimum proof below the claim
*minus* those steps; and whether the gold proof uses a universal fact at all, which FLD mostly words
with "some"/"somebody" ("Some man is unplanned if the one is a kind of a shadowed").
"""
import argparse
import collections
import json
import math
import os
import random
import re

from benchmarks.audit_fld_proofs import Parser, certify, constants, ground
from semforge.domains.fld import FLDDomain, from_row, load, split_facts

KEYS = ["ALBERT_API_KEY", "ALBERT_API_KEY_2", "ALBERT_API_KEY_3", "ALBERT_API_KEY_4"]
LABELS = ("entailment", "contradiction", "neutral")
PROMPT = """Premises:
{premises}

Hypothesis: {hypothesis}

Using only the premises, is the hypothesis true (entailment), false (contradiction), or undetermined
(neutral)? Reason step by step, then end with a line "Answer: entailment", "Answer: contradiction" or
"Answer: neutral"."""


def build(per_steps, seed, split="test"):
    data, render = load(split), FLDDomain(rows=[]).render
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    rows, count = [], collections.Counter()
    for i in order:
        inst = from_row(data[i], i)
        if inst.gold == "neutral" or inst.steps is None or count[inst.steps] >= per_steps:
            continue
        try:
            facts = [Parser(f).parse() for _, f in inst.facts if f != "fake_formula"]
            hyp = Parser(inst.hyp_formula).parse()
        except (ValueError, AttributeError, AssertionError):
            continue
        consts = set()
        for f in [*facts, hyp]:
            constants(f, consts)
        if isinstance(hyp, tuple) and hyp[0] in ("forall", "exists"):
            continue
        r = certify([g for f in facts for g in ground(f, consts)], hyp, inst.gold)
        if r is None:
            continue
        premises, hypothesis = render(inst)
        count[inst.steps] += 1
        rows.append({"uid": i, "gold": inst.gold, "claimed": inst.steps, **features(data[i]), **r.analysis.profile,
                     "prompt": PROMPT.format(premises=premises, hypothesis=hypothesis)})
    return rows


def features(row):
    """FLD's instantiation steps in its gold proof, and whether that proof uses a universal fact."""
    inst, formula = from_row(row, 0), split_facts(row["facts_formula"])
    used = set(re.findall(r"fact\d+", (row["proofs_formula"] or [""])[0]))
    return {"instantiations": inst.instantiations,
            "universal": any(formula.get(f, "").lstrip().startswith("(x)") for f in used)}


def parse(text):
    m = re.findall(r"Answer:\s*\**\s*(entailment|contradiction|neutral)", str(text), flags=re.I)
    return m[-1].lower() if m else None


def wilson(k, n, z=1.96):
    if not n:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def accuracy(rs):
    ok = [r for r in rs if r["pred"] is not None]
    k = sum(r["pred"] == r["gold"] for r in ok)
    return {"n": len(ok), "acc": round(k / len(ok), 3) if ok else None, "ci95": wilson(k, len(ok)),
            "unparsed": len(rs) - len(ok)}


def summarize(rows):
    by = lambda key: {k: accuracy([r for r in rows if key(r) == k]) for k in sorted({key(r) for r in rows})}
    # within a claimed length: a genuine shortcut (min size below the claim without instantiation
    # steps) or not; a universal fact in the gold proof or not
    within = {}
    for s in sorted({r["claimed"] for r in rows}):
        rs = [r for r in rows if r["claimed"] == s]
        within[s] = {"shortcut": accuracy([r for r in rs if shortcut(r)]),
                     "no_shortcut": accuracy([r for r in rs if not shortcut(r)]),
                     "universal": accuracy([r for r in rs if r["universal"]]),
                     "no_universal": accuracy([r for r in rs if not r["universal"]]),
                     "no_mandatory_step": accuracy([r for r in rs if r["mandatory"] == 0])}
    return {"all": accuracy(rows), "by_claimed": by(lambda r: r["claimed"]),
            "by_min_size": by(lambda r: min(r["min_size"], 8)), "by_mandatory": by(lambda r: min(r["mandatory"], 8)),
            "by_universal": by(lambda r: r["universal"]), "by_shortcut": by(shortcut),
            "within_claimed": within, "logistic": logistic(rows)}


def shortcut(r):
    return r["min_size"] < r["claimed"] - r["instantiations"]


COVARIATES = {"claimed": lambda r: r["claimed"], "min_size": lambda r: r["min_size"],
              "universal": lambda r: r["universal"], "shortcut": shortcut}
# Does necessity add to the smallest proof? Rows with a smallest proof only (forced share is undefined at 0).
FORCED = {"claimed": lambda r: r["claimed"] - r["instantiations"], "min_size": lambda r: r["min_size"],
          "forced_share": lambda r: r["mandatory"] / r["min_size"] if r["min_size"] else None,
          "universal": lambda r: r["universal"]}


def logistic(rows, boot=500, seed=0, covariates=COVARIATES):
    """correct ~ covariates (standardized; default claimed + min_size + universal + shortcut), bootstrap
    95% intervals. Rows where a covariate is None are left out."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    ok = [r for r in rows if r["pred"] is not None and all(f(r) is not None for f in covariates.values())]
    X = np.array([[f(r) for f in covariates.values()] for r in ok], float)
    y = np.array([r["pred"] == r["gold"] for r in ok], int)
    X = (X - X.mean(0)) / X.std(0)
    fit = lambda X, y: LogisticRegression(C=1e6).fit(X, y).coef_[0]
    rng, coefs = np.random.default_rng(seed), []
    for _ in range(boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            coefs.append(fit(X[i], y[i]))
    lo, hi = np.percentile(coefs, [2.5, 97.5], axis=0)
    return {name: {"coef": round(float(c), 3), "ci95": [round(float(a), 3), round(float(b), 3)]}
            for name, c, a, b in zip(covariates, fit(X, y), lo, hi)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-steps", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="albert/deepseek-v4-flash-0731")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--rpm", type=int, default=45, help="per Albert key")
    ap.add_argument("--out", default="results/llm_depth")
    ap.add_argument("--refit", nargs="+", help="saved replies (.jsonl): add the forced-share fit to their summaries")
    args = ap.parse_args()
    if args.refit:
        for path in args.refit:
            rows, summary_path = [json.loads(line) for line in open(path)], path[:-len(".jsonl")] + ".summary.json"
            summary = json.load(open(summary_path))
            summary["logistic_forced"] = logistic(rows, boot=1000, covariates=FORCED)
            json.dump(summary, open(summary_path, "w"), indent=1)
            print(path, json.dumps(summary["logistic_forced"]))
        return
    os.makedirs(args.out, exist_ok=True)
    rows_path = f"{args.out}/rows_s{args.seed}_n{args.per_steps}.jsonl"
    if os.path.exists(rows_path):
        rows = [json.loads(line) for line in open(rows_path)]
    else:
        rows = build(args.per_steps, args.seed)
        with open(rows_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    if rows and "universal" not in rows[0]:  # rows built before the confounds were recorded
        data = load("test")
        rows = [{**r, **features(data[r["uid"]])} for r in rows]
        with open(rows_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    print(len(rows), "rows", sorted(collections.Counter(r["claimed"] for r in rows).items()), flush=True)

    import litlm
    albert = args.model.startswith("albert/")
    replies = litlm.complete([r["prompt"] for r in rows], model=args.model,
                             api_key_envs=KEYS if albert else None, per_key_rpm=args.rpm if albert else None,
                             num_retries=2, max_tokens=8192, timeout=300, caching=True,
                             reasoning_effort=args.reasoning or None)
    tag = args.model.split("/")[-1] + (f"-{args.reasoning}" if args.reasoning else "")
    for r, t in zip(rows, replies):
        r["pred"], r["reply_tail"] = parse(t), str(t)[-300:]
    with open(f"{args.out}/{tag}.jsonl", "w") as f:
        f.writelines(json.dumps({k: v for k, v in r.items() if k != "prompt"}) + "\n" for r in rows)
    summary = {"model": args.model, "reasoning": args.reasoning, "rows": rows_path, **summarize(rows)}
    json.dump(summary, open(f"{args.out}/{tag}.summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
