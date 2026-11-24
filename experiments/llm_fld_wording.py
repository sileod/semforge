"""Does wording cause the FLD×2 universal-fact errors? Same rows, universals reworded as universals.

llm_realized_depth.py finds that LLM errors on FLD×2 track one surface property, whether the gold proof
uses a universal fact, and that FLD×2 mostly words its universals with "something"/"someone"/"some man",
which models read as particular. That is a correlation: rows with universal facts may also be harder for
other reasons. Here the logic is held fixed and only the quantifier word changes. Each fact whose formula
is a universal "(x): ..." and whose English opens with the indefinite that realizes x is reworded:

  Something that adsorbs Tiflis does not unfasten shanghai.  ->  Everything that adsorbs Tiflis ...
  Some person is filariid if ...                             ->  Every person is filariid if ...
  If something is nosocomial then it does dissonate ...      ->  For everything, if it is nosocomial then ...

Conditionals whose indefinite takes a relative clause ("If something that ...") are left as they are.

Since "(x):" is the outermost operator, the universal reading is the formula's literal reading whatever
the rest of the sentence says. Facts already worded "every"/"all" and facts with other openings are left
as they are. The same rows as llm_realized_depth (results/llm_depth/rows_s0_n120.jsonl), the same prompt
and model settings; original prompts hit the reply cache. Paired comparison on rows whose gold proof uses
a reworded fact. Replies are sampled, so one reply per row confounds the rewording with sampling noise:
with --samples k each condition is asked k times (sample i appends i newlines, which only defeats the
reply cache; sample 0 is the original run), and rows are compared by their mean accuracy over samples,
with a bootstrap interval over rows. The single-sample McNemar test is reported too.

  python -m experiments.llm_fld_wording --model albert/deepseek-v4-flash-0731
"""
import argparse
import json
import math
import os
import re

from experiments.llm_realized_depth import KEYS, parse
from semforge.domains.fld import load, split_facts

PERSON = {"someone": "everyone", "somebody": "everybody", "something": "everything"}
OPEN = re.compile(r"^(Something|Someone|Somebody|Some (?:man|person|thing|one))\b")
IF_OPEN = re.compile(r"^If (something|someone|somebody|some man|some person) (?!that |who |which )")


def reword(sentence):
    """The sentence with its opening indefinite made universal, or None if it opens otherwise."""
    m = OPEN.match(sentence)
    if m:
        w = m.group(1).lower()
        new = PERSON.get(w) or "every " + w.split()[1]
        return new.capitalize() + sentence[m.end():]
    m = IF_OPEN.match(sentence)
    if m:
        w = m.group(1)
        every = PERSON.get(w, "every " + w.split()[-1])
        return f"For {every}, if it " + sentence[m.end():]  # FLD×2's own pronoun for any variable
    return None


def universal_facts(row):
    formula, text = split_facts(row["facts_formula"]), split_facts(row["facts"])
    return {f: text[f] for f, form in formula.items() if form.lstrip().startswith("(x)") and f in text}


def rewrite(row, prompt):
    """The prompt with every rewordable universal fact reworded; the fact ids reworded."""
    done = []
    for f, s in universal_facts(row).items():
        new = reword(s)
        if new and s in prompt:
            prompt = prompt.replace(s, new)
            done.append(f)
    return prompt, done


def gold_facts(row):
    return set(re.findall(r"fact\d+", (row["proofs_formula"] or [""])[0]))


def mcnemar(b, c):
    """Exact two-sided McNemar p-value for b (original right, reworded wrong) and c (the reverse)."""
    n, k = b + c, min(b, c)
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default="results/llm_depth/rows_s0_n120.jsonl")
    ap.add_argument("--model", default="albert/deepseek-v4-flash-0731")
    ap.add_argument("--reasoning", default="low")
    ap.add_argument("--rpm", type=int, default=45, help="per Albert key")
    ap.add_argument("--out", default="results/llm_depth/wording")
    ap.add_argument("--samples", type=int, default=3, help="replies per row and condition")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = load("test")
    rows = [json.loads(line) for line in open(args.rows)]
    for r in rows:
        row = data[r["uid"]]
        r["reworded_prompt"], done = rewrite(row, r["prompt"])
        r["reworded"] = done
        r["gold_reworded"] = bool(set(done) & gold_facts(row))
    treated = [r for r in rows if r["gold_reworded"]]
    print(len(rows), "rows;", sum(bool(r["reworded"]) for r in rows), "with a reworded fact;",
          len(treated), "with a reworded fact in the gold proof", flush=True)

    import litlm
    albert = args.model.startswith("albert/")
    ask = lambda prompts: litlm.complete(prompts, model=args.model, api_key_envs=KEYS if albert else None,
                                         per_key_rpm=args.rpm if albert else None, num_retries=2,
                                         max_tokens=8192, timeout=300, caching=True,
                                         reasoning_effort=args.reasoning or None)
    for cond, key in (("orig", "prompt"), ("new", "reworded_prompt")):
        for i in range(args.samples):
            replies = ask([r[key] + "\n" * i for r in treated])
            for r, t in zip(treated, replies):
                r.setdefault(f"preds_{cond}", []).append(parse(t))
                if i == 0:
                    r[f"pred_{cond}"], r[f"tail_{cond}"] = parse(t), str(t)[-300:]
    both = [r for r in treated if r["pred_orig"] and r["pred_new"]]
    ok = lambda r, k: r[k] == r["gold"]
    b = sum(ok(r, "pred_orig") and not ok(r, "pred_new") for r in both)
    c = sum(ok(r, "pred_new") and not ok(r, "pred_orig") for r in both)
    acc = lambda k: round(sum(ok(r, k) for r in both) / len(both), 3) if both else None
    neutral = lambda k: sum(r[k] == "neutral" and not ok(r, k) for r in both)
    import numpy as np
    mean_ok = lambda r, cond: np.mean([p == r["gold"] for p in r[f"preds_{cond}"] if p] or [np.nan])
    rows_k = [r for r in treated if not np.isnan(mean_ok(r, "orig")) and not np.isnan(mean_ok(r, "new"))]
    o = np.array([mean_ok(r, "orig") for r in rows_k])
    n = np.array([mean_ok(r, "new") for r in rows_k])
    rng = np.random.default_rng(0)
    boot = [(n - o)[i].mean() for i in (rng.integers(0, len(o), len(o)) for _ in range(2000))]
    summary = {"model": args.model, "reasoning": args.reasoning, "rows": len(rows), "treated": len(treated),
               "samples": args.samples, "rows_sampled": len(rows_k),
               "mean_acc_original": round(float(o.mean()), 3), "mean_acc_reworded": round(float(n.mean()), 3),
               "diff": round(float((n - o).mean()), 3),
               "diff_ci95": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
               "unparsed_original": sum(p is None for r in treated for p in r["preds_orig"]),
               "unparsed_reworded": sum(p is None for r in treated for p in r["preds_new"]),
               "single_sample": {"paired": len(both), "acc_original": acc("pred_orig"), "acc_reworded": acc("pred_new"),
                                 "orig_right_new_wrong": b, "new_right_orig_wrong": c, "mcnemar_p": mcnemar(b, c),
                                 "neutral_errors_original": neutral("pred_orig"),
                                 "neutral_errors_reworded": neutral("pred_new")}}
    tag = args.model.split("/")[-1] + (f"-{args.reasoning}" if args.reasoning else "")
    with open(f"{args.out}/{tag}.jsonl", "w") as f:
        f.writelines(json.dumps({k: v for k, v in r.items() if k not in ("prompt",)}) + "\n" for r in treated)
    json.dump(summary, open(f"{args.out}/{tag}.summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
