"""Does training an LLM on certified-deep rows teach depth? One property varies; the rest is matched.

Arms differ only in their entailment/contradiction rows, all from the monadic pack and one renderer:
  shallow  shape rows with at most 5 mandatory steps (certified)
  deep     shape rows with at least 10 mandatory steps (certified)
  forward  FLD-style forward rows (1-20 planned steps, depth not certified)
They have the same number of rows, the same label mix and the same premise-count histogram (selection
over 2-premise bins), and share one set of neutral rows. Shallow rows get more distractors at
construction so that their premise counts reach the deep ones'.

Evaluation: held-out shape rows (another seed) by mandatory bucket, FLD×2 test rows (FLD's own English,
experiments/llm_realized_depth.py), and propositional shape rows (another pack).

  python -m experiments.llm_depth_training build --n 3000 --out results/llm_train/data
  python -m experiments.llm_depth_training train --data results/llm_train/data --arm deep --seed 0 \\
      --model Qwen/Qwen3-0.6B-Base --out results/llm_train/runs
"""
import argparse
import collections
import json
import os
import random

from experiments.llm_common import LABELS, Keep, commit, bucket, generate, like, match, reformat, report
from experiments.llm_realized_depth import parse  # noqa: F401  (same answer format as the API runs)

def build(args):
    """E/C rows per arm matched across arms (_match); each arm's neutrals drawn from one shape pool to
    match its own E/C rows on certified depth and premise count, so that depth cannot separate E/C
    from N inside an arm. (A first design shared one set of neutrals across arms: each model then
    called rows of the other depth neutral.)"""
    os.makedirs(args.out, exist_ok=True)
    over = 5 * args.n
    pk = args.pack
    pad = args.pad  # same random padding everywhere; selection then matches premise counts
    raw = {"shallow": generate(pk, "shape", 1, over, Keep(hi=args.shallow_max), pad=pad),
           "deep": generate(pk, "shape", 2, over, Keep(lo=args.deep_min), pad=pad),
           "forward": generate(pk, "forward", 3, over, Keep())}
    arms = match(raw, args.n, random.Random(0))
    pool = generate(pk, "shape", 4, 12 * args.n, Keep(labels=("neutral",)), pad=pad)
    rng = random.Random(1)
    for a, rows in arms.items():
        arms[a] = rows + like(rows, pool, args.n // 2, rng)
        with open(f"{args.out}/train_{a}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in arms[a])
    if args.tests_from:  # the earlier build's tests (monadic held-out, FLD x2, propositional), plus this pack's
        tests = {t: [json.loads(line) for line in open(f"{args.tests_from}/test_{t}.jsonl")]
                 for t in ("shape_heldout", "propositional", "fld")}
        if pk != "monadic":
            tests[f"{pk}_heldout"] = generate(pk, "shape", 5, args.test, Keep(labels=LABELS))
    else:
        tests = {"shape_heldout": generate("monadic", "shape", 5, args.test, Keep(labels=LABELS)),
                 "propositional": generate("propositional", "shape", 6, args.test // 2, Keep(labels=LABELS))}
        fld = [json.loads(line) for line in open("results/llm_depth/rows_s0_n120.jsonl")]
        tests["fld"] = [{"gold": r["gold"], "mandatory": r["mandatory"], "claimed": r["claimed"],
                         "universal": r["universal"], "text": reformat(r["prompt"])} for r in fld]
    for t, rows in tests.items():
        with open(f"{args.out}/test_{t}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    stats = {a: {"rows": len(r), "labels": collections.Counter(x["gold"] for x in r),
                 "mandatory": {lab: collections.Counter(bucket(x["mandatory"]) for x in r if x["gold"] == lab)
                               for lab in LABELS},
                 "mean_premises": {lab: round(sum(x["premises"] for x in r if x["gold"] == lab) /
                                              max(1, sum(x["gold"] == lab for x in r)), 2) for lab in LABELS}}
             for a, r in arms.items()}
    stats["tests"] = {t: len(r) for t, r in tests.items()}
    stats["commit"] = commit()
    json.dump(stats, open(f"{args.out}/stats.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    if args.lora:  # models too large to fine-tune fully on one 24 GB GPU
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=args.lora, lora_alpha=2 * args.lora, target_modules="all-linear",
                                                 task_type="CAUSAL_LM"))
    base = model.get_base_model() if args.lora else model
    if args.arm.endswith("+proof"):  # queue syntax: arm+proof:seed
        args.arm, args.proof = args.arm[:-len("+proof")], True
    if args.arm.endswith("+cot"):  # queue syntax: arm+cot:seed
        args.arm, args.cot = args.arm[:-len("+cot")], True
    name = args.arm + ("+proof" if args.proof else "") + ("+cot" if args.cot else "")
    name += {"shuffle": "", "easy_first": "+easy", "hard_first": "+hard"}[args.order]
    lead = tok(COT_LEAD, add_special_tokens=False).input_ids
    # large vocabularies (gemma: 262k): logits at supervised positions only, the same loss in less memory
    sparse = model.config.vocab_size > 200_000 or args.sparse_loss  # also for long rows (FLD×2, ProverQA)
    rows = [] if args.arm == "none" else [json.loads(line) for line in open(f"{args.data}/train_{args.arm}.jsonl")]
    random.Random(args.seed).shuffle(rows)
    if args.order != "shuffle":  # each epoch runs through the rows in this order (ties keep the shuffle)
        rows.sort(key=lambda r: r.get("mandatory") or 0, reverse=args.order == "hard_first")
    answer = {lab: tok(" " + lab, add_special_tokens=False).input_ids for lab in LABELS}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(rows) // args.batch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 20) * max(0.0, 1 - s / max(steps, 1)))
    model.train()
    log, curve = [], []
    first = {}  # row index -> gold log-prob just before the model first trains on it (prequential, per row)
    for s in range(steps):
        batch = [rows[(s * args.batch + i) % len(rows)] for i in range(args.batch)]
        if args.row_losses and s * args.batch < len(rows):
            _, lp = score(base, tok, batch, answer, args.max_len, batch=args.batch)
            for i, l in enumerate(lp):
                first.setdefault(s * args.batch + i, l[LABELS.index(batch[i]["gold"])])
        ids, labels, part = [], [], []
        for r in batch:
            p = tok(r["text"], add_special_tokens=False).input_ids[-args.max_len:]
            a = answer[r["gold"]] + [tok.eos_token_id]
            b = []  # proof supervision: the label, then a certified proof, scored as a separate loss
            if args.proof and r.get("proof"):
                a = answer[r["gold"]]
                b = tok("\nProof:\n" + r["proof"], add_special_tokens=False).input_ids[:args.max_proof] + [tok.eos_token_id]
            if args.cot:  # the proof first (neutral: none found), then the label after COT_END
                proof = r.get("proof") or "No proof found."
                b = tok(proof, add_special_tokens=False).input_ids[:args.max_proof] + tok(COT_END, add_special_tokens=False).input_ids
                p = p + lead
                ids.append(p + b + a)
                labels.append([-100] * len(p) + b + a)
                part.append([0] * len(p) + [2] * len(b) + [1] * len(a))
                continue
            ids.append(p + a + b)
            labels.append([-100] * len(p) + a + b)
            part.append([0] * len(p) + [1] * len(a) + [2] * len(b))
        width = max(map(len, ids))
        pad = lambda xs, v: [x + [v] * (width - len(x)) for x in xs]
        x = torch.tensor(pad(ids, tok.pad_token_id or 0)).cuda()
        y = torch.tensor(pad(labels, -100)).cuda()
        mask = torch.tensor(pad([[1] * len(i) for i in ids], 0)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if not (args.proof or args.cot or sparse):
                loss = model(input_ids=x, attention_mask=mask, labels=y).loss
            else:  # label loss + proof loss, each a token mean; logits only at supervised positions
                k = torch.tensor(pad(part, 0)).cuda()[:, 1:]
                h = base.model(input_ids=x, attention_mask=mask).last_hidden_state[:, :-1][k > 0]
                ce = torch.nn.functional.cross_entropy(base.lm_head(h).float(), y[:, 1:][k > 0], reduction="none")
                kk = k[k > 0]
                loss = ce[kk == 1].mean() + (ce[kk == 2].mean() if (kk == 2).any() else 0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {s}")
        curve.append(round(loss.item(), 4))  # every step: the prequential (epiplexity) estimate needs the full curve
        if s % 50 == 0:
            log.append({"step": s, "loss": round(loss.item(), 4)})
            print(log[-1], flush=True)
    del opt, sched
    if args.lora:  # the evaluation code reads model.model / model.lm_head: a plain model with merged weights
        model = model.merge_and_unload()
    model.eval()
    model.gradient_checkpointing_disable()  # and so the cache works for cot generation
    os.makedirs(args.out, exist_ok=True)
    if args.save:  # the trained student, e.g. to score generated rows (experiments/teacher_policy.py)
        model.save_pretrained(args.save)
        tok.save_pretrained(args.save)
    if args.row_losses:  # per-row learning gain: final gold log-prob minus the one before first training on it
        _, lp = score(model, tok, rows, answer, args.max_len)
        with open(f"{args.out}/{name}_s{args.seed}.rows.jsonl", "w") as f:
            f.writelines(json.dumps({"text": r["text"], "gold": r["gold"], "first": first.get(i),
                                     "final": l[LABELS.index(r["gold"])]}) + "\n" for i, (r, l) in enumerate(zip(rows, lp)))
    results = {"arm": name, "proof": args.proof, "cot": args.cot, "seed": args.seed, "model": args.model, "lora": args.lora, "lr": args.lr, "order": args.order, "epochs": args.epochs, "steps": steps, "log": log, "loss_curve": curve,
               "max_len": args.max_len}  # longer prompts keep their last max_len tokens: counted per set below
    overlong = lambda rows: sum(len(tok(r["text"], add_special_tokens=False).input_ids) > args.max_len for r in rows)
    import glob
    for t in sorted(os.path.basename(f)[:-6] for f in glob.glob(f"{args.data}/test_*.jsonl") +
                    ([] if args.cot else glob.glob(f"{args.data}/val_*.jsonl"))):  # cot: generation is slow
        rows_t = [json.loads(line) for line in open(f"{args.data}/{t}.jsonl")]
        t = t[5:] if t.startswith("test_") else t
        derived = derive(model, tok, rows_t, args) if args.cot else None
        preds, logps = score(model, tok, rows_t, answer, args.max_len, prefixes=derived and [d[0] for d in derived])
        results[t] = {**summarize(rows_t, preds), "truncated": overlong(rows_t)}
        with open(f"{args.out}/{name}_s{args.seed}_{t}.preds.jsonl", "w") as f:
            f.writelines(json.dumps({"gold": r["gold"], "pred": p, "mandatory": r["mandatory"], "logp": logps[k],
                                     **({"proof": derived[k][1]} if derived else {})}) + "\n"
                         for k, (r, p) in enumerate(zip(rows_t, preds)))
        print(t, json.dumps(results[t]), flush=True)
    json.dump(results, open(f"{args.out}/{name}_s{args.seed}.json", "w"), indent=1)


COT_LEAD, COT_END = "\nProof:\n", "\nSo the label is:"


def derive(model, tok, rows, args, batch=16):
    """(token prefix, proof text) per row for cot scoring: prompt, COT_LEAD, the model's greedy proof up
    to COT_END (appended when the proof runs out of budget without it)."""
    import torch
    lead, end = (tok(x, add_special_tokens=False).input_ids for x in (COT_LEAD, COT_END))
    model.config.use_cache = True
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            ps = [tok(r["text"], add_special_tokens=False).input_ids[-args.max_len:] + lead for r in rows[i:i + batch]]
            width = max(map(len, ps))
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            x = torch.tensor([[pad] * (width - len(q)) + q for q in ps]).cuda()  # left padding
            mask = torch.tensor([[0] * (width - len(q)) + [1] * len(q) for q in ps]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                g = model.generate(input_ids=x, attention_mask=mask, max_new_tokens=args.max_proof + len(end),
                                   do_sample=False, stop_strings=[COT_END.strip()], tokenizer=tok, pad_token_id=pad)
            for q, new in zip(ps, g[:, width:].tolist()):
                text = tok.decode(new, skip_special_tokens=True)
                cut = text.find(COT_END.strip())
                proof = (text[:cut] if cut >= 0 else text).rstrip()
                out.append((q + tok(proof, add_special_tokens=False).input_ids + end, proof))
    return out


def score(model, tok, rows, answer, max_len, batch=16, prefixes=None):
    """The label whose answer tokens have the highest total log-probability, and those totals (in LABELS
    order) per row. `prefixes`: token ids to score after, instead of each row's prompt (cot: prompt and
    the model's own proof)."""
    import torch
    preds, logps = [], []
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            totals = []
            for lab in LABELS:
                ids, starts = [], []
                for j, r in enumerate(chunk):
                    p = prefixes[i + j] if prefixes else tok(r["text"], add_special_tokens=False).input_ids[-max_len:]
                    ids.append(p + answer[lab])
                    starts.append(len(p))
                width = max(map(len, ids))
                x = torch.tensor([s + [tok.pad_token_id or 0] * (width - len(s)) for s in ids]).cuda()
                mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in ids]).cuda()
                a = answer[lab]
                rows_i = torch.arange(len(chunk)).cuda()[:, None]
                cols = torch.tensor([[st - 1 + k for k in range(len(a))] for st in starts]).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = model.model(input_ids=x, attention_mask=mask).last_hidden_state[rows_i, cols]
                    logp = model.lm_head(hidden).float().log_softmax(-1)  # only the answer positions
                totals.append(logp[..., torch.tensor(a).cuda()].diagonal(dim1=1, dim2=2).sum(-1).tolist())
            preds += [LABELS[max(range(3), key=lambda k: totals[k][j])] for j in range(len(chunk))]
            logps += [[round(totals[k][j], 4) for k in range(3)] for j in range(len(chunk))]
    return preds, logps


def summarize(rows, preds):
    acc = lambda pairs: round(sum(p == r["gold"] for r, p in pairs) / len(pairs), 4) if pairs else None
    pairs = list(zip(rows, preds))
    ec = [(r, p) for r, p in pairs if r["gold"] != "neutral"]
    return {"all": acc(pairs), "ec": acc(ec), "n": len(pairs),
            "by_mandatory": {b: acc([(r, p) for r, p in ec if r["mandatory"] is not None and bucket(r["mandatory"]) == b])
                             for b in ("1-3", "4-7", "8-12", "13+")},
            "by_label": {lab: acc([(r, p) for r, p in pairs if r["gold"] == lab]) for lab in LABELS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=3000, help="E/C rows per arm")
    b.add_argument("--test", type=int, default=1200)
    b.add_argument("--out", default="results/llm_train/data")
    b.add_argument("--tests-from", help="reuse the test files of an earlier build")
    b.add_argument("--pack", default="monadic")
    b.add_argument("--shallow-max", type=int, default=5)
    b.add_argument("--deep-min", type=int, default=10)
    b.add_argument("--pad", type=int, default=0, help="up to this many extra irrelevant derivations per shape row")
    t = sub.add_parser("train")
    t.add_argument("--data", default="results/llm_train/data")
    t.add_argument("--arm", required=True, help="shallow, deep, forward, or none (the untrained model)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--order", choices=("shuffle", "easy_first", "hard_first"), default="shuffle",
                   help="row order within each epoch, by mandatory steps")
    t.add_argument("--max-len", type=int, default=768)
    t.add_argument("--row-losses", action="store_true", help="write each training row's gold log-prob before the "
                   "model first trains on it and after training (per-row prequential gain, an epiplexity proxy)")
    t.add_argument("--save", help="directory to save the trained model in")
    t.add_argument("--lora", type=int, default=0, help="LoRA rank on every linear layer (0: full fine-tuning)")
    t.add_argument("--sparse-loss", action="store_true", help="logits at supervised positions only (same loss, less memory)")
    t.add_argument("--out", default="results/llm_train/runs")
    t.add_argument("--proof", action="store_true", help="also train on each row's certified proof, after the label")
    t.add_argument("--max-proof", type=int, default=384, help="proof tokens kept")
    t.add_argument("--cot", action="store_true", help="train on the certified proof before the label; "
                   "at test, generate a proof, then score the labels after it")
    r = sub.add_parser("report")
    r.add_argument("--runs", default="results/llm_train/runs_v2")
    args = ap.parse_args()
    {"build": build, "train": train, "report": lambda a: report(a.runs)}[args.cmd](args)


if __name__ == "__main__":
    main()
