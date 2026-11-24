"""Proof derivation: train on written proofs, then check the proofs a model writes, not only its labels.

A row numbers its premises; the target is a proof whose steps cite them, then the answer:
  s1. Every cat is an animal. [from 2, 5]
  s2. Tom is an animal. [from 1, s1]
  Answer: entailment
(neutral: "No proof." then the answer). The checker reads only the cited premise numbers: a proof is
grounded when the cited premises alone give the answer (H for entailment, not-H for contradiction), and
minimal when no cited premise can be dropped. The steps' wording is not checked.

Training arms (monadic pack; E/C rows matched on label and premise count, plus the same share of shape
neutrals, as in llm_depth_training):
  shape    certified construction, the smallest certified proof
  forward  forward generation (depth not certified), the smallest certified proof
  planted  the same forward rows, the derivation the generator planted
  label    the shape rows with the answer only
  fld      FLD×2 training rows with FLD's own proofs (external reference)
Tests, each with an exact checker: held-out shape rows (monadic, another seed), propositional shape rows
(another pack), FLD×2 test (z3 over FLD's formulas), ProofWriter OWA depth-5 (Horn chase over its logical
forms) and ProverQA dev (z3 over its translations).

  python -m experiments.llm_proof_derivation build --n 3000 --out results/llm_proof/data
  python -m experiments.llm_proof_derivation train --data results/llm_proof/data --arm shape --seed 10 \\
      --out results/llm_proof/runs
  python -m experiments.llm_proof_derivation report --runs results/llm_proof/runs
"""
import argparse
import collections
import glob
import json
import os
import random
import re

from experiments.llm_common import LABELS, Keep, bucket, commit, like, match

INSTRUCTION = ("Prove the hypothesis or its negation from the numbered premises, citing them, "
               "or say that neither follows.")
FORMAT = "{background}Premises:\n{premises}\n\nHypothesis: {hypothesis}\n\n" + INSTRUCTION + "\nProof:\n"
PROVERQA = "opendatalab/ProverQA"
PROOFWRITER = os.environ.get(  # ProofWriter OWA depth-5 test, as downloaded by scripts/reproduce_audits.sh
    "PROOFWRITER", "data/proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-test.jsonl")


def prompt(premises, hypothesis, background=()):
    bg = "Background:\n" + "\n".join(background) + "\n\n" if background else ""
    return FORMAT.format(background=bg, premises="\n".join(f"[{i + 1}] {s}" for i, s in enumerate(premises)),
                         hypothesis=hypothesis)


def target(steps, label):
    """steps: (statement, cited premise indices (0-based), cited step indices (0-based))."""
    lines = [f"s{k + 1}. {s.rstrip('.')}. [from {', '.join([str(i + 1) for i in ps] + [f's{j + 1}' for j in ss])}]"
             for k, (s, ps, ss) in enumerate(steps)]
    return "\n".join(lines or ["No proof."]) + f"\nAnswer: {label}"


def read(text):
    """(answer or None, cited premise indices, 0-based) from a generated proof."""
    m = re.search(r"Answer:\s*(entailment|contradiction|neutral)", text)
    cited = {int(x) - 1 for group in re.findall(r"\[from ([^\]]*)\]", text) for x in re.findall(r"(?<![s\d])(\d+)", group)}
    return (m.group(1) if m else None), cited


# --- constructed rows ----------------------------------------------------------------------------------

def _json(f):
    return f if isinstance(f, str) else [_json(x) for x in f]


def _formula(f):
    return f if isinstance(f, str) else tuple(_formula(x) for x in f)


def _proof_steps(order, cite, statement):
    """order: (head, parents) bottom-up. A parent is a premise (index), an earlier step, or neither
    (background, not cited)."""
    done, steps = {}, []
    for head, parents in order:
        ps = sorted({cite[p] for p in parents if p in cite})
        ss = sorted({done[p] for p in parents if p in done and p not in cite})
        done.setdefault(head, len(steps))
        steps.append((statement(head), ps, ss))
    return steps


def constructed_shard(pack, strategy, seed, want, keep):
    from semforge.construct.archive import profile_cell
    from semforge.construct.calculus import analyze, rule_of
    from semforge.construct.forward import forward
    from semforge.construct.oracle import Undecided
    from semforge.domains.constructed import ConstructedDomain
    d, rng, out = ConstructedDomain(pack, strategy=strategy), random.Random(seed), []
    calc, realizer = d.pack.calculus, d.pack.render.__self__
    key = profile_cell("mandatory", calculus=calc)
    tries = 0
    while len(out) < want and tries < 50 * want:
        tries += 1
        try:
            if strategy == "forward":
                got = forward(d.pack, rng, rng.randint(1, 20))
                if not got or not d._renders(got[0]):
                    continue
                (inst, plan), label = got, got[0].label
            else:
                [(inst, label)] = d.sample(rng, rng.randint(0, 5)) or [(None, None)]
                plan = None
                if inst is None:
                    continue
            cell = key(inst)
        except Undecided:
            continue
        row = {"gold": label, "mandatory": cell[1] if cell else None, "premises": len(inst.premises),
               "strategy": strategy}
        if not keep(row):
            continue
        r = random.Random(repr((inst.premises, inst.hypothesis)))  # as in render: the same lexicon
        lex = realizer.lexicon([*inst.premises, inst.hypothesis], r)
        sents = [realizer.statement(p, "premise", lex, r) for p in inst.premises]
        hyp = realizer.statement(inst.hypothesis, "hypothesis", lex, r)
        say = lambda f: realizer.statement(f, "premise", lex, r)
        cite = {calc.canonical(p): i for i, p in enumerate(inst.premises)}
        row.update(text=prompt(sents, hyp, realizer.sig.theory),
                   check={"kind": "constructed", "pack": pack, "premises": _json(list(inst.premises)),
                          "hypothesis": _json(inst.hypothesis)})
        if label == "neutral":
            row["proof"] = target([], label)
        else:
            goal = inst.hypothesis if label == "entailment" else d.pack.negate(inst.hypothesis)
            a = analyze(calc, inst.premises, goal, inst.universe)
            if not a.proofs:
                continue
            best = min(a.proofs, key=lambda p: (sum(rule_of(x) is not None for x in p), sorted(map(repr, p))))
            order = sorted(((x[2], x[3]) for x in best if rule_of(x) is not None),
                           key=lambda hp: (a.closure.round[hp[0]], repr(hp[0])))
            row["proof"] = target(_proof_steps(order, cite, say), label)
            if plan is not None:  # the planted derivation, bottom-up, nodes by index
                order = [(calc.canonical(plan.nodes[c]), [calc.canonical(plan.nodes[i]) for i in ps])
                         for _, c, ps in reversed(plan.steps)]
                row["planted"] = target(_proof_steps(order, cite, say), label)
        out.append(row)
    return out


def constructed(pack, strategy, seed, want, keep, jobs=32):
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(jobs) as ex:
        parts = ex.map(constructed_shard, *zip(*[(pack, strategy, 1000 * seed + j, -(-want // jobs), keep)
                                                 for j in range(jobs)]))
        return [r for part in parts for r in part][:want]


# --- external rows -------------------------------------------------------------------------------------

def fld_rows(split, n, rng, with_proof):
    from semforge.domains.fld import LABEL, load, split_facts
    data = load(split)
    out = []
    for i in rng.sample(range(len(data)), min(len(data), 3 * n)):
        x = data[i]
        text, formula = split_facts(x["facts"]), split_facts(x["facts_formula"])
        ids = sorted(text, key=lambda f: int(f[4:]))
        cite = {f: k for k, f in enumerate(ids)}
        label = LABEL[x["world_assump_label"]]
        row = {"gold": label, "mandatory": x["depth"], "premises": len(ids),
               "text": prompt([text[f] for f in ids], x["hypothesis"]),
               "check": {"kind": "fld", "facts": [[f, formula.get(f, "fake_formula")] for f in ids],
                         "hypothesis": x["hypothesis_formula"]}}
        if with_proof:
            if label == "neutral":
                row["proof"] = target([], label)
            else:
                proof, steps, done = (x["proofs"] or [""])[0], [], {}
                for step in filter(None, (s.strip() for s in proof.split(";"))):
                    body, _, head = step.partition("->")
                    name, _, stmt = head.partition(":")
                    name = name.strip()
                    if name == "hypothesis":  # FLD words not-H only when it has a negative hypothesis
                        stmt = x["hypothesis"] if label == "entailment" else \
                            x["negative_hypothesis"] or f"It is not the case that {x['hypothesis'][0].lower()}{x['hypothesis'][1:]}"
                    used = re.findall(r"(fact\d+|int\d+)", body)
                    steps.append((stmt.strip(), sorted({cite[u] for u in used if u in cite}),
                                  sorted({done[u] for u in used if u in done})))
                    done[name] = len(steps) - 1
                if not steps:
                    continue
                row["proof"] = target(steps, label)
        out.append(row)
    return out


def proofwriter_rows(n, rng):
    """Questions balanced over answers and QDep; one sentence per triple and rule, in that order."""
    items = [(r, q) for line in open(PROOFWRITER) for r in [json.loads(line)] for q in r["questions"].values()]
    by = collections.defaultdict(list)
    for r, q in items:
        by[(str(q["answer"]), q["QDep"])].append((r, q))
    per = -(-n // len(by))
    out = []
    for k in sorted(by):
        for r, q in rng.sample(by[k], min(per, len(by[k]))):
            sents = [*r["triples"].items(), *r["rules"].items()]
            label = {"True": "entailment", "False": "contradiction"}.get(str(q["answer"]), "neutral")
            out.append({"gold": label, "mandatory": q["QDep"], "premises": len(sents),
                        "text": prompt([v["text"] for _, v in sents], q["question"]),
                        "check": {"kind": "proofwriter", "sentences": [[name, v["representation"]] for name, v in sents],
                                  "goal": q["representation"]}})
    return out


def proverqa_rows(per_level, rng):
    from huggingface_hub import hf_hub_download
    out = []
    for level in ("easy", "medium", "hard"):
        dev = json.load(open(hf_hub_download(PROVERQA, f"dev/{level}.json", repo_type="dataset")))
        for x in rng.sample(dev, per_level):
            sents = list(x["nl2fol"])
            label = {"A": "entailment", "B": "contradiction", "C": "neutral"}[x["answer"]]
            out.append({"gold": label, "mandatory": {"easy": 1, "medium": 4, "hard": 8}[level], "premises": len(sents),
                        "text": prompt(sents, x["question"].split("uncertain?", 1)[1].strip()),
                        "check": {"kind": "proverqa", "x": {k: x[k] for k in ("nl2fol", "conclusion_fol", "answer")}}})
    return out


# --- checking ------------------------------------------------------------------------------------------

class Checker:
    """gives(cited) -> do the cited premises (0-based) alone give the row's gold conclusion? None: the
    checker could not decide."""

    def __init__(self, row):
        c, self.kind = row["check"], row["check"]["kind"]
        self.gold = row["gold"]
        if self.kind == "constructed":
            from semforge.construct.packs import get
            self.pack = get(c["pack"])
            self.prem = [_formula(p) for p in c["premises"]]
            h = _formula(c["hypothesis"])
            self.goal = h if self.gold == "entailment" else self.pack.negate(h)
        elif self.kind == "fld":
            import z3
            from semforge.domains.fld import FLDDomain, _Parser
            env, self.d = {}, FLDDomain(rows=[])
            self.facts = [z3.BoolVal(True) if f == "fake_formula" else _Parser(f, env).parse() for _, f in c["facts"]]
            h = _Parser(c["hypothesis"], env).parse()
            self.goal = h if self.gold == "entailment" else z3.Not(h)
        elif self.kind == "proofwriter":
            from benchmarks.audit_proofwriter import VARS, ground, literal
            self.sents = []
            for name, rep in c["sentences"]:
                if name.startswith("triple"):
                    self.sents.append(("fact", literal(rep)))
                else:
                    body, head = rep.split("->")
                    self.sents.append(("rule", (name, [literal(x) for x in re.findall(r"\([^()]*\)", body)], literal(head))))
            g = literal(c["goal"])
            self.goal = g if self.gold == "entailment" else g[:3] + ("-" if g[3] == "+" else "+",)
            lits = [f for k, v in self.sents for f in ([v] if k == "fact" else v[1] + [v[2]])]
            self.entities = sorted({a for f in lits for a in (f[0], f[2])} - VARS)
            self.ground = ground
        else:
            from benchmarks.audit_proverqa import Problem
            import z3
            self.p = Problem(c["x"], 5000)
            self.goal = self.p.goal if self.gold == "entailment" else z3.Not(self.p.goal)
        self.n = len(self.prem if self.kind == "constructed" else self.facts if self.kind == "fld"
                     else self.sents if self.kind == "proofwriter" else self.p.premises)

    def gives(self, cited):
        cited = sorted(i for i in cited if 0 <= i < self.n)
        try:
            if self.kind == "constructed":
                return self.pack.entails([self.prem[i] for i in cited], self.goal)
            if self.kind == "fld":
                return self.d._entails(dict(enumerate(self.facts)), self.goal, cited)
            if self.kind == "proofwriter":
                from benchmarks.audit_proofwriter import chase
                facts = {self.sents[i][1] for i in cited if self.sents[i][0] == "fact"}
                rules = [self.sents[i][1] for i in cited if self.sents[i][0] == "rule"]
                return self.goal in chase(facts, self.ground(rules, self.entities))[0]
            return self.p.entails(cited, self.goal)
        except Exception:  # z3 timeout or an undecidable query: not counted either way
            return None


def grade(row, text):
    """answer, grounded (cited premises give the gold conclusion), minimal (and none can be dropped)."""
    answer, cited = read(text)
    out = {"answer": answer, "cited": len(cited), "grounded": None, "minimal": None}
    if row["gold"] == "neutral" or answer != row["gold"]:
        return out
    ch = Checker(row)
    out["grounded"] = ch.gives(cited)
    if out["grounded"]:
        out["minimal"] = not any(ch.gives(cited - {i}) for i in cited)
    return out


# --- build / train / report ----------------------------------------------------------------------------

def write(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def build(args):
    """E/C rows matched per logic, so every arm has the same logics in the same proportions."""
    os.makedirs(args.out, exist_ok=True)
    over, k = 4 * args.n // len(args.packs), args.n // len(args.packs)
    arms = {"shape": [], "forward": []}
    for i, pack in enumerate(args.packs):
        raw = {"shape": constructed(pack, "shape", 1 + 10 * i, over, Keep()),
               "forward": constructed(pack, "forward", 3 + 10 * i, over, Keep())}
        got = match(raw, k, random.Random(i))
        neutrals = constructed(pack, "shape", 4 + 10 * i, 8 * k, Keep(labels=("neutral",)))
        for a in got:
            arms[a] += got[a] + like(got[a], neutrals, k // 2, random.Random(f"{i}:{a}"))
    if args.packs == ["monadic"]:  # the single-logic study's controls
        arms["planted"] = [{**r, "proof": r.get("planted", r["proof"])} for r in arms["forward"]]
        arms["label"] = [{**r, "proof": f"Answer: {r['gold']}"} for r in arms["shape"]]
    fld = fld_rows("train", 2 * args.n, random.Random(2), with_proof=True)
    ec = [r for r in fld if r["gold"] != "neutral"][:args.n]
    arms["fld"] = ec + [r for r in fld if r["gold"] == "neutral"][:args.n // 2]
    for a, rows in arms.items():
        random.Random(a).shuffle(rows)
        write(f"{args.out}/train_{a}.jsonl", rows)
    tests = {"monadic": constructed("monadic", "shape", 5, args.test, Keep(labels=LABELS)),
             "propositional": constructed("propositional", "shape", 6, args.test, Keep(labels=LABELS)),
             "fld": fld_rows("test", args.test, random.Random(7), with_proof=False)[:args.test],
             "proofwriter": proofwriter_rows(args.test, random.Random(8)),
             "proverqa": proverqa_rows(args.test // 3, random.Random(9))}
    for t, rows in tests.items():
        write(f"{args.out}/test_{t}.jsonl", rows)
    stats = {k: {"rows": len(v), "labels": collections.Counter(r["gold"] for r in v)}
             for k, v in [*(("train_" + a, r) for a, r in arms.items()), *(("test_" + t, r) for t, r in tests.items())]}
    json.dump({"commit": commit(), "args": vars(args), "stats": stats}, open(f"{args.out}/build.json", "w"), indent=1)
    print(json.dumps(stats, indent=1))


def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).cuda()
    model.gradient_checkpointing_enable()
    if args.lora:
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(r=args.lora, lora_alpha=2 * args.lora, target_modules="all-linear",
                                                 task_type="CAUSAL_LM"))
    base = model.get_base_model() if args.lora else model
    rows = [] if args.arm == "none" else [json.loads(line) for line in open(f"{args.data}/train_{args.arm}.jsonl")]
    rows = rows[:args.limit] if args.limit else rows
    random.Random(args.seed).shuffle(rows)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = args.epochs * len(rows) // args.batch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 20) * max(0.0, 1 - s / max(steps, 1)))
    model.train()
    log = []
    for s in range(steps):
        batch = [rows[(s * args.batch + i) % len(rows)] for i in range(args.batch)]
        total = 0.0
        for c in range(0, args.batch, args.micro or args.batch):  # micro-batches: same loss, less memory
            ids, labels = [], []
            for r in batch[c:c + (args.micro or args.batch)]:
                p = tok(r["text"], add_special_tokens=False).input_ids[-args.max_len:]
                t = tok(r["proof"], add_special_tokens=False).input_ids[:args.max_new] + [tok.eos_token_id]
                ids.append(p + t)
                labels.append([-100] * len(p) + t)
            width = max(map(len, ids))
            pad = lambda xs, v: [x + [v] * (width - len(x)) for x in xs]
            x, y = torch.tensor(pad(ids, tok.pad_token_id or 0)).cuda(), torch.tensor(pad(labels, -100)).cuda()
            mask = torch.tensor(pad([[1] * len(i) for i in ids], 0)).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):  # logits at supervised positions only
                k = y[:, 1:] != -100
                h = base.model(input_ids=x, attention_mask=mask).last_hidden_state[:, :-1][k]
                ce = torch.nn.functional.cross_entropy(base.lm_head(h).float(), y[:, 1:][k], reduction="sum")
            n_tok = sum(sum(v != -100 for v in l[1:]) for l in labels)
            (ce / max(n_tok, 1) * len(ids) / args.batch).backward()
            total += ce.item() / max(n_tok, 1) * len(ids) / args.batch
        loss = torch.tensor(total)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {s}")
        if s % 50 == 0:
            log.append({"step": s, "loss": round(loss.item(), 4)})
            print(log[-1], flush=True)
    del opt, sched
    if args.lora:
        model = model.merge_and_unload()
    model.eval()
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    os.makedirs(args.out, exist_ok=True)
    results = {"arm": args.arm, "seed": args.seed, "model": args.model, "lora": args.lora, "steps": steps, "log": log}
    for f in sorted(glob.glob(f"{args.data}/test_*.jsonl")):
        t = os.path.basename(f)[5:-6]
        rows_t = [json.loads(line) for line in open(f)][:args.test_limit or None]
        texts = generate(model, tok, rows_t, args)
        graded = [grade(r, g) for r, g in zip(rows_t, texts)]
        with open(f"{args.out}/{args.arm}_s{args.seed}_{t}.preds.jsonl", "w") as fh:
            fh.writelines(json.dumps({"gold": r["gold"], "mandatory": r["mandatory"], "output": g, **x}) + "\n"
                          for r, g, x in zip(rows_t, texts, graded))
        results[t] = summarize(rows_t, graded)
        print(t, json.dumps(results[t]), flush=True)
    json.dump(results, open(f"{args.out}/{args.arm}_s{args.seed}.json", "w"), indent=1)


def generate(model, tok, rows, args, batch=32):
    """Greedy proofs, `batch` prompts at a time; a batch that runs out of GPU memory is split in half."""
    import torch
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def run(chunk):
        ps = [tok(r["text"], add_special_tokens=False).input_ids[-args.max_len:] for r in chunk]
        width = max(map(len, ps))
        x = torch.tensor([[pad] * (width - len(q)) + q for q in ps]).cuda()  # left padding
        mask = torch.tensor([[0] * (width - len(q)) + [1] * len(q) for q in ps]).cuda()
        g = None
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                g = model.generate(input_ids=x, attention_mask=mask, max_new_tokens=args.max_new, do_sample=False,
                                   pad_token_id=pad, eos_token_id=tok.eos_token_id)
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
        if g is None:  # retried outside the handler, so the failed attempt's tensors are freed first
            del x, mask
            torch.cuda.empty_cache()
            return run(chunk[:len(chunk) // 2]) + run(chunk[len(chunk) // 2:])
        return [tok.decode(new, skip_special_tokens=True) for new in g[:, width:].tolist()]
    return [t for i in range(0, len(rows), batch) for t in run(rows[i:i + batch])]


def summarize(rows, graded):
    """macro: label accuracy averaged over gold labels; on E/C rows: correct, grounded (correct and the
    cited premises give it), minimal (grounded, no cited premise removable), undecided checks."""
    acc = [sum(g["answer"] == lab for r, g in zip(rows, graded) if r["gold"] == lab) / n
           for lab in LABELS if (n := sum(r["gold"] == lab for r in rows))]
    ec = [(r, g) for r, g in zip(rows, graded) if r["gold"] != "neutral"]
    share = lambda xs: round(sum(xs) / len(ec), 4) if ec else None
    return {"macro": round(sum(acc) / len(acc), 4), "n": len(rows), "ec": len(ec),
            "correct": share([g["answer"] == r["gold"] for r, g in ec]),
            "grounded": share([bool(g["grounded"]) for _, g in ec]), "minimal": share([bool(g["minimal"]) for _, g in ec]),
            "undecided": share([g["answer"] == r["gold"] and g["grounded"] is None for r, g in ec]),
            "no_answer": round(sum(g["answer"] is None for g in graded) / len(graded), 4)}


def report(args):
    runs = collections.defaultdict(list)
    for f in glob.glob(f"{args.runs}/*_s*_*.preds.jsonl"):
        arm, rest = os.path.basename(f)[:-len(".preds.jsonl")].split("_s", 1)
        seed, test = rest.split("_", 1)
        rows = [json.loads(line) for line in open(f)]
        ec = [r for r in rows if r["gold"] != "neutral"]
        acc = [sum(r["answer"] == lab for r in rows if r["gold"] == lab) / n
               for lab in LABELS if (n := sum(r["gold"] == lab for r in rows))]
        runs[(test, arm)].append((sum(acc) / len(acc), sum(r["answer"] == r["gold"] for r in ec) / len(ec),
                                  sum(bool(r["grounded"]) for r in ec) / len(ec), sum(bool(r["minimal"]) for r in ec) / len(ec)))
    print(f"{'test':13s}{'arm':10s}{'seeds':>6s}{'macro':>8s}{'E/C ok':>8s}{'ground':>8s}{'minimal':>8s}")
    for (test, arm), v in sorted(runs.items()):
        m = [sum(x[k] for x in v) / len(v) for k in range(4)]
        print(f"{test:13s}{arm:10s}{len(v):6d}" + "".join(f"{x:8.3f}" for x in m))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=3000, help="E/C rows per arm")
    b.add_argument("--test", type=int, default=600)
    b.add_argument("--out", default="results/llm_proof/data")
    b.add_argument("--packs", nargs="+", default=["monadic"], help="training logics (E/C rows split evenly)")
    t = sub.add_parser("train")
    t.add_argument("--data", default="results/llm_proof/data")
    t.add_argument("--arm", required=True, help="shape, forward, planted, label, fld, or none")
    t.add_argument("--seed", type=int, default=10)
    t.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--epochs", type=int, default=2)
    t.add_argument("--max-len", type=int, default=1024, help="prompt tokens kept (the last ones)")
    t.add_argument("--max-new", type=int, default=512, help="proof tokens trained on and generated")
    t.add_argument("--lora", type=int, default=0)
    t.add_argument("--micro", type=int, default=0, help="micro-batch size (0: the whole batch at once)")
    t.add_argument("--limit", type=int, default=0, help="training rows used (0: all; smoke tests)")
    t.add_argument("--test-limit", type=int, default=0, help="rows per test (0: all; smoke tests)")
    t.add_argument("--out", default="results/llm_proof/runs")
    r = sub.add_parser("report")
    r.add_argument("--runs", default="results/llm_proof/runs")
    args = ap.parse_args()
    {"build": build, "train": train, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
