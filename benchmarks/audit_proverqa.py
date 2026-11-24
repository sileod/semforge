"""ProverQA (ProverGen, ICLR 2025) at the premise level: are its claimed reasoning chains what the problem needs?

ProverQA gives each problem its context sentences with a FOL translation (`nl2fol`), a FOL conclusion, a
gold answer (True / False / Uncertain) and a reasoning chain whose steps cite facts and rules in English.
The audit needs no calculus: it is semantic, by z3 over the given translations.
- label agreement: z3's label (entailment, contradiction, neutral) against the gold answer;
- claimed premises: the cited facts and rules that are context sentences (the others are intermediate
  conclusions of earlier steps);
- sufficiency: do the claimed premises alone give the gold conclusion (H, or not-H for False)?
- necessity: a premise is mandatory when removing it alone from the whole context loses the conclusion
  (every proof uses it). Share of claimed premises that are mandatory, and of mandatory ones claimed;
- shortcut: a minimal support (deletion-based, claimed premises tried for removal first) smaller than
  the claimed premises, i.e. the conclusion follows from fewer premises than the chain uses.
Neutral rows get label agreement only (no chain to audit). Queries z3 cannot settle within the timeout
are counted, never guessed.

  python -m benchmarks.audit_proverqa --out results/audit/proverqa.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import z3

GOLD = {"A": "entailment", "B": "contradiction", "C": "neutral"}
TOKEN = re.compile(r"\s*(∀|∃|¬|∧|∨|→|↔|⊕|\(|\)|,|[A-Za-z_][A-Za-z0-9_'.-]*)")


class Unparsed(ValueError):
    pass


class Parser:
    """FOL as ProverQA writes it: ∀x / ∃x, ¬ ∧ ∨ ⊕ → ↔, predicates over constants and variables.
    Precedence: ¬ > ∧ > ∨, ⊕ > → > ↔ (→ and ↔ right-associative); a quantifier scopes over the
    formula that follows it."""

    def __init__(self, text, env):
        self.toks, self.i, self.env = self._lex(text), 0, env

    @staticmethod
    def _lex(text):
        out, i = [], 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            m = TOKEN.match(text, i)
            if not m or m.end() == i:
                raise Unparsed(f"cannot read {text[i:i + 20]!r}")
            out.append(m.group(1))
            i = m.end()
        return out

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, want=None):
        t = self.peek()
        if t is None or (want and t != want):
            raise Unparsed(f"expected {want or 'a token'}, got {t}")
        self.i += 1
        return t

    def parse(self):
        f = self.iff({})
        if self.peek() is not None:
            raise Unparsed(f"trailing {self.toks[self.i:]}")
        return f

    def iff(self, bound):
        a = self.imp(bound)
        if self.peek() == "↔":
            self.take()
            return a == self.iff(bound)
        return a

    def imp(self, bound):
        a = self.disj(bound)
        if self.peek() == "→":
            self.take()
            return z3.Implies(a, self.imp(bound))
        return a

    def disj(self, bound):
        a = self.conj(bound)
        while self.peek() in ("∨", "⊕"):
            op = self.take()
            b = self.conj(bound)
            a = z3.Or(a, b) if op == "∨" else z3.Xor(a, b)
        return a

    def conj(self, bound):
        a = self.unary(bound)
        while self.peek() == "∧":
            self.take()
            a = z3.And(a, self.unary(bound))
        return a

    def unary(self, bound):
        t = self.peek()
        if t == "¬":
            self.take()
            return z3.Not(self.unary(bound))
        if t in ("∀", "∃"):
            self.take()
            v = self.take()
            x = z3.Const(f"{v}_{len(bound)}", self.env["sort"])
            body = self.unary({**bound, v: x})
            return z3.ForAll([x], body) if t == "∀" else z3.Exists([x], body)
        if t == "(":
            self.take()
            f = self.iff(bound)
            self.take(")")
            return f
        name = self.take()
        if self.peek() != "(":
            return z3.Bool(f"p_{name}")  # a proposition
        self.take("(")
        args = [self.term(bound)]
        while self.peek() == ",":
            self.take()
            args.append(self.term(bound))
        self.take(")")
        key = (name, len(args))
        if key not in self.env["preds"]:
            self.env["preds"][key] = z3.Function(name, *[self.env["sort"]] * len(args), z3.BoolSort())
        return self.env["preds"][key](*args)

    def term(self, bound):
        t = self.take()
        if t in bound:
            return bound[t]
        if t not in self.env["consts"]:
            self.env["consts"][t] = z3.Const(f"c_{t}", self.env["sort"])
        return self.env["consts"][t]


class Problem:
    def __init__(self, x, timeout_ms):
        self.env = {"sort": z3.DeclareSort("U"), "preds": {}, "consts": {}}
        self.sentences = list(x["nl2fol"])
        self.premises = [Parser(x["nl2fol"][s], self.env).parse() for s in self.sentences]
        self.goal = Parser(x["conclusion_fol"], self.env).parse()
        self.gold = GOLD[x["answer"]]
        self.timeout_ms = timeout_ms

    def entails(self, idx, goal):
        s = z3.Solver()
        s.set("timeout", self.timeout_ms)
        s.add(*(self.premises[i] for i in idx), z3.Not(goal))
        r = s.check()
        if r == z3.unknown:
            raise TimeoutError
        return r == z3.unsat

    def label(self):
        every = range(len(self.premises))
        if self.entails(every, z3.BoolVal(False)):
            return "inconsistent"
        if self.entails(every, self.goal):
            return "entailment"
        return "contradiction" if self.entails(every, z3.Not(self.goal)) else "neutral"

    def minimal_support(self, goal, first):
        """Deletion-based minimal support of `goal`, trying the premises in `first` for removal first."""
        keep = list(range(len(self.premises)))
        for i in sorted(keep, key=lambda i: (i not in first, i)):
            rest = [j for j in keep if j != i]
            if self.entails(rest, goal):
                keep = rest
        return keep


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def claimed(x, sentences):
    """Indices of context sentences cited in the reasoning chain (fact and rule lines)."""
    index = {norm(s): i for i, s in enumerate(sentences)}
    out, cited = set(), 0
    for line in x["reasoning"].splitlines():
        m = re.match(r"\s*(fact\d*|rule\d*)\s*:\s*(.+)", line)
        if m:
            cited += 1
            i = index.get(norm(m.group(2)))
            if i is not None:
                out.add(i)
    steps = len(re.findall(r"^\s*conclusion\s*:", x["reasoning"], flags=re.M))
    return out, steps, cited


def audit(x, timeout_ms):
    p = Problem(x, timeout_ms)
    row = {"id": x["id"], "gold": p.gold, "premises": len(p.premises), "label": p.label()}
    if p.gold == "neutral" or row["label"] != p.gold:
        return row
    goal = p.goal if p.gold == "entailment" else z3.Not(p.goal)
    used, steps, cited = claimed(x, p.sentences)
    every = range(len(p.premises))
    mandatory = {i for i in every if not p.entails([j for j in every if j != i], goal)}
    support = p.minimal_support(goal, used)
    row.update({"steps": steps, "cited_lines": cited, "claimed": len(used),
                "sufficient": p.entails(sorted(used), goal),
                "mandatory": len(mandatory), "claimed_mandatory": len(used & mandatory),
                "minimal_support": len(support), "support_outside_claim": len(set(support) - used)})
    return row


def summarize(rows):
    ec = [r for r in rows if "claimed" in r]
    mean = lambda k, rs: round(sum(r[k] for r in rs) / len(rs), 2) if rs else None
    share = lambda f, rs: round(sum(map(f, rs)) / len(rs), 3) if rs else None
    return {
        "rows": len(rows),
        "label_agreement": share(lambda r: r["label"] == r["gold"], rows),
        "by_gold": {g: share(lambda r: r["label"] == g, [r for r in rows if r["gold"] == g]) for g in GOLD.values()},
        "z3_labels": dict(Counter(r["label"] for r in rows)),
        "ec_audited": len(ec),
        "mean_steps": mean("steps", ec), "mean_claimed_premises": mean("claimed", ec),
        "mean_mandatory": mean("mandatory", ec), "mean_minimal_support": mean("minimal_support", ec),
        "claim_sufficient": share(lambda r: r["sufficient"], ec),
        "claimed_premises_mandatory": round(sum(r["claimed_mandatory"] for r in ec) / max(1, sum(r["claimed"] for r in ec)), 3),
        "mandatory_premises_claimed": round(sum(r["claimed_mandatory"] for r in ec) / max(1, sum(r["mandatory"] for r in ec)), 3),
        "shortcut": share(lambda r: r["minimal_support"] < r["claimed"], ec),
        "rows_with_unneeded_claimed_premise": share(lambda r: r["claimed_mandatory"] < r["claimed"], ec),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", default="easy,medium,hard")
    ap.add_argument("--n", type=int, default=500, help="rows per level (the dev split has 500)")
    ap.add_argument("--timeout-ms", type=int, default=5000)
    ap.add_argument("--out", default="results/audit/proverqa.json")
    args = ap.parse_args()
    from huggingface_hub import hf_hub_download
    out = {"source": "opendatalab/ProverQA dev", "levels": {}}
    for level in args.levels.split(","):
        data = json.load(open(hf_hub_download("opendatalab/ProverQA", f"dev/{level}.json", repo_type="dataset")))
        rows, failed = [], Counter()
        for x in data[:args.n]:
            try:
                rows.append(audit(x, args.timeout_ms))
            except Unparsed:
                failed["unparsed"] += 1
            except TimeoutError:
                failed["z3_timeout"] += 1
        out["levels"][level] = {**summarize(rows), "dropped": dict(failed)}
        out.setdefault("rows", {})[level] = rows
        print(level, json.dumps(out["levels"][level]), flush=True)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
