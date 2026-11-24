"""FLD×2 (Morishita et al., NeurIPS 2024) as a semforge domain.

Facts and hypotheses come with formulas in FLD's notation: atoms `{A}`, `{A}{a}` (predicate and constant),
`{A}x` (predicate and variable); connectives ¬, &, v, ->; quantifier prefixes `(x):` and `(Ex):`. They are
parsed to z3 first-order formulas over one uninterpreted sort, so a label is certified by z3 (PROVED iff the
facts entail the hypothesis, DISPROVED iff they entail its negation) and justifications are minimal fact
sets, enumerated with MARCO as for the other domains.

A fact's kind is its logical skeleton (predicates -> P, constants -> c), e.g. `∀x: ¬Px -> (¬Px & Px)`;
literal facts have no kind, like facts in the logic domain. FLD's gold proof is kept, so the audit can
also ask whether the gold proof is the only justification (no shortcut) and whether z3 agrees with the label.
"""
from __future__ import annotations

import functools
import re
from dataclasses import dataclass, replace

import z3

from .base import Domain

LABEL = {"PROVED": "entailment", "DISPROVED": "contradiction", "UNKNOWN": "neutral"}
TOKEN = re.compile(r"\s*(\(E?x\):|¬|&|v(?![A-Za-z])|->|\(|\)|\{[A-Za-z]+\}(?:\{[a-z]+\}|x)?)")
D = z3.DeclareSort("D")


class _Parser:
    """Recursive descent: imp := or ('->' imp)? ; or := and ('v' and)* ; and := un ('&' un)* ;
    un := '¬' un | quant | '(' imp ')' | atom ; quant := '(x):' imp | '(Ex):' imp."""

    def __init__(self, text, env):
        self.toks, pos = [], 0
        text = text.strip()
        while pos < len(text):
            m = TOKEN.match(text, pos)
            if not m:
                raise ValueError(f"cannot tokenize {text[pos:pos + 20]!r} in {text!r}")
            self.toks.append(m.group(1))
            pos = m.end()
        self.i, self.env, self.var = 0, env, None

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, t=None):
        tok = self.peek()
        if t is not None and tok != t:
            raise ValueError(f"expected {t}, got {tok}")
        self.i += 1
        return tok

    def parse(self):
        e = self.imp()
        if self.peek() is not None:
            raise ValueError(f"trailing {self.toks[self.i:]}")
        return e

    def imp(self):
        a = self.or_()
        if self.peek() == "->":
            self.take()
            return z3.Implies(a, self.imp())
        return a

    def or_(self):
        xs = [self.and_()]
        while self.peek() == "v":
            self.take()
            xs.append(self.and_())
        return xs[0] if len(xs) == 1 else z3.Or(xs)

    def and_(self):
        xs = [self.un()]
        while self.peek() == "&":
            self.take()
            xs.append(self.un())
        return xs[0] if len(xs) == 1 else z3.And(xs)

    def un(self):
        t = self.peek()
        if t == "¬":
            self.take()
            return z3.Not(self.un())
        if t in ("(x):", "(Ex):"):
            self.take()
            x, outer = z3.Const(f"x{self.i}", D), self.var
            self.var = x
            body = self.imp()
            self.var = outer
            return (z3.ForAll if t == "(x):" else z3.Exists)([x], body)
        if t == "(":
            self.take()
            e = self.imp()
            self.take(")")
            return e
        return self.atom(self.take())

    def atom(self, t):
        m = re.fullmatch(r"\{([A-Za-z]+)\}(?:\{([a-z]+)\}|(x))?", t)
        if not m:
            raise ValueError(f"bad atom {t!r}")
        p, c, v = m.groups()
        if c is None and v is None:
            return self.env.setdefault(("prop", p), z3.Bool(f"p_{p}"))
        f = self.env.setdefault(("pred", p), z3.Function(f"P_{p}", D, z3.BoolSort()))
        if v:
            if self.var is None:
                raise ValueError(f"free variable in {t}")
            return f(self.var)
        return f(self.env.setdefault(("const", c), z3.Const(f"c_{c}", D)))


def skeleton(formula: str) -> str:
    s = re.sub(r"\{[A-Za-z]+\}\{[a-z]+\}", "Pc", formula)
    s = re.sub(r"\{[A-Za-z]+\}x", "Px", s)
    s = re.sub(r"\{[A-Za-z]+\}", "P", s)
    s = s.replace("(x):", "∀x:").replace("(Ex):", "∃x:")
    return re.sub(r"\s+", " ", s).strip()


def rule_shape(skel: str) -> str:
    """A skeleton without negations and with atoms unified: comparable to the logic domain's rule shapes."""
    return re.sub(r"P[cx]", "P", skel.replace("¬", "")).replace("  ", " ")


def is_literal(formula: str) -> bool:
    return re.fullmatch(r"¬?\{[A-Za-z]+\}(\{[a-z]+\})?", formula.strip()) is not None


def split_facts(s: str) -> dict:
    parts = re.split(r"(?:^|\s)(fact\d+):\s*", s.strip())
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


@dataclass
class FLDInstance:
    uid: str
    facts: list          # premises kept: (fact id, formula) pairs
    formula: dict        # fact id -> formula string
    text: dict           # fact id -> English sentence
    hyp_formula: str
    hyp_text: str
    gold: str            # FLD label mapped to E/C/N
    gold_support: frozenset  # fact ids used by FLD's first gold proof
    steps: int | None
    depth: int | None
    instantiations: int = 0  # steps of the gold proof that only instantiate a universal fact


def instantiation_steps(proof: str, formula: dict) -> int:
    """Steps of an FLD proof whose sole premise is a universal fact (`fact3 -> int1: ...`, fact3 = (x): ...).
    FLD counts them in `steps`; a calculus over ground instances does not."""
    n = 0
    for step in proof.split(";"):
        ids = re.findall(r"(fact\d+|int\d+)", step.split("->")[0]) if "->" in step else []
        n += len(ids) == 1 and ids[0].startswith("fact") and formula.get(ids[0], "").lstrip().startswith("(x)")
    return n


def from_row(row, uid):
    formula, text = split_facts(row["facts_formula"]), split_facts(row["facts"])
    proof = (row["proofs_formula"] or [""])[0] + " " + " ".join(row["negative_proofs"] or [])
    used = frozenset(re.findall(r"fact\d+", (row["proofs_formula"] or [""])[0]))
    facts = [(f, formula[f]) for f in sorted(formula, key=lambda f: int(f[4:]))]
    return FLDInstance(uid, facts, formula, text, row["hypothesis_formula"],
                       row["hypothesis"], LABEL[row["world_assump_label"]], used, row["steps"], row["depth"],
                       instantiation_steps((row["proofs_formula"] or [""])[0], formula))


class FLDDomain(Domain):
    """Justifications of FLD×2 labels re-derived with z3. `timeout_ms` bounds each entailment check; an
    instance with any undecided check has no certified label and is skipped."""
    name = "fld"

    def __init__(self, rows=None, timeout_ms=5000):
        self.rows = rows if rows is not None else load()
        self.timeout_ms = timeout_ms

    # --- certification
    def _encode(self, inst):
        env = {}  # translation distractors ("fake_formula") have no logical content
        facts = {f: _Parser(f[1], env).parse() if f[1] != "fake_formula" else z3.BoolVal(True) for f in inst.facts}
        hyp = _Parser(inst.hyp_formula, env).parse()
        return facts, hyp

    def _entails(self, facts, goal, keep):
        s = z3.Solver()
        s.set("timeout", self.timeout_ms)
        s.add(*[facts[f] for f in keep])
        s.add(z3.Not(goal))
        r = s.check()
        if r == z3.unknown:
            raise TimeoutError
        return r == z3.unsat

    def label(self, inst):
        facts, hyp = self._encode(inst)
        try:
            if self._entails(facts, hyp, inst.facts):
                return "entailment"
            if self._entails(facts, z3.Not(hyp), inst.facts):
                return "contradiction"
        except TimeoutError:
            return None
        return "neutral"

    def justifications(self, inst, lab=None, limit=64):
        from ..core.lattice import Oracle, enumerate_lattice
        lab = lab or self.label(inst)
        if lab not in ("entailment", "contradiction"):
            return None, False
        facts, hyp = self._encode(inst)
        goal = hyp if lab == "entailment" else z3.Not(hyp)
        o = Oracle(lambda s: self._entails(facts, goal, s))
        try:
            sups, _, complete = enumerate_lattice(inst.facts, o, limit=limit)
        except TimeoutError:
            return None, False
        return sups, complete

    def kind(self, fact):
        return None if is_literal(fact[1]) or fact[1] == "fake_formula" else skeleton(fact[1])

    def without(self, inst, drop):
        return replace(inst, facts=[f for f in inst.facts if f not in drop])

    def render(self, inst):
        return " ".join(inst.text.get(f, form) for f, form in inst.facts), inst.hyp_text

    def sample(self, rng, level=None):
        """One FLD×2 training row, with its label as z3 certifies it (skipped if undecided)."""
        while True:
            i = rng.randrange(len(self.rows))
            inst = from_row(self.rows[i], i)
            lab = self.label(inst)
            if lab is not None:
                return [(inst, lab)]

    def annotations(self, inst, p):
        """FLD's own annotations, to check against the certified analysis."""
        ids = lambda sup: frozenset(f for f, _ in sup)
        return {"gold_label": inst.gold, "steps": inst.steps,
                "gold_proof_is_a_justification": (inst.gold_support in {ids(j) for j in p.justifications}) if p else None,
                "gold_proof_size": len(inst.gold_support)}

    def world(self, inst):
        return f"fld:{inst.uid}"

    def premises(self, inst):
        return list(inst.facts)


@functools.lru_cache(maxsize=None)
def load(split="train"):
    from datasets import load_dataset
    return load_dataset("hitachi-nlp/FLDx2", split=split)
