"""FOLIO with the corrected annotations of "Fixing FOLIO and MALLS" (2606.02837): validation stories and
conclusions, first-order formulas parsed into terms and decided by z3.

Data: https://github.com/dslab-uniud/Fixing-FOLIO-and-MALLS, file
core/dataset/datasets/folio/dataset_rawpairs.jsonl (path in $SEMFORGE_FOLIO, or the default below).
The corrected story formula is one conjunction; its top-level conjuncts are the premises, so premises are
finer than FOLIO's English sentences. Distinct constants denote distinct objects by default (the
reading FOLIO's labels assume: without it, 6 of 202 corrected labels are not entailed, with it 4, each
needing unstated background knowledge such as "bees are animals" or "taller is asymmetric").

Terms: ("forall", var, body), ("exists", var, body), ("not", f), ("and", f, g, ...), ("or", ...),
("xor", f, g), ("->", f, g), ("<->", f, g), ("=", s, t), and atoms ("Pred", arg, ...).
"""
from __future__ import annotations

import json
import os
import re

import z3

PATH = os.environ.get("SEMFORGE_FOLIO", os.path.expanduser(
    "~/data/folio_fixed/repo/core/dataset/datasets/folio/dataset_rawpairs.jsonl"))
TOKEN = re.compile(r"\s*(∀|∃|¬|∧|∨|⊕|→|↔|≠|=|\(|\)|,|[^\W\d][\w']*)")
BINARY = [("↔", "<->"), ("→", "->"), ("∨", "or"), ("⊕", "xor"), ("∧", "and")]  # loosest first


class ParseError(ValueError):
    pass


def tokens(s):
    out, i, s = [], 0, s.strip()
    while i < len(s):
        m = TOKEN.match(s, i)
        if not m:
            raise ParseError(f"bad character at {s[i:i + 10]!r}")
        out.append(m.group(1))
        i = m.end()
    return out


class Parser:
    """Quantifiers take the formula right after them (`∀x (A) ∧ ∀x (B)` is a conjunction); `→` and `↔`
    associate to the right."""

    def __init__(self, s):
        self.t, self.i = tokens(s), 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, tok=None):
        x = self.peek()
        if tok is not None and x != tok:
            raise ParseError(f"expected {tok!r}, got {x!r}")
        self.i += 1
        return x

    def parse(self):
        f = self.binary(0)
        if self.peek() is not None:
            raise ParseError(f"trailing {self.t[self.i:self.i + 3]}")
        return f

    def binary(self, level):
        if level == len(BINARY):
            return self.unary()
        sym, op = BINARY[level]
        left = self.binary(level + 1)
        if op in ("->", "<->"):
            if self.peek() == sym:
                self.eat()
                return (op, left, self.binary(level))
            return left
        args = [left]
        while self.peek() == sym:
            self.eat()
            args.append(self.binary(level + 1))
        if len(args) == 1:
            return left
        return (op, *args) if op != "xor" else _xor(args)

    def unary(self):
        x = self.peek()
        if x == "¬":
            self.eat()
            return ("not", self.unary())
        if x in ("∀", "∃"):
            self.eat()
            v = self.eat()
            return ("forall" if x == "∀" else "exists", v, self.unary())
        if x == "(":
            self.eat()
            f = self.binary(0)
            self.eat(")")
            return f
        name = self.eat()
        if self.peek() == "(":
            self.eat()
            args = [self.eat()]
            while self.peek() == ",":
                self.eat()
                args.append(self.eat())
            self.eat(")")
            return (name, *args)
        if self.peek() in ("=", "≠"):  # a term equation
            op, rhs = self.eat(), self.eat()
            return ("=", name, rhs) if op == "=" else ("not", ("=", name, rhs))
        raise ParseError(f"unexpected {name!r}")


def _xor(args):
    f = args[0]
    for g in args[1:]:
        f = ("xor", f, g)
    return f


def parse(s):
    return Parser(s).parse()


def conjuncts(f):
    return list(f[1:]) if isinstance(f, tuple) and f[0] == "and" else [f]


def free_vars(f, bound=frozenset(), variables=None):
    """Variables of f not bound by a quantifier; `variables` are names ever quantified over."""
    if f[0] in ("forall", "exists"):
        return free_vars(f[2], bound | {f[1]}, variables)
    if f[0] in ("not", "and", "or", "xor", "->", "<->"):
        return set().union(*(free_vars(g, bound, variables) for g in f[1:]))
    return {a for a in f[1:] if a in variables and a not in bound}


def quantified(f, out=None):
    out = set() if out is None else out
    if f[0] in ("forall", "exists"):
        out.add(f[1])
        quantified(f[2], out)
    elif f[0] in ("not", "and", "or", "xor", "->", "<->"):
        for g in f[1:]:
            quantified(g, out)
    return out


class Z3:
    """One uninterpreted sort; predicates by name and arity; constants without unique names."""

    def __init__(self):
        self.sort, self.fns, self.consts = z3.DeclareSort("Entity"), {}, {}

    def term(self, a, env):
        if a in env:
            return env[a]
        if a not in self.consts:
            self.consts[a] = z3.Const(f"c_{a}", self.sort)
        return self.consts[a]

    def formula(self, f, env=None):
        env = env or {}
        op = f[0]
        if op in ("forall", "exists"):
            v = z3.Const(f"v{len(env)}_{f[1]}", self.sort)
            body = self.formula(f[2], {**env, f[1]: v})
            return z3.ForAll([v], body) if op == "forall" else z3.Exists([v], body)
        if op == "not":
            return z3.Not(self.formula(f[1], env))
        if op in ("and", "or"):
            return (z3.And if op == "and" else z3.Or)([self.formula(g, env) for g in f[1:]])
        if op == "xor":
            return z3.Xor(self.formula(f[1], env), self.formula(f[2], env))
        if op == "->":
            return z3.Implies(self.formula(f[1], env), self.formula(f[2], env))
        if op == "<->":
            return self.formula(f[1], env) == self.formula(f[2], env)
        if op == "=":
            return self.term(f[1], env) == self.term(f[2], env)
        key = (op, len(f) - 1)
        if key not in self.fns:
            self.fns[key] = z3.Function(f"{op}_{len(f) - 1}", *[self.sort] * (len(f) - 1), z3.BoolSort())
        return self.fns[key](*(self.term(a, env) for a in f[1:]))


class Undecided(RuntimeError):
    pass


class Inconsistent(Undecided):
    """The premises are unsatisfiable in this encoding, so every conclusion would be entailed vacuously."""


def entails(premises, goal, timeout_ms=10_000, unique_names=True) -> bool:
    m, s = Z3(), z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(*(m.formula(p) for p in premises), z3.Not(m.formula(goal)))
    if unique_names and len(m.consts) > 1:
        s.add(z3.Distinct(*m.consts.values()))
    r = s.check()
    if r == z3.unknown:
        raise Undecided(s.reason_unknown())
    return r == z3.unsat


FALSE = ("and", ("Zz0",), ("not", ("Zz0",)))


def label(premises, goal, **kw):
    """z3's label; raises Inconsistent for an unsatisfiable premise set (three FOLIO stories under unique
    names), whose conclusions are all vacuously entailed and whose minimal premise sets are unsat cores."""
    if entails(premises, FALSE, **kw):
        raise Inconsistent("premises are unsatisfiable")
    if entails(premises, goal, **kw):
        return "entailment"
    if entails(premises, ("not", goal), **kw):
        return "contradiction"
    return "neutral"


LABELS = {"True": "entailment", "False": "contradiction", "Uncertain": "neutral"}


def load(path=PATH):
    """[(id, premises, conclusion, label, row)]; rows that do not parse or have free variables are
    returned with premises None and the error in row["error"]."""
    rows = [json.loads(line) for line in open(path)]
    stories = {r["id"].split("_")[1]: r for r in rows if r["id"].startswith("story")}
    out = []
    for r in rows:
        if not r["id"].startswith("concl"):
            continue
        story = stories[r["id"].split("_")[1]]
        try:
            prem, goal = conjuncts(parse(story["FOL_sentence_new"])), parse(r["FOL_sentence_new"])
            variables = set().union(*(quantified(f) for f in prem + [goal]), {"x", "y", "z", "w"})
            if any(free_vars(f, variables=variables) for f in prem + [goal]):
                raise ParseError("free variable")
        except ParseError as e:
            out.append((r["id"], None, None, LABELS[r["label"]], {**r, "error": str(e)}))
            continue
        out.append((r["id"], prem, goal, LABELS[r["label"]], r))
    return out
