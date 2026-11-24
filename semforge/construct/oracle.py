"""The z3 oracle, compiled from a signature: entailment over any pack with no pack-specific solver code.

Sorts become z3 sorts: entity sorts are uninterpreted (with all constants pairwise distinct when the
sort has unique names), predicate sorts are unary functions to Bool, propositional atoms are Bools.
Each symbol's `z3` function receives the model and its translated arguments. Entailment is
unsatisfiability of premises ∧ ¬goal, so labels are exact whenever z3 decides the fragment (it does for
every pack here: propositional, monadic first-order, finite-domain constraints).

With `Signature.world`, formulas are true at worlds (the standard translation): atoms become predicates
over that sort, symbols get their formula arguments as functions of a world, and entailment is local
consequence at a fresh world.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import count, product

import z3

from .signature import PROP, Signature


class Undecided(RuntimeError):
    """z3 returned unknown (e.g. incomplete quantifier instantiation): no certifiable label."""


class Model:
    """Translation context for one entailment query; symbol `z3` functions use `var` and `fn`."""

    def __init__(self, sig: Signature, formulas):
        self.sig, self.kinds = sig, sig.infer(formulas)
        for axiom in sig.axioms:  # constants a theory axiom quantifies over exist whether or not the query names them
            for c, sort in getattr(axiom, "declares", {}).items():
                if self.kinds.setdefault(c, sort) != sort:
                    raise ValueError(f"{c!r} declared as {sort} but used as {self.kinds[c]}")
        self._sorts, self._fns, self._n = {}, {}, count()
        self.consts = defaultdict(list)
        self.terms = {c: self._term(c, s) for c, s in self.kinds.items()}

    def sort(self, name):
        if name not in self._sorts:
            self._sorts[name] = z3.DeclareSort(name)
        return self._sorts[name]

    def _term(self, c, sort):
        if sort == PROP:
            return z3.Function(f"p_{c}", self.sort(self.sig.world), z3.BoolSort()) if self.sig.world \
                else z3.Bool(f"p_{c}")
        spec = self.sig.sorts[sort]
        if spec.kind == "predicate":
            return z3.Function(f"{sort}_{c}", self.sort(spec.over), z3.BoolSort())
        t = z3.Const(f"{sort}_{c}", self.sort(sort))
        self.consts[sort].append(t)
        return t

    def var(self, sort):
        """A fresh bound variable of an entity sort, for quantifier symbols."""
        return z3.Const(f"_x{next(self._n)}", self.sort(sort))

    def fn(self, name, *sorts):
        """A pack function, e.g. fn("colour", "Person", "Colour"): the last sort is the range ("Int" and
        PROP are z3's integers and Booleans)."""
        if name not in self._fns:
            *dom, rng = sorts
            builtin = {PROP: z3.BoolSort(), "Int": z3.IntSort()}
            sort = lambda s: builtin[s] if s in builtin else self.sort(s)
            self._fns[name] = z3.Function(name, *map(sort, dom), sort(rng))
        return self._fns[name]

    def formula(self, f, w=None):
        """f as a z3 Bool; at world w when the signature has worlds."""
        if isinstance(f, str):
            return self.terms[f](w) if self.sig.world else self.terms[f]
        op, args = f[0], f[1:]
        if op == "not":
            return z3.Not(self.formula(args[0], w))
        if op == "and":
            return z3.And([self.formula(a, w) for a in args])
        if op == "or":
            return z3.Or([self.formula(a, w) for a in args])
        if op == "->":
            return z3.Implies(self.formula(args[0], w), self.formula(args[1], w))
        sym = self.sig.symbols[op]
        if not self.sig.world:
            return sym.z3(self, *(self.formula(a) if s == PROP else self.terms[a] for a, s in zip(args, sym.args)))
        return sym.z3(self, *((lambda v, a=a: self.formula(a, v)) if s == PROP else self.terms[a]
                              for a, s in zip(args, sym.args)), w)

    def axioms(self, unique_names: bool = True):
        distinct = [z3.Distinct(cs) for s, cs in self.consts.items() if self.sig.sorts[s].distinct and len(cs) > 1]
        return (distinct if unique_names else []) + [a(self) for a in self.sig.axioms]


# Frame properties of a binary relation symbol r over an entity sort, as Signature.axioms entries.

def _rel(m, r, sort):
    return m.fn(r, sort, sort, PROP)


def reflexive(r, sort):
    def axiom(m):
        x = m.var(sort)
        return z3.ForAll([x], _rel(m, r, sort)(x, x))
    return axiom


def irreflexive(r, sort):
    def axiom(m):
        x = m.var(sort)
        return z3.ForAll([x], z3.Not(_rel(m, r, sort)(x, x)))
    return axiom


def symmetric(r, sort):
    def axiom(m):
        x, y = m.var(sort), m.var(sort)
        return z3.ForAll([x, y], z3.Implies(_rel(m, r, sort)(x, y), _rel(m, r, sort)(y, x)))
    return axiom


def transitive(r, sort):
    def axiom(m):
        x, y, z = m.var(sort), m.var(sort), m.var(sort)
        R = _rel(m, r, sort)
        return z3.ForAll([x, y, z], z3.Implies(z3.And(R(x, y), R(y, z)), R(x, z)))
    return axiom


# Closed-world axioms for a finite, fully stated structure (model checking rather than open entailment).
# Each declares the constants it is about (`axiom.declares`), and Model registers them before translating
# the query, so the structure never depends on which names the premises and goal happen to mention.

def _declaring(axiom, declares):
    axiom.declares = declares
    return axiom


def domain_closure(sort, constants):
    """The named constants are all the elements of `sort`."""
    def axiom(m):
        x = m.var(sort)
        return z3.ForAll([x], z3.Or([x == m.terms[c] for c in constants] or [z3.BoolVal(False)]))
    return _declaring(axiom, {c: sort for c in constants})


def exactly(r, sort, pairs):
    """Relation symbol r over `sort` holds of exactly the listed pairs of constants (its completion)."""
    def axiom(m):
        x, y = m.var(sort), m.var(sort)
        listed = [z3.And(x == m.terms[a], y == m.terms[b]) for a, b in pairs]
        return z3.ForAll([x, y], _rel(m, r, sort)(x, y) == z3.Or(listed or [z3.BoolVal(False)]))
    return _declaring(axiom, {c: sort for pair in pairs for c in pair})


def true_exactly_at(atom, sort, worlds):
    """Propositional atom `atom` (with `Signature.world`) is true at exactly the listed worlds."""
    def axiom(m):
        x = m.var(sort)
        listed = [x == m.terms[w] for w in worlds]
        return z3.ForAll([x], m.terms[atom](x) == z3.Or(listed or [z3.BoolVal(False)]))
    return _declaring(axiom, {atom: PROP, **{w: sort for w in worlds}})


def query(sig: Signature, premises, goal, domain: int = 0, unique_names: bool = True) -> z3.Solver:
    """A solver holding premises ∧ ¬goal (and the theory axioms): unsat iff the premises entail the goal.
    With `domain` = k > 0, every sort is also bounded to at most k elements: a model then is still a
    countermodel (non-entailment is exact), but unsat proves nothing. Without `unique_names`, distinct
    constants may denote the same object (for schematic checks, where they are metavariables)."""
    m = Model(sig, [*premises, goal])
    w = z3.Const("_actual", m.sort(sig.world)) if sig.world else None
    s = z3.Solver()
    s.add(*m.axioms(unique_names), *(m.formula(p, w) for p in premises), z3.Not(m.formula(goal, w)))
    for name, sort in (m._sorts.items() if domain else ()):
        x, elems = z3.Const(f"_d_{name}", sort), [z3.Const(f"_e{i}_{name}", sort) for i in range(domain)]
        s.add(z3.ForAll([x], z3.Or([x == e for e in elems])))
    return s


def z3_entails(sig: Signature, timeout_ms: int = 10_000):
    """Entailment by z3. A query z3 cannot settle within the timeout (quantified frame axioms over nested
    modalities can make it search indefinitely) raises Undecided: the instance is dropped, never guessed."""
    def entails(premises, goal) -> bool:
        s = query(sig, premises, goal)
        s.set("timeout", timeout_ms)
        r = s.check()
        if r == z3.unknown:
            raise Undecided(f"z3 undecided: {s.reason_unknown()}")
        return r == z3.unsat
    return entails


def countermodel(sig: Signature, premises, goal, timeout_ms: int = 10_000, sizes=(1, 2, 3, 4, 6, 8)):
    """A finite model of premises ∧ ¬goal, as a certificate `semforge.certcheck` evaluates without z3:
    {"smt2": the assertions, "model": {"sorts": {sort: [elements]}, "functions": {name: {"else": value,
    "rows": [[*args, value]]}}}}: a total table, listing the argument tuples whose value is not `else`. Small models are tried first (every sort bounded by each of `sizes`, then
    unbounded) to keep tables short. None when there is none (entailment), z3 does not find one in time,
    or a function takes integer arguments (its table would be infinite)."""
    for k in (*sizes, 0):
        s = query(sig, premises, goal, domain=k)
        s.set("timeout", timeout_ms if k == 0 else min(timeout_ms, 1000))
        r = s.check()
        if r == z3.sat:
            break
        if r == z3.unsat and k == 0:
            return None
    else:
        return None
    m, decls, seen = s.model(), {}, set()
    todo = list(s.assertions())
    while todo:
        e = todo.pop()
        if e.get_id() in seen:
            continue
        seen.add(e.get_id())
        if z3.is_quantifier(e):
            todo.append(e.body())
        elif z3.is_app(e):
            if e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                decls[e.decl().name()] = e.decl()
            todo.extend(e.children())
    sorts = {}
    for d in decls.values():
        for srt in [d.domain(i) for i in range(d.arity())] + [d.range()]:
            if srt.kind() == z3.Z3_UNINTERPRETED_SORT and srt.name() not in sorts:
                if srt not in m.sorts():
                    return None
                sorts[srt.name()] = (srt, list(m.get_universe(srt)))
    value = lambda v: (z3.is_true(v) if z3.is_bool(v) else v.as_long() if z3.is_int_value(v) else str(v))
    elements = lambda srt: ([z3.BoolVal(True), z3.BoolVal(False)] if srt.kind() == z3.Z3_BOOL_SORT
                            else sorts[srt.name()][1] if srt.kind() == z3.Z3_UNINTERPRETED_SORT else None)
    functions = {}
    for name, d in decls.items():
        domains = [elements(d.domain(i)) for i in range(d.arity())]
        if any(x is None for x in domains):
            return None
        table = [[*map(value, args), value(m.eval(d(*args), model_completion=True))] for args in product(*domains)]
        default = max({json.dumps(r[-1]): r[-1] for r in table}.values(), key=[r[-1] for r in table].count)
        functions[name] = {"else": default, "rows": [r for r in table if r[-1] != default]}
    return {"smt2": s.to_smt2(),
            "model": {"sorts": {n: [str(x) for x in u] for n, (_, u) in sorts.items()}, "functions": functions}}
