"""Packs from Datalog programs: every Horn clause is both a rule of the calculus and an axiom of the
semantics, so any (negation-free) Datalog theory becomes a pack with no pack-specific code.

  from semforge.construct.packs.datalog import from_datalog
  pack = from_datalog("kinship", KINSHIP, templates={"parent": "{0} is a parent of {1}", ...})

Clauses are `head(X, Y) :- body(X, Z), body2(Z, Y).` with capitalised variables. Predicates are
relations over one entity sort. Premises are ground atoms; a hypothesis is an atom or its negation, so
a negated derivable atom is a contradiction and anything else is neutral (open world: not derivable
means unknown). Facts in the program (clauses without a body) are axioms of the theory.
"""
from __future__ import annotations

import re

import z3

from ..pack import declare
from ..signature import PROP, Signature, Sort, Symbol, rule

NAMES = ["Ada", "Ben", "Cleo", "Dov", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jade", "Kai", "Lior", "Mia", "Noor",
         "Otto", "Pia", "Quil", "Ravi", "Suki", "Tom", "Uma", "Vic", "Wim", "Xia", "Yan", "Zoe"]
ATOM = re.compile(r"([a-z]\w*)\s*\(([^)]*)\)")


def parse(text) -> list:
    """[(head, [body atoms])] with each atom (predicate, args); `%` starts a comment."""
    atoms = lambda s: [(m.group(1), tuple(x.strip() for x in m.group(2).split(","))) for m in ATOM.finditer(s)]
    return [(atoms(head)[0], atoms(body)) for head, _, body in
            (c.partition(":-") for c in re.sub(r"%.*", "", text).split(".") if c.strip())]


def _clause_text(head, body, template):
    """A clause as an English rule over its variables: "For any X, Y and Z: if ..., then ..."."""
    fill = lambda p, args: re.sub(r"\{(\d+)(?::\w+)?\}", lambda m: args[int(m.group(1))], template(p))
    listed = lambda xs, sep: xs[0] if len(xs) == 1 else sep.join(xs[:-1]) + (sep.rstrip() if len(xs) > 2 else "") + " and " + xs[-1]
    names = sorted({a for _, args in [head, *body] for a in args})
    concl = fill(*head)
    if not body:
        return f"For any {listed(names, ', ')}: {concl}." if names else concl[0].upper() + concl[1:] + "."
    return f"For any {listed(names, ', ')}: if {listed([fill(*b) for b in body], ', ')}, then {concl}."


def from_datalog(name: str, text: str, templates: dict = None, negated: dict = None):
    """A pack whose rules are the program's clauses with a body and whose axioms are all its clauses."""
    clauses = parse(text)
    arity = {}
    for h, body in clauses:
        for p, args in [h, *body]:
            if arity.setdefault(p, len(args)) != len(args):
                raise ValueError(f"{p} used with arities {arity[p]} and {len(args)}")
    templates, negated = templates or {}, negated or {}

    rel = lambda m, p: m.fn(p, *(["Entity"] * arity[p]), PROP)

    def axiom(h, body):
        def build(m):
            env = {}
            var = lambda x: env.setdefault(x, m.var("Entity"))
            app = lambda p, args: rel(m, p)(*map(var, args))
            concl = app(*h)
            prem = [app(*b) for b in body]
            return z3.ForAll(list(env.values()), z3.Implies(z3.And(prem), concl) if prem else concl)
        return build

    default = lambda p: "{0} " + p.replace("_", " ") + " " + " and ".join(f"{{{i}}}" for i in range(1, arity[p]))
    symbols = [Symbol(p, ("Entity",) * k, (templates.get(p) or default(p),), (negated[p],) if p in negated else (),
                      lambda m, *a, p=p: rel(m, p)(*a)) for p, k in sorted(arity.items())]
    sig = Signature.of(sorts=[Sort("Entity", tuple(NAMES))], symbols=symbols, connectives=("not",),
                       axioms=tuple(axiom(h, b) for h, b in clauses),
                       theory=tuple(_clause_text(h, b, lambda p: templates.get(p) or default(p)) for h, b in clauses))
    rules = [rule(f"{h[0]}_{i}", ", ".join(f"({p} {' '.join(a)})" for p, a in body) + f" |- {h[0]} {' '.join(h[1])}",
                  kind=h[0]) for i, (h, body) in enumerate(clauses) if body]
    return declare(name, sig, rules)


KINSHIP = """
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
sibling(X, Y) :- parent(Z, X), parent(Z, Y).
cousin(X, Y) :- parent(A, X), sibling(A, B), parent(B, Y).
uncle(X, Y) :- sibling(X, Z), parent(Z, Y).
"""

PACK = from_datalog("kinship", KINSHIP, templates={
    "parent": "{0} is a parent of {1}", "grandparent": "{0} is a grandparent of {1}",
    "ancestor": "{0} is an ancestor of {1}", "sibling": "{0} and {1} are siblings",
    "cousin": "{0} is a cousin of {1}", "uncle": "{0} is an aunt or uncle of {1}"}, negated={
    "parent": "{0} is not a parent of {1}", "grandparent": "{0} is not a grandparent of {1}",
    "ancestor": "{0} is not an ancestor of {1}", "sibling": "{0} and {1} are not siblings",
    "cousin": "{0} is not a cousin of {1}", "uncle": "{0} is not an aunt or uncle of {1}"})
