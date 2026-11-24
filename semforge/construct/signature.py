"""Signatures: what a pack's formulas are made of, declared once and used everywhere.

A signature lists sorts (kinds of things: people, colours, predicates) and symbols (atomic sentence
forms over them, e.g. "x's house is v"). From it the engine derives, with no pack code:
  - unification and equality modulo symmetric/unordered arguments (`unordered`)
  - which metavariables stand for whole formulas, and what a free formula slot becomes (`leaf`)
  - the sort of every constant in an instance (`infer`), for lexicons, z3 and checks
  - the z3 semantics (oracle.py) and the English realization (realize.py)
  - a check of every rule against the symbols it uses (`check_rule`)

Interpreted symbols (`Symbol.holds`) are theory relations computed on demand: arithmetic or order over a
closed sort, written in rules like any symbol ("(plus A D C)") but never stated or stored as facts.

Connectives (not, and, or, ->) are built in: formulas are atoms (propositional constants), symbol
applications, or connectives over formulas. Rules are written in a small prefix DSL (`rule`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from .calculus import Rule
from .terms import is_var, variables

PROP = "Prop"
CONNECTIVES = {"not": 1, "and": 2, "or": 2, "->": 2}


@dataclass(frozen=True)
class Sort:
    name: str
    lexicon: Union[tuple, Callable] = ()  # words for constants, or rng, n -> n distinct phrases
    kind: str = "entity"                  # entity | predicate (unary, over `over`)
    over: Optional[str] = None
    distinct: bool = True                 # unique names: distinct constants denote distinct objects
    fixed: tuple = ()                     # (constant, phrase) pairs: a closed sort whose constants have fixed
                                          # meanings (houses, values of a puzzle); construction binds to them


@dataclass(frozen=True)
class Symbol:
    name: str
    args: tuple                     # argument sorts; PROP for formula arguments
    templates: tuple = ()           # realizations: "{0}'s house is {1}"; "{0:a}" adds an article, "{0:s}" a plural
    negated: tuple = ()             # realizations of its negation (default: "it is not the case that")
    z3: Optional[Callable] = None   # (model, *translated args) -> z3 Bool; with Signature.world, PROP args
                                    # are world -> z3 Bool functions and the current world comes last
    unordered: tuple = ()           # groups of argument indices whose order is irrelevant
    leaf: bool = True               # may fill an unconstrained formula slot in construction
    holds: Optional[Callable] = None  # interpreted: (*constants) -> bool, over closed sorts only. Its true
                                      # instances are theory facts computed when a rule needs them (never
                                      # premises, never derived); `z3` must agree, which check.py verifies


@dataclass
class Signature:
    sorts: dict = field(default_factory=dict)    # name -> Sort
    symbols: dict = field(default_factory=dict)  # name -> Symbol
    connectives: tuple = tuple(CONNECTIVES)
    prop_lexicon: Optional[Callable] = None      # rng, n -> phrases for propositional atoms
    axioms: tuple = ()                           # theory axioms, model -> z3 formula (see oracle.reflexive)
    world: Optional[str] = None                  # entity sort that formulas are true *at* (nested modalities)
    nesting: int = 2                             # how deep leaf symbols over formulas nest in a free slot
    background: tuple = ()                       # ground theory facts (e.g. house order): every closure has
                                                 # them; never an instance premise, never rendered
    theory: tuple = ()                           # English sentences stating `axioms` and `background`, rendered
                                                 # before every premise list: a label must not rest on unstated theory
    solved_grounding: bool = False               # bind closed-sort metavariables so a proof's nodes are jointly
                                                 # satisfiable (z3), not at random. For functional symbols (one
                                                 # score per person), where leaves constrain each other through
                                                 # paths the proof does not use

    @property
    def background_heads(self) -> frozenset:
        """Heads of theory facts: listed `background` facts and interpreted symbols."""
        return frozenset(f[0] for f in self.background) | frozenset(self.interpreted)

    @property
    def interpreted(self) -> dict:
        return {n: s for n, s in self.symbols.items() if s.holds is not None}

    def check_rule(self, rule) -> None:
        """Raises ValueError on a rule that cannot mean what it says: an unknown symbol, a wrong number of
        arguments, a formula where a constant belongs, a metavariable used at two sorts, a conclusion
        variable no premise binds (such a rule never fires), or an interpreted symbol concluded or left
        with a variable that only it binds (its facts are computed, and only over the instance's constants)."""
        where, sorts = f"rule {rule.name}", {}

        def walk(f, want):
            if is_var(f):
                if sorts.setdefault(f, want) != want:
                    raise ValueError(f"{where}: {f[1:]} is used as {sorts[f]} and as {want}")
                return
            if not isinstance(f, tuple):
                return
            if want != PROP:
                raise ValueError(f"{where}: a formula ({f[0]} ...) where a {want} belongs")
            if f[0] in CONNECTIVES:
                args = (PROP,) * CONNECTIVES[f[0]]
            elif f[0] in self.symbols:
                args = self.symbols[f[0]].args
            else:
                raise ValueError(f"{where}: unknown symbol {f[0]!r}")
            if len(f) - 1 != len(args):
                raise ValueError(f"{where}: {f[0]} takes {len(args)} arguments, not {len(f) - 1}")
            for x, s in zip(f[1:], args):
                walk(x, s)

        for f in (*rule.premises, rule.conclusion):
            walk(f, PROP)
        head = lambda f: f[0] if isinstance(f, tuple) else None
        used = set().union(set(), *map(variables, rule.premises))
        bound = set().union(set(), *(variables(p) for p in rule.premises if head(p) not in self.interpreted))
        if free := variables(rule.conclusion) - used:
            raise ValueError(f"{where}: {', '.join(sorted(v[1:] for v in free))} in the conclusion but no premise")
        if head(rule.conclusion) in self.interpreted:
            raise ValueError(f"{where}: concludes the interpreted symbol {head(rule.conclusion)}")
        if self.interpreted and any(is_var(p) for p in rule.premises):
            raise ValueError(f"{where}: a bare formula premise cannot range over computed theory facts")
        for p in rule.premises:
            if head(p) in self.interpreted and (only := variables(p) - bound - variables(rule.conclusion)):
                raise ValueError(f"{where}: {', '.join(sorted(v[1:] for v in only))} bound only by interpreted {p[0]}")

    def check_interpreted(self) -> None:
        """Interpreted symbols range over closed sorts (their facts are enumerable) and have z3 semantics."""
        for name, sym in self.interpreted.items():
            open_ = [a for a in sym.args if a == PROP or not getattr(self.sorts.get(a), "fixed", ())]
            if open_ or sym.z3 is None or sym.unordered:
                raise ValueError(f"interpreted symbol {name} needs closed argument sorts, z3 semantics and "
                                 f"ordered arguments")

    @classmethod
    def of(cls, sorts=(), symbols=(), **kw):
        return cls({s.name: s for s in sorts}, {s.name: s for s in symbols}, **kw)

    @property
    def unordered(self) -> dict:
        """head -> groups of tuple positions (1-based), for terms.unifiers / terms.canonical."""
        return {n: tuple(tuple(i + 1 for i in g) for g in s.unordered) for n, s in self.symbols.items() if s.unordered}

    def formula_vars(self, f) -> set:
        """Metavariables standing for whole formulas in f."""
        if is_var(f):
            return {f}
        if not isinstance(f, tuple):
            return set()
        if f[0] in CONNECTIVES:
            return set().union(*(self.formula_vars(x) for x in f[1:]))
        sym = self.symbols.get(f[0])
        return set().union(*(self.formula_vars(x) for x, s in zip(f[1:], sym.args) if s == PROP)) if sym else set()

    def leaf(self, fresh, rng, depth: Optional[int] = None):
        """What a free formula slot becomes: a random leaf symbol over fresh constants, or (with no
        leaf symbols) a fresh propositional atom (None). Leaf symbols over formulas (modalities) nest up
        to `depth` (default `nesting`), and alternate with plain atoms when the signature has a
        propositional lexicon."""
        depth = self.nesting if depth is None else depth
        leaves = [s for s in self.symbols.values() if s.leaf and (depth > 0 or PROP not in s.args)]
        if not leaves or (self.prop_lexicon and rng.random() < 1 / (len(leaves) + 1)):
            return None
        s = rng.choice(leaves)
        return (s.name, *((self.leaf(fresh, rng, depth - 1) or fresh.var()) if a == PROP else fresh.var()
                          for a in s.args))

    def infer(self, formulas) -> dict:
        """Constant -> sort, by position; propositional atoms get PROP. Raises on a sort clash."""
        out = {}

        def put(c, sort):
            if out.setdefault(c, sort) != sort:
                raise ValueError(f"{c!r} used as {out[c]} and {sort}")

        def walk(f):
            if isinstance(f, str):
                put(f, PROP)
            elif f[0] in CONNECTIVES:
                for x in f[1:]:
                    walk(x)
            else:
                sym = self.symbols[f[0]]
                if len(f) - 1 != len(sym.args):
                    raise ValueError(f"{f[0]} takes {len(sym.args)} arguments: {f!r}")
                for x, s in zip(f[1:], sym.args):
                    walk(x) if s == PROP else put(x, s)
        for f in formulas:
            walk(f)
        return out


def negate(f):
    """Negation without stacking: the negation of not-x is x."""
    return f[1] if isinstance(f, tuple) and f[0] == "not" else ("not", f)


# --- rule DSL -----------------------------------------------------------------------------------------
# Prefix notation: "(eq X V), (ne X Y) |- not (eq Y V)". Capitalised words are metavariables; a
# parenthesised group or a comma-separated item is a head followed by its arguments.

TOKEN = re.compile(r"\s*(\(|\)|,|\|-|[^\s(),]+)")


def _tokens(text):
    pos, out = 0, []
    text = text.strip()
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"cannot read {text[pos:]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


def _seq(toks, i, stop):
    items = []
    while i < len(toks) and toks[i] not in stop:
        if toks[i] == "(":
            x, i = _seq(toks, i + 1, (")",))
            i += 1
        else:
            x, i = (f"?{toks[i]}" if toks[i][0].isupper() else toks[i]), i + 1
        items.append(x)
    if not items:
        raise ValueError(f"empty expression in {' '.join(toks)!r}")
    return (items[0] if len(items) == 1 else tuple(items)), i


def parse(text):
    """One formula in the DSL."""
    toks = _tokens(text)
    f, i = _seq(toks, 0, ())
    return f


def rule(name: str, text: str, kind: Optional[str] = None) -> Rule:
    """A rule from 'premise, premise |- conclusion'."""
    toks = _tokens(text)
    if "|-" not in toks:
        raise ValueError(f"rule {name}: missing |- in {text!r}")
    premises, i = [], 0
    while toks[i] != "|-":
        f, i = _seq(toks, i, (",", "|-"))
        premises.append(f)
        if toks[i] == ",":
            i += 1
    conclusion, _ = _seq(toks, i + 1, ())
    return Rule(name, tuple(premises), conclusion, kind=kind)
