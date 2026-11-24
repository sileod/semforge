"""English realization from a signature: templates per symbol, lexicons per sort, built-in connectives.

Constants get distinct words drawn from their sort's lexicon (fictional by default, so world knowledge
cannot decide anything). A symbol picks one of its templates per sentence, which varies the surface form
without changing the formula. Connectives are realized generically and bracketed whenever an operand
would otherwise read ambiguously; disjunction chains are flattened into a list.

A formula may read better in another, equivalent form depending on its role (premise or hypothesis).
`surface(formula, role)` picks that form before rendering; the templates do not change. Equivalence is
the pack's business (its logic licenses the rewrite), and `semforge.construct.check` verifies it with
the pack's oracle.
"""
from __future__ import annotations

import re

from .signature import CONNECTIVES, PROP, Signature, negate

SLOT = re.compile(r"\{(\d+)(?::(a|s))?\}")


def article(word: str) -> str:
    return ("an " if word[:1].lower() in "aeiou" else "a ") + word


def plural(word: str) -> str:
    return word + ("es" if word.endswith(("s", "x", "z", "ch", "sh")) else "s")


FORMS = {None: str, "a": article, "s": plural}


class Phrase(str):
    """A propositional atom's wording that knows its own negation ("Ann swims" / "Ann does not swim"), so
    "not a" reads as a plain clause instead of "it is false that ...". Lexicons may return plain strings."""
    def __new__(cls, text: str, negated: str = None):
        out = super().__new__(cls, text)
        out.negated = negated
        return out


def verbatim(f, role):
    return f


def classical_surface(f, role):
    """As a hypothesis, "if A, then B" invites a conditional reading that material implication does not
    make, so it is stated as what it claims: "A is false, or B is true"; its negation as "A, and B is
    false". Top level only. Licensed by classical logic: nonclassical packs do not use it."""
    if role != "hypothesis" or isinstance(f, str):
        return f
    if f[0] == "->":
        return ("or", negate(f[1]), f[2])
    if f[0] == "not" and isinstance(f[1], tuple) and f[1][0] == "->":
        return ("and", f[1][1], negate(f[1][2]))
    return f


class Realizer:
    def __init__(self, sig: Signature, surface=verbatim):
        self.sig = sig
        self.surface = surface

    def lexicon(self, formulas, rng) -> dict:
        by_sort = {}
        for c, s in sorted(self.sig.infer(formulas).items()):
            by_sort.setdefault(s, []).append(c)
        lex = {}
        for s, cs in by_sort.items():
            if s != PROP and self.sig.sorts[s].fixed:
                lex.update((c, dict(self.sig.sorts[s].fixed)[c]) for c in cs)
                continue
            source = self.sig.prop_lexicon if s == PROP else self.sig.sorts[s].lexicon
            words = source(rng, len(cs)) if callable(source) else rng.sample(list(source), len(cs))
            lex.update(zip(cs, words))
        return lex

    def _atomic(self, f) -> bool:
        return isinstance(f, str) or f[0] not in CONNECTIVES

    def _negates(self, g, lex) -> bool:
        """The atom has its own negated wording: a symbol's negated template, or a Phrase's negation."""
        if isinstance(g, str):
            return getattr(lex.get(g), "negated", None) is not None
        return bool(self.sig.symbols[g[0]].negated)

    def _plain(self, f, lex) -> bool:
        """Reads as one clause: an atomic sentence, or a negation the symbol words itself ("x is not a P").
        A symbol over formulas ("it must be that ...") is bracketed, since its scope is otherwise unclear."""
        if not isinstance(f, str) and f[0] not in CONNECTIVES and PROP in self.sig.symbols[f[0]].args:
            return False
        return self._atomic(f) or (f[0] == "not" and self._atomic(f[1]) and self._negates(f[1], lex))

    def _fill(self, template, f, lex, rng):
        """Fill a symbol's template; formula arguments are realized recursively (nested modalities)."""
        sorts = self.sig.symbols[f[0]].args

        def slot(m):
            i = int(m.group(1))
            return self._operand(f[1 + i], lex, rng) if sorts[i] == PROP else FORMS[m.group(2)](lex[f[1 + i]])
        return SLOT.sub(slot, template)

    def sentence(self, f, lex, rng) -> str:
        if isinstance(f, str):
            return lex[f]
        op, args = f[0], f[1:]
        if op not in CONNECTIVES:
            return self._fill(rng.choice(self.sig.symbols[op].templates), f, lex, rng)
        if op == "not":
            g = args[0]
            if isinstance(g, str):
                return getattr(lex[g], "negated", None) or f"it is false that {lex[g]}"
            if g[0] == "not" and isinstance(g[1], str):
                return f"it is not false that {lex[g[1]]}"
            if g[0] not in CONNECTIVES and self.sig.symbols[g[0]].negated:
                return self._fill(rng.choice(self.sig.symbols[g[0]].negated), g, lex, rng)
            # "both ... and" and "neither ... nor" mark their own scope: no brackets needed
            if g[0] == "or" and len(self._disjuncts(g)) == 2:
                a, b = (self._operand(x, lex, rng) for x in self._disjuncts(g))
                return f"neither {a} nor {b}"
            if g[0] == "and":
                return f"it is not the case that {self.sentence(g, lex, rng)}"
            return f"it is not the case that ({self.sentence(g, lex, rng)})"
        if op == "or":
            ds = self._disjuncts(f)
            parts = [self._operand(x, lex, rng, flat=len(ds) > 2) for x in ds]  # a list needs no brackets
            return f"either {parts[0]} or {parts[1]}" if len(parts) == 2 else \
                "at least one of these holds: " + "; ".join(parts)
        a, b = (self._operand(x, lex, rng, flat=op == "->" and (not isinstance(x, tuple) or x[0] != "->"))
                for x in args)
        return f"both {a} and {b}" if op == "and" else f"if {a}, then {b}"

    def _operand(self, x, lex, rng, flat=False):
        s = self.sentence(x, lex, rng)
        return s if self._plain(x, lex) or flat else f"({s})"

    def _disjuncts(self, f):
        return [y for x in f[1:] for y in (self._disjuncts(x) if isinstance(x, tuple) and x[0] == "or" else [x])]

    def render(self, premises, hypothesis, rng):
        lex = self.lexicon([*premises, hypothesis], rng)
        return ("\n".join([*self.sig.theory, *(self.statement(p, "premise", lex, rng) for p in premises)]),
                self.statement(hypothesis, "hypothesis", lex, rng))

    def statement(self, f, role, lex, rng) -> str:
        t = self.sentence(self.surface(f, role), lex, rng)
        return t[0].upper() + t[1:] + "."
