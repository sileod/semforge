"""Inference calculi as data, and exact structural analysis of what a conclusion requires.

A calculus is a set of named rule schemas `premises ⊢ conclusion` over formulas with metavariables.
Analysis runs the provenance chase (semforge.core.provenance) with every *rule application* (step)
counted as a premise of what it derives. A minimal support is then a minimal proof: a set of steps
and given premises from which the conclusion follows, minimal under inclusion. Two proofs using the
same premises but different steps are distinct; so are proofs applying a rule a different number of
times. From one fixpoint:

  proofs              minimal proofs (sets of steps and premises)
  premise_supports    minimal premise sets with some proof (the notion of minimal justification
                      used for semantic oracles; several proofs can share one)
  required_rules      multiset of rule applications in every proof (multiset intersection)
  required_premises   premises in every proof
  depth, size         shortest proof height, fewest steps

All are exact relative to the calculus restricted to a finite universe of formulas (`Universe`), which
bounds rules that build larger formulas (hypothetical syllogism, Barbara, conjunction introduction), and are
None/flagged when the chase was capped. They say nothing about other calculi or about cognition.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable, Optional

from ..core.provenance import provenance_chase
from .terms import canonical, is_ground, is_var, matches, subformulas, subst, unifiers, variables


@dataclass(frozen=True)
class Rule:
    name: str
    premises: tuple
    conclusion: object
    kind: Optional[str] = None  # reasoning kind certificates count; schemas that differ only in argument
                                # order or position (symmetric constraints, domain slots) share one

    @property
    def family(self) -> str:
        return self.kind or self.name


class Universe:
    """Formulas a derivation may pass through: subformulas of the inputs, their negations, and the
    pack's closed connectives (e.g. implication, or "all A are B") applied to those. A membership test,
    never enumerated."""

    def __init__(self, formulas: Iterable, negate: Callable, closed: tuple = ("->",), schematic=frozenset(),
                 stacked: bool = True):
        """`schematic`: atoms standing for arbitrary formulas (most general instances); their negations are
        left out, since negating an instance of one may cancel a negation instead of adding one. `stacked`:
        also admit the double negation of each negated subformula, which a rule such as modus tollens on a
        negated antecedent concludes (from "if not A then B" and "not B", "not not A")."""
        base = {f for x in formulas for f in subformulas(x)}
        self.literals = base | {negate(f) for f in base if f not in schematic}
        if stacked:
            self.literals |= {("not", f) for f in base if isinstance(f, tuple) and f[0] == "not" and f not in schematic}
        self.closed, self.negate = frozenset(closed), negate

    def _built(self, f) -> bool:
        return isinstance(f, tuple) and f[0] in self.closed and all(x in self.literals for x in f[1:])

    def __contains__(self, f) -> bool:
        return f in self.literals or self._built(f) or self._built(self.negate(f))


class Background:
    """Theory facts: in every closure, stated by no instance. Listed facts (`Signature.background`) are
    stored; the true instances of interpreted symbols (`Symbol.holds`) are computed when a rule asks for
    them. Either kind is used the same way: membership, and the facts matching a premise pattern."""

    def __init__(self, listed=(), signature=None):
        self.listed = tuple(dict.fromkeys(listed))  # in signature order: construction draws from it
        self._listed = frozenset(self.listed)
        self.interpreted = {} if signature is None else {
            name: (sym.holds, tuple(tuple(c for c, _ in signature.sorts[a].fixed) for a in sym.args))
            for name, sym in signature.interpreted.items()}

    def __contains__(self, f) -> bool:
        if f in self._listed:
            return True
        spec = self.interpreted.get(_head(f))
        return spec is not None and len(f) == len(spec[1]) + 1 and \
            all(a in dom for a, dom in zip(f[1:], spec[1])) and spec[0](*f[1:])

    def __bool__(self) -> bool:
        return bool(self.listed or self.interpreted)

    def interprets(self, p) -> bool:
        return _head(p) in self.interpreted

    def solutions(self, p, s):
        """(fact, s2) for each true instance of the interpreted pattern p extending s: the unbound
        arguments range over their closed sorts, and a bound one outside its sort makes no fact."""
        holds, domains = self.interpreted[p[0]]
        q = subst(p, s)
        if any(not is_var(a) and a not in dom for a, dom in zip(q[1:], domains)):
            return
        free = [i for i, a in enumerate(q[1:], 1) if is_var(a)]
        for values in product(*(domains[i - 1] for i in free)):
            f = list(q)
            for i, v in zip(free, values):
                f[i] = v
            f = tuple(f)
            if holds(*f[1:]):
                for s2 in matches(q, f, s):  # rejects a repeated variable bound two ways
                    yield f, s2

    def facts(self, constants=None) -> frozenset:
        """Listed facts and the true interpreted facts over `constants` (all of a sort's, by default):
        what a certificate states as theory, enough for a checker to re-derive everything."""
        out = set(self.listed)
        for name, (holds, domains) in self.interpreted.items():
            doms = [d if constants is None else [c for c in d if c in constants] for d in domains]
            out |= {(name, *args) for args in product(*doms) if holds(*args)}
        return frozenset(out)


class Calculus:
    def __init__(self, rules: Iterable[Rule], signature=None):
        """`signature` (construct.signature.Signature), if given, supplies unordered argument groups
        (matching and equality modulo symmetry), typed leaves for construction and theory facts, and
        every rule is checked against it."""
        self.rules = {r.name: r for r in rules}
        self.signature = signature
        if signature is not None:
            signature.check_interpreted()
            for r in self.rules.values():
                signature.check_rule(r)
        self.unordered = signature.unordered if signature is not None else {}
        self.background = tuple(self.canonical(f) for f in signature.background) if signature is not None else ()
        self.background_set = Background(self.background, signature)  # membership covers interpreted facts
        if self.background_set:
            heads = signature.background_heads
            assert all(any(not (isinstance(p, tuple) and p[0] in heads) for p in r.premises) for r in self.rules.values()), \
                "a rule with only background premises would derive from the theory alone"

    def canonical(self, f):
        return canonical(f, self.unordered) if self.unordered else f

    def kinds(self, names) -> Counter:
        """Schema names (a plan, or a Counter of them) as a multiset of reasoning kinds."""
        names = names.elements() if isinstance(names, Counter) else names
        return Counter(self.rules[n].family for n in names)

    @property
    def families(self) -> list:
        return sorted({r.family for r in self.rules.values()})

    def instances(self, rule: Rule, closure, universe: Universe):
        """Ground applications of `rule` over the closure: (conclusion, parents). Structured premises
        are matched first; once a premise is ground it is a set lookup."""
        prems = sorted(enumerate(rule.premises), key=lambda ip: (self.background_set.interprets(ip[1]), is_var(ip[1])))
        closure = set(closure)

        def join(k, s, parents):
            if k == len(prems):
                c = self.canonical(subst(rule.conclusion, s))
                if is_ground(c) and c in universe:
                    yield c, tuple(parents[i] for i in range(len(prems)))
                return
            i, p = prems[k]
            if self.background_set.interprets(p):
                for f, s2 in self.background_set.solutions(p, s):
                    yield from join(k + 1, s2, {**parents, i: f})
                return
            p = subst(p, s)
            if is_ground(p):
                p = self.canonical(p)
                if p in closure:
                    yield from join(k + 1, s, {**parents, i: p})
                return
            for f in closure:
                for s2 in unifiers(p, f, s, self.unordered):
                    yield from join(k + 1, s2, {**parents, i: f})

        yield from join(0, {}, {})


ANY = object()  # index bucket of facts a bare metavariable premise may match


def _head(f):
    return f[0] if isinstance(f, tuple) else None


class Closure:
    """Everything the calculus derives from some premises inside a universe, as a hypergraph.

    A semi-naive chase: in round r, only joins with at least one premise first derived in round r-1
    are tried, with that premise matched first and the others looked up by head symbol (or by value,
    once ground). `round[f]` is the height of f's shallowest derivation; `edges[f]` lists every rule
    application deriving f as (rule, parents). Matching, the expensive part, happens once here;
    certificates are read off the hypergraph.

    Introduction rules (A, B ⊢ A ∧ B), whose conclusion must already be a formula of the universe and
    fixes every premise, run backward instead: their ground instances are read off the universe once,
    and each round only checks which have all premises derived. Forward, they would pair every new
    fact with every fact and discard nearly all of the results.
    """

    def __init__(self, calculus: "Calculus", premises: Iterable, universe: "Universe"):
        self.calculus, self.universe = calculus, universe
        self.round, self.edges = {}, {}
        self._by_head, self._by_arg = {}, {}
        self._schedule, backward = [], []
        for r in calculus.rules.values():
            if self._backward(r, universe):
                backward.append((r, self._instances_from_universe(r, universe)))
            else:
                # interpreted premises last: they are solved over a sort, cheapest once their arguments are bound
                self._schedule.append((r, sorted(range(len(r.premises)),
                                                 key=lambda i: (calculus.background_set.interprets(r.premises[i]),
                                                                is_var(r.premises[i])))))
        delta = []
        for f in [*premises, *calculus.background]:
            if f not in self.round:
                self.round[f] = 0
                self._index(f)
                delta.append(f)
        # Background facts alone derive nothing (every rule has a non-background premise: Calculus
        # checks it), so they are known from the start but never the new fact a join starts from.
        delta = [f for f in delta if f not in calculus.background_set]
        r = 0
        while delta:
            r += 1
            new = {}
            for rule, order in self._schedule:
                for j in range(len(rule.premises)):
                    for f in delta:
                        for s in matches(rule.premises[j], f, {}, calculus.unordered):
                            for head, parents in self._join(rule, order, j, s, {j: f}):
                                if head not in self.round and head not in new:
                                    new[head] = []
                                step = (rule, parents)
                                bucket = self.edges.setdefault(head, [])
                                if step not in bucket:
                                    bucket.append(step)
                                    for x in parents:  # a computed theory fact, known from the start once used
                                        self.round.setdefault(x, 0)
            fresh = set(delta)
            for rule, insts in backward:
                for head, parents in insts:
                    if any(x in fresh for x in parents) and all(x in self.round or x in calculus.background_set
                                                                for x in parents):
                        if head not in self.round and head not in new:
                            new[head] = []
                        bucket = self.edges.setdefault(head, [])
                        if (rule, parents) not in bucket:
                            bucket.append((rule, parents))
                            for x in parents:
                                self.round.setdefault(x, 0)
            for f in new:
                self.round[f] = r
                self._index(f)
            delta = list(new)

    @staticmethod
    def _backward(rule, universe) -> bool:
        c = rule.conclusion
        # then membership in the universe means being one of its literals (see Universe.__contains__)
        return (isinstance(c, tuple) and c[0] not in universe.closed and c[0] != "not" and "not" not in universe.closed
                and variables(c) >= set().union(*map(variables, rule.premises)))

    def _instances_from_universe(self, rule, universe):
        out, unordered = [], self.calculus.unordered
        for lit in universe.literals:
            if _head(lit) != rule.conclusion[0]:
                continue
            head = self.calculus.canonical(lit)
            for s in matches(rule.conclusion, head, {}, unordered):
                parents = tuple(self.calculus.canonical(subst(p, s)) for p in rule.premises)
                if (head, parents) not in out:
                    out.append((head, parents))
        return out

    def _index(self, f):
        """Facts by head, and by (head, constant argument): a fact matching a pattern has each of the
        pattern's constant arguments among its own (at any position, so unordered groups are covered)."""
        h = _head(f)
        self._by_head.setdefault(h, []).append(f)
        if isinstance(f, tuple):
            for a in set(f[1:]):
                if isinstance(a, str):
                    self._by_arg.setdefault((h, a), []).append(f)

    def _candidates(self, p):
        if is_var(p):
            return list(self.round)
        h = _head(p)
        keys = [(h, a) for a in p[1:] if isinstance(a, str) and not is_var(a)] if isinstance(p, tuple) else []
        if keys:
            return min((self._by_arg.get(k, ()) for k in keys), key=len)
        return self._by_head.get(h, ())

    def _join(self, rule, order, fixed, s, bound):
        rest = [i for i in order if i != fixed]
        unordered = self.calculus.unordered

        def go(k, s, bound):
            if k == len(rest):
                c = self.calculus.canonical(subst(rule.conclusion, s))
                if is_ground(c) and c in self.universe:
                    yield c, tuple(bound[i] for i in range(len(rule.premises)))
                return
            i = rest[k]
            if self.calculus.background_set.interprets(rule.premises[i]):  # computed, never stored
                for f, s2 in self.calculus.background_set.solutions(rule.premises[i], s):
                    yield from go(k + 1, s2, {**bound, i: f})
                return
            p = subst(rule.premises[i], s)
            if is_ground(p):
                p = self.calculus.canonical(p)
                if p in self.round:
                    yield from go(k + 1, s, {**bound, i: p})
                return
            for f in self._candidates(p):
                for s2 in matches(p, f, s, unordered):
                    yield from go(k + 1, s2, {**bound, i: f})
        yield from go(0, s, bound)

    def ancestors(self, goal) -> set:
        """Facts some derivation of goal passes through."""
        seen, stack = set(), [goal]
        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            for _, parents in self.edges.get(f, ()):
                stack.extend(parents)
        return seen


def step_id(rule: Rule, head, parents) -> tuple:
    return ("step", rule.family, head, parents)


def _same_step(rule: Rule, head, parents) -> tuple:
    return rule.family, head, frozenset(parents)


def rule_of(x) -> Optional[str]:
    """Rule name of a step id, None for a premise."""
    return x[1] if isinstance(x, tuple) and len(x) == 4 and x[0] == "step" else None


def rule_counts(proof) -> Counter:
    return Counter(r for r in map(rule_of, proof) if r is not None)


class Analysis:
    """Certificates for one goal, layered by cost.

    Cheap and exact, from the chase hypergraph alone:
      derivable, depth (round of first derivation = height of the shallowest proof),
      mandatory_steps / mandatory_premises (in every proof: removing it alone makes the goal
      underivable; only a witness proof's members need testing), topology.
    On demand (minimal-proof enumeration over the goal's ancestors, capped at `cap`):
      proofs, n_proofs, required_rules (multiset intersection), size, premise_supports, exact.
    `requires(want)` answers "are these rule applications mandatory?" from the mandatory steps.
    """

    def __init__(self, calculus, closure, premises, goal, cap=256):
        self.calculus, self.closure, self.goal, self.cap = calculus, closure, goal, cap
        self.premises = [p for p in premises]
        self.derivable = closure is not None and goal in closure.round
        self._enum = None
        if self.derivable:
            self.relevant = closure.ancestors(goal)
            self.edges = [(rule, head, parents) for head in self.relevant for rule, parents in closure.edges.get(head, ())]
            # background facts (theory) start proofs like premises, but are never reported as premises
            premises = set(self.premises)
            self.given = {p for p in self.relevant if p in premises or p in calculus.background_set}

    # --- cheap ------------------------------------------------------------------------------------------

    @property
    def depth(self) -> Optional[int]:
        return self.closure.round[self.goal] if self.derivable else None

    def _reaches(self, skip_edge=None, skip_premise=None) -> bool:
        """Is the goal derivable from the given premises over the relevant edges, without one of them?"""
        have = {p for p in self.given if p != skip_premise}
        waiting, users = {}, {}
        frontier = list(have)
        skip = None if skip_edge is None else _same_step(*self.edges[skip_edge])
        for k, (rule, head, parents) in enumerate(self.edges):
            if skip is not None and _same_step(rule, head, parents) == skip:  # also its permuted copies
                continue
            need = {x for x in parents if x not in have}
            if not need:
                if head not in have:
                    have.add(head)
                    frontier.append(head)
            else:
                waiting[k] = len(need)
                for x in need:
                    users.setdefault(x, []).append(k)
        while frontier:
            f = frontier.pop()
            for k in users.get(f, ()):
                waiting[k] -= 1
                if waiting[k] == 0:
                    head = self.edges[k][1]
                    if head not in have:
                        have.add(head)
                        frontier.append(head)
        return self.goal in have

    def proof_without(self, step) -> Optional[list]:
        """One proof of the goal that never makes `step` (a step id; permuted copies included), as a list
        of step ids, or None if every proof makes it. The witness of a failed necessity claim."""
        skip = (step[1], step[2], frozenset(step[3]))
        have, producer = set(self.given), {}
        waiting, users, frontier = {}, {}, list(have)
        for k, (rule, head, parents) in enumerate(self.edges):
            if _same_step(rule, head, parents) == skip:
                continue
            need = {x for x in parents if x not in have}
            if not need:
                if head not in have:
                    have.add(head)
                    producer[head] = k
                    frontier.append(head)
            else:
                waiting[k] = len(need)
                for x in need:
                    users.setdefault(x, []).append(k)
        while frontier:
            f = frontier.pop()
            for k in users.get(f, ()):
                waiting[k] -= 1
                if waiting[k] == 0:
                    head = self.edges[k][1]
                    if head not in have:
                        have.add(head)
                        producer[head] = k
                        frontier.append(head)
        if self.goal not in have:
            return None
        out, stack, seen = [], [self.goal], set()
        while stack:
            f = stack.pop()
            if f in seen or f not in producer:
                continue
            seen.add(f)
            rule, head, parents = self.edges[producer[f]]
            out.append(step_id(rule, head, parents))
            stack.extend(parents)
        return out

    def _witness(self):
        """Edge indices and premises of one proof (each fact through an edge of earliest parents)."""
        rnd, best = self.closure.round, {}
        for k, (rule, head, parents) in enumerate(self.edges):
            key = max((rnd[x] for x in parents), default=-1)
            if key < rnd[head] and (head not in best or key < best[head][0]):
                best[head] = (key, k)
        edges, prem, stack, seen = set(), set(), [self.goal], set()
        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            if rnd[f] == 0:
                prem.add(f)
                continue
            k = best[f][1]
            edges.add(k)
            stack.extend(self.edges[k][2])
        return edges, prem

    def _dependencies(self) -> dict:
        """For each relevant fact, the steps and premises without which it is not derivable, as a bitset
        over `tokens` (a step with its permuted copies, or a non-background premise), all computed at
        once. A derivation through an edge depends on the edge and on what its parents depend on;
        alternatives (other edges, being given) intersect. Per bit this is the complement of
        reachability with that token removed, a least fixed point, so the bitsets are the greatest fixed
        point: start from all bits (also right for facts on cycles) and descend. Returns (tokens, deps)."""
        bg = self.calculus.background_set
        tokens = {}
        for rule, head, parents in self.edges:
            tokens.setdefault(_same_step(rule, head, parents), 1 << len(tokens))
        for p in self.given:
            if p not in bg:
                tokens.setdefault(("premise", p), 1 << len(tokens))
        top = (1 << len(tokens)) - 1
        incoming = {}
        for rule, head, parents in self.edges:
            incoming.setdefault(head, []).append((tokens[_same_step(rule, head, parents)], parents))
        deps = dict.fromkeys(self.relevant, top)
        order = sorted(self.relevant, key=lambda f: (self.closure.round[f], repr(f)))
        changed = True
        while changed:
            changed = False
            for f in order:
                value = tokens.get(("premise", f), 0) if f in self.given else top
                for own, parents in incoming.get(f, ()):
                    for p in parents:
                        own |= deps[p]
                    value &= own
                if value != deps[f]:
                    deps[f], changed = value, True
        return tokens, deps

    @property
    def mandatory_steps(self) -> frozenset:
        """Rule applications every proof makes (exact, no enumeration)."""
        if not self.derivable:
            return frozenset()
        if not hasattr(self, "_mandatory"):
            edges, prem = self._witness()  # every mandatory step and premise is in any one proof
            tokens, deps = self._dependencies()
            need = deps[self.goal]
            self._mandatory = frozenset(step_id(*self.edges[k]) for k in edges
                                        if need & tokens[_same_step(*self.edges[k])])
            self._mandatory_premises = frozenset(p for p in prem if need & tokens.get(("premise", p), 0))
        return self._mandatory

    @property
    def mandatory_premises(self) -> frozenset:
        self.mandatory_steps
        return self._mandatory_premises if self.derivable else frozenset()

    required_premises = mandatory_premises

    @property
    def mandatory_kinds(self) -> Counter:
        """Kinds of the mandatory steps: a lower bound on `required_rules`, usually equal to it."""
        return rule_counts(self.mandatory_steps)

    @property
    def topology(self) -> dict:
        """Proof shape every proof shares, read off the mandatory steps: `spine`, the longest chain of
        mandatory steps each feeding the next (certified sequential depth); `joins`, mandatory steps
        combining two or more mandatory sub-derivations (0 for a chain)."""
        steps = self.mandatory_steps
        made = {x[2]: x for x in steps}  # head -> mandatory step
        memo = {}

        def chain(x):
            if x not in memo:
                memo[x] = 1 + max((chain(made[p]) for p in x[3] if p in made), default=0)
            return memo[x]
        return {"spine": max(map(chain, steps), default=0),
                "joins": sum(sum(p in made for p in x[3]) >= 2 for x in steps)}

    @property
    def shared(self) -> int:
        """Mandatory steps whose conclusion feeds two or more mandatory steps: a lemma reused, which
        the tree-shaped topology cannot tell from the same lemma derived twice."""
        made = {x[2] for x in self.mandatory_steps}
        uses = Counter(p for x in self.mandatory_steps for p in set(x[3]) if p in made)
        return sum(n >= 2 for n in uses.values())

    @property
    def profile(self) -> Optional[dict]:
        """The goal's proof structure as one record (`semforge.audit.PROFILE` fields), the same for every
        pack and audited source; None when the goal is not derivable.
          min_depth  height of the shallowest derivation
          min_size   fewest rule applications in any proof
          mandatory  rule applications every proof makes; spine and joins: their topology; shared:
                     those whose conclusion two of them use
          proofs     minimal proofs (inclusion-minimal sets of steps and premises)
          supports   minimal premise sets among them: one support with several proofs is the same
                     premises combined in different orders (re-routing), several is over-determination
          exact      False when proof enumeration hit its cap: min_size is then an upper bound and
                     proofs a lower bound; min_depth and mandatory are exact either way"""
        if not self.derivable:
            return None
        return {"min_depth": self.depth, "min_size": self.size, "mandatory": len(self.mandatory_steps),
                **self.topology, "shared": self.shared, "proofs": self.n_proofs,
                "supports": len(self.premise_supports), "exact": self.exact}

    def requires(self, want: Counter, enumerate: bool = False) -> Optional[bool]:
        """Are the rule applications in `want` (a multiset of kinds) made by mandatory steps, i.e. the
        same applications in every proof? Exact and cheap. With `enumerate`, a weaker property is
        decided when that fails: every proof makes that many applications of each kind, possibly
        different ones (None if enumeration hit the cap)."""
        if not self.derivable:
            return False
        if not want - self.mandatory_kinds:
            return True
        if not enumerate:
            return False
        if not self.exact:
            return None
        return not want - self.required_rules

    # --- on demand ----------------------------------------------------------------------------------

    def _enumerate(self):
        if self._enum is None:
            by_rule, same = {}, set()
            for rule, head, parents in self.edges:  # one application per step modulo premise order, so a
                key = _same_step(rule, head, parents)  # rule matching its premises two ways is one proof
                if key not in same:
                    same.add(key)
                    by_rule.setdefault(rule.name, []).append((head, parents))
            prov = provenance_chase(sorted(self.given, key=repr), list(self.calculus.rules.values()),
                                    lambda r, have: [(h, ps) for h, ps in by_rule.get(r.name, ())
                                                     if all(x in have for x in ps)],
                                    lambda x: None if isinstance(x, Rule) else x, self.cap, step_id=step_id)
            proofs = prov.minimal_supports(self.goal, allow_incomplete=True) if self.goal in prov.supports else []
            self._enum = (prov.is_exact(self.goal), proofs)
        return self._enum

    @property
    def exact(self) -> bool:
        return self._enumerate()[0] if self.derivable else True

    @property
    def proofs(self) -> list:
        """Minimal proofs: frozensets of premises and step ids."""
        return self._enumerate()[1] if self.derivable else []

    @property
    def n_proofs(self) -> int:
        return len(self.proofs)

    @property
    def required_rules(self) -> Counter:
        """Rule applications every proof makes, as a multiset (intersection over proofs)."""
        if not self.proofs:
            return Counter()
        out = rule_counts(self.proofs[0])
        for p in self.proofs[1:]:
            out &= rule_counts(p)
        return out

    @property
    def size(self) -> Optional[int]:
        return min(sum(rule_counts(p).values()) for p in self.proofs) if self.proofs else None

    @property
    def premise_supports(self) -> list:
        bg = self.calculus.background_set
        prem = {frozenset(x for x in p if rule_of(x) is None and x not in bg) for p in self.proofs}
        return [s for s in prem if not any(t < s for t in prem)]


def analyze(calculus: Calculus, premises: Iterable, goal, universe: Universe, cap: int = 256) -> Analysis:
    premises, goal = [calculus.canonical(p) for p in premises], calculus.canonical(goal)
    return Analysis(calculus, Closure(calculus, premises, universe), premises, goal, cap)
