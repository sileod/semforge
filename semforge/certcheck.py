"""Independent checker for semforge proof certificates: trust the claims without trusting the generator.

Shares no code with `semforge.construct` (own matcher, universe membership and chase), so a bug in the
generator's semi-naive closure, unification or ablation cannot make a false claim pass. A certificate
(JSON, from `semforge.construct.synth.export_certificate`) holds the rules as schemas, the premises, the
goal, the finite universe derivations may pass through, ground background facts of the theory (usable
by every derivation, never premises), one proof, and the steps claimed mandatory. The checker verifies:

  1. the proof: each step instantiates a rule of the stated kind, uses only premises and earlier
     conclusions, stays in the universe, and the proof reaches the goal;
  2. the mandatory steps, soundly and completely: a step of the proof is mandatory iff, with every
     application of it removed (same kind, conclusion and premise set), a fresh saturation of the
     premises inside the universe no longer reaches the goal; the claimed list must be exactly those.
     (A mandatory step is made by every proof, so by the certificate's proof: no other can exist.)
  3. the mandatory premises: likewise, without each premise the proof uses.

A nogood certificate (`semforge.construct.atlas.nogood_certificates`) instead holds a most general
instance of a shortcut-atlas fragment: the planned derivation (`planned`), one of its steps (`avoids`)
and a proof of the same goal that never makes that step. Both derivations are checked, so the planned
step is shown not mandatory on that instance; the extension to every instance is the atlas theorem
(the paper, Appendix A), not re-checked here.

Claims are relative to the rules and universe in the certificate. A labelled certificate states its
`hypothesis`. An entailment (contradiction) label rests on a proof whose goal is the hypothesis (its
negation) and the soundness of the rules (`check.prove_rule`). A neutral certificate's proof is
counterfactual: it names the `withheld` premises it adds, and the remaining premises must not derive the
hypothesis. The label itself rests on two countermodels (`countermodels`, from `oracle.countermodel`):
finite structures in which the premises hold and the hypothesis is false (`refutes: "entailment"`), or
true (`refutes: "contradiction"`). The checker evaluates the certificate's own label queries over each
model's tables (`smt2`: premises and the negated hypothesis; `smt2_contradiction`: premises and the
hypothesis), after checking the tables against the declared signature, so no solver is trusted. An
entailment or contradiction certificate carries the one countermodel refuting the opposite label, which
shows its premises consistent (an inconsistent premise set would prove any label). What is
trusted is the translation of the premises into those queries, which is the pack's semantics; any
solver can re-check it.

Formulas are JSON: an atom is a string, an application a list [head, *args]; "?X" is a metavariable.
"""
from __future__ import annotations

import itertools
import json
import math
import re
import sys

LABELS = ("entailment", "contradiction", "neutral")


def _t(x):
    return tuple(_t(a) for a in x) if isinstance(x, list) else x


def _var(x):
    return isinstance(x, str) and x.startswith("?")


class Checker:
    def __init__(self, cert: dict):
        self.unordered = {h: [tuple(g) for g in gs] for h, gs in cert.get("unordered", {}).items()}
        self.rules = [(r["kind"], [_t(p) for p in r["premises"]], _t(r["conclusion"])) for r in cert["rules"]]
        self.closed = set(cert["universe"]["closed"])
        self.literals = {self.canon(_t(f)) for f in cert["universe"]["literals"]}
        self.premises = [self.canon(_t(p)) for p in cert["premises"]]
        self.goal = self.canon(_t(cert["goal"]))
        self.background = [self.canon(_t(f)) for f in cert.get("background", ())]  # theory facts, never premises

    # --- formulas -----------------------------------------------------------------------------------

    def canon(self, x):
        if not isinstance(x, tuple):
            return x
        y = [x[0], *(self.canon(a) for a in x[1:])]
        for g in self.unordered.get(x[0], ()):
            for i, v in zip(g, sorted((y[i] for i in g), key=repr)):
                y[i] = v
        return tuple(y)

    @staticmethod
    def negate(f):
        return f[1] if isinstance(f, tuple) and f[0] == "not" else ("not", f)

    def in_universe(self, f) -> bool:
        def built(g):
            return isinstance(g, tuple) and g[0] in self.closed and all(a in self.literals for a in g[1:])
        return f in self.literals or built(f) or built(self.negate(f))

    def subst(self, x, s):
        if _var(x):
            return s.get(x, x)
        return tuple(self.subst(a, s) if i else a for i, a in enumerate(x)) if isinstance(x, tuple) else x

    def match(self, p, f, s):
        """All extensions of s with subst(p) equal to f modulo unordered argument groups."""
        if _var(p):
            if p in s:
                if s[p] == f:
                    yield s
            else:
                yield {**s, p: f}
            return
        if not isinstance(p, tuple):
            if p == f:
                yield s
            return
        if not isinstance(f, tuple) or p[0] != f[0] or len(p) != len(f):
            return
        groups = self.unordered.get(p[0], ())
        for perms in itertools.product(*(itertools.permutations(g) for g in groups)):
            q = list(p)
            for g, perm in zip(groups, perms):
                for i, j in zip(g, perm):
                    q[i] = p[j]
            yield from self._args(q[1:], f[1:], s)

    def _args(self, ps, fs, s):
        if not ps:
            yield s
            return
        for s2 in self.match(ps[0], fs[0], s):
            yield from self._args(ps[1:], fs[1:], s2)

    def _ground(self, x) -> bool:
        return not _var(x) and (not isinstance(x, tuple) or all(self._ground(a) for a in x[1:]))

    # --- claims -------------------------------------------------------------------------------------

    def instance_of(self, kind, parents, head) -> bool:
        for k, ps, c in self.rules:
            if k != kind or len(ps) != len(parents):
                continue
            for s in self._args(list(ps), list(parents), {}):
                if self.canon(self.subst(c, s)) == head:
                    return True
        return False

    def saturate(self, premises, skip=None) -> set:
        """Everything the rules derive from premises inside the universe (semi-naive), never applying
        the step `skip` = (kind, head, frozenset(parents))."""
        have = set(premises) | set(self.background)
        delta = set(have)
        while delta:
            new = set()
            for kind, ps, c in self.rules:
                for j in range(len(ps)):
                    for f in delta:
                        for s in self.match(ps[j], f, {}):
                            for s2, parents in self._join(ps, j, f, s, have):
                                h = self.canon(self.subst(c, s2))
                                if h in have or h in new or not self._ground(h) or not self.in_universe(h):
                                    continue
                                if skip is not None and (kind, h, frozenset(parents)) == skip:
                                    continue
                                new.add(h)
            have |= new
            delta = new
        return have

    def _join(self, ps, fixed, f, s, have):
        def go(i, s, bound):
            if i == len(ps):
                yield s, bound
                return
            if i == fixed:
                yield from go(i + 1, s, bound + [f])
                return
            for g in have:
                for s2 in self.match(ps[i], g, s):
                    yield from go(i + 1, s2, bound + [g])
        yield from go(0, s, [])

    def proof(self, proof: list, name: str = "proof") -> tuple:
        """Check a proof step by step; returns (errors, its steps as (kind, head, frozenset(parents)))."""
        errors, known, steps = [], set(self.premises) | set(self.background), []
        for n, st in enumerate(proof):
            parents, head = [self.canon(_t(p)) for p in st["parents"]], self.canon(_t(st["head"]))
            steps.append((st["kind"], head, frozenset(parents)))
            if not all(p in known for p in parents):
                errors.append(f"{name} step {n}: uses a formula neither given nor derived earlier")
            if not self.instance_of(st["kind"], parents, head):
                errors.append(f"{name} step {n}: not an instance of a {st['kind']} rule")
            if not self.in_universe(head):
                errors.append(f"{name} step {n}: conclusion outside the universe")
            if steps[-1] in steps[:-1]:
                errors.append(f"{name} step {n}: repeats step {steps.index(steps[-1])}")
            known.add(head)
        if self.goal not in known:
            errors.append(f"the {name} does not reach the goal")
        return errors, steps

    def check(self, cert: dict) -> list:
        """Failed claims, as messages; empty when the certificate checks."""
        errors, steps = self.proof(cert["proof"])
        if "avoids" in cert:  # a nogood certificate: the planned derivation has a step this proof avoids
            planned_errors, planned = self.proof(cert["planned"], "planned derivation")
            a = cert["avoids"]
            avoided = (a["kind"], self.canon(_t(a["head"])), frozenset(self.canon(_t(p)) for p in a["parents"]))
            errors += planned_errors
            if avoided not in planned:
                errors.append("the avoided step is not a step of the planned derivation")
            if avoided in steps:
                errors.append("the bypassing proof makes the step it claims to avoid")
        if "label" in cert:
            errors += self.label(cert)
        if "mandatory_steps" not in cert:
            return errors
        # A mandatory step is made by every proof, so by this one: testing each of its steps finds them
        # all, and the claimed list must be exactly that set (sound and complete). Likewise premises.
        claimed = set(cert["mandatory_steps"])
        for n in sorted(claimed - set(range(len(steps))), key=repr):
            errors.append(f"claimed mandatory step {n!r} is not a step of the proof")
        for n, st in enumerate(steps):
            needed = self.goal not in self.saturate(self.premises, skip=st)
            if n in claimed and not needed:
                errors.append(f"step {n} ({st[0]}) is claimed mandatory but the goal is derivable without it")
            if needed and n not in claimed:
                errors.append(f"step {n} ({st[0]}) is mandatory but not claimed")
        claimed = {self.canon(_t(p)) for p in cert.get("mandatory_premises", ())}
        used = {p for _, _, ps in steps for p in ps if p in set(self.premises)}
        if self.goal in set(self.premises):  # a zero-step proof uses the goal itself
            used.add(self.goal)
        for p in sorted(used | claimed, key=repr):
            needed = self.goal not in self.saturate([q for q in self.premises if q != p])
            if p in claimed and not needed:
                errors.append(f"premise {p!r} is claimed mandatory but the goal is derivable without it")
            if needed and p not in claimed:
                errors.append(f"premise {p!r} is mandatory but not claimed")
        if "planned" in cert:  # requested exposure: the intended derivation, and its bypassed applications
            errors += self.exposure(cert)
        return errors

    def exposure(self, cert: dict) -> list:
        """An exposure claim splits the planned derivation's applications into mandatory (every proof
        makes them; checked as for the proof above) and bypassed, each with a witness proof of the goal
        that never makes the claimed application. Both directions are verified, so the claimed number of
        bypassed applications is exact."""
        errors, planned = self.proof(cert["planned"], "planned derivation")
        listed = [b.get("step") for b in cert.get("bypassed", ())]
        if len(set(listed)) != len(listed) or any(not isinstance(i, int) or not 0 <= i < len(planned) for i in listed):
            return errors + ["bypassed steps must name distinct steps of the planned derivation"]
        bypassed = set(listed)
        for b in cert.get("bypassed", ()):
            berrors, bsteps = self.proof(b["proof"], "bypass proof")
            errors += berrors
            if planned[b["step"]] in bsteps:
                errors.append(f"the bypass proof for planned step {b['step']} makes the step it claims to avoid")
        for n, st in enumerate(planned):
            needed = self.goal not in self.saturate(self.premises, skip=st)
            if n in bypassed and needed:
                errors.append(f"planned step {n} ({st[0]}) is claimed bypassable but every proof makes it")
            if n not in bypassed and not needed:
                errors.append(f"planned step {n} ({st[0]}) is bypassable but not reported as such")
        return errors


    def label(self, cert: dict) -> list:
        """A label is a claim about the hypothesis: the proof reaches it (entailment) or its negation
        (contradiction) from the premises as given. A neutral certificate's proof is counterfactual: it
        reaches the hypothesis only once the `withheld` premises are restored, so those must be among the
        premises, and the remaining ones (the actual problem) must not reach it under the rules."""
        label = cert["label"]
        if label not in LABELS:
            return [f"unknown label {label!r}"]
        if "hypothesis" not in cert:
            return ["a labelled certificate must state its hypothesis"]
        h = self.canon(_t(cert["hypothesis"]))
        goal = self.canon(self.negate(h)) if label == "contradiction" else h
        errors = [] if self.goal == goal else [f"the goal is not what a {label} label needs (the "
                                                   f"{'negated ' * (label == 'contradiction')}hypothesis)"]
        withheld = [self.canon(_t(p)) for p in cert.get("withheld", ())]
        if label != "neutral":
            return errors + (["only a neutral certificate may withhold premises"] if withheld else [])
        if not withheld:
            return errors + ["a neutral certificate must name the premises its proof adds (withheld)"]
        if not set(withheld) <= set(self.premises):
            errors.append("a withheld premise is not among the certificate's premises")
        actual = [p for p in self.premises if p not in set(withheld)]
        if self.goal in self.saturate(actual):
            errors.append("the actual premises derive the hypothesis: the label is not neutral")
        return errors


def _sexprs(text):
    """SMT-LIB text -> nested lists of tokens; |quoted| symbols lose their bars."""
    stack, out = [[]], []
    for tok in re.findall(r"\|[^|]*\||[()]|[^\s()|]+", re.sub(r";[^\n]*", "", text)):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced parentheses: a ')' closes nothing")
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok.strip("|"))
    if len(stack) != 1:
        raise ValueError("unbalanced parentheses: a '(' is never closed")
    return stack[0]


class ModelChecker:
    """Evaluates SMT-LIB assertions over a finite model: sorts are lists of elements, functions total
    tables; quantifiers range over a sort's elements (or Booleans). Integers are only values, never
    quantified over."""

    def __init__(self, model: dict):
        self.sorts = {s: list(xs) for s, xs in model["sorts"].items()}
        self.raw = {f: t for f, t in model["functions"].items()}
        self.tables = {f: ({tuple(r[:-1]): r[-1] for r in t["rows"]}, t["else"]) for f, t in model["functions"].items()}
        self.signature = {}  # function -> (argument sorts, result sort), from the declarations checked

    def domain(self, sort):
        if sort == "Bool":
            return [True, False]
        if not self.sorts.get(sort):
            raise ValueError(f"quantifier over {sort}, which has no finite nonempty domain")
        return self.sorts[sort]

    def member(self, sort, v) -> bool:
        if sort == "Bool":
            return isinstance(v, bool)
        if sort == "Int":
            return isinstance(v, int) and not isinstance(v, bool)
        return isinstance(sort, str) and v in self.sorts.get(sort, ())

    def declarations(self, cmds) -> list:
        """The model must interpret the declared signature: every declared sort a nonempty set of distinct
        elements, every declared function a table whose arguments and values have the declared sorts."""
        errors = []
        for c in cmds:
            if c[0] == "declare-sort":
                xs = self.sorts.get(c[1])
                if not xs:
                    errors.append(f"sort {c[1]} has no elements in the model")
                elif len(set(xs)) != len(xs):
                    errors.append(f"sort {c[1]} lists an element twice")
        for c in cmds:
            if c[0] not in ("declare-fun", "declare-const"):
                continue
            name, args, result = (c[1], c[2], c[3]) if c[0] == "declare-fun" else (c[1], [], c[2])
            self.signature[name] = (tuple(args), result)
            if any(not isinstance(a, str) or (a not in ("Bool", "Int") and a not in self.sorts) for a in [*args, result]):
                errors.append(f"{name}: a sort of its declaration has no domain in the model")
                continue
            if name not in self.raw:
                errors.append(f"no table for {name}")
                continue
            t = self.raw[name]
            if not self.member(result, t["else"]):
                errors.append(f"{name}: default value {t['else']!r} is not of sort {result}")
            for r in t["rows"]:
                if len(r) != len(args) + 1 or not all(self.member(a, x) for a, x in zip(args, r)) \
                        or not self.member(result, r[-1]):
                    errors.append(f"{name}: row {r!r} does not match ({' '.join(args)}) {result}")
                    break
        return errors

    def ev(self, t, env):
        if isinstance(t, str):
            if t in env:
                return env[t]
            if t in ("true", "false"):
                return t == "true"
            if t.isdigit():
                return int(t)
            return self.apply(t, ())
        if not t:
            raise ValueError("empty expression")
        op, args = t[0], t[1:]
        if isinstance(op, list):
            raise ValueError(f"unsupported term {t!r}")
        if op == "!":
            return self.ev(args[0], env)
        if op == "let":
            return self.ev(args[1], {**env, **{v: self.ev(x, env) for v, x in args[0]}})
        if op in ("forall", "exists"):
            names = [v for v, _ in args[0]]
            envs = ({**env, **dict(zip(names, xs))} for xs in itertools.product(*(self.domain(s) for _, s in args[0])))
            return (all if op == "forall" else any)(self.ev(args[1], e) for e in envs)
        if op == "and":
            return all(self.ev(a, env) for a in args)
        if op == "or":
            return any(self.ev(a, env) for a in args)
        if op == "=>":
            *hyps, concl = args
            return not all(self.ev(a, env) for a in hyps) or self.ev(concl, env)
        if op == "ite":
            return self.ev(args[1] if self.ev(args[0], env) else args[2], env)
        v = [self.ev(a, env) for a in args]
        if op == "not":
            return not v[0]
        if op == "=":
            return all(x == v[0] for x in v[1:])
        if op == "distinct":
            return len(set(v)) == len(v)
        if op == "xor":
            return sum(map(bool, v)) % 2 == 1
        if op == "-":
            return -v[0] if len(v) == 1 else v[0] - sum(v[1:])
        if op == "+":
            return sum(v)
        if op == "*":
            return math.prod(v)
        if op in ("<", "<=", ">", ">="):
            cmp = {"<": int.__lt__, "<=": int.__le__, ">": int.__gt__, ">=": int.__ge__}[op]
            return all(cmp(a, b) for a, b in zip(v, v[1:]))
        return self.apply(op, tuple(v))

    def apply(self, f, args):
        if f not in self.signature:
            raise ValueError(f"{f} is not declared")
        sorts, _ = self.signature[f]
        if len(args) != len(sorts) or not all(self.member(s, x) for s, x in zip(sorts, args)):
            raise ValueError(f"{f} applied to arguments outside its declared sorts")
        rows, default = self.tables[f]
        return rows.get(args, default)

    def check(self, smt2: str) -> list:
        try:
            cmds = _sexprs(smt2)
        except ValueError as e:
            return [f"malformed query: {e}"]
        bad = [c for c in cmds if not (isinstance(c, list) and c and isinstance(c[0], str))]
        if bad:
            return [f"malformed command {bad[0]!r}"]  # a bare token or () at top level
        if not any(c[0] == "assert" for c in cmds):
            return ["the query asserts nothing"]
        if any(c[0] == "assert" and len(c) != 2 for c in cmds):
            return ["an assert takes exactly one term"]
        errors = self.declarations(cmds)
        if errors:
            return errors
        for n, cmd in enumerate(c for c in cmds if c[0] == "assert"):
            try:
                if not self.ev(cmd[1], {}):
                    errors.append(f"assertion {n} is false in the model")
            except (ValueError, KeyError, TypeError, IndexError) as e:
                errors.append(f"assertion {n}: {e}")
        return errors


def check_countermodels(cert: dict) -> list:
    """A neutral label needs a model refuting entailment and one refuting contradiction. An entailment
    (contradiction) needs the model refuting contradiction (entailment): the premises hold with the
    hypothesis true (false), so they are consistent and the label is not vacuous. Each model is
    checked against the certificate's own label queries (`smt2`: premises and not the hypothesis;
    `smt2_contradiction`: premises and the hypothesis), never against assertions it carries itself, so a
    model of some other problem cannot stand in."""
    errors, refuted = [], set()
    queries = {"entailment": cert.get("smt2"), "contradiction": cert.get("smt2_contradiction")}
    for cm in cert.get("countermodels", ()):
        if cm.get("refutes") not in queries:
            errors.append(f"countermodel refuting {cm.get('refutes')!r}: not a label it can refute")
            continue
        if queries[cm["refutes"]] is None:
            errors.append(f"countermodel refuting {cm['refutes']}: the certificate has no query to check it against")
            continue
        errs = ModelChecker(cm["model"]).check(queries[cm["refutes"]])
        errors += [f"countermodel refuting {cm['refutes']}: {e}" for e in errs]
        if not errs:
            refuted.add(cm["refutes"])
    need = {"neutral": {"entailment", "contradiction"}, "entailment": {"contradiction"},
            "contradiction": {"entailment"}}.get(cert.get("label"), set())
    if need - refuted:
        why = "" if cert["label"] == "neutral" else " (premises not shown consistent)"
        errors.append(f"{cert['label']} label without checked countermodels for {sorted(need - refuted)}{why}")
    return errors


def check(cert: dict) -> list:
    return Checker(cert).check(cert) + check_countermodels(cert)


if __name__ == "__main__":  # python -m semforge.certcheck certificates.jsonl (a certificate, or a row with one, per line)
    bad = 0
    for line in open(sys.argv[1]):
        row = json.loads(line)
        cert = {"id": row.get("id"), **row["certificate"]} if "certificate" in row else row
        errs = check(cert)
        bad += bool(errs)
        for e in errs:
            print(cert.get("id", "?"), e)
    print(f"{bad} certificate(s) failed")
    sys.exit(bool(bad))
