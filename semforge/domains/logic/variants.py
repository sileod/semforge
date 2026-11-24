"""Certified variants for logic_depth NLI (Datalog with signed atoms, labels from the full chase).

Operators
  invariant  rename            permute entities within each type (all premises and the hypothesis)
  invariant  shuffle           permute premise order
  invariant  distractor:*      add one premise that leaves the label unchanged, graded by strength:
               irrelevant      a fact sharing no predicate or entity with the hypothesis
               lexical         a fact with the hypothesis predicate about other entities
               near_miss_<s>   a rule concluding the foil (the atom that would change the label) whose
                               body lacks exactly s facts; adding them is verified to fire it
  twin       remove_necessary  delete one premise that occurs in every proof of the target (E/C -> N)
  twin       complete          add one fact that makes a neutral hypothesis provable or refutable

Rules keep one rendering template across all variants of an instance, so a variant's text differs
from the base only by the edit.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field, replace

from ...core.provenance import logic_depth_provenance
from ...core.variation import Orbit, Variant, certify

ANSWER = {"entailment": "Yes", "contradiction": "No", "neutral": "Maybe"}


def _L():
    from reasoning_core.tasks import logic_depth as L
    return L


@dataclass(frozen=True)
class NLIInstance:
    theory: object
    hyp: object
    order: tuple                      # facts and rules in display order
    rids: dict = field(default_factory=dict, compare=False, hash=False)


def label(inst):
    L = _L()
    res = L.chase(inst.theory)
    if res.inconsistent:
        return None
    return L._label_for_atom(res, inst.hyp)


def _rid(rule):
    return int(hashlib.sha1(repr(rule).encode()).hexdigest(), 16) % 997


def render(inst):
    L = _L()
    pack = inst.theory.domain_pack
    lines = []
    for x in inst.order:
        if isinstance(x, L.Rule):
            lines.append(L.rule_text(x, rid=inst.rids.get(x, _rid(x)), pack=pack)[0])
        else:
            lines.append(L.atom_text(x, pack) + ".")
    return lines, L.atom_text(inst.hyp, pack) + "."


def prompt(inst):
    lines, hyp = render(inst)
    return ("Premises:\n" + "\n".join(lines) + f"\n\nHypothesis: {hyp}\n\n"
            "Is the hypothesis true given the premises? Answer Yes if the premises entail it, No if they "
            "entail its negation, and Maybe otherwise. End your reply with exactly one line: "
            "'Answer: Yes', 'Answer: No' or 'Answer: Maybe'.")


def _with(inst, facts=None, rules=None, order=None, hyp=None):
    L = _L()
    t = inst.theory
    theory = L.Theory(list(t.facts if facts is None else facts), list(t.rules if rules is None else rules),
                      t.denials, t.pred_sigs, t.entities, t.domain_pack)
    return NLIInstance(theory, inst.hyp if hyp is None else hyp,
                       tuple(inst.order if order is None else order), inst.rids)


def from_case(case):
    L = _L()
    order = tuple(case.theory.facts) + tuple(case.theory.rules)
    inst = NLIInstance(case.theory, case.hyp, order, {r: _rid(r) for r in case.theory.rules})
    return inst


def _insert(order, item, rng):
    order = list(order)
    order.insert(rng.randrange(len(order) + 1), item)
    return order


# ---- invariants ----------------------------------------------------------------------------------

def rename(inst, rng):
    L = _L()
    mapping = {}
    for typ, names in inst.theory.entities.items():
        names = list(names)
        perm = names[:]
        for _ in range(10):
            rng.shuffle(perm)
            if perm != names:
                break
        mapping.update(zip(names, perm))
    if all(mapping.get(a, a) == a for a in inst.hyp.args):
        return None

    def atom(a):
        return L.Atom(a.pred, tuple(mapping.get(x, x) for x in a.args), a.sign)

    def rule(r):
        body = tuple(atom(b) if isinstance(b, L.Atom) else
                     L.Not(atom(b.atom)) if isinstance(b, L.Not) else b for b in r.body)
        return L.Rule(body, atom(r.head), r.name, r.shape)

    facts = [atom(a) for a in inst.theory.facts]
    rules = [rule(r) for r in inst.theory.rules]
    remap = {**dict(zip(inst.theory.facts, facts)), **dict(zip(inst.theory.rules, rules))}
    rids = {remap[r]: rid for r, rid in inst.rids.items() if r in remap}
    out = _with(inst, facts, rules, [remap[x] for x in inst.order], atom(inst.hyp))
    return replace(out, rids=rids)


def shuffle(inst, rng):
    order = list(inst.order)
    for _ in range(10):
        rng.shuffle(order)
        if order != list(inst.order):
            return _with(inst, order=order)
    return None


# ---- distractors ---------------------------------------------------------------------------------

def _unknown_atoms(inst, closure, preds, entity_pool):
    L = _L()
    out = []
    for sig in inst.theory.pred_sigs.values():
        if sig.name not in preds:
            continue
        pools = [[e for e in inst.theory.entities[t] if e in entity_pool] for t in sig.arg_types]
        for args in __import__("itertools").product(*pools):
            if len(args) == 2 and args[0] == args[1]:
                continue
            for sign in (True, False):
                a = L.Atom(sig.name, tuple(args), sign)
                if a not in closure and L.opposite(a) not in closure:
                    out.append(a)
    return out


def distractor_fact(inst, rng, grade):
    L = _L()
    res = L.chase(inst.theory)
    h = inst.hyp
    all_ents = {e for es in inst.theory.entities.values() for e in es}
    if grade == "lexical":
        cands = [a for a in _unknown_atoms(inst, res.closure, {h.pred}, all_ents) if a.args != h.args]
    else:
        preds = {p for p in inst.theory.pred_sigs if p != h.pred}
        cands = _unknown_atoms(inst, res.closure, preds, all_ents - set(h.args))
    rng.shuffle(cands)
    for a in cands[:20]:
        edited = _with(inst, facts=[*inst.theory.facts, a], order=_insert(inst.order, a, rng))
        if label(edited) == label(inst):
            return edited, {"strength": grade, "added": repr(a)}
    return None, {}


def foil(label_, hyp, rng):
    L = _L()
    if label_ == "entailment":
        return L.opposite(hyp)
    if label_ == "contradiction":
        return hyp
    return rng.choice([hyp, L.opposite(hyp)])


def near_miss(inst, rng, missing):
    """A rule concluding the foil whose body has one satisfied atom (if any) and `missing` unknown ones."""
    L = _L()
    res = L.chase(inst.theory)
    base_label = L._label_for_atom(res, inst.hyp)
    f = foil(base_label, inst.hyp, rng)
    ents = list(dict.fromkeys(f.args))
    var = {e: v for e, v in zip(ents, ("?x", "?y", "?z"))}

    def gen(a):
        return L.Atom(a.pred, tuple(var[x] for x in a.args), a.sign)

    satisfied = [a for a in res.closure if a.pred != f.pred and set(a.args) <= set(ents)]
    unknown = [a for a in _unknown_atoms(inst, res.closure, set(inst.theory.pred_sigs), set(ents))
               if a.pred != f.pred and set(a.args) <= set(ents)]
    for _ in range(30):
        if len(unknown) < missing:
            return None, {}
        miss = rng.sample(unknown, missing)
        if len({(m.pred, m.args) for m in miss}) < missing:
            continue
        body = ([rng.choice(satisfied)] if satisfied else []) + miss
        rule = L.Rule(tuple(gen(b) for b in body), gen(f), "distractor", "near_miss")
        edited = _with(inst, rules=[*inst.theory.rules, rule], order=_insert(inst.order, rule, rng))
        edited = replace(edited, rids={**inst.rids, rule: _rid(rule)})
        if label(edited) != base_label:
            continue
        fired = L.close_with(edited.theory, miss)
        if not (fired.inconsistent or f in fired.closure):
            continue
        return edited, {"strength": f"near_miss_{missing}", "foil": repr(f), "missing": [repr(m) for m in miss]}
    return None, {}


# ---- twins ---------------------------------------------------------------------------------------

def remove_necessary(inst, rng):
    L = _L()
    res = L.chase(inst.theory)
    lab = L._label_for_atom(res, inst.hyp)
    target = L.target_for(lab, inst.hyp)
    if target is None:
        return None, {}
    source = {x: i for i, x in enumerate(inst.order)}
    nec = logic_depth_provenance(inst.theory, source).necessary(target)
    if not nec:
        return None, {}
    premises = sorted((inst.order[i] for i in nec), key=lambda x: (isinstance(x, L.Rule), repr(x)))
    facts = [p for p in premises if not isinstance(p, L.Rule)]
    pick = rng.choice(facts or premises)
    edited = _with(inst, facts=[a for a in inst.theory.facts if a != pick],
                   rules=[r for r in inst.theory.rules if r != pick],
                   order=[x for x in inst.order if x != pick])
    return edited, {"removed": repr(pick), "is_rule": isinstance(pick, L.Rule)}


def complete(inst, rng, hint=None):
    L = _L()
    res = L.chase(inst.theory)
    if L._label_for_atom(res, inst.hyp) != "neutral":
        return None, {}
    cands = ([hint] if hint is not None else []) + rng.sample(
        list(L._all_ground_atoms(inst.theory)), min(150, sum(1 for _ in L._all_ground_atoms(inst.theory))))
    for a in cands:
        if a in res.closure or L.opposite(a) in res.closure or a in (inst.hyp, L.opposite(inst.hyp)):
            continue
        edited = _with(inst, facts=[*inst.theory.facts, a], order=_insert(inst.order, a, rng))
        if label(edited) in ("entailment", "contradiction"):
            return edited, {"added": repr(a)}
    return None, {}


def orbit(case, rng, near_miss_strengths=(1, 2)):
    inst = from_case(case)
    lab = label(inst)
    if lab is None:
        return None
    base = Variant("base", "base", inst, lab)
    vs = []

    def keep(edited, op, kind, meta=None):
        v = certify(base, edited, op, kind, label, meta)
        if v is not None:
            vs.append(v)

    keep(rename(inst, rng), "rename", "invariant")
    keep(shuffle(inst, rng), "shuffle", "invariant")
    for grade in ("irrelevant", "lexical"):
        e, m = distractor_fact(inst, rng, grade)
        keep(e, f"distractor:{grade}", "invariant", m)
    for s in near_miss_strengths:
        e, m = near_miss(inst, rng, s)
        keep(e, f"distractor:near_miss_{s}", "invariant", m)
    if lab == "neutral":
        e, m = complete(inst, rng, getattr(case, "neutral_completion", None))
        keep(e, "twin:complete", "twin", m)
    else:
        e, m = remove_necessary(inst, rng)
        keep(e, "twin:remove_necessary", "twin", m)
    return Orbit(base, vs)


# ---- pattern enforcement -------------------------------------------------------------------------

def _kind(rule):
    return f"{rule.shape}{'' if rule.head.sign else '-'}"


def enforce_pattern(inst, keep):
    """Make the rule kinds of one minimal proof `keep` required, by deleting a minimum set of premises.

    Every minimal support whose rule kinds do not include all of `keep`'s is a shortcut. A minimum
    hitting set of the shortcuts that avoids `keep` removes them all; `keep` survives, and the rules are
    monotone, so the label is unchanged and no new proof appears. Returns (instance, meta) or (None, {}).
    """
    import z3
    L = _L()
    lab = label(inst)
    target = L.target_for(lab, inst.hyp) if lab else None
    if target is None:
        return None, {}
    prov = logic_depth_provenance(inst.theory, {x: i for i, x in enumerate(inst.order)})
    if not prov.is_exact(target):
        return None, {}
    sups = prov.minimal_supports(target)
    keep = frozenset(keep)
    if keep not in sups:
        return None, {}
    want = {_kind(inst.order[i]) for i in keep if isinstance(inst.order[i], L.Rule)}
    bad = [s for s in sups if not want <= {_kind(inst.order[i]) for i in s if isinstance(inst.order[i], L.Rule)}]
    if not bad:
        return inst, {"removed": [], "kinds": sorted(want)}
    cand = sorted(set().union(*bad) - keep)
    if any(not (s - keep) for s in bad):
        return None, {}
    var = {i: z3.Bool(f"d{i}") for i in cand}
    opt = z3.Optimize()
    for s in bad:
        opt.add(z3.Or([var[i] for i in s - keep]))
    opt.minimize(z3.Sum([z3.If(v, 1, 0) for v in var.values()]))
    if opt.check() != z3.sat:
        return None, {}
    model = opt.model()
    drop = {inst.order[i] for i in cand if z3.is_true(model.eval(var[i], model_completion=True))}
    edited = _with(inst, facts=[a for a in inst.theory.facts if a not in drop],
                   rules=[r for r in inst.theory.rules if r not in drop],
                   order=[x for x in inst.order if x not in drop])
    if label(edited) != lab:
        return None, {}
    return edited, {"removed": [repr(x) for x in drop], "kinds": sorted(want)}


SHAPES = ("u_imp", "u_and", "rel_y", "rel_x", "converse", "composition")


def expand_fact(inst, rng, shape, required_of):
    """Replace one fact of the unique proof by a rule of `shape` plus fresh facts that fire it.

    The removed fact is no longer available, so every proof of the target goes through the new rule and
    its kind becomes required. Verified, not assumed: the label must be unchanged and `required_of(new)`
    (rule kinds common to all minimal proofs) must contain the new rule's kind. Returns (instance, meta).
    """
    L = _L()
    lab = label(inst)
    target = L.target_for(lab, inst.hyp) if lab else None
    if target is None:
        return None, {}
    prov = logic_depth_provenance(inst.theory, {x: i for i, x in enumerate(inst.order)})
    if not prov.is_exact(target):
        return None, {}
    sups = prov.minimal_supports(target)
    if len(sups) != 1:
        return None, {}
    closure = L.chase(inst.theory).closure
    sigs, ents = inst.theory.pred_sigs, inst.theory.entities
    unary = [s for s in sigs.values() if len(s.arg_types) == 1]
    binary = [s for s in sigs.values() if len(s.arg_types) == 2]
    facts = [inst.order[i] for i in sups[0] if not isinstance(inst.order[i], L.Rule)]
    rng.shuffle(facts)

    def other(typ, avoid):
        pool = [e for e in ents[typ] if e not in avoid]
        return rng.choice(pool) if pool else None

    def fresh(a):
        return a not in closure and L.opposite(a) not in closure

    for f in facts:
        sig = sigs[f.pred]
        for _ in range(20):
            body, facts_new = None, None
            if len(f.args) == 1:
                a = f.args[0]
                same = [s for s in unary if s.name != f.pred and s.arg_types == sig.arg_types]
                if shape == "u_imp" and same:
                    p = rng.choice(same)
                    body, facts_new = (L.Atom(p.name, ("?x",)),), [L.Atom(p.name, (a,))]
                elif shape == "u_and" and len(same) >= 2:
                    p, q = rng.sample(same, 2)
                    body, facts_new = (L.Atom(p.name, ("?x",)), L.Atom(q.name, ("?x",))), [L.Atom(p.name, (a,)), L.Atom(q.name, (a,))]
                elif shape in ("rel_y", "rel_x"):
                    rels = [r for r in binary if r.arg_types[0 if shape == "rel_y" else 1] == sig.arg_types[0]]
                    if not rels:
                        break
                    r = rng.choice(rels)
                    t_other = r.arg_types[1 if shape == "rel_y" else 0]
                    us = [s for s in unary if s.arg_types == (t_other,) and s.name != f.pred]
                    b = other(t_other, {a})
                    if not us or b is None:
                        break
                    u = rng.choice(us)
                    if shape == "rel_y":
                        body = (L.Atom(r.name, ("?x", "?y")), L.Atom(u.name, ("?y",)))
                        facts_new = [L.Atom(r.name, (a, b)), L.Atom(u.name, (b,))]
                    else:
                        body = (L.Atom(r.name, ("?y", "?x")), L.Atom(u.name, ("?y",)))
                        facts_new = [L.Atom(r.name, (b, a)), L.Atom(u.name, (b,))]
            elif len(f.args) == 2:
                a, c = f.args
                if shape == "converse":
                    rels = [r for r in binary if r.name != f.pred and r.arg_types == sig.arg_types[::-1]]
                    if rels:
                        r = rng.choice(rels)
                        body, facts_new = (L.Atom(r.name, ("?y", "?x")),), [L.Atom(r.name, (c, a))]
                elif shape == "composition":
                    firsts = [r for r in binary if r.name != f.pred and r.arg_types[0] == sig.arg_types[0]]
                    if firsts:
                        r = rng.choice(firsts)
                        seconds = [s for s in binary if s.name != f.pred and s.arg_types == (r.arg_types[1], sig.arg_types[1])]
                        b = other(r.arg_types[1], {a, c})
                        if seconds and b is not None:
                            s = rng.choice(seconds)
                            body = (L.Atom(r.name, ("?x", "?y")), L.Atom(s.name, ("?y", "?z")))
                            facts_new = [L.Atom(r.name, (a, b)), L.Atom(s.name, (b, c))]
            if body is None:
                break
            if not all(fresh(x) for x in facts_new) or len(set(facts_new)) < len(facts_new):
                continue
            head_vars = ("?x",) if len(f.args) == 1 else ("?x", "?z" if shape == "composition" else "?y")
            name = "signed" if shape == "u_and" and not f.sign else shape
            rule = L.Rule(body, L.Atom(f.pred, head_vars, f.sign), "expand", name)
            order = [x for x in inst.order if x != f]
            for x in [rule, *facts_new]:
                order = _insert(order, x, rng)
            edited = _with(inst, facts=[x for x in inst.theory.facts if x != f] + facts_new,
                           rules=[*inst.theory.rules, rule], order=order)
            edited = replace(edited, rids={**inst.rids, rule: _rid(rule)})
            if label(edited) != lab:
                continue
            req = required_of(edited)
            if req is None or _kind(rule) not in dict(req):
                continue
            return edited, {"replaced": repr(f), "rule": repr(rule), "kind": _kind(rule)}
    return None, {}
