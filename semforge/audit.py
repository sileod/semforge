"""Audit any reasoning dataset or generator with the same certified, calculus-free measures.

  python -m semforge.audit --sources fld zebralogic constructed_monadic_shape constructed_monadic_forward \
      --n 300 --out results/audit/audit.json

Every source is reduced to the same object: premises, a hypothesis, a label, and an entailment oracle
(z3 or a CSP solver). For each E/C instance the audit enumerates the *minimal premise sets* that still
yield the label (MARCO, capped), and reads off:

  label_agrees       the dataset's own label equals the certified one (generated sources: by construction)
  supports           number of minimal premise sets (> 1: an alternative justification exists)
  min_support        size of the smallest one: how many premises the label really needs
  mandatory          premises in every minimal set (the certified bottlenecks)
  distractors        share of premises in no minimal set (never needed for the label)
  required kinds     the kind multiset common to every minimal set (the reasoning pattern)

and, where the source states it, whether its own claimed proof survives (FLD: the gold proof's premises
form a minimal set, and whether another minimal set exists). Summaries are taken at an equal number of E/C
instances, so diversity counts are comparable: distinct required-kind multisets, their entropy, and
distinct support-shape multisets. Kinds are each source's premise shapes (FLD: formula skeletons,
ZebraLogic: clue templates, constructed: formula skeletons in the pack's symbols).

These measures need no calculus, so they apply to datasets we cannot regenerate. Proof-level
certificates (mandatory rule applications, spine) exist only for constructed sources and are reported
alongside, not mixed in.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter

from .core.lattice import Oracle, enumerate_lattice


def entropy(c: Counter) -> float:
    n = sum(c.values())
    return -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0


def skeleton(f, names=None) -> str:
    """A formula's shape: constants and atoms renamed in order of first occurrence (a, b, ...)."""
    names = {} if names is None else names
    if not isinstance(f, tuple):
        return names.setdefault(f, "abcdefghijklmnopqrstuvwxyz"[len(names) % 26])
    return f"{f[0]}({','.join(skeleton(x, names) for x in f[1:])})"


# --- proof level: one record per derivable goal, one summary for every source --------------------------

PROFILE = ("min_depth", "min_size", "mandatory", "spine", "joins", "shared", "proofs", "supports", "exact")


def summarize_profiles(profiles, planted=None) -> dict:
    """Summary of proof profiles (Analysis.profile, or the same fields from a native audit). `planted`:
    the source's own claimed depth per row, if it states one.

    Zero mandatory steps has three causes, reported apart: the goal is given (min_size 0), it is one
    step away by several rules (1), or it has several proofs that share no step (2 or more: not
    shallow). The last splits by premise supports: `zero_rerouted` when every minimal proof uses the
    same premises (the steps can be recombined, e.g. chains under hypothetical syllogism), and
    `zero_overdetermined` when several premise sets suffice; sources without supports count in neither. `forced_share` is mandatory / min_size over rows with min_size > 0:
    the part of the smallest proof that no proof can avoid."""
    ps = [p for p in profiles if p is not None]
    if not ps:
        return {"n": 0}
    def mean(k):  # None when the source does not measure k
        xs = [p[k] for p in ps if p.get(k) is not None]
        return round(sum(xs) / len(xs), 2) if xs else None
    zero = [p for p in ps if p["mandatory"] == 0]
    sized = [p for p in ps if p["min_size"]]
    out = {"n": len(ps), **{k: mean(k) for k in PROFILE if k != "exact"},
           "exact": round(sum(bool(p.get("exact", True)) for p in ps) / len(ps), 3),
           "zero_mandatory": round(len(zero) / len(ps), 3),
           "zero_given": round(sum(p["min_size"] == 0 for p in zero) / len(ps), 3),
           "zero_one_step": round(sum(p["min_size"] == 1 for p in zero) / len(ps), 3),
           "zero_alternatives": round(sum((p["min_size"] or 0) >= 2 for p in zero) / len(ps), 3),
           "zero_rerouted": round(sum((p["min_size"] or 0) >= 2 and p.get("supports") == 1 for p in zero) / len(ps), 3),
           "zero_overdetermined": round(sum((p["min_size"] or 0) >= 2 and (p.get("supports") or 0) > 1 for p in zero) / len(ps), 3),
           "multi_support": round(sum((p.get("supports") or 0) > 1 for p in ps) / len(ps), 3),
           "forced_share": round(sum(p["mandatory"] for p in sized) / max(1, sum(p["min_size"] for p in sized)), 3),
           "multi_proof": round(sum(p["proofs"] > 1 for p in ps) / len(ps), 3)}
    if planted is not None:
        pl = [x for x, p in zip(planted, profiles) if p is not None]
        out["planted"] = round(sum(pl) / len(pl), 2)
        out["min_size_below_planted"] = round(sum(p["min_size"] < x for x, p in zip(pl, ps)) / len(ps), 3)
    return out


# --- sources: premises, oracle-backed minimal supports, kinds, and what the source claims -------------

class FLD:
    name = "fld"

    def __init__(self, **kw):
        from .domains.fld import FLDDomain
        self.d = FLDDomain(**kw)

    def sample(self, rng):
        self.seen = getattr(self, "seen", Counter())
        while True:  # neutral rows have no justification to audit: counted, then skipped
            (inst, lab), = self.d.sample(rng)
            self.seen[(inst.gold, lab)] += 1
            if lab != "neutral":
                return inst, lab, {"label_agrees": inst.gold == lab, "claimed_steps": inst.steps}

    def premises(self, inst):
        return self.d.premises(inst)

    def supports(self, inst, lab, limit):
        return self.d.justifications(inst, lab, limit=limit)

    def kind(self, p):
        return self.d.kind(p)

    def claims(self, inst, sups):
        ids = lambda s: frozenset(f for f, _ in s)
        gold = inst.gold_support
        return {"gold_is_minimal": gold in {ids(s) for s in sups},
                "gold_has_alternative": any(ids(s) != gold for s in sups)}


class Zebra:
    name = "zebralogic"

    def __init__(self, **kw):
        from .domains.zebralogic import ZebraDomain
        self.d = ZebraDomain(**kw)

    def sample(self, rng):
        inst, lab = rng.choice([x for x in self.d.sample(rng) if x[1] != "neutral"])
        return inst, lab, {}

    def premises(self, inst):
        return self.d.premises(inst)

    def supports(self, inst, lab, limit):
        return self.d.justifications(inst, lab, limit=limit)

    def kind(self, p):
        return self.d.kind(p)

    def claims(self, inst, sups):
        return {}


class FOLIO:
    """Corrected FOLIO validation conclusions (semforge.domains.folio); the label audited is z3's."""
    name = "folio"

    def __init__(self, **kw):
        from .domains import folio
        self.f = folio
        self.rows = [r for r in folio.load() if r[1] is not None]
        self.order = None

    def sample(self, rng):
        if self.order is None:
            self.order = rng.sample(self.rows, len(self.rows))
        while self.order:
            id_, prem, goal, gold, _ = self.order.pop()
            try:
                lab = self.f.label(prem, goal)
            except self.f.Undecided:
                continue
            if lab != "neutral":
                return (prem, goal, id_), lab, {"label_agrees": gold == lab}
        raise StopIteration("FOLIO exhausted")

    def premises(self, inst):
        return inst[0]

    def supports(self, inst, lab, limit):
        prem, goal, _ = inst
        g = goal if lab == "entailment" else ("not", goal)
        sups, _, complete = enumerate_lattice(prem, Oracle(lambda s: self.f.entails(list(s), g)), limit=limit)
        return sups, complete

    def kind(self, p):
        """Skeleton with predicates and constants renamed; None for a ground literal."""
        f = p[1] if p[0] == "not" else p
        if f[0] not in ("forall", "exists", "and", "or", "xor", "->", "<->", "not", "="):
            return None
        names = {}

        def sk(g):
            if not isinstance(g, tuple):
                return names.setdefault(g, "abcdefghijklmnopqrstuvwxyz"[len(names) % 26])
            if g[0] in ("forall", "exists"):
                return f"{g[0]} {sk(g[1])}.{sk(g[2])}"
            if g[0] in ("not", "and", "or", "xor", "->", "<->", "="):
                return f"{g[0]}({','.join(map(sk, g[1:]))})"
            return f"P{len(g) - 1}({','.join(map(sk, g[1:]))})"
        return sk(p)

    def claims(self, inst, sups):
        return {}


class Constructed:
    """A semforge generator, audited semantically like any dataset (MARCO over pack.entails), with its
    proof-level certificate reported alongside."""

    def __init__(self, pack, strategy):
        from .domains.constructed import ConstructedDomain
        self.d, self.name = ConstructedDomain(pack=pack, strategy=strategy), f"constructed_{pack}_{strategy}"
        self.made = self.seconds = 0

    def sample(self, rng):
        while True:
            t = time.perf_counter()
            out = self.d.sample(rng, rng.randint(0, 5))
            self.seconds += time.perf_counter() - t
            out = [x for x in out if x[1] != "neutral"]
            if out:
                self.made += 1
                inst, lab = out[0]
                return inst, lab, {"label_agrees": True}

    def premises(self, inst):
        return list(inst.premises)

    def supports(self, inst, lab, limit):
        pack = self.d.pack
        goal = inst.hypothesis if lab == "entailment" else pack.negate(inst.hypothesis)
        sups, _, complete = enumerate_lattice(inst.premises, Oracle(lambda s: pack.entails(list(s), goal)), limit=limit)
        return sups, complete

    def kind(self, p):
        """The premise's skeleton; None for a literal (an atom or a leaf symbol application, possibly
        negated), as FLD gives literal facts no kind."""
        f = p[1] if isinstance(p, tuple) and p[0] == "not" else p
        sig = self.d.pack.signature
        if isinstance(f, str) or (sig is not None and f[0] in sig.symbols and sig.symbols[f[0]].leaf
                                  and "Prop" not in sig.symbols[f[0]].args):
            return None
        return skeleton(p)

    def claims(self, inst, sups):
        a = self.d.annotations(inst, None).get("certified") or {}
        return {"certified_mandatory_steps": a.get("mandatory"), "certified_spine": a.get("spine")}


def source(name):
    if name == "fld":
        return FLD()
    if name == "zebralogic":
        return Zebra()
    if name == "folio":
        return FOLIO()
    if name.startswith("constructed_"):
        pack, strategy = name[len("constructed_"):].rsplit("_", 1)
        return Constructed(pack, strategy)
    raise KeyError(name)


# --- audit --------------------------------------------------------------------------------------------

def audit_one(src, rng, limit):
    inst, lab, meta = src.sample(rng)
    t = time.perf_counter()
    try:
        sups, complete = src.supports(inst, lab, limit)
    except Exception as e:  # an undecided oracle call: no certified supports
        return {"label": lab, "error": type(e).__name__, **meta}
    if not sups:
        return {"label": lab, "error": "no_support", **meta}
    prem = src.premises(inst)
    used = set().union(*map(set, sups))
    mandatory = set.intersection(*map(set, sups))
    kinds = [Counter(k for k in map(src.kind, s) if k is not None) for s in sups]
    required = kinds[0].copy()
    for k in kinds[1:]:
        required &= k
    return {"label": lab, **meta, "complete": complete, "premises": len(prem), "supports": len(sups),
            "min_support": min(map(len, sups)), "mandatory": len(mandatory),
            "distractor_share": round(1 - len(used) / len(prem), 3),
            "required": sorted(required.items()), "shape": sorted(Counter(map(lambda k: tuple(sorted(k.items())), kinds)).items(), key=repr),
            "audit_s": round(time.perf_counter() - t, 3), **src.claims(inst, sups)}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarize(rows, n):
    ok = [r for r in rows if "error" not in r and r["complete"]][:n]
    req = Counter(json.dumps(r["required"]) for r in ok)
    out = {"audited": len(rows), "certified_complete": len(ok),
           "undecided_or_incomplete": len(rows) - len([r for r in rows if "error" not in r and r["complete"]]),
           "errors": dict(Counter(r["error"] for r in rows if "error" in r)),
           "label_agrees": mean([float(r["label_agrees"]) for r in rows if "label_agrees" in r]),
           "unique_justification": mean([float(r["supports"] == 1) for r in ok]),
           "supports": mean([r["supports"] for r in ok]), "min_support": mean([r["min_support"] for r in ok]),
           "mandatory": mean([r["mandatory"] for r in ok]), "premises": mean([r["premises"] for r in ok]),
           "distractor_share": mean([r["distractor_share"] for r in ok]),
           "distinct_required_kinds": len(req), "required_kinds_entropy": round(entropy(req), 2),
           "distinct_shapes": len({json.dumps(r["shape"]) for r in ok}),
           "kind_vocabulary": len({k for r in ok for k, _ in r["required"]}),
           "audit_s_per_instance": mean([r["audit_s"] for r in ok])}
    for k in ("gold_is_minimal", "gold_has_alternative"):
        if any(k in r for r in ok):
            out[k] = mean([float(r[k]) for r in ok if k in r])
    for k in ("certified_mandatory_steps", "certified_spine"):
        if any(r.get(k) is not None for r in ok):
            out[k] = mean([r.get(k) for r in ok])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300, help="E/C instances per source with complete supports")
    ap.add_argument("--limit", type=int, default=64, help="minimal supports enumerated per instance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    report = {}
    for name in args.sources:
        src, rng, rows = source(name), random.Random(args.seed), []
        while sum("error" not in r and r["complete"] for r in rows) < args.n and len(rows) < 3 * args.n:
            try:
                rows.append(audit_one(src, rng, args.limit))
            except StopIteration:  # a finite dataset ran out
                break
        report[name] = summarize(rows, args.n)
        if isinstance(src, Constructed):
            report[name]["generated_per_s"] = round(src.made / src.seconds, 1)
        if isinstance(src, FLD):  # (FLD label, certified label) over every row drawn, neutrals included
            report[name]["label_pairs"] = {f"{g}->{c}": v for (g, c), v in sorted(src.seen.items())}
        print(name, json.dumps(report[name]), flush=True)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"args": vars(args), "report": report}, open(args.out, "w"), indent=1)
        json.dump(rows, open(os.path.join(os.path.dirname(args.out) or ".", f"{name}_rows.json"), "w"))


if __name__ == "__main__":
    main()
