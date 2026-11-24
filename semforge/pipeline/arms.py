"""Matched source arms and a held-out target from rows, for any domain.

Arms are matched exactly: within every (family, level, label, rung, required size) stratum each arm takes the
same number of E/C rows, and all arms share the same neutral rows, so no arm can buy difficulty or more
required reasoning. They differ only in which E/C rows:
  narrow       rows of the k most frequent patterns of their family
  vanilla      a random draw: the generator's own pattern distribution
  diverse      greedy least-covered selection over natural rows
  vanilla_all, diverse_all
               the same pair without the narrow cap (capping every stratum by the narrow pool confines
               the three arms above to what the narrow patterns cover, usually the easy strata)
  variants     least-covered selection over natural rows plus certified variants (enforced or expanded);
               matched per (family, level, label, rung) only, since adding required reasoning is its point

The target tgt_<domain>_heldout tests transfer to unseen patterns in unseen worlds. Worlds (one sample call
with all its questions and variants; a ZebraLogic puzzle; an FLD row) are split first, a --target_worlds
share going to the target side; patterns outside the narrow set are then split by hash. Sources use source
worlds and non-held patterns only, the target the reverse, so no context and no pattern is shared. Rows
whose justification enumeration was incomplete are dropped.

  python -m semforge.pipeline.arms --rows results/patterns/csp/rows.jsonl --out results/transfer/data3
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
from dataclasses import dataclass

from semforge.core.patterns import select_diverse

from .common import certified, pattern_key, read_jsonl, required_size, write_jsonl


def _hash(x, salt=""):
    return int(hashlib.sha1((salt + repr(x)).encode()).hexdigest(), 16)


def key(r):
    return pattern_key(r, family=True)


def descriptor(r):
    """Pattern key and its parts (kinds with counts, kinds, rung, family), for least-covered selection."""
    return key(r), [*(f"{a}x{b}" for a, b in r["signature"]), *(a for a, _ in r["signature"]),
                    f"rung:{r['rung']}", f"family:{r['family']}"]


def coarse(r):
    return r["family"], r["level"], r["label"], r["rung"]


def stratum(r):
    return (*coarse(r), min(required_size(r), 5))


@dataclass
class Split:
    sources: dict      # arm name -> rows (E/C plus the shared neutral rows), shuffled
    target: list
    stats: dict


def build(rows, n_ec=1600, n_neutral=800, narrow_k=3, target_worlds=0.3, seed=0) -> Split:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    to_target = lambda r: _hash(r["world"], "world") % 1000 < 1000 * target_worlds
    ec = [r for r in rows if certified(r)]
    natural = [r for r in ec if r["variant"] == "natural"]

    freq = collections.Counter(map(key, natural))
    narrow_set = set()
    for fam in {k[0] for k in freq}:
        narrow_set |= {k for k, _ in collections.Counter({k: c for k, c in freq.items() if k[0] == fam}).most_common(narrow_k)}
    held = {k for k in set(map(key, ec)) if k not in narrow_set and _hash(k) % 2 == 0}

    eligible = [r for r in natural if key(r) not in held and not to_target(r)]
    variants = [r for r in ec if r["variant"] != "natural" and key(r) not in held and not to_target(r)]
    neutral = [r for r in rows if r["label"] == "neutral"]
    shared_neutral = [r for r in neutral if not to_target(r)][:n_neutral]
    target_neutral = [r for r in neutral if to_target(r)]

    by = collections.defaultdict(list)
    for r in eligible:
        by[stratum(r)].append(r)
    by_coarse_variants = collections.defaultdict(list)
    for r in variants:
        by_coarse_variants[coarse(r)].append(r)
    share = n_ec / len(eligible)
    arms = collections.defaultdict(list)
    coarse_q, coarse_pool = collections.Counter(), collections.defaultdict(list)
    for st, pool in sorted(by.items()):
        narrow_pool = [r for r in pool if key(r) in narrow_set]
        q = min(round(len(pool) * share), len(narrow_pool))  # every capped arm can fill the same count
        if q == 0:
            continue
        arms["narrow"] += narrow_pool[:q]
        arms["vanilla"] += pool[:q]
        arms["diverse"] += select_diverse(pool, q, descriptor)
        coarse_q[st[:-1]] += q
        coarse_pool[st[:-1]] += pool
    for st, pool in sorted(by.items()):
        q = round(len(pool) * share)
        arms["vanilla_all"] += pool[:q]
        arms["diverse_all"] += select_diverse(pool, q, descriptor)
    if variants:
        for st, q in sorted(coarse_q.items()):
            arms["variants"] += select_diverse(coarse_pool[st] + by_coarse_variants[st], q, descriptor)

    sources = {}
    for name, part in arms.items():
        part = part + shared_neutral
        rng.shuffle(part)
        sources[name] = part
    target = [r for r in ec if key(r) in held and to_target(r)]
    target += target_neutral[:max(1, len(target) // 2)]
    rng.shuffle(target)

    source_worlds = {r["world"] for part in sources.values() for r in part}
    assert not source_worlds & {r["world"] for r in target}, "world leak between sources and target"
    stats = {"domain": rows[0]["domain"], "rows": len(rows), "natural_ec": len(natural), "patterns": len(freq),
             "narrow_patterns": len(narrow_set), "heldout_patterns": len(held),
             "arms": {name: arm_stats(part) for name, part in sources.items()},
             "target": {"n": len(target), "labels": dict(collections.Counter(r["label"] for r in target)),
                        "worlds": len({r["world"] for r in target})}}
    return Split(sources, target, stats)


def arm_stats(part):
    pats = [r for r in part if r["signature"]]
    return {"n": len(part), "labels": dict(collections.Counter(r["label"] for r in part)),
            "patterns": len({key(r) for r in pats}),
            "variant_rows": sum(r["variant"] != "natural" for r in part),
            "search_share": round(sum(r["rung"] == "search" for r in pats) / len(pats), 3),
            "mean_required": round(sum(map(required_size, pats)) / len(pats), 2),
            "mean_premises": round(sum(r["n_premises"] for r in part) / len(part), 2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n_ec", type=int, default=1600, help="E/C rows per arm (before stratum capping)")
    ap.add_argument("--n_neutral", type=int, default=800)
    ap.add_argument("--narrow_k", type=int, default=3, help="patterns per family in the narrow arm")
    ap.add_argument("--target_worlds", type=float, default=0.3, help="share of worlds reserved for the target")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/transfer/data3")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    split = build(read_jsonl(args.rows), args.n_ec, args.n_neutral, args.narrow_k, args.target_worlds, args.seed)
    dom = split.stats["domain"]
    for name, part in split.sources.items():
        write_jsonl(f"{args.out}/src_{dom}_{name}.jsonl", part)
    write_jsonl(f"{args.out}/tgt_{dom}_heldout.jsonl", split.target)
    json.dump(split.stats, open(f"{args.out}/{dom}_arms.json", "w"), indent=1)
    print(json.dumps(split.stats, indent=1))


if __name__ == "__main__":
    main()
