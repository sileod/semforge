"""reasoning-core ReferenceTracking (pure tracking items): how many moves does the answer depend on, against
n_steps, and how often does the last move naming the target ball give the answer?

  PYTHONPATH=. python benchmarks/audit_tracking.py
"""
import random, re, collections, statistics as st, json
import semforge.rc  # noqa: puts the reasoning_core checkout on the path
from reasoning_core.tasks.tracking import ReferenceTracking, ReferenceTrackingConfig

def required(meta):
    """Moves the answer depends on, walking back from the last move that affects the target: a named move
    states its destination (1 move; a pronoun move also needs its antecedent), a bulk move or a swap needs
    the target's box just before it (recurse)."""
    target, raw, res = meta.target, meta.moves, meta.resolved_moves
    place = dict(meta.initial_placement); hist = []  # placement before each move
    for r in res:
        hist.append(dict(place))
        if m := re.match(r"Move (b\d+) from (x\d+) to (x\d+)", r): place[m[1]] = m[3]
        elif m := re.match(r"Move all contents of (x\d+) to (x\d+)", r):
            for b, x in list(place.items()):
                if x == m[1]: place[b] = m[2]
        elif m := re.match(r"Swap the balls in (x\d+) and (x\d+)", r):
            for b, x in list(place.items()):
                if x == m[1]: place[b] = m[2]
                elif x == m[2]: place[b] = m[1]
    used, i = 0, len(res) - 1
    while i >= 0:
        r, before = res[i], hist[i]
        affects = (re.match(rf"Move {target} from", r) or
                   (m := re.match(r"Move all contents of (x\d+)", r)) and before[target] == m[1] or
                   (m := re.match(r"Swap the balls in (x\d+) and (x\d+)", r)) and before[target] in (m[1], m[2]))
        if affects:
            used += 1
            if r.startswith(f"Move {target} from"):
                return used + raw[i].startswith("Move it")
        i -= 1
    return used

def last_mention(meta):
    for s in reversed(meta.moves):
        if re.search(rf"\b{meta.target}\b", s): return re.findall(r"x\d+", s)[-1]
    return meta.initial_placement[meta.target]

out = {}
for level in range(0, 9, 2):
    random.seed(level)
    c = ReferenceTrackingConfig(); c.apply_difficulty(level); c.winograd_p = 0.0
    t = ReferenceTracking(config=c)
    rows = [t.generate_entry() for _ in range(600)]
    req = [required(e.metadata) for e in rows]
    lm = [last_mention(e.metadata) == e.answer for e in rows]
    out[level] = {"n_steps": c.n_steps, "required_mean": round(st.mean(req), 2), "required_dist": dict(sorted(collections.Counter(min(r, 5) for r in req).items())),
                  "last_mention_acc": round(st.mean(lm), 3), "indirect_desc": round(st.mean(not e.metadata.question.startswith(f"Where is {e.metadata.target} ") for e in rows), 2)}
    print(level, out[level])
