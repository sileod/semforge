"""Kim & Schuster (2023) boxes dataset: how many operations does the queried box's content depend on, against
`numops` (operations that touch the box), and how far does a per-object "last mention" rule get?

Operations are read as state assertions: "Move the X from Box A to Box B" leaves X in B and not in A,
"Remove the X from Box A" leaves X out of A, "Put the X into Box A" puts it in, "Move the contents of Box A
to Box B" moves whatever A holds then. An operation is required when deleting it changes the queried
box's contents. The data is password-protected upstream; it is read from outside this repository.

With `--semforge N`, N queries per numops level are also certified in semforge's `boxes` logic: the story
becomes premises (an operation on several objects becomes one step per object), each object's membership
in the queried box a goal, and an operation is required when replacing its steps by `idle` loses a goal.
The two counts must agree.

  python benchmarks/audit_boxes.py <boxes-dataset-v1 directory> [--semforge 200]
"""
import collections
import json
import re
import sys

OBJ = r"the ([a-z]+(?: [a-z]+)*?)"


def parse(sentence):
    parts = re.split(r"(?<=\.) ", sentence)
    init = {}
    for clause in parts[0].rstrip(".").split(", "):
        m = re.match(r"Box (\d+) (?:contains (.+)|is empty)", clause)
        init[int(m[1])] = set(re.findall(r"the ([a-z]+(?: [a-z]+)*?)(?= and |$)", m[2])) if m[2] else set()
    ops = []
    for s in parts[1:-1]:
        if m := re.match(r"Move the contents of Box (\d+) to Box (\d+)\.", s):
            ops.append(("contents", None, int(m[1]), int(m[2])))
        elif m := re.match(r"Move (.+) from Box (\d+) to Box (\d+)\.", s):
            ops.append(("move", objects(m[1]), int(m[2]), int(m[3])))
        elif m := re.match(r"Remove (.+) from Box (\d+)\.", s):
            ops.append(("remove", objects(m[1]), int(m[2]), None))
        elif m := re.match(r"Put (.+) into Box (\d+)\.", s):
            ops.append(("put", objects(m[1]), None, int(m[2])))
        else:
            raise ValueError(s)
    q = int(re.match(r"Box (\d+)", parts[-1])[1])
    return init, ops, q


def objects(text):
    return frozenset(re.findall(r"the ([a-z]+(?: [a-z]+)*?)(?= and |$)", text))


def run(init, ops):
    """Box contents after the operations, each object in one place at a time (None: in no box)."""
    loc = {o: b for b, v in init.items() for o in v}
    for kind, objs, a, b in ops:
        if kind == "contents":
            for o, x in list(loc.items()):
                if x == a:
                    loc[o] = b
        else:
            for o in objs:
                loc[o] = None if kind == "remove" else b
    box = {b: set() for b in init}
    for o, x in loc.items():
        if x is not None:
            box[x].add(o)
    return box


def last_mention(init, ops, q):
    """Per object: in the box iff the last operation naming the object puts it into q (or, never named
    after, it started there)."""
    out = set()
    for o in {o for v in init.values() for o in v} | {o for op in ops if op[1] for o in op[1]}:
        last = next((op for op in reversed(ops) if op[1] and o in op[1]), None)
        if (last is None and o in init.get(q, ())) or (last and last[0] in ("move", "put") and last[3] == q):
            out.add(o)
    return out


def certify(init, ops, q):
    """(gold contents, required operation indices) from semforge's boxes logic."""
    from semforge.construct.packs.boxes import NOWHERE, OBJECTS, PACK, closure
    calc = PACK.calculus
    names = sorted({o for v in init.values() for o in v} | {o for op in ops if op[1] for o in op[1]})
    slot = dict(zip(names, OBJECTS))
    box = lambda i: f"box{i}"
    start = {o: box(b) for b, v in init.items() for o in v}
    prem = [("in", slot[o], start.get(o, NOWHERE), "t0") for o in names]
    steps, owner = [], []  # one event per step; the operation each step comes from
    for k, (kind, objs, a, b) in enumerate(ops):
        if kind == "contents":
            steps.append(("contents", box(a), box(b))); owner.append(k)
            continue
        for o in sorted(objs):
            steps.append({"move": ("move", slot[o], box(a), box(b)), "remove": ("remove", slot[o], box(a)),
                          "put": ("put", slot[o], box(b))}[kind]); owner.append(k)
    end = f"t{len(steps)}"
    events = [(*e, f"t{i}") for i, e in enumerate(steps)]
    gold = run(init, ops)[q]
    goals = [("in", slot[o], box(q), end) if o in gold else ("not", ("in", slot[o], box(q), end)) for o in names]

    def derives(evs):
        have = closure(prem + evs, goals)
        return all(calc.canonical(g) in have for g in goals)
    assert derives(events), "the story does not certify its own answer"
    need = {owner[i] for i in range(len(events))
            if not derives([("idle", f"t{j}") if j == i else e for j, e in enumerate(events)])}
    return gold, need


def main(root, sample=0):
    for split in ("t5_boxes_nso_exp2_max3", "t5_boxes_nso_exp2_max3_move_contents"):
        by = collections.defaultdict(lambda: [0, 0, 0, 0])
        checked, agree = collections.Counter(), collections.Counter()
        for line in open(f"{root}/{split}/test-t5.jsonl"):
            r = json.loads(line)
            init, ops, q = parse(r["sentence"])
            gold = run(init, ops)[q]
            need = sum(run(init, ops[:i] + ops[i + 1:])[q] != gold for i in range(len(ops)))
            c = by[min(r["numops"], 5)]
            if sample and checked[min(r["numops"], 5)] < sample:
                checked[min(r["numops"], 5)] += 1
                _, sf = certify(init, ops, q)
                hand = {i for i in range(len(ops)) if run(init, ops[:i] + ops[i + 1:])[q] != gold}
                agree[min(r["numops"], 5)] += sf == hand
            c[0] += 1; c[1] += need; c[2] += last_mention(init, ops, q) == gold; c[3] += need < r["numops"]
        print(split)
        for k in sorted(by):
            n, need, lm, fewer = by[k]
            print(f"  numops {k}{'+' if k == 5 else ''}: n={n:6d}  required ops {need / n:.2f}  "
                  f"fewer than numops {fewer / n:.2f}  last-mention exact {lm / n:.2f}" +
                  (f"  semforge agrees {agree[k]}/{checked[k]}" if sample else ""))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[sys.argv.index("--semforge") + 1]) if "--semforge" in sys.argv else 0)
