"""PlanBench at the plan level: how much of a plan is forced? Mandatory steps for planning are action
landmarks, ground actions that every plan contains (Hoffmann, Porteous & Sebastia, JAIR 2004).

  python -m benchmarks.audit_planbench --domain <dir>/generated_domain.pddl --instances <dir>/generated_basic

Exact, by breadth-first search over the (small) state spaces of PlanBench STRIPS instances. Per instance:
  - optimal plan length (the nominal depth), and the length of PlanBench's own gold plan;
  - number of optimal plans;
  - action landmarks: removing the action makes the goal unreachable (only actions of one plan qualify);
  - optimal landmarks: removing it makes the optimal plan longer (only actions of one optimal plan);
  - fact landmarks among the facts of one optimal plan's states, excluding the initial state and goal.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import statistics
import time
from collections import Counter, deque


def sexprs(text):
    stack = [[]]
    for tok in re.findall(r"[()]|[^\s()]+", re.sub(r";[^\n]*", "", text.lower())):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    return stack[0][0]


def section(expr, key):
    return next((x for x in expr if isinstance(x, list) and x and x[0] == key), None)


def atoms(e):
    """Positive and negative atoms of a STRIPS conjunction."""
    if not e:
        return [], []
    if e[0] == "and":
        pos, neg = [], []
        for x in e[1:]:
            p, n = atoms(x)
            pos += p
            neg += n
        return pos, neg
    if e[0] == "not":
        return [], [tuple(e[1])]
    return [tuple(e)], []


class Task:
    def __init__(self, domain_text, problem_text):
        dom, prob = sexprs(domain_text), sexprs(problem_text)
        self.objects = [o for o in section(prob, ":objects")[1:] if o != "-" and not o.startswith("?")]
        self.init = frozenset(tuple(a) for a in section(prob, ":init")[1:])
        goal = section(prob, ":goal")[1]
        self.goal = frozenset(atoms(goal)[0])
        schemas = []
        for a in (x for x in dom if isinstance(x, list) and x and x[0] == ":action"):
            name, params = a[1], [p for p in a[a.index(":parameters") + 1] if p.startswith("?")]
            pre = atoms(a[a.index(":precondition") + 1])[0]
            add, dele = atoms(a[a.index(":effect") + 1])
            schemas.append((name, params, pre, add, dele))
        fluent = {f[0] for _, _, _, add, dele in schemas for f in add + dele}
        self.actions = []  # (name, pre, add, del), static preconditions checked against init
        for name, params, pre, add, dele in schemas:
            for objs in itertools.product(self.objects, repeat=len(params)):
                s = dict(zip(params, objs))
                g = lambda fs: [tuple(s.get(x, x) for x in f) for f in fs]
                gp = g(pre)
                if all(f in self.init for f in gp if f[0] not in fluent):
                    self.actions.append(((name, *objs), frozenset(f for f in gp if f[0] in fluent),
                                         frozenset(g(add)), frozenset(g(dele))))
        static = {f for f in self.init if f[0] not in fluent}
        self.init = self.init - static
        self.goal = self.goal - static

    def search(self, banned=frozenset(), banned_fact=None):
        """BFS from init: (optimal length or None, parents dict, goal states at that depth, depth map)."""
        dist, preds, frontier, goals = {self.init: 0}, {self.init: []}, [self.init], []
        if banned_fact in self.init:
            return None, preds, [], dist
        while frontier and not goals:
            nxt = []
            for st in frontier:
                for a, pre, add, dele in self.actions:
                    if a in banned or not pre <= st:
                        continue
                    t = (st - dele) | add
                    if banned_fact is not None and banned_fact in t:
                        continue
                    if t not in dist:
                        dist[t], preds[t] = dist[st] + 1, []
                        nxt.append(t)
                    if dist[t] == dist[st] + 1:
                        preds[t].append((st, a))
            frontier = nxt
            goals = [s for s in frontier if self.goal <= s]
        if self.goal <= self.init:
            return 0, preds, [self.init], dist
        return (dist[goals[0]] if goals else None), preds, goals, dist

    def reachable(self, banned):
        """Is the goal reachable without the banned actions (any length)?"""
        return self.search(banned)[0] is not None

    @staticmethod
    def one_plan(preds, goal_state):
        plan, states, st = [], [goal_state], goal_state
        while preds[st]:
            st, a = preds[st][0]
            plan.append(a)
            states.append(st)
        return plan[::-1], states[::-1]

    @staticmethod
    def count_plans(preds, goals):
        memo = {}

        def n(st):
            if st not in memo:
                memo[st] = 1 if not preds[st] else sum(n(p) for p, _ in preds[st])
            return memo[st]
        return sum(n(g) for g in goals)


def audit(task: Task):
    opt, preds, goals, _ = task.search()
    if opt is None:
        return {"solvable": False}
    plan, states = task.one_plan(preds, goals[0])
    landmarks = [a for a in dict.fromkeys(plan) if not task.reachable(frozenset([a]))]
    opt_landmarks = [a for a in dict.fromkeys(plan) if task.search(frozenset([a]))[0] != opt]
    facts = {f for st in states[1:] for f in st} - task.init - task.goal
    fact_landmarks = [f for f in facts if task.search(banned_fact=f)[0] is None]
    return {"solvable": True, "objects": len(task.objects), "optimal": opt,
            "optimal_plans": task.count_plans(preds, goals), "landmarks": len(landmarks),
            "optimal_landmarks": len(opt_landmarks), "fact_landmarks": len(fact_landmarks),
            "landmark_kinds": [a[0] for a in landmarks]}


def summarize(rows):
    rows = [r for r in rows if r.get("solvable")]
    mean = lambda k: round(statistics.mean(r[k] for r in rows), 2)
    by_len = {}
    for r in rows:
        by_len.setdefault(r["optimal"], []).append(r)
    return {
        "instances": len(rows), "objects": mean("objects"), "optimal_length": mean("optimal"),
        "gold_length": mean("gold") if all("gold" in r for r in rows) else None,
        "gold_optimal": round(sum(r.get("gold") == r["optimal"] for r in rows) / len(rows), 3),
        "landmarks": mean("landmarks"), "optimal_landmarks": mean("optimal_landmarks"),
        "fact_landmarks": mean("fact_landmarks"),
        "landmark_share": round(sum(r["landmarks"] for r in rows) / max(1, sum(r["optimal"] for r in rows)), 3),
        "zero_landmarks": round(sum(r["landmarks"] == 0 for r in rows) / len(rows), 3),
        "unique_optimal_plan": round(sum(r["optimal_plans"] == 1 for r in rows) / len(rows), 3),
        "median_optimal_plans": statistics.median(r["optimal_plans"] for r in rows),
        "by_optimal_length": {k: {"n": len(v), "landmarks": round(statistics.mean(r["landmarks"] for r in v), 2),
                                  "optimal_landmarks": round(statistics.mean(r["optimal_landmarks"] for r in v), 2),
                                  "median_optimal_plans": statistics.median(r["optimal_plans"] for r in v)}
                              for k, v in sorted(by_len.items())},
        "landmark_kinds": Counter(k for r in rows for k in r["landmark_kinds"]).most_common(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--info", help="PlanBench info.json with gold plans")
    ap.add_argument("--n", type=int, default=0, help="first n instances (0: all)")
    ap.add_argument("--out")
    args = ap.parse_args()
    domain = open(args.domain).read()
    info = json.load(open(args.info)) if args.info else {}
    files = sorted(os.listdir(args.instances), key=lambda f: int(re.sub(r"\D", "", f) or 0))
    rows, t = [], time.time()
    for f in files[:args.n or None]:
        r = {"instance": f, **audit(Task(domain, open(os.path.join(args.instances, f)).read()))}
        if f in info:
            r["gold"] = info[f]["length"]
        rows.append(r)
    s = summarize(rows)
    s["seconds"] = round(time.time() - t, 1)
    print(json.dumps({k: v for k, v in s.items() if k != "by_optimal_length"}, indent=1))
    print(json.dumps(s["by_optimal_length"]))
    if args.out:
        json.dump({"summary": s, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
