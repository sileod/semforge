"""ZebraLogicBench grid puzzles as a semforge domain.

A template-aware parser maps each clue to a reasoning_core CSP formula over house-position variables
(one per attribute value, all-different within a category): same house -> EqVar, "is not" -> NeVar,
left/right of -> Lt, directly left/right -> Linear, next to / one or two between -> Distance,
"in the k-th house" -> Eq, "not in" -> Ne. Puzzles whose clues do not parse, or that mention a surface
form shared by two categories, are skipped and counted. z3 recovers the unique solution (the public
release ships blank grids). Each cell `value in house h` is then an E/C instance with certified
justifications; clue kinds are the template names, so ZebraLogic and reasoning_core's assignment family
are compared on the same footing (see `operator_kind` for the shared operator vocabulary).
"""
from __future__ import annotations

import functools
import random
import re
from dataclasses import dataclass, field

from .base import Domain

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}
NUM = {"one": 1, "two": 2, "three": 3, "four": 4}


def _U():
    from reasoning_core.tasks import _csp_utils as U
    return U


ALIAS = {"jan": "january", "feb": "february", "mar": "march", "apr": "april", "jun": "june", "jul": "july",
         "aug": "august", "sept": "september", "oct": "october", "nov": "november", "dec": "december",
         "brit": "british", "dane": "danish", "swede": "swedish"}


def _norm(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def _word_match(w, t, loose):
    if t == w or t in (w + "s", w + "es", ALIAS.get(w)) or (w.endswith("s") and t == w[:-1]):
        return True
    if w.endswith("y") and t == w[:-1] + "ies":
        return True
    return loose and len(w) >= 6 and t.startswith(w[:5])


def _mentions(tokens, values):
    """Non-overlapping value mentions: exact or inflected forms first, then stems on the rest."""
    used, found = set(), []
    for loose in (False, True):
        for key, words in sorted(values.items(), key=lambda kv: -len(kv[1])):
            for i in range(len(tokens)):
                end = _match_at(words, tokens, i, loose)
                if end is None or set(range(i, end)) & used:
                    continue
                used |= set(range(i, end))
                found.append((i, key))
    return [k for _, k in sorted(found)]


def _match_at(words, tokens, i, loose):
    """End of a match of `words` at token i; a word may also span two tokens it was split into ("f-150")."""
    for w in words:
        if i < len(tokens) and _word_match(w, tokens[i], loose):
            i += 1
        elif i + 1 < len(tokens) and tokens[i] + tokens[i + 1] == w:
            i += 2
        else:
            return None
    return i


@dataclass
class Puzzle:
    pid: str
    n: int
    categories: dict            # category -> list of values
    clues: list                 # formulas
    templates: dict             # formula -> template name
    text: str                   # puzzle text before the clues
    clue_text: dict = field(default_factory=dict)  # formula -> original clue sentence
    solution: dict = field(default_factory=dict)   # value -> house


def parse(row):
    U = _U()
    text = row["puzzle"]
    head, _, body = text.partition("## Clues:")
    n = int(re.search(r"There are (\d+) houses", head).group(1))
    cats = {}
    for line in head.splitlines():
        vals = re.findall(r"`([^`]+)`", line)
        if line.strip().startswith("-") and vals:
            cats[f"c{len(cats)}"] = vals
    variables, words = {}, {}
    for c, vals in cats.items():
        for v in vals:
            variables[(c, v)] = U.Var(f"{c}_{re.sub(r'[^a-z0-9]', '_', v.lower())}", range(1, n + 1))
            words[(c, v)] = _norm(v)
    if len({tuple(w) for w in words.values()}) < len(words):
        return None, "ambiguous surface form: " + str(sorted(words.values()))
    clues, templates, clue_text = [], {}, {}
    for line in body.strip().splitlines():
        m = re.match(r"\s*\d+\.\s*(.*)", line)
        if not m:
            continue
        raw = m.group(1).strip()
        s = raw.lower()
        found = list(dict.fromkeys(_mentions(_norm(s), words)))
        vs = [variables[x] for x in found]
        ordm = re.search(r"\b(first|second|third|fourth|fifth|sixth)\b house", s)
        if len(vs) == 1 and ordm:
            k = ORD[ordm.group(1)]
            f, t = (U.Ne(vs[0], k), "not_at") if re.search(r"\bnot\b", s) else (U.Eq(vs[0], k), "found_at")
        elif len(vs) == 2:
            a, b = vs
            if "directly left of" in s:
                f, t = U.Linear([1, -1], [b, a], "==", 1), "direct_left"
            elif "directly right of" in s:
                f, t = U.Linear([1, -1], [a, b], "==", 1), "direct_left"
            elif "somewhere to the left of" in s or "somewhere left of" in s:
                f, t = U.Lt(a, b), "left_of"
            elif "somewhere to the right of" in s or "somewhere right of" in s:
                f, t = U.Lt(b, a), "left_of"
            elif "next to each other" in s or "next to" in s:
                f, t = U.Distance(a, b, 1), "next_to"
            elif (bm := re.search(r"there (?:is|are) (one|two|three|four) houses? between", s)):
                k = NUM[bm.group(1)] + 1
                f, t = U.Distance(a, b, k), f"{bm.group(1)}_between"
            elif re.search(r"\bis not\b", s):
                f, t = U.NeVar(a, b), "not_same"
            elif re.search(r"\bis\b", s):
                f, t = U.EqVar(a, b), "same_house"
            else:
                return None, f"unparsed: {s}"
        else:
            return None, f"unparsed: {s}"
        clues.append(f)
        templates[f] = t
        clue_text[f] = raw
    base = [U.AllDifferent([variables[(c, v)] for v in vals]) for c, vals in cats.items()]
    p = Puzzle(row["id"], n, cats, clues, templates, head.rstrip(), clue_text)
    p.variables, p.base = variables, base
    s = U.CSPSolver(list(variables.values()), base)
    sol = {}
    for key, var in variables.items():
        vals = s.possible_values(var, clues)
        if len(vals) != 1:
            return None, "not unique (parse error or ambiguity)"
        sol[key] = vals[0]
    p.solution = sol
    return p, "ok"


@dataclass
class Cell:
    puzzle: Puzzle
    key: tuple       # (category, value)
    house: int
    clues: list


class ZebraDomain(Domain):
    name = "zebralogic"

    def __init__(self, puzzles=None):
        self.puzzles = puzzles if puzzles is not None else load()[0]
        self.templates = {f: t for p in self.puzzles for f, t in p.templates.items()}

    def family(self, inst):
        return f"zebra_{inst.puzzle.n}x{len(inst.puzzle.categories)}"

    def sample(self, rng, level=None, per_label=1):
        p = rng.choice(self.puzzles)  # level is ignored: puzzle size is the family
        out = []
        keys = list(p.variables)
        for want in ("entailment", "contradiction"):
            for _ in range(per_label):
                k = rng.choice(keys)
                h = p.solution[k] if want == "entailment" else rng.choice([x for x in range(1, p.n + 1) if x != p.solution[k]])
                out.append((Cell(p, k, h, list(p.clues)), want))
        for _ in range(per_label):
            n = self._neutral(p, rng.choice(keys), rng)
            if n is not None:
                out.append((n, "neutral"))
        return out

    def _neutral(self, p, k, rng):
        """Drop clues in random order until cell k is undetermined; ask about one of its still-possible houses."""
        s, clues = self._solver(Cell(p, k, 0, [])), list(p.clues)
        for c in rng.sample(clues, len(clues)):
            clues.remove(c)
            vals = s.possible_values(p.variables[k], clues)
            if len(vals) > 1:
                return Cell(p, k, rng.choice(vals), clues)
        return None

    def _solver(self, inst):
        U = _U()
        return U.CSPSolver(list(inst.puzzle.variables.values()), inst.puzzle.base)

    def label(self, inst):
        vals = self._solver(inst).possible_values(inst.puzzle.variables[inst.key], inst.clues)
        if not vals:
            return None
        if inst.house not in vals:
            return "contradiction"
        return "entailment" if len(vals) == 1 else "neutral"

    def justifications(self, inst, lab=None, limit=64):
        from .csp.oracles import forces_value, refutes
        from ..core.lattice import enumerate_lattice
        U = _U()
        lab = lab or self.label(inst)
        var, s = inst.puzzle.variables[inst.key], self._solver(inst)
        if lab == "entailment":
            o = forces_value(s, var, inst.house, pool=inst.clues)
        elif lab == "contradiction":
            o = refutes(s, [U.Eq(var, inst.house)])
        else:
            return None, False
        sups, _, complete = enumerate_lattice(inst.clues, o, limit=limit)
        return sups, complete

    def kind(self, clue):
        return self.templates.get(clue)

    def rung(self, inst, lab):
        U = _U()
        variables = list(inst.puzzle.variables.values())
        domains, _, _ = U.propagate(variables, [*inst.puzzle.base, *inst.clues])
        d = domains[inst.puzzle.variables[inst.key]]
        if (lab == "entailment" and d == [inst.house]) or (lab == "contradiction" and inst.house not in d):
            return "propagation"
        return "search"

    def without(self, inst, drop):
        return Cell(inst.puzzle, inst.key, inst.house, [c for c in inst.clues if c not in drop])

    def world(self, inst):
        return inst.puzzle.pid

    def render(self, inst):
        """Only the clues the instance keeps, in their original wording and order."""
        c, v = inst.key
        lines = [f"{i}. {inst.puzzle.clue_text[f]}" for i, f in enumerate(inst.clues, 1)]
        return inst.puzzle.text + "\n\n## Clues:\n" + "\n".join(lines), f"The person with attribute `{v}` is in house {inst.house}."

    def premises(self, inst):
        return list(inst.clues)


# clue template -> reasoning_core operator name, the kind vocabulary of the CSP domain
TEMPLATE_OPERATOR = {"same_house": "eqvar", "not_same": "nevar", "left_of": "lt", "direct_left": "linear",
                     "next_to": "distance", "one_between": "distance", "two_between": "distance",
                     "three_between": "distance", "found_at": "eq", "not_at": "ne"}


def operator_kind(clue):
    return _U().operator_name(clue)


@functools.lru_cache(maxsize=None)
def load(limit=None):
    from datasets import load_dataset
    rows = load_dataset("allenai/ZebraLogicBench", "grid_mode")["test"]
    puzzles, status = [], {}
    for r in rows.select(range(limit or len(rows))):
        p, why = parse(r)
        status[why.split(":")[0]] = status.get(why.split(":")[0], 0) + 1
        if p is not None:
            puzzles.append(p)
    return puzzles, status
