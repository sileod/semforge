"""Small task adapters; reasoning_core imports remain optional and lazy.

Each oracle names its selectable premises and fixes the rest of the problem.
No adapter interprets a minimum support as a universal reasoning difficulty.
"""
from functools import lru_cache

from ..core.lattice import Oracle
from ..core.provenance import logic_depth_provenance


def linear_oracle(equations, variables, target=None, *, background=(), objective="determines"):
    """Oracle over equation indices using exact rational linear algebra.

    `determines` means a consistent system fixes the target, with other variables
    allowed to remain free. The full pool must be consistent, so all subsets are
    consistent and this property is monotone. `inconsistent` instead finds equation
    conflict cores. Variable domains are unrestricted; background equations stay fixed.
    """
    import sympy as sp

    equations, variables, background = tuple(equations), tuple(variables), tuple(background)
    if objective not in {"determines", "inconsistent"}:
        raise ValueError("objective must be determines or inconsistent")
    if objective == "determines" and target not in variables:
        raise ValueError("target must be a listed variable")
    def row(eq):
        if eq is True or eq is sp.S.true:
            return sp.zeros(1, len(variables)), sp.zeros(1, 1)
        if eq is False or eq is sp.S.false:
            return sp.zeros(1, len(variables)), sp.ones(1, 1)
        return sp.linear_eq_to_matrix([eq], variables)

    rows = [row(eq) for eq in (*background, *equations)]
    n_background = len(background)

    def matrices(indices):
        if not indices <= set(range(len(equations))):
            raise ValueError("unknown equation index")
        active = [*rows[:n_background], *(rows[n_background + i] for i in sorted(indices))]
        if not active:
            return sp.zeros(0, len(variables)), sp.zeros(0, 1)
        return sp.Matrix.vstack(*(a for a, _ in active)), sp.Matrix.vstack(*(b for _, b in active))

    full_a, full_b = matrices(frozenset(range(len(equations))))
    if objective == "determines" and full_a.rank() != full_a.row_join(full_b).rank():
        raise ValueError("determination requires a consistent full equation pool")
    target_index = variables.index(target) if objective == "determines" else None

    def holds(indices):
        a, b = matrices(indices)
        if objective == "inconsistent":
            return a.rank() != a.row_join(b).rank()
        return all(v[target_index] == 0 for v in a.nullspace())

    return Oracle(holds)


def word_problem_oracle(metadata):
    """Relation indices determining the asked count, with the revealed count fixed.

    Matches reasoning_core's relational_proof_core_size: unrestricted rational
    unknowns, not additional integer/positivity constraints. Process problems do
    not expose a selectable equation pool and are intentionally excluded.
    """
    import sympy as sp

    if metadata["family"] != "relational":
        raise ValueError("expected a relational math word problem")
    symbols = {name: sp.Symbol(name) for name in metadata["names"]}
    equations = []
    for op, a, b, k, c in metadata["relations"]:
        rhs = symbols[b] + symbols[c] if op == "combine" else {
            "times": lambda: k * symbols[b],
            "more": lambda: symbols[b] + k,
            "fewer": lambda: symbols[b] - k,
            "frac": lambda: symbols[b] / k,
        }[op]()
        equations.append(sp.Eq(symbols[a], rhs))
    return linear_oracle(equations, tuple(symbols.values()), symbols[metadata["asked"]],
                         background=[sp.Eq(symbols[metadata["given"]], metadata["given_value"])])


def logic_support_oracle(case):
    """Rendered line IDs that still derive a monotone case's target, checked by chase."""
    from reasoning_core.tasks import logic_depth as L

    if case.target is None:
        raise ValueError("a derived target is required")
    if any(L.body_not_atoms(r.body) for r in case.theory.rules):
        raise ValueError("negation as failure is not monotone")
    if case.res.inconsistent:
        raise ValueError("a consistent case is required")
    theory, source = case.theory, case.source
    # Missing IDs must not silently turn selectable premises into background.
    if any(x not in source for x in (*theory.facts, *theory.rules)):
        raise ValueError("every fact and rule needs a source ID")
    universe = frozenset(source.values())

    def holds(indices):
        if not indices <= universe:
            raise ValueError("unknown premise ID")
        sub = L.Theory([a for a in theory.facts if source[a] in indices],
                       [r for r in theory.rules if source[r] in indices],
                       theory.denials, theory.pred_sigs, theory.entities, theory.domain_pack)
        return case.target in L.chase(sub).closure

    return Oracle(holds)


def logic_derivation_audit(case, *, cap=256, max_candidates=128):
    """Keep support minimality and canonical proof-trace uniqueness separate."""
    from reasoning_core.tasks.logic_derivation import canonical_trace

    prov = logic_depth_provenance(case.theory, case.source, cap=cap)
    exact = prov.is_exact(case.target)
    trace = canonical_trace(case.theory, case.res, case.source, case.target,
                            max_depth=case.derivation.depth, max_candidates=max_candidates)
    return {
        "provenance_exact": exact,
        "minimal_support_count": len(prov.minimal_supports(case.target)) if exact else None,
        "minimum_premises": prov.min_size(case.target),
        "minimum_depth": prov.min_depth(case.target),
        "necessary": sorted(prov.necessary(case.target)) if exact else None,
        "trace_unique": trace.unique if trace else False,
        "trace_truncated": trace.truncated if trace else False,
        "trace_steps": trace.size if trace else None,
    }


def bounded_plans(model, cues=(), *, cap=2):
    """At most `cap` valid plans for the displayed STRIPS model and exact horizon.

    Independent of reasoning_core's phase-specific pruning. Cues are zero-based
    (step, action) pairs. No new actions, initial facts or goals are introduced.
    """
    cues = dict(cues)
    horizon = model["horizon"]
    if cap < 1 or any(i not in range(horizon) for i in cues):
        raise ValueError("invalid plan cap or cue position")
    actions = sorted(model["actions"], key=lambda a: a["call"])
    if len({a["call"] for a in actions}) != len(actions):
        raise ValueError("action calls must be unique")

    @lru_cache(None)
    def visit(state, step):
        if step == horizon:
            return ((),) if (set(model["goal_true"]) <= state
                             and not set(model["goal_false"]) & state) else ()
        out = []
        for action in actions:
            if step in cues and cues[step] != action["call"]:
                continue
            if not set(action["pre_true"]) <= state or set(action["pre_false"]) & state:
                continue
            nxt = (state - set(action["delete"])) | set(action["add"])
            for tail in visit(frozenset(nxt), step + 1):
                out.append((action["call"],) + tail)
                if len(out) == cap:
                    return tuple(out)
        return tuple(out)

    return visit(frozenset(model["initial_true"]), 0)


def planning_cue_oracle(model, reference):
    """Step indices whose reference-action cues isolate the fixed valid plan.

    All selectable cues agree with the reference, so adding them cannot remove
    that plan. This restriction makes unique-plan isolation monotone.
    """
    reference = tuple(reference)
    if len(reference) != model["horizon"] or bounded_plans(model, enumerate(reference)) != (reference,):
        raise ValueError("reference must be a valid plan of the displayed horizon")
    universe = frozenset(range(len(reference)))

    def holds(indices):
        if not indices <= universe:
            raise ValueError("unknown cue position")
        return bounded_plans(model, ((i, reference[i]) for i in indices)) == (reference,)

    return Oracle(holds)
