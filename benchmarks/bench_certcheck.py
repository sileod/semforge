"""Independent certificate checking: export certificates from constructed instances, re-check them with
`semforge.certcheck` (no shared code), and confirm the checker rejects corrupted certificates.

Mutations, each applied to every certificate that admits it:
  - false_mandatory: a proof step the generator found *not* mandatory is claimed mandatory;
  - omitted_mandatory: a mandatory step is left out of the claimed list (the checker is complete too);
  - bad_step: a proof step's conclusion is replaced by another premise;
  - missing_premise: a mandatory premise is deleted from the premises;
  - bad_model: (neutral) every table of a countermodel is made constant at another value;
  - dropped_model: (neutral) one of the two countermodels is removed;
  - flipped_label: the label is changed (entailment <-> contradiction, neutral -> entailment);
  - swapped_models: (neutral) each countermodel claims to refute the other label.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time

from semforge.certcheck import check
from semforge.domains.constructed import ConstructedDomain
from semforge.construct.synth import export_certificate


FLIP = {"entailment": "contradiction", "contradiction": "entailment"}


def mutations(cert, rng):
    n = len(cert["proof"])
    free = [i for i in range(n) if i not in cert["mandatory_steps"]]
    if free:
        c = copy.deepcopy(cert)
        c["mandatory_steps"].append(rng.choice(free))
        yield "false_mandatory", c
    if cert["mandatory_steps"]:
        c = copy.deepcopy(cert)
        c["mandatory_steps"].remove(rng.choice(c["mandatory_steps"]))
        yield "omitted_mandatory", c
    if n:
        c = copy.deepcopy(cert)
        i = rng.randrange(n)
        others = [p for p in c["premises"] if p != c["proof"][i]["head"]]
        if others:
            c["proof"][i]["head"] = rng.choice(others)
            yield "bad_step", c
    if cert["mandatory_premises"]:
        c = copy.deepcopy(cert)
        c["premises"].remove(rng.choice(c["mandatory_premises"]))
        c["mandatory_premises"] = []
        yield "missing_premise", c
    if cert.get("countermodels"):
        c = copy.deepcopy(cert)
        model = c["countermodels"][0]["model"]
        values = [x for xs in model["sorts"].values() for x in xs]
        for t in model["functions"].values():
            other = [v for v in ([True, False] if isinstance(t["else"], bool) else values) if v != t["else"]]
            t["else"], t["rows"] = (other[0] if other else t["else"]), []
        yield "bad_model", c
        c = copy.deepcopy(cert)
        c["countermodels"].pop()
        yield "dropped_model", c
        c = copy.deepcopy(cert)
        for cm in c["countermodels"]:
            cm["refutes"] = FLIP[cm["refutes"]]
        yield "swapped_models", c
    yield "flipped_label", {**cert, "label": FLIP.get(cert["label"], "entailment")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+", default=["propositional", "monadic", "csp", "modal", "modal_s4", "modal_nested",
                                                    "modal_nested_s4", "order", "kinship"])
    ap.add_argument("--strategies", nargs="+", default=["shape", "forward"])
    ap.add_argument("--n", type=int, default=50, help="certificates per pack and strategy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/construct/certcheck.json")
    args = ap.parse_args()
    rows = []
    for name, strategy in [(p, s) for p in args.packs for s in args.strategies]:
        d, rng = ConstructedDomain(name, strategy=strategy), random.Random(args.seed)
        certs = []
        while len(certs) < args.n:
            out = d.sample(rng, 6)
            if out and (c := export_certificate(d.pack, out[0][0])):
                certs.append(c)
        t, failed = time.time(), sum(bool(check(c)) for c in certs)
        dt = (time.time() - t) / len(certs)
        caught, tried = {}, {}
        for c in certs:
            for kind, m in mutations(c, rng):
                tried[kind] = tried.get(kind, 0) + 1
                caught[kind] = caught.get(kind, 0) + bool(check(m))
        r = dict(pack=name, strategy=strategy, certificates=len(certs), failed=failed, s_per_check=round(dt, 3),
                 neutral=sum(c["label"] == "neutral" for c in certs),
                 mean_steps=round(sum(len(c["proof"]) for c in certs) / len(certs), 1),
                 mean_mandatory=round(sum(len(c["mandatory_steps"]) for c in certs) / len(certs), 1),
                 mean_universe=round(sum(len(c["universe"]["literals"]) for c in certs) / len(certs)),
                 mutations={k: f"{caught[k]}/{tried[k]}" for k in tried})
        rows.append(r)
        print(json.dumps(r), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
