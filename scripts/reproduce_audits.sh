#!/usr/bin/env bash
# Rerun every audit reported in docs/audit.md, from public data, into results/audit/ (or $OUT).
#
#   pip install -e ".[audit]"
#   bash scripts/reproduce_audits.sh            # all audits (a few hours on one CPU, mostly the proof audits)
#   bash scripts/reproduce_audits.sh quick      # the same, at small n, as a smoke test (minutes)
#
# FLD×2 and ZebraLogic come from the Hugging Face hub. The other sources are fetched once into $DATA:
#   ProofWriter  https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip
#   PlanBench    https://github.com/karthikv792/LLMs-Planning (commit fc638a1 audited)
#   FOLIO fixed  https://github.com/dslab-uniud/Fixing-FOLIO-and-MALLS
# All runs are seeded; outputs are deterministic up to solver timeouts.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA=${DATA:-$HOME/data}
OUT=${OUT:-results/audit}
QUICK=${1:-}
n() { if [ "$QUICK" = quick ]; then echo "$2"; else echo "$1"; fi; }
mkdir -p "$OUT/planbench"

# --- data ------------------------------------------------------------------------------------------
PW=$DATA/proofwriter-dataset-V2020.12.3
if [ ! -d "$PW" ]; then
  curl -sSLo "$DATA/proofwriter.zip" https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip
  unzip -q "$DATA/proofwriter.zip" -d "$DATA"
fi
PB=$DATA/planbench/repo
[ -d "$PB" ] || git clone -q https://github.com/karthikv792/LLMs-Planning "$PB"
BW=$PB/llm_planning_analysis/instances/blocksworld
FOLIO_REPO=$DATA/folio_fixed/repo
[ -d "$FOLIO_REPO" ] || git clone -q https://github.com/dslab-uniud/Fixing-FOLIO-and-MALLS "$FOLIO_REPO"
export SEMFORGE_FOLIO=$FOLIO_REPO/core/dataset/datasets/folio/dataset_rawpairs.jsonl

run() { echo "== $*" >&2; "$@"; }

# --- premise level: minimal supports by MARCO, one oracle per source ----------------------------------
for src in fld zebralogic; do
  run python -m semforge.audit --sources $src --n "$(n 300 30)" --out "$OUT/$src.json" > "$OUT/$src.log"
done
run python -m semforge.audit --sources folio --n "$(n 200 30)" --out "$OUT/folio.json" > "$OUT/folio.log"
for pack in propositional monadic csp modal; do
  for strategy in forward shape; do
    run python -m semforge.audit --sources constructed_${pack}_$strategy --n "$(n 300 30)" \
      --out "$OUT/constructed_${pack}_$strategy.json" > "$OUT/constructed_${pack}_$strategy.log"
  done
done
run python -m semforge.audit --sources constructed_order_shape constructed_order_forward --n "$(n 300 30)" \
  --out "$OUT/constructed_order.json" > "$OUT/constructed_order.log"
AUDIT_ROOT=$OUT run python -m benchmarks.audit_matched fld constructed_monadic_forward constructed_monadic_shape > "$OUT/matched.json"

# --- proof level: mandatory steps and minimum proofs (semforge.audit.summarize_profiles) ---------------
run python -m benchmarks.audit_fld_proofs --n "$(n 400 40)" --out "$OUT/fld_proofs.json" > "$OUT/fld_proofs.log"
run python -m benchmarks.audit_fld_proofs --depth 5 --n "$(n 400 40)" --out "$OUT/fld_depth5.json" > "$OUT/fld_depth5.log"
run python -m benchmarks.audit_fld_proofs --fld-rules --n "$(n 400 40)" --out "$OUT/fld_rules.json" > "$OUT/fld_rules.log"
run python -m benchmarks.audit_fld_proofs --fld-rules --depth 5 --n "$(n 400 40)" --out "$OUT/fld_rules_depth5.json" > "$OUT/fld_rules_depth5.log"
run python -m benchmarks.audit_zebra_proofs --n "$(n 400 40)" --out "$OUT/zebra_proofs.json" > "$OUT/zebra_proofs.log"
run python -m benchmarks.audit_grid_vs_zebra --n "$(n 60 10)" --out "$OUT/grid_vs_zebra_n60.json" > "$OUT/grid_vs_zebra_n60.log"
run python -m benchmarks.audit_proofwriter --data "$PW/OWA/depth-5/meta-test.jsonl" --per-depth "$(n 300 20)" \
  --out "$OUT/proofwriter_owa_d5.json" > "$OUT/proofwriter_owa_d5.log"

# --- plan level: action landmarks, exact BFS -------------------------------------------------------
run python -m benchmarks.audit_planbench --domain "$BW/generated_domain.pddl" --instances "$BW/generated_basic" \
  --info "$BW/info.json" --n "$(n 0 20)" --out "$OUT/planbench/blocksworld.json" > "$OUT/planbench/blocksworld.log"
run python -m benchmarks.audit_planbench --domain "$BW/mystery/generated_domain.pddl" \
  --instances "$BW/mystery/generated_basic" --n "$(n 0 20)" \
  --out "$OUT/planbench/mystery.json" > "$OUT/planbench/mystery.log"

python -m benchmarks.audit_proverqa --n "$(n 500 40)" --out "$OUT/proverqa.json" > "$OUT/proverqa.log"
python -m benchmarks.audit_logicgraph --n "$(n 0 20)" --out "$OUT/logicgraph.json" > "$OUT/logicgraph.log"

echo "done: $OUT" >&2
