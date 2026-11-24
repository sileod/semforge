# Reproducing the paper

Every number in *Certify What You Plant* comes from these entry points. Data builds are deterministic
for a given code version.

- **Dataset audits.** `bash scripts/reproduce_audits.sh` downloads ProofWriter (OWA depth-5 test),
  PlanBench and the corrected FOLIO, reads FLD×2 and ZebraLogic from the Hugging Face hub, and reruns every
  audit; `bash scripts/reproduce_audits.sh quick` does the same at small sample sizes in minutes. ProverQA
  and LogicGraph are audited by `benchmarks/audit_proverqa.py` and `benchmarks/audit_logicgraph.py` (also
  run by the script). FLD×2 under its own rules: `benchmarks/audit_fld_proofs.py`.
- **Construction (ProofPlant) benchmarks.** `benchmarks/bench_scale.py` (success and time by size),
  `bench_baselines.py` (forward generation, filtering and per-instance rejection), `bench_coverage.py`
  (proof-shape coverage), `bench_certcheck.py` (certificate corruption tests), `stress_atlas.py` (atlas
  soundness stress test).
- **Calculus robustness.** `experiments/calculus_robustness.py` (fused two-step rules, rank stability).
- **LLM error analysis.** `experiments/llm_realized_depth.py` (what predicts errors on FLD×2),
  `experiments/llm_fld_wording.py` (quantifier rewording), `experiments/llm_constructed_depth.py`
  (constructed problems by mandatory steps). These query hosted models through an OpenAI-compatible API
  whose keys are read from the environment.
- **Training sets and runs.** `experiments/llm_structure_mixture.py` (subcommands `build`, `forward`) and
  `experiments/llm_depth_training.py` build and train the depth-mixture sets; `experiments/llm_regmix.py`
  runs the mixture search and recomputes the transfer tables and their paired bootstrap intervals
  (`report`, `compare`); `fldlike_rows.py` builds the FLD×2-style baseline and `external_rows.py`,
  `tptp_nl_rows.py`, `modalbench_rows.py` and `folio_rows.py` the external tests; `planned_rows.py`
  (with `--polarity`) the certified-depth contrast, `multiplicity_rows.py` the proof-multiplicity
  contrast, `kripke_rows.py` the Kripke problem types, and `llm_proof_derivation.py` the proof-writing
  experiments. `experiments/cue_reliability.py audit` measures the hypothesis-negation cue of any set.

Training uses Qwen3 (0.6B full fine-tuning; 1.7B to 8B with LoRA), SmolLM2 and Gemma-3 base models on one
GPU per run; install with `pip install -e '.[train]'` (the paper's runs used torch 2.7.1, transformers
5.17.0 and peft 0.15.2). The LLM error analysis needs `pip install -e '.[llm]'`. Seeds are given per
table in the paper.
