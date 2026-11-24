"""Transfer study with tasknet: each target alone vs target + one source arm, at equal steps.

Every treatment shares the target-only baseline of its seed, so arms compare as paired deltas. Targets
train on a small fixed sample (few-shot) and evaluate on a disjoint held-out split; sources train on all
of their rows (src_<arm>.jsonl from semforge.pipeline.arms). Real targets load from the Hub (HF below);
any other target name reads tgt_<name>.jsonl from --data.

  python -m semforge.pipeline.transfer --data results/transfer/data3 --arms csp_vanilla csp_diverse \
      --targets csp_heldout wanli --steps 600 --attn sdpa --batch 4 --accum 4 --out results/transfer/run2
"""
from __future__ import annotations

import argparse

from datasets import ClassLabel, Dataset, DatasetDict

from .common import read_jsonl

NAMES = {"ld": ["entailment", "contradiction", "neutral"], "sl": [0, 1]}


def load(path, kind, n_train=None, n_eval=None, seed=0):
    rows = read_jsonl(path)
    names = NAMES[kind]
    ds = Dataset.from_dict({"premises": [r["premises"] for r in rows],
                            "hypothesis": [r["hypothesis"] for r in rows],
                            "label": [names.index(r["label"]) for r in rows]})
    ds = ds.cast_column("label", ClassLabel(names=[str(n) for n in names]))
    if n_train is None:
        return DatasetDict(train=ds)
    ds = ds.shuffle(seed=seed)
    n_eval = min(n_eval or len(ds) - n_train, len(ds) - n_train)
    return DatasetDict(train=ds.select(range(n_train)), validation=ds.select(range(n_train, n_train + n_eval)))


HF = {  # name: (dataset, s1, s2, label column, label map, train split, eval split)
    "wanli": ("alisawuffles/WANLI", "premise", "hypothesis", "gold", None, "train", "test"),
    "folio": ("tasksource/folio", "premises", "conclusion", "label",
              {"True": "entailment", "False": "contradiction", "Uncertain": "neutral"}, "train", "validation"),
    "control": ("tasksource/ConTRoL-nli", "premise", "hypothesis", "label", None, "train", "test"),
    "semfrag": ("tasksource/semantic_fragments_nli", "sentence1", "sentence2", "gold_label", None, "train", "test"),
}


def load_hf(name, n_train, n_eval, seed=0):
    from datasets import load_dataset
    path, s1, s2, y, mapping, tr, ev = HF[name]
    raw = load_dataset(path)
    names = NAMES["ld"]

    def conv(split, n):
        d = raw[split].shuffle(seed=seed)
        rows = [r for r in d.select(range(min(len(d), 3 * n))) if (mapping or {}).get(r[y], r[y]) in names][:n]
        ds = Dataset.from_dict({"premises": [r[s1] for r in rows], "hypothesis": [r[s2] for r in rows],
                                "label": [names.index((mapping or {}).get(r[y], r[y])) for r in rows]})
        return ds.cast_column("label", ClassLabel(names=names))

    return DatasetDict(train=conv(tr, n_train), validation=conv(ev, n_eval))


def task(name, data):
    import tasknet as tn
    return tn.Classification(dataset=data, s1="premises", s2="hypothesis", y="label", name=name)


def main(argv=None):
    import tasknet as tn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="results/transfer/data3")
    ap.add_argument("--arms", nargs="+", default=["narrow", "vanilla", "diverse"],
                    help="source files src_<arm>.jsonl, e.g. csp_narrow logic_variants")
    ap.add_argument("--targets", nargs="+", default=["wanli", "semfrag"])
    ap.add_argument("--target_train", type=int, default=200)
    ap.add_argument("--target_eval", type=int, default=1000)
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=2048, help="tasknet drops longer rows, so keep ConTRoL whole")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--no_bf16", action="store_true")
    ap.add_argument("--attn", default="flash_attention_2", help="'' for the model default")
    ap.add_argument("--pack_encoder", action="store_true",
                    help="packed-encoders kernels for ModernBERT-family encoders (torch 2.11; same model)")
    ap.add_argument("--out", default="results/transfer/run")
    args = ap.parse_args(argv)
    sources = [task(f"src_{a}", load(f"{args.data}/src_{a}.jsonl", "ld")) for a in args.arms]
    targets = [task(t, load_hf(t, args.target_train, args.target_eval) if t in HF else
                    load(f"{args.data}/tgt_{t}.jsonl", "sl" if t.startswith("sl_") else "ld",
                         args.target_train, args.target_eval))
               for t in args.targets]
    result = tn.transfer(sources=sources, targets=targets, model=args.model, steps=args.steps,
                         seeds=args.seeds, source_fraction=0.5, output_dir=args.out,
                         training_args={"per_device_train_batch_size": args.batch, "learning_rate": args.lr,
                                        "gradient_accumulation_steps": args.accum,
                                        "max_length": args.max_length, "bf16": not args.no_bf16,
                                        **({"attn_implementation": args.attn} if args.attn else {}),
                                        **({"pack_encoder": True} if args.pack_encoder else {})})
    print(result.matrix())
    print(result.summary())


if __name__ == "__main__":
    main()
