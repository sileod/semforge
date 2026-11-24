"""Row I/O and the few statistics every pipeline stage shares.

A row is one certified instance as JSON: domain, family, world, level, label, premises, hypothesis,
variant, and its dependency pattern (signature = required kinds with counts, rung) when it has one.
"""
from __future__ import annotations

import json
import math
from collections import Counter


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def pattern_key(row, family=False):
    """(signature, rung), optionally prefixed by family; hashable."""
    k = (tuple(map(tuple, row["signature"])), row["rung"])
    return (row["family"], *k) if family else k


def required_size(row):
    return sum(c for _, c in row["signature"])


def certified(row):
    """An E/C row whose justifications were enumerated exhaustively (its pattern is exact)."""
    return row["signature"] is not None and row["complete"]


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    return -sum(v / n * math.log2(v / n) for v in counts.values()) if n else 0.0
