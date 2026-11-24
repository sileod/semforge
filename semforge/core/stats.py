"""Small, explicit generation accounting; nested stages may overlap in time."""
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import wraps
from time import perf_counter
from unittest.mock import patch


@dataclass
class Attempt:
    outcome: str = "accepted"


class GenerationStats:
    def __init__(self):
        self.outcomes = defaultdict(Counter)
        self.seconds = Counter()

    @contextmanager
    def measure(self, stage):
        attempt = Attempt()
        start = perf_counter()
        try:
            yield attempt
        except BaseException as exc:
            attempt.outcome = f"error:{type(exc).__name__}"
            raise
        finally:
            self.outcomes[stage][attempt.outcome] += 1
            self.seconds[stage] += perf_counter() - start

    def report(self):
        out = {}
        for stage, counts in sorted(self.outcomes.items()):
            accepted = counts["accepted"]
            total = sum(counts.values())
            out[stage] = {
                "attempts": total, "accepted": accepted, "outcomes": dict(counts),
                "seconds": self.seconds[stage],
                "attempts_per_accepted": total / accepted if accepted else None,
                "seconds_per_accepted": self.seconds[stage] / accepted if accepted else None,
            }
        return out


@contextmanager
def profile_calls(module, classifiers, stats, *, prefix=""):
    """Observe existing functions without changing their return values or RNG use.

    Classifiers map function names to result -> outcome functions. Counters describe
    stage calls, not a disjoint partition of rejected generator draws. Patches are
    restored on failure; this profiling helper is for single-threaded experiments.
    """
    def wrapper(name, original, classify):
        @wraps(original)
        def call(*args, **kwargs):
            with stats.measure(prefix + name) as attempt:
                result = original(*args, **kwargs)
                attempt.outcome = classify(result)
                return result
        return call

    with ExitStack() as stack:
        for name, classify in classifiers.items():
            original = getattr(module, name)
            stack.enter_context(patch.object(module, name, wrapper(name, original, classify)))
        yield
