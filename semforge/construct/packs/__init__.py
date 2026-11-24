"""Packs: one module per logic, each exposing PACK."""
from importlib import import_module

NAMES = ("propositional", "monadic", "csp", "modal", "modal_s4", "modal_nested", "modal_nested_s4", "modal_hybrid", "order", "grid", "kinship", "planning", "arithmetic", "boxes", "monadic_syllogistic")
MODULES = {"kinship": "datalog", "monadic_syllogistic": "monadic"}  # packs declared in a module of another name


def get(name: str):
    if name not in NAMES:
        raise KeyError(f"unknown pack {name!r}; known: {', '.join(NAMES)}")
    module = import_module(f".{MODULES.get(name, name)}", __name__)
    return module.SYLLOGISTIC if name == "monadic_syllogistic" else module.PACK
