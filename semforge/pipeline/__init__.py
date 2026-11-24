"""Data pipeline: rows (certified instances with patterns) -> arms (matched sources, held-out target) -> transfer.

Each stage is a module with a `main()` CLI: python -m semforge.pipeline.{rows,arms,compare,transfer}.
"""
