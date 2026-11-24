"""Locate the reasoning_core checkout these adapters were written against, and record its commit."""
import os
import subprocess
import sys

RC_PATH = os.environ.get("SEMFORGE_RC", os.path.join(os.path.dirname(__file__), "..", "..", "rc-jev-docs"))
RC_PATH = os.path.abspath(RC_PATH)
if RC_PATH not in sys.path:
    sys.path.insert(0, RC_PATH)


def rc_commit():
    try:
        return subprocess.check_output(["git", "-C", RC_PATH, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None
