"""Wrapper around axbench/scripts/train.py that first patches pyvene to know
about Qwen3.

Usage (drop-in replacement, same CLI as axbench/scripts/train.py):

    torchrun --nproc_per_node=1 \
        axbench/mcp-protect/train_with_qwen3_patch.py \
        --config <path> --dump_dir <path>

The patch must be imported BEFORE pyreft / pyvene / transformers, or before
any pyvene `IntervenableModel` is constructed. Importing it at the top of
this file (and before we import the underlying `train.main`) is sufficient.
"""
from __future__ import annotations

# Patch pyvene FIRST. patch_pyvene_qwen3 calls _patch() at import time, which
# imports pyvene.models.intervenable_modelcard and registers Qwen3 entries.
# This must happen before train.py imports pyreft (which imports pyvene).
import patch_pyvene_qwen3  # noqa: F401  (import-for-side-effect)

import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    # axbench/scripts/train.py uses sibling-import-style: `from args.training_args
    # import TrainingArgs`. That requires `axbench/scripts` to be on sys.path.
    repo_root = Path(__file__).resolve().parents[2]  # .../axbench/
    train_path = repo_root / "axbench" / "scripts" / "train.py"
    scripts_dir = train_path.parent

    if not train_path.exists():
        raise FileNotFoundError(f"train.py not found at {train_path}")

    # Make `from args...` and other sibling imports inside train.py work.
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    # runpy.run_path executes train.py with __name__ == "__main__", which
    # triggers train.py's `if __name__ == "__main__": main()` block.
    # train.py reads sys.argv, so we leave sys.argv as-is (torchrun already
    # set up the right argv for us).
    runpy.run_path(str(train_path), run_name="__main__")


if __name__ == "__main__":
    main()
