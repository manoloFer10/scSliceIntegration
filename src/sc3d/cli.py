"""Command-line entrypoint: `python -m sc3d <root>` or `sc3d <root>`."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .core.pipeline import run
from .io.nifti import NiftiStackLoader


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute 3D per-object metrics for segmented NIfTI stacks.")
    parser.add_argument("root", type=Path, help="Root folder containing Day_X/Condition/stack subfolders.")
    parser.add_argument("--aggregate-dir", type=Path, default=None,
                        help="Where to write all_samples_metrics.csv (default: root).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    run(NiftiStackLoader(), args.root, args.aggregate_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
