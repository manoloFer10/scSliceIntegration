"""Command-line entrypoint.

Subcommands:
  metrics <root> [--aggregate-dir DIR]   full per-object metrics (original behavior)
  count <root> [--out DIR]               count 3-D objects in an existing NIfTI tree
  convert <data> [--out ROOT] [seg opts] TIF → NIfTI (re-render with given hyperparams)
  count-leica <data> [--out DIR] [opts]  convert then count, end to end

Backward compatible: `python -m sc3d <root> [...]` (no subcommand) dispatches to
`metrics`, so existing docs/habits keep working.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import config
from .core.counting import count_tree
from .core.pipeline import run
from .io.nifti import NiftiStackLoader

_SUBCOMMANDS = {"metrics", "count", "convert", "count-leica"}


def _add_seg_opts(p: argparse.ArgumentParser) -> None:
    """Segmentation hyperparameters for the converter (defaults = config r1 point)."""
    p.add_argument("--threshold", type=float, default=config.COUNT_THRESHOLD,
                   help=f"Absolute GFP intensity floor (default: {config.COUNT_THRESHOLD}).")
    p.add_argument("--min-volume-um3", type=float, default=config.COUNT_MIN_VOLUME_UM3,
                   dest="min_volume_um3",
                   help=f"Minimum object volume in µm³ (default: {config.COUNT_MIN_VOLUME_UM3}); "
                        "only used when --min-voxels is 0/negative.")
    p.add_argument("--min-voxels", type=int, default=config.COUNT_MIN_VOXELS,
                   dest="min_voxels",
                   help=f"Minimum object size as a direct voxel count (default: {config.COUNT_MIN_VOXELS}); "
                        "takes precedence over --min-volume-um3. Pass 0 to use the µm³ filter instead.")
    p.add_argument("--denoise-radius", type=int, default=config.COUNT_DENOISE_RADIUS,
                   dest="denoise_radius",
                   help=f"XY median denoise radius in voxels (default: {config.COUNT_DENOISE_RADIUS}).")
    p.add_argument("--fully-connected", action=argparse.BooleanOptionalAction,
                   default=config.COUNT_FULLY_CONNECTED,
                   help="26-connectivity for connected components (default: on).")


def _seg_kwargs(args: argparse.Namespace) -> dict:
    # min_voxels > 0 wins (direct voxel filter); 0/negative falls back to the µm³ path.
    min_voxels = args.min_voxels if args.min_voxels and args.min_voxels > 0 else None
    return dict(
        threshold=args.threshold,
        min_volume_um3=args.min_volume_um3,
        min_voxels=min_voxels,
        denoise_radius=args.denoise_radius,
        fully_connected=args.fully_connected,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sc3d",
        description="3D per-object metrics and GFP+ cell counting for NIfTI stacks.",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    p_metrics = sub.add_parser("metrics", parents=[common],
                               help="Compute full per-object metrics for a NIfTI tree.")
    p_metrics.add_argument("root", type=Path,
                           help="Root folder containing Day_X/Condition/stack subfolders.")
    p_metrics.add_argument("--aggregate-dir", type=Path, default=None,
                           help="Where to write all_samples_metrics.csv (default: root).")

    p_count = sub.add_parser("count", parents=[common],
                             help="Count 3-D objects per stack in an existing NIfTI tree.")
    p_count.add_argument("root", type=Path, nargs="?", default=Path(config.LEICA_OUT_DIRNAME),
                         help=f"NIfTI tree to count (default: {config.LEICA_OUT_DIRNAME}).")
    p_count.add_argument("--out", type=Path, default=None,
                         help=f"Where to write {config.COUNTS_CSV} (default: root).")

    p_conv = sub.add_parser("convert", parents=[common],
                            help="Convert raw Leica TIF series to a NIfTI tree.")
    p_conv.add_argument("data", type=Path, help="Raw data root (Analysis_<TISSUE>/Analysis_Day N/Series*).")
    p_conv.add_argument("--out", type=Path, default=Path(config.LEICA_OUT_DIRNAME),
                        help=f"Output NIfTI tree root (default: {config.LEICA_OUT_DIRNAME}).")
    _add_seg_opts(p_conv)

    p_cl = sub.add_parser("count-leica", parents=[common],
                          help="Convert raw Leica TIFs then count objects, end to end.")
    p_cl.add_argument("data", type=Path, help="Raw data root.")
    p_cl.add_argument("--out", type=Path, default=Path("out"),
                      help=f"Output dir; the tree is written to <out>/{config.LEICA_OUT_DIRNAME} "
                           f"and {config.COUNTS_CSV} to <out> (default: out).")
    _add_seg_opts(p_cl)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatibility: a bare root (not a known subcommand, not a flag) →
    # dispatch to `metrics` so `python -m sc3d <root> [...]` keeps working.
    if argv and argv[0] not in _SUBCOMMANDS and not argv[0].startswith("-"):
        argv = ["metrics"] + argv

    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.command == "metrics":
        run(NiftiStackLoader(), args.root, args.aggregate_dir)
    elif args.command == "count":
        count_tree(args.root, args.out)
    elif args.command == "convert":
        from .preprocess.leica import convert
        convert(args.data, args.out, **_seg_kwargs(args))
    elif args.command == "count-leica":
        from .preprocess.leica import convert
        tree = args.out / config.LEICA_OUT_DIRNAME
        try:
            convert(args.data, tree, **_seg_kwargs(args))
        except Exception:
            logging.getLogger(__name__).exception("convert stage failed")
            return 1
        count_tree(tree, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
