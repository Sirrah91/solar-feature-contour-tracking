"""
Run sliding-window logistic regression and save results as parquet.

Example
-------
python -m scr.convective_regimes.scripts.run_regression \\
    --filter_modes sunspots pores \\
    --phases all forming stable decaying \\
    --boundary b550.0
"""
from scr.config.env import initialize_environment
initialize_environment()

import socket
from typing import Sequence

import numpy as np

from scr.convective_regimes.analysis.logistic import sliding_gamma_analysis
from scr.convective_regimes.io.loaders import extract_q, load_all
from scr.convective_regimes.io.filenames import regression_filename
from scr.convective_regimes.settings import DATA_DIR
from scr.devtools.parser import (
    CustomArgumentParser,
    CustomFormatter,
    inplace_process_nargs1_args,
    print_args,
    farewell,
)
from scr.utils.filesystem import check_dir
from scr.io.parquet import save_parquet


def get_parser() -> CustomArgumentParser:
    parser = CustomArgumentParser(
        allow_abbrev=False,
        add_help=True,
        description=(
            "Sliding-window logistic regression in inclination space.\n\n"
            "Runs two models per (filter_mode, phase) combination:\n"
            "  UP   umbra/penumbra boundary  (Ic > ic_up)\n"
            "  PQ   penumbra/quiet-sun boundary  (Ic < ic_pq)\n\n"
            "Example:\n"
            "  python run_regression.py --filter_modes sunspots --phases stable"
        ),
        formatter_class=CustomFormatter,
    )

    data = parser.add_argument_group("data settings")
    data.add_argument(
        "--filter_modes",
        type=str,
        nargs="+",
        default=["sunspots", "pores"],
        choices=["sunspots", "pores"],
        help="Object types to process.",
    )
    data.add_argument(
        "--phases",
        type=str,
        nargs="+",
        default=["all", "forming", "stable", "decaying"],
        choices=["all", "forming", "stable", "decaying"],
        help=(
            "'all' loads all three phases combined; individual names load one at a time."
        ),
    )
    data.add_argument(
        "--boundary",
        type=str,
        nargs=1,
        default="b550.0",
        help="Region boundary key used when loading pixel arrays.",
    )
    data.add_argument(
        "--data_dir",
        type=str,
        nargs=1,
        default=DATA_DIR,
        help=f"Directory containing npz input files and receiving parquet output.",
    )

    model = parser.add_argument_group("model settings")
    model.add_argument(
        "--ic_up",
        type=float,
        nargs=1,
        default=0.5,
        help="Ic threshold for umbra-penumbra (UP) target.",
    )
    model.add_argument(
        "--ic_pq",
        type=float,
        nargs=1,
        default=0.9,
        help="Ic threshold for penumbra-quiet-sun (PQ) target.",
    )
    model.add_argument(
        "--window_half_width",
        type=float,
        nargs=1,
        default=5.0,
        help="Half-width of sliding inclination window (deg).",
    )
    model.add_argument(
        "--min_samples",
        type=int,
        nargs=1,
        default=1000,
        help="Minimum pixels per window to fit the model.",
    )
    model.add_argument(
        "--gamma_min",
        type=float,
        nargs=1,
        default=5.0,
        help="First window centre (deg).",
    )
    model.add_argument(
        "--gamma_max",
        type=float,
        nargs=1,
        default=86.0,
        help="Exclusive upper bound for window centres (deg).",
    )
    model.add_argument(
        "--gamma_step",
        type=float,
        default=2.0,
        help="Step between window centres (deg).",
    )

    return parser


def run_chain(args) -> None:
    check_dir(args.data_dir)

    gamma_centers = np.arange(args.gamma_min, args.gamma_max, args.gamma_step)

    for filter_mode in args.filter_modes:
        for phase in args.phases:
            print(f"Processing: {filter_mode} / {phase}")

            bver = extract_q(
                load_all([filter_mode], [phase], data_dir=args.data_dir),
                q="Br", boundary=args.boundary,
            )
            bhor = extract_q(
                load_all([filter_mode], [phase], data_dir=args.data_dir),
                q="Bhor", boundary=args.boundary,
            )
            gamma = extract_q(
                load_all([filter_mode], [phase], data_dir=args.data_dir),
                q="Binc", boundary=args.boundary,
            )
            Ic = extract_q(
                load_all([filter_mode], [phase], data_dir=args.data_dir),
                q="Ic", boundary=args.boundary,
            )

            results_up = sliding_gamma_analysis(
                gamma=gamma,
                bver=bver,
                bhor=bhor,
                target=(Ic > args.ic_up).astype(int),
                gamma_centers=gamma_centers,
                window_half_width=args.window_half_width,
                min_samples=args.min_samples,
            )

            results_pq = sliding_gamma_analysis(
                gamma=gamma,
                bver=bver,
                bhor=bhor,
                target=(Ic < args.ic_pq).astype(int),
                gamma_centers=gamma_centers,
                window_half_width=args.window_half_width,
                min_samples=args.min_samples,
            )

            filename_UP = regression_filename(
                data_dir=args.data_dir,
                object_type=args.object_type,
                phase=phase,
                region="UP",
            )
            filename_PQ = regression_filename(
                data_dir=args.data_dir,
                object_type=args.object_type,
                phase=phase,
                region="PQ",
            )

            save_parquet(
                regression_filename(data_dir=args.data_dir, object_type=filter_mode, phase=phase, region="UP"),
                results_up
            )
            save_parquet(
                regression_filename(data_dir=args.data_dir, object_type=filter_mode, phase=phase, region="PQ"),
                results_pq
            )
            print(f"  Saved UP ({len(results_up)} rows) and PQ ({len(results_pq)} rows)")


def main(sent_args: Sequence[str] | None = None) -> None:
    hostname = socket.gethostname()
    print(f"Running on: {hostname}\n")

    parser = get_parser()
    args = parser.parse_args(sent_args)

    inplace_process_nargs1_args(parser, args)
    print_args(args)

    run_chain(args)

    print(f"\nPipeline Complete!")
    farewell()


if __name__ == "__main__":
    main()
