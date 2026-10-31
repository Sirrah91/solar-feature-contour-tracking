"""
Compute 2D penumbra probability maps and save as npz.

Example
-------
python -m src.convective_regimes.cli.run_probabilities \\
    --filter_modes sunspots pores \\
    --phases all \\
    --quantities Bhor B Br \\
    --hires
"""
from src.config.env import initialize_environment
initialize_environment()

from os import path
import socket
from typing import Sequence

import argparse
import numpy as np

from src.convective_regimes.analysis.probability import analyse_penumbra_probability
from src.convective_regimes.io.loaders import extract_q, load_all
from src.convective_regimes.io.filenames import probability_filename
from src.convective_regimes.settings import DATA_DIR
from src.devtools.parser import (
    CustomArgumentParser,
    CustomFormatter,
    inplace_process_nargs1_args,
    print_args,
    farewell,
)
from src.utils.filesystem import mkdir
from src.io.npz import save_npz


_REGIONS = {
    "penumbra": (0.5, 0.9),
    "umbra":    (0.0, 0.5),
    "QS":       (0.9, np.inf),
}


def parse_ic_edge(value: str) -> float:
    value = value.strip().lower()

    if value.startswith("neg"):
        return -float(value[3:])

    return float(value)


def get_parser() -> CustomArgumentParser:
    parser = CustomArgumentParser(
        allow_abbrev=False,
        add_help=True,
        description=(
            "Compute 2D conditional penumbra probability in (B, gamma) space.\n\n"
            "Example:\n"
            "  python run_probabilities.py --filter_modes sunspots --phases stable "
            "--hires"
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
        help="Phases to process.",
    )
    data.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=None,
        choices=list(_REGIONS.keys()),
        help="Intensity regions to compute maps for.",
    )
    data.add_argument(
        "--ic_subbins",
        type=parse_ic_edge,
        nargs="+",
        default=None,
        help=(
            "Optional Ic edges subdividing the penumbra into darker/brighter "
            "sub-bins, e.g. --ic_subbins 0.5 0.6 0.7 0.8 0.9. Each "
            "consecutive pair defines a sub-region; probability is computed "
            "relative to the full penumbra population (first..last edge), "
            "not the whole image."
        ),
    )
    data.add_argument(
        "--quantities",
        type=str,
        nargs="+",
        default=["Bhor", "B", "Br"],
        choices=["Bhor", "B", "Br"],
        help="Magnetic field quantities to use as the x-axis.",
    )
    data.add_argument(
        "--boundary",
        type=str,
        nargs=1,
        default="b605.0",
        help="Region boundary key for loading pixel arrays.",
    )
    data.add_argument(
        "--data_dir",
        type=str,
        nargs=1,
        default=DATA_DIR,
        help=f"Directory with npz input files and output destination.",
    )

    model = parser.add_argument_group("histogram settings")
    model.add_argument(
        "--n_B_bins",
        type=int,
        nargs=1,
        default=150,
        help="Number of field-strength bins.",
    )
    model.add_argument(
        "--n_g_bins",
        type=int,
        nargs=1,
        default=45,
        help="Number of inclination bins.",
    )
    model.add_argument(
        "--min_count",
        type=int,
        default=50,
        help="Bins with fewer total counts are masked to NaN.",
    )
    model.add_argument(
        "--subbin_min_count",
        type=int,
        default=None,
        help="min_count override for --ic_subbins maps (defaults to --min_count).",
    )

    return parser


def _compute_and_save(*, filter_mode, phase, region_name, ic_bounds,
                       ic_reference, gamma, Ic, data_all, args, min_count):
    for quantity in args.quantities:
        B = extract_q(data_all, q=quantity, boundary=args.boundary)

        pmap = analyse_penumbra_probability(
            B=B, gamma=gamma, Ic=Ic,
            ic_penumbra=ic_bounds,
            ic_reference=ic_reference,
            min_count=min_count,
            n_B_bins=args.n_B_bins,
            n_g_bins=args.n_g_bins,
        )

        filename = probability_filename(
            data_dir=args.data_dir, object_type=filter_mode,
            phase=phase, region=region_name, quantity=quantity,
        )
        save_npz(
            filename,
            probability=pmap.probability,
            counts=pmap.counts,
            x_bins=pmap.x_bins,
            y_bins=pmap.y_bins,
        )
        print(f"  Saved: {path.basename(filename)}")


def run_chain(args) -> None:
    mkdir(args.data_dir)

    for filter_mode in args.filter_modes:
        for phase in args.phases:
            print(f"Processing: {filter_mode} / {phase}")

            data_all = load_all([filter_mode], [phase], data_dir=args.data_dir)
            gamma = extract_q(data_all, q="Binc", boundary=args.boundary)
            Ic = extract_q(data_all, q="Ic", boundary=args.boundary)

            if args.regions:
                for region in args.regions:
                    _compute_and_save(
                        filter_mode=filter_mode, phase=phase,
                        region_name=region, ic_bounds=_REGIONS[region],
                        ic_reference=None,
                        gamma=gamma, Ic=Ic, data_all=data_all, args=args,
                        min_count=args.min_count,
                    )

            if args.ic_subbins:
                edges = args.ic_subbins
                # full_range = (edges[0], edges[-1])
                subbin_min_count = args.subbin_min_count or args.min_count
                for lo, hi in zip(edges[:-1], edges[1:]):
                    _compute_and_save(
                        filter_mode=filter_mode, phase=phase,
                        region_name=f"Ic_{lo:.2f}-{hi:.2f}",
                        ic_bounds=(lo, hi), ic_reference=None,
                        gamma=gamma, Ic=Ic, data_all=data_all, args=args,
                        min_count=subbin_min_count,
                    )


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
