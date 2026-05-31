"""
Compute 2D penumbra probability maps and save as npz.

Example
-------
python -m scr.convective_regimes.scripts.run_probabilities \\
    --filter_modes sunspots pores \\
    --phases all \\
    --quantities Bhor B Br \\
    --hires
"""
from scr.config.env import initialize_environment
initialize_environment()

from os import path
import socket
from typing import Sequence

import numpy as np

from scr.convective_regimes.analysis.probability import analyse_penumbra_probability
from scr.convective_regimes.io.loaders import extract_q, load_all
from scr.convective_regimes.io.filenames import probability_filename
from scr.convective_regimes.settings import DATA_DIR
from scr.devtools.parser import (
    CustomArgumentParser,
    CustomFormatter,
    inplace_process_nargs1_args,
    print_args,
    farewell,
)
from scr.utils.filesystem import check_dir
from scr.io.npz import save_npz


_REGIONS = {
    "penumbra": (0.5, 0.9),
    "umbra":    (0.0, 0.5),
    "QS":       (0.9, np.inf),
}


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
        default=list(_REGIONS.keys()),
        choices=list(_REGIONS.keys()),
        help="Intensity regions to compute maps for.",
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

    return parser


def run_chain(args) -> None:
    check_dir(args.data_dir)

    for filter_mode in args.filter_modes:
        for phase in args.phases:
            print(f"Processing: {filter_mode} / {phase}")

            data_all = load_all([filter_mode], [phase], data_dir=args.data_dir)
            gamma = extract_q(data_all, q="Binc", boundary=args.boundary)
            Ic = extract_q(data_all, q="Ic", boundary=args.boundary)

            for region in args.regions:
                ic_penumbra = _REGIONS[region]

                for quantity in args.quantities:
                    B = extract_q(data_all, q=quantity, boundary=args.boundary)

                    pmap = analyse_penumbra_probability(
                        B=B,
                        gamma=gamma,
                        Ic=Ic,
                        ic_penumbra=ic_penumbra,
                        min_count=args.min_count,
                        n_B_bins=args.n_B_bins,
                        n_g_bins=args.n_g_bins,
                    )

                    filename = probability_filename(
                        data_dir=args.data_dir,
                        object_type=filter_mode,
                        phase=phase,
                        region=region,
                        quantity=quantity,
                    )
                    save_npz(
                        filename,
                        probability=pmap.probability,
                        counts=pmap.counts,
                        x_bins=pmap.x_bins,
                        y_bins=pmap.y_bins,
                    )
                    print(f"  Saved: {path.basename(filename)}")


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
