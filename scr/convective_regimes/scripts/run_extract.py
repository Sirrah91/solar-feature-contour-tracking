"""
Extract per-pixel data from sunspot phase tracks and save as npz.

Example
-------
python -m scr.convective_regimes.scripts.run_extract \\
    --filter_mode sunspots \\
    --phase stable \\
    --sunspot_type removed \\
    --levels 550.0 605.0
"""
from scr.config.env import initialize_environment
initialize_environment()

from os import path
import socket
from typing import Sequence

import numpy as np

from scr.config.filtering import gimme_filtering_kwargs
from scr.convective_regimes.processing.extract import extract_pixel_data
from scr.convective_regimes.io.filenames import extract_filename
from scr.convective_regimes.settings import DATA_DIR, SUNSPOTS_PHASES_DIR_TEMPLATE
from scr.devtools.parser import (
    CustomArgumentParser,
    CustomFormatter,
    inplace_process_nargs1_args,
    print_args,
    farewell,
)
from scr.io.npz import save_npz
from scr.pipelines.io.load_phase_tracks import load_filtered_phase_tracks
from scr.utils.filesystem import check_dir


def get_parser() -> CustomArgumentParser:
    parser = CustomArgumentParser(
        allow_abbrev=False,
        add_help=True,
        description=(
            "Extract per-pixel magnetic and intensity data for sunspot/pore regions.\n\n"
            "Iterates over all images in the filtered phase tracks, builds region masks,\n"
            "and saves per-pixel arrays as a compressed npz file.\n\n"
            "Example:\n"
            "  python run_extract.py --filter_mode sunspots --phase stable "
            "--sunspot_type removed --levels 550.0"
        ),
        formatter_class=CustomFormatter,
    )

    core = parser.add_argument_group("core settings")
    core.add_argument(
        "--filter_mode",
        type=str,
        required=True,
        choices=["sunspots", "pores"],
        nargs=1,
        help="Object type to process.",
    )
    core.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["forming", "stable", "decaying"],
        nargs=1,
        help="Evolution phase to process.",
    )
    core.add_argument(
        "--sunspot_type",
        type=str,
        required=True,
        choices=["collapsed", "removed"],
        nargs=1,
        help="Outer boundary collapsing method.",
    )
    core.add_argument(
        "--levels",
        type=float,
        required=True,
        nargs="+",
        help="Magnetic field contour levels [G] defining B-threshold regions.",
    )

    paths = parser.add_argument_group("path settings")
    paths.add_argument(
        "--data_dir",
        type=str,
        nargs=1,
        default=DATA_DIR,
        help=f"Output directory for npz files.",
    )
    paths.add_argument(
        "--sunspots_dir",
        type=str,
        nargs=1,
        default=None,
        help=(
            "Root directory containing sunspot phase tracks. "
            "Defaults to SUNSPOTS_PHASES_DIR_TEMPLATE.format(sunspot_type=...)."
        ),
    )

    return parser


def run_chain(args) -> None:
    if args.sunspots_dir is None:
        args.sunspots_dir = SUNSPOTS_PHASES_DIR_TEMPLATE.format(
            sunspot_type=args.sunspot_type
        )

    sunspots_phases, df = load_filtered_phase_tracks(
        nosuffix_filename=path.join(args.sunspots_dir, "sunspots_phases"),
        filtering_options={
            "phase": {"mode": "frame-wise", "exact_value": args.phase}
        } | gimme_filtering_kwargs(args.filter_mode),
        filter_phase_tracks=True,
        drop_unknown=True,
    )
    print(f"{args.filter_mode}, {args.phase}: {len(df)} rows")

    data = extract_pixel_data(
        sunspots_phases=sunspots_phases,
        df=df,
        levels=args.levels,
        phase=args.phase,
    )

    save_dict = {
        f"{q}_{region}": np.asarray(data[region][q], dtype=object)
        for region in data
        for q in data[region]
    }
    save_dict.update({
        "phase": args.phase,
        "filter_mode": args.filter_mode,
    })

    filename = extract_filename(
        data_dir=args.data_dir,
        object_type=args.filter_mode,
        phase=args.phase,
    )
    check_dir(filename, is_file=True)
    save_npz(filename=filename, **save_dict)
    print(f"Saved: {filename}")


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
