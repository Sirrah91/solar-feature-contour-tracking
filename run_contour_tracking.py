from scr.config.env import initialize_environment
initialize_environment()

import argparse
import socket
from os import path
from glob import glob
from typing import Sequence

from scr.pipelines.processing.build_contour_tracks import track_contours

from scr.config.paths import PATH_TRACKS
from scr.utils.filesystem import check_dir
from scr.utils.naming import get_initial_track_filename
from scr.utils.metadata import merge_metadata
from scr.io.tracks import save_track_file

from scr.devtools.parser import (
    CustomArgumentParser,
    CustomFormatter,
    inplace_process_nargs1_args,
    print_args,
    farewell,
)


def get_parser(
        parser: CustomArgumentParser | None = None,
        add_help_flag: bool = True
) -> CustomArgumentParser:
    if parser is None:
        parser = CustomArgumentParser(
            allow_abbrev=False,
            add_help=False,
            description=(
                "Extract and track contours from AR data based on user-defined "
                "thresholds and conditions.\n\n"
                "Example:\n"
                "  python run_contour_tracking.py "
                "--data_dir /path/to/fits "
                "--contour_quantity Ic "
                "--contour_level 0.9"
            ),
            formatter_class=CustomFormatter,
        )

    # Positional arguments
    positional = parser.add_argument_group("positional arguments")

    # Input options
    input = parser.add_argument_group("input settings")
    input.add_argument(
        "--data_dir",
        type=str,
        required=True,
        nargs=1,
        help="Directory containing input FITS files."
    )

    # Contour extraction settings
    contour = parser.add_argument_group("contour extraction")
    contour.add_argument(
        "--contour_quantity",
        type=str,
        required=True,
        nargs=1,
        choices=["Ic", "B", "Bp", "Bt", "Br", "Bhor"],
        help="Quantity used to compute the contour."
    )
    contour.add_argument(
        "--contour_level",
        type=float,
        required=True,
        nargs=1,
        help="Value to define the contour level."
    )

    # Filtering based on area and persistence
    filtering = parser.add_argument_group("filtering options")
    filtering.add_argument(
        "--max_gap",
        type=int,
        default=3,
        nargs=1,
        help="Maximum number of consecutive frames a tracked contour can be absent and "
             "still be considered part of a valid track."
    )
    filtering.add_argument(
        "--min_area",
        type=float,
        default=3.0,
        nargs=1,
        help="Minimum contour area in pixels² for contours to be considered."
    )

    # Morphology and contour merging
    morph = parser.add_argument_group("morphology and merging")
    morph.add_argument(
        "--iou_threshold",
        type=float,
        default=0.3,
        nargs=1,
        help="Minimum IoU (Intersection over Union) required to merge two contours into one region."
    )
    morph.add_argument(
        "--registration",
        action="store_true",
        help="Enable image alignment (registration) before contour tracking."
    )

    # Output options
    output = parser.add_argument_group("output options")
    output.add_argument(
        "--track_outdir",
        type=str,
        default=PATH_TRACKS,
        nargs=1,
        help="Output directory for saving results."
    )
    output.add_argument(
        "--track_output_name",
        type=str,
        default="",
        nargs=1,
        help="Base name for saved output files. If empty, it is construct from other inputs."
    )

    # Create a proper "optional arguments" group for help
    # Only add the -h flag if specifically requested
    if add_help_flag:
        optional = parser.add_argument_group("optional arguments")
        parser.safe_add_argument(
            optional,
            "-h", "--help",
            action="help",
            default=argparse.SUPPRESS,
            help="Show this help message and exit."
        )

    return parser


def run_chain(args: argparse.Namespace) -> str:
    """
    Handles file discovery, contour tracking, and saving.
    Returns the path to the generated track file.
    """
    # 1. Data Preparation
    # Discovering images
    args.image_paths = sorted(glob(path.join(f"{args.data_dir}", "*.fits")))

    # 2. Internal logic for missing args
    if not getattr(args, "track_output_name", None) or not args.track_output_name:
        args.track_output_name = get_initial_track_filename(
            data_dir=args.data_dir,
            quantity=args.contour_quantity,
            level=args.contour_level,
        )

    check_dir(args.track_outdir)

    # 3. Execution
    tracks, metadata, events = track_contours(
        image_paths=args.image_paths,
        contour_quantity=args.contour_quantity,
        contour_level=args.contour_level,
        min_area=args.min_area,
        max_gap=args.max_gap,
        iou_threshold=args.iou_threshold,
        registration=args.registration,
    )

    # 4. Saving
    track_file = path.join(args.track_outdir, args.track_output_name)
    save_track_file(
        filename=track_file,
        tracks=tracks,
        metadata=merge_metadata(left=vars(args), right=metadata, placeholders=""),
        events=events,
    )

    return track_file


def main(sent_args: Sequence[str] | None = None) -> None:
    hostname = socket.gethostname()
    print(f"Running on: {hostname}\n")

    parser = get_parser()
    args = parser.parse_args(sent_args)

    inplace_process_nargs1_args(parser, args)
    print_args(args)

    final_output = run_chain(args)

    print(f"\nPipeline Complete!")
    print(f"Final Output: {final_output}")
    farewell()


if __name__ == "__main__":
    main()
