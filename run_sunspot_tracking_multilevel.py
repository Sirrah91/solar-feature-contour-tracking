from scr.config.env import initialize_environment
initialize_environment()

import argparse
import socket
from os import path
from glob import glob
from typing import Sequence

from scr.pipelines.processing.build_sunspots_from_levels import track_and_merge_sunspots

from scr.config.paths import PATH_SUNSPOTS
from scr.utils.filesystem import check_dir
from scr.utils.naming import (
    get_initial_track_filename,
    track_to_sunspot_filename,
)
from scr.utils.metadata import merge_metadata
from scr.io.sunspots import save_sunspot_file
from scr.tracks.labels import tracks_label

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
                "Extract and track contours from AR data based on user-defined thresholds and conditions.\n\n"
                "Example:\n"
                "  python run_contour_tracking.py "
                "--data_dir /path/to/fits "
                "--contour_quantity Ic "
                "--component 0.9 3 "
                "--component 0.65 3 "
                "--component 0.5\n"
            ),
            formatter_class=CustomFormatter
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

    # Contour extraction settings
    components = parser.add_argument_group("component extraction")
    components.add_argument(
        "--component",
        metavar="COMPONENT",
        nargs="+",
        action="append",
        required=True,
        help=(
            "Define a component to associate with outer contours.\n"
            "Two possible syntaxes:\n"
            "  1) LEVEL [NAME]            -> e.g., 0.5 inner\n"
            "  2) LEVEL [MIN_AREA] -> e.g., 0.65 3\n"
            "  3) LEVEL [MIN_AREA] [NAME] -> e.g., 0.65 3 middle\n"
            "Arguments:\n"
            "  LEVEL      (float)  Required: contour threshold level.\n"
            "  MIN_AREA   (float)  Optional: minimum area (default: 0.0).\n"
            "  NAME       (str)    Optional: component name (default: 'quantity-LEVEL').\n"
            "You can repeat --component multiple times to define multiple components.\n"
        ),
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
        "--min_frames",
        type=int,
        default=0,
        nargs=1,
        help="Minimum lifetime (in frames) of outer tracks.",
    )
    filtering.add_argument(
        "--collapse_nested",
        action="store_true",
        help="Collapse nested contours.",
    )
    filtering.add_argument(
        "--remove_nested",
        action="store_true",
        help="Remove nested frames from outer tracks.",
    )

    filtering.add_argument(
        "--min_vertices",
        type=int,
        default=4,
        help="Minimum number of vertices a contour must have to be kept. "
             "Contours with fewer points are discarded.",
    )
    filtering.add_argument(
        "--max_healing_gap",
        type=float,
        default=0.0,
        help="Maximum distance (in pixels) between first and last contour point "
             "to automatically 'snap' them together if the contour is almost closed. "
             "Set >0 to heal minor numerical gaps.",
    )
    filtering.add_argument(
        "--max_closing_gap",
        type=float,
        default=0.0,
        help="Maximum distance (in pixels) between first and last contour point "
             "to forcibly close the contour by appending the first point. "
             "Use this for incomplete contours.",
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

    # Containment
    containment = parser.add_argument_group("containment settings")
    containment.add_argument(
        "--containment_mode",
        type=str,
        choices=["strict", "covers", "robust"],
        default="covers",
        nargs=1,
        help="Containment criterion for component association.",
    )
    containment.add_argument(
        "--min_containment",
        type=float,
        default=0.8,
        nargs=1,
        help="Minimum containment fraction (robust mode only).",
    )

    # Output options
    output = parser.add_argument_group("output options")
    output.add_argument(
        "--sunspot_outdir",
        type=str,
        default=PATH_SUNSPOTS,
        nargs=1,
        help="Output directory.",
    )
    output.add_argument(
        "--sunspot_output_name",
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
    Handles file discovery, contour tracking, tracks-to-sunspot association, and saving.
    Returns the path to the generated sunspot file.
    """
    # 1. Data Preparation
    # Discovering images
    args.image_paths = sorted(glob(path.join(f"{args.data_dir}", "*.fits")))
    args.inner_contour_quantity = args.contour_quantity  # metadata consistency

    # Assemble components configuration
    components_cfg = []

    for i, entry in enumerate(args.component):
        if len(entry) < 1:
            raise ValueError("--component requires at least LEVEL")

        level = float(entry[0])

        if len(entry) == 1:
            min_area = 0.0
            name = tracks_label(quantity=args.inner_contour_quantity, level=level)

        elif len(entry) == 2:
            # Detect if 2nd arg is numeric (MIN_AREA) or string (NAME)
            try:
                min_area = float(entry[1])
                name = tracks_label(quantity=args.inner_contour_quantity, level=level)
            except ValueError:
                min_area = 0.0
                name = entry[1]
        else:
            min_area = float(entry[1])
            name = entry[2]

        if i == 0:
            args.contour_level = level  # file naming
            args.min_area = min_area  # metadata consistency

        components_cfg.append({
            "name": name,
            "level": level,
            "min_area": min_area,
        })
    args.components = components_cfg

    # Assemble filtering configuration (always present)
    filtering_cfg = {
        "min_frames": args.min_frames,
        "collapse_nested": args.collapse_nested,
        "remove_nested": args.remove_nested,
        "min_vertices": args.min_vertices,
        "max_healing_gap": args.max_healing_gap,
        "max_closing_gap": args.max_closing_gap,
    }
    args.filtering = filtering_cfg

    # 2. Internal logic for missing args
    if not getattr(args, "sunspot_output_name", None) or not args.sunspot_output_name:
        args.sunspot_output_name = track_to_sunspot_filename(
            get_initial_track_filename(
                data_dir=args.data_dir,
                quantity=args.contour_quantity,
                level=args.contour_level,
            )
        )

    check_dir(args.sunspot_outdir)

    # 3. Execution
    sunspots, metadata, events = track_and_merge_sunspots(
        image_paths=args.image_paths,
        contour_quantity=args.contour_quantity,
        components=args.components,
        max_gap=args.max_gap,
        iou_threshold=args.iou_threshold,
        registration=args.registration,
        filtering=args.filtering,
        containment_mode=args.containment_mode,
        min_containment=args.min_containment,
    )

    # 4. Saving
    sunspot_file = path.join(args.sunspot_outdir, args.sunspot_output_name)
    save_sunspot_file(
        filename=sunspot_file,
        sunspots=sunspots,
        metadata=merge_metadata(left=vars(args), right=metadata, placeholders=""),
        events=events,
    )

    return sunspot_file


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
