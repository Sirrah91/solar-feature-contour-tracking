from scr.config.env import initialize_environment
initialize_environment()

import argparse
import socket
from os import path
from typing import Sequence

from scr.pipelines.processing.stats_computation import compute_stats_from_sunspots

from scr.config.paths import PATH_SUNSPOTS_STATS
from scr.utils.filesystem import check_dir
from scr.utils.naming import sunspot_to_stats_filename
from scr.utils.metadata import merge_metadata
from scr.io.sunspots import save_sunspot_file

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
                "Recompute statistics.\n\n"
                "Example:\n"
                "  python run_calc_stats.py "
                "--sunspot_input_path /path/to/sunspots/sunspot_file.npz "
                "--quantities Ic "
                "--stat_types sunspots"),
            formatter_class=CustomFormatter
        )

    # Positional arguments
    positional = parser.add_argument_group("positional arguments")

    # Input options
    input = parser.add_argument_group("input settings")
    input.add_argument(
        "--sunspot_input_path",
        type=str,
        required=True,
        nargs=1,
        help="Input file with tracked sunspots (.npz).",
    )

    # Quantity options
    quantity = parser.add_argument_group("quantity settings")
    quantity.add_argument(
        "--quantities",
        type=str,
        choices=["Ic", "B", "Bp", "Bt", "Br", "Bhor"],
        required=True,
        nargs="+",
        help="Quantities used to compute the statistics.",
    )

    # Contour settings
    contour = parser.add_argument_group("contour sampling")
    contour.add_argument(
        "--max_vertex_spacing",
        type=float,
        default=0.5,
        nargs=1,
        help=(
            "Maximum allowed distance (in pixels) between adjacent contour vertices. "
            "Edges longer than this value will be equidistantly resampled to ensure "
            "high-resolution sampling of underlying map data."
        )
    )

    # Header options
    header = parser.add_argument_group("header options")
    header.add_argument(
        "--header_index",
        type=int,
        default=0,
        nargs=1,
        help="Header index to use for the foreshortening correction.",
    )

    # Output options
    output = parser.add_argument_group("output options")
    output.add_argument(
        "--stats_outdir",
        type=str,
        default=PATH_SUNSPOTS_STATS,
        nargs=1,
        help="Output directory for saving results.",
    )
    output.add_argument(
        "--stats_output_name",
        type=str,
        default="",
        nargs=1,
        help="Base name for saved output files. If empty, it is construct from other inputs.",
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
    Handles statistics computation and saving.
    Returns the path to the generated sunspot-stat file.
    """
    # 1. Internal logic for missing args
    if not getattr(args, "stats_output_name", None) or not args.stats_output_name:
        args.stats_output_name = sunspot_to_stats_filename(
            sunspot_path=args.sunspot_input_path,
            quantities=args.quantities,
        )

    check_dir(args.stats_outdir)

    # 2. Execution
    sunspots, stats, metadata, events = compute_stats_from_sunspots(
        sunspot_path=args.sunspot_input_path,
        quantities=args.quantities,
        header_index=args.header_index,
        max_vertex_spacing=args.max_vertex_spacing,
    )

    # 3. Saving
    sunspot_stat_file = path.join(args.stats_outdir, args.stats_output_name)
    save_sunspot_file(
        filename=sunspot_stat_file,
        sunspots=sunspots,
        stats=stats,
        metadata=merge_metadata(left=vars(args), right=metadata, placeholders=""),
        events=events
    )

    return sunspot_stat_file


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
