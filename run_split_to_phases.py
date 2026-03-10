from scr.config.env import initialize_environment
initialize_environment()

import argparse
import socket
from os import path
from glob import glob
from typing import Sequence

from scr.pipelines.processing.split_to_phases import compute_phase_split

from scr.config.paths import (
    PATH_SUNSPOTS_PHASES,
    SLOPES_BASENAME,
)
from scr.utils.filesystem import check_dir
from scr.utils.naming import get_sunspot_phases_stem, ensure_extension
from scr.io.npz import save_npz
from scr.io.parquet import save_parquet

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
                "Split track contours to evolutionary phases.\n\n"
                "Example:\n"
                "  python run_split_to_phases.py "
                "--stats_dir /path/to/sunspots_stats"
            ),
            formatter_class=CustomFormatter
        )

    # Positional arguments
    positional = parser.add_argument_group("positional arguments")

    # Input options
    input = parser.add_argument_group("input settings")
    input.add_argument(
        "--stats_dir",
        type=str,
        required=True,
        nargs=1,
        help="Directory containing tracked sunspots with statistics (.npz files)."
    )

    # Slope options
    slope_opts = parser.add_argument_group("slope arguments")
    slope_opts.add_argument(
        "--slope_filename",
        type=str,
        default=SLOPES_BASENAME,
        nargs=1,
        help="File path for precomputed slopes (absolute, or relative to --phases_outdir)."
    )
    slope_opts.add_argument(
        "--collect_new_slopes",
        action="store_true",
        help="Flag to trigger new slope computation."
    )

    # Output options
    output = parser.add_argument_group("output options")
    output.add_argument(
        "--phases_outdir",
        type=str,
        default=PATH_SUNSPOTS_PHASES,
        nargs=1,
        help="Output directory for saving results."
    )
    output.add_argument(
        "--phases_output_name",
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


def run_chain(args: argparse.Namespace) -> tuple[str, str]:
    """
    Handles splitting sunspots to evolutionary phases and saving.
    Returns the paths to the generated sunspot-phases files.
    """
    # 1. Internal logic for missing args
    if not getattr(args, "phases_output_name", None) or not args.phases_output_name:
        args.phases_output_name = get_sunspot_phases_stem()

    check_dir(args.phases_outdir)

    # 2. Execution
    sunspots_phases, combined_df = compute_phase_split(
        stats_paths=sorted(glob(path.join(args.stats_dir, "*.npz"))),
        slope_path=path.join(args.phases_outdir, args.slope_filename),
        collect_new_slopes=args.collect_new_slopes,
    )

    # 3. Saving
    outfile = path.join(args.phases_outdir, args.phases_output_name)
    npz_file = ensure_extension(outfile, extension=".npz")
    parquet_file = ensure_extension(outfile, extension=".parquet")

    save_npz(filename=npz_file, sunspots_phases=sunspots_phases)
    save_parquet(filename=parquet_file, df=combined_df)

    return npz_file, parquet_file


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
