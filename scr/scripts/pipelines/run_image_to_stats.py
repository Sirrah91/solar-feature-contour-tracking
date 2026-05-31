from scr.config.env import initialize_environment
initialize_environment()

import sys
import socket
import argparse
from typing import Sequence

# Import your atomic pipeline modules
import run_contour_tracking as step1
import run_sunspot_association as step2
import run_calc_stats as step3

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
    """
    Builds the unified parser by merging all stage-specific parsers.
    """
    if parser is None:
        parser = CustomArgumentParser(
            allow_abbrev=False,
            add_help=False,
            description="Master Pipeline: Images -> Tracks -> Sunspots -> Stats",
            formatter_class=CustomFormatter,
        )

    # Merge parsers from sub-steps (silencing their individual help flags)
    parser = step1.get_parser(parser, add_help_flag=False)
    parser = step2.get_parser(parser, add_help_flag=False)
    parser = step3.get_parser(parser, add_help_flag=False)

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
    Executes the three-stage pipeline sequentially using a single Namespace.
    """
    # --- STAGE 1: CONTOUR TRACKING ---
    print("\n>>> STAGE 1: CONTOUR TRACKING")
    track_file_path = step1.run_chain(args)
    print(f"Generated Track File: {track_file_path}")

    # --- STAGE 2: SUNSPOT ASSOCIATION ---
    print("\n>>> STAGE 2: SUNSPOT ASSOCIATION")
    args.track_input_path = track_file_path  # Ensure this matches STAGE 1 output
    sunspot_file_path = step2.run_chain(args)
    print(f"Generated Sunspot File: {sunspot_file_path}")

    # --- STAGE 3: STATISTICS COMPUTATION ---
    print("\n>>> STAGE 3: STATISTICS COMPUTATION")
    args.sunspot_input_path = sunspot_file_path  # Ensure this matches STAGE 2 output
    final_file = step3.run_chain(args)
    print(f"Generated Statistics File: {final_file}")

    return final_file


def launch():
    """
    Pre-processes raw terminal input to auto-calculate required paths.
    """
    cmd = sys.argv[1:]

    # If user asks for help, don't try to calculate paths
    if "-h" in cmd or "--help" in cmd:
        main(cmd)
        return

    # Inject PLACEHOLDER paths into the command list if not already provided
    if "--track_input_path" not in cmd:
        cmd.extend(["--track_input_path", ""])
    if "--sunspot_input_path" not in cmd:
        cmd.extend(["--sunspot_input_path", ""])

    main(cmd)


def main(sent_args: Sequence[str] | None = None):
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
    launch()
