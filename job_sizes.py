# from scr.config.env import initialize_environment
# initialize_environment()

import argparse
from os import path
from glob import glob
from typing import Sequence
import numpy as np

from scr.devtools.parser import (
    CustomArgumentParser,
    CustomFormatter,
    inplace_process_nargs1_args,
)


def get_parser(
        parser: CustomArgumentParser | None = None,
        add_help_flag: bool = True
) -> CustomArgumentParser:
    if parser is None:
        parser = CustomArgumentParser(
            allow_abbrev=False,
            add_help=False,
            description="Calculate memory requirements to process sunspot contours.\n\n"
                        "Example:\n"
                        "  python job_sizes.py --data_dir /path/to/data",
            formatter_class=CustomFormatter
        )

    # Positional arguments
    positional = parser.add_argument_group("positional arguments")

    # Core options
    core = parser.add_argument_group("core settings")
    core.add_argument(
        "--data_dir",
        type=str,
        required=True,
        nargs=1,
        help="Directory containing input files."
    )
    core.add_argument(
        "--parse_quantities",
        action="store_true",
        help="Enable parsing of filename dots to determine quantity count."
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


def run_chain(args: argparse.Namespace) -> float:
    """
    Handles memory estimate.
    Returns the job size.
    """
    # 1. Determine if we are looking at a file or a directory
    if path.isfile(args.data_dir):
        image_filenames = [args.data_dir]
        is_single_file = True
    else:
        image_filenames = glob(path.join(args.data_dir, "*"))
        is_single_file = False

    total_normalized_bytes = 0.

    for filename in image_filenames:
        if not path.isfile(filename):
            continue

        file_size = path.getsize(filename)

        n_quant = 1  # Default to "single unit" behavior

        # Only parse if requested AND it's not a single file override
        if args.parse_quantities and not is_single_file:
            bare_filename = path.basename(filename)
            parts = bare_filename.split(".")

            if len(parts) >= 2:
                # e.g., 'data.ABC.fits' -> parts[-2] is 'ABC' -> len is 3
                n_quant = max(1, len(parts[-2]))

        total_normalized_bytes += file_size / n_quant

    mem_gb_per_quantity = total_normalized_bytes / (1024.0 ** 3)

    # print usage for qsub
    print(f"{np.clip(np.ceil(mem_gb_per_quantity * 1.5), a_min=1., a_max=None):.0f}")

    return mem_gb_per_quantity


def main(sent_args: Sequence[str] | None = None) -> None:
    parser = get_parser()
    args, _ = parser.parse_known_args(sent_args)

    inplace_process_nargs1_args(parser, args)

    run_chain(args)


if __name__ == "__main__":
    main()
