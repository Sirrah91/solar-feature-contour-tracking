import argparse
import sys
from pprint import pprint


class CustomArgumentParser(
    argparse.ArgumentParser
):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\nUse 'python {self.prog} --help' to see available options.\n")

    def safe_add_argument(self, group, *args, **kwargs):
        """
        Adds an argument to a specific group ONLY if it doesn't exist
        in the global parser yet.
        """
        # Check if any of the flags (e.g., '-o', '--outdir') already exist
        if not any(flag in self._option_string_actions for flag in args):
            return group.add_argument(*args, **kwargs)
        return None  # Skip if already exists


class CustomFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter
):
    def _get_default_metavar_for_optional(self, action):
        return action.dest

    def _get_help_string(self, action) -> str:
        help_str = action.help
        if "%(default)" not in help_str and action.default is not argparse.SUPPRESS:
            default = action.default
            if isinstance(default, list | tuple):
                default = " ".join(str(x) for x in default)
            return f"{help_str} (default: {default})"
        return help_str


def inplace_process_nargs1_args(
        parser: argparse.ArgumentParser,
        args: argparse.Namespace
) -> None:
    """
    Function to process arguments with nargs=1.
    Converts nargs=1 arguments from lists to single values.
    """
    for action in parser._actions:
        if hasattr(args, action.dest) and action.nargs == 1:
            value = getattr(args, action.dest)
            if isinstance(value, list):
                setattr(args, action.dest, value[0])


def print_args(
        args: argparse.Namespace
) -> None:
    pprint(vars(args))
    print("")


def farewell() -> None:
    print("\nAll done, bye.")
