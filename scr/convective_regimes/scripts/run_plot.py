"""
Generate all paper plots for convective regime analysis.

Example
-------
python -m scr.convective_regimes.scripts.run_plot \\
    --plots regression probability \\
    --phases all stable \\
    --fig_format pdf
"""
from scr.config.env import initialize_environment
initialize_environment()

import socket
from typing import Sequence


from scr.convective_regimes.settings import (
    DATA_DIR, FIG_FORMAT, FIGURE_DIR, SUNSPOTS_PHASES_DIR_TEMPLATE,
)
from scr.devtools.parser import (
    CustomArgumentParser,
    CustomFormatter,
    inplace_process_nargs1_args,
    print_args,
    farewell,
)
from scr.utils.filesystem import check_dir

_ALL_PLOTS = [
    "regression",
    "probability",
    "histograms_2d",
    "histograms_1d",
    "filling",
    "regions_example",
]


def get_parser() -> CustomArgumentParser:
    parser = CustomArgumentParser(
        allow_abbrev=False,
        add_help=True,
        description=(
            "Generate paper plots for convective regime analysis.\n\n"
            "Example:\n"
            "  python run_plot.py --plots all --phases all stable"
        ),
        formatter_class=CustomFormatter,
    )

    what = parser.add_argument_group("plot selection")
    what.add_argument(
        "--plots",
        type=str,
        nargs="+",
        default=["all"],
        choices=_ALL_PLOTS + ["all"],
        help=(
            "Which plots to generate. 'all' runs every plot type."
        ),
    )
    what.add_argument(
        "--phases",
        type=str,
        nargs="+",
        default=["all", "forming", "stable", "decaying"],
        choices=["all", "forming", "stable", "decaying"],
        help=(
            "Phases to plot. Applies to per-phase plots "
            "(regression, probability, histograms_2d, filling)."
        ),
    )

    paths = parser.add_argument_group("path settings")
    paths.add_argument(
        "--data_dir",
        type=str,
        nargs=1,
        default=DATA_DIR,
        help="Directory containing npz/parquet inputs.",
    )
    paths.add_argument(
        "--figure_outdir",
        type=str,
        nargs=1,
        default=FIGURE_DIR,
        help=f"Output directory for figures.",
    )
    paths.add_argument(
        "--sunspot_type",
        type=str,
        nargs=1,
        default="removed",
        choices=["collapsed", "removed"],
        help=(
            "Sunspot type for histograms_1d and regions_example "
            "(determines which phase-track directory is used)."
        ),
    )

    style = parser.add_argument_group("style settings")
    style.add_argument(
        "--fig_format",
        type=str,
        nargs=1,
        default=FIG_FORMAT,
        choices=["pdf", "png", "svg", "eps"],
        help=f"Output figure format.",
    )

    return parser


def run_chain(args) -> None:
    check_dir(args.figure_outdir)

    plots = _ALL_PLOTS if "all" in args.plots else args.plots

    sunspots_phases_dir = SUNSPOTS_PHASES_DIR_TEMPLATE.format(
        sunspot_type=args.sunspot_type
    )

    shared_kw = dict(
        data_dir=args.data_dir,
        figure_outdir=args.figure_outdir,
        fig_format=args.fig_format,
    )

    if "regions_example" in plots:
        from scr.convective_regimes.plotting.regions_example import plot_regions_example
        print("Plotting: regions_example")
        plot_regions_example(
            sunspot_type=args.sunspot_type,
            sunspots_phases_dir=sunspots_phases_dir,
            figure_outdir=args.figure_outdir,
            fig_format=args.fig_format,
        )

    if "histograms_1d" in plots:
        from scr.convective_regimes.plotting.histograms import plot_1d_histograms
        print("Plotting: histograms_1d")
        plot_1d_histograms(
            sunspot_type=args.sunspot_type,
            sunspots_phases_dir=sunspots_phases_dir,
            figure_outdir=args.figure_outdir,
            fig_format=args.fig_format,
        )

    per_phase_plots = {
        "regression": "scr.convective_regimes.plotting.regression",
        "probability": "scr.convective_regimes.plotting.probability",
        "histograms_2d": "scr.convective_regimes.plotting.histograms",
        "filling": "scr.convective_regimes.plotting.filling",
    }
    per_phase_funcs = {
        "regression": "plot_regression",
        "probability": "plot_probability_counts",
        "histograms_2d": "plot_2d_histograms",
        "filling": "plot_flux_in_target_region",
    }

    for plot_name, module_path in per_phase_plots.items():
        if plot_name not in plots:
            continue
        import importlib
        module = importlib.import_module(module_path)
        func = getattr(module, per_phase_funcs[plot_name])

        for phase in args.phases:
            print(f"Plotting: {plot_name} / {phase}")
            func(phase, **shared_kw)


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
