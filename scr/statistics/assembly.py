from os import path

from scr.config.quantities import order_quantities
from scr.utils.filesystem import check_dir
from scr.utils.nested import nested_cast
from scr.io.sunspots import load_sunspot_file, save_sunspot_file
from scr.io.discovery.stats import discover_stat_files


def combine_single_stat_files(
        sunspots_file: str,
        schema_file: str | None = None,
        outdir: str | None = None,
        complete_lengths: int = 6,
        force_completeness: bool = False,
) -> None:
    """
    Combine separate stat files (each containing one or more quantities)
    into a single unified sunspot file.

    Parameters
    ----------
    sunspots_file : str
        Base sunspot file (used for discovery pattern).
    schema_file : str, optional
        Reference file to enforce nested schema consistency.
    outdir : str, optional
        Output directory (defaults to input directory).
    complete_lengths : int
        Expected number of stat files.
    force_completeness : bool
        If True, enforce that all expected files are present.
    """

    files = discover_stat_files(
        sunspots_file,
        complete_lengths=complete_lengths,
        force_completeness=force_completeness,
    )

    if not files:
        print(f"No stat files found for {sunspots_file}")
        return

    indir, base_name = path.split(sunspots_file)
    outdir = outdir or indir
    check_dir(outdir)

    merged_stats = {}
    sunspots = metadata = events = None

    # ---- Load and merge ----
    for i, fname in enumerate(files):
        spts, stats, meta, ev = load_sunspot_file(fname)

        if i == 0:
            sunspots = spts
            metadata = meta
            events = ev

        # Check duplicate quantities
        overlap = set(merged_stats).intersection(stats)
        if overlap:
            raise ValueError(
                f"Duplicate quantities detected across files: {overlap}"
            )

        merged_stats.update(stats)

    # ---- Optional schema enforcement ----
    if schema_file and path.isfile(schema_file):
        ref_tracks, ref_stats, ref_meta, _ = load_sunspot_file(schema_file)

        sunspots = nested_cast(ref_tracks, sunspots)
        merged_stats = nested_cast(ref_stats, merged_stats)
        metadata = nested_cast(ref_meta, metadata)

    # ---- Build output filename dynamically ----
    quantities = order_quantities(list(merged_stats.keys()))
    suffix = "_".join(quantities)
    outname = base_name.replace(".npz", f"_{suffix}.npz")

    save_sunspot_file(
        filename=path.join(outdir, outname),
        sunspots=sunspots,
        stats=merged_stats,
        metadata=metadata,
        events=events,
    )

    print(f"Combined {len(files)} files → {outname}")
