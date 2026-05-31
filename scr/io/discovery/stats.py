from glob import glob


def discover_stat_files(
        sunspots_file: str,
        complete_lengths: int = 6,
        force_completeness: bool = True,
) -> list[str]:
    files = glob(f"{sunspots_file.replace('.npz', '')}*")

    if force_completeness and len(files) != complete_lengths:
        raise IOError(f"Not enough files for {sunspots_file}")

    return sorted(files)
