from os import path
from glob import glob

from scr.config.paths import PATH_SUNSPOTS
from scr.utils.types_alias import Quantity
from scr.io.sunspots import load_sunspot_file, save_sunspot_file

from scr.statistics.pipeline.evolution import compute_sunspot_statistics_evolution
from scr.statistics.postprocessing.propagation import propagate_stat_parameter


def main() -> None:
    QUANTITIY: Quantity = "Bhor"
    PROPAGATED_PARAMETER = "fractal_dimensions"

    sunspots_files = sorted(glob(path.join(PATH_SUNSPOTS, "*.npz")))

    for sunspots_file in sunspots_files:
        sunspots, stats, metadata, events = load_sunspot_file(sunspots_file)

        image_paths = metadata["image_paths"]

        stats = compute_sunspot_statistics_evolution(
            sunspots=sunspots,
            image_paths=image_paths,
            quantities=[QUANTITIY],
            max_vertex_spacing=0.5,
            header_index=0,
        )

        propagate_stat_parameter(
            stats,
            source_quantity=QUANTITIY,
            target_quantities=["Ic", "B", "Bp", "Bt", "Br", "Bhor"],
            param=PROPAGATED_PARAMETER,
        )

        save_sunspot_file(
            filename=sunspots_file.replace(".npz", "_PROPAGATED.npz"),
            sunspots=sunspots,
            stats=stats,
            metadata=metadata,
            events=events,
        )
