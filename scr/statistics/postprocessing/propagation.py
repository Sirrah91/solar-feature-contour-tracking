from scr.utils.types_alias import StatsByQuantity, Quantity


def propagate_stat_parameter(
        stats: StatsByQuantity,
        source_quantity: Quantity,
        target_quantities: list[Quantity],
        param: str,
) -> None:
    for obs_id, areas in stats.get(source_quantity, {}).items():
        for area, frames in areas.items():
            for frame_id, params in frames.items():
                if param not in params:
                    continue  # skip if `param` not computed for this frame

                value = params[param]
                for q in target_quantities:
                    stats.setdefault(q, {}) \
                         .setdefault(obs_id, {}) \
                         .setdefault(area, {}) \
                         .setdefault(frame_id, {})[param] = value
