from scr.utils.types_alias import Tracks, Events


def relabel_tracks_by_lifetime(
    tracks: Tracks,
    events: Events | None = None,
) -> tuple[Tracks, Events | None]:
    """
    Relabel track IDs sorted by descending lifetime.
    Optionally updates event logs consistently.
    """

    # Sort by descending lifetime
    sorted_items = sorted(
        tracks.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    # Build mapping old_id -> new_id
    id_map = {
        old_id: new_id
        for new_id, (old_id, _) in enumerate(sorted_items)
    }

    # Rebuild tracks
    new_tracks = {
        id_map[old_id]: track
        for old_id, track in tracks.items()
    }

    # Update events if provided
    if events is not None:
        new_events = []

        for e in events:
            e_new = e.copy()

            if e["type"] == "merge":
                e_new["parents"] = [
                    id_map[p] for p in e["parents"] if p in id_map
                ]
                e_new["survivor"] = id_map.get(e["survivor"])

            elif e["type"] == "split":
                e_new["parent"] = id_map.get(e["parent"])
                e_new["children"] = [
                    id_map[c] for c in e["children"] if c in id_map
                ]

            new_events.append(e_new)

        return new_tracks, new_events

    return new_tracks, None
