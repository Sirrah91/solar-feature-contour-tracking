from scr.utils.types_alias import Events


def prune_events_after_filtering(
        events: Events,
        remaining_ids: set[int],
) -> Events:
    """
    Keep event unchanged if at least one referenced ID still exists.

    Historical information inside the event is preserved.
    """

    pruned = []

    for e in events:

        if e["type"] == "merge":

            keep = (
                any(p in remaining_ids for p in e["parents"])
                or e["survivor"] in remaining_ids
            )

            if keep:
                pruned.append(e)

        elif e["type"] == "split":

            keep = (
                e["parent"] in remaining_ids
                or any(c in remaining_ids for c in e["children"])
            )

            if keep:
                pruned.append(e)

        else:
            # Unknown event type → keep defensively
            pruned.append(e)

    return pruned
