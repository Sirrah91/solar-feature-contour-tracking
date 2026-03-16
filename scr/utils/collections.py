from collections import defaultdict
from typing import Sequence


class NestedDefault:
    """
    Pickleable callable that returns a nested defaultdict.

    `factories` defines the factory at each depth level.
    """

    def __init__(self, factories: Sequence[type]) -> None:
        if not factories:
            raise ValueError("factories must not be empty")
        self.factories = tuple(factories)  # make immutable & pickle-stable

    def __call__(self) -> defaultdict:
        factory = self.factories[0]

        if len(self.factories) == 1:
            return defaultdict(factory)

        return defaultdict(NestedDefault(self.factories[1:]))


def nested_defaultdict(
        *,
        factory: type | Sequence[type],
        depth: int = 1,
) -> defaultdict:
    """
    Create a nested defaultdict.

    Parameters
    ----------
    factory : type or Sequence[type]
        Either:
            - single factory type (repeated `depth` times)
            - sequence of factories, one per depth level

    depth : int
        Used only if `factory` is a single type.

    Examples
    --------
    nested_defaultdict(depth=3, factory=list)
    nested_defaultdict(factory=[dict, dict, list])
    """

    if isinstance(factory, Sequence) and not isinstance(factory, type):
        # sequence of factories
        return NestedDefault(factory)()

    # single factory: intermediate levels are dict, leaf is `factory`
    return NestedDefault([dict] * (depth - 1) + [factory])()
