from collections.abc import Callable, Iterator

from .fineweb2 import fineweb2

Source = Callable[..., Iterator[str]]

REGISTRY: dict[str, Source] = {
    "fineweb2": fineweb2
}
