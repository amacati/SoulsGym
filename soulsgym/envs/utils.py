"""Utility functions for the SoulsGym environments."""

from typing import Any, Callable

from gymnasium.error import RetriesExceededError

from soulsgym.exception import SoulsGymException


def max_retries(retries: int = 3) -> Callable:
    """Decorator factory to retry a function `retries` times.

    Args:
        retries: The number of retries.
    """

    def decorator(fn: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for i in range(retries):
                try:
                    return fn(*args, **kwargs)
                except SoulsGymException as e:
                    if i == retries - 1:
                        raise RetriesExceededError(
                            "Function '{}' failed after {} retries".format(fn.__name__, retries)
                        ) from e
                    continue

        return wrapper

    return decorator
