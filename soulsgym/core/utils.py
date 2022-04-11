"""Miscellaneous utils functions."""
from __future__ import annotations
from typing import Union
import numpy as np


def wrap_to_pi(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap an angle into the interval of [-pi, pi].

    Args:
        x: Single angle or array of angles in radians.

    Returns:
        The wrapped angle.
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class Singleton(object):
    """Implement a Singleton parent class for inheritance."""

    def __new__(cls) -> Singleton:
        """Create a new class only if singleton has no instance so far, else return instance."""
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton, cls).__new__(cls)
        return cls.instance
