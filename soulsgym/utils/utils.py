"""Miscellaneous utils functions."""
from __future__ import annotations
from typing import Union
import numpy as np


def distance(coords_a: list, coords_b: list, flat: bool = True) -> np.ndarray:
    """Calculate the distance between two positions.

    Args:
        coords_a: First position.
        coords_b: Second position.
        flat: Toggle between 3D and projected 2D distance.

    Returns:
        The distance in meters.
    """
    coords_a, coords_b = np.array(coords_a), np.array(coords_b)
    dim = 2 if flat else 3
    return np.linalg.norm(coords_a[0:dim] - coords_b[0:dim])


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
