"""Miscellaneous utils functions."""
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
