import numpy as np


def distance(coords_a, coords_b, flat=True):
    coords_a, coords_b = np.array(coords_a), np.array(coords_b)
    dim = 2 if flat else 3
    return np.linalg.norm(coords_a[0:dim] - coords_b[0:dim])
