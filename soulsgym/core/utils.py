"""Utility module for ``soulsgym``."""
from __future__ import annotations
from typing import Union, Any
from threading import Lock
from weakref import WeakValueDictionary

import numpy as np


def wrap_to_pi(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Wrap an angle into the interval of [-pi, pi].

    Args:
        x: Single angle or array of angles in radians.

    Returns:
        The wrapped angle.
    """
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class Singleton(type):
    """Metaclass for singletons."""

    # Enables automatic __del__ call after all instances have gone out of scope
    # See https://stackoverflow.com/questions/43619748/destroying-a-singleton-object-in-python
    _instances = WeakValueDictionary()
    _lock = Lock()

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """Create the class if no object is already instantiated, otherwise return the instance."""
        with cls._lock:
            if cls not in cls._instances:
                # This variable declaration is required to force a strong reference on the instance
                instance = super(Singleton, cls).__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]
