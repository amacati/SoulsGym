"""Utility module for ``soulsgym``."""
from __future__ import annotations
from typing import Union, Any
from threading import Lock
from weakref import WeakValueDictionary
import win32gui
import psutil

import numpy as np


def get_window_id(app_name: str) -> int:
    """Get an application window ID.

    Args:
        app_name: The name of the application.

    Returns:
        The application's window ID.
    """
    apps = []
    _app_name = app_name.lower().replace(" ", "")
    # Lambda function handles the EnumWindows callback and appends the app ID and name to the list
    win32gui.EnumWindows(lambda hwnd, apps: apps.append((hwnd, win32gui.GetWindowText(hwnd))), apps)
    for app in apps:
        if app[1].lower().replace(" ", "") == _app_name:
            return app[0]
    raise RuntimeError(f"It appears {app_name} is not open. Please launch the game!")


def get_pid(app_name: str) -> int:
    """Get an application process ID.

    Args:
        app_name: The name of the application.

    Returns:
        The application's process ID.
    """
    _app_name = app_name.lower().replace(" ", "")
    for proc in psutil.process_iter():
        if proc.name().lower().rstrip(".exe") == _app_name:
            return proc.pid
    raise RuntimeError(f"It appears {app_name} is not open. Please launch the game!")


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
