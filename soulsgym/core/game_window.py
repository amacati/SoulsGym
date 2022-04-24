"""The ``GameWindow`` is primarily a wrapper around the ``mss`` module which enables screen capture.

It also allows us to focus the Dark Souls III application on gym start. We do not provide the images
as observation output as we deem training on the images as too complex for now. Screen capturing
also puts unnecessary additional load on the gym. We note however that the ``GameWindow`` is fully
capable if desired to provide a visual game observation.

Note:
    ``mss`` refers to the `pypi package <https://pypi.org/project/mss/>`_, **not** the conda-forge
    package!
"""
import time
from typing import Callable, Optional

import numpy as np
import mss
import win32gui
import win32api
import win32con
import win32com
import win32com.client


class GameWindow:
    """Provide an interface with the game window.

    The game resolution is particularly critical for the performance of the screen capture since a
    larger game window directly corresponds to larger images and more required computational power
    for processing.
    """

    def __init__(self, processing: Optional[Callable] = None):
        """Initialize the monitor and screen frame grab.

        We offer an optional ``processing`` callable which can be used to transform images from the
        screen capture.

        Args:
            processing: Optional function for raw image processing.
        """
        self._app_id = self._get_ds_app_id()
        self._monitor = self._get_monitor()
        self._sct = mss.mss()
        self._process_fn = processing or self._default_processing

    def screenshot(self, return_raw: bool = False) -> np.ndarray:
        """Fetch the current screen.

        Args:
            return_raw: Option to get the unprocessed frame.

        Returns:
            The current game screenshot.
        """
        raw = np.array(self._sct.grab(self._monitor))
        if return_raw:
            return raw
        return self._process_fn(raw)

    def focus_application(self):
        """Shift the application focus of Windows to the game application.

        Also sets the cursor within the game window.
        """
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')  # Bug fix for shell use with SetForegroundWindow.
        win32gui.SetForegroundWindow(self._app_id)
        time.sleep(0.1)
        left, top, _, _ = win32gui.GetWindowRect(self._app_id)
        win32api.SetCursorPos((left + 100, top + 100))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, left + 100, top + 5, 0, 0)

    def _get_monitor(self) -> dict:
        """Get the window pixel positions.

        Returns:
            A dictionary containing the pixel coordinates of `top`, `left`, `width`, `height`.
        """
        left, top, right, bottom = win32gui.GetWindowRect(self._app_id)
        width = right - left
        height = bottom - top
        monitor = {"top": top + 46, "left": left + 11, "width": width - 22, "height": height - 56}
        return monitor

    def _window_enumeration_handler(self, hwnd: int, top_windows: list):
        """Handle the EnumWindows callback.

        Args:
            hwnd: A handle to a top-level window.
            top_windows: The application-defined value given in EnumWindows.
        """
        top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))

    def _get_ds_app_id(self) -> int:
        """Get the Dark Souls III application ID.

        Returns:
            The app ID.
        """
        top_windows = []
        win32gui.EnumWindows(self._window_enumeration_handler, top_windows)
        ds_app_id = 0
        for apps in top_windows:
            if apps[1] == "DARK SOULS III":
                ds_app_id = apps[0]
                break
        if not ds_app_id:
            raise RuntimeError("It appears DS3 is not open. Please launch the game!")
        return ds_app_id

    @staticmethod
    def _default_processing(raw: np.ndarray) -> np.ndarray:
        """Identity processing function.

        Args:
            raw: Raw input image.

        Returns:
            The unprocessed input image.
        """
        return raw
