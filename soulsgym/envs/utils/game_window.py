import time
from typing import Callable
import numpy as np
import mss
import cv2
import win32gui, win32api, win32con, win32com, win32com.client


class GameWindow:
    """
    This class manages interactions with the game window.
    """

    def __init__(self, processing: Callable = None):
        """
        Initializes the monitor and screen frame grab.

        Args:
            processing: Optional function for raw image processing.
        """
        self._app_id = self._get_ds_app_id()
        self._monitor = self._get_monitor()
        self._sct = mss.mss()
        if processing is None:
            self._process_func = self._default_processing
        else:
            self._process_func = processing

    def screenshot(self, return_raw: bool = False) -> np.ndarray:
        """
        Fetches the current screen.

        Args:
            return_raw: Option to get the unprocessed frame.

        Returns:
            The current game screenshot.
        """
        raw = np.array(self._sct.grab(self._monitor))
        if return_raw:
            return raw
        return self._process_func(raw)

    def focus_application(self) -> None:
        """
        Shifts the application focus of windows to the game application.

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
        left, top, right, bottom = win32gui.GetWindowRect(self._app_id)
        width = right - left
        height = bottom - top
        monitor = {"top": top + 46, "left": left + 11, "width": width - 22, "height": height - 56}
        return monitor

    def _window_enumeration_handler(self, hwnd, top_windows):
        top_windows.append((hwnd, win32gui.GetWindowText(hwnd)))

    def _get_ds_app_id(self) -> int:
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
        grey = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        small = cv2.resize(grey, (400, 225))
        return small
