"""The ``GameWindow`` is a wrapper around the ``window_capture`` submodule for screen capture.

Window capture itself is implemented in C++ to enable fast and efficient screen capture.
``GameWindow`` also allows us to focus the Dark Souls III application on gym start.
"""
from typing import Callable
import time

import numpy as np
import cv2
import win32gui
import win32api
import win32con

from soulsgym.core.game_window.window_capture import WindowCapture
from soulsgym.exception import InvalidGameSettings


class GameWindow:
    """Provide an interface with the game window.

    The game resolution is particularly critical for the performance of the screen capture since a
    larger game window directly corresponds to larger images and more required computational power
    for processing.
    """
    window_ids = {"DarkSoulsIII": "DARK SOULS III", "EldenRing": "ELDEN RING™"}
    game_resolution = {"DarkSoulsIII": (800, 450), "EldenRing": (800, 450)}

    def __init__(self,
                 game_id: str,
                 processing: Callable | None = None,
                 img_height: int | None = None,
                 img_width: int | None = None):
        """Initialize the monitor and screen frame grab.

        We offer an optional ``processing`` callable which can be used to transform images from the
        screen capture.

        Args:
            game_id: The name of the game.
            processing: Optional function for raw image processing.
        """
        self.game_id = game_id
        self.img_height = img_height or 90
        self.img_width = img_width or 160
        self._process_fn = processing or self._default_processing
        # Initialize the window capture
        self.hwnd = win32gui.FindWindow(None, self.window_ids[game_id])
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(game_id))
        self._window_capture = WindowCapture()
        self._window_capture.open(self.hwnd)
        # Check if the game has the expected resolution
        self._crop_height = None
        self._crop_width = None
        self._check_resolution()  # Also sets the crop height and width

    @property
    def img_resolution(self) -> tuple[int, int]:
        return self.img_height, self.img_width

    @img_resolution.setter
    def img_resolution(self, resolution: tuple[int, int]):
        assert len(resolution) == 2, "Image resolution must be a tuple of length 2."
        game_width, game_height = self.game_resolution[self.game_id][::-1]
        assert 0 < resolution[0] <= game_width, f"Image width must be in (0, {game_width}]."
        assert 0 < resolution[1] <= game_height, f"Image height must be in (0, {game_height}]."
        self.img_height, self.img_width = resolution

    def get_img(self, return_raw: bool = False) -> np.ndarray:
        """Fetch the current image from the targeted application.

        Args:
            return_raw: Option to get the unprocessed frame.

        Returns:
            The current game screenshot.
        """
        img = self._window_capture.get_img()
        if return_raw:
            return img
        return self._process_fn(img)

    def focus_application(self):
        """Shift the application focus of Windows to the game application.

        Also sets the cursor within the game window.
        """
        win32gui.SetForegroundWindow(self.hwnd)
        time.sleep(0.1)
        left, top, _, _ = win32gui.GetWindowRect(self.hwnd)
        win32api.SetCursorPos((left + 100, top + 100))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, left + 100, top + 5, 0, 0)

    def _default_processing(self, img: np.ndarray) -> np.ndarray:
        """Default processing function.

        Resizes the input to (img_width, img_heigth).

        Args:
            img: Input image.

        Returns:
            The processed input image.
        """
        img = img[self._crop_height[0]:self._crop_height[1],
                  self._crop_widths[0]:self._crop_widths[1]]
        if img.shape == (self.img_height, self.img_width, 3):
            return img
        return cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)

    def _check_resolution(self):
        """Check if the game has the expected resolution.

        Raises:
            InvalidGameSettings: If the game has not the expected resolution.
        """
        img = self.get_img(return_raw=True)
        resolution = self.game_resolution[self.game_id][::-1]  # Numpy img dimensions are reversed
        # The resolution can differ by a few pixels due to the window border. Height deviates more
        # because of the title bar.
        if img.shape[0] - resolution[0] > 50 or img.shape[1] - resolution[1] > 20:
            raise InvalidGameSettings("Game resolution does not match: {}x{} vs {}x{}".format(
                img.shape[0], img.shape[1], resolution[0], resolution[1]))
        if img.shape[0] < resolution[0] or img.shape[1] < resolution[1]:
            raise InvalidGameSettings("Desired resolution too big: {}x{} vs {}x{}".format(
                img.shape[0], img.shape[1], resolution[0], resolution[1]))
        self._crop_height = [img.shape[0] - resolution[0], img.shape[0]]
        if self._crop_height[0] >= 2:  # Usually the bottom border is 1-2 pixels wide
            self._crop_height = [self._crop_height[0] - 1, self._crop_height[1] - 1]
        # Crop both sides of the window equally
        dw = img.shape[1] - resolution[1]
        self._crop_widths = [round(dw / 2), img.shape[1] - (dw - round(dw / 2))]
