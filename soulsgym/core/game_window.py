"""The ``GameWindow`` is primarily a wrapper around the ``mss`` module which enables screen capture.

It also allows us to focus the Dark Souls III application on gym start. We do not provide the images
as observation output as we deem training on the images as too complex for now. Screen capturing
also puts unnecessary additional load on the gym. We note however that the ``GameWindow`` is fully
capable if desired to provide a visual game observation.
"""
from typing import Callable

import numpy as np
import win32gui
import win32ui
import win32con
import cv2


class GameWindow:
    """Provide an interface with the game window.

    The game resolution is particularly critical for the performance of the screen capture since a
    larger game window directly corresponds to larger images and more required computational power
    for processing.
    """
    window_ids = {"DarkSoulsIII": "DARK SOULS III", "EldenRing": "ELDEN RING™"}

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
        self.hwnd = win32gui.FindWindow(None, self.window_ids[game_id])
        if not self.hwnd:
            raise Exception('Window not found: {}'.format(game_id))
        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        self.w = window_rect[2] - window_rect[0]
        self.h = window_rect[3] - window_rect[1]
        # account for the window border and titlebar and cut them off
        self.cropped_x = 8  # Border pixels
        self.cropped_y = 31  # Title bar pixels
        self.w -= self.cropped_x * 2
        self.h -= self.cropped_y + self.cropped_x
        self._process_fn = processing or self._default_processing
        self.img_height = img_height or 90
        self.img_width = img_width or 160

    def screenshot(self, return_raw: bool = False) -> np.ndarray:
        """Fetch the current image from the targeted application.

        Args:
            return_raw: Option to get the unprocessed frame.

        Returns:
            The current game screenshot.
        """
        window_device_context = win32gui.GetWindowDC(self.hwnd)
        device_context = win32ui.CreateDCFromHandle(window_device_context)
        c_device_context = device_context.CreateCompatibleDC()
        data_bit_map = win32ui.CreateBitmap()
        data_bit_map.CreateCompatibleBitmap(device_context, self.w, self.h)
        c_device_context.SelectObject(data_bit_map)
        c_device_context.BitBlt((0, 0), (self.w, self.h), device_context,
                                (self.cropped_x, self.cropped_y), win32con.SRCCOPY)
        img_buffer = data_bit_map.GetBitmapBits(True)
        img = np.frombuffer(img_buffer, dtype='uint8').reshape((self.h, self.w, 4))
        img = img[..., [2, 1, 0]]
        # Free resources
        device_context.DeleteDC()
        c_device_context.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, window_device_context)
        win32gui.DeleteObject(data_bit_map.GetHandle())
        img = np.ascontiguousarray(img)
        if return_raw:
            return img
        return self._process_fn(img)

    def focus_application(self):
        """Shift the application focus of Windows to the game application.

        Also sets the cursor within the game window.
        """
        win32gui.SetForegroundWindow(self.hwnd)

    def _default_processing(self, img: np.ndarray) -> np.ndarray:
        """Default processing function.

        Resizes the input to (img_width, img_heigth).

        Args:
            img: Input image.

        Returns:
            The processed input image.
        """
        return cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
