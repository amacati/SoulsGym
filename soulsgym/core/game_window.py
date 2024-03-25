"""The ``GameWindow`` is a wrapper around the ``windows_capture`` library.

The capture mechanism itself is implemented in ``rust`` to enable fast and efficient screen capture.
``GameWindow`` also allows us to focus the Dark Souls III application on gym start.
"""

import platform
import time
from itertools import groupby
from multiprocessing import Event
from typing import Callable

import cv2
import numpy as np

if platform.system() == "Windows":  # Windows imports, ignore for unix to make imports work
    import win32api
    import win32con
    import win32gui
    from windows_capture import Frame, InternalCaptureControl, WindowsCapture


class GameWindow:
    """Provide an interface with the game window.

    The game resolution is particularly critical for the performance of the screen capture since a
    larger game window directly corresponds to larger images and more required computational power
    for processing.
    """

    window_ids = {"DarkSoulsIII": "DARK SOULS III", "EldenRing": "ELDEN RING™"}
    game_resolution = {"DarkSoulsIII": (800, 450), "EldenRing": (800, 450)}
    _initial_timeout = 5

    def __init__(
        self,
        game_id: str,
        processing: Callable | None = None,
        img_height: int | None = None,
        img_width: int | None = None,
    ):
        """Initialize the monitor and screen frame grab.

        We offer an optional ``processing`` callable which can be used to transform images from the
        screen capture.

        Args:
            game_id: The name of the game.
            processing: Optional function for raw image processing.
            img_height: The height of the captured image.
            img_width: The width of the captured image.
        """
        self.game_id = game_id
        self.img_height = img_height or 90
        self.img_width = img_width or 160
        self._process_fn = processing or self._default_processing
        self.hwnd = win32gui.FindWindow(None, self.window_ids[game_id])
        # Configure the windows capture module and wait for the first frame to arrive. If we do not
        # receive a frame within 5 seconds, we raise an error.
        self._latest_frame: Frame | None = None
        self.capture = WindowsCapture(
            cursor_capture=None,
            draw_border=None,
            monitor_index=None,
            window_name=self.window_ids[game_id],
        )
        self._close_capture = Event()

        @self.capture.event
        def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
            self._latest_frame = frame
            if self._close_capture.is_set():
                capture_control.close()

        @self.capture.event
        def on_closed():
            self._latest_frame = None
            raise RuntimeError("Game window capture closed unexpectedly.")

        self.capture.start_free_threaded()
        t_start = time.time()
        while self._latest_frame is None and time.time() - t_start < self._initial_timeout:
            time.sleep(0.1)
        if self._latest_frame is None:
            raise RuntimeError("Failed to capture the game window. Is the game paused?")
        # The image we get from the game capture module game window initially does not match the
        # desired resolution. We therefore determine the necessary crop indices to remove the image
        # padding. See function docs for more details.
        self._crop_heights, self._crop_widths = self._determine_image_crop()

    @property
    def img_resolution(self) -> tuple[int, int]:
        """The resolution of the captured image."""
        return self.img_height, self.img_width

    @img_resolution.setter
    def img_resolution(self, resolution: tuple[int, int]):
        assert len(resolution) == 2, "Image resolution must be a tuple of length 2."
        game_width, game_height = self.game_resolution[self.game_id][::-1]
        assert 0 < resolution[0] <= game_width, f"Image width must be in (0, {game_width}]."
        assert 0 < resolution[1] <= game_height, f"Image height must be in (0, {game_height}]."
        self.img_height, self.img_width = resolution

    @property
    def focused(self) -> bool:
        """Check if the game window is currently focused.

        Returns:
            True if the game window is currently focused, False otherwise.
        """
        return win32gui.GetForegroundWindow() == self.hwnd

    @property
    def img(self) -> np.ndarray:
        """The latest processed game screenshot.

        Returns:
            The processed image.
        """
        assert self._latest_frame is not None, "No frame available."
        return self._process_fn(self._latest_frame.frame_buffer[..., :3])

    @property
    def raw_img(self) -> np.ndarray:
        """The latest raw game screenshot.

        Returns:
            The raw image.
        """
        assert self._latest_frame is not None, "No frame available."
        return self._latest_frame.frame_buffer[..., :3]

    def focus(self):
        """Shift the application focus of Windows to the game application.

        Also sets the cursor within the game window.
        """
        win32gui.SetForegroundWindow(self.hwnd)
        time.sleep(0.1)
        left, top, _, _ = win32gui.GetWindowRect(self.hwnd)
        win32api.SetCursorPos((left + 100, top + 100))
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, left + 100, top + 5, 0, 0)

    def close(self):
        """Close the game window capture."""
        self._close_capture.set()
        self._latest_frame = None

    def _default_processing(self, img: np.ndarray) -> np.ndarray:
        """Default processing function.

        Resizes the input to (img_width, img_heigth).

        Args:
            img: Input image.

        Returns:
            The processed input image.
        """
        img = img[
            self._crop_heights[0] : self._crop_heights[1],
            self._crop_widths[0] : self._crop_widths[1],
        ]
        if img.shape == (self.img_height, self.img_width, 3):
            return img
        return cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)

    def _determine_image_crop(self) -> tuple[np.ndarray, np.ndarray]:
        """Determine the necessary crop to remove the image padding and title bar.

        The image we get from the game capture module game window initially does not match the
        desired resolution. Windows adds a border to the window as well as a title bar we need to
        crop. This border and title bar vary for each user and can be of different sizes. Therefore,
        we need to determine the cropping of the image at runtime for preprocessing.

        Raises:
            RuntimeError: If the crop detection fails due to unexpected window or crop sizes.
        """
        img = self.raw_img
        desired_resolution = self.game_resolution[self.game_id][::-1]
        # Detect all rows and columns in the image that are completely black
        black_pixels = np.all(img == 0, axis=2)
        black_rows, black_columns = (
            np.all(black_pixels == 1, axis=0),
            np.all(black_pixels == 1, axis=1),
        )
        # Generate a list of the lengths of the black pixel rows and columns
        img_column_paddings = [len(list(v)) for k, v in groupby(black_columns) if k == 1]
        img_row_paddings = [len(list(v)) for k, v in groupby(black_rows) if k == 1]
        # The initial values for the crop remove the black padding around the borders of the image
        top_crop = 0 if not black_rows[0] else img_column_paddings[0]
        bottom_crop = img.shape[0] if not black_rows[-1] else img.shape[0] - img_column_paddings[-1]
        left_crop = 0 if not black_columns[0] else img_row_paddings[0]
        right_crop = img.shape[1] if not black_columns[-1] else img.shape[1] - img_row_paddings[-1]
        # Check if the image resolution can still be cropped to the desired resolution and
        res_diff = np.array([bottom_crop - top_crop, right_crop - left_crop]) - desired_resolution
        if np.any(res_diff < 0):
            raise RuntimeError(
                (
                    "Image autocrop failed: Resolution differences are below zero "
                    f"({res_diff}). Please set the game resolution to 800x450."
                )
            )
        if res_diff[0] > 50 or res_diff[1] > 10:
            raise RuntimeError(
                (
                    f"Image resolution difference is too great {res_diff}."
                    "Please set the game resolution to 800x450."
                )
            )
        # Finally, crop the image to the desired resolution. The side difference is distributed
        # equally to the left and right side of the image. The top difference is assigned most of
        # the difference, since it still contains the window title bar that needs to be cropped out.
        left_crop = left_crop + res_diff[1] // 2
        right_crop = right_crop - (res_diff[1] - res_diff[1] // 2)
        if res_diff[1] > 0:
            bottom_crop = bottom_crop - 1
            top_crop = top_crop + res_diff[0] - 1
        return np.array([top_crop, bottom_crop]), np.array([left_crop, right_crop])
