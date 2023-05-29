import numpy as np

from soulsgym.core.game_window._C.window_capture import WindowCapture as _WindowCapture


class WindowCapture:
    """Wrapper for the C++ WindowCapture class.

    Since it is tedious to compile the C++ code for each platform, and SoulsGym can only be used on
    Windows x64 systems, we directly import the compiled .pyd file. This avoids installing it with
    pybind11 and pip. The wrapper also allows us to include docstraings and type hints.
    """

    def __init__(self):
        """Create an internal WindowCapture object from the C++ bindings."""
        self._window_capture = _WindowCapture()

    def open(self, hwnd: int):
        """Open the window of the given handle.

        When we open the window, an internal frame pool is attached to the window and notifies the
        main thread whenever a new frame is available. Since we don't fetch the actual image, this
        is fast and consumes negligible resources.

        Note: You have to open a window before you can call :meth:`.WindowCapture.get_img`!

        Args:
            hwnd: Handle of the window to open. Note that this is not the process ID, but the window
                handle.
        """
        return self._window_capture.open(hwnd)

    def get_img(self) -> np.ndarray:
        """Get the current image of the window.

        Returns:
            The current image of the window as a numpy array.
        """
        return self._window_capture.get_img()

    def close(self):
        """Close the window and stop the frame pool."""
        return self._window_capture.close()
