#include "pch.h"
#include "window_capture.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(window_capture, m)
{

    py::class_<WindowCapture>(m, "WindowCapture", R"pbdoc(
        Window capture class using DirectX to efficiently grab the current image from a Windows window.

        The main functionalities are exposed with Pybind11 to Python.
    )pbdoc")
        .def(py::init<>())
        .def("open", &WindowCapture::open, R"pbdoc(
        Open the window of a process handle and start the image signaling threads.

        When we open the window, an internal frame pool is attached to the window and notifies the
        main thread whenever a new frame is available. Since we don't fetch the actual image, this
        is fast and consumes negligible resources.

        Note: You have to open a window before you can call :meth:`.WindowCapture.get_img`!

        Args:
            hwnd: Handle of the window to open. Note that this is not the process ID, but the window
                handle.
    )pbdoc")
        .def("get_img", &WindowCapture::get_img, py::return_value_policy::reference_internal, R"pbdoc(
        Get the current image of the window as array.

        Returns:
            The current image of the window as a numpy array.
    )pbdoc")
        .def("close", &WindowCapture::close, R"pbdoc(
        Close the current window and stop the frame pool.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
