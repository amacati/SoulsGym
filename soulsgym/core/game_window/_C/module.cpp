#include "pch.h"
#include "window_capture.h"
#include <pybind11-global/pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(window_capture, m) {

    py::class_<WindowCapture>(m, "WindowCapture", R"pbdoc(
        Window capture class using DirectX to efficiently grab the current image from a Windows window.

        The class is a Python binding to the actual C++ implementation.
    )pbdoc")
        .def(py::init<>())
        .def("open", &WindowCapture::open, R"pbdoc(
        Open a window and start the image signaling threads.
    )pbdoc")
        .def("get_img", &WindowCapture::get_img, py::return_value_policy::reference_internal, R"pbdoc(
        Get the current image of the window as array.
    )pbdoc")
        .def("close", &WindowCapture::close, R"pbdoc(
        Close the current window.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
