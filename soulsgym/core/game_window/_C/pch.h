#pragma once
#define NOMINMAX  // Avoid errors with std::max because windows headers define it as a macro at some point (seriously? guys?)

// Needs to come before C++/WinRT headers
#include <wil/cppwinrt.h>

// WinRT
#include <winrt/Windows.Foundation.Metadata.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.Display.h>
#include <winrt/Windows.Graphics.DirectX.h>
#include <winrt/Windows.Graphics.DirectX.Direct3d11.h>

// Interop
#include <windows.graphics.capture.interop.h>

// Windows Imaging Component
#include <wincodec.h>

// STL
#include <atomic>
#include <vector>
#include <stdexcept>

// D3D
#include <d3d11_4.h>
#include <d2d1_3.h>

// WIL
#include <wil/resource.h>
