/* Largely based on https://github.com/robmikh/SimpleRecorder.
 */
#pragma once
#include "pch.h"
#include "utils.h"
#include "pybind11-global/pybind11/pybind11.h"
#include "pybind11-global/pybind11/numpy.h"

namespace winrt
{
    using namespace Windows::Graphics::Capture;
    using namespace Windows::Graphics::DirectX;
    using namespace Windows::Graphics::Display;
    using namespace Windows::Graphics::DirectX::Direct3D11;
}

namespace py = pybind11;

class WindowCapture
{
public:
    WindowCapture();

    void open(int window_handle);

    void close();

    py::array_t<UINT8> get_img();

    ~WindowCapture();

private:
    winrt::com_ptr<ID3D11Texture2D> get_texture();
    void on_frame_arrived(winrt::Direct3D11CaptureFramePool const &framePool, winrt::Windows::Foundation::IInspectable const &);

    std::atomic<bool> m_is_open = false;
    winrt::IDirect3DDevice m_device = {nullptr};
    winrt::GraphicsCaptureItem m_gc_item = {nullptr};
    winrt::com_ptr<ID3D11Device> m_d3d_device = {nullptr};
    winrt::com_ptr<ID3D11DeviceContext> m_d3d_context{nullptr};

    winrt::Direct3D11CaptureFramePool m_frame_pool{nullptr};
    winrt::GraphicsCaptureSession m_session{nullptr};

    wil::shared_event m_capture_event;
    std::mutex m_mutex;
    winrt::Direct3D11CaptureFrame m_frame = {nullptr};
    LONG call_count = 0;

    winrt::DirectXPixelFormat m_pixel_format = winrt::DirectXPixelFormat::B8G8R8A8UIntNormalized;
};

WindowCapture::WindowCapture()
{
    // Check if screen capturing is supported
    if (!winrt::GraphicsCaptureSession::IsSupported())
        throw std::runtime_error("System does not support Windows GraphicsCaptureSession");
}

void WindowCapture::open(int window_handle)
{
    if (m_is_open)
        return;
    auto d3d_device = utils::CreateD3DDevice();
    auto dxgi_device = d3d_device.as<IDXGIDevice>();
    m_device = utils::CreateDirect3DDevice(dxgi_device.get());
    m_gc_item = utils::CreateCaptureItemForWindow((HWND)IntToPtr(window_handle));

    m_d3d_device = utils::GetDXGIInterfaceFromObject<ID3D11Device>(m_device);
    m_d3d_device->GetImmediateContext(m_d3d_context.put());
    // Creating our frame pool with CreateFreeThreaded means that we
    // will be called back from the frame pool's internal worker thread
    // instead of the thread we are currently on. It also disables the
    // DispatcherQueue requirement.
    m_frame_pool = winrt::Direct3D11CaptureFramePool::CreateFreeThreaded(m_device, m_pixel_format, 1, m_gc_item.Size());
    m_session = m_frame_pool.CreateCaptureSession(m_gc_item);

    m_capture_event = wil::shared_event(wil::EventOptions::ManualReset);

    m_frame_pool.FrameArrived({this, &WindowCapture::on_frame_arrived});
    m_session.StartCapture();
    m_is_open.store(true, std::memory_order_release);
}

void WindowCapture::close()
{
    if (!m_is_open.load(std::memory_order_acquire))
        return;
    m_is_open.store(false);
    // End the capture
    m_session.Close();
    m_frame_pool.Close();

    m_device.Close();
    m_d3d_context.detach();
    m_d3d_device.detach();

    m_frame_pool = nullptr;
    m_session = nullptr;
    m_gc_item = nullptr;
}

winrt::com_ptr<ID3D11Texture2D> WindowCapture::get_texture()
{
    auto success = m_capture_event.wait();
    if (!success)
        throw std::runtime_error("Capture wait returned unsuccessful");
    winrt::Direct3D11CaptureFrame local_frame = { nullptr };  // Declare in outer scope so that it is defined
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_capture_event.ResetEvent();
        local_frame = m_frame;
    }
    auto texture = utils::GetDXGIInterfaceFromObject<ID3D11Texture2D>(local_frame.Surface());
    return utils::CopyD3DTexture(m_d3d_device, texture, true);
}

py::array_t<UINT8> WindowCapture::get_img()
{
    if (!m_is_open.load(std::memory_order_acquire))
        throw std::runtime_error("Tried to get an image without an open window");
    auto texture = get_texture();
    // Get description for conversion into numpy array
    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);
    std::vector<size_t> shape{desc.Height, desc.Width, 3};
    auto data = utils::texture_to_vector(texture, m_device, m_d3d_context);
    return utils::vector_as_numpy<UINT8>(data, shape);
}

void WindowCapture::on_frame_arrived(winrt::Direct3D11CaptureFramePool const &framePool, winrt::Windows::Foundation::IInspectable const &)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_frame = m_frame_pool.TryGetNextFrame();
    m_capture_event.SetEvent();
}

WindowCapture::~WindowCapture()
{
    if (m_is_open.load(std::memory_order_acquire))
    {
        close();
    }
}