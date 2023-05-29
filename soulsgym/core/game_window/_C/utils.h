#pragma once
#include "pch.h"
#include "pybind11-global/pybind11/numpy.h"

/* This utility header is mainly a selective copy of all necessary definitions from the robmikh.common 0.22 beta library.
   See https://github.com/robmikh/robmikh.common
 */

namespace py = pybind11;

namespace utils {

    extern "C"
    {
        HRESULT __stdcall CreateDirect3D11DeviceFromDXGIDevice(::IDXGIDevice* dxgiDevice,
            ::IInspectable** graphicsDevice);

        HRESULT __stdcall CreateDirect3D11SurfaceFromDXGISurface(::IDXGISurface* dgxiSurface,
            ::IInspectable** graphicsSurface);
    }

    struct __declspec(uuid("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1"))
        IDirect3DDxgiInterfaceAccess : ::IUnknown
    {
        virtual HRESULT __stdcall GetInterface(GUID const& id, void** object) = 0;
    };

    template <typename T>
    auto GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object)
    {
        auto access = object.as<IDirect3DDxgiInterfaceAccess>();
        winrt::com_ptr<T> result;
        winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
        return result;
    }

    inline auto CopyD3DTexture(winrt::com_ptr<ID3D11Device> const& device, winrt::com_ptr<ID3D11Texture2D> const& texture, bool asStagingTexture)
    {
        winrt::com_ptr<ID3D11DeviceContext> context;
        device->GetImmediateContext(context.put());

        D3D11_TEXTURE2D_DESC desc = {};
        texture->GetDesc(&desc);
        // Clear flags that we don't need
        desc.Usage = asStagingTexture ? D3D11_USAGE_STAGING : D3D11_USAGE_DEFAULT;
        desc.BindFlags = asStagingTexture ? 0 : D3D11_BIND_SHADER_RESOURCE;
        desc.CPUAccessFlags = asStagingTexture ? D3D11_CPU_ACCESS_READ : 0;
        desc.MiscFlags = 0;

        // Create and fill the texture copy
        winrt::com_ptr<ID3D11Texture2D> textureCopy;
        winrt::check_hresult(device->CreateTexture2D(&desc, nullptr, textureCopy.put()));
        context->CopyResource(textureCopy.get(), texture.get());

        return textureCopy;
    }

    inline auto CreateD3DDevice(D3D_DRIVER_TYPE const type, UINT flags, winrt::com_ptr<ID3D11Device>& device)
    {
        WINRT_ASSERT(!device);

        return D3D11CreateDevice(nullptr, type, nullptr, flags, nullptr, 0, D3D11_SDK_VERSION, device.put(),
            nullptr, nullptr);
    }

    inline auto CreateD3DDevice(UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT)
    {
        winrt::com_ptr<ID3D11Device> device;
        HRESULT hr = CreateD3DDevice(D3D_DRIVER_TYPE_HARDWARE, flags, device);
        if (DXGI_ERROR_UNSUPPORTED == hr)
        {
            hr = CreateD3DDevice(D3D_DRIVER_TYPE_WARP, flags, device);
        }

        winrt::check_hresult(hr);
        return device;
    }

    inline auto CreateDirect3DDevice(IDXGIDevice* dxgi_device)
    {
        winrt::com_ptr<::IInspectable> d3d_device;
        winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi_device, d3d_device.put()));
        return d3d_device.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
    }

    inline auto CreateCaptureItemForWindow(HWND hwnd)
    {
        auto interop_factory = winrt::get_activation_factory<winrt::Windows::Graphics::Capture::GraphicsCaptureItem, IGraphicsCaptureItemInterop>();
        winrt::Windows::Graphics::Capture::GraphicsCaptureItem item = { nullptr };
        winrt::check_hresult(interop_factory->CreateForWindow(hwnd, winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>(), winrt::put_abi(item)));
        return item;
    }

    // Own utility functions

    std::vector<uint8_t> texture_to_vector(const winrt::com_ptr<ID3D11Texture2D>& texture,
        const winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice& device,
        const winrt::com_ptr<ID3D11DeviceContext> context)
    {
        D3D11_TEXTURE2D_DESC desc;
        texture->GetDesc(&desc);

        // Create a staging resource with similar parameters as the original texture
        D3D11_TEXTURE2D_DESC stagingDesc{};
        stagingDesc.Width = desc.Width;
        stagingDesc.Height = desc.Height;
        stagingDesc.MipLevels = desc.MipLevels;
        stagingDesc.ArraySize = desc.ArraySize;
        stagingDesc.Format = desc.Format;
        stagingDesc.SampleDesc = desc.SampleDesc;
        stagingDesc.Usage = D3D11_USAGE_STAGING;
        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

        auto d3d_device = utils::GetDXGIInterfaceFromObject<ID3D11Device>(device);

        winrt::com_ptr<ID3D11Texture2D> stagingTexture;

        winrt::check_hresult(d3d_device->CreateTexture2D(&stagingDesc, nullptr, stagingTexture.put()));

        // Copy the texture data to the staging resource
        context->CopyResource(stagingTexture.get(), texture.get());

        // Map the staging resource to retrieve a pointer to the data
        D3D11_MAPPED_SUBRESOURCE mappedResource{};  // RGBA array of the current screen
        winrt::check_hresult(context->Map(stagingTexture.get(), 0, D3D11_MAP_READ, 0, &mappedResource));

        // Create a std::vector wrapping the mapped data pointer
        std::vector<uint8_t> rgbData(desc.Width * desc.Height * 3);
        for (size_t y = 0; y < desc.Height; ++y)
        {
            for (size_t x = 0; x < desc.Width; ++x)
            {
                uint8_t* pixel = reinterpret_cast<uint8_t*>(mappedResource.pData) + y * mappedResource.RowPitch + x * 4;
                rgbData[(y * desc.Width + x) * 3 + 0] = pixel[2]; // Red channel
                rgbData[(y * desc.Width + x) * 3 + 1] = pixel[1]; // Green channel
                rgbData[(y * desc.Width + x) * 3 + 2] = pixel[0]; // Blue channel
            }
        }

        // Unmap the staging resource
        context->Unmap(stagingTexture.get(), 0);

        return rgbData;
    }

    template<typename T>
    py::array_t<T>vector_as_numpy(std::vector<T>& data, std::vector<size_t>& shape)
    {
        std::vector<size_t> stride(shape.size(), 0);
        size_t elm_stride = sizeof(T);
        auto shape_it = shape.rbegin();
        auto stride_it = stride.rbegin();
        for (; stride_it != stride.rend(); stride_it++, shape_it++) {
            *stride_it = elm_stride;
            elm_stride *= *shape_it;
        }
        return py::array_t<T>(shape, stride, data.data());
    }
}
