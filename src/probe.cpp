#include <Windows.h>
#include <objbase.h>
#include <dmo.h>
#include <NuiApi.h>
#include <chrono>
#include <dmo.h>
#include <iostream>
#include <objbase.h>
#include <thread>
int main() {
    int count = 0;
    HRESULT hr = NuiGetSensorCount(&count);
    std::cout << "Kinect SDK 1.8; enumeration HRESULT=0x" << std::hex << hr << std::dec
              << "; sensors=" << count << "\n";
    for (int i = 0; i < count; ++i) {
        INuiSensor *s = nullptr;
        hr = NuiCreateSensorByIndex(i, &s);
        if (FAILED(hr) || !s)
            continue;
        std::cout << "Sensor " << i << " status=0x" << std::hex << s->NuiStatus() << std::dec << "\n";
        hr = s->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR |
                              NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX |
                              NUI_INITIALIZE_FLAG_USES_SKELETON);
        std::cout << "Initialize=0x" << std::hex << hr << std::dec << "\n";
        if (SUCCEEDED(hr)) {
            HANDLE rgb = nullptr, depth = nullptr;
            auto a = s->NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, nullptr,
                                           &rgb);
            auto b = s->NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX,
                                           NUI_IMAGE_RESOLUTION_640x480, 0, 2, nullptr, &depth);
            auto c = s->NuiSkeletonTrackingEnable(nullptr, 0);
            std::cout << "640x480 RGB/depth, skeleton HRESULTs: " << std::hex << a << "," << b << "," << c
                      << std::dec << "\n";
            unsigned nr = 0, nd = 0, ns = 0;
            auto start = std::chrono::steady_clock::now();
            while (std::chrono::steady_clock::now() - start < std::chrono::seconds(3)) {
                NUI_IMAGE_FRAME f{};
                if (SUCCEEDED(a) && SUCCEEDED(s->NuiImageStreamGetNextFrame(rgb, 0, &f))) {
                    ++nr;
                    s->NuiImageStreamReleaseFrame(rgb, &f);
                }
                if (SUCCEEDED(b) && SUCCEEDED(s->NuiImageStreamGetNextFrame(depth, 0, &f))) {
                    ++nd;
                    s->NuiImageStreamReleaseFrame(depth, &f);
                }
                NUI_SKELETON_FRAME sk{};
                if (SUCCEEDED(s->NuiSkeletonGetNextFrame(0, &sk)))
                    ++ns;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::cout << "3-second observed frames RGB=" << nr << " depth=" << nd << " skeleton=" << ns
                      << "\n";
            s->NuiShutdown();
        }
        s->Release();
    }
}
