#include "exposure.hpp"
#include <Windows.h>
#include <NuiSensorLib.h>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <chrono>
namespace kf {
struct KinectExposure::Impl {
    NUISENSOR_HANDLE handle{};
    bool changed{};
    UINT32 originalACS{};
    UINT32 command(NUISENSOR_RGB_COMMAND_TYPE cmd, UINT32 arg=0) {
        static std::atomic<UINT32> sequence{};
        NUISENSOR_RGB_CHANGE_STREAM_SETTING request{};
        request.NumCommands=1;request.SequenceId=++sequence;
        request.Commands[0].Cmd=cmd;request.Commands[0].Arg=arg;
        NUISENSOR_RGB_CHANGE_STREAM_SETTING_REPLY reply{};reply.NumStatus=1;
        std::array<BYTE,NUISENSOR_SEND_SCRATCH_SPACE_REQUIRED> scratch{};
        HANDLE event=CreateEventW(nullptr,TRUE,FALSE,nullptr);
        if(!event)throw std::runtime_error("Cannot create Kinect exposure event");
        OVERLAPPED overlapped{};
        overlapped.hEvent=reinterpret_cast<HANDLE>(reinterpret_cast<UINT_PTR>(event)|1);
        BOOL ok=NuiSensor_ColorChangeCameraSettings(handle,scratch.data(),DWORD(scratch.size()),
            &request,sizeof(request),&reply,sizeof(reply),&overlapped);
        DWORD error=ok?ERROR_SUCCESS:GetLastError();
        if(!ok && error==ERROR_IO_PENDING) {
            const DWORD wait=WaitForSingleObject(event,2000);
            if(wait!=WAIT_OBJECT_0)CancelIoEx(handle,&overlapped);
            DWORD bytes{};
            ok=GetOverlappedResult(handle,&overlapped,&bytes,TRUE);
            error=ok?ERROR_SUCCESS:GetLastError();
            if(wait!=WAIT_OBJECT_0){ok=FALSE;error=ERROR_TIMEOUT;}
        }
        CloseHandle(event);
        if(!ok)throw std::runtime_error("Kinect exposure command "+std::to_string(cmd)+" failed (Windows "+std::to_string(error)+")");
        if(reply.NumStatus!=1 || reply.CommandListStatus || reply.Status[0].Status)
            throw std::runtime_error("Kinect rejected exposure command "+std::to_string(cmd)+
                " ("+std::to_string(reply.CommandListStatus)+","+std::to_string(reply.Status[0].Status)+")");
        return reply.Status[0].Data;
    }
    void restore() {
        if(!changed)return;
        command(NUISENSOR_RGB_COMMAND_SET_EXPOSURE_MODE,0);
        command(NUISENSOR_RGB_COMMAND_SET_ACS,originalACS);
        changed=false;
    }
    ~Impl(){if(handle){try{restore();}catch(...){}NuiSensor_Shutdown(handle);}}
};
KinectExposure::KinectExposure()=default;
KinectExposure::~KinectExposure()=default;
bool KinectExposure::open(const std::wstring& sensorId) {
    p_.reset();
    std::array<NUISENSOR_DEVICE_INFO,16> devices{};
    const auto count=std::min<ULONG>(NuiSensor_FindAllDevices(devices.data(),ULONG(devices.size())),ULONG(devices.size()));
    for(ULONG i=0;i<count;++i) {
        auto p=std::make_unique<Impl>();
        if(!NuiSensor_InitializeEx(&p->handle,devices[i].DevicePath))continue;
        NUISENSOR_SERIAL_NUMBER serial{};
        if(!NuiSensor_GetSerialNumber(p->handle,&serial))continue;
        wchar_t id[65]{};static_assert(sizeof(serial)==128);
        std::memcpy(id,&serial,sizeof(serial));
        if(sensorId!=id)continue;
        p->originalACS=p->command(NUISENSOR_RGB_COMMAND_GET_ACS);
        p_=std::move(p);return true;
    }
    return false;
}
void KinectExposure::prioritize30() {
    if(!p_)throw std::runtime_error("Kinect exposure control unavailable");
    p_->changed=true;
    try {
        p_->command(NUISENSOR_RGB_COMMAND_SET_ACS,0);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        p_->command(NUISENSOR_RGB_COMMAND_SET_EXPOSURE_MODE,3);
        p_->command(NUISENSOR_RGB_COMMAND_SET_EXPOSURE_TIME_MS,std::bit_cast<UINT32>(33.3f));
        const float ms=std::bit_cast<float>(p_->command(NUISENSOR_RGB_COMMAND_GET_EXPOSURE_TIME_MS));
        if(!std::isfinite(ms) || std::abs(ms-33.3f)>.1f || ms>33.34f)
            throw std::runtime_error("Kinect did not accept 33.3 ms exposure (reported "+std::to_string(ms)+" ms)");
    }catch(...){try{p_->restore();}catch(...){}throw;}
}
void KinectExposure::automatic(){
    if(!p_)return;
    if(p_->changed)p_->restore();
    else p_->command(NUISENSOR_RGB_COMMAND_SET_EXPOSURE_MODE,0);
}
}
