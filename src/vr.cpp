#include "io.hpp"
#include <Windows.h>
#include <openvr.h>
#include <winsock2.h>
#include <ws2tcpip.h>
namespace kf {
struct VrInput::Impl {
    vr::IVRSystem *system{};
    std::uint64_t epoch{};
    std::uint64_t universe{};
    double retry{};
    ~Impl() {
        if (system)
            vr::VR_Shutdown();
    }
};
VrInput::VrInput() : p_(std::make_unique<Impl>()) {}
VrInput::~VrInput() = default;
std::uint64_t VrInput::epoch() const {
    return p_->epoch;
}
VrSample VrInput::poll() {
    VrSample sample;
    sample.host = now();
    auto &p = *p_;
    if (!p.system && sample.host >= p.retry) {
        p.retry = sample.host + 3;
        vr::EVRInitError error; // Background does not start or replace a scene application.
        p.system = vr::VR_Init(&error, vr::VRApplication_Background);
        if (error != vr::VRInitError_None) {
            p.system = nullptr;
            status = vr::VR_GetVRInitErrorAsEnglishDescription(error);
        } else {
            ++p.epoch;
            p.universe=p.system->GetUint64TrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd,vr::Prop_CurrentUniverseId_Uint64);
            status = "SteamVR standing-space pose input connected";
        }
    }
    sample.epoch = p.epoch;
    if (!p.system)
        return sample;
    vr::VREvent_t event;
    while (p.system->PollNextEvent(&event, sizeof event)) {
        if (event.eventType == vr::VREvent_Quit) {
            vr::VR_Shutdown();
            p.system = nullptr;
            ++p.epoch;
            status = "SteamVR stopped; calibration invalidated";
            return sample;
        }
        if (event.eventType == vr::VREvent_ChaperoneUniverseHasChanged) {
            const auto universe=event.data.chaperone.m_nCurrentUniverse;
            if(p.universe && universe && p.universe!=universe){++p.epoch;sample.epoch=p.epoch;}
            if(universe)p.universe=universe;
        }
        // Same-universe chaperone edits, space drag and seated recentering change
        // the standing transform, not the physical device coordinate system.
    }
    std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> poses{};
    p.system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, poses.data(),
                                              uint32_t(poses.size()));
    sample.host = now();
    auto rawToStanding=p.system->GetRawZeroPoseToStandingAbsoluteTrackingPose();
    M3 rawRotation;
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)rawRotation.a[r][c]=rawToStanding.m[r][c];
    Q q=quaternion(rawRotation);V3 t{rawToStanding.m[0][3],rawToStanding.m[1][3],rawToStanding.m[2][3]};
    sample.standingToRaw={q.conjugate(),q.conjugate().rotate(-t)};
    sample.rawTransformValid=finite(t) && std::isfinite(dot(q,q));
    std::array<uint32_t, 3> ids{
        vr::k_unTrackedDeviceIndex_Hmd,
        p.system->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_LeftHand),
        p.system->GetTrackedDeviceIndexForControllerRole(vr::TrackedControllerRole_RightHand)};
    for (int i = 0; i < 3; ++i) {
        if (ids[i] >= poses.size())
            continue;
        auto &pose = poses[ids[i]];
        if (!pose.bDeviceIsConnected || !pose.bPoseIsValid ||
            pose.eTrackingResult != vr::TrackingResult_Running_OK)
            continue;
        M3 m;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                m.a[a][b] = pose.mDeviceToAbsoluteTracking.m[a][b];
        sample.devices[i] = {{pose.mDeviceToAbsoluteTracking.m[0][3], pose.mDeviceToAbsoluteTracking.m[1][3],
                              pose.mDeviceToAbsoluteTracking.m[2][3]},
                             quaternion(m),
                             true};
        if(i>0) {
            vr::VRControllerState_t buttons{};
            if(p.system->GetControllerState(ids[i],&buttons,sizeof buttons)) {
                sample.triggerAvailable[i-1]=true;
                sample.triggerPressed[i-1]=(buttons.ulButtonPressed & vr::ButtonMaskFromId(vr::k_EButton_SteamVR_Trigger))!=0;
            }
        }
    }
    return sample;
}
OscOutput::OscOutput() {
    WSADATA data{};
    initialized_ = WSAStartup(MAKEWORD(2, 2), &data) == 0;
    if (initialized_) {
        socket_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        u_long nonblocking = 1;
        ioctlsocket(socket_, FIONBIO, &nonblocking);
    }
}
OscOutput::~OscOutput() {
    if (socket_ != INVALID_SOCKET)
        closesocket(socket_);
    if (initialized_)
        WSACleanup();
}
bool OscOutput::send(std::span<const std::uint8_t> b) {
    if (b.empty() || socket_ == INVALID_SOCKET)
        return false;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(9000);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    return sendto(socket_, reinterpret_cast<const char *>(b.data()), int(b.size()), 0,
                  reinterpret_cast<sockaddr *>(&address), sizeof address) == int(b.size());
}
} // namespace kf
