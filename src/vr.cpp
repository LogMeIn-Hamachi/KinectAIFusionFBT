#include "io.hpp"
#include "runtime_lifetime.hpp"
#include <Windows.h>
#include <openvr.h>
#include <winsock2.h>
#include <ws2tcpip.h>
namespace kf {
namespace {
// Distinguish separate VrInput instances in this process. The counter is not
// persisted and is never used as a room identifier across application runs.
std::atomic_uint64_t nextVrEpoch{1};
std::uint64_t trackingSource(vr::IVRSystem* system) {
    std::string identity;
    for(auto property:{vr::Prop_TrackingSystemName_String,vr::Prop_SerialNumber_String,vr::Prop_ModelNumber_String}) {
        char value[512]{};vr::ETrackedPropertyError error;
        system->GetStringTrackedDeviceProperty(0,property,value,sizeof value,&error);
        if(error!=vr::TrackedProp_Success || !value[0])return 0;
        identity+=value;identity+='\n';
    }
    // Local profile matching only; do not expose device serials in diagnostics.
    std::uint64_t key=14695981039346656037ull;
    for(unsigned char c:identity){key^=c;key*=1099511628211ull;}
    return key?key:1;
}
}
struct VrInput::Impl {
    vr::IVRSystem *system{};
    std::uint64_t epoch{};
    std::uint64_t universe{};
    std::uint64_t source{};
    bool sourceAvailable{};
    double retry{},nextIdentity{},settleUntil{};
    ~Impl() {
        auto lifetime=openVrLifetime.write();
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
        auto lifetime=openVrLifetime.write();
        p.retry = sample.host + 3;
        vr::EVRInitError error; // Background does not start or replace a scene application.
        p.system = vr::VR_Init(&error, vr::VRApplication_Background);
        if (error != vr::VRInitError_None) {
            p.system = nullptr;
            status = vr::VR_GetVRInitErrorAsEnglishDescription(error);
        } else {
            p.epoch=nextVrEpoch++;
            p.universe=p.system->GetUint64TrackedDeviceProperty(vr::k_unTrackedDeviceIndex_Hmd,vr::Prop_CurrentUniverseId_Uint64);
            p.source=trackingSource(p.system);p.sourceAvailable=p.source!=0;p.nextIdentity=sample.host+1;
            p.settleUntil=sample.host+.75;sample.referenceEvents|=VrRuntimeStarted;
            status = "SteamVR standing-space pose input connected";
        }
    }
    sample.epoch = p.epoch;
    if (!p.system)
        return sample;
    auto lifetime=openVrLifetime.read();
    vr::VREvent_t event;
    while (p.system->PollNextEvent(&event, sizeof event)) {
        if (event.eventType == vr::VREvent_Quit) {
            lifetime.unlock();
            auto shutdown=openVrLifetime.write();
            vr::VR_Shutdown();
            p.system = nullptr;
            p.epoch=nextVrEpoch++;sample.epoch=p.epoch;
            sample.referenceEvents|=VrRuntimeStopped;
            status = "SteamVR stopped; saved alignment retained";
            return sample;
        }
        if (event.eventType == vr::VREvent_ChaperoneUniverseHasChanged) {
            const auto universe=event.data.chaperone.m_nCurrentUniverse;
            if(p.universe && universe && p.universe!=universe){
                p.epoch=nextVrEpoch++;sample.referenceEvents|=VrUniverseChanged;p.settleUntil=sample.host+.75;
            }
            if(universe)p.universe=universe;
        }
        if(event.eventType==vr::VREvent_StandingZeroPoseReset)sample.referenceEvents|=VrStandingReset;
        if(event.eventType==vr::VREvent_SeatedZeroPoseReset)sample.referenceEvents|=VrSeatedReset;
        if(event.eventType==vr::VREvent_ChaperoneRoomSetupStarting) {
            sample.referenceEvents|=VrRoomSetup;
            p.epoch=nextVrEpoch++;p.settleUntil=sample.host+.75;
        }
        // CommitWorkingCopy is also used by playspace movers. It is not proof
        // that a physical room setup occurred and must not invalidate alignment.
        if(event.eventType==vr::VREvent_ChaperoneRoomSetupCommitted)sample.referenceEvents|=VrChaperoneCommitted;
        // Same-universe chaperone edits, space drag and seated recentering change
        // the standing transform, not the physical device coordinate system.
    }
    if(sample.host>=p.nextIdentity) {
        p.nextIdentity=sample.host+1;
        const auto source=trackingSource(p.system);
        p.sourceAvailable=source!=0;
        if(source && source!=p.source) {
            p.source=source;p.epoch=nextVrEpoch++;p.settleUntil=sample.host+.75;
            sample.referenceEvents|=VrSourceChanged;
        }
    }
    sample.epoch=p.epoch;sample.referenceSource=p.source;sample.referenceUniverse=p.universe;
    std::array<vr::TrackedDevicePose_t, vr::k_unMaxTrackedDeviceCount> poses{};
    p.system->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0, poses.data(),
                                              uint32_t(poses.size()));
    sample.host = now();
    auto rawToStanding=p.system->GetRawZeroPoseToStandingAbsoluteTrackingPose();
    M3 rawRotation;
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)rawRotation.a[r][c]=rawToStanding.m[r][c];
    Q q=quaternion(rawRotation);V3 t{rawToStanding.m[0][3],rawToStanding.m[1][3],rawToStanding.m[2][3]};
    sample.standingToRaw={q.conjugate(),q.conjugate().rotate(-t)};
    sample.rawTransformValid=finiteRigid(sample.standingToRaw) && p.sourceAvailable &&
        p.system->IsTrackedDeviceConnected(0) && sample.host>=p.settleUntil;
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
