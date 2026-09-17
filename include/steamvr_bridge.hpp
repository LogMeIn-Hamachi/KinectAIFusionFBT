#pragma once
#include "body_tracker.hpp"
#include <Windows.h>
#include <cstring>
namespace kf {
inline constexpr uint32_t bridgeMagic=0x4b465442,bridgeVersion=2;
struct BridgePose {
    double position[3]{},rotation[4]{1,0,0,0},velocity[3]{},angularVelocity[3]{},validUntil{};
    uint32_t valid{};
};
struct BridgePacket {
    uint32_t magic{bridgeMagic},version{bridgeVersion},bytes{sizeof(BridgePacket)},enabled{};
    double published{};
    std::array<BridgePose,3> poses{};
};
struct BridgeMemory {BridgePacket packet;double driverHeartbeat{};uint32_t driverReady{};};
inline bool validPacket(const BridgePacket& p,double time) {
    if(p.magic!=bridgeMagic || p.version!=bridgeVersion || p.bytes!=sizeof(p) || !p.enabled ||
       !std::isfinite(p.published) || p.published>time+.01 || time-p.published>.25)return false;
    for(auto& v:p.poses)if(v.valid) {
        V3 position{v.position[0],v.position[1],v.position[2]},velocity{v.velocity[0],v.velocity[1],v.velocity[2]};
        V3 angVel{v.angularVelocity[0],v.angularVelocity[1],v.angularVelocity[2]};
        Q q{v.rotation[0],v.rotation[1],v.rotation[2],v.rotation[3]};
        if(!finite(position) || norm(position)>50 || !finite(velocity) || norm(velocity)>4.01 ||
           !finite(angVel) || norm(angVel)>25.01 ||
           !std::isfinite(dot(q,q)) || std::abs(dot(q,q)-1)>.001 || !std::isfinite(v.validUntil) || v.validUntil>p.published+.25)return false;
    }
    return true;
}
struct BridgeTrackingStatus {bool connected{},valid{};};
inline BridgeTrackingStatus bridgeTrackingStatus(const BridgePacket& packet,int role,double time) {
    if(role<0 || role>=3 || !validPacket(packet,time))return {};
    const auto& pose=packet.poses[role];
    // A live device with a temporarily unobserved foot is still connected.
    // Tracking validity must expire, without hot-unplugging the virtual device.
    return {true,pose.valid && time<=pose.validUntil};
}
inline BridgePacket trackerPacket(const State& state,const Calibration& cal,const VrSample& vr,double time,bool enabled) {
    BridgePacket packet;packet.published=time;
    if(!enabled || !cal.valid || !trackingReferenceValid(cal,vr))return packet;
    auto delivery=deliveryState(state,time);packet.enabled=1;
    const auto cameraToRaw=composeRigid(cal.standingToRaw,cal.transform);
    const double age=std::max(0.0,time-state.host);
    const double extraDt=std::clamp(age-.04,0.0,0.06);
    for(int i=0;i<3;++i) {
        auto t=delivery.trackers[i];auto& out=packet.poses[i];
        if(t.valid && extraDt>0)t.p+=bounded(t.velocity,4)*extraDt;
        // SteamVR is right-handed. The OSC/Unity Z reflection does not belong here.
        // Fixed physical reference: SteamVR applies the live playspace offset.
        // Using the current inverse here cancels OVR movement for our trackers.
        auto p=cameraToRaw.apply(t.p);
        auto q=normalized(cameraToRaw.q*t.q);
        auto v=cameraToRaw.q.rotate(bounded(t.velocity,4));
        auto w=cameraToRaw.q.rotate(bounded(t.angularVelocity,20));
        if(age>.10)v={};
        out.position[0]=p.x;out.position[1]=p.y;out.position[2]=p.z;
        out.rotation[0]=q.w;out.rotation[1]=q.x;out.rotation[2]=q.y;out.rotation[3]=q.z;
        out.velocity[0]=v.x;out.velocity[1]=v.y;out.velocity[2]=v.z;
        out.angularVelocity[0]=w.x;out.angularVelocity[1]=w.y;out.angularVelocity[2]=w.z;
        out.validUntil=(t.observedHost>0?t.observedHost:state.lastObserved)+outputHoldSeconds;
        out.valid=t.valid;
    }
    return packet;
}
// Local-session IPC. Both sides use a non-blocking mutex: no render-thread wait,
// torn reads, sockets or external service. Driver heartbeat is separate from poses.
class SteamVrBridge {
    HANDLE mapping_{},mutex_{};
    BridgeMemory* memory_{};
public:
    explicit SteamVrBridge(bool testOnly=false) {
        mutex_=CreateMutexW(nullptr,FALSE,testOnly?L"Local\\KinectFBT_TestMutex_v2":L"Local\\KinectFBT_PoseMutex_v2");
        mapping_=CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,0,sizeof(BridgeMemory),testOnly?L"Local\\KinectFBT_TestPoses_v2":L"Local\\KinectFBT_Poses_v2");
        if(mapping_)memory_=static_cast<BridgeMemory*>(MapViewOfFile(mapping_,FILE_MAP_ALL_ACCESS,0,0,sizeof(BridgeMemory)));
    }
    ~SteamVrBridge(){if(memory_)UnmapViewOfFile(memory_);if(mapping_)CloseHandle(mapping_);if(mutex_)CloseHandle(mutex_);}
    SteamVrBridge(const SteamVrBridge&)=delete;
    bool ready()const{return memory_ && mutex_;}
    bool lock(){if(!ready())return false;auto r=WaitForSingleObject(mutex_,0);return r==WAIT_OBJECT_0 || r==WAIT_ABANDONED;}
    bool publish(const BridgePacket& p,bool& driverReady) {
        if(!lock())return false;
        memory_->packet=p;
        driverReady=memory_->driverReady && p.published-memory_->driverHeartbeat>=0 && p.published-memory_->driverHeartbeat<1;
        ReleaseMutex(mutex_);return true;
    }
    bool read(BridgePacket& p,bool active=true) {
        if(!lock())return false;
        p=memory_->packet;memory_->driverHeartbeat=now();memory_->driverReady=active;
        ReleaseMutex(mutex_);return true;
    }
    bool driverReady() {
        if(!lock())return false;
        bool yes=memory_->driverReady && now()-memory_->driverHeartbeat<1;
        ReleaseMutex(mutex_);return yes;
    }
};
}
