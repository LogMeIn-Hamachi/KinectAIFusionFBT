#include "steamvr_bridge.hpp"
#include <openvr_driver.h>
#include <cstdio>
namespace {
constexpr const char* serials[]{"KinectFBT_Waist","KinectFBT_LeftFoot","KinectFBT_RightFoot"};
constexpr const char* roles[]{"TrackerRole_Waist","TrackerRole_LeftFoot","TrackerRole_RightFoot"};
class Device final:public vr::ITrackedDeviceServerDriver {
    std::atomic<vr::TrackedDeviceIndex_t> index_{vr::k_unTrackedDeviceIndexInvalid};
    vr::DriverPose_t pose_{};
    std::mutex mutex_;
    int role_;
    kf::V3 currentPos_{};
    kf::V3 currentVel_{};
    kf::Q currentRot_{1, 0, 0, 0};
    kf::V3 currentAngVel_{};
    kf::V3 errorOffsetPos_{};
    kf::Q errorOffsetRot_{1, 0, 0, 0};
    double errorStartTime_{0};
    double lastPacketPublished_{0};
    double lastTime_{0};
    bool initialized_{false};
public:
    explicit Device(int role):role_(role){}
    vr::EVRInitError Activate(uint32_t index)override {
        index_=index;
        auto props=vr::VRProperties()->TrackedDeviceToPropertyContainer(index);
        vr::VRProperties()->SetStringProperty(props,vr::Prop_SerialNumber_String,serials[role_]);
        vr::VRProperties()->SetStringProperty(props,vr::Prop_ModelNumber_String,"Kinect FBT virtual tracker");
        vr::VRProperties()->SetStringProperty(props,vr::Prop_ManufacturerName_String,"Kinect FBT");
        vr::VRProperties()->SetStringProperty(props,vr::Prop_TrackingSystemName_String,"kinect_fbt");
        vr::VRProperties()->SetStringProperty(props,vr::Prop_ControllerType_String,"vive_tracker");
        vr::VRProperties()->SetStringProperty(props,vr::Prop_InputProfilePath_String,"{kinect_fbt}/input/tracker_profile.json");
        vr::VRProperties()->SetStringProperty(props,vr::Prop_RenderModelName_String,"{htc}vr_tracker_vive_1_0");
        vr::VRProperties()->SetInt32Property(props,vr::Prop_ControllerRoleHint_Int32,vr::TrackedControllerRole_OptOut);
        vr::VRProperties()->SetBoolProperty(props,vr::Prop_WillDriftInYaw_Bool,false);
        vr::VRProperties()->SetBoolProperty(props,vr::Prop_DeviceIsWireless_Bool,false);
        vr::VRProperties()->SetBoolProperty(props,vr::Prop_DeviceProvidesBatteryStatus_Bool,false);
        // Set defaults only; retain roles that the user assigned in SteamVR.
        std::string key="/devices/kinect_fbt/"+std::string(serials[role_]);
        vr::VRProperties()->SetStringProperty(props,vr::Prop_RegisteredDeviceType_String,key.c_str());
        char current[128]{};vr::EVRSettingsError error;
        vr::VRSettings()->GetString("trackers",key.c_str(),current,sizeof(current),&error);
        if(error!=vr::VRSettingsError_None || !current[0])vr::VRSettings()->SetString("trackers",key.c_str(),roles[role_]);
        return vr::VRInitError_None;
    }
    void Deactivate()override{index_=vr::k_unTrackedDeviceIndexInvalid;initialized_=false;}
    void EnterStandby()override{}
    void* GetComponent(const char*)override{return nullptr;}
    void DebugRequest(const char*,char* out,uint32_t n)override{if(n)out[0]=0;}
    vr::DriverPose_t GetPose()override{std::lock_guard l(mutex_);return pose_;}
    void update(const kf::BridgePacket& packet,double time) {
        vr::DriverPose_t p{};
        p.qWorldFromDriverRotation.w=p.qDriverFromHeadRotation.w=p.qRotation.w=1;
        auto& in=packet.poses[role_];
        p.deviceIsConnected=kf::validPacket(packet,time) && in.valid && time<=in.validUntil;
        p.poseIsValid=p.deviceIsConnected;
        p.result=p.poseIsValid?vr::TrackingResult_Running_OK:vr::TrackingResult_Uninitialized;
        if(p.poseIsValid) {
            kf::V3 targetPos{in.position[0],in.position[1],in.position[2]};
            kf::Q targetRot{in.rotation[0],in.rotation[1],in.rotation[2],in.rotation[3]};
            kf::V3 targetVel{in.velocity[0],in.velocity[1],in.velocity[2]};
            kf::V3 targetAngVel{in.angularVelocity[0],in.angularVelocity[1],in.angularVelocity[2]};

            auto angDelta=[](kf::V3 w,double dt) -> kf::Q {
                double speed=kf::norm(w);
                if(speed<1e-6 || dt<=0.0)return {1,0,0,0};
                return kf::axisAngle(w/speed,speed*dt);
            };

            if(!initialized_ || time-lastTime_>0.25 || time<lastTime_) {
                currentPos_=targetPos;
                currentVel_=targetVel;
                currentRot_=targetRot;
                currentAngVel_=targetAngVel;
                errorOffsetPos_={};
                errorOffsetRot_={1,0,0,0};
                errorStartTime_=time;
                lastPacketPublished_=packet.published;
                initialized_=true;
            } else {
                // When a fresh packet arrives from the engine, capture the prediction discrepancy
                // between our previous trajectory and the new target.
                if(packet.published>lastPacketPublished_+1e-5) {
                    double dtPacket=std::clamp(time-packet.published,0.0,0.10);
                    kf::V3 predictedNewPos=targetPos+targetVel*dtPacket;
                    kf::Q predictedNewRot=kf::normalized(angDelta(targetAngVel,dtPacket)*targetRot);

                    kf::V3 rawError=currentPos_-predictedNewPos;
                    if(kf::norm(rawError)<0.25) {
                        errorOffsetPos_=rawError;
                        errorOffsetRot_=kf::continuous(currentRot_,predictedNewRot)*predictedNewRot.conjugate();
                        errorStartTime_=time;
                    } else {
                        errorOffsetPos_={};
                        errorOffsetRot_={1,0,0,0};
                    }
                    lastPacketPublished_=packet.published;
                }

                // Linear dead-reckoning: advance continuously along target velocity.
                double dtFromPub=std::clamp(time-packet.published,0.0,0.10);
                kf::V3 linearPos=targetPos+targetVel*dtFromPub;
                kf::Q linearRot=kf::normalized(angDelta(targetAngVel,dtFromPub)*targetRot);

                // Smooth Hermite blend to dissolve error offset over 40 ms (zero jump, zero rubber-banding).
                constexpr double blendWindow=0.040;
                double elapsed=time-errorStartTime_;
                double decay=(elapsed>=0.0 && elapsed<blendWindow)?(1.0-elapsed/blendWindow):0.0;
                double smoothDecay=decay*decay*(3.0-2.0*decay);

                currentPos_=linearPos+errorOffsetPos_*smoothDecay;
                currentRot_=kf::normalized(kf::blend(linearRot,errorOffsetRot_*linearRot,smoothDecay));
                currentVel_=targetVel;
                currentAngVel_=targetAngVel;
            }
            lastTime_=time;

            p.poseTimeOffset=0.0;
            p.vecPosition[0]=currentPos_.x;p.vecPosition[1]=currentPos_.y;p.vecPosition[2]=currentPos_.z;
            p.vecVelocity[0]=currentVel_.x;p.vecVelocity[1]=currentVel_.y;p.vecVelocity[2]=currentVel_.z;
            p.qRotation={currentRot_.w,currentRot_.x,currentRot_.y,currentRot_.z};
            p.vecAngularVelocity[0]=currentAngVel_.x;p.vecAngularVelocity[1]=currentAngVel_.y;p.vecAngularVelocity[2]=currentAngVel_.z;
        } else {
            initialized_=false;
        }
        {std::lock_guard l(mutex_);pose_=p;}
        if(index_!=vr::k_unTrackedDeviceIndexInvalid)vr::VRServerDriverHost()->TrackedDevicePoseUpdated(index_,p,sizeof(p));
    }
};
class Provider final:public vr::IServerTrackedDeviceProvider {
    std::unique_ptr<kf::SteamVrBridge> bridge_;
    std::array<std::unique_ptr<Device>,3> devices_;
    kf::BridgePacket packet_;
public:
    vr::EVRInitError Init(vr::IVRDriverContext* context)override {
        VR_INIT_SERVER_DRIVER_CONTEXT(context);
        bridge_=std::make_unique<kf::SteamVrBridge>();
        if(!bridge_->ready())return vr::VRInitError_Driver_Failed;
        for(int i=0;i<3;++i) {
            devices_[i]=std::make_unique<Device>(i);
            if(!vr::VRServerDriverHost()->TrackedDeviceAdded(serials[i],vr::TrackedDeviceClass_GenericTracker,devices_[i].get()))return vr::VRInitError_Driver_Failed;
        }
        vr::VRDriverLog()->Log("Kinect FBT: waist and two foot trackers registered; waiting for app output.");
        return vr::VRInitError_None;
    }
    void Cleanup()override {
        if(bridge_)bridge_->read(packet_,false);
        for(auto& d:devices_)d.reset();bridge_.reset();VR_CLEANUP_SERVER_DRIVER_CONTEXT();
    }
    const char*const* GetInterfaceVersions()override{return vr::k_InterfaceVersions;}
    void RunFrame()override{if(!bridge_)return;bridge_->read(packet_);double time=kf::now();for(auto& d:devices_)if(d)d->update(packet_,time);}
    bool ShouldBlockStandbyMode()override{return false;}
    void EnterStandby()override{}
    void LeaveStandby()override{}
};
Provider provider;
}
extern "C" __declspec(dllexport) void* HmdDriverFactory(const char* name,int* error) {
    if(name && std::strcmp(name,vr::IServerTrackedDeviceProvider_Version)==0){if(error)*error=vr::VRInitError_None;return &provider;}
    if(error)*error=vr::VRInitError_Init_InterfaceNotFound;return nullptr;
}
