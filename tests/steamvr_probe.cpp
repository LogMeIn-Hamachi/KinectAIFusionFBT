#include "steamvr_bridge.hpp"
#include <openvr.h>
#include <iostream>
int main() {
    kf::SteamVrBridge bridge;
    std::cout<<"driver_heartbeat="<<bridge.driverReady()<<'\n';
    vr::EVRInitError error;
    auto system=vr::VR_Init(&error,vr::VRApplication_Background);
    if(error || !system){std::cerr<<vr::VR_GetVRInitErrorAsEnglishDescription(error)<<'\n';return 2;}
    unsigned count=0;
    for(unsigned i=0;i<vr::k_unMaxTrackedDeviceCount;++i) {
        char serial[256]{};
        system->GetStringTrackedDeviceProperty(i,vr::Prop_SerialNumber_String,serial,sizeof(serial));
        if(std::string(serial).starts_with("KinectFBT_")) {
            char profile[512]{};
            system->GetStringTrackedDeviceProperty(i,vr::Prop_InputProfilePath_String,profile,sizeof(profile));
            std::cout<<"tracker="<<serial<<" class="<<system->GetTrackedDeviceClass(i)
                <<" connected="<<system->IsTrackedDeviceConnected(i)<<" profile="<<profile<<'\n';
            ++count;
        }
    }
    bool heartbeat=bridge.driverReady();
    std::cout<<"tracker_count="<<count<<" driver_heartbeat="<<heartbeat<<'\n';
    vr::VR_Shutdown();
    return count==3 && heartbeat?0:1;
}
