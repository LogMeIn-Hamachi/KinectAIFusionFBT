#include "steamvr_bridge.hpp"
#include "alignment.hpp"
#include <openvr_driver.h>
#include <iostream>
using namespace kf;
void check(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(){try {
    State s;s.host=10;s.lastObserved=10;auto& t=s.trackers[0];t.valid=true;t.observedHost=10;t.p={.2,1.1,2};t.q=axisAngle({0,1,0},.4);t.velocity={.1,0,0};
    Calibration cal;cal.valid=true;cal.transform={axisAngle({0,1,0},.3),{1,0,-2}};
    VrSample vr;vr.rawTransformValid=true;vr.standingToRaw={axisAngle({0,1,0},-.7),{.4,.1,.8}};
    check(bindTrackingReference(cal,vr),"Live calibration reference could not be captured");
    auto p=trackerPacket(s,cal,vr,10.02,true);check(validPacket(p,10.02),"Valid SteamVR packet rejected");
    auto expected=vr.standingToRaw.apply(cal.transform.apply(t.p+t.velocity*.02));
    auto& v=p.poses[0];check(norm(V3{v.position[0],v.position[1],v.position[2]}-expected)<1e-9,"Standing/raw conversion or reflection incorrect");
    Q q{v.rotation[0],v.rotation[1],v.rotation[2],v.rotation[3]};
    check(std::abs(dot(q,vr.standingToRaw.q*cal.transform.q*t.q))>.999999,"SteamVR rotation convention mismatch");
    {
        State all=s;
        for(int i=0;i<3;++i){all.trackers[i]=t;all.trackers[i].p={.2*(i-1),i?0.04:1.1,2};}
        const auto physical=trackerPacket(all,cal,vr,10.02,true);
        // OVR applies a standing-space translation/rotation to real devices.
        // Virtual trackers must receive exactly the same movement, once.
        for(Rigid drag:{Rigid{{},{0,1.2,0}},Rigid{axisAngle({0,1,0},1.3),{.7,-.6,1.1}},Rigid{}}) {
            auto shifted=vr;shifted.standingToRaw=composeRigid(vr.standingToRaw,inverseRigid(drag));
            auto packet=trackerPacket(all,cal,shifted,10.02,true);
            check(validPacket(packet,10.02),"Playspace move disabled tracking");
            for(int i=0;i<3;++i) {
                const auto& a=physical.poses[i];const auto& b=packet.poses[i];
                V3 rawA{a.position[0],a.position[1],a.position[2]},rawB{b.position[0],b.position[1],b.position[2]};
                check(norm(rawA-rawB)<1e-9,"Virtual playspace move leaked into physical tracker coordinates");
                auto expectedStanding=drag.apply(vr.standingToRaw.inverse(rawA));
                check(norm(shifted.standingToRaw.inverse(rawB)-expectedStanding)<1e-9,"Waist/feet cancelled or doubled playspace movement");
                Q rawQ{b.rotation[0],b.rotation[1],b.rotation[2],b.rotation[3]};
                Q expectedQ=drag.q*cal.transform.q*all.trackers[i].q;
                check(std::abs(dot(shifted.standingToRaw.q.conjugate()*rawQ,expectedQ))>.999999,"Playspace turn did not rotate tracker orientation");
            }
            DevicePose device{{.3,1.4,-.2},axisAngle({1,0,0},.4),true};
            shifted.devices[1]={drag.apply(device.p),drag.q*device.q,true};
            auto reference=calibrationVr(shifted,cal);
            check(norm(reference.devices[1].p-device.p)<1e-9 && std::abs(dot(reference.devices[1].q,device.q))>.999999,"Controller constraints reacted to a virtual space drag");
            auto oscTransform=currentStandingCalibration(cal,shifted);
            check(norm(oscTransform.apply(t.p)-drag.apply(cal.transform.apply(t.p)))<1e-9,"OSC playspace transform disagreed with native trackers");
        }
        auto restarted=vr;restarted.epoch++;
        {
            PoseHistory history;auto before=vr;before.host=5;
            before.devices[1]={{.3,1.4,-.2},axisAngle({1,0,0},.4),true};
            auto after=before;after.host=5.02;
            Rigid drag{axisAngle({0,1,0},1.1),{.6,1.2,-.4}};
            after.standingToRaw=composeRigid(vr.standingToRaw,inverseRigid(drag));
            // A real 2 cm motion must survive; the large virtual drag must not.
            after.devices[1].p=drag.apply(before.devices[1].p+V3{.02,0,0});
            after.devices[1].q=drag.q*before.devices[1].q;
            history.add(before);history.add(after);
            auto middle=history.at(5.01);check(bool(middle),"No time-matched VR sample");
            auto physicalMiddle=calibrationVr(*middle,cal);
            check(norm(physicalMiddle.devices[1].p-before.devices[1].p-V3{.01,0,0})<1e-9,"Pose interpolation mixed different virtual playspaces");
            check(std::abs(dot(physicalMiddle.devices[1].q,before.devices[1].q))>.999999,"Pose interpolation invented controller rotation during drag");
            after.rawTransformValid=false;history.clear();history.add(before);history.add(after);
            check(!history.at(5.01)->rawTransformValid,"Missing physical reference was interpolated as valid");
        }
        check(!trackerPacket(all,cal,restarted,10.02,true).enabled,"SteamVR restart reused an old live reference");
        check(!calibrationVr(restarted,cal).devices[1].valid,"Old-reference VR observations accepted after restart");
        auto unconfirmed=cal;unconfirmed.rawReferenceValid=false;
        check(!trackerPacket(all,unconfirmed,vr,10.02,true).enabled,"Unconfirmed tracking reference emitted poses");
    }
    check(!trackerPacket(s,cal,vr,10.02,false).enabled,"Paused app published trackers");
    vr.rawTransformValid=false;check(!trackerPacket(s,cal,vr,10.02,true).enabled,"Unknown raw origin accepted");
    check(!validPacket(p,10.3),"Dead writer did not expire");
    p.poses[0].position[1]=std::numeric_limits<double>::quiet_NaN();check(!validPacket(p,10.02),"Nonfinite driver pose accepted");
    p.poses[0].position[1]=1;p.version=42;check(!validPacket(p,10.02),"Incompatible protocol accepted");
    SteamVrBridge writer(true),reader(true);check(writer.ready() && reader.ready(),"Shared pose channel failed");
    p=BridgePacket{};p.published=now();bool active=false;
    check(writer.publish(p,active),"IPC write failed");BridgePacket copy;
    check(reader.read(copy),"IPC read failed");check(copy.published==p.published,"IPC torn/corrupt pose");
    check(writer.driverReady(),"Driver heartbeat missing");
    check(reader.read(copy,false),"IPC shutdown failed");check(!writer.driverReady(),"Stopped driver retained ready status");
    for(int step=0;step<5;++step) {
        auto waiting=alignmentCue(step,120,false,true),settle=alignmentCue(step,-3),hold=alignmentCue(step,0);
        check(waiting.waitingForReady && !waiting.collecting && waiting.seconds==0,"Waiting for readiness must have no deadline");
        check(!settle.collecting && settle.seconds==3,"Ready capture lacks settling time");
        check(hold.collecting && hold.seconds==3,"Calibration hold countdown incorrect");
        check(alignmentCue(step,0,false,true).speech==waiting.speech,"Waiting repeats or cuts off instruction");
        check(waiting.instruction.find("Palms")==std::string::npos,"Complicated palm pose instruction retained");
    }
    check(!alignmentCue(4,8,true).collecting,"Calibration collects after completion");
    HMODULE dll=LoadLibraryW(L"driver_kinect_fbt.dll");check(dll!=nullptr,"SteamVR driver DLL cannot load");
    using Factory=void*(*)(const char*,int*);auto factory=reinterpret_cast<Factory>(GetProcAddress(dll,"HmdDriverFactory"));
    check(factory!=nullptr,"Missing SteamVR driver factory");int error=0;
    check(!factory("invalid_interface",&error) && error!=0,"Driver factory accepts invalid interface");
    check(factory(vr::IServerTrackedDeviceProvider_Version,&error) && error==0,"Driver provider interface unavailable");
    FreeLibrary(dll);
    std::cout<<"SteamVR transform, packet validation, stale disconnect, IPC, driver loading and calibration pacing passed.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
