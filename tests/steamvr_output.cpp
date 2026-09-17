#include "steamvr_bridge.hpp"
#include "alignment.hpp"
#include "tracker_smoothing.hpp"
#include "osc_tracking.hpp"
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
    {
        auto status=bridgeTrackingStatus(p,0,10.02);
        check(status.connected && status.valid,"Live tracker is not valid/connected");
        auto missing=p;missing.poses[0].valid=0;
        status=bridgeTrackingStatus(missing,0,10.02);
        check(status.connected && !status.valid,"Missing pose hot-unplugged a live device");
        missing=p;missing.published=10.25;
        status=bridgeTrackingStatus(missing,0,10.25);
        check(status.connected && !status.valid,"Expired observation stayed valid or disconnected device");
        check(!bridgeTrackingStatus(missing,0,10.6).connected,"Dead writer remained connected");
        missing.enabled=0;
        check(!bridgeTrackingStatus(missing,0,10.25).connected,"Stopped output remained connected");
    }
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
    for(double renderHz:{90.,120.,144.}) {
        TrackerSmoothing filter;
        filter.update({},Q{},1,1);
        auto target=axisAngle({0,1,0},pi/2);
        double previousDistance=.2,previousAngle=pi/2;
        for(int i=1;i<=int(renderHz*.2);++i) {
            double time=1+i/renderHz;
            filter.update({.2,0,0},i%2?target:-target,1+1./30,time);
            double distance=norm(filter.position()-V3{.2,0,0});
            double angle=angleBetween(filter.rotation(),target);
            check(distance<=previousDistance+1e-9 && filter.position().x<=.2,"Driver position overshot or oscillated");
            check(angle<=previousAngle+1e-9,"Driver rotation reversed or lost quaternion continuity");
            previousDistance=distance;previousAngle=angle;
        }
        check(previousDistance<.001 && previousAngle<.01,"Actual driver filter failed to converge");
        filter.update({2,0,0},Q{},2,2);
        check(norm(filter.position()-V3{2,0,0})<1e-9,"Driver retained stale history after gap/teleport");
        filter.reset();
        filter.update({.1,0,0},target,3,3);
        check(norm(filter.position()-V3{.1,0,0})<1e-9 && angleBetween(filter.rotation(),target)<1e-6,"Tracker reacquisition retained old smoothing");
    }
    // The same 30 Hz observations must produce the same measured motion at
    // different compositor rates. Repeated packets must not remeasure error.
    for(double renderHz:{90.,120.,144.}) {
        TrackerSmoothing filter;double maximumLag=0;
        for(int i=0;i<=int(renderHz*3);++i) {
            double elapsed=i/renderHz;
            int sample=int(std::floor(elapsed*30+1e-8));double observed=1+sample/30.;
            V3 target{sample/30.*.2,0,0};
            filter.update(target,axisAngle({0,1,0},sample/30.*.4),observed,1+elapsed);
            if(elapsed>1)maximumLag=std::max(maximumLag,target.x-filter.position().x);
        }
        check(std::abs(filter.speed()-.2)<.001,"Driver adaptive speed depends on compositor rate/filter error");
        check(maximumLag<.015,"Corrected driver added excessive movement lag");
        // Stop at the last point: fresh identical samples must remove velocity.
        for(int i=1;i<=30;++i)filter.update({.6,0,0},axisAngle({0,1,0},1.2),4+i/30.,4+i/30.);
        check(filter.speed()<1e-6,"Stationary target retained adaptive speed");
    }
    // OSC and native driver consume identical predicted poses and smoothing.
    // Compare physical motion at both camera rates, including live OVR changes.
    auto oscVr=vr;oscVr.rawTransformValid=true;
    for(double cameraHz:{15.,30.}) {
        OscTracking osc;
        std::array<TrackerSmoothing,3> native;
        State moving=s;moving.body.id=42;
        for(int tick=0;tick<375;++tick) {
            double time=20+tick/125.;
            double sample=20+std::floor((time-20)*cameraHz+1e-7)/cameraHz;
            moving.host=moving.lastObserved=sample;
            for(int i=0;i<3;++i) {
                auto& tracker=moving.trackers[i];tracker.valid=true;tracker.observedHost=sample;
                tracker.p={.15*(sample-20)+i*.2,i?0.05:1.0,2};
                tracker.velocity={.15,0,0};tracker.q=axisAngle({0,1,0},.4*(sample-20));
            }
            auto space=oscVr;space.rawTransformValid=true;
            Rigid drag=tick>=100 && tick<250?Rigid{axisAngle({0,1,0},.8),{.5,1,-.4}}:
                tick>=250 && tick<300?Rigid{{},{25,0,0}}:Rigid{};
            space.standingToRaw=composeRigid(oscVr.standingToRaw,inverseRigid(drag));
            space.devices[0].q=axisAngle({1,0,0},std::sin(time)); // gaze cannot move trackers
            auto packet=trackerPacket(moving,cal,space,time,true);
            auto output=osc.update(moving,cal,space,time);
            for(int i=0;i<3;++i) {
                const auto& pose=packet.poses[i];
                native[i].update({pose.position[0],pose.position[1],pose.position[2]},
                    {pose.rotation[0],pose.rotation[1],pose.rotation[2],pose.rotation[3]},
                    pose.validUntil-outputHoldSeconds,time);
                auto toStanding=inverseRigid(space.standingToRaw);
                check(output[i].valid,"OSC lost a valid tracker");
                check(norm(output[i].p-toStanding.apply(native[i].position()))<1e-9,"OSC prediction/smoothing or space drag differs from native output");
                check(angleBetween(output[i].q,toStanding.q*native[i].rotation())<1e-6,"OSC rotation differs from native smoothing");
            }
            check(!oscBundle(output,{}).empty(),"Smoothed OSC output failed serialization");
        }
        moving.host=moving.lastObserved=24;
        moving.trackers[0].observedHost=moving.trackers[1].observedHost=24;
        auto partial=osc.update(moving,cal,oscVr,24.01);
        check(partial[0].valid && partial[1].valid && !partial[2].valid,"OSC did not expire a foot independently");
        auto bytes=oscBundle(partial,{});
        std::string wire(bytes.begin(),bytes.end());
        check(wire.find("/tracking/trackers/3/")==std::string::npos,"OSC refreshed an expired foot");
        check(oscBundle(osc.update(moving,cal,oscVr,25),{}).empty(),"OSC sent stale poses after capture stopped");
        auto restarted=oscVr;restarted.epoch++;
        check(oscBundle(osc.update(moving,cal,restarted,24.02),{}).empty(),"OSC reused a reference across a SteamVR restart");
        auto changed=cal;changed.transform.t.x+=.1;
        auto reacquired=osc.update(moving,changed,oscVr,24.01);
        auto expectedPose=trackerPacket(moving,changed,oscVr,24.01,true).poses[0];
        auto expectedPos=inverseRigid(oscVr.standingToRaw).apply({expectedPose.position[0],expectedPose.position[1],expectedPose.position[2]});
        check(norm(reacquired[0].p-expectedPos)<1e-9,"OSC retained filter history across calibration change");
        // Change calibration while tracking remains live: even a small change
        // must not be mistaken for motion to smooth across.
        auto changedAgain=changed;changedAgain.transform.t.x+=.04;
        auto activeChange=osc.update(moving,changedAgain,oscVr,24.02);
        auto activePacket=trackerPacket(moving,changedAgain,oscVr,24.02,true).poses[0];
        auto activeExpected=inverseRigid(oscVr.standingToRaw).apply({activePacket.position[0],activePacket.position[1],activePacket.position[2]});
        check(norm(activeChange[0].p-activeExpected)<1e-9,"Active OSC calibration change was smoothed as motion");
        osc.reset();auto legacy=cal;legacy.rawReferenceValid=false;
        auto legacyOutput=osc.update(moving,legacy,VrSample{},24.01);
        check(legacyOutput[0].valid && norm(legacyOutput[0].p-legacy.transform.apply(moving.trackers[0].p+V3{.0015,0,0}))<1e-8,"Legacy OSC-only calibration regressed");
    }
    HMODULE dll=LoadLibraryW(L"driver_kinect_fbt.dll");check(dll!=nullptr,"SteamVR driver DLL cannot load");
    using Factory=void*(*)(const char*,int*);auto factory=reinterpret_cast<Factory>(GetProcAddress(dll,"HmdDriverFactory"));
    check(factory!=nullptr,"Missing SteamVR driver factory");int error=0;
    check(!factory("invalid_interface",&error) && error!=0,"Driver factory accepts invalid interface");
    check(factory(vr::IServerTrackedDeviceProvider_Version,&error) && error==0,"Driver provider interface unavailable");
    FreeLibrary(dll);
    std::cout<<"SteamVR transform, packet validation, stale disconnect, IPC, driver loading and calibration pacing passed.\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
