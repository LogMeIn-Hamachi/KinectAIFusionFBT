#include "alignment.hpp"
#include "io.hpp"
#include "tilt.hpp"
#include "vr_overlay.hpp"
#include <iostream>
using namespace kf;
int main() {
    try {
        int checks=0;
        auto check=[&](bool ok,const char *message){if(!ok)throw std::runtime_error(message);++checks;};
        Rigid truth{axisAngle({0,1,0},1.1)*axisAngle({1,0,0},-.45),{.6,1.2,-1.4}};
        std::array<V3,3> expected{{{}, {.045,-.065,.12},{-.035,-.08,.085}}};
        auto samples=[&](int rotationMode=0) {
            std::vector<AlignmentObservation> observations;
            for(int k=0;k<16;++k)for(int d=1;d<=2;++d)for(int tick=0;tick<5;++tick) {
                V3 c{(d==1?-.3:.3)+.12*std::sin(k*.9),.45+.2*std::cos(k*.6),2+.22*std::sin(k*.7)};
                Q q=rotationMode==1?Q{}:axisAngle({0,1,0},std::sin(k*1.2+d)*.9);
                if(rotationMode==0)q=q*axisAngle({1,0,0},std::cos(k*.71)*.65)*axisAngle({0,0,1},std::sin(k*.4)*.55);
                DevicePose pose{truth.apply(c)-q.rotate(expected[d]),q,true};
                observations.push_back({100+k+tick*.15,d,c,pose,{}});
            }
            return observations;
        };
        auto input=samples();std::array<V3,3> fitted{};
        auto result=calibrateControllers(input,fitted);
        if(!result.valid)std::cerr<<result.reason<<'\n';
        check(result.valid,"Controller-only solver accepts varied headless observations");
        check(norm(result.transform.t-truth.t)<1e-5 && norm(result.transform.apply({.3,.5,2})-truth.apply({.3,.5,2}))<1e-5,"Recover tilted camera transform without headset observations");
        check(norm(fitted[1]-expected[1])<1e-5 && norm(fitted[2]-expected[2])<1e-5,"Learn distinct controller-to-wrist offsets");
        auto same=samples(1);check(!calibrateControllers(same,fitted).valid,"Reject fixed controller orientations: unknown wrist offsets cannot be solved");
        for(auto& s:same)s.offset=expected[s.device];
        auto fixed=calibrateKnownOffsets(same);
        check(fixed.valid && norm(fixed.transform.t-truth.t)<1e-5,"Known offsets align fixed controller orientations without offset relearning");
        auto validation=same;for(auto& s:validation)s.camera.x+=.16;
        check(!calibrateKnownOffsets(same,validation).valid,"Separate bad check pose must reject an otherwise good fit");
        check(calibrateKnownOffsets(same,same).valid,"Good independent observations agree with fixed fit");
        Plane floor{truth.q.conjugate().rotate({0,1,0}),truth.t.y,true};
        auto grounded=calibrateKnownOffsets(same,same,&floor);
        check(grounded.valid && norm(grounded.transform.t-truth.t)<1e-5,"Floor-constrained wrist alignment recovers camera transform");
        check(norm(grounded.transform.q.rotate(floor.n)-V3{0,1,0})<1e-8 && std::abs(grounded.transform.t.y-floor.d)<1e-9,"Alignment maps Kinect floor exactly onto SteamVR floor");
        auto floorBias=same;for(auto& s:floorBias)s.pose.p.y+=.025;
        auto floorFit=calibrateKnownOffsets(floorBias,floorBias,&floor);
        check(floorFit.valid && std::abs(floorFit.transform.t.y-floor.d)<1e-9,"Small wrist bias cannot lift the calibrated floor");
        auto wrongFloor=floor;wrongFloor.d+=.18;
        check(!calibrateKnownOffsets(same,same,&wrongFloor).valid,"Inconsistent VR floor must not silently force alignment");
        auto floorFull=calibrateControllers(input,fitted,&floor);
        check(floorFull.valid && norm(fitted[1]-expected[1])<1e-5,"Full offset-learning setup supports floor constraint");
        StableFloor stableFloor;
        for(int i=0;i<25;++i)stableFloor.add(1+i/30.,floor);
        check(stableFloor.value().valid,"Stable observed floor was not accepted");
        stableFloor.add(2,wrongFloor);
        check(!stableFloor.value().valid,"Sudden floor-height change retained stale floor reference");
        for(double angleScale:{.7,1.})for(double fps:{15.,30.})for(bool badCheck:{false,true})for(bool useFloor:{false,true}) {
            VrSample referenceVr;referenceVr.epoch=3;referenceVr.rawTransformValid=true;
            referenceVr.standingToRaw={axisAngle({0,1,0},.4),{1,.2,-2}};
            GuidedAlignment routine;routine.reset(108,&referenceVr);Settings qs;
            double finished=0,entered=100;int previousStage=-1;
            for(int tick=0;tick<int(100*fps) && !routine.done();++tick) {
                Frame f;f.host=100+tick/fps;f.vr.host=f.host;Body b;b.id=42;
                if(useFloor)f.floor=floor;
                auto cue=routine.cue(f.host);int stage=cue.step;
                if(stage!=previousStage){entered=f.host;previousStage=stage;}
                if(cue.waitingForReady && f.host-entered>=10)routine.capturePose(f.host);
                for(int d=1;d<=2;++d){int j=d==1?LWrist:RWrist;double side=d==1?-1.:1.;
                    // Compact wrists: all ahead of the torso, elbows bent, below shoulders.
                    V3 c{side*(stage==1?.32:.25),stage==0||stage==1?1.05:stage==4?(d==1?1.25:1.05):1.25,
                         stage==3?1.65:1.8};
                    Q q=stage==1?axisAngle({0,1,0},side*.8*angleScale):stage==2?axisAngle({1,0,0},1.1*angleScale):stage==4 && d==1?axisAngle({1,0,0},.7*angleScale):Q{};
                    b.joints[j]={c,1,.01,1};f.vr.devices[d]={truth.apply(c)-q.rotate(expected[d]),q,true};
                    if(stage==4 && badCheck && routine.feedback().find("did not agree")==std::string::npos)f.vr.devices[d].p.x+=.15;
                }
                Rigid drag{axisAngle({0,1,0},.6*std::sin(f.host)),{.06*std::sin(f.host),.3*std::cos(f.host),.2}};
                f.vr.epoch=3;f.vr.rawTransformValid=true;
                f.vr.standingToRaw=composeRigid(referenceVr.standingToRaw,inverseRigid(drag));
                for(int d=1;d<=2;++d){f.vr.devices[d].p=drag.apply(f.vr.devices[d].p);f.vr.devices[d].q=drag.q*f.vr.devices[d].q;}
                f.bodies.push_back(b);routine.add(f,42,qs);
                if(f.host<113)check(routine.size()==0,"Preparation and movement cannot enter guided fit");
                if(stage<4 && !routine.done())check(!routine.result().valid,"Alignment accepted before independent check");
                finished=f.host-100;
                if(badCheck && routine.feedback().find("did not agree")!=std::string::npos)break;
            }
            if(!routine.result().valid && !badCheck)std::cerr<<routine.result().reason<<'\n';
            check(routine.done()!=badCheck && routine.result().valid!=badCheck,"One routine learns offsets and rejects bad independent check");
            check(finished>=75 && finished<90,"Guided routine respects manually chosen preparation time");
            if(!badCheck) {
                check(norm(routine.result().transform.t-truth.t)<1e-5,"Guided routine recovers tilted transform");
                check(norm(routine.offsets()[1]-expected[1])<1e-5,"Guided routine learns unknown controller offsets");
            }else {
                check(routine.cue(100+finished).waitingForReady && routine.cue(100+finished).step==4,"Bad check restarted the whole routine");
                auto earlier=routine.size();routine.capturePose(105+finished);
                for(int tick=0;tick<int(12*fps) && !routine.done();++tick) {
                    Frame f;f.host=105+finished+tick/fps;f.vr=referenceVr;f.vr.host=f.host;Body b;b.id=42;
                    for(int d=1;d<=2;++d) {
                        V3 c{d==1?-.25:.25,d==1?1.25:1.05,1.8};int j=d==1?LWrist:RWrist;
                        Q q=d==1?axisAngle({1,0,0},.7*angleScale):Q{};
                        b.joints[j]={c,1,.01,1};f.vr.devices[d]={truth.apply(c)-q.rotate(expected[d]),q,true};
                    }
                    f.bodies.push_back(b);routine.add(f,42,qs);
                }
                check(routine.done() && routine.result().valid && routine.size()>earlier,"Retry of final check lost earlier valid poses or did not recover");
            }
        }
        {
            // Noisy wrists, unfamiliar controller axes and temporary occlusion
            // must work in this same routine without a different setup path.
            GuidedAlignment routine;routine.reset(108);double finished=0,stageStart=0;int previousStage=-1;
            for(int tick=0;tick<100*30 && !routine.done();++tick) {
                Frame f;f.host=100+tick/30.;f.vr.host=f.host;f.floor=floor;Body b;b.id=42;
                auto cue=routine.cue(f.host);int stage=cue.step;
                if(stage!=previousStage){stageStart=f.host;previousStage=stage;}
                if(cue.waitingForReady && f.host-stageStart>=6)routine.capturePose(f.host);
                for(int d=1;d<=2;++d) {
                    int j=d==1?LWrist:RWrist;double side=d==1?-1.:1.;
                    // Compact wrists: all ahead of the torso, elbows bent, below shoulders.
                    V3 c{side*(stage==1?.32:.25),stage==0||stage==1?1.05:stage==4?(d==1?1.25:1.05):1.25,
                         stage==3?1.65:1.8};
                    Q q=stage==1?axisAngle({0,1,0},side*.8):stage==2?axisAngle({1,0,0},1.1):stage==4 && d==1?axisAngle({1,0,0},.7):Q{};
                    q=q*axisAngle(unit(V3{.3,1,.7}),d*.6);
                    auto noise=V3{std::sin(tick*1.7),std::cos(tick*.8),std::sin(tick*.9)}*.007;
                    b.joints[j]={c+noise,1,.01,1};
                    f.vr.devices[d]={truth.apply(c)-q.rotate(expected[d]),q,true};
                    if(stage==2 && f.host-stageStart<20)b.joints[j].confidence=0;
                }
                f.bodies.push_back(b);routine.add(f,42,{});finished=f.host-100;
            }
            if(!routine.result().valid)std::cerr<<routine.result().reason<<'\n';
            check(routine.done() && routine.result().valid,"Noisy/rebased controllers with a visibility pause failed unified routine");
            std::cout<<"Unified calibration with temporary wrist loss finished in "<<finished<<" seconds\n";
            check(finished>=70 && finished<85,"Visibility pause did not extend current pose fairly");
            check(norm(routine.result().transform.t-truth.t)<.02,"Noisy unified alignment lost transform accuracy");
        }
        {
            GuidedAlignment missing;missing.reset(100);Frame f;
            Body waitingBody;waitingBody.id=42;waitingBody.joints[LWrist]={{-.3,1,2},1,.01,1};waitingBody.joints[RWrist]={{.3,1,2},1,.01,1};f.bodies={waitingBody};
            f.vr.devices[1]={{-.3,1,2},{},true};f.vr.devices[2]={{.3,1,2},{},true};
            for(int i=0;i<120*30;++i){f.host=100+i/30.;f.vr.host=f.host;missing.add(f,42,{});}
            check(!missing.done() && missing.size()==0 && missing.cue(f.host).waitingForReady,"Positioning time started capture or timed out without user readiness");
            missing.capturePose(f.host);
            f.bodies.clear();
            for(int i=0;i<24*30;++i){f.host+=1./30;missing.add(f,42,{});}
            check(!missing.done() && !missing.result().valid && missing.cue(f.host).waitingForReady && missing.cue(f.host).step==0,"Invisible wrists did not return to retrying the same pose");
            // A held trigger cannot skip multiple poses; stale input cannot arm.
            GuidedAlignment buttons;buttons.reset(100);f={};f.host=100;f.vr.host=100;
            f.vr.devices[1].valid=true;f.vr.triggerAvailable[0]=true;f.vr.triggerPressed[0]=true;
            buttons.add(f,42,{});check(buttons.cue(100).waitingForReady,"Already-held trigger armed calibration");
            f.host=f.vr.host=101;f.vr.triggerPressed[0]=false;buttons.add(f,42,{});
            f.host=102;f.vr.host=100;f.vr.triggerPressed[0]=true;buttons.add(f,42,{});
            check(buttons.cue(102).waitingForReady,"Stale trigger input armed calibration");
            f.vr.host=f.host;buttons.add(f,42,{});
            check(buttons.cue(102).waitingForReady,"Tracking recovery interpreted held trigger as a new press");
            f.vr.triggerPressed[0]=false;buttons.add(f,42,{});
            f.vr.triggerPressed[0]=true;buttons.add(f,42,{});
            check(!buttons.cue(102).waitingForReady && !buttons.cue(102).collecting && buttons.cue(102).seconds==3,"Fresh trigger press did not start settling countdown");
            check(buttons.size()==0,"Trigger motion entered calibration samples");
            f.bodies={waitingBody};f.vr.devices[1]={{-.3,1,2},{},true};f.vr.devices[2]={{.3,1,2},{},true};
            for(int i=0;i<15*30;++i){f.host=f.vr.host=105+i/30.;buttons.add(f,42,{});}
            check(buttons.cue(f.host).step==1 && buttons.cue(f.host).waitingForReady,"Holding trigger across captured pose automatically captured the next pose");
        }
        {
            GuidedAlignment retry;retry.reset(100);Frame f;f.host=f.vr.host=100;
            f.vr.devices[1].valid=f.vr.devices[2].valid=true;
            f.vr.triggerAvailable={true,true};f.vr.triggerPressed={true,true};
            retry.add(f,42,{});
            for(int attempt=0;attempt<20;++attempt) {
                // Left trigger stays held; right alone releases and presses.
                f.host=f.vr.host=101+attempt*30.;f.vr.triggerPressed[1]=false;retry.add(f,42,{});
                f.host=f.vr.host=f.host+.1;f.vr.triggerPressed[1]=true;retry.add(f,42,{});
                auto cue=retry.cue(f.host);
                check(!cue.waitingForReady && !cue.collecting && cue.seconds==3,"Repeated retry did not start preparation with other trigger held");
                check(retry.cue(f.host+3).collecting,"Preparation did not transition to capture");
                f.host=f.vr.host=f.host+24;retry.add(f,42,{});
                check(retry.retrying() && retry.cue(f.host).waitingForReady && retry.stage()==0,"Repeated wrist loss froze or advanced calibration");
            }
            f.host=f.vr.host=f.host+1;f.vr.triggerPressed[1]=false;retry.add(f,42,{});
            f.host=f.vr.host=f.host+.1;f.vr.triggerPressed[1]=true;retry.add(f,42,{});
            Body recovered;recovered.id=42;
            recovered.joints[LWrist]={{-.3,1,2},1,.01,1};recovered.joints[RWrist]={{.3,1,2},1,.01,1};
            f.bodies={recovered};f.vr.devices[1]={{-.3,1,2},{},true};f.vr.devices[2]={{.3,1,2},{},true};
            for(int i=0;i<300;++i){f.host=f.vr.host=f.host+1./30;retry.add(f,42,{});}
            check(retry.stage()==1 && retry.cue(f.host).waitingForReady,"Valid capture after twenty failed retries did not recover");
            std::string calls;
            auto upload=[&]{calls+='U';};auto flush=[&]{calls+='F';};
            auto submit=[&]{calls+='S';return true;};auto show=[&]{calls+='V';return true;};
            check(presentOverlayFrame(false,upload,submit,flush,show) && calls=="USFV","Overlay GPU work not flushed after submission before initial show");
            calls.clear();
            check(presentOverlayFrame(true,upload,submit,flush,show) && calls=="USF","Visible overlay was shown repeatedly");
            calls.clear();
            check(!presentOverlayFrame(false,upload,[&]{calls+='S';return false;},flush,show) && calls=="USF","Failed overlay submit skipped GPU flush or showed invalid texture");
            OverlayRefresh refresh;OverlayState state;state.active=true;state.isRetry=true;
            check(refresh.due(state,100),"First overlay frame suppressed");
            refresh.complete(state,100,true);
            check(!refresh.due(state,100.5) && !refresh.due(state,101.01),"Unchanged overlay repeatedly reloaded its image");
            refresh.reset();
            for(int attempt=0;attempt<20;++attempt) {
                double t=102+attempt;
                check(refresh.due(state,t),"Failed identical overlay state suppressed retry");
                refresh.complete(state,t,false);
                check(!refresh.due(state,t+.1) && refresh.due(state,t+.26),"Overlay failure retries not bounded");
            }
            refresh.complete(state,123,true);state.agreement="new result";
            check(refresh.due(state,123.11),"Overlay completion text change suppressed");
            refresh.reset();check(refresh.due(state,123.12),"Hidden overlay did not reset presentation");
            state.step=4;state.waitingForReady=true;state.collecting=false;
            refresh.complete(state,124,true);
            check(!refresh.due(state,130),"Idle final pose reuploaded unchanged image");
            for(int attempt=0;attempt<20;++attempt) {
                double t=131+attempt*4.;
                state.waitingForReady=false;state.collecting=true;state.leftSamples=state.rightSamples=12;
                check(refresh.due(state,t),"Stage five capture was not displayed after retry");
                refresh.complete(state,t,false);
                check(refresh.due(state,t+.26),"Stage five failed texture update froze presentation");
                refresh.complete(state,t+.26,true);
                state.waitingForReady=true;state.collecting=false;state.isRetry=true;
                state.leftSamples=state.rightSamples=0;
                check(refresh.due(state,t+1),"Stage five retry card was suppressed");
                refresh.complete(state,t+1,true);
            }
            state.done=true;state.success=true;state.isRetry=false;
            check(refresh.due(state,220),"Stage five completion card was suppressed");

        }
        auto oneAxis=samples(2);check(!calibrateControllers(oneAxis,fitted).valid,"Reject single-axis orientation degeneracy");
        auto missing=input;std::erase_if(missing,[](auto &s){return s.device==2;});
        check(!calibrateControllers(missing,fitted).valid,"Require both tracked controllers");
        auto nan=input;nan[3].camera.x=std::numeric_limits<double>::quiet_NaN();
        check(!calibrateControllers(nan,fitted).valid,"Reject nonfinite observations");
        auto noisy=input;
        for(size_t i=0;i<noisy.size();++i) {
            noisy[i].camera+=V3{std::sin(i*1.7),std::cos(i*1.3),std::sin(i*.9)}*.003;
            if(i%23==0)noisy[i].camera+=V3{.12,-.1,.08};
        }
        auto robust=calibrateControllers(noisy,fitted);
        check(robust.valid && norm(robust.transform.t-truth.t)<.025,"Robust fit tolerates small Kinect noise and isolated bad samples");
        auto badHand=input;
        for(size_t i=0;i<badHand.size();++i)if(badHand[i].device==1 && (i/10)%2==0)badHand[i].camera+=V3{.3,-.25,.2};
        check(!calibrateControllers(badHand,fitted).valid,"A good controller cannot hide a bad controller");
        // Change controller origins/axes without any headset identity or preset.
        for(auto &s:input) {
            Q rebase=axisAngle({0,0,1},.7);
            s.pose.q=s.pose.q*rebase;
            s.pose.p=truth.apply(s.camera)-s.pose.q.rotate(expected[s.device]*1.4);
        }
        auto other=calibrateControllers(input,fitted);
        check(other.valid && norm(fitted[1]-expected[1]*1.4)<1e-5,"Different controller origin conventions do not need a headset preset");

        // Eight compact poses matching the spoken routine, not a large room walk.
        std::vector<AlignmentObservation> guided;
        for(int k=0;k<8;++k)for(int d=1;d<=2;++d)for(int tick=0;tick<8;++tick) {
            double side=d==1?-1.:1.;
            V3 c{side*.28,.1+(k==1?.2:k==3?(d==1?.2:0):k==7?(d==2?.2:0):0),2-(k==2?.25:0)};
            Q q;
            if(k==4)q=axisAngle({0,0,1},side*1.2);
            if(k==5)q=axisAngle({1,0,0},-.9);
            if(k==6)q=axisAngle({0,1,0},-side*1.);
            if(k==7)q=axisAngle({0,0,1},-side*.8)*axisAngle({1,0,0},.4);
            guided.push_back({100+k+tick*.15,d,c,{truth.apply(c)-q.rotate(expected[d]),q,true},{}});
        }
        auto compact=calibrateControllers(guided,fitted);
        if(!compact.valid)std::cerr<<compact.reason<<'\n';
        check(compact.valid && norm(compact.transform.t-truth.t)<1e-5,"Eight compact guided poses identify transform and offsets");

        AlignmentSession session;session.reset(true);Settings settings;
        auto clean=samples();
        for(int k=0;k<16;++k)for(int tick=0;tick<40;++tick) {
            Frame f;f.host=100+k*2+tick/30.;f.vr.host=f.host;
            Body b;b.id=42;
            for(int d=1;d<=2;++d) {
                auto &s=clean[k*10+(d-1)*5];int joint=d==1?LWrist:RWrist;
                b.joints[joint].p=s.camera;b.joints[joint].confidence=1;
                f.vr.devices[d]=s.pose;
            }
            // Head confidence is zero and HMD pose is invalid throughout.
            f.bodies.push_back(b);session.add(f,42,settings);
        }
        auto held=session.finish();check(held.valid,"Live sample collection accepts head entirely absent");
        check(session.agreement(held).find("not used; head visibility is optional")!=std::string::npos,"Feedback does not ask for head visibility");
        check(norm(session.offsets()[2]-expected[2])<1e-5,"Session exposes accepted wrist offsets for persistence");

        AlignmentSession hidden;hidden.reset(true);
        for(int k=0;k<40;++k) {
            Frame f;f.host=100+k/30.;f.vr.host=f.host;Body b;b.id=42;f.bodies.push_back(b);
            PosePrior prior;prior.host=f.host;prior.valid=true;prior.selectedId=42;
            for(int d=1;d<=2;++d){int j=d==1?LWrist:RWrist;prior.available[j]=true;prior.points[j]={d*.3,0,2};f.vr.devices[d]={{d*.3,0,2},{},true};}
            hidden.add(f,42,settings,&prior);
        }
        check(hidden.size()==0,"Hidden learned wrists without depth support are not calibration evidence");
        TiltLimiter limiter;
        check(TiltLimiter::target(26,1)==27 && TiltLimiter::target(-26,-1)==-27,"Motor targets clamp at hardware limits");
        for(int i=0;i<15;++i){check(limiter.claim(i*2.),"Permit spaced motor commands");check(!limiter.claim(i*2.+.5),"Reject rapid repeat motor commands");}
        check(!limiter.ready(47.9) && limiter.ready(48),"Enforce 20 second rest after 15 changes");
        check(limiter.claim(48) && limiter.consecutive==1,"Rest resets motor command count");
        auto path=std::filesystem::temp_directory_path()/"kf-controller-calibration-test.txt";
        saveCalibration(path,held,settings);Calibration restored;Settings restoredSettings;
        check(loadCalibration(path,restored,restoredSettings),"Accepted saved calibration remains loadable");
        held.valid=false;held.spread=0;saveCalibration(path,held,settings);
        check(!loadCalibration(path,restored,restoredSettings),"Persisted camera-move invalidation survives restart");
        settings.deviceOffsets=expected;saveCalibration(path,held,settings,true);bool known=false;
        check(!loadCalibration(path,restored,restoredSettings,&known) && known,"Learned offsets survive camera alignment invalidation");
        saveCalibration(path,held,settings,false);loadCalibration(path,restored,restoredSettings,&known);
        check(!known,"Unknown offsets do not silently enable quick alignment");
        {
            Calibration accepted;accepted.valid=true;accepted.spread=.2;accepted.rms=.025;
            accepted.transform={axisAngle({0,1,0},.3),{.4,.1,-1}};
            VrSample vr;vr.epoch=9;vr.rawTransformValid=true;
            vr.standingToRaw={axisAngle({0,1,0},-.2),{1,0,0}};
            bindTrackingReference(accepted,vr);
            SavedAlignment saved;saved.remember(accepted);
            // HMD/controller tracking may disappear while the physical reference
            // remains valid. Confirming it must not require fresh pose samples.
            vr.devices={};auto confirmed=saved.confirm(vr);
            check(confirmed.valid && saved.available(),"Headset removal discarded accepted alignment");
            check(norm(confirmed.transform.t-accepted.transform.t)<1e-9,"Confirmation changed calibration transform");
            auto sleeping=vr;sleeping.rawTransformValid=false;
            check(!saved.confirm(sleeping).valid && saved.available(),"Temporary VR loss discarded saved alignment");
            check(saved.confirm(vr).valid,"Wake-up in same reference could not restore alignment");
            auto dragged=vr;dragged.standingToRaw.t.y+=1;
            confirmed=saved.confirm(dragged);
            check(confirmed.valid && norm(confirmed.standingToRaw.t-accepted.standingToRaw.t)<1e-9,"Confirmation rebound alignment to virtual space drag");
            auto restarted=vr;restarted.epoch++;
            check(!saved.confirm(restarted).valid && saved.confirm(restarted).reason.find("restarted")!=std::string::npos,"Real origin change silently rebound old alignment");
            // An unfinished/failed attempt operates on a separate live result.
            auto attempted=accepted;attempted.valid=false;attempted.spread=0;
            check(saved.confirm(vr).valid,"Unaccepted attempt overwrote saved transform");
            saved.invalidate();check(!saved.confirm(vr).valid,"Camera/offset invalidation reused stale alignment");
            saved.remember(accepted);saved.forgetLiveReference();
            check(saved.confirm(restarted).valid,"Explicit saved confirmation after app/capture restart failed");
            saved.remember(attempted,false);check(!saved.confirm(vr).valid,"Another sensor reused previous accepted alignment");
            attempted=accepted;attempted.rms=.2;saved.remember(attempted);
            check(!saved.confirm(vr).valid,"Bad saved quality accepted");
            saveCalibration(path,accepted,settings,true);Calibration loaded;
            check(loadCalibration(path,loaded,restoredSettings),"Accepted alignment failed to reload");
            saved.remember(loaded);
            check(saved.confirm(vr).valid,"Disk-loaded accepted alignment could not be confirmed");
        }
        std::filesystem::remove(path);
        std::cout<<checks<<" controller calibration and motor limit checks passed. No hardware moved.\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
