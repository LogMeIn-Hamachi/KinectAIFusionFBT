#include "body_tracker.hpp"
#include "sam3d_geometry.hpp"
#include "osc_tracking.hpp"
#include <iostream>
using namespace kf;
#include "foot_frames_fixture.hpp"
void check(bool b,const char* msg){if(!b)throw std::runtime_error(msg);}
PosePrior sample(double time,bool depth=true) {
    PosePrior p;p.host=time;p.selectedId=7;p.articulationValid=true;p.valid=depth;p.rootAnchors=depth?6:0;
    p.registrationRms=.02;p.available.fill(true);p.supported.fill(depth);
    auto& v=p.points;for(auto& a:v)a={0,1,2};
    v[Hip]={0,1,2};v[LHip]={-.12,1,2};v[RHip]={.12,1,2};v[Neck]={0,1.5,2};v[Head]={0,1.68,2};
    for(int s=0;s<2;++s){double sign=s?1.:-1.;
        v[LShoulder+s]={sign*.22,1.48,2};v[LElbow+s]={sign*.35,1.2,2};v[LWrist+s]={sign*.5,1.,2};
        v[LKnee+s]={sign*.12,.55,2};v[LAnkle+s]={sign*.12,.10,2};
        v[LHeel+s]={sign*.12,.035,2.045};v[LToe+s]={sign*.12,.035,1.85};v[LSmallToe+s]={sign*.18,.035,1.87};
    }
    p.imageRoot=v[Hip];return p;
}
int main(){try {
    // Smooth constant-speed input through the real body/estimator/output path.
    // Detect stop-go presentation even when average position accuracy is good.
    for(double fps:{15.,30.})for(double speed:{.03,.06,.2}) {
        BodyTracker body;Estimator estimator;estimator.select(7);
        estimator.settings.contacts=false;estimator.settings.vrConstraints=false;
        Calibration calibration;calibration.valid=true;VrSample vr;vr.rawTransformValid=true;
        bindTrackingReference(calibration,vr);
        OscTracking output;State state;int lastSample=-1;double sum=0,sum2=0,velocitySum=0;int count=0;
        V3 previous{};
        for(int tick=0;tick<720;++tick) {
            double elapsed=tick/120.;int index=int(std::floor(elapsed*fps+1e-8));
            if(index!=lastSample) {
                Frame frame;frame.host=100+index/fps;frame.vr=vr;
                auto pose=sample(frame.host);
                for(auto& point:pose.points)point.x+=speed*index/fps;
                pose.imageRoot=pose.points[Hip];
                Body sdk;sdk.id=7;sdk.player=1;
                for(int j=0;j<J;++j)sdk.joints[j]={pose.points[j],.9,.02,1};
                frame.bodies={sdk};
                auto fitted=body.update(pose,7,frame,calibration,estimator.settings);
                state=estimator.process(frame,nullptr,calibration,&fitted);lastSample=index;
            }
            auto trackers=output.update(state,calibration,vr,100+elapsed+.03);
            check(trackers[0].valid,"Constant movement lost tracker validity");
            if(elapsed>3) {
                double renderedSpeed=(trackers[0].p.x-previous.x)*120;
                sum+=renderedSpeed;sum2+=renderedSpeed*renderedSpeed;velocitySum+=state.trackers[0].velocity.x;++count;
            }
            previous=trackers[0].p;
        }
        double mean=sum/count,variation=std::sqrt(std::max(0.,sum2/count-mean*mean))/speed;
        check(std::abs(velocitySum/count-speed)<speed*.01,"Steady motion velocity was suppressed");
        check(variation<.02,"Straight motion still pulses between camera samples");
        std::cout<<"Motion ramp "<<fps<<" Hz / "<<speed<<" m/s: predicted speed "<<velocitySum/count
            <<", rendered speed variation "<<variation<<"\n";
    }
    for(double fps:{15.,30.})for(V3 axis:{V3{1,0,0},V3{0,1,0},V3{0,0,1}}) {
        BodyTracker body;Estimator estimator;estimator.select(7);
        estimator.settings.contacts=false;estimator.settings.vrConstraints=false;
        Calibration calibration;calibration.valid=true;VrSample vr;vr.rawTransformValid=true;
        bindTrackingReference(calibration,vr);OscTracking output;State state;int lastSample=-1;
        V3 previous{};double worstVelocityError=0;int measured=0;
        for(int tick=0;tick<1440;++tick) {
            double elapsed=tick/120.;int index=int(std::floor(elapsed*fps+1e-8));
            if(index!=lastSample) {
                double sampleTime=index/fps,phase=std::fmod(sampleTime,6.);
                auto pose=sample(100+sampleTime);
                double distance=.06*(phase<3?phase:6-phase);
                for(int j:{LAnkle,LHeel,LToe,LSmallToe})pose.points[j]+=V3{0,.2,0}+axis*distance;
                pose.points[LKnee]+=V3{0,.1,0}+axis*(distance*.5);
                Frame frame;frame.host=pose.host;frame.vr=vr;
                Body sdk;sdk.id=7;sdk.player=1;
                for(int j=0;j<J;++j)sdk.joints[j]={pose.points[j],.9,.02,1};frame.bodies={sdk};
                auto fitted=body.update(pose,7,frame,calibration,estimator.settings);
                state=estimator.process(frame,nullptr,calibration,&fitted);lastSample=index;
            }
            auto trackers=output.update(state,calibration,vr,100+elapsed+.03);
            check(trackers[1].valid,"Lifted leg motion lost tracking");
            double phase=std::fmod(elapsed,6.),sinceTurn=std::fmod(elapsed,3.);
            if(elapsed>1 && sinceTurn>1.2 && sinceTurn<2.8) {
                double expected=phase<3?.06:-.06;
                double speed=dot(trackers[1].p-previous,axis)*120;
                worstVelocityError=std::max(worstVelocityError,std::abs(speed-expected));++measured;
            }
            previous=trackers[1].p;
        }
        check(measured>100 && worstVelocityError<.006,"Lifted leg still pulses during straight movement or reversal recovery");
        std::cout<<"Leg axis "<<axis.x<<','<<axis.y<<','<<axis.z<<" at "<<fps<<" Hz: max steady velocity error "<<worstVelocityError<<" m/s\n";
    }
    // Verify native MHR foot axes against independently exported neutral mesh
    // landmarks, including Kinect's reflected camera and arbitrary body turns.
    for(bool reflected:{false,true})for(int turn=0;turn<12;++turn) {
        Sam3dPrediction p;p.hasFootRotations=true;
        Sam3dCamera cam;cam.valid=true;cam.rows={V3{reflected?-1.:1.,0,0},V3{0,1,0},V3{0,0,1}};
        Q rotation=axisAngle(unit(V3{1,.7,.3}),turn*.4);
        for(int side=0;side<2;++side)for(int c=0;c<3;++c) {
            V3 v=rotation.rotate({neutralFeet[side][c],neutralFeet[side][3+c],neutralFeet[side][6+c]});
            p.footRotations[side].a[0][c]=v.x;p.footRotations[side].a[1][c]=v.y;p.footRotations[side].a[2][c]=v.z;
        }
        auto actual=sam3dFootOrientations(p,cam);
        for(int side=0;side<2;++side) {
            auto transform=[&](V3 v){v=rotation.rotate(v);return cam.toSdk({v.x,-v.y,-v.z});};
            auto big=transform(neutralLandmarks[side][0]),small=transform(neutralLandmarks[side][1]),heel=transform(neutralLandmarks[side][2]);
            V3 z=unit(heel-big),r=(small-big)*(side?1.:-1.);r=unit(r-z*dot(r,z));
            Q expected=basis(r,cross(z,r));
            check(actual[side] && std::abs(dot(expected,*actual[side]))>.99999,"Native foot axes/reflection mismatch");
        }
        p.footRotations[0].a[0][0]=std::numeric_limits<double>::quiet_NaN();
        check(!sam3dFootOrientations(p,cam)[0],"Nonfinite foot orientation accepted");
    }
    for(double fps:{15.,22.,30.}) {
        {
            // Two identical physical sequences, one with continuous OVR drag/turn.
            // Exercise body correction, contact and final tracker filtering together.
            BodyTracker normal,moved;Estimator a,b;a.select(7);b.select(7);
            Calibration cal;cal.valid=true;cal.transform={axisAngle({0,1,0},.6),{.4,.2,-1}};
            VrSample reference;reference.epoch=4;reference.rawTransformValid=true;
            reference.standingToRaw={axisAngle({0,1,0},-.3),{1,.1,-2}};
            check(bindTrackingReference(cal,reference),"Missing test VR reference");
            for(int i=0;i<int(fps*6);++i) {
                Frame fa;fa.host=300+i/fps;fa.floor={{0,1,0},0,true};fa.vr=reference;fa.vr.host=fa.host;
                auto p=sample(fa.host,i%int(fps*2)<fps);
                double sway=.08*std::sin(i/fps*2);
                for(int j:{Hip,LHip,RHip,Neck,Head,LShoulder,RShoulder,LElbow,RElbow,LWrist,RWrist})p.points[j].x+=sway;
                p.imageRoot=p.points[Hip];
                for(int side=0;side<2;++side){p.footSurfaces[side]={(side?1.:-1.)*.12,.1,1.96};p.footSurfaceValid[side]=true;}
                Body sdk;sdk.id=7;sdk.player=1;for(int j=0;j<J;++j)sdk.joints[j]={p.points[j],.9,.02,1};fa.bodies={sdk};
                for(int d=0;d<3;++d)fa.vr.devices[d]={cal.transform.apply(p.points[d==0?Head:d==1?LWrist:RWrist]+V3{.025,0,0}),cal.transform.q,true};
                Frame fb=fa;Rigid drag{axisAngle({0,1,0},.7*std::sin(i/fps)),{.7*std::sin(i/fps*2),1.2*std::sin(i/fps),-.4}};
                fb.vr.standingToRaw=composeRigid(reference.standingToRaw,inverseRigid(drag));
                for(auto& d:fb.vr.devices){d.p=drag.apply(d.p);d.q=drag.q*d.q;}
                auto pa=normal.update(p,7,fa,cal,a.settings),pb=moved.update(p,7,fb,cal,b.settings);
                check(pa.valid==pb.valid,"Playspace drag changed body visibility");
                auto sa=a.process(fa,nullptr,cal,&pa),sb=b.process(fb,nullptr,cal,&pb);
                for(int n=0;n<3;++n){
                    check(sa.trackers[n].valid==sb.trackers[n].valid,"Playspace drag changed tracker validity");
                    check(norm(sa.trackers[n].p-sb.trackers[n].p)<1e-8,"Playspace drag changed physical tracking/contact");
                    check(std::abs(dot(sa.trackers[n].q,sb.trackers[n].q))>.999999,"Playspace drag changed physical orientation");
                }
            }
        }
        {
            // Hip sway corrupts the shared translation and SAM foot estimates,
            // but selected-person ankle depth stays still. Exercise the complete
            // body/estimator path, including weak root depth with visible feet.
            BodyTracker body;Estimator estimator,legacy;estimator.select(7);legacy.select(7);Frame frame;
            frame.floor={{0,1,0},0,true};State result;V3 initial{};
            double maxSlide=0,legacySlide=0;bool preservedEvidence=false;
            for(int i=0;i<int(fps*6);++i) {
                frame.host=200+i/fps;auto p=sample(frame.host,i%int(fps*2)<fps);
                double x=i<fps?0:.12*std::sin((frame.host-201)*2*pi*.7);
                for(int j:{Hip,LHip,RHip,Neck,Head,LShoulder,RShoulder,LElbow,RElbow,LWrist,RWrist})p.points[j].x+=x;
                p.imageRoot=p.points[Hip];
                for(int side=0;side<2;++side) {
                    p.points[LKnee+side].x+=x*.5;
                    for(int j:{LAnkle+side,LHeel+side,LToe+side,LSmallToe+side})p.points[j].x+=x*.5;
                    p.footSurfaces[side]={(side?1.:-1.)*.12+(i%2?.002:-.002),.10,1.96};
                    p.footSurfaceValid[side]=true;p.footOrientationValid[side]=true;
                }
                Body sdk;sdk.id=7;sdk.player=1;for(int j=0;j<J;++j)sdk.joints[j]={p.points[j],.9,.02,1};frame.bodies={sdk};
                auto tracked=body.update(p,7,frame,{},{});
                check(tracked.valid,"Sway regression lost body");
                if(tracked.depthPredicted && tracked.footSurfaceValid[0])preservedEvidence=true;
                result=estimator.process(frame,nullptr,{},&tracked);
                auto old=tracked;old.footSurfaceValid={};auto previous=legacy.process(frame,nullptr,{},&old);
                if(i==int(fps)-1)initial=result.trackers[1].p;
                if(i>=fps) {
                    maxSlide=std::max(maxSlide,std::abs(result.trackers[1].p.x-initial.x));
                    legacySlide=std::max(legacySlide,std::abs(previous.trackers[1].p.x-initial.x));
                }
            }
            check(preservedEvidence,"Foot observations disappeared with weak root depth");
            check(maxSlide<.006,"Stationary depth-supported foot skated with noisy hip sway");
            check(legacySlide>.02,"Hip sway regression did not exercise the old skating failure");
            std::cout<<fps<<" Hz noisy hip sway slide "<<legacySlide*1000<<" -> "<<maxSlide*1000<<" mm\n";
            double initialY=result.trackers[1].p.y;
            for(int i=1;i<=int(fps*.2);++i) {
                frame.host+=1/fps;auto p=sample(frame.host);double lift=std::min(.2,i/fps);
                for(int j:{LKnee,LAnkle,LHeel,LToe,LSmallToe})p.points[j].y+=lift;
                p.footSurfaceValid={true,true};p.footSurfaces={V3{-.12,.10+lift,1.96},V3{.12,.10,1.96}};
                p.footOrientationValid={true,true};Body sdk;sdk.id=7;sdk.player=1;
                for(int j=0;j<J;++j)sdk.joints[j]={p.points[j],.9,.02,1};frame.bodies={sdk};
                auto tracked=body.update(p,7,frame,{},{});result=estimator.process(frame,nullptr,{},&tracked);
            }
            check(result.trackers[1].p.y>initialY+.10,"Independent contact prevented a deliberate foot lift");
        }
        {
            Plane floor;floor.valid=true;PlantStabilizer plant;V3 surface{0,.09,2};
            auto settle=[&](){plant={};for(int i=0;i<int(fps);++i)plant.update({0,.01,2},{},{},floor,true,1/fps,surface);};
            settle();check(plant.planted && plant.surfaceAnchored,"Independent depth plant did not acquire");
            // A lateral model velocity is not physical foot movement.
            auto offset=plant.update({.09,.01,2},{},{.3,0,0},floor,false,1/fps,surface);
            check(plant.planted && std::abs(offset.x+.09)<1e-9,"Root drift released independent foot anchor");
            // A real slow floor slide must release, even without lifting.
            double released=0;
            for(int i=1;i<=int(fps*.5);++i){double shift=.18*i/fps;
                plant.update({shift,.01,2},{},{.18,0,0},floor,true,1/fps,surface+V3{shift,0,0});
                if(!plant.planted){released=i/fps;break;}}
            check(released>0 && released<=.27,"Independent anchor stuck to a real sliding step");
            settle();plant.update({0,.01,2},{},{},floor,true,1/fps,surface+V3{0,.04,0});
            check(!plant.planted,"Depth-observed lift waited for smoothed model motion");
            settle();plant.update({0,.01,2},{},{},floor,true,1/fps,surface+V3{0,.3,0});
            check(!plant.planted,"Large depth-observed lift was mistaken for lost visibility");
            settle();plant.update({0,.01,2},axisAngle({0,1,0},.4),{},floor,true,1/fps,surface);
            check(!plant.planted,"Independent depth plant blocked a foot pivot");
            settle();plant.update({0,.01,2},{},{},floor,true,1/fps,surface+V3{.03,0,0});
            plant.update({0,.01,2},{},{},floor,true,1/fps,surface);
            check(plant.planted,"Single small depth outlier released a foot");
            settle();plant.update({0,.01,2},{},{},floor,true,1/fps);
            check(plant.planted,"One missing surface released plant");
            for(int i=0;i<int(fps*.3);++i)plant.update({0,.01,2},{},{},floor,false,1/fps);
            check(!plant.planted,"Missing ankle depth held a planted foot indefinitely");
            settle();plant.update({.2,.01,2},{},{},floor,true,1/fps,surface);
            check(!plant.planted,"Independent contact hid an excessive model divergence");
            // Tilted cameras must lock the floor plane, not camera X/Y axes.
            Q tilt=axisAngle({1,0,0},.6);Plane sloped{tilt.rotate({0,1,0}),.8,true};
            V3 origin=sloped.n*-.8;plant={};
            for(int i=0;i<int(fps);++i)plant.update(origin+tilt.rotate({0,.01,2}),tilt,{},sloped,true,1/fps,origin+tilt.rotate(surface));
            auto sole=origin+tilt.rotate({.09,.01,2.04});
            auto correction=plant.update(sole,tilt,tilt.rotate({.3,0,.1}),sloped,true,1/fps,origin+tilt.rotate(surface));
            auto drift=sole+correction-plant.anchor;
            check(plant.planted && norm(drift-sloped.n*dot(drift,sloped.n))<1e-9,"Tilted camera broke planted horizontal position");
            std::cout<<fps<<" Hz real sliding-step release "<<released*1000<<" ms\n";
        }
        {
            // Same camera pose and headset POSITION, different headset gaze.
            // Compare the full body -> tracker pipeline without floor locking,
            // so contact cannot hide an orientation-to-position coupling.
            BodyTracker neutral,looking;Estimator a,b;a.select(7);b.select(7);
            Settings settings;settings.contacts=false;a.settings=b.settings=settings;
            Calibration cal;cal.valid=true;cal.transform={axisAngle({0,1,0},.7)*axisAngle({1,0,0},-.3),{.4,1.2,-1}};
            double maxFootDifference=0;PosePrior pa,pb;Frame fa,fb;
            for(int i=0;i<int(fps*6);++i) {
                double time=100+i/fps;auto p=sample(time);p.available[Head]=false;
                fa.host=time;fa.vr.host=time;fa.vr.epoch=12;
                fa.vr.devices[0]={cal.transform.apply(p.points[Neck]+V3{0,.2,-.08}),cal.transform.q,true};
                for(int d=1;d<3;++d)fa.vr.devices[d]={cal.transform.apply(p.points[d==1?LWrist:RWrist]),{},i<fps*2};
                Body sdk;sdk.id=7;sdk.player=1;for(int j=0;j<J;++j)sdk.joints[j]={p.points[j],.9,.02,1};fa.bodies={sdk};fb=fa;
                if(i>=fps*2)fb.vr.devices[0].q=cal.transform.q*axisAngle({1,0,0},1.4*std::sin((time-102)*3))*
                    axisAngle({0,1,0},1.5*std::sin((time-102)*2))*axisAngle({0,0,1},.6*std::sin((time-102)*4));
                pa=neutral.update(p,7,fa,cal,settings);pb=looking.update(p,7,fb,cal,settings);
                auto sa=a.process(fa,nullptr,cal,&pa),sb=b.process(fb,nullptr,cal,&pb);
                if(i>=fps*2) {
                    check(pa.valid && pb.valid,"Head-look test lost camera body");
                    for(int side=1;side<3;++side)maxFootDifference=std::max(maxFootDifference,norm(sa.trackers[side].p-sb.trackers[side].p));
                    check(norm(sa.trackers[0].p-sb.trackers[0].p)<1e-9,"Headset gaze moved pelvis");
                }
            }
            check(maxFootDifference<1e-9,"Looking down/sideways/tilting raised or moved feet");
            // Retain actual headset/body translation after removing gaze coupling.
            for(int i=0;i<int(fps);++i) {
                fa.host+=1/fps;fa.vr.host=fa.host;auto p=sample(fa.host);p.available[Head]=false;
                for(auto& v:p.points)v.x+=.25;p.imageRoot=p.points[Hip];
                fa.vr.devices[0].p=cal.transform.apply(p.points[Neck]+V3{0,.2,-.08});
                pa=neutral.update(p,7,fa,cal,settings);
            }
            check(pa.valid && std::abs(pa.points[Hip].x-.25)<.025,"Position-only anchoring stopped following actual translation");
            std::cout<<fps<<" Hz headset gaze foot displacement "<<maxFootDifference*1000<<" mm\n";
        }
        {
            // Hip sway with truly stationary feet. The feet must not inherit
            // lag from a changing pelvis coordinate frame, even without contact.
            BodyTracker sway;Frame frame;PosePrior output;double maxSlide=0;
            for(int i=0;i<int(fps*5);++i) {
                frame.host=1+i/fps;auto p=sample(frame.host);
                double x=i<fps?0:.12*std::sin((frame.host-2)*2*pi*.7);
                for(int j:{Hip,LHip,RHip,Neck,Head,LShoulder,RShoulder,LElbow,RElbow,LWrist,RWrist})p.points[j].x+=x;
                p.points[LKnee].x+=x*.5;p.points[RKnee].x+=x*.5;p.imageRoot=p.points[Hip];
                output=sway.update(p,7,frame,{},{});
                if(i>fps*2)maxSlide=std::max(maxSlide,std::abs(output.points[LAnkle].x+.12));
            }
            check(maxSlide<.012,"Pelvis-relative smoothing made stationary feet skate during hip sway");
            std::cout<<fps<<" Hz hip sway maximum foot displacement "<<maxSlide*1000<<" mm\n";
        }
        {
            BodyTracker body;Estimator estimator;estimator.select(7);Frame frame;frame.floor={{0,1,0},0,true};State result;
            for(int i=0;i<int(fps*2);++i) {
                frame.host=10+i/fps;auto p=sample(frame.host);Body sdk;sdk.id=7;sdk.player=1;
                for(int j=0;j<J;++j)sdk.joints[j]={p.points[j],.9,.02,1};frame.bodies={sdk};
                auto tracked=body.update(p,7,frame,{},{});result=estimator.process(frame,nullptr,{},&tracked);
            }
            check(std::abs(result.trackers[1].p.y-.035)<.006,"Foot tracker is not mounted over the grounded sole");
            check(result.trackers[1].p.z<result.body.joints[LAnkle].p.z-.04,"Foot tracker remains at shin instead of instep");
            double initialY=result.trackers[1].p.y;
            for(int i=1;i<=int(fps*.2);++i) {
                frame.host+=1/fps;auto p=sample(frame.host);
                for(int j:{LKnee,LAnkle,LHeel,LToe,LSmallToe})p.points[j].y+=std::min(.20,i/fps);
                Body sdk;sdk.id=7;sdk.player=1;for(int j=0;j<J;++j)sdk.joints[j]={p.points[j],.9,.02,1};frame.bodies={sdk};
                auto tracked=body.update(p,7,frame,{},{});result=estimator.process(frame,nullptr,{},&tracked);
            }
            check(result.trackers[1].p.y>initialY+.10,"Rest stabilization prevented a deliberate foot lift");
        }
        {
            BodyTracker common;Frame frame;PosePrior output;
            double lo=10,hi=-10;
            for(int i=0;i<int(fps*4);++i) {
                double time=1+i/fps,sign=i%2?1.:-1.;auto p=sample(time);
                auto root=p.points[Hip];auto turn=axisAngle({0,0,1},sign*.04);
                for(auto& v:p.points)v=root+turn.rotate(v-root)+V3{sign*.02,0,0};
                p.imageRoot=p.points[Hip];frame.host=time;
                output=common.update(p,7,frame,{},{});
                if(i>fps*2){lo=std::min(lo,output.points[LAnkle].x);hi=std::max(hi,output.points[LAnkle].x);}
            }
            check(hi-lo<.025,"Shared root/pelvis wobble bypassed body smoothing");
            check(output.resting[LAnkle],"Resting shared jitter was classified as motion");
            double started=frame.host,lag=0;
            for(int i=1;i<=int(fps);++i) {
                frame.host=started+i/fps;auto p=sample(frame.host);
                for(auto& v:p.points)v.x+=i/fps;p.imageRoot=p.points[Hip];
                output=common.update(p,7,frame,{},{});
                if(i>fps*.5)lag=std::max(lag,i/fps-output.points[Hip].x);
            }
            check(lag<.10,"Rest smoothing added excessive movement lag");
            std::cout<<fps<<" Hz shared-body jitter "<<(hi-lo)*1000<<" mm; moving root lag "<<lag*1000<<" mm\n";
        }
        BodyTracker tracker;Frame f;Settings settings;PosePrior out;
        double lo=10,hi=-10,maximumBoneError=0;std::vector<double> costs;
        for(int i=0;i<int(fps*4);++i){double t=1+i/fps;auto p=sample(t);
            if(i>fps){p.points[LAnkle].x+=i%2?.035:-.035;p.points[LKnee].x+=i%2?.025:-.025;}
            f.host=t;double start=now();out=tracker.update(p,7,f,{},settings);costs.push_back((now()-start)*1000);
            check(out.valid && out.bodyFitted,"Measured body invalid");
            if(i>fps*2){lo=std::min(lo,out.points[LAnkle].x);hi=std::max(hi,out.points[LAnkle].x);}
            maximumBoneError=std::max(maximumBoneError,std::abs(norm(out.points[LKnee]-out.points[LAnkle])-.45));
        }
        check(hi-lo<.025,"Articulation noise did not settle");
        check(maximumBoneError<.015,"Fitted leg length changed excessively");
        // Alternating whole-leg hypotheses must not be mistaken for rapid kicks;
        // a sustained real bend must still be adopted promptly.
        for(int i=0;i<int(fps);++i){f.host+=1/fps;auto p=sample(f.host);
            if(i%2)for(int j:{LKnee,LAnkle,LHeel,LToe,LSmallToe})p.points[j].z-=.4;
            out=tracker.update(p,7,f,{},settings);
            check(out.points[LAnkle].z>1.85,"Alternating leg branch made the foot jump");
        }
        for(int i=0;i<int(fps*.6);++i){f.host+=1/fps;auto p=sample(f.host);
            for(int j:{LKnee,LAnkle,LHeel,LToe,LSmallToe})p.points[j].z-=.4;
            out=tracker.update(p,7,f,{},settings);
        }
        check(out.points[LAnkle].z<1.8,"Sustained real leg change did not recover");
        double last=f.host;
        for(int i=1;i<int(fps*1.5);++i){f.host=last+i/fps;auto p=sample(f.host,false);out=tracker.update(p,7,f,{},settings);
            check(out.valid && out.depthPredicted,"Fresh articulation failed during depth gap");
            check(std::abs(out.points[Hip].z-2)<.005,"Unobserved root drifted");
            check(out.rootAnchors==0 && !out.supported[LAnkle],"Predicted root claimed depth support");
        }
        for(int i=1;i<int(fps);++i){f.host+=1/fps;out=tracker.update(sample(f.host,false),7,f,{},settings);}
        check(!out.valid,"Unobserved metric position never expired");
        tracker.reset();f.host=10;out=tracker.update(sample(10,false),7,f,{},settings);
        check(!out.valid,"Unanchored monocular pose initialized world position");
        f.host+=1/fps;out=tracker.update(sample(f.host),7,f,{},settings);
        // Headset movement cannot drag a valid camera body, even with a hidden
        // head, a changing estimated neck length and two visible controllers.
        Calibration cal;cal.valid=true;tracker.reset();f.vr={};
        for(int i=0;i<int(fps*5);++i) {
            f.host=30+i/fps;f.vr.host=f.host;f.vr.epoch=2;auto p=sample(f.host);
            p.available[Head]=false;p.points[Neck].y+=.12*std::sin(i/fps*3);
            f.vr.devices[0]={{.3*std::sin(i/fps),1.2+.4*std::sin(i/fps*2),2.1},axisAngle({1,0,0},1.2),true};
            for(int d=1;d<=2;++d)f.vr.devices[d]={p.points[d==1?LWrist:RWrist],{},true};
            out=tracker.update(p,7,f,cal,settings);
            check(out.valid && norm(out.points[Hip]-V3{0,1,2})<1e-9,"Headset/neck variation dragged camera pelvis");
        }
        // Both controllers may correct a small horizontal bias, but cannot
        // lift the body or turn correction into an ever-growing root feedback.
        tracker.reset();f.vr={};
        double previousX=0,maxSpeed=0;
        for(int i=0;i<int(fps*8);++i) {
            f.host=40+i/fps;f.vr.host=f.host;auto p=sample(f.host);
            for(int d=1;d<=2;++d)f.vr.devices[d]={p.points[d==1?LWrist:RWrist]+V3{.08,.06,0},{},true};
            out=tracker.update(p,7,f,cal,settings);
            check(out.valid && std::abs(out.points[Hip].y-1)<1e-9,"Controller drift correction changed body height");
            check(out.points[Hip].x<=.035001,"Horizontal controller correction exceeded its bound");
            maxSpeed=std::max(maxSpeed,(out.points[Hip].x-previousX)*fps);previousX=out.points[Hip].x;
        }
        check(out.vrAnchored && out.points[Hip].x>.025 && maxSpeed<.016,"Gentle correction missing or moving too fast");
        // Actual depth-observed crouching follows even if the headset stays up.
        for(int i=0;i<int(fps);++i) {
            f.host+=1/fps;f.vr.host=f.host;auto p=sample(f.host);
            for(auto& v:p.points)v.y-=.25;p.imageRoot=p.points[Hip];
            f.vr.devices={};f.vr.devices[0]={{0,1.7,2},{},true};
            out=tracker.update(p,7,f,cal,settings);
        }
        check(out.valid && std::abs(out.points[Hip].y-.75)<.025,"Headset pinned the body above a camera-observed crouch");
        // No headset or wrist constraint may keep unobserved depth alive forever.
        for(int i=0;i<int(fps*3);++i) {
            f.host+=1/fps;f.vr.host=f.host;auto p=sample(f.host,false);
            for(int d=1;d<=2;++d)f.vr.devices[d]={p.points[d==1?LWrist:RWrist],{},true};
            out=tracker.update(p,7,f,cal,settings);
        }
        check(!out.valid,"VR devices masked prolonged loss of metric body observations");
        // A solitary inconsistent controller must not move the shared root.
        tracker.reset();f.vr={};
        for(int i=0;i<int(fps);++i){f.host=40+i/fps;f.vr.host=f.host;
            f.vr.devices[1]={{2,2,2},{},true};out=tracker.update(sample(f.host),7,f,cal,settings);}
        check(norm(out.points[Hip]-V3{0,1,2})<.01,"Bad controller moved body root");
        f.host+=1/fps;auto other=sample(f.host,false);other.selectedId=8;
        check(!tracker.update(other,8,f,cal,settings).valid,"Body history leaked between people");
        PlantStabilizer plant;Plane floor;floor.valid=true;V3 offset{};
        for(int i=0;i<int(fps);++i)offset=plant.update({0,.01,2},{},{},floor,true,1/fps);
        check(plant.planted,"Stationary observed foot did not plant");
        for(int i=0;i<int(fps*.3);++i)offset=plant.update({.02,.01,2},{},{},floor,true,1/fps);
        check(std::abs(offset.x+.02)<.003,"Plant did not resist horizontal skating");
        check(std::abs(offset.y+.01)<.003,"Observed resting sole remained floating above the floor");
        plant.update({.02,.01,2},{},{},floor,false,1/fps);
        check(plant.planted,"One missing depth sample released an established plant");
        plant.update({.02,.09,2},{},{0,.3,0},floor,true,1/fps);
        check(!plant.planted,"Lifting foot remained planted");
        for(int i=0;i<int(fps);++i)plant.update({0,.01,2},{},{},floor,true,1/fps);
        plant.update({0,.01,2},axisAngle({0,1,0},.4),{},floor,true,1/fps);
        check(!plant.planted,"Pivot did not release foot");
        State state;state.host=20;state.lastObserved=20;state.trackers[0].valid=true;
        state.trackers[0].observedHost=20;state.trackers[0].velocity={1,0,0};
        state.trackers[1].valid=true;state.trackers[1].observedHost=19;
        auto held=deliveryState(state,20.15);
        check(held.trackers[0].valid && !held.trackers[1].valid,"Tracker freshness was coupled");
        check(std::abs(held.trackers[0].p.x-.04)<1e-9,"Stall extrapolated beyond 40 ms");
        check(!deliveryState(state,20.25).trackers[0].valid,"Output never expired");
        std::sort(costs.begin(),costs.end());
        std::cout<<fps<<" Hz: ankle peak-to-peak "<<(hi-lo)*1000<<" mm from 70 mm input; bone error "<<maximumBoneError*1000
            <<" mm; body fit median "<<costs[costs.size()/2]<<" ms, max "<<costs.back()<<" ms\n";
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
