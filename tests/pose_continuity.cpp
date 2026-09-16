#include "pose_continuity.hpp"
#include <iostream>
using namespace kf;
void check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
PosePrior pose(double t,bool depth=true) {
    PosePrior p;p.host=t;p.selectedId=7;p.valid=depth;p.articulationValid=true;
    p.rootAnchors=depth?5:0;p.registrationRms=.02;p.available.fill(true);p.supported.fill(depth);
    for(auto& v:p.points)v={0,1,2};
    p.points[Hip]={0,1,2};p.points[LAnkle]={-.1,0,2};p.points[RAnkle]={.1,0,2};
    return p;
}
int main(){try{
    for(double fps:{15.,22.,30.}) {
        PoseContinuity continuity;
        auto p=pose(1);auto out=continuity.update(p,7,1);
        check(out.valid && !out.depthPredicted,"First measured root rejected");
        Frame frame;frame.host=1;Body sdk;sdk.id=7;sdk.player=1;
        for(auto& j:sdk.joints)j={{.5,.5,2.3},.9,.03,1};
        frame.bodies.push_back(sdk);Estimator estimator;estimator.select(7);
        estimator.process(frame,nullptr,{},&out);
        unsigned bridges{};
        for(int i=1;i<=int(fps*.3);++i) {
            p=pose(1+i/fps,false);
            // The image model's absolute scale/translation fluctuates, but its
            // ray and fresh leg articulation remain usable.
            for(auto& v:p.points)v=v*1.3;
            p.points[LAnkle].y+=.2;
            out=continuity.update(p,7,p.host);
            check(out.valid && out.depthPredicted && out.rootAnchors==0,"Brief depth loss switched to SDK");
            check(std::none_of(out.supported.begin(),out.supported.end(),[](bool b){return b;}),"Predicted depth claimed observed support");
            check(std::abs(out.points[Hip].z-2)<1e-9,"Monocular depth jump moved the retained root");
            check(norm((out.points[LAnkle]-out.points[Hip])-(p.points[LAnkle]-p.points[Hip]))<1e-9,"Fresh articulation frozen or distorted");
            frame.host=p.host;auto state=estimator.process(frame,nullptr,{},&out);
            check(state.body.joints[LAnkle].source==5 && state.learnedPosition[1],"Wrong confident SDK ankle replaced a bridged SAM pose");
            check(state.contacts[0].state==Contact::Air,"Predicted depth planted a foot");
            ++bridges;
        }
        for(double t=out.host+1/fps;t<1.6;t+=1/fps){p=pose(t,false);out=continuity.update(p,7,t);}
        check(!out.valid,"Depth hold outlived its limit");
        double t=out.host?out.host:1.6;
        // Use monotonically spaced times after loss; reacquisition must persist.
        t=1.6;p=pose(t);out=continuity.update(p,7,t);
        check(!out.valid,"A single returning anchor frame reacquired after loss");
        for(int i=1;i<=int(fps*.2)+1;++i){p=pose(t+i/fps);out=continuity.update(p,7,p.host);}
        check(out.valid && !out.depthPredicted,"Stable depth did not reacquire");
        p=pose(out.host+1/fps,false);p.selectedId=8;
        check(!continuity.update(p,8,p.host).valid,"Depth history leaked to another person");
        continuity.reset();p=pose(3);continuity.update(p,7,3);
        p=pose(3+1/fps,false);p.articulationValid=false;
        check(!continuity.update(p,7,p.host).valid,"Invalid articulation bridged");
        p=pose(4,false);
        check(!continuity.update(p,7,4).valid,"Old root survived long frame gap");
        SourceTransition transition;V3 previous{};
        previous=transition.update({},previous,5,1);
        auto switched=transition.update({.3,0,0},previous,1,1+1/fps);
        check(norm(switched-previous)<1e-9,"Source handoff snapped");
        previous=switched;
        for(int i=1;i<=int(fps*.4);++i)previous=transition.update({.3,0,0},previous,1,1+(i+1)/fps);
        check(norm(previous-V3{.3,0,0})<1e-9,"Source transition did not reach new evidence");
        std::cout<<fps<<" Hz: "<<bridges<<" bridged frames; bounded depth hold, reacquisition, identity and source continuity passed\n";
    }
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
