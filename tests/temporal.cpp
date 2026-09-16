#include "core.hpp"
#include <iostream>
using namespace kf;
static void check(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
int main() {
    try {
        for(double fps:{15.,30.,45.}) {
            SoleCorrection sole;
            double previous=0,maximumStep=0;
            for(int i=0;i<int(fps*2);++i) {
                double value=sole.update(i%2?-.015:0,1/fps);
                maximumStep=std::max(maximumStep,std::abs(value-previous));previous=value;
                check(std::abs(value)<=.015,"Contact correction exceeded its geometric bound");
            }
            check(maximumStep<=.06/fps+1e-9,"Contact correction snapped at a state boundary");
            for(int i=0;i<int(fps);++i)sole.update(0,1/fps);
            check(std::abs(sole.value)<1e-5,"Floor correction retained a released foot");
            LearnedPositionFilter position;
            RotationEvidence rotation;
            double lo=1,hi=-1,qlo=1,qhi=-1,time=1;
            for(int i=0;i<int(fps*5);++i,time+=1/fps) {
                double sign=i%2?1.:-1.;
                auto value=position.update({sign*.02,0,2},time);
                rotation.update(axisAngle({0,1,0},sign*8*pi/180),true,time,0,.65,8,.045,true);
                if(i>fps*2) {
                    lo=std::min(lo,value.x);hi=std::max(hi,value.x);
                    auto forward=rotation.value.rotate({0,0,1});double yaw=std::atan2(forward.x,forward.z);
                    qlo=std::min(qlo,yaw);qhi=std::max(qhi,yaw);
                }
            }
            check(hi-lo<.010,"Alternating 4 cm position noise did not settle");
            check((qhi-qlo)*180/pi<4,"Alternating 16 degree rotation noise did not settle");
            check(position.resting,"Alternating noise incorrectly identified as coherent motion");
            double stepStart=time,stepResponse=0;
            for(int i=0;i<int(fps);++i,time+=1/fps) {
                auto p=position.update({.2,0,2},time);
                if(stepResponse==0 && p.x>.18)stepResponse=time-stepStart+1/fps;
            }
            check(stepResponse>0 && stepResponse<=.15,"Rest smoothing delayed a real step by more than 150 ms");
            LearnedPositionFilter slow;
            double maximumLag=0;
            for(int i=0;i<int(fps*6);++i) {
                double t=i/fps,target=.03*t;
                auto p=slow.update({target,0,2},1+t);
                if(t>1)maximumLag=std::max(maximumLag,target-p.x);
            }
            check(maximumLag<.015,"Slow deliberate movement stuck in rest deadband");
            auto reset=position.update({1,0,2},time+.3);
            check(norm(reset-V3{1,0,2})<1e-9,"Long tracking gap retained stale filter history");
            std::cout<<fps<<" Hz: position peak-to-peak "<<(hi-lo)*1000<<" mm (input 40); yaw "
                     <<(qhi-qlo)*180/pi<<" deg (input 16); step90 "<<stepResponse*1000
                     <<" ms; slow-motion lag "<<maximumLag*1000<<" mm\n";
            std::cout<<"Contact toggle max step "<<maximumStep*1000<<" mm (previously up to 15).\n";
        }
        std::cout<<"Temporal settling, movement response, slow-motion and gap checks passed.\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
