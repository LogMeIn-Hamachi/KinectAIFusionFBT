#include "core.hpp"
#include "neural_cadence.hpp"
#include "tracking_health.hpp"
#include <iostream>
using namespace kf;
static void check(bool ok,const char *message) {if(!ok)throw std::runtime_error(message);}
int main() {
    try {
        // Cadence is a wall-time ceiling, including slow cameras and jitter.
        for(double cameraHz:{15.,20.,30.})for(int choice:{1,2,3}) {
            NeuralCadence cadence;int runs=0;
            double desired=choice==1?30:choice==2?20:15;
            for(int i=0;i<int(cameraHz*12);++i) {
                double t=1+i/cameraHz+(i%2?.0002:0);
                if(cadence.due(t,choice))++runs;
            }
            check(std::abs(runs-12*std::min(desired,cameraHz))<=1,"Cadence divided slow camera frames or drifted");
        }
        {
            NeuralCadence cadence;double t=1;
            auto run=[&](double seconds,double cost,double queue=0.) {
                for(int i=0;i<int(seconds*30);++i,t+=1./30)
                    if(cadence.due(t,0))cadence.observe(t,cost,queue);
            };
            // Ignore startup compilation; normal 29ms worker fits the budget.
            run(2./30,500);run(4,29);
            check(cadence.hz()==30,"Startup or healthy inference selected a low rate");
            run(5,58);check(cadence.hz()==15,"Sustained overload did not throttle");
            run(5,38);check(cadence.hz()==20,"Auto stayed stuck at 15 under moderate load");
            run(5,27);check(cadence.hz()==30,"Auto failed to recover to full rate");
            run(1./30,45);run(3,27);check(cadence.hz()==30,"One transient lowered the rate");
            run(4,27,24);check(cadence.hz()==20,"Real queue backlog did not reduce cadence");
            run(5,27);check(cadence.hz()==30,"Queue recovery left Auto throttled");
            for(int i=0;i<300;++i,t+=1./30)cadence.due(t,0);
            check(cadence.hz()==30,"Frames without new inference changed Auto");
            check(cadence.due(t+2,3),"Long stall did not allow immediate inference");
            check(!cadence.due(t+2.01,3),"Stall caused catch-up inference burst");
            check(cadence.due(t+2.02,1),"Mode change did not reset deadline");
            check(cadence.due(t+2.021,3,true),"Replay incorrectly skipped inference");
            check(cadence.due(1,3),"Clock reset retained a future deadline");
        }
        {
            TrackingHealth health;State state;
            ProcessingHistory history;
            for(int i=0;i<200;++i){ProcessingTiming t;t.workerMs=i;history.add(t);}
            check(history.size()==180 && history.workerPercentile(.5)==110 && history.workerPercentile(.95)==191,
                  "Bounded timing history or percentiles incorrect");
            history.clear();check(history.size()==0 && history.workerPercentile(.95)==0,"Old timing samples survived restart");
            for(int i=0;i<61;++i) {health.inferred(1+i/30.);health.snapshot(1+i/30.,20);}
            check(std::abs(health.snapshot(3,20).neuralHz-30)<.01,"Actual neural rate incorrect");
            auto stalled=health.snapshot(3,20);stalled.age(6);
            check(stalled.neuralHz==0 && stalled.neuralAgeMs==3000,"Stalled camera GUI retained fresh neural stats");
            check(health.snapshot(6,20).neuralHz==0,"Stopped inference still displayed a live rate");
            health.inferred(7.,6.95);auto ages=health.snapshot(7.02,20);
            check(std::abs(ages.neuralAgeMs-20)<1e-8 && std::abs(ages.sourceAgeMs-70)<1e-8,
                  "Inference completion age was confused with source-image age");
            state.trackers[0].valid=true;state.learnedPosition[0]=true;
            health.frame(false,false,false,state);
            health.frame(true,false,false,state);
            state.learnedPosition[0]=false;health.frame(false,false,true,state);
            state.trackers[0].valid=false;health.frame(false,true,true,state);
            auto stats=health.snapshot(6,20);
            check(stats.reusedFrames==1 && stats.inferenceErrors==1 && stats.missingPriorFrames==2,"Health event counts incorrect");
            check(stats.sourceChanges[0]==1 && stats.validityLosses[0]==1,"Reuse confused with tracking loss");
            health.resetSource();health.frame(false,false,false,state);
            check(health.snapshot(6,20).validityLosses[0]==1,"Identity reset counted as a tracker loss");
        }
        for(double fps:{15.,30.}) {
            PresentationVelocity motion;double maximum=0;
            for(int i=0;i<int(fps*4);++i) {
                auto velocity=motion.update({i%2?.002:-.002,0,0},1+i/fps,true);
                if(i>fps)maximum=std::max(maximum,norm(velocity));
            }
            check(maximum<.004,"Alternating noise was converted into presentation motion");
            check(norm(motion.update({1,0,0},6,true))==0,"Gap retained old presentation velocity");
            check(norm(motion.update({1.001,0,0},6.02,false))==0,"Invalid tracker retained velocity");
            check(norm(motion.update({2,0,0},6.03,true))==0,"Reacquisition carried stale velocity");
        }
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
