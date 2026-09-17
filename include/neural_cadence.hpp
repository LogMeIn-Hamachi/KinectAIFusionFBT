#pragma once
#include <algorithm>
#include <cmath>

namespace kf {
// A ceiling on new image inference, not a fraction of arriving camera frames.
// Auto measures sustained worker cost/backlog; it does not measure GPU utilization.
class NeuralCadence {
    int choice_{-1}, level_{}, samples_{};
    double due_{}, lastFrame_{}, lastSample_{}, cost_{}, queue_{};
    double pressureSince_{}, recoverySince_{};
    void change(int level) {
        level_=level; due_=0; pressureSince_=recoverySince_=0;
    }
public:
    void reset() { *this=NeuralCadence{}; }
    int hz() const { return choice_==2?20:choice_==3?15:choice_==1?30:level_==2?15:level_==1?20:30; }
    double workMs() const { return cost_; }
    bool due(double time,int choice,bool replay=false) {
        choice=std::clamp(choice,0,3);
        if(choice!=choice_ || time<lastFrame_ || time-lastFrame_>1) {
            reset(); choice_=choice;
        }
        lastFrame_=time;
        if(replay)return true;
        if(due_>0 && time+.001<due_)return false;
        const double period=1.0/hz();
        // Keep the phase for 20 Hz on a 30 fps camera, without catch-up bursts
        // after a stall or further dividing an already slow camera stream.
        if(due_==0 || time-due_>=period)due_=time+period;
        else due_+=period;
        return true;
    }
    void observe(double time,double workMs,double queueMs) {
        if(!std::isfinite(workMs) || !std::isfinite(queueMs) || workMs<0 || queueMs<0)return;
        // Model startup/first-use compilation must not select a lasting low rate.
        if(++samples_<=2) {lastSample_=time;return;}
        double dt=std::clamp(time-lastSample_,.001,.2);lastSample_=time;
        double alpha=1-std::exp(-dt/.35);
        if(samples_==3){cost_=workMs;queue_=queueMs;}
        else {cost_+=(workMs-cost_)*alpha;queue_+=(queueMs-queue_)*alpha;}
        if(choice_!=0)return;
        bool pressure=level_==0?(cost_>34.5 || queue_>18):level_==1?(cost_>50.5 || queue_>30):false;
        bool recovery=level_==2?(cost_<46 && queue_<14):level_==1?(cost_<31.5 && queue_<10):false;
        if(pressure) {
            recoverySince_=0;
            if(!pressureSince_)pressureSince_=time;
            if(time-pressureSince_>=.6)change(level_+1);
        } else if(recovery) {
            pressureSince_=0;
            if(!recoverySince_)recoverySince_=time;
            if(time-recoverySince_>=2)change(level_-1);
        } else pressureSince_=recoverySince_=0;
    }
};
}
