#pragma once
#include "core.hpp"
namespace kf {
struct TrackingStats {
    uint64_t inferences{}, reusedFrames{}, inferenceErrors{}, missingPriorFrames{};
    std::array<uint64_t,trackerCount> validityLosses{}, sourceChanges{};
    double neuralHz{}, neuralAgeMs{}, workerMs{};
    double lastInferenceCompletion{};
    void age(double time) {
        neuralAgeMs=lastInferenceCompletion>0?std::max(0.,time-lastInferenceCompletion)*1000:0;
        if(!lastInferenceCompletion || neuralAgeMs>2000)neuralHz=0;
    }
};
class TrackingHealth {
    TrackingStats stats_;
    std::deque<double> inferenceTimes_;
    double lastInference_{};
    std::array<bool,trackerCount> valid_{}, learned_{};
public:
    void resetSource() {valid_={};learned_={};}
    void inferred(double time) {
        ++stats_.inferences;
        lastInference_=time;inferenceTimes_.push_back(time);
    }
    void frame(bool reused,bool failed,bool missing,const State& state) {
        stats_.reusedFrames+=reused;stats_.inferenceErrors+=failed;stats_.missingPriorFrames+=missing;
        for(int i=0;i<trackerCount;++i) {
            bool valid=state.trackers[i].valid;
            if(valid_[i] && !valid)++stats_.validityLosses[i];
            if(valid_[i] && valid && learned_[i]!=state.learnedPosition[i])++stats_.sourceChanges[i];
            valid_[i]=valid;learned_[i]=state.learnedPosition[i];
        }
    }
    TrackingStats snapshot(double time,double workerMs) {
        // Event interval rate, including idle time; expires rather than showing
        // the last successful inference rate forever after the player disappears.
        while(inferenceTimes_.size()>1 && inferenceTimes_[1]<time-2)inferenceTimes_.pop_front();
        stats_.neuralHz=inferenceTimes_.size()>1?
            double(inferenceTimes_.size()-1)/std::max(.001,time-inferenceTimes_.front()):0;
        stats_.lastInferenceCompletion=lastInference_;stats_.age(time);
        stats_.workerMs=workerMs;
        return stats_;
    }
};
}
