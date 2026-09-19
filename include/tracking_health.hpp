#pragma once
#include "core.hpp"
#include <ostream>
namespace kf {
struct ProcessingTiming {
    double host{},sourceAgeMs{},captureMs{},queueMs{},workerMs{},inferenceMs{};
    double cropMs{},encoderHostMs{},decoderHostMs{};
    int cadenceHz{},rootSupports{};
    bool fresh{},reused{},depthPredicted{},leftPlanted{},rightPlanted{},live{};
};
class ProcessingHistory {
    std::deque<ProcessingTiming> samples_;
public:
    void add(const ProcessingTiming& sample){samples_.push_back(sample);if(samples_.size()>180)samples_.pop_front();}
    void clear(){samples_.clear();}
    size_t size()const{return samples_.size();}
    double workerPercentile(double fraction)const {
        if(samples_.empty())return 0;
        std::vector<double> values;for(const auto& s:samples_)if(std::isfinite(s.workerMs))values.push_back(s.workerMs);
        if(values.empty())return 0;
        std::sort(values.begin(),values.end());
        return values[size_t(std::ceil(std::clamp(fraction,0.,1.)*(values.size()-1)))];
    }
    void write(std::ostream& out)const {
        out<<"host,source_age_ms,capture_ms,queue_ms,worker_ms,fresh_inference_ms,sam_crop_ms,sam_encoder_host_call_ms,sam_decoder_host_call_ms,cadence_hz,root_supports,fresh,reused,depth_predicted,left_planted,right_planted,live\n";
        for(const auto& s:samples_)out<<s.host<<','<<s.sourceAgeMs<<','<<s.captureMs<<','<<s.queueMs<<','<<s.workerMs<<','<<s.inferenceMs<<','
            <<s.cropMs<<','<<s.encoderHostMs<<','<<s.decoderHostMs<<','<<s.cadenceHz<<','<<s.rootSupports<<','<<s.fresh<<','<<s.reused<<','
            <<s.depthPredicted<<','<<s.leftPlanted<<','<<s.rightPlanted<<','<<s.live<<'\n';
    }
};
struct TrackingStats {
    uint64_t inferences{}, reusedFrames{}, inferenceErrors{}, missingPriorFrames{};
    std::array<uint64_t,trackerCount> validityLosses{}, sourceChanges{};
    double neuralHz{}, neuralAgeMs{}, sourceAgeMs{}, workerMs{};
    double lastInferenceCompletion{},lastInferenceSource{};
    void age(double time) {
        neuralAgeMs=lastInferenceCompletion>0?std::max(0.,time-lastInferenceCompletion)*1000:0;
        sourceAgeMs=lastInferenceSource>0?std::max(0.,time-lastInferenceSource)*1000:0;
        if(!lastInferenceCompletion || neuralAgeMs>2000)neuralHz=0;
    }
};
class TrackingHealth {
    TrackingStats stats_;
    std::deque<double> inferenceTimes_;
    double lastInference_{},lastSource_{};
    std::array<bool,trackerCount> valid_{}, learned_{};
public:
    void resetSource() {valid_={};learned_={};}
    void inferred(double time,double source=0) {
        ++stats_.inferences;
        lastInference_=time;inferenceTimes_.push_back(time);
        lastSource_=source;
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
        stats_.lastInferenceCompletion=lastInference_;stats_.lastInferenceSource=lastSource_;stats_.age(time);
        stats_.workerMs=workerMs;
        return stats_;
    }
};
}
