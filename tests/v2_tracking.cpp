#include "engine.hpp"
#include <iostream>
#include <iomanip>
using namespace kf;
double percentile(std::vector<double> v,double p) {
    if(v.empty())return 0;std::sort(v.begin(),v.end());return v[std::min(v.size()-1,size_t(v.size()*p))];
}
int main(int argc,char** argv) {
    try {
        if(argc<2)return 2;
        Engine engine(std::filesystem::absolute(argv[1]));engine.start();
        std::vector<double> inference,capture,arrival,queue;
        double start=now(),duration=argc>2?std::stod(argv[2]):25;uint64_t last=0;
        unsigned selected=0,bodyFrames=0,learned=0,valid=0;View s;
        while(now()-start<duration) {
            s=engine.view();
            if(s.frames!=last && s.frame) {
                last=s.frames;
                if(s.frame->sensorVersion!=2)throw std::runtime_error("Expected live Kinect v2");
                if(!selected && s.frame->bodies.size()==1){selected=s.frame->bodies.front().id;engine.select(selected);}
                bodyFrames+=!s.frame->bodies.empty();
                if(s.inferenceMs>0 && s.frames>10) {
                    inference.push_back(s.inferenceMs);capture.push_back(s.frame->captureMs);
                    arrival.push_back(s.arrivalToEstimateMs);queue.push_back(s.queueMs);
                    learned+=s.state.learnedPosition[0] && s.state.learnedPosition[1] && s.state.learnedPosition[2];
                    valid+=s.state.trackers[0].valid && s.state.trackers[1].valid && s.state.trackers[2].valid;
                }
                if(s.sent || s.output || s.recording)throw std::runtime_error("Diagnostic must not send OSC or record images");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        engine.stop();
        std::cout<<s.sensor<<'\n'<<s.inference<<'\n'<<s.notice<<'\n'<<std::setprecision(6)
            <<"frames="<<s.frames<<" dropped="<<s.dropped<<" body_frames="<<bodyFrames<<" inferred_frames="<<inference.size()
            <<" learned_all_trackers="<<learned<<" valid_all_trackers="<<valid
            <<" inference_median_ms="<<percentile(inference,.5)<<" inference_p95_ms="<<percentile(inference,.95)
            <<" capture_median_ms="<<percentile(capture,.5)<<" arrival_median_ms="<<percentile(arrival,.5)
            <<" arrival_p95_ms="<<percentile(arrival,.95)<<" queue_median_ms="<<percentile(queue,.5)
            <<" zero_OSC="<<(s.sent==0)<<" zero_recordings="<<!s.recording<<'\n';
        return inference.empty()?2:0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
