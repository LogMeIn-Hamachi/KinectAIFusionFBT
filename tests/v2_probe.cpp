#include "io.hpp"
#include "sam3d_geometry.hpp"
#include "sam3d_model.hpp"
#include <iostream>
#include <thread>
#include <iomanip>
using namespace kf;
int main(int argc,char** argv) {
    try {
        Sam3dModel model;
        if(argc>2) {
            auto root=std::filesystem::absolute(argv[2]);
            model.load(root/"assets/sam3d-optimized/backbone.onnx",root);
        }
        KinectV2Capture capture;
        if(!capture.open())throw std::runtime_error("Kinect v2 is not available");
        std::cout<<"Kinect v2 available. No images recorded; no OSC.\n"<<std::flush;
        double start=now(),first=0,last=0;unsigned count=0,bodyFrames=0,cameras=0,crops=0;
        std::vector<double> costs,rms,inference;double seconds=argc>1?std::stod(argv[1]):10;
        while(now()-start<seconds) {
            if(auto f=capture.poll()) {
                if(!count)first=f->host;last=f->host;++count;costs.push_back(f->captureMs);
                auto projection=fitColorProjection(*f);auto camera=sam3dCamera(projection);
                cameras+=camera.valid;rms.push_back(projection.rms);
                if(model.ready() && camera.valid) {
                    double t=now();
                    // Fixed central crop measures processing cost even with nobody
                    // present; this does not measure tracking quality or accuracy.
                    model.infer(*f,sam3dCrop(480,0,1440,1080),camera);
                    if(count>5)inference.push_back((now()-t)*1000);
                }
                bodyFrames+=!f->bodies.empty();
                if(!f->bodies.empty())crops+=bool(playerCrop(*f,f->bodies.front().id));
                if(count==1)std::cout<<"RGB "<<f->width<<'x'<<f->height<<" depth "<<f->depthWidth<<'x'<<f->depthHeight
                    <<" projection rms "<<projection.rms<<" camera valid "<<camera.valid<<" exposure_ms "<<f->exposureMs<<" color_interval_ms "<<f->colorIntervalMs<<'\n';
            } else std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if(!count)throw std::runtime_error("No synchronized Kinect v2 frames received");
        std::sort(costs.begin(),costs.end());std::sort(rms.begin(),rms.end());
        std::sort(inference.begin(),inference.end());
        std::cout<<std::setprecision(6)<<"frames="<<count<<" fps="<<(count-1)/(last-first)<<" dropped="<<capture.dropped
            <<" capture_median_ms="<<costs[costs.size()/2]<<" capture_p95_ms="<<costs[costs.size()*95/100]
            <<" camera_valid="<<cameras<<" projection_rms_median="<<rms[rms.size()/2]
            <<" body_frames="<<bodyFrames<<" crop_frames="<<crops<<'\n';
        if(!inference.empty())std::cout<<"fixed_crop_inference_frames="<<inference.size()
            <<" inference_median_ms="<<inference[inference.size()/2]<<" inference_p95_ms="<<inference[inference.size()*95/100]
            <<"; throughput test only, not a tracking-quality test\n";
        return cameras==count?0:2;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
