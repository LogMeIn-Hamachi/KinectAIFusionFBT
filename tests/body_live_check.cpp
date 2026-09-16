#include "engine.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace kf;
int main(int argc,char** argv){try {
    if(argc!=3)return 2;
    if(std::filesystem::exists(argv[2]))throw std::runtime_error("Output already exists");
    Engine engine(std::filesystem::absolute(argv[1]));engine.start();
    std::ofstream csv(argv[2]);csv<<std::setprecision(12);
    csv<<"time,host,inference_ms,fit_ms,queue_ms,arrival_ms,dropped,calibration,vr_devices,pose_source";
    for(auto name:{"hip","left","right"})for(auto field:{"valid","source","x","y","z","qw","qx","qy","qz","sigma"})csv<<','<<name<<'_'<<field;
    csv<<",left_native,right_native\n";
    double wait=now(),start=0;unsigned id=0,count=0;uint64_t last=0;
    while((!start && now()-wait<60) || (start && now()-start<30)) {
        auto s=engine.view();
        if(s.frames!=last && s.frame) {
            last=s.frames;
            if(!id && s.frame->bodies.size()==1){id=s.frame->bodies[0].id;engine.select(id);engine.useSavedCalibration();}
            if(id && s.inferenceMs>0) {
                if(!start){start=now();std::cout<<"BEGIN: 30-second live check. Stand, turn, sit/recline. No OSC or images.\n"<<std::flush;}
                unsigned devices=0;for(auto& d:s.frame->vr.devices)devices+=d.valid;
                csv<<now()-start<<','<<s.state.host<<','<<s.inferenceMs<<','<<s.state.fitMs<<','<<s.queueMs<<','<<s.arrivalToEstimateMs<<','<<s.dropped
                   <<','<<s.calibration.valid<<','<<devices<<','<<s.poseSource;
                for(int i=0;i<3;++i){auto t=s.state.trackers[i];int j=i==0?Hip:i==1?LAnkle:RAnkle;
                    csv<<','<<t.valid<<','<<int(s.state.body.joints[j].source)<<','<<t.p.x<<','<<t.p.y<<','<<t.p.z<<','<<t.q.w<<','<<t.q.x<<','<<t.q.y<<','<<t.q.z<<','<<t.positionSigma;}
                csv<<','<<s.state.nativeFootDirection[0]<<','<<s.state.nativeFootDirection[1]<<'\n';++count;
            }
            if(s.output || s.sent || s.recording)throw std::runtime_error("Diagnostic must not output OSC or record images");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    engine.stop();std::cout<<"Completed "<<count<<" frames; saved numbers only; OSC disabled.\n";
    return count>100?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
