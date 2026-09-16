#include "engine.hpp"
#include <iostream>
int main(int argc,char** argv){try{
    if(argc!=2)return 2;
    kf::Engine engine(std::filesystem::absolute(argv[1]));
    engine.chooseExposure(true);
    engine.start();
    const double start=kf::now();std::string status;
    while(kf::now()-start<25) {
        const auto view=engine.view();
        if(view.exposureStatus!=status){status=view.exposureStatus;std::cout<<"time="<<kf::now()-start<<" exposure="<<status<<std::endl;}
        if(view.sent || view.output || view.recording)throw std::runtime_error("Unexpected output or recording");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const auto view=engine.view();engine.stop();
    std::cout<<"frames="<<view.frames<<" inference="<<view.inference<<" zero_OSC="<<(view.sent==0)
        <<" zero_recording="<<!view.recording<<std::endl;
    if(!view.frame || view.frame->sensorVersion!=2 || view.frames<30 || view.exposureStatus.find("30 fps priority")!=0)
        throw std::runtime_error("App pipeline exposure startup failed");
    if(view.frame->exposureMs>33.34 || view.frame->colorIntervalMs>34)throw std::runtime_error("Unexpected camera timing");
    std::cout<<"App pipeline startup passed: "<<view.frame->exposureMs<<" ms exposure, "<<view.frame->colorIntervalMs<<" ms interval\n";
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
