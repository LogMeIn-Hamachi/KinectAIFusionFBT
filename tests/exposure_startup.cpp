#include "io.hpp"
#include <iostream>
#include <thread>
int main(int argc,char** argv) {
    try {
        unsigned failures{};
        const int cycles=argc>1?std::stoi(argv[1]):5;
        for(int i=0;i<cycles;++i) {
            kf::KinectCapture camera;
            const double start=kf::now();camera.open();
            std::cout<<"cycle="<<i<<" open_s="<<kf::now()-start<<" status="<<camera.exposureStatus<<std::endl;
            double first{},last{},exposure{},interval{};unsigned frames{};
            while(kf::now()-start<9) {
                if(auto f=camera.poll()) {
                    if(f->sensorVersion!=2)throw std::runtime_error("Expected Kinect v2");
                    if(!frames){first=kf::now();std::cout<<"first_frame_s="<<first-start<<std::endl;}
                    last=kf::now();++frames;exposure=f->exposureMs;interval=f->colorIntervalMs;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            const bool pass=frames>30 && exposure<=33.34 && interval<=34 && camera.exposureStatus.find("30 fps priority")==0;
            failures+=!pass;
            std::cout<<"frames="<<frames<<" fps="<<(last>first?(frames-1)/(last-first):0)
                <<" exposure_ms="<<exposure<<" interval_ms="<<interval<<" status="<<camera.exposureStatus<<" pass="<<pass<<std::endl;
            camera.close();
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }
        std::cout<<"failures="<<failures<<std::endl;return failures?1:0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 2;}
}
