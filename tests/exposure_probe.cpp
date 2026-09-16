#include "io.hpp"
#include <iostream>
#include <thread>
int main(){
    try {
        kf::KinectV2Capture capture;
        if(!capture.open())throw std::runtime_error("No Kinect v2 available");
        auto measure=[&](auto& source,const char* label,int seconds,bool expect30=false) {
            const double start=kf::now();double first{},last{},exposure{},interval{},minExposure=1e9,maxExposure{},maxInterval{};unsigned n{},slow{};
            while(kf::now()-start<seconds) {
                if(auto f=source.poll();f && kf::now()-start>1) {
                    last=f->rgbStamp/1000.0;if(!n)first=last;++n;
                    exposure=f->exposureMs;interval=f->colorIntervalMs;
                    minExposure=std::min(minExposure,exposure);maxExposure=std::max(maxExposure,exposure);
                    maxInterval=std::max(maxInterval,interval);slow+=interval>34;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::cout<<label<<" frames="<<n<<" fps="<<(last>first?(n-1)/(last-first):0)
                <<" exposure_ms="<<exposure<<" interval_ms="<<interval
                <<" min_exposure_ms="<<minExposure<<" max_exposure_ms="<<maxExposure
                <<" max_interval_ms="<<maxInterval<<" slow_frames="<<slow<<std::endl;
            if(n<2)throw std::runtime_error("No usable camera frames");
            // Camera-command readback accepts 33.3 ms, but SDK frame metadata on
            // this sensor reports 32.24 ms. Check a real increase over 30 ms and
            // unchanged cadence, rather than equating the two reporting APIs.
            if(expect30 && (minExposure<=30.0 || maxExposure>33.34 || slow || (n-1)/(last-first)<29))
                throw std::runtime_error("30 fps exposure/cadence check failed");
        };
        measure(capture,"automatic_before",5);
        capture.prioritize30(true);measure(capture,"priority_30",20,true);
        capture.prioritize30(false);measure(capture,"automatic_restored",5);
        capture.close();
        kf::KinectCapture facade;facade.open();
        measure(facade,"app_capture_default",7);
        std::cout<<facade.exposureStatus<<std::endl;
        if(facade.exposureStatus!="Automatic exposure")throw std::runtime_error("Default exposure control not applied");
        facade.close();
        facade.open(true);
        measure(facade,"app_capture_priority_30",12,true);
        if(facade.exposureStatus.find("30 fps priority")!=0)throw std::runtime_error("Priority 30 selection not applied");
        facade.close();
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
