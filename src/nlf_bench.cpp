#include "nlf_model.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace kf;
int main(int argc,char **argv) {
    try {
        if(argc<3)throw std::runtime_error("Usage: kf_nlf_bench geometry <prefix> OR model <prefix> <runtime-dir> <onnx> [repeats]");
        std::string mode=argv[1],prefix=argv[2];
        Frame frame;frame.host=1;frame.bgra.resize(640*480*4);
        std::ifstream image(prefix+".bgra",std::ios::binary);
        image.read(reinterpret_cast<char*>(frame.bgra.data()),frame.bgra.size());
        std::ifstream meta(prefix+".txt");Sam3dCamera camera;camera.valid=true;Crop crop;
        for(auto &v:camera.intrinsics)meta>>v;
        meta>>crop.cx>>crop.cy>>crop.w>>crop.h;
        if(!meta || !image)throw std::runtime_error("Missing image/camera/crop inputs");
        BodyPrediction prediction;
        if(mode=="geometry") {
            auto warp=nlfWarp(crop,camera);auto tensor=nlfImage(frame,warp);
            std::ofstream output(prefix+".crop.bin",std::ios::binary);
            output.write(reinterpret_cast<char*>(tensor.data()),tensor.size()*sizeof(float));
            std::ofstream metadata(prefix+".warp.txt");metadata<<std::setprecision(15);
            for(auto v:warp.rotation)metadata<<v.x<<' '<<v.y<<' '<<v.z<<' ';
            for(auto v:warp.intrinsics)metadata<<v<<' ';
            for(auto v:warp.sourceProjection)metadata<<v.x<<' '<<v.y<<' '<<v.z<<' ';
            std::array<float,J*6> points{};std::ifstream data(prefix+".points.bin",std::ios::binary);
            if(!data)return 0;
            data.read(reinterpret_cast<char*>(points.data()),sizeof(points));
            if(!data)throw std::runtime_error("Short NLF point data");
            prediction=nlfDecode(points,warp,camera,1);
        } else if(mode=="model" && argc>=5) {
            NlfModel model;double start=now();model.load(argv[4],argv[3]);
            std::cout<<"load_seconds="<<now()-start<<'\n';
            int repeats=argc>5?std::stoi(argv[5]):20;
            if(repeats<1 || repeats>10000)throw std::runtime_error("Invalid repeat count");
            std::vector<double> times;std::array<double,6> components{};
            for(int i=0;i<repeats;++i) {
                start=now();prediction=model.infer(frame,crop,camera);times.push_back((now()-start)*1000);
                for(int j=0;j<6;++j)components[j]+=model.timings()[j]/repeats;
            }
            std::sort(times.begin(),times.end());
            std::cout<<"median_ms="<<times[times.size()/2]<<" p95_ms="<<times[std::min(times.size()-1,size_t(times.size()*.95))]<<'\n';
            std::cout<<"crop_ms="<<components[0]<<" network_ms="<<components[1]<<" decode_ms="<<components[2]<<'\n';
        } else throw std::runtime_error("Invalid NLF benchmark command");
        std::ofstream csv(prefix+"."+mode+".csv");csv<<"x,y,z,u,v,available\n"<<std::setprecision(12);
        for(int j=0;j<J;++j){auto p=prediction.cameraPoints[j];auto uv=prediction.landmarks[j].uv;csv<<p.x<<','<<p.y<<','<<p.z<<','<<uv.x<<','<<uv.y<<','<<prediction.available[j]<<'\n';}
        return 0;
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
