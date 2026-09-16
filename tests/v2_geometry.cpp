#include "io.hpp"
#include "sam3d_geometry.hpp"
#include <iostream>
using namespace kf;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main() {
    auto path=std::filesystem::temp_directory_path()/("kf-v2-roundtrip-"+std::to_string(now())+".kfr");
    try {
        Frame f;f.sensorVersion=2;f.width=1920;f.height=1080;f.depthWidth=512;f.depthHeight=424;
        f.host=f.arrival=10;f.captureMs=7;f.exposureMs=16;f.colorIntervalMs=33.3333;
        f.bgra.resize(1920*1080*4);f.depth.resize(512*424);f.mapping.resize(512*424);
        Body body;body.id=123;body.player=1;f.bodies.push_back(body);
        for(int y=0;y<424;++y)for(int x=0;x<512;++x) {
            size_t i=size_t(y)*512+x;
            int u=x*1919/511,v=y*1079/423;
            float z=.6f+float((x*37+y*23)%3000)/1000;
            f.mapping[i]={float((960-u)*z/1000.),float((540-v)*z/1000.),z,u,v};
            bool player=u>1000 && u<1500 && v>200 && v<1000;
            f.depth[i]=uint16_t((unsigned(z*1000)<<3)|(player?1:0));
        }
        auto projection=fitColorProjection(f);auto camera=sam3dCamera(projection);
        require(projection.valid && projection.rms<.001 && camera.valid,"Full-HD projection fit");
        require(std::abs(camera.intrinsics[0]-1000)<.01 && std::abs(camera.intrinsics[2]-960)<.01,
            "Full-HD camera intrinsics must not retain v1 dimensions");
        auto uv=projection.project({-.6,0,2});require(std::abs(uv.x-1260)<.01,"Projection beyond v1 image boundary");
        indexRegistration(f);require(f.colorIndex.size()==1920*1080,"Full-HD registration index");
        unsigned indexed=0;for(auto i:f.colorIndex)indexed+=i>=0;
        require(indexed<=512*424 && indexed>200000,"No invented depth pixels or v1 field-of-view clipping");
        auto crop=playerCrop(f,123);require(crop && crop->cx>1000 && crop->cy>500,"Player crop beyond v1 bounds");
        auto input=sam3dImage(f,*crop);require(input.size()==3*512*512,"SAM workload stays fixed at 512 square");
        auto original=f.mapping[12345];
        {RecordingWriter writer;writer.open(path,"v2 synthetic geometry test");writer.write(f);Frame old;writer.write(old);writer.close();}
        {RecordingReader reader;reader.open(path);auto roundtrip=reader.next();
            require(roundtrip && roundtrip->sensorVersion==2 && roundtrip->depthWidth==512 &&
                roundtrip->depthHeight==424 && roundtrip->width==1920 && roundtrip->height==1080,"V2 recording dimensions");
            require(roundtrip->depth==f.depth && roundtrip->bgra==f.bgra && roundtrip->mapping[12345].u==original.u &&
                roundtrip->mapping[12345].z==original.z && roundtrip->exposureMs==16,"V2 recording preserves native samples");
            require(roundtrip->bodies.front().id==123,"Recording body identity");
            auto old=reader.next();require(old && old->sensorVersion==1 && old->width==640 && old->depthWidth==640,"V1 frame roundtrip");
            require(!reader.next(),"Recording end");}
        {RecordingWriter writer;writer.open(path,"invalid dimensions");f.depthWidth=640;writer.write(f);writer.close();}
        bool rejected=false;try {RecordingReader reader;reader.open(path);reader.next();}catch(...){rejected=true;}
        require(rejected,"Reject mismatched native depth dimensions");
        std::filesystem::remove(path);
        std::cout<<"V2 native depth, full-HD projection, crop, fixed inference size and recording checks passed.\n";
        return 0;
    }catch(const std::exception& e){std::filesystem::remove(path);std::cerr<<e.what()<<'\n';return 1;}
}
