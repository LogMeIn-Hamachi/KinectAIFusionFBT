#include "nlf_geometry.hpp"
#include <iostream>
using namespace kf;
int main() {
    try {
        Sam3dCamera camera;camera.valid=true;camera.intrinsics={510,0,320,0,510,240,0,0,1};camera.rows={V3{1,0,0},V3{0,1,0},V3{0,0,1}};
        auto warp=nlfWarp({320,240,400,400},camera);
        std::array<float,J*6> raw{};std::array<V3,J> expected{};
        for(int j=0;j<J;++j) {
            V3 p{(j%5-2)*.12,(j/5-2)*.12,2.2+(j%3)*.08};expected[j]=p;
            raw[j*6]=float(warp.intrinsics[0]*p.x/p.z+128);raw[j*6+1]=float(warp.intrinsics[4]*p.y/p.z+128);
            auto relative=p-V3{-.9,-.8,1.2};raw[j*6+2]=float(relative.x);raw[j*6+3]=float(relative.y);raw[j*6+4]=float(relative.z);raw[j*6+5]=.05f;
        }
        auto prediction=nlfDecode(raw,warp,camera,10);
        for(int j=0;j<J;++j)if(norm(prediction.cameraPoints[j]-expected[j])>.002 || !prediction.available[j])
            throw std::runtime_error("NLF perspective units/reconstruction mismatch");
        unsigned rejected=0;
        try {nlfDecode(std::span<const float>(raw).first(12),warp,camera,10);}catch(...){++rejected;}
        auto bad=raw;bad[3]=std::numeric_limits<float>::quiet_NaN();
        try {nlfDecode(bad,warp,camera,10);}catch(...){++rejected;}
        bad=raw;for(int j=0;j<J;++j)bad[j*6+5]=1;
        try {nlfDecode(bad,warp,camera,10);}catch(...){++rejected;}
        try {nlfWarp({320,240,-1,400},camera);}catch(...){++rejected;}
        camera.valid=false;try {nlfWarp({320,240,400,400},camera);}catch(...){++rejected;}
        Frame frame;try {nlfImage(frame,warp);}catch(...){++rejected;}
        if(rejected!=6)throw std::runtime_error("NLF malformed/uncertain inputs not rejected");
        frame.bgra.assign(640*480*4,255);auto white=nlfImage(frame,warp);
        if(std::any_of(white.begin(),white.end(),[](float v){return std::abs(v-1)>1e-6;}))throw std::runtime_error("White NLF image changed");
        camera.valid=true;auto off=nlfWarp({-300,240,200,300},camera);auto padded=nlfImage(frame,off);
        Frame hd;hd.width=1920;hd.height=1080;hd.bgra.assign(1920*1080*4,255);
        auto hdWhite=nlfImage(hd,warp);
        if(hdWhite.size()!=3*256*256 || std::any_of(hdWhite.begin(),hdWhite.end(),[](float v){return std::abs(v-1)>1e-6;}))
            throw std::runtime_error("NLF full-HD input changed fixed crop size or intensity");
        if(std::any_of(padded.begin(),padded.end(),[](float v){return !std::isfinite(v) || v<0 || v>1;}))throw std::runtime_error("Invalid padded NLF image");
        std::cout<<"NLF metric reconstruction, units, malformed inputs, uncertainty, padding and image range passed.\n";
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
