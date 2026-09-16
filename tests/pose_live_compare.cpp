#include "io.hpp"
#include "sam3d_model.hpp"
#include "pose_continuity.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
using namespace kf;
int main(int argc,char** argv){try{
    if(argc!=3)return 2;
    const auto root=std::filesystem::absolute(argv[1]);
    if(std::filesystem::exists(argv[2]))throw std::runtime_error("Output already exists");
    Sam3dModel model;model.load(root/"assets/sam3d-optimized/backbone.onnx",root);
    KinectCapture camera;camera.open(false);
    Estimator before(false),after;PoseContinuity continuity;
    unsigned id{},count{};uint32_t epoch=~0u;
    std::ofstream csv(argv[2]);csv<<std::setprecision(12);
    csv<<"time,raw_pose,stable_pose,bridged,depth_age,anchors,inference_ms,continuity_ms";
    for(auto prefix:{"before","after"})for(int j=0;j<3;++j)for(auto field:{"valid","source","x","y","z"})csv<<','<<prefix<<j<<'_'<<field;
    csv<<'\n';
    std::cout<<"Model ready. Begin 30-second movement check. No images or OSC.\n"<<std::flush;
    double start=now();
    while(now()-start<30) {
        auto f=camera.poll();if(!f){std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
        if(f->sensorVersion!=2)throw std::runtime_error("Expected Kinect v2");
        if(!id && f->bodies.size()==1){id=f->bodies[0].id;before.select(id);after.select(id);}
        if(f->epoch!=epoch){before.select(id);after.select(id);continuity.reset();epoch=f->epoch;}
        indexRegistration(*f);auto projection=fitColorProjection(*f);f->colorProjection=std::make_shared<ColorProjection>(projection);
        auto intrinsics=sam3dCamera(projection);auto crop=playerCrop(*f,id);
        Sam3dEvidence evidence;double ms{};
        if(crop && intrinsics.valid){crop->w=crop->h=std::max(crop->w,crop->h);double t=now();
            auto prediction=model.infer(*f,*crop,intrinsics);ms=(now()-t)*1000;
            evidence=sam3dEvidence(*f,id,prediction,intrinsics);}
        const double t=now();auto stable=continuity.update(evidence.prior,id,f->host);const double cost=(now()-t)*1000;
        auto a=before.process(*f,&evidence.keypoints,{},&evidence.prior);
        auto b=after.process(*f,&evidence.keypoints,{},stable.valid?&stable:nullptr);
        csv<<now()-start<<','<<evidence.prior.valid<<','<<stable.valid<<','<<stable.depthPredicted<<','<<stable.depthAge
            <<','<<evidence.prior.rootAnchors<<','<<ms<<','<<cost;
        for(auto s:{a,b})for(int j=0;j<3;++j){auto tracker=s.trackers[j];int joint=j==0?Hip:j==1?LAnkle:RAnkle;
            csv<<','<<tracker.valid<<','<<int(s.body.joints[joint].source)<<','<<tracker.p.x<<','<<tracker.p.y<<','<<tracker.p.z;}
        csv<<'\n';if(!csv)throw std::runtime_error("Cannot write comparison");++count;
    }
    camera.close();std::cout<<"Compared "<<count<<" frames. No camera images saved; no OSC.\n";
    return count>100 && id?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
