#include "io.hpp"
#include "exposure.hpp"
#include <Windows.h>
#include <Kinect.h>
#include <wrl/client.h>
#include <cstring>
#include <sstream>
#include <map>
#include <thread>

namespace kf {
using Microsoft::WRL::ComPtr;
namespace {
void checkV2(HRESULT hr, const char *operation) {
    if (FAILED(hr)) {
        std::ostringstream s;
        s << operation << " failed (0x" << std::hex << static_cast<unsigned long>(hr)
          << "). Check Kinect v2 power, direct USB 3.0 connection and official runtime.";
        throw std::runtime_error(s.str());
    }
}
template<class T> void append(std::vector<uint8_t>& out, const T& v) {
    auto p=reinterpret_cast<const uint8_t*>(&v);out.insert(out.end(),p,p+sizeof(v));
}
}
struct KinectV2Capture::Impl {
    HMODULE runtime{};
    ComPtr<IKinectSensor> sensor;
    ComPtr<ICoordinateMapper> mapper;
    ComPtr<IMultiSourceFrameReader> reader;
    std::vector<CameraSpacePoint> geometry{512*424};
    std::vector<ColorSpacePoint> colors{512*424};
    std::vector<uint8_t> calibration;
    std::map<UINT64,uint32_t> ids;
    uint32_t nextId{}, frameId{};
    INT64 lastStamp{};
    double lastFrame{};
    std::string key;
    std::wstring sensorId;
    std::unique_ptr<KinectExposure> exposure;
    ClockMap clock;
    ~Impl() {
        exposure.reset();
        reader.Reset();mapper.Reset();
        if(sensor) sensor->Close();
        sensor.Reset();
        if(runtime)FreeLibrary(runtime);
    }
};
KinectV2Capture::KinectV2Capture()=default;
KinectV2Capture::~KinectV2Capture()=default;
void KinectV2Capture::close(){p_.reset();}
bool KinectV2Capture::open() {
    close();dropped=0;
    auto p=std::make_unique<Impl>();
    p->runtime=LoadLibraryExW(L"Kinect20.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    if(!p->runtime)return false;
    using GetSensor=HRESULT(WINAPI*)(IKinectSensor**);
    auto get=reinterpret_cast<GetSensor>(GetProcAddress(p->runtime,"GetDefaultKinectSensor"));
    if(!get)return false;
    checkV2(get(p->sensor.GetAddressOf()),"Find Kinect v2");
    if(!p->sensor)return false;
    checkV2(p->sensor->Open(),"Open Kinect v2");
    BOOLEAN available=FALSE;
    for(int retry=0;retry<100;++retry) {
        p->sensor->get_IsAvailable(&available);
        if(available)break;
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    if(!available)return false;
    checkV2(p->sensor->get_CoordinateMapper(p->mapper.GetAddressOf()),"Kinect v2 coordinate mapper");
    checkV2(p->sensor->OpenMultiSourceFrameReader(FrameSourceTypes_Color|FrameSourceTypes_Depth|
        FrameSourceTypes_BodyIndex|FrameSourceTypes_Body,p->reader.GetAddressOf()),"Kinect v2 synchronized streams");
    WCHAR id[256]{};
    checkV2(p->sensor->get_UniqueKinectId(256,id),"Kinect v2 identity");
    p->sensorId=id;
    p->key="kinect-v2-";
    for(auto ch:id) {
        if(!ch)break;
        // Hex encoding gives a stable path-safe key without truncating identifiers.
        constexpr char hex[]="0123456789abcdef";
        for(int shift=12;shift>=0;shift-=4)p->key+=hex[(ch>>shift)&15];
    }
    if(p->key.size()>180)throw std::runtime_error("Unexpected Kinect v2 identity length");
    CameraIntrinsics intrinsics{};
    checkV2(p->mapper->GetDepthCameraIntrinsics(&intrinsics),"Kinect v2 depth intrinsics");
    const uint32_t tag=0x3243464b;append(p->calibration,tag);append(p->calibration,intrinsics);
    for(auto ch:id){append(p->calibration,ch);if(!ch)break;}
    p->lastFrame=now();p_=std::move(p);return true;
}
bool KinectV2Capture::healthy() const {
    BOOLEAN available=FALSE;
    return p_ && SUCCEEDED(p_->sensor->get_IsAvailable(&available)) && available && now()-p_->lastFrame<3;
}
void KinectV2Capture::prioritize30(bool enabled) {
    if(!p_)throw std::runtime_error("Kinect v2 is not open");
    if(!p_->exposure) {
        auto control=std::make_unique<KinectExposure>();
        if(!control->open(p_->sensorId))throw std::runtime_error("Cannot access this Kinect's exposure control");
        p_->exposure=std::move(control);
    }
    try {
        if(enabled)p_->exposure->prioritize30();
        else p_->exposure->automatic();
    }catch(...) {
        // A failed device transaction can leave the control handle unusable.
        // Cleanup restores automatic mode where possible; retry opens a fresh handle.
        p_->exposure.reset();
        throw;
    }
}
std::string KinectV2Capture::calibrationKey() const {return p_?p_->key:std::string{};}
std::shared_ptr<Frame> KinectV2Capture::poll() {
    if(!p_)return {};
    auto& s=*p_;
    ComPtr<IMultiSourceFrame> multi;
    if(FAILED(s.reader->AcquireLatestFrame(multi.GetAddressOf())))return {};
    const double start=now();
    ComPtr<IColorFrameReference> cr;ComPtr<IDepthFrameReference> dr;
    ComPtr<IBodyIndexFrameReference> ir;ComPtr<IBodyFrameReference> br;
    ComPtr<IColorFrame> color;ComPtr<IDepthFrame> depth;
    ComPtr<IBodyIndexFrame> index;ComPtr<IBodyFrame> bodies;
    if(FAILED(multi->get_ColorFrameReference(cr.GetAddressOf())) ||
       FAILED(multi->get_DepthFrameReference(dr.GetAddressOf())) ||
       FAILED(multi->get_BodyIndexFrameReference(ir.GetAddressOf())) ||
       FAILED(multi->get_BodyFrameReference(br.GetAddressOf())) ||
       FAILED(cr->AcquireFrame(color.GetAddressOf())) || FAILED(dr->AcquireFrame(depth.GetAddressOf())) ||
       FAILED(ir->AcquireFrame(index.GetAddressOf())) || FAILED(br->AcquireFrame(bodies.GetAddressOf())))return {};
    INT64 ct{},dt{},it{},bt{};
    checkV2(color->get_RelativeTime(&ct),"Color timestamp");
    checkV2(depth->get_RelativeTime(&dt),"Depth timestamp");
    checkV2(index->get_RelativeTime(&it),"Body index timestamp");
    checkV2(bodies->get_RelativeTime(&bt),"Body timestamp");
    if(dt<=s.lastStamp)return {};
    ComPtr<IColorCameraSettings> settings;TIMESPAN exposure{},interval{};
    if(SUCCEEDED(color->get_ColorCameraSettings(settings.GetAddressOf()))) {
        settings->get_ExposureTime(&exposure);settings->get_FrameInterval(&interval);
    }
    // Automatic exposure can legitimately reduce the colour stream to 15 Hz.
    const INT64 expected=interval>=300000 && interval<=1000000?interval:333333;
    if(s.lastStamp && dt-s.lastStamp>expected*3/2)dropped+=uint64_t((dt-s.lastStamp+expected/2)/expected)-1;
    s.lastStamp=dt;
    if(std::abs(ct-dt)>180000 || std::abs(it-dt)>10000 || std::abs(bt-dt)>180000){++dropped;return {};}
    auto f=std::make_shared<Frame>();
    f->sensorVersion=2;f->width=1920;f->height=1080;f->depthWidth=512;f->depthHeight=424;
    f->exposureMs=double(exposure)/10000;f->colorIntervalMs=double(interval)/10000;
    f->arrival=start;f->rgbStamp=ct/10000;f->depthStamp=dt/10000;f->skeletonStamp=bt/10000;
    f->rgbId=f->depthId=f->skeletonId=++s.frameId;
    f->host=s.clock.map(f->depthStamp,f->arrival);f->epoch=s.clock.resets;
    f->bgra.resize(size_t(f->width)*f->height*4);
    checkV2(color->CopyConvertedFrameDataToArray(UINT(f->bgra.size()),f->bgra.data(),ColorImageFormat_Bgra),"Kinect v2 color conversion");
    UINT count{},indexCount{};UINT16* raw{};BYTE* players{};
    checkV2(depth->AccessUnderlyingBuffer(&count,&raw),"Kinect v2 depth");
    checkV2(index->AccessUnderlyingBuffer(&indexCount,&players),"Kinect v2 body index");
    if(count!=512*424 || indexCount!=count)throw std::runtime_error("Unexpected Kinect v2 depth dimensions");
    checkV2(s.mapper->MapDepthFrameToCameraSpace(count,raw,count,s.geometry.data()),"Kinect v2 metric geometry");
    checkV2(s.mapper->MapDepthFrameToColorSpace(count,raw,count,s.colors.data()),"Kinect v2 color registration");
    f->depth.resize(count);f->mapping.resize(count);
    for(size_t i=0;i<count;++i) {
        // Keep the native depth grid. Never inflate it into fabricated full-HD depth.
        const unsigned mm=raw[i];
        const unsigned player=players[i]<BODY_COUNT?players[i]+1:0;
        f->depth[i]=mm<=8191?uint16_t((mm<<3)|player):0;
        auto g=s.geometry[i];auto c=s.colors[i];
        int u=-1,v=-1;
        if(std::isfinite(c.X) && std::isfinite(c.Y) && c.X>=0 && c.X<1920 && c.Y>=0 && c.Y<1080) {
            u=int(std::lround(c.X));v=int(std::lround(c.Y));
        }
        if(mm<500 || mm>4500 || !finite({g.X,g.Y,g.Z}))f->mapping[i]={0,0,0,-1,-1};
        else f->mapping[i]={g.X,g.Y,g.Z,u,v};
    }
    Vector4 floor{};
    if(SUCCEEDED(bodies->get_FloorClipPlane(&floor))) {
        double n=norm({floor.x,floor.y,floor.z});
        if(n>.9 && n<1.1)f->floor={{floor.x/n,floor.y/n,floor.z/n},floor.w/n,true};
    }
    IBody* native[BODY_COUNT]{};
    HRESULT hr=bodies->GetAndRefreshBodyData(BODY_COUNT,native);
    std::array<ComPtr<IBody>,BODY_COUNT> owned;
    for(int i=0;i<BODY_COUNT;++i)owned[i].Attach(native[i]);
    checkV2(hr,"Kinect v2 bodies");
    constexpr std::array<std::pair<int,JointType>,18> map{{
        {LShoulder,JointType_ShoulderLeft},{RShoulder,JointType_ShoulderRight},
        {LElbow,JointType_ElbowLeft},{RElbow,JointType_ElbowRight},
        {LWrist,JointType_WristLeft},{RWrist,JointType_WristRight},
        {LHip,JointType_HipLeft},{RHip,JointType_HipRight},
        {LKnee,JointType_KneeLeft},{RKnee,JointType_KneeRight},
        {LAnkle,JointType_AnkleLeft},{RAnkle,JointType_AnkleRight},
        {Head,JointType_Head},{Neck,JointType_Neck},{Hip,JointType_SpineBase},{Nose,JointType_Head},
        {LToe,JointType_FootLeft},{RToe,JointType_FootRight}}};
    std::map<UINT64,uint32_t> currentIds;
    const uint32_t blobTag=0x3242464b;append(f->sdkSkeletonBlob,blobTag);append(f->sdkSkeletonBlob,bt);
    for(int i=0;i<BODY_COUNT;++i) {
        BOOLEAN tracked=FALSE;UINT64 id{};
        if(!owned[i] || FAILED(owned[i]->get_IsTracked(&tracked)) || !tracked)continue;
        checkV2(owned[i]->get_TrackingId(&id),"Kinect v2 body identity");
        ::Joint joints[JointType_Count]{};
        checkV2(owned[i]->GetJoints(JointType_Count,joints),"Kinect v2 body joints");
        auto previous=s.ids.find(id);
        Body body;body.id=previous==s.ids.end()?++s.nextId:previous->second;body.player=uint8_t(i+1);
        currentIds[id]=body.id;
        s.ids[id]=body.id;
        append(f->sdkSkeletonBlob,body.player);append(f->sdkSkeletonBlob,id);append(f->sdkSkeletonBlob,joints);
        for(auto[to,from]:map) {
            auto j=joints[from];auto p=j.Position;
            if(!finite({p.X,p.Y,p.Z}))continue;
            bool good=j.TrackingState==TrackingState_Tracked;
            body.joints[to]={{p.X,p.Y,p.Z},good?.85:j.TrackingState==TrackingState_Inferred?.2:0,good?.035:.15,1};
        }
        f->bodies.push_back(body);
    }
    // Brief occlusion must not change our ID when the SDK retains its 64-bit ID.
    // Bound history without ever truncating SDK identifiers into a 32-bit value.
    if(s.ids.size()>1024)s.ids=std::move(currentIds);
    f->calibrationBlob=s.calibration;f->captureMs=(now()-start)*1000;s.lastFrame=now();return f;
}
} // namespace kf
