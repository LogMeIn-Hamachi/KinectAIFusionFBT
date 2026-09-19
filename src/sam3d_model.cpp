#include "sam3d_model.hpp"
#include <Windows.h>
#include <fstream>
namespace kf {
Sam3dModel::~Sam3dModel(){binding_.reset();featureTensor_=Ort::Value{nullptr};session_.reset();if(decoder_ && destroy_)destroy_(decoder_);}
void Sam3dModel::load(const std::filesystem::path &path,const std::filesystem::path &libs) {
    binding_.reset();featureTensor_=Ort::Value{nullptr};featureBuffer_=nullptr;session_.reset();
    if(decoder_ && destroy_){destroy_(decoder_);decoder_=nullptr;}
    auto manifest=std::filesystem::path(path.wstring()+L".files.sha256");
    std::ifstream in(manifest);
    if(!in) throw std::runtime_error("SAM model integrity manifest is missing");
    std::string line;bool graph=false,decoderAsset=false;unsigned files=0;
    while(std::getline(in,line)) {
        if(line.size()<67 || line.substr(64,2)!="  ") throw std::runtime_error("Invalid SAM manifest");
        auto relative=std::filesystem::path(line.substr(66));
        if(relative.is_absolute() || relative.has_root_name()) throw std::runtime_error("Invalid SAM asset path");
        for(const auto &part:relative) if(part=="..") throw std::runtime_error("Invalid SAM asset path");
        if(sha256(path.parent_path()/relative)!=line.substr(0,64)) throw std::runtime_error("SAM asset checksum mismatch");
        graph|=relative==path.filename();decoderAsset|=relative=="decoder.pt";++files;
    }
    if(!graph || !decoderAsset || files<3) throw std::runtime_error("Incomplete SAM asset manifest");
    hash_=sha256(manifest);
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(4);options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning","0");
    options.AddConfigEntry("session.inter_op.allow_spinning","0");
    options.AddConfigEntry("session.disable_cpu_ep_fallback","1");
    auto runtime=libs/"sam3d-runtime";
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    AddDllDirectory(runtime.c_str());
    // Keep LibTorch loaded for process lifetime: FreeLibrary can deadlock its
    // worker shutdown under Windows' loader lock. Decoder instances are released.
    auto dll=GetModuleHandleW(L"kf_sam3d_decoder.dll");
    if(!dll)dll=LoadLibraryExW((runtime/L"kf_sam3d_decoder.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!dll)throw std::runtime_error("Native SAM decoder runtime missing: "+std::to_string(GetLastError()));
    auto create=reinterpret_cast<void*(*)(const char*,int)>(GetProcAddress(dll,"kf_decoder_create_precision"));
    auto precisionName=reinterpret_cast<const char*(*)(void*)>(GetProcAddress(dll,"kf_decoder_precision"));
    decode_=reinterpret_cast<Decode>(GetProcAddress(dll,"kf_decoder_run_gpu_size"));
    auto buffer=reinterpret_cast<int(*)(void*,float**,size_t*,int*)>(GetProcAddress(dll,"kf_decoder_feature_buffer"));
    destroy_=reinterpret_cast<void(*)(void*)>(GetProcAddress(dll,"kf_decoder_destroy"));
    error_=reinterpret_cast<const char*(*)()>(GetProcAddress(dll,"kf_decoder_error"));
    decoderTimings_=reinterpret_cast<void(*)(void*,double*)>(GetProcAddress(dll,"kf_decoder_timings"));
    if(!create||!precisionName||!decode_||!buffer||!destroy_||!error_)throw std::runtime_error("SAM Kinect v2 decoder runtime update is missing");
    auto bytes=(path.parent_path()/"decoder.pt").u8string();std::string decoderPath(bytes.begin(),bytes.end());
    // Explicit, reversible precision choice. Old executables use the retained
    // legacy create API, which continues to select strict FP32.
    std::string precision="tf32",extra;
    auto precisionFile=libs/"decoder-precision.txt";
    if(std::filesystem::exists(precisionFile)) {
        std::ifstream config(precisionFile);
        if(!(config>>precision) || (config>>extra) || (precision!="tf32" && precision!="fp32"))
            throw std::runtime_error("decoder-precision.txt must contain tf32 or fp32");
    }
    decoder_=create(decoderPath.c_str(),precision=="tf32"?1:0);
    if(!decoder_)throw std::runtime_error(error_());
    decoderPrecision_=precisionName(decoder_);
    env_.RegisterExecutionProviderLibrary("NvTensorRTRTXExecutionProvider",(libs/"onnxruntime_providers_nv_tensorrt_rtx.dll").c_str());
    std::vector<Ort::ConstEpDevice> devices;
    for(auto &d:env_.GetEpDevices()) if(std::string(d.EpName())=="NvTensorRTRTXExecutionProvider") {devices.push_back(d);break;}
    if(devices.empty()) throw std::runtime_error("SAM requires the NVIDIA GPU runtime");
    wchar_t system[32768];GetSystemDirectoryW(system,32768);
    auto device=devices.front().Device();
    std::string key=hash_.substr(0,16)+"_"+sha256(libs/"onnxruntime.dll").substr(0,12)+"_"+
        sha256(libs/"tensorrt_rtx_1_6.dll").substr(0,12)+"_"+
        sha256(std::filesystem::path(system)/"nvcuda.dll").substr(0,12)+"_"+
        std::to_string(device.DeviceId())+"_sam_mixed_ws512";
    auto cache=libs/"cache"/key;std::filesystem::create_directories(cache);
    Ort::KeyValuePairs ep;ep.Add("nv_max_workspace_size","536870912");
    ep.Add("nv_runtime_cache_path",cache.string().c_str());
    auto stream=reinterpret_cast<uint64_t(*)(void*)>(GetProcAddress(dll,"kf_decoder_stream"));
    if(!stream)throw std::runtime_error("Native decoder stream interface missing");
    ep.Add("user_compute_stream",std::to_string(stream(decoder_)).c_str());
    options.AppendExecutionProvider_V2(env_,devices,ep);
    session_=std::make_unique<Ort::Session>(env_,path.c_str(),options);
    if(session_->GetInputCount()!=1 || session_->GetOutputCount()!=1)
        throw std::runtime_error("Wrong SAM model interface");
    Ort::AllocatorWithDefaultOptions allocator;
    auto type=session_->GetInputTypeInfo(0);auto tensor=type.GetTensorTypeAndShapeInfo();
    if(std::string(session_->GetInputNameAllocated(0,allocator).get())!="image" ||
       tensor.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || tensor.GetShape()!=std::vector<int64_t>{1,3,512,512})
        throw std::runtime_error("Unsupported SAM model input");
    auto outputType=session_->GetOutputTypeInfo(0);auto output=outputType.GetTensorTypeAndShapeInfo();
    const std::array<int64_t,4> featureShape{1,1280,32,32};
    if(std::string(session_->GetOutputNameAllocated(0,allocator).get())!="features" ||
       output.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
       output.GetShape()!=std::vector<int64_t>(featureShape.begin(),featureShape.end()))
        throw std::runtime_error("Wrong SAM encoder output shape");
    size_t count=0;int deviceId=-1;
    if(!buffer(decoder_,&featureBuffer_,&count,&deviceId))throw std::runtime_error(error_());
    if(!featureBuffer_ || count!=1280*32*32 || deviceId<0)throw std::runtime_error("Invalid SAM GPU feature buffer");
    // Decoder owns this stable allocation for its entire CUDA graph lifetime.
    // Both stages use the decoder's CUDA stream, so no host feature copy is needed.
    Ort::MemoryInfo gpuMemory("Cuda",OrtDeviceAllocator,deviceId,OrtMemTypeDefault);
    featureTensor_=Ort::Value::CreateTensor<float>(gpuMemory,featureBuffer_,count,featureShape.data(),featureShape.size());
    binding_=std::make_unique<Ort::IoBinding>(*session_);
    binding_->BindOutput("features",featureTensor_);
}
Sam3dPrediction Sam3dModel::infer(const Frame &frame,Crop crop,const Sam3dCamera &camera) {
    if(!ready() || !camera.valid) throw std::runtime_error("SAM model/camera is not ready");
    double start=now();
    sam3dImage(frame,crop,image_);
    timings_[0]=(now()-start)*1000;start=now();
    float center[]{float(crop.cx),float(crop.cy)},scale[]{float(crop.w),float(crop.h)};
    auto intrinsics=camera.intrinsics;
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    std::array<int64_t,4> shape{1,3,512,512};
    auto input=Ort::Value::CreateTensor<float>(memory,image_.data(),image_.size(),shape.data(),shape.size());
    binding_->BindInput("image",input);
    session_->Run(Ort::RunOptions{nullptr},*binding_);
    timings_[1]=(now()-start)*1000;start=now();
    std::array<float,210> xyz{};std::array<float,140> uv{};std::array<float,1143> rotations{};
    const float imageSize[]{float(frame.width),float(frame.height)};
    if(!decode_(decoder_,featureBuffer_,center,scale,intrinsics.data(),imageSize,xyz.data(),uv.data(),rotations.data()))throw std::runtime_error(error_());
    timings_[2]=(now()-start)*1000;
    if(decoderTimings_)decoderTimings_(decoder_,timings_.data()+3);
    for(auto span:{std::span<const float>(xyz),std::span<const float>(uv),std::span<const float>(rotations)})
        for(float v:span)if(!std::isfinite(v))throw std::runtime_error("Nonfinite SAM output");
    Sam3dPrediction out;out.host=frame.host;
    for(int i=0;i<70;++i) {out.cameraPoints[i]={xyz[i*3],xyz[i*3+1],xyz[i*3+2]};out.imagePoints[i]={uv[i*2],uv[i*2+1]};}
    for(int side=0;side<2;++side)for(int r=0;r<3;++r)for(int c=0;c<3;++c)
        out.footRotations[side].a[r][c]=rotations[(side==0?23:7)*9+r*3+c];
    out.hasFootRotations=true;
    return out;
}
}
