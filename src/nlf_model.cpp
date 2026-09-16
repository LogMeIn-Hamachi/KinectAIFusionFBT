#include "nlf_model.hpp"
#include <Windows.h>
#include <fstream>
namespace kf {
void NlfModel::load(const std::filesystem::path &path,const std::filesystem::path &libs) {
    session_.reset();std::string expected;
    std::ifstream checksum(std::filesystem::path(path.wstring()+L".sha256"));checksum>>expected;
    hash_=sha256(path);
    if(expected.size()!=64 || expected!=hash_)throw std::runtime_error("NLF model checksum missing or mismatched");
    Ort::SessionOptions options;options.SetIntraOpNumThreads(2);options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning","0");
    options.AddConfigEntry("session.inter_op.allow_spinning","0");
    options.AddConfigEntry("session.disable_cpu_ep_fallback","1");
    env_.RegisterExecutionProviderLibrary("NvTensorRTRTXExecutionProvider",(libs/"onnxruntime_providers_nv_tensorrt_rtx.dll").c_str());
    std::vector<Ort::ConstEpDevice> devices;
    for(auto &d:env_.GetEpDevices())if(std::string(d.EpName())=="NvTensorRTRTXExecutionProvider"){devices.push_back(d);break;}
    if(devices.empty())throw std::runtime_error("NLF requires an NVIDIA RTX GPU");
    wchar_t system[32768];GetSystemDirectoryW(system,32768);
    auto hardware=devices.front().Device();
    std::string key=hash_.substr(0,16)+"_"+sha256(libs/"onnxruntime.dll").substr(0,12)+"_"+
        sha256(libs/"tensorrt_rtx_1_6.dll").substr(0,12)+"_"+sha256(std::filesystem::path(system)/"nvcuda.dll").substr(0,12)+"_"+
        std::to_string(hardware.VendorId())+"_"+std::to_string(hardware.DeviceId())+"_nlf_b1_tf32_ws128";
    auto cache=libs/"cache"/key;std::filesystem::create_directories(cache);
    Ort::KeyValuePairs ep;ep.Add("nv_max_workspace_size","134217728");ep.Add("enable_cuda_graph","1");ep.Add("nv_runtime_cache_path",cache.string().c_str());
    options.AppendExecutionProvider_V2(env_,devices,ep);
    auto session=std::make_unique<Ort::Session>(env_,path.c_str(),options);
    if(session->GetInputCount()!=1 || session->GetOutputCount()!=1)throw std::runtime_error("Wrong NLF model interface");
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputType=session->GetInputTypeInfo(0),outputType=session->GetOutputTypeInfo(0);
    auto input=inputType.GetTensorTypeAndShapeInfo(),output=outputType.GetTensorTypeAndShapeInfo();
    if(std::string(session->GetInputNameAllocated(0,allocator).get())!="image" ||
       std::string(session->GetOutputNameAllocated(0,allocator).get())!="points" ||
       input.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || input.GetShape()!=std::vector<int64_t>{1,3,256,256} ||
       output.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || output.GetShape()!=std::vector<int64_t>{1,J,6})
        throw std::runtime_error("Unsupported NLF graph input/output");
    // Compile/warm the fixed batch-one graph before publishing readiness.
    std::vector<float> image(3*256*256,.5f);std::array<int64_t,4> shape{1,3,256,256};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto tensor=Ort::Value::CreateTensor<float>(memory,image.data(),image.size(),shape.data(),4);
    const char *ins[]{"image"},*outs[]{"points"};
    for(int i=0;i<3;++i)session->Run(Ort::RunOptions{nullptr},ins,&tensor,1,outs,1);
    session_=std::move(session);
}
BodyPrediction NlfModel::infer(const Frame &frame,Crop crop,const Sam3dCamera &camera) {
    if(!session_)throw std::runtime_error("NLF model is not ready");
    double start=now();auto warp=nlfWarp(crop,camera);auto image=nlfImage(frame,warp);
    timings_[0]=(now()-start)*1000;start=now();
    std::array<int64_t,4> shape{1,3,256,256};
    auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
    auto input=Ort::Value::CreateTensor<float>(memory,image.data(),image.size(),shape.data(),4);
    const char *ins[]{"image"},*outs[]{"points"};
    auto result=session_->Run(Ort::RunOptions{nullptr},ins,&input,1,outs,1);
    timings_[1]=(now()-start)*1000;start=now();
    auto out=nlfDecode({result[0].GetTensorData<float>(),J*6},warp,camera,frame.host);
    timings_[2]=(now()-start)*1000;
    return out;
}
}
