// Native LibTorch runtime boundary. No Python library or interpreter is linked.
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#include <array>
#include <string>
#include <chrono>

namespace {
thread_local std::string lastError;
struct Decoder {
    torch::jit::Module module;
    std::vector<torch::Tensor> inputs;
    std::vector<torch::jit::IValue> arguments;
    std::vector<torch::Tensor> outputs;
    c10::cuda::CUDAStream stream=c10::cuda::getStreamFromPool(true);
    at::cuda::CUDAGraph graph;
    bool captured=false;
    std::array<double,3> timings{};
    bool tf32=false;
    explicit Decoder(const char *path,bool tensorFloat=false):module(torch::jit::load(path)),tf32(tensorFloat) {
        module.eval();
        // Select kernels before warmup and graph capture. Geometry, reductions,
        // inputs and outputs retain FP32 storage in both modes.
        at::globalContext().setAllowTF32CuBLAS(tf32);
        at::globalContext().setAllowTF32CuDNN(tf32);
        at::set_num_threads(4);
        torch::jit::setGraphExecutorOptimize(false);
        const std::vector<int64_t> shapes[]{{1,1280,32,32},{1,2},{1,2},{1,3,3},{1,2}};
        for(const auto &shape:shapes){inputs.push_back(torch::zeros(shape,torch::TensorOptions().device(torch::kCUDA)));arguments.push_back(inputs.back());}
    }
    void infer(const float *features,const float *center,const float *scale,const float *camera,float *xyz,float *uv,float *rotations,bool resident=false,const float *imageSize=nullptr) {
        torch::InferenceMode inferenceGuard;
        c10::cuda::CUDAStreamGuard streamGuard(stream);
        auto clock=[] {return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();};
        double start=clock();
        float wh[]{640,480};
        if(imageSize) {
            if(!std::isfinite(imageSize[0]) || !std::isfinite(imageSize[1]) || imageSize[0]<1 || imageSize[1]<1 || imageSize[0]>4096 || imageSize[1]>4096)
                throw std::runtime_error("Invalid SAM image dimensions");
            wh[0]=imageSize[0];wh[1]=imageSize[1];
        }
        const float *data[]{features,center,scale,camera,wh};
        if(resident) {
            if(features!=inputs[0].data_ptr<float>())throw std::runtime_error("Wrong SAM GPU feature buffer");
            // Reduce on the GPU; only one boolean crosses to the host. Never
            // dereference a device pointer in the application's CPU validation.
            if(!torch::isfinite(inputs[0]).all().item<bool>())throw std::runtime_error("Nonfinite SAM image features");
        }
        for(int i=resident?1:0;i<5;++i) inputs[i].copy_(torch::from_blob(const_cast<float*>(data[i]),inputs[i].sizes(),torch::kFloat32));
        timings[0]=(clock()-start)*1000;start=clock();
        if(!captured) {
            for(int i=0;i<5;++i) module.forward(arguments);
            stream.synchronize();
            graph.capture_begin();
            auto tuple=module.forward(arguments).toTuple();
            for(const auto &v:tuple->elements())outputs.push_back(v.toTensor());
            graph.capture_end();captured=true;
        }
        graph.replay();
        timings[1]=(clock()-start)*1000;start=clock();
        float *dest[]{xyz,uv,rotations};size_t sizes[]{210,140,1143};
        for(int i=0;i<3;++i) {
            if(outputs[i].numel()!=sizes[i] || outputs[i].scalar_type()!=torch::kFloat32) throw std::runtime_error("Wrong native decoder output");
            cudaMemcpyAsync(dest[i],outputs[i].data_ptr<float>(),sizes[i]*sizeof(float),cudaMemcpyDeviceToHost,stream);
        }
        stream.synchronize();
        timings[2]=(clock()-start)*1000;
    }
};
}
extern "C" {
__declspec(dllexport) void *kf_decoder_create(const char *path) {
    try {torch::InferenceMode guard;return new Decoder(path);}
    catch(const std::exception &e){lastError=e.what();return nullptr;}
}
__declspec(dllexport) void *kf_decoder_create_precision(const char *path,int precision) {
    try {
        if(precision!=0 && precision!=1)throw std::runtime_error("Unsupported decoder precision");
        torch::InferenceMode guard;return new Decoder(path,precision==1);
    } catch(const std::exception &e){lastError=e.what();return nullptr;}
}
__declspec(dllexport) const char *kf_decoder_precision(void *handle) {
    return static_cast<Decoder*>(handle)->tf32?"TF32":"FP32";
}
__declspec(dllexport) int kf_decoder_run(void *handle,const float *features,const float *center,const float *scale,const float *camera,float *xyz,float *uv,float *rotations) {
    try {if(!handle)throw std::runtime_error("Decoder is not loaded");static_cast<Decoder*>(handle)->infer(features,center,scale,camera,xyz,uv,rotations);return 1;}
    catch(const std::exception &e){lastError=e.what();return 0;}
}
__declspec(dllexport) int kf_decoder_feature_buffer(void *handle,float **pointer,size_t *count,int *device) {
    try {
        if(!handle || !pointer || !count || !device)throw std::runtime_error("Invalid GPU buffer request");
        auto &tensor=static_cast<Decoder*>(handle)->inputs[0];
        *pointer=tensor.data_ptr<float>();*count=tensor.numel();*device=tensor.get_device();return 1;
    }catch(const std::exception &e){lastError=e.what();return 0;}
}
__declspec(dllexport) int kf_decoder_run_gpu(void *handle,const float *features,const float *center,const float *scale,const float *camera,float *xyz,float *uv,float *rotations) {
    try {if(!handle)throw std::runtime_error("Decoder is not loaded");static_cast<Decoder*>(handle)->infer(features,center,scale,camera,xyz,uv,rotations,true);return 1;}
    catch(const std::exception &e){lastError=e.what();return 0;}
}
__declspec(dllexport) int kf_decoder_run_gpu_size(void *handle,const float *features,const float *center,const float *scale,const float *camera,const float *imageSize,float *xyz,float *uv,float *rotations) {
    try {
        if(!handle || !imageSize)throw std::runtime_error("Missing decoder or image dimensions");
        static_cast<Decoder*>(handle)->infer(features,center,scale,camera,xyz,uv,rotations,true,imageSize);return 1;
    }catch(const std::exception &e){lastError=e.what();return 0;}
}
__declspec(dllexport) void kf_decoder_destroy(void *handle){
    delete static_cast<Decoder*>(handle);
    // Switching to NLF must release SAM's inactive tensor/graph memory to VRChat.
    try { c10::cuda::CUDACachingAllocator::emptyCache(); } catch(...) {}
}
__declspec(dllexport) const char *kf_decoder_error(){return lastError.c_str();}
__declspec(dllexport) uint64_t kf_decoder_stream(void *handle){return reinterpret_cast<uint64_t>(static_cast<Decoder*>(handle)->stream.stream());}
__declspec(dllexport) void kf_decoder_timings(void *handle,double *output){auto &t=static_cast<Decoder*>(handle)->timings;std::copy(t.begin(),t.end(),output);}
}
