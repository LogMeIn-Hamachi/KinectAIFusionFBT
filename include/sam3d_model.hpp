#pragma once
#include "sam3d_geometry.hpp"
#include <onnxruntime_cxx_api.h>
namespace kf {
// Native runtime for mixed FP16/FP32 or selectively FP8-quantized encoders.
class Sam3dModel {
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,"Kinect-SAM3D"};
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::IoBinding> binding_;
    Ort::Value featureTensor_{nullptr};
    float *featureBuffer_{};
    std::string hash_;
    std::string decoderPrecision_;
    void *decoder_{};
    using Decode=int(*)(void*,const float*,const float*,const float*,const float*,const float*,float*,float*,float*);
    Decode decode_{};
    void (*destroy_)(void*){};
    const char *(*error_)(){};
    void (*decoderTimings_)(void*,double*){};
    std::array<double,6> timings_{};
    std::vector<float> image_;
  public:
    ~Sam3dModel();
    void load(const std::filesystem::path &, const std::filesystem::path &libraries);
    Sam3dPrediction infer(const Frame &,Crop,const Sam3dCamera &);
    bool ready() const {return bool(session_) && bool(binding_) && decoder_;}
    const std::string &hash() const {return hash_;}
    const std::string &decoderPrecision() const {return decoderPrecision_;}
    const auto &timings() const {return timings_;}
};
}
