#pragma once
#include "nlf_geometry.hpp"
#include <onnxruntime_cxx_api.h>
namespace kf {
class NlfModel {
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,"Kinect-NLF"};
    std::unique_ptr<Ort::Session> session_;
    std::string hash_;
    std::array<double,6> timings_{};
  public:
    void load(const std::filesystem::path &,const std::filesystem::path &);
    BodyPrediction infer(const Frame &,Crop,const Sam3dCamera &);
    bool ready() const {return bool(session_);}
    const std::string &hash() const {return hash_;}
    const auto &timings() const {return timings_;}
};
}
