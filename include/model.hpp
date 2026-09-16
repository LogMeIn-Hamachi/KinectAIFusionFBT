#pragma once
#include "core.hpp"
#include <onnxruntime_cxx_api.h>
namespace kf {
std::vector<float> preprocess(const Frame &, Crop);
Keypoints decodeSimCC(std::span<const float> x, std::span<const float> y, Crop crop);
class PoseModel {
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "KinectRGBD"};
    std::unique_ptr<Ort::Session> session_;
    std::string inputName_, xName_, yName_, backend_, hash_;

  public:
    void load(const std::filesystem::path &model, bool gpu, const std::filesystem::path &libraries);
    Keypoints infer(const Frame &, Crop);
    std::pair<std::vector<float>, std::vector<float>> run(std::vector<float> &tensor);
    bool ready() const { return bool(session_); }
    std::string backend() const { return backend_; }
    std::string hash() const { return hash_; }
};
} // namespace kf
