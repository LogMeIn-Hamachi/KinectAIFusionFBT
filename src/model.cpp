#include "model.hpp"
#include <Windows.h>
#include <fstream>
namespace kf {
std::vector<float> preprocess(const Frame &f, Crop c) {
    if (f.width <= 0 || f.height <= 0 || f.bgra.size() != size_t(f.width) * f.height * 4 || c.w <= 0 ||
        c.h <= 0)
        throw std::runtime_error("Invalid RGB/crop");
    std::vector<float> out(3 * 192 * 256);
    const double mean[]{123.675, 116.28, 103.53}, sd[]{58.395, 57.12, 57.375};
    for (int y = 0; y < 256; ++y)
        for (int x = 0; x < 192; ++x) {
            // Match OpenCV warpAffine's 10-bit affine accumulation and 5-bit interpolation table.
            // Rounding the final floating coordinate alone differs at crop/padding boundaries.
            int fixedX =
                (int(std::lrint((c.cx - c.w / 2) * 1024)) + 16 + int(std::lrint(c.w / 192 * x * 1024))) >> 5;
            int fixedY = (int(std::lrint((c.cy - c.h / 2 + c.h / 256 * y) * 1024)) + 16) >> 5;
            int x0 = fixedX >> 5, y0 = fixedY >> 5;
            double fx = (fixedX & 31) / 32.0, fy = (fixedY & 31) / 32.0;
            for (int ch = 0; ch < 3; ++ch) {
                double value = 0;
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx) {
                        int xx = x0 + dx, yy = y0 + dy;
                        if (xx >= 0 && xx < f.width && yy >= 0 && yy < f.height)
                            value += f.bgra[(size_t(yy) * f.width + xx) * 4 + (2 - ch)] * (dx ? fx : 1 - fx) *
                                     (dy ? fy : 1 - fy);
                    }
                out[(ch * 256 + y) * 192 + x] = float((std::round(value) - mean[ch]) / sd[ch]);
            }
        }
    return out;
}
Keypoints decodeSimCC(std::span<const float> x, std::span<const float> y, Crop c) {
    if (x.size() != 26 * 384 || y.size() != 26 * 512)
        throw std::runtime_error("Expected Halpe26 SimCC [1,26,384] and [1,26,512]");
    Keypoints out;
    for (int j = 0; j < J; ++j) {
        auto xs = x.subspan(j * 384, 384), ys = y.subspan(j * 512, 512);
        auto ix = std::max_element(xs.begin(), xs.end()), iy = std::max_element(ys.begin(), ys.end());
        double score = std::min(*ix, *iy);
        if (!std::isfinite(score) || score <= 0) {
            out[j] = {{}, 0};
            continue;
        }
        out[j] = {c.toImage({double(ix - xs.begin()) / 2, double(iy - ys.begin()) / 2}), score};
    }
    return out;
}
void PoseModel::load(const std::filesystem::path &path, bool gpu, const std::filesystem::path &libs) {
    session_.reset();
    hash_ = sha256(path);
    Ort::SessionOptions options;
    options.AddFreeDimensionOverrideByName("batch", 1);
    options.SetIntraOpNumThreads(4);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.AddConfigEntry("session.inter_op.allow_spinning", "0");
    backend_ = "ORT CPU FP32";
    Ort::SessionOptions cpuOptions = options.Clone();
    std::vector<Ort::ConstEpDevice> gpuDevices;
    bool cacheEnabled = false;
    if (gpu) {
        env_.RegisterExecutionProviderLibrary("NvTensorRTRTXExecutionProvider",
                                              (libs / "onnxruntime_providers_nv_tensorrt_rtx.dll").c_str());
        for (auto &device : env_.GetEpDevices())
            if (std::string(device.EpName()) == "NvTensorRTRTXExecutionProvider") {
                gpuDevices.push_back(device);
                break;
            }
        if (gpuDevices.empty())
            throw std::runtime_error("NVIDIA TensorRT RTX device unavailable");
        Ort::KeyValuePairs ep;
        ep.Add("nv_max_workspace_size", "536870912");
        try {
            wchar_t system[32768];
            GetSystemDirectoryW(system, 32768);
            auto hardware = gpuDevices.front().Device();
            std::string key = hash_.substr(0, 16) + "_" + sha256(libs / "onnxruntime.dll").substr(0, 12) +
                              "_" + sha256(libs / "tensorrt_rtx_1_6.dll").substr(0, 12) + "_" +
                              sha256(std::filesystem::path(system) / "nvcuda.dll").substr(0, 12) + "_" +
                              std::to_string(hardware.VendorId()) + "_" +
                              std::to_string(hardware.DeviceId()) + "_batch1_fp32_ws512";
            auto cache = libs / "cache" / key;
            std::filesystem::create_directories(cache);
            ep.Add("nv_runtime_cache_path", cache.string().c_str());
            cacheEnabled = true;
        } catch (...) { /* Read-only portable folders may still run without a compiled cache. */
        }
        options.AppendExecutionProvider_V2(env_, gpuDevices, ep);
        backend_ = "NVIDIA TensorRT RTX (FP32 graph)";
    }
    try {
        session_ = std::make_unique<Ort::Session>(env_, path.c_str(), options);
    } catch (const Ort::Exception &) {
        if (!gpu || !cacheEnabled)
            throw;
        // Recover from a stale or invalid cache without deleting files or changing the model.
        options = cpuOptions.Clone();
        Ort::KeyValuePairs fresh;
        fresh.Add("nv_max_workspace_size", "536870912");
        options.AppendExecutionProvider_V2(env_, gpuDevices, fresh);
        session_ = std::make_unique<Ort::Session>(env_, path.c_str(), options);
        backend_ += "; cache bypassed";
    }
    Ort::AllocatorWithDefaultOptions a;
    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 2)
        throw std::runtime_error("Unsupported pose model interface");
    inputName_ = session_->GetInputNameAllocated(0, a).get();
    xName_ = session_->GetOutputNameAllocated(0, a).get();
    yName_ = session_->GetOutputNameAllocated(1, a).get();
    auto shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4 || (shape[0] != 1 && shape[0] != -1) || shape[1] != 3 || shape[2] != 256 ||
        shape[3] != 192) {
        std::string detail;
        for (auto n : shape)
            detail += std::to_string(n) + ",";
        throw std::runtime_error("Model shape does not match pinned preprocessing: " + detail);
    }
    std::vector<float> warm(3 * 256 * 192);
    for (int i = 0; i < 3; ++i)
        run(warm);
}
std::pair<std::vector<float>, std::vector<float>> PoseModel::run(std::vector<float> &data) {
    if (!session_ || data.size() != 3 * 256 * 192)
        throw std::runtime_error("Model not ready / wrong tensor size");
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> shape{1, 3, 256, 192};
    auto input =
        Ort::Value::CreateTensor<float>(memory, data.data(), data.size(), shape.data(), shape.size());
    const char *in[]{inputName_.c_str()};
    const char *out[]{xName_.c_str(), yName_.c_str()};
    auto values = session_->Run(Ort::RunOptions{nullptr}, in, &input, 1, out, 2);
    std::pair<std::vector<float>, std::vector<float>> result;
    for (int i = 0; i < 2; ++i) {
        auto info = values[i].GetTensorTypeAndShapeInfo();
        auto expected = std::vector<int64_t>{1, 26, i == 0 ? 384 : 512};
        if (info.GetShape() != expected || info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Unexpected model outputs");
        auto p = values[i].GetTensorData<float>();
        auto &dest = i == 0 ? result.first : result.second;
        dest.assign(p, p + info.GetElementCount());
        for (float v : dest)
            if (!std::isfinite(v))
                throw std::runtime_error("Nonfinite inference output");
    }
    return result;
}
Keypoints PoseModel::infer(const Frame &f, Crop c) {
    auto input = preprocess(f, c);
    auto [x, y] = run(input);
    return decodeSimCC(x, y, c);
}
} // namespace kf
