#include "sam3d_geometry.hpp"
#include "io.hpp"
#include <onnxruntime_cxx_api.h>
#include <Windows.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace kf;
namespace {
std::string quoted(const std::string &s) {
    std::string r = "\"";
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') r += '\\';
        if (c < 32) throw std::runtime_error("Control character in tensor name");
        r += char(c);
    }
    return r + '"';
}
size_t bytesPerElement(ONNXTensorElementDataType t) {
    switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: return 2;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return 8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return 4;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return 1;
    default: throw std::runtime_error("Unsupported tensor element type");
    }
}
std::vector<uint8_t> readBytes(const std::filesystem::path &p, size_t expected) {
    if (std::filesystem::file_size(p) != expected)
        throw std::runtime_error("Tensor file size mismatch: " + p.string());
    std::vector<uint8_t> b(expected);
    std::ifstream in(p, std::ios::binary);
    if (!in.read(reinterpret_cast<char *>(b.data()), std::streamsize(b.size())))
        throw std::runtime_error("Cannot read tensor: " + p.string());
    return b;
}
std::string verifyAssets(const std::filesystem::path &model) {
    auto manifest = std::filesystem::path(model.wstring() + L".files.sha256");
    std::ifstream in(manifest);
    if (!in) throw std::runtime_error("Missing model integrity manifest; run scripts/sam3d_assets.py seal");
    std::string line;
    bool found = false;
    while (std::getline(in, line)) {
        if (line.size() < 67 || line.substr(64, 2) != "  ")
            throw std::runtime_error("Invalid integrity manifest");
        auto name = line.substr(66);
        auto relative = std::filesystem::path(std::u8string(name.begin(), name.end()));
        if (relative.is_absolute() || relative.has_root_name())
            throw std::runtime_error("Absolute path in model manifest");
        for (const auto &part : relative)
            if (part == "..") throw std::runtime_error("Parent path in model manifest");
        auto path = model.parent_path() / relative;
        if (sha256(path) != line.substr(0, 64))
            throw std::runtime_error("Model asset checksum mismatch: " + relative.string());
        found |= relative == model.filename();
    }
    if (!found) throw std::runtime_error("Model graph absent from integrity manifest");
    return sha256(manifest);
}
void finiteOutput(const Ort::Value &v) {
    auto info = v.GetTensorTypeAndShapeInfo();
    auto n = info.GetElementCount();
    switch (info.GetElementType()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        for (size_t i = 0; i < n; ++i)
            if (!std::isfinite(v.GetTensorData<float>()[i])) throw std::runtime_error("Nonfinite FP32 output");
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        for (size_t i = 0; i < n; ++i)
            if (!std::isfinite(v.GetTensorData<Ort::Float16_t>()[i].ToFloat())) throw std::runtime_error("Nonfinite FP16 output");
        break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
        for (size_t i = 0; i < n; ++i)
            if (!std::isfinite(v.GetTensorData<Ort::BFloat16_t>()[i].ToFloat())) throw std::runtime_error("Nonfinite BF16 output");
        break;
    default: break;
    }
}
}
int main(int argc, char **argv) {
    try {
        if (argc >= 4 && std::string(argv[1]) == "samples") {
            const std::filesystem::path output = argv[3];
            if (std::filesystem::exists(output)) throw std::runtime_error("Sample output already exists");
            int stride = argc > 4 ? std::stoi(argv[4]) : 30;
            if (stride < 1) throw std::runtime_error("Invalid sample stride");
            RecordingReader reader;
            reader.open(argv[2]);
            std::filesystem::create_directories(output);
            unsigned index = 0, saved = 0;
            while (auto f = reader.next()) {
                unsigned frameIndex = index++;
                if (frameIndex % stride || !f->runConfig) continue;
                auto id = f->runConfig->selectedId;
                auto crop = playerCrop(*f, id);
                if (!crop) continue;
                auto projection = fitColorProjection(*f);
                if (!projection.valid) continue;
                auto base = output / std::to_string(frameIndex);
                saveBgraBmp(base.string() + ".bmp", *f);
                std::ofstream meta(base.string() + ".json");
                // Undo only the known 1.25 padding. The retained .75 aspect is
                // idempotent under SAM's aspect expansion, so its square crop agrees.
                auto c = *crop;
                meta << std::setprecision(15) << "{\"frame\":" << frameIndex << ",\"host\":" << f->host
                     << ",\"selected_id\":" << id << ",\"bbox\":[" << c.cx-c.w/2/1.25 << ',' << c.cy-c.h/2/1.25
                     << ',' << c.cx+c.w/2/1.25 << ',' << c.cy+c.h/2/1.25 << "],\"projection_rms_px\":" << projection.rms
                     << ",\"projection\":[";
                const auto &p = projection.p;
                const double matrix[]{p[0]*640,p[1]*640,p[2]*640,p[3]*640,
                                      p[4]*480,p[5]*480,p[6]*480,p[7]*480,p[8],p[9],1,p[10]};
                for (int j=0;j<12;++j) {if(j)meta<<',';meta<<matrix[j];}
                meta << "],\"sdk\":[";
                auto b = std::find_if(f->bodies.begin(),f->bodies.end(),[&](auto &body){return body.id==id;});
                if(b != f->bodies.end()) for(int j=0;j<J;++j) {
                    if(j)meta<<',';
                    auto v=b->joints[j]; meta<<'['<<v.p.x<<','<<v.p.y<<','<<v.p.z<<','<<v.confidence<<']';
                }
                meta << "]}\n";
                if (!meta) throw std::runtime_error("Cannot write sample metadata");
                ++saved;
            }
            std::cout << "Saved " << saved << " local recorded RGB samples with camera projection and SDK observations.\n";
            return saved ? 0 : 1;
        }
        if (argc == 8 && std::string(argv[1]) == "crop") {
            Frame f;
            f.bgra = readBytes(argv[2], 640 * 480 * 4);
            auto crop = sam3dCrop(std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]), std::stod(argv[7]));
            auto data = sam3dImage(f, crop);
            if (std::filesystem::exists(argv[3])) throw std::runtime_error("Crop output already exists");
            std::ofstream out(argv[3], std::ios::binary);
            out.write(reinterpret_cast<const char *>(data.data()), std::streamsize(data.size() * sizeof(float)));
            if (!out) throw std::runtime_error("Cannot write crop output");
            std::cout << crop.cx << ' ' << crop.cy << ' ' << crop.w << ' ' << crop.h << '\n';
            return 0;
        }
        bool batchMode=argc>=2 && std::string(argv[1])=="batch";
        if (argc < 7 || (std::string(argv[1]) != "run" && !batchMode)) {
            std::cout << "kf_sam3d_bench run <model.onnx> <runtime-dir> <inputs-dir> <new-output-dir> <cpu|gpu> [iterations]\n"
                         "Inputs: input_0.bin, input_0.shape, ... in ONNX input order. Native offline benchmark; no capture or OSC.\n"
                         "kf_sam3d_bench crop <640x480.bgra> <new-output.bin> <x1> <y1> <x2> <y2>\n";
            return argc == 1 ? 0 : 1;
        }
        const auto model = std::filesystem::absolute(argv[2]);
        const auto libs = std::filesystem::absolute(argv[3]);
        const auto inputsDir = std::filesystem::absolute(argv[4]);
        const auto output = std::filesystem::absolute(argv[5]);
        const std::string mode = argv[6];
        if (mode != "cpu" && mode != "gpu") throw std::runtime_error("Backend must be cpu or gpu");
        int iterations = argc > 7 ? std::stoi(argv[7]) : 30;
        if (iterations < 1 || iterations > 1000) throw std::runtime_error("Iterations must be 1..1000");
        if (std::filesystem::exists(output)) throw std::runtime_error("Output must be a new directory");
        auto fingerprint = verifyAssets(model);
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SAM3D-native-validation");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(4);
        options.SetInterOpNumThreads(1);
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        options.AddConfigEntry("session.inter_op.allow_spinning", "0");
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (mode == "gpu") {
            env.RegisterExecutionProviderLibrary("NvTensorRTRTXExecutionProvider", (libs / "onnxruntime_providers_nv_tensorrt_rtx.dll").c_str());
            std::vector<Ort::ConstEpDevice> devices;
            for (auto &d : env.GetEpDevices())
                if (std::string(d.EpName()) == "NvTensorRTRTXExecutionProvider") { devices.push_back(d); break; }
            if (devices.empty()) throw std::runtime_error("NVIDIA TensorRT RTX device unavailable");
            // Fail rather than reporting a CPU-fallback run as GPU acceleration.
            options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
            Ort::KeyValuePairs ep;
            ep.Add("nv_max_workspace_size", "536870912");
            if(std::getenv("KF_SAM_VERBOSE")) {
                options.SetLogSeverityLevel(0);
                ep.Add("nv_detailed_build_log","1");
            }
            options.AppendExecutionProvider_V2(env, devices, ep);
        }
        std::filesystem::create_directories(output);
        options.EnableProfiling((output / "ort-profile").c_str());
        double start = now();
        Ort::Session session(env, model.c_str(), options);
        double loadMs = (now() - start) * 1000;
        bool profiling=true;
        auto evaluate=[&](const std::filesystem::path &inputsDir,const std::filesystem::path &output) {
        std::filesystem::create_directories(output);
        Ort::AllocatorWithDefaultOptions allocator;
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<std::string> names, outNames;
        std::vector<std::vector<uint8_t>> data;
        std::vector<Ort::Value> tensors;
        std::ostringstream inputReport;
        for (size_t i = 0; i < session.GetInputCount(); ++i) {
            auto infoOwner = session.GetInputTypeInfo(i);
            auto info = infoOwner.GetTensorTypeAndShapeInfo();
            auto declared = info.GetShape();
            std::vector<int64_t> shape;
            std::ifstream shapeFile(inputsDir / ("input_" + std::to_string(i) + ".shape"));
            if (!shapeFile) throw std::runtime_error("Missing input shape file");
            int64_t dim;
            while (shapeFile >> dim) shape.push_back(dim);
            if (!shapeFile.eof() || shape.size() != declared.size()) throw std::runtime_error("Invalid input rank/shape");
            size_t elements = 1;
            for (size_t j = 0; j < shape.size(); ++j) {
                if (shape[j] <= 0 || (declared[j] > 0 && declared[j] != shape[j]) || size_t(shape[j]) > (1ull << 30) / elements)
                    throw std::runtime_error("Input dimension mismatch/overflow");
                elements *= size_t(shape[j]);
            }
            size_t bytes = elements * bytesPerElement(info.GetElementType());
            if (bytes > (1ull << 30)) throw std::runtime_error("Input tensor exceeds 1 GiB limit");
            auto path = inputsDir / ("input_" + std::to_string(i) + ".bin");
            data.push_back(readBytes(path, bytes));
            tensors.push_back(Ort::Value::CreateTensor(memory, data.back().data(), bytes, shape.data(), shape.size(), info.GetElementType()));
            finiteOutput(tensors.back());
            names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
            if (i) inputReport << ',';
            inputReport << "{\"name\":" << quoted(names.back()) << ",\"sha256\":" << quoted(sha256(path)) << '}';
        }
        for (size_t i = 0; i < session.GetOutputCount(); ++i) outNames.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        std::vector<const char *> ins, outs;
        for (auto &n : names) ins.push_back(n.c_str());
        for (auto &n : outNames) outs.push_back(n.c_str());
        auto run = [&] { return session.Run(Ort::RunOptions{nullptr}, ins.data(), tensors.data(), tensors.size(), outs.data(), outs.size()); };
        for (int i = 0; i < 3; ++i) { auto warm = run(); for (auto &v : warm) finiteOutput(v); }
        // Profiling is useful for placement diagnosis, but would inflate timed runs.
        if(profiling) {auto profile=session.EndProfilingAllocated(allocator);profiling=false;}
        std::vector<double> times;
        std::vector<Ort::Value> result;
        std::ofstream timing(output / "timings.csv");
        timing << "iteration,host_input_to_host_output_ms\n";
        for (int i = 0; i < iterations; ++i) {
            start = now(); result = run();
            times.push_back((now() - start) * 1000);
            for (auto &v : result) finiteOutput(v);
            timing << i << ',' << std::setprecision(10) << times.back() << '\n';
        }
        std::ofstream report(output / "result.json");
        report << "{\"model_sha256\":" << quoted(sha256(model)) << ",\"asset_manifest_sha256\":" << quoted(fingerprint)
               << ",\"backend\":" << quoted(mode == "gpu" ? "TensorRT RTX, CPU fallback disabled" : "ORT CPU reference")
               << ",\"tensor_core_execution_verified\":false,\"load_ms\":" << loadMs << ",\"iterations\":" << iterations
               << ",\"inputs\":[" << inputReport.str() << "],\"outputs\":[";
        for (size_t i = 0; i < result.size(); ++i) {
            auto info = result[i].GetTensorTypeAndShapeInfo();
            auto path = output / ("output_" + std::to_string(i) + ".bin");
            std::ofstream file(path, std::ios::binary);
            file.write(static_cast<const char *>(result[i].GetTensorRawData()), std::streamsize(info.GetElementCount() * bytesPerElement(info.GetElementType())));
            if (!file) throw std::runtime_error("Cannot write model output");
            if (i) report << ',';
            report << "{\"name\":" << quoted(outNames[i]) << ",\"onnx_type\":" << int(info.GetElementType()) << ",\"shape\":[";
            auto shape = info.GetShape();
            for (size_t j = 0; j < shape.size(); ++j) { if (j) report << ','; report << shape[j]; }
            report << "]}";
        }
        std::sort(times.begin(), times.end());
        auto quantile = [&](double q) { return times[size_t(std::ceil(q * times.size())) - 1]; };
        report << "],\"median_ms\":" << quantile(.5) << ",\"p95_ms\":" << quantile(.95) << ",\"p99_ms\":" << quantile(.99) << "}\n";
        if (!report || !timing) throw std::runtime_error("Cannot write benchmark report");
        std::cout << "Native graph benchmark: median=" << quantile(.5) << " ms, p95=" << quantile(.95)
                  << " ms. Includes host tensor transfers; excludes capture, crop and fusion.\n";
        };
        if(batchMode) {
            unsigned count=0;
            for(const auto &entry:std::filesystem::directory_iterator(inputsDir))
                if(entry.is_directory() && std::filesystem::exists(entry.path()/"input_0.bin")) {
                    evaluate(entry.path(),output/entry.path().filename());++count;
                }
            if(!count) throw std::runtime_error("No input folders found for batch evaluation");
        } else evaluate(inputsDir,output);
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
    return 0;
}
