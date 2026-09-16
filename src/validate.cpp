#include "io.hpp"
#include "model.hpp"
#include "engine.hpp"
#include "sam3d_model.hpp"
#include "nlf_model.hpp"
#include "body_tracker.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <openvr.h>
using namespace kf;
int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::cout << "kf_validate model <model.onnx> <library-dir> [cpu|gpu] [tensor.bin]\nkf_validate "
                         "replay <file.kfr> <model.onnx> <output.csv> [raw|filtered|fused]\n";
            return 0;
        }
        std::string command = argv[1];
        if(command=="tilt-read") {
            KinectCapture sensor;sensor.open();auto angle=sensor.elevation();
            if(!angle)throw std::runtime_error("Kinect did not report a motor angle");
            std::cout<<"Kinect motor angle: "<<*angle<<" degrees. Read only; motor not moved, no frames recorded.\n";
            return 0;
        }
        if(command=="sam-images" && (argc==5 || argc==6)) {
            // Offline RGB-only diagnosis; no capture, metric registration or OSC.
            auto root=std::filesystem::absolute(argv[2]);
            auto input=std::filesystem::path(argv[3]), output=std::filesystem::path(argv[4]);
            if(std::filesystem::exists(output)) throw std::runtime_error("Output already exists");
            Sam3dModel sam;sam.load(argc==6?std::filesystem::absolute(argv[5]):root/"assets/sam3d/backbone.onnx",root);
            std::filesystem::create_directories(output);
            unsigned count=0;
            for(const auto &entry:std::filesystem::directory_iterator(input)) {
                if(entry.path().extension()!=L".bgra")continue;
                Frame frame;frame.host=1;frame.bgra.resize(640*480*4);
                if(std::filesystem::file_size(entry.path())!=frame.bgra.size())throw std::runtime_error("Invalid RGB size");
                std::ifstream image(entry.path(),std::ios::binary);
                image.read(reinterpret_cast<char *>(frame.bgra.data()),frame.bgra.size());
                // Supplied per-image camera and crop; video crops use approximate
                // intrinsics and must not be interpreted as a metric accuracy test.
                Sam3dCamera camera;camera.valid=true;Crop crop;
                auto metaPath=entry.path();metaPath.replace_extension(L".txt");
                std::ifstream meta(metaPath);
                for(auto &v:camera.intrinsics)meta>>v;
                meta>>crop.cx>>crop.cy>>crop.w>>crop.h;
                if(!meta)throw std::runtime_error("Invalid image camera/crop metadata");
                auto prediction=sam.infer(frame,crop,camera);
                std::ofstream data(output/(entry.path().stem().string()+".csv"));
                data<<"x,y,z,u,v\n"<<std::setprecision(12);
                for(int j=0;j<70;++j) {
                    auto p=prediction.cameraPoints[j];auto uv=prediction.imagePoints[j];
                    data<<p.x<<','<<p.y<<','<<p.z<<','<<uv.x<<','<<uv.y<<'\n';
                }
                ++count;
            }
            std::cout<<"RGB-only model diagnostic images="<<count<<"; no metric accuracy claim, no OSC.\n";
            return count?0:1;
        }
        if (command == "trim" && argc >= 5) {
            if (std::filesystem::absolute(argv[2]) == std::filesystem::absolute(argv[3]) ||
                std::filesystem::exists(argv[3]))
                throw std::runtime_error("Trim output must be a new file");
            double seconds = std::stod(argv[4]);
            if (!std::isfinite(seconds) || seconds <= 0)
                throw std::runtime_error("Invalid trim duration");
            RecordingReader reader;
            RecordingWriter writer;
            reader.open(argv[2]);
            writer.open(argv[3], reader.metadata + "\nTrimmed to requested duration.");
            double first = -1, last = 0;
            unsigned count = 0;
            while (auto frame = reader.next()) {
                if (first < 0)
                    first = frame->host;
                if (frame->host - first > seconds)
                    break;
                writer.write(*frame);
                if (argc >= 6 && count % 30 == 0) {
                    std::filesystem::create_directories(argv[5]);
                    saveBgraBmp(std::filesystem::path(argv[5]) / (std::to_string(count) + ".bmp"), *frame);
                }
                last = frame->host;
                ++count;
            }
            writer.close();
            std::cout << "trimmed_frames=" << count << " span_s=" << last - first << '\n';
            return 0;
        }
        if (command == "vr-info") {
            vr::EVRInitError error;
            auto system = vr::VR_Init(&error, vr::VRApplication_Background);
            if (!system)
                throw std::runtime_error(vr::VR_GetVRInitErrorAsEnglishDescription(error));
            for (uint32_t i = 0; i < vr::k_unMaxTrackedDeviceCount; ++i) {
                if (!system->IsTrackedDeviceConnected(i))
                    continue;
                auto type = system->GetTrackedDeviceClass(i);
                if (type != vr::TrackedDeviceClass_HMD && type != vr::TrackedDeviceClass_Controller)
                    continue;
                std::cout << "device=" << i << " class=" << int(type)
                          << " role=" << int(system->GetControllerRoleForTrackedDeviceIndex(i)) << '\n';
                for (auto property : {vr::Prop_ModelNumber_String, vr::Prop_ManufacturerName_String,
                                      vr::Prop_ControllerType_String, vr::Prop_RenderModelName_String,
                                      vr::Prop_TrackingSystemName_String}) {
                    char value[1024]{};
                    vr::ETrackedPropertyError e;
                    system->GetStringTrackedDeviceProperty(i, property, value, sizeof(value), &e);
                    if (e == vr::TrackedProp_Success)
                        std::cout << int(property) << '=' << value << '\n';
                }
            }
            vr::VR_Shutdown();
            return 0;
        }
        if (command == "profile" && argc >= 3) {
            const double seconds = std::clamp(std::stod(argv[2]), 1.0, 120.0);
            Engine engine(std::filesystem::absolute(std::filesystem::path(argv[0]).parent_path()));
            engine.start();
            double start = now();
            std::uint64_t last = 0, valid = 0, inferred = 0;
            std::uint32_t selected = 0;
            std::array<std::uint64_t, 6> modes{};
            std::vector<double> inference, fit, queue, arrival;
            while (now() - start < seconds) {
                auto v = engine.view();
                if (!selected && v.frame && !v.frame->bodies.empty()) {
                    selected = v.frame->bodies.front().id;
                    engine.select(selected); // Diagnostic locks one identity; never switches to a bystander.
                }
                if (v.frames != last) {
                    last = v.frames;
                    ++modes[size_t(v.state.mode)];
                    if (std::all_of(v.state.trackers.begin(), v.state.trackers.end(),
                                    [](auto &t) { return t.valid; }))
                        ++valid;
                    if (v.inferenceMs > 0) {
                        inference.push_back(v.inferenceMs);
                        ++inferred;
                    }
                    fit.push_back(v.state.fitMs);
                    queue.push_back(v.queueMs);
                    arrival.push_back(v.arrivalToEstimateMs);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            auto result = engine.view();
            engine.stop();
            auto report = [](const char *name, std::vector<double> values) {
                if (values.empty())
                    return;
                std::sort(values.begin(), values.end());
                std::cout << name << "_median_ms=" << values[values.size() / 2] << '\n'
                          << name << "_p95_ms=" << values[size_t((values.size() - 1) * .95)] << '\n'
                          << name << "_p99_ms=" << values[size_t((values.size() - 1) * .99)] << '\n';
            };
            std::cout << "duration_s=" << seconds << "\nbackend=" << result.inference
                      << "\nsensor=" << result.sensor << "\nselected_id=" << selected
                      << "\nprocessed_frames=" << result.frames << "\nsampled_frames=" << arrival.size()
                      << "\ninferred_frames=" << inferred << "\nvalid_three_tracker_frames=" << valid
                      << "\nqueue_drops=" << result.dropped << "\nOSC_packets=" << result.sent << '\n';
            for (size_t i = 0; i < modes.size(); ++i)
                std::cout << "mode_" << modeName(Mode(i)) << '=' << modes[i] << '\n';
            report("preprocess_and_inference", inference);
            report("fit", fit);
            report("arrival_to_inference_start", queue);
            report("arrival_to_estimate", arrival);
            std::cout << "No images saved. OSC disabled. Timing excludes display and receiver.\n";
            return result.frames ? 0 : 2;
        }
        if (command == "capture" && argc >= 3) {
            double seconds = std::clamp(std::stod(argv[2]), 1.0, 120.0);
            KinectCapture capture;
            capture.open();
            RecordingWriter writer;
            if (argc > 3)
                writer.open(argv[3],
                            "Explicit local live diagnostic capture; raw acquisition; no OSC output");
            double start = now(), first = 0, last = 0;
            size_t frames = 0, bodyFrames = 0;
            std::vector<double> skews, ages, projectionErrors;
            std::vector<std::uint32_t> ids;
            while (now() - start < seconds) {
                if (auto f = capture.poll()) {
                    if (!frames)
                        first = f->host;
                    last = f->host;
                    ++frames;
                    if (!f->bodies.empty())
                        ++bodyFrames;
                    for (auto &b : f->bodies)
                        if (std::find(ids.begin(), ids.end(), b.id) == ids.end())
                            ids.push_back(b.id);
                    skews.push_back(double(std::abs(f->rgbStamp - f->depthStamp)));
                    ages.push_back((now() - f->arrival) * 1000);
                    if (frames % 10 == 0) {
                        auto projection = fitColorProjection(*f);
                        if (projection.valid)
                            projectionErrors.push_back(projection.rms);
                    }
                    if (writer.active())
                        writer.write(*f);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            writer.close();
            std::sort(skews.begin(), skews.end());
            std::sort(ages.begin(), ages.end());
            std::cout << "paired_frames=" << frames << "\nbody_frames=" << bodyFrames
                      << "\npair_drops=" << capture.dropped << "\n";
            if (frames > 1)
                std::cout << "paired_rate_hz=" << (frames - 1) / (last - first)
                          << "\nrgb_depth_skew_p95_ms=" << skews[size_t((frames - 1) * .95)]
                          << "\nSDK_mapping_arrival_to_ready_p95_ms=" << ages[size_t((frames - 1) * .95)]
                          << "\n";
            std::cout << "body_ids=";
            for (auto id : ids)
                std::cout << id << ',';
            std::cout << "\nvalid_color_projection_checks=" << projectionErrors.size() << "\n";
            if (!projectionErrors.empty())
                std::cout << "last_projection_rms_pixels=" << projectionErrors.back() << "\n";
            return frames ? 0 : 2;
        }
        if (command == "preprocess" && argc == 8) {
            Frame frame;
            frame.bgra.resize(640 * 480 * 4);
            std::ifstream input(argv[2], std::ios::binary);
            input.read(reinterpret_cast<char *>(frame.bgra.data()), frame.bgra.size());
            if (!input)
                throw std::runtime_error("Expected a 640x480 BGRA frame");
            Crop crop =
                cropBox(std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]), std::stod(argv[7]));
            auto tensor = preprocess(frame, crop);
            std::ofstream output(argv[3], std::ios::binary);
            output.write(reinterpret_cast<char *>(tensor.data()), tensor.size() * sizeof(float));
            if (!output)
                throw std::runtime_error("Cannot write preprocessing tensor");
            return 0;
        }
        if (command == "model" && argc >= 4) {
            PoseModel model;
            bool gpu = argc > 4 && std::string(argv[4]) == "gpu";
            double t = now();
            model.load(argv[2], gpu, argv[3]);
            std::cout << "backend=" << model.backend() << "\nmodel_sha256=" << model.hash()
                      << "\nload_and_warmup_ms=" << (now() - t) * 1000 << "\n";
            std::vector<float> tensor(3 * 192 * 256);
            if (argc > 5) {
                std::ifstream f(argv[5], std::ios::binary);
                f.read(reinterpret_cast<char *>(tensor.data()), tensor.size() * 4);
                if (!f)
                    throw std::runtime_error("Tensor input file is short");
            } else {
                for (size_t i = 0; i < tensor.size(); ++i)
                    tensor[i] = float(std::sin(i * 0.013) * 1.5);
            }
            std::vector<double> times;
            std::pair<std::vector<float>, std::vector<float>> outputs;
            for (int i = 0; i < 50; ++i) {
                t = now();
                outputs = model.run(tensor);
                times.push_back((now() - t) * 1000);
            }
            std::sort(times.begin(), times.end());
            std::cout << "iterations=50\nmedian_ms=" << times[25] << "\np95_ms=" << times[47]
                      << "\np99_ms=" << times[49] << "\n";
            std::string prefix = gpu ? "gpu" : "cpu";
            std::ofstream out(prefix + "-outputs.bin", std::ios::binary);
            if (!out)
                throw std::runtime_error("Cannot write validation tensors in current directory");
            out.write(reinterpret_cast<char *>(outputs.first.data()), outputs.first.size() * 4);
            out.write(reinterpret_cast<char *>(outputs.second.data()), outputs.second.size() * 4);
            return 0;
        }
        if ((command == "replay" || command=="sam-replay" || command=="nlf-replay" || command=="sam-cache") && argc >= 5) {
            RecordingReader reader;
            reader.open(argv[2]);
            std::cout << "metadata=" << reader.metadata << "\n";
            PoseModel model;
            Sam3dModel sam;
            NlfModel nlf;
            bool cached=command=="sam-cache";
            std::ifstream cache,cacheIndex,rotationCache;
            if(cached) {
                cache.open(argv[3],std::ios::binary);
                rotationCache.open(std::string(argv[3])+".feet",std::ios::binary);
                auto name=std::string(argv[3]);auto suffix=std::string(".sam3d.bin");
                if(!name.ends_with(suffix))throw std::runtime_error("Cache must have its original .csv.sam3d.bin filename");
                cacheIndex.open(name.substr(0,name.size()-suffix.size()));
                std::string header;std::getline(cacheIndex,header);
                if(!cache || !cacheIndex || !header.starts_with("depth_id,host,"))throw std::runtime_error("Missing SAM cache/index");
                std::cout<<"Offline fusion of cached model outputs; frame IDs/times checked; no GPU, capture or OSC.\n";
            }
            Estimator estimator;
            BodyTracker continuity;
            std::string mode = !cached && argc > 5 ? argv[5] : "fused";
            int stride=cached && argc>5?std::stoi(argv[5]):1;
            if(stride<1 || stride>10)throw std::runtime_error("Invalid cache stride");
            estimator.settings.baseline = mode == "raw" ? 1 : mode == "filtered" ? 2 : 0;
            if(command=="nlf-replay") {
                auto root=std::filesystem::absolute(argv[3]);nlf.load(root/"assets/nlf/pose.onnx",root);
                std::cout<<"NLF-S NVIDIA TensorRT comparison on recorded geometry; no OSC.\n";
            } else if(command=="sam-replay") {
                auto root=std::filesystem::absolute(argv[3]);
                auto encoder=argc>6?std::filesystem::absolute(argv[6]):root/"assets/sam3d/backbone.onnx";
                sam.load(encoder,root);
                std::cout<<"Explicit model comparison: Fast SAM on the recorded geometry/settings; decoder "<<sam.decoderPrecision()<<"; no OSC.\n";
            } else if (!cached && !estimator.settings.baseline)
                model.load(argv[3], false, std::filesystem::path(argv[0]).parent_path());
            Calibration cal;
            std::ofstream csv(argv[4]);
            if (!csv)
                throw std::runtime_error("Cannot write replay CSV");
            std::ofstream samData,rotationData;
            if(sam.ready()){samData.open(std::string(argv[4])+".sam3d.bin",std::ios::binary);rotationData.open(std::string(argv[4])+".sam3d.bin.feet",std::ios::binary);}
            csv << "depth_id,host,mode,fit_ms,hip_x,hip_y,hip_z,left_x,left_y,left_z,right_x,right_y,right_z,"
                   "left_sigma,right_sigma,hip_valid,left_valid,right_valid,left_contact,right_contact,"
                   "ambiguous";
            for (auto name : {"hip", "left", "right"})
                csv << ',' << name << "_qw," << name << "_qx," << name << "_qy," << name << "_qz," << name
                    << "_sigma_x_rad," << name << "_sigma_y_rad," << name << "_sigma_z_rad";
            for (int j = 0; j < J; ++j)
                for (auto field :
                     {"x", "y", "z", "confidence", "source", "rgb_u", "rgb_v", "rgb_score", "depth_rejection",
                      "depth_samples", "player_samples", "depth_spread", "prior_delta"})
                    csv << ",joint_" << j << '_' << field;
            for (int j = 0; j < J; ++j)
                csv << ",sdk_" << j << "_u,sdk_" << j << "_v,sdk_" << j << "_confidence";
            csv << ",inference_ms,pose_present,pose_registration_rms,hip_inferred,left_inferred,right_inferred,crop_ms,encoder_ms,decoder_ms,decoder_upload_ms,decoder_launch_ms,decoder_readback_ms,pose_root_anchors,pose_anchor_valid,hip_AI_position,left_AI_position,right_AI_position,left_sole_height,right_sole_height,left_native_rotation,right_native_rotation\n";
            std::uint32_t id = 0;
            std::uint32_t recordedSelection = ~0u;
            std::uint32_t epoch = ~0u, lastDepth = ~0u;
            std::optional<Crop> crop;
            double cropTime = 0;
            size_t count = 0,cacheFrame=0;
            while (auto f = reader.next()) {
                indexRegistration(*f);
                if (epoch != f->epoch) {
                    estimator.select(id);
                    continuity.reset();
                    epoch = f->epoch;
                    lastDepth = ~0u;
                    crop.reset();
                }
                if (lastDepth == f->depthId)
                    continue;
                lastDepth = f->depthId;
                std::optional<Sam3dPrediction> cachedPrediction;
                if(cached) {
                    std::array<float,350> data;
                    std::string line;std::getline(cacheIndex,line);
                    std::istringstream row(line);std::string depth,host;
                    std::getline(row,depth,',');std::getline(row,host,',');
                    if(!cache.read(reinterpret_cast<char *>(data.data()),sizeof(data)) ||
                       depth.empty() || std::stoul(depth)!=f->depthId || std::abs(std::stod(host)-f->host)>1e-5)
                        throw std::runtime_error("Cached prediction does not match recording frame");
                    std::array<float,18> feet{};
                    if(rotationCache.is_open() && !rotationCache.read(reinterpret_cast<char*>(feet.data()),sizeof(feet)))
                        throw std::runtime_error("Truncated SAM rotation cache");
                    if(cacheFrame++%stride)continue;
                    if(std::isfinite(data[0])) {
                        cachedPrediction.emplace();cachedPrediction->host=f->host;
                        if(rotationCache.is_open() && std::isfinite(feet[0])) {
                            for(int side=0;side<2;++side)for(int r=0;r<3;++r)for(int c=0;c<3;++c)
                                cachedPrediction->footRotations[side].a[r][c]=feet[side*9+r*3+c];
                            cachedPrediction->hasFootRotations=true;
                        }
                        for(int j=0;j<70;++j) {
                            cachedPrediction->cameraPoints[j]={data[j*5],data[j*5+1],data[j*5+2]};
                            cachedPrediction->imagePoints[j]={data[j*5+3],data[j*5+4]};
                        }
                    }
                }
                if (f->runConfig) {
                    if (f->runConfig->selectedId != recordedSelection) {
                        recordedSelection = f->runConfig->selectedId;
                        if (id != recordedSelection) {
                            id = f->runConfig->selectedId;
                            estimator.select(id);
                            continuity.reset();
                            crop.reset();
                        }
                    }
                    estimator.settings = f->runConfig->settings;
                    estimator.settings.baseline = mode == "raw" ? 1 : mode == "filtered" ? 2 : 0;
                    estimator.restoreLengths(f->runConfig->lengths);
                    cal = f->runConfig->calibration;
                    if (model.ready() && !f->runConfig->modelHash.empty() &&
                        model.hash() != f->runConfig->modelHash)
                        throw std::runtime_error("Recording model hash mismatch");
                }
                if (!id && !f->runConfig && !f->bodies.empty()) {
                    id = f->bodies[0].id;
                    estimator.select(id);
                }
                std::optional<Keypoints> kp;
                std::optional<PosePrior> prior;
                std::optional<Sam3dPrediction> samPrediction;
                std::optional<BodyPrediction> nlfPrediction;
                auto recoveredId = estimator.reconcileIdentity(*f, cal);
                if (recoveredId != id) {
                    id = recoveredId;
                    crop.reset();
                }
                if (auto current = playerCrop(*f, id)) {
                    crop = current;
                    cropTime = f->host;
                }
                double inferStart=now(),inferenceMs=0;
                if(cachedPrediction) {
                    samPrediction=cachedPrediction;
                    auto evidence=sam3dEvidence(*f,id,*samPrediction,sam3dCamera(fitColorProjection(*f)));
                    kp=evidence.keypoints;prior=evidence.prior;
                } else if (!cached && (model.ready() || sam.ready() || nlf.ready()) && estimator.settings.inference && id && crop && f->host - cropTime < .12) {
                    if(nlf.ready()) {
                        auto camera=sam3dCamera(fitColorProjection(*f));
                        if(camera.valid) {
                            nlfPrediction=nlf.infer(*f,*crop,camera);
                            auto evidence=bodyPoseEvidence(*f,id,*nlfPrediction,camera);
                            kp=evidence.keypoints;prior=evidence.prior;
                        }
                    } else if(sam.ready()) {
                        auto camera=sam3dCamera(fitColorProjection(*f));
                        if(camera.valid) {
                            auto c=*crop;c.w=c.h=std::max(c.w,c.h);
                            samPrediction=sam.infer(*f,c,camera);
                            auto evidence=sam3dEvidence(*f,id,*samPrediction,camera);
                            kp=evidence.keypoints;prior=evidence.prior;
                        }
                    } else kp = kinectImageLabels(model.infer(*f, *crop));
                    inferenceMs=(now()-inferStart)*1000;
                }
                const double bodyStart=now();
                if(estimator.settings.baseline==0 && estimator.settings.inference && estimator.settings.depth) {
                    auto stable=continuity.update(prior?*prior:PosePrior{},id,*f,cal,estimator.settings);
                    prior=stable.valid?std::optional<PosePrior>{stable}:std::nullopt;
                }else continuity.reset();
                const double bodyMs=(now()-bodyStart)*1000;
                auto s = estimator.process(*f, kp ? &*kp : nullptr, cal,prior?&*prior:nullptr);
                s.fitMs+=bodyMs;
                csv << std::setprecision(12) << f->depthId << ',' << f->host << ',' << int(s.mode) << ','
                    << s.fitMs;
                for (auto &t : s.trackers)
                    csv << ',' << t.p.x << ',' << t.p.y << ',' << t.p.z;
                csv << ',' << s.trackers[1].positionSigma << ',' << s.trackers[2].positionSigma;
                for (const auto &t : s.trackers)
                    csv << ',' << t.valid;
                csv << ',' << int(s.contacts[0].state) << ',' << int(s.contacts[1].state) << ','
                    << s.ambiguous;
                for (const auto &t : s.trackers)
                    csv << ',' << t.q.w << ',' << t.q.x << ',' << t.q.y << ',' << t.q.z << ','
                        << t.angularSigma.x << ',' << t.angularSigma.y << ',' << t.angularSigma.z;
                for (int j = 0; j < J; ++j) {
                    const auto &joint = s.body.joints[j];
                    auto key = kp ? (*kp)[j] : Keypoint{};
                    csv << ',' << joint.p.x << ',' << joint.p.y << ',' << joint.p.z << ',' << joint.confidence
                        << ',' << int(joint.source) << ',' << key.uv.x << ',' << key.uv.y << ',' << key.score;
                    const auto &support = s.depthSupport[j];
                    csv << ',' << support.rejection << ',' << support.samples << ',' << support.playerSamples
                        << ',' << support.spread << ',' << support.priorDelta;
                }
                auto projection = fitColorProjection(*f);
                auto sdk = std::find_if(f->bodies.begin(), f->bodies.end(),
                                        [&](const Body &body) { return body.id == id; });
                for (int j = 0; j < J; ++j) {
                    Joint raw = sdk != f->bodies.end() ? sdk->joints[j] : Joint{};
                    V2 uv = projection.valid && raw.confidence > 0 ? projection.project(raw.p) : V2{};
                    csv << ',' << uv.x << ',' << uv.y << ',' << raw.confidence;
                }
                csv << ','<<inferenceMs<<','<<(bool(samPrediction)||bool(nlfPrediction))<<','<<(prior?prior->registrationRms:0);
                for(bool learned:s.learnedDirection)csv<<','<<learned;
                for(double timing:nlf.ready()?nlf.timings():sam.timings())csv<<','<<timing;
                csv<<','<<(prior?prior->rootAnchors:0)<<','<<(prior && prior->valid);
                for(bool inferred:s.learnedPosition)csv<<','<<inferred;
                for(int side=0;side<2;++side) {
                    auto sole=s.body.joints[LAnkle+side].p-s.trackers[side+1].q.rotate({0,estimator.settings.soleOffset,0});
                    csv<<','<<(f->floor.valid?f->floor.height(sole):std::numeric_limits<double>::quiet_NaN());
                }
                csv << ','<<s.nativeFootDirection[0]<<','<<s.nativeFootDirection[1]<<'\n';
                if(sam.ready()) {
                    std::array<float,350> data;
                    data.fill(std::numeric_limits<float>::quiet_NaN());
                    if(samPrediction)for(int j=0;j<70;++j) {
                        auto p=samPrediction->cameraPoints[j];auto uv=samPrediction->imagePoints[j];
                        data[j*5]=float(p.x);data[j*5+1]=float(p.y);data[j*5+2]=float(p.z);data[j*5+3]=float(uv.x);data[j*5+4]=float(uv.y);
                    }
                    samData.write(reinterpret_cast<char*>(data.data()),sizeof(data));
                    std::array<float,18> feet;feet.fill(std::numeric_limits<float>::quiet_NaN());
                    if(samPrediction && samPrediction->hasFootRotations)for(int side=0;side<2;++side)for(int r=0;r<3;++r)for(int c=0;c<3;++c)
                        feet[side*9+r*3+c]=float(samPrediction->footRotations[side].a[r][c]);
                    rotationData.write(reinterpret_cast<char*>(feet.data()),sizeof(feet));
                    if(!samData || !rotationData)throw std::runtime_error("Cannot write SAM comparison data");
                }
                if (!csv)
                    throw std::runtime_error("Replay CSV write failed");
                ++count;
            }
            std::cout << "replayed_frames=" << count << "\n";
            if(cached && (cache.peek()!=std::char_traits<char>::eof() || cacheIndex.peek()!=std::char_traits<char>::eof() || (rotationCache.is_open() && rotationCache.peek()!=std::char_traits<char>::eof())))
                throw std::runtime_error("Extra cached frames after recording ended");
            return 0;
        }
        throw std::runtime_error("Unknown command or missing arguments");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
