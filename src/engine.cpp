#include "engine.hpp"
#include "sam3d_model.hpp"
#include "nlf_model.hpp"
#include "body_tracker.hpp"
#include "steamvr_bridge.hpp"
#include "vr_overlay.hpp"
#include <timeapi.h>
#include <iomanip>
#include <sstream>
namespace kf {
Engine::Engine(std::filesystem::path root) : root_(std::move(root)) {
    overlay_ = std::make_unique<VrOverlay>();
    view_.modelChoice=std::filesystem::exists(root_/"assets/sam3d-optimized/backbone.onnx")?3:
        (std::filesystem::exists(root_/"assets/sam3d-fp8/backbone.onnx")?2:
        (std::filesystem::exists(root_/"assets/nlf/pose.onnx")?1:0));
    std::ifstream modelFile(root_/"tracking-model.txt");std::string name;
    if(modelFile>>name){if(name=="nlf")view_.modelChoice=1;else if(name=="sam")view_.modelChoice=0;else if(name=="sam-fp8")view_.modelChoice=2;else if(name=="sam-optimized")view_.modelChoice=3;}
    std::ifstream outputFile(root_/"tracking-output.txt");std::string outputName;
    if(outputFile>>outputName)view_.steamVrOutput=outputName!="osc";
    calibrationFile_=root_/"calibration.txt";
    std::ifstream cadenceFile(root_/"tracking-cadence.txt");int savedCadence=0;
    if(cadenceFile>>savedCadence && savedCadence>=0 && savedCadence<=3) {
        view_.cadenceChoice=savedCadence;
        const char* names[]{"Auto (GPU adaptive)","30 Hz (Full AI)","20 Hz (Balanced)","15 Hz (Low GPU / Heavy VRChat)"};
        view_.cadenceStatus=names[savedCadence];
    }
    calibrationAccepted_ = loadCalibration(calibrationFile_, view_.calibration, view_.settings,&view_.wristOffsetsReady);
}
Engine::~Engine() {
    stop();
}
View Engine::view() const {
    std::lock_guard l(mutex_);
    auto copy = view_;
    copy.tiltWait=std::max(0.,tiltLimiter_.nextAllowed-now());
    if(copy.collecting) {
        auto cue=alignment_.cue(now());copy.calibrationSecondsRemaining=cue.seconds;
        copy.calibrationWaiting=cue.waitingForReady;
        copy.calibrationPrompt=cue.instruction+"\n"+(cue.waitingForReady?
            "Take your time. Squeeze either trigger or click Capture pose when ready.":cue.collecting?
            (cue.seconds?"Hold still: "+std::to_string(cue.seconds)+" seconds":"Keep holding while both wrists are measured."):
            "Settle into position. Capture in "+std::to_string(cue.seconds)+" seconds.");
        copy.calibrationSpeech=cue.speech;
    }
    if(copy.bodyCollecting) {
        double elapsed=now()-bodyCaptureStart_;
        copy.bodySecondsRemaining=std::max(0.,(elapsed<0?0:4)-elapsed);
        copy.bodySpeech=elapsed<0?"Stand naturally with your feet apart and arms relaxed. Capture begins in eight seconds.":"Hold still. Capturing proportions for four seconds.";
        copy.bodyPrompt=(elapsed<0?"Get into position: ":"Hold still — capturing: ")+std::to_string(int(std::ceil(copy.bodySecondsRemaining)))+" seconds";
    }
    if(copy.running && !copy.replay && copy.state.host>0) {
        auto delivered=deliveryState(copy.state,now());
        copy.state.mode=delivered.mode;
        for(int i=0;i<3;++i)copy.state.trackers[i].valid=delivered.trackers[i].valid;
    }
    return copy;
}
void Engine::message(const std::string &s) {
    std::lock_guard l(mutex_);
    view_.notice = s;
}
void Engine::start(const std::filesystem::path &replay) {
    stop();
    measurements_.reset();
    records_.reset();
    run_ = true;
    selection_ = 0;
    {
        std::lock_guard l(mutex_);
        view_.running = true;
        view_.replay = !replay.empty();
        view_.output = false;
        view_.frames = view_.sent = view_.dropped = 0;
        view_.state = {};
        view_.samOverlay.reset();view_.poseSource.clear();view_.modelHash.clear();
        view_.frame.reset();
        view_.exposureStatus.clear();
        view_.tiltAngle.reset();view_.tiltPending=false;tiltTarget_.reset();
        view_.calibration.valid = false;
        view_.calibration.rawReferenceValid=false;
        view_.notice = "Select and lock the player after a body appears.";
    }
    recorder_ = std::thread(&Engine::recordLoop, this);
    process_ = std::thread(&Engine::processLoop, this);
    capture_ = std::thread(&Engine::captureLoop, this, replay);
    output_ = std::thread(&Engine::outputLoop, this);
}
void Engine::stop() {
    run_ = false;
    record_ = false;
    measurements_.close();
    records_.close();
    for (auto *t : {&capture_, &process_, &output_, &recorder_})
        if (t->joinable())
            t->join();
    std::lock_guard l(mutex_);
    view_.running = false;
    view_.bodyCollecting=false;calibrateBody_=false;
    view_.recording = false;
    view_.output = false;
    view_.collecting = false;
    view_.sensor = "Stopped";
    view_.tiltAngle.reset();view_.tiltPending=false;tiltTarget_.reset();
    if (overlay_) overlay_->hide();
}
void Engine::select(uint32_t id) {
    selection_ = id;
    std::lock_guard l(mutex_);
    view_.output = false;
    view_.collecting = false;
    view_.bodyCollecting=false;calibrateBody_=false;
    view_.notice = "Player locked. Keep feet and wrists visible; Align to VR does not need your head in view.";
}
void Engine::settings(Settings s) {
    std::lock_guard l(mutex_);
    for (int i = 0; i < 3; ++i)
        if (norm(s.deviceOffsets[i] - view_.settings.deviceOffsets[i]) > 1e-9) {
            view_.calibration.valid = false;
            if(i>0)view_.wristOffsetsReady=false;
            calibrationAccepted_ = false;
            view_.collecting = false;
            view_.output = false;
            view_.calibration.spread = 0;
            view_.notice = "Device offsets changed; collect a new camera-to-VR alignment.";
        }
    view_.settings = s;
}
void Engine::chooseModel(int choice) {
    std::lock_guard l(mutex_);
    if(view_.running || choice<0 || choice>3)return;
    std::ofstream file(root_/"tracking-model.txt",std::ios::trunc);
    file<<(choice==3?"sam-optimized":choice==2?"sam-fp8":choice==1?"nlf":"sam")<<'\n';file.flush();
    if(!file){view_.notice="Could not save tracking model selection.";return;}
    view_.modelChoice=choice;view_.modelHash.clear();view_.inference="Not loaded";
    view_.notice="Tracking model selected. Press Start, lock your player, then confirm saved alignment.";
}
void Engine::bodyCalibration() {
    std::lock_guard l(mutex_);
    if(!view_.running || view_.replay || !selection_ || view_.collecting || view_.bodyCollecting){view_.notice="Start and lock your player before capturing proportions.";return;}
    bodyCaptureStart_=now()+8;view_.bodyCollecting=true;calibrateBody_=true;
    view_.notice="Stand naturally, with feet apart and arms relaxed. Eight seconds to get ready, then a four-second capture.";
}
void Engine::chooseExposure(bool prefer30) {
    std::lock_guard l(mutex_);
    if(view_.running)return;
    view_.prefer30=prefer30;
    view_.notice=prefer30?"30 fps priority selected. Press Start. Dim scenes may look darker or grainier.":
        "Automatic exposure selected. In dim light the colour camera may slow to 15 fps.";
}
void Engine::chooseCadence(int choice) {
    std::lock_guard l(mutex_);
    if(choice<0 || choice>3)return;
    view_.cadenceChoice=choice;
    std::ofstream file(root_/"tracking-cadence.txt",std::ios::trunc);
    file<<choice<<'\n';
    const char* names[]{"Auto (GPU adaptive)","30 Hz (Full AI)","20 Hz (Balanced)","15 Hz (Low GPU / Heavy VRChat)"};
    view_.cadenceStatus=names[choice];
    view_.notice=std::string("GPU tracking cadence: ")+names[choice]+".";
}
void Engine::beginCalibration() {
    std::lock_guard l(mutex_);
    if (!view_.frame || !selection_ || view_.replay || view_.bodyCollecting) {
        view_.notice = "Start live tracking and lock your player first.";
        return;
    }
    if(view_.tiltPending || !tiltLimiter_.ready(now())) {
        view_.notice="Wait for the tilt motor to settle before aligning.";return;
    }
    view_.collecting = true;
    view_.output = false;
    view_.calibration.valid = false;
    if(!view_.frame->vr.rawTransformValid) {
        view_.collecting=false;view_.notice="Connect SteamVR and both controllers before aligning.";return;
    }
    alignment_.reset(now(),&view_.frame->vr);
    calibrationAccepted_ = false;
    view_.calibrationSamples = 0;
    view_.calibrationDetail = alignment_.feedback();
    view_.notice = "Take your time on each pose. Squeeze either trigger, or click Capture pose, only when ready. No timed positioning and no exact palm twists; your head may be out of view.";
}
void Engine::captureAlignmentPose() {
    std::lock_guard l(mutex_);
    if(view_.collecting)alignment_.capturePose(now());
}
void Engine::cancelCalibration() {
    std::lock_guard l(mutex_);
    if(!view_.collecting)return;
    view_.collecting=false;view_.calibration.valid=false;view_.output=false;
    view_.notice="Alignment cancelled. Saved alignment and earlier device offsets were not overwritten.";
    if (overlay_) overlay_->hide();
}
void Engine::tilt(int direction) {
    std::lock_guard l(mutex_);
    if(view_.frame && view_.frame->sensorVersion==2){view_.notice="Kinect v2 has no motor. Adjust its mount by hand, then Align to VR again.";return;}
    if(!view_.running || view_.replay || !view_.tiltAngle) {view_.notice="Start live Kinect capture to adjust its motor.";return;}
    if(view_.recording) {view_.notice="Stop recording before adjusting camera tilt.";return;}
    if(view_.tiltPending || !tiltLimiter_.ready(now())) {view_.notice="Wait for the tilt motor cooldown.";return;}
    int target=TiltLimiter::target(*view_.tiltAngle,direction);
    if(target==*view_.tiltAngle){view_.notice="Kinect is at its tilt limit.";return;}
    // Persist invalidation before moving, so restarting cannot confirm a stale
    // transform. The motor target is never replayed automatically on startup.
    auto invalid=view_.calibration;invalid.valid=false;invalid.spread=0;
    saveCalibration(calibrationFile_,invalid,view_.settings,view_.wristOffsetsReady);
    view_.calibration=invalid;calibrationAccepted_=false;
    view_.output=false;view_.collecting=false;view_.calibrationDetail.clear();
    view_.bodyCollecting=false;calibrateBody_=false;
    tiltLimiter_.claim(now());tiltTarget_=target;view_.tiltPending=true;
    view_.notice="Adjusting camera tilt. Align to VR again after choosing the final camera angle.";
}
void Engine::useSavedCalibration() {
    std::lock_guard l(mutex_);
    if(view_.collecting || view_.bodyCollecting){view_.notice="Finish the current calibration first.";return;}
    if (!view_.frame || view_.replay || !calibrationAccepted_ || view_.calibration.spread < 0.07 ||
        view_.calibration.rms >= calibrationMaxRms) {
        view_.notice = "No previously accepted rigid calibration. Collect a new alignment.";
        return;
    }
    if(!view_.calibration.rawReferenceValid || view_.calibration.rawEpoch!=view_.frame->vr.epoch)
        if(!bindTrackingReference(view_.calibration,view_.frame->vr)) {
            view_.notice="SteamVR tracking reference is unavailable; reconnect the headset first.";return;
        }
    view_.calibration.valid = true;
    view_.notice = "Saved alignment confirmed. Space drag now moves all trackers together. Reset OVR offsets before confirming after an app restart.";
}
void Engine::chooseOutput(bool steamVr) {
    std::lock_guard l(mutex_);if(view_.output)return;
    std::ofstream file(root_/"tracking-output.txt");file<<(steamVr?"steamvr":"osc")<<'\n';file.flush();
    if(!file){view_.notice="Could not save output preference.";return;}
    view_.steamVrOutput=steamVr;
    view_.notice=steamVr?"SteamVR trackers selected. Restart SteamVR once after driver installation.":"OSC output selected.";
}
void Engine::output(bool enabled) {
    std::lock_guard l(mutex_);
    if (enabled && (!view_.calibration.valid || view_.replay || view_.collecting)) {
        view_.notice = "Live output requires an accepted camera-to-SteamVR alignment.";
        return;
    }
    view_.output = enabled;
    view_.notice = view_.steamVrOutput
        ? (enabled ? "Sending waist and foot trackers to SteamVR. Use VRChat's FBT calibration."
                   : "SteamVR tracker output paused.")
        : (enabled ? "Sending hips and two feet to localhost:9000. Enable OSC and calibrate FBT in VRChat."
                   : "OSC output paused.");
}
void Engine::toggleRecord() {
    std::lock_guard l(mutex_);
    if (record_) {
        record_ = false;
        view_.recording = false;
        view_.notice = "Recording stopped. Files are local in recordings/.";
        return;
    }
    if (!run_ || view_.replay) {
        view_.notice = "Local recording is available during live capture.";
        return;
    }
    std::filesystem::create_directories(root_ / "recordings");
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    recordPath_ = root_ / "recordings" / (std::to_string(ms) + ".kfr");
    std::ostringstream s;
    s << "{\"format\":2,\"app\":\"0.1.0\",\"rgb\":\"BGRA; dimensions in each frame\",\"depth\":\"native grid, packed "
         "millimeters/player\",\"model_sha256\":\"";
    s << view_.modelHash;
    s << "\",\"baseline\":" << view_.settings.baseline << ",\"depth_enabled\":" << view_.settings.depth
      << ",\"constraints\":" << view_.settings.constraints << ",\"contacts\":" << view_.settings.contacts
      << ",\"sole_offset_m\":" << view_.settings.soleOffset << ",\"offsets\":[";
    for (int i = 0; i < 3; ++i) {
        if (i)
            s << ',';
        auto v = view_.settings.deviceOffsets[i];
        s << '[' << v.x << ',' << v.y << ',' << v.z << ']';
    }
    s << "],\"calibration_valid\":" << view_.calibration.valid << "}";
    metadata_ = s.str();
    ++recordGeneration_;
    record_ = true;
    view_.recording = true;
    view_.notice = "Recording RGB and native depth locally (roughly 230-400 MB/s); disk drops are counted.";
}
void Engine::captureLoop(std::filesystem::path replay) {
    try {
        if (!replay.empty()) {
            RecordingReader reader;
            reader.open(replay);
            {
                std::lock_guard l(mutex_);
                view_.sensor = "Deterministic replay - network output disabled";
            } // Replay handshakes each frame so disk speed cannot change estimator inputs.
            double previous = 0;
            while (run_) {
                auto f = reader.next();
                if (!f)
                    break;
                if (previous > 0 && f->host > previous) {
                    double end = now() + std::min(f->host - previous, 2.0);
                    while (run_ && now() < end)
                        std::this_thread::sleep_for(std::chrono::milliseconds(3));
                }
                previous = f->host;
                std::uint64_t before = view().frames;
                measurements_.push(f);
                while (run_ && view().frames == before)
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            message("Replay finished. No OSC packets were sent.");
            return;
        }
        KinectCapture sensor;
        VrInput vr;
        PoseHistory history;
        double nextRetry = 0, lastVr = 0, lastTiltRead=0;
        std::uint64_t lastEpoch = 0;
        unsigned captureEpoch = 0;
        while (run_) {
            double t = now();
            if (t - lastVr >= 0.008) {
                auto pose = vr.poll();
                if (lastEpoch && pose.epoch != lastEpoch) {
                    std::lock_guard l(mutex_);
                    view_.calibration.valid = false;
                    view_.output = false;
                    view_.notice = "SteamVR origin changed or restarted; repeat alignment.";
                    calibrationAccepted_ = false;
                    view_.collecting = false;
                }
                lastEpoch = pose.epoch;
                history.add(pose);
                lastVr = t;
                std::lock_guard l(mutex_);
                view_.vr = vr.status;
            }
            if (t >= nextRetry && !sensor.healthy()) {
                try {
                    sensor.open(view().prefer30);
                    ++captureEpoch;
                    {
                        std::lock_guard l(mutex_);
                        view_.sensor = sensor.status;
                        view_.exposureStatus = sensor.exposureStatus;
                        auto key=sensor.calibrationKey();
                        auto path=root_/(key.empty()?"calibration.txt":("calibration-"+key+".txt"));
                        if(path!=calibrationFile_) {
                            calibrationFile_=path;
                            view_.calibration={};
                            // Keep user settings if this sensor has no saved alignment.
                            calibrationAccepted_=loadCalibration(path,view_.calibration,view_.settings,&view_.wristOffsetsReady);
                            view_.notice=calibrationAccepted_?"Sensor found. Lock your player and confirm this sensor's saved alignment.":
                                "New Kinect: lock your player, capture proportions, then Align to VR.";
                        }
                        view_.calibration.valid = false;
                        if (captureEpoch > 1) {
                            calibrationAccepted_ = false;
                            view_.collecting = false;
                        }
                        view_.output = false;
                    }
                } catch (const std::exception &e) {
                    sensor.close();
                    std::lock_guard l(mutex_);
                    view_.sensor = e.what();
                    view_.tiltAngle.reset();view_.tiltPending=false;tiltTarget_.reset();
                }
                nextRetry = t + 3;
            }
            std::optional<int> target;
            {std::lock_guard l(mutex_);target=tiltTarget_;tiltTarget_.reset();}
            if(target) {
                try {sensor.elevation(*target);++captureEpoch;}
                catch(const std::exception &e) {
                    std::lock_guard l(mutex_);
                    view_.notice=e.what();tiltLimiter_.nextAllowed=now()+20;
                }
                std::lock_guard l(mutex_);view_.tiltPending=false;
                tiltLimiter_.nextAllowed=std::max(tiltLimiter_.nextAllowed,now()+1.);
            }
            if(t-lastTiltRead>.5) {
                auto angle=sensor.elevation();lastTiltRead=t;
                std::lock_guard l(mutex_);view_.tiltAngle=angle;
            }
            try {
                auto frame=sensor.poll();
                {std::lock_guard l(mutex_);view_.exposureStatus=sensor.exposureStatus;}
                if (frame) {
                    frame->epoch += captureEpoch * 1000;
                    if (auto poses = history.at(frame->host))
                        frame->vr = *poses;
                    auto snapshot = view();
                    auto cfg = std::make_shared<ReplayConfig>();
                    cfg->settings = snapshot.settings;
                    cfg->calibration = snapshot.calibration;
                    cfg->selectedId = selection_;
                    cfg->lengths = snapshot.lengths;
                    cfg->modelHash = snapshot.modelHash;
                    frame->runConfig = cfg;
                    if (record_)
                        records_.push(frame);
                    measurements_.push(std::move(frame));
                    std::lock_guard l(mutex_);
                    view_.dropped = sensor.dropped + measurements_.dropped;
                    view_.recordDrops = records_.dropped;
                }
            } catch (const std::exception &e) {
                sensor.close();
                message(e.what());
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } catch (const std::exception &e) {
        message(e.what());
    }
}
void Engine::processLoop() {
    try {
        PoseModel model;
        Sam3dModel sam;
        NlfModel nlf;
        BodyTracker continuity;
        const int choice=view().modelChoice;
        const std::string poseName=choice==1?"NLF-S":"SAM";
        {
            std::lock_guard l(mutex_);
            view_.inference = "Loading and warming pose model...";
        }
        try {
            if(choice==1) {
                nlf.load(root_/"assets/nlf/pose.onnx",root_);
                std::lock_guard l(mutex_);
                view_.inference="NLF-S: NVIDIA accelerated body and feet";
                view_.modelHash=nlf.hash();
            } else if(choice>=2 || (useGpu_ && std::filesystem::exists(root_/"assets/sam3d/backbone.onnx"))) {
                sam.load(root_/(choice==3?"assets/sam3d-optimized/backbone.onnx":choice==2?"assets/sam3d-fp8/backbone.onnx":"assets/sam3d/backbone.onnx"),root_);
                std::lock_guard l(mutex_);
                view_.inference=std::string("Fast SAM body: TensorRT ")+(choice>=2?"selective FP8":"FP16")+" encoder + native CUDA "+sam.decoderPrecision()+" decoder"+(choice==3?"; optimized sampling":"");
                view_.modelHash=sam.hash();
            } else {
            std::ifstream expected(root_ / "assets/pose.sha256");
            std::string hash;
            expected >> hash;
            if (hash.size() != 64 || sha256(root_ / "assets/pose.onnx") != hash)
                throw std::runtime_error("Pose model hash missing/mismatched");
            try {
                model.load(root_ / "assets/pose.onnx", useGpu_, root_);
            } catch (const std::exception &e) {
                message(std::string("NVIDIA backend unavailable: ") + e.what() +
                        ". Loading CPU FP32 fallback.");
                model.load(root_ / "assets/pose.onnx", false, root_);
            }
            std::lock_guard l(mutex_);
            view_.inference = model.backend();
            view_.modelHash = model.hash();
            }
        } catch (const std::exception &e) {
            std::lock_guard l(mutex_);
            view_.inference = std::string("SDK fallback: ") + e.what();
        }
        Estimator estimator;
        std::uint32_t id = 0, lastDepth = ~0u, lastEpoch = ~0u;
        std::uint32_t recordedSelection = ~0u;
        std::optional<Crop> lastCrop;
        double cropTime = 0;
        std::vector<Body> bodySamples;
        bool collectingBody = false;
        bool samReplayNotice=false;
        std::optional<Sam3dPrediction> lastSamPrediction;
        std::optional<Sam3dPrediction> prevSamPrediction;
        std::optional<BodyPrediction> lastNlfPrediction;
        std::optional<BodyPrediction> prevNlfPrediction;
        double lastInferenceHost = 0;
        double prevInferenceHost = 0;
        double prevNlfInferenceHost = 0;
        uint64_t inferenceFrameCount = 0;
        int autoThrottleRemaining = 0;
        int autoThrottleLevel = 0;
        double lastMeasuredInferenceMs = 0;
        while (run_) {
            auto next = measurements_.pop(!view().replay);
            if (!next)
                continue;
            auto frame = *next;
            indexRegistration(*frame);
            if(frame->sensorVersion==2)frame->colorProjection=std::make_shared<ColorProjection>(fitColorProjection(*frame));
            auto config = view();
            if (config.replay && frame->runConfig) {
                if((sam.ready() || nlf.ready()) && !samReplayNotice) {
                    message("Replay comparison with "+poseName+"; recorded geometry and settings retained. OSC is disabled.");
                    samReplayNotice=true;
                }
                config.settings = frame->runConfig->settings;
                // Preserve user-selected ablations while restoring all recorded geometric parameters.
                config.settings.baseline = view().settings.baseline;
                config.calibration = frame->runConfig->calibration;
                if (recordedSelection != frame->runConfig->selectedId) {
                    recordedSelection = frame->runConfig->selectedId;
                    selection_ = recordedSelection;
                }
                if (model.ready() && !frame->runConfig->modelHash.empty() &&
                    frame->runConfig->modelHash != model.hash())
                    throw std::runtime_error("Replay model hash differs from the recorded model");
            }
            if (frame->epoch != lastEpoch) {
                if (lastEpoch != ~0u) {
                    std::lock_guard l(mutex_);
                    view_.calibration.valid = false;
                    view_.output = false;
                    config.calibration.valid = false;
                    calibrationAccepted_ = false;
                    view_.collecting = false;
                }
                estimator.select(id);
                continuity.reset();
                collectingBody=false;{std::lock_guard l(mutex_);view_.bodyCollecting=false;}
                lastDepth = ~0u;
                lastEpoch = frame->epoch;
                lastCrop.reset();
                lastSamPrediction.reset();
                prevSamPrediction.reset();
                lastNlfPrediction.reset();
                prevNlfPrediction.reset();
                lastInferenceHost = 0;
                prevInferenceHost = 0;
                prevNlfInferenceHost = 0;
            }
            if (frame->depthId == lastDepth)
                continue;
            lastDepth = frame->depthId;
            if (id != selection_) {
                id = selection_;
                estimator.select(id);
                continuity.reset();
                collectingBody=false;{std::lock_guard l(mutex_);view_.bodyCollecting=false;}
                lastCrop.reset();
                lastSamPrediction.reset();
                prevSamPrediction.reset();
                lastNlfPrediction.reset();
                prevNlfPrediction.reset();
                lastInferenceHost = 0;
                prevInferenceHost = 0;
                prevNlfInferenceHost = 0;
            }
            estimator.settings = config.settings;
            if (config.replay && frame->runConfig)
                estimator.restoreLengths(frame->runConfig->lengths);
            auto recoveredId = estimator.reconcileIdentity(*frame, config.calibration);
            if (recoveredId != id) {
                // Do not overwrite a concurrent explicit player selection from the GUI.
                auto expectedId = id;
                if (selection_.compare_exchange_strong(expectedId, recoveredId)) {
                    id = recoveredId;
                    continuity.reset();
                    collectingBody=false;calibrateBody_=false;
                    {std::lock_guard l(mutex_);view_.bodyCollecting=false;}
                    lastCrop.reset();
                    lastSamPrediction.reset();
                    prevSamPrediction.reset();
                    lastNlfPrediction.reset();
                    prevNlfPrediction.reset();
                    lastInferenceHost = 0;
                    prevInferenceHost = 0;
                    prevNlfInferenceHost = 0;
                    message("Player recovered after Kinect changed its body ID; headset and both controllers "
                            "matched.");
                } else
                    continue;
            }
            if (calibrateBody_.exchange(false)) {
                bodySamples.clear();
                collectingBody = true;
                message("Get ready for the timed proportions capture.");
            }
            if (auto crop = playerCrop(*frame, id)) {
                lastCrop = crop;
                cropTime = frame->host;
            }
            std::optional<Keypoints> keypoints;
            std::optional<PosePrior> learned;
            std::optional<Keypoints> samOverlay;
            double begin = now(), inferenceMs = 0;
            ++inferenceFrameCount;
            int cadence = config.cadenceChoice; // 0 Auto, 1 Full (30Hz), 2 Balanced (20Hz), 3 Low GPU (15Hz)
            bool shouldInfer = true;
            if (config.replay) {
                shouldInfer = true;
            } else if (cadence == 1) {
                shouldInfer = true;
            } else if (cadence == 2) {
                shouldInfer = (inferenceFrameCount % 3 != 0); // 20 Hz (2 of 3)
            } else if (cadence == 3) {
                shouldInfer = (inferenceFrameCount % 2 != 0); // 15 Hz (1 of 2)
            } else {
                // Auto: graduated throttling based on GPU contention.
                // 30 Hz frame budget is 33.3ms. SAM on RTX 5070 Ti takes ~22-26ms.
                // Level 2 (15 Hz): severe contention (inference > 32ms or queue > 20ms)
                // Level 1 (20 Hz): moderate contention (inference > 28.5ms or queue > 14ms)
                if (lastMeasuredInferenceMs > 32.0 || config.queueMs > 20.0) {
                    autoThrottleLevel = 2; // 15 Hz
                    autoThrottleRemaining = 30;
                } else if (lastMeasuredInferenceMs > 28.5 || config.queueMs > 14.0) {
                    if (autoThrottleLevel < 1) autoThrottleLevel = 1; // 20 Hz
                    autoThrottleRemaining = 30;
                }
                if (autoThrottleRemaining > 0) {
                    --autoThrottleRemaining;
                    if (autoThrottleLevel == 2) {
                        shouldInfer = (inferenceFrameCount % 2 != 0); // 15 Hz
                    } else {
                        shouldInfer = (inferenceFrameCount % 3 != 0); // 20 Hz
                    }
                } else {
                    autoThrottleLevel = 0;
                    shouldInfer = true; // 30 Hz full
                }
            }
            if ((model.ready() || sam.ready() || nlf.ready()) && id && config.settings.inference && config.settings.baseline == 0 &&
                lastCrop && frame->host - cropTime < 0.12) {
                try {
                    // SDK v1 RGB is mirrored relative to the anatomical labels used by the model.
                    // Registration already addresses the original pixels: swap labels, not pixels.
                    if(nlf.ready()) {
                        auto camera=sam3dCamera(fitColorProjection(*frame));
                        if(camera.valid) {
                            if (shouldInfer || !lastNlfPrediction || frame->host - lastInferenceHost > 0.12) {
                                prevNlfPrediction = lastNlfPrediction;
                                prevNlfInferenceHost = lastInferenceHost;
                                lastNlfPrediction = nlf.infer(*frame, *lastCrop, camera);
                                lastInferenceHost = frame->host;
                                inferenceMs = (now() - begin) * 1000;
                                lastMeasuredInferenceMs = inferenceMs;
                            }
                            if (lastNlfPrediction) {
                                auto pred = *lastNlfPrediction;
                                pred.host = frame->host;
                                double dtInfer = lastInferenceHost - prevNlfInferenceHost;
                                double dtSkip = frame->host - lastInferenceHost;
                                if (!shouldInfer && prevNlfPrediction && dtInfer > 0.015 && dtInfer < 0.15 && dtSkip > 0 && dtSkip < 0.12) {
                                    double ratio = std::clamp(dtSkip / dtInfer, 0.0, 1.0);
                                    for (size_t i = 0; i < pred.landmarks.size(); ++i) {
                                        V2 imgVel = {lastNlfPrediction->landmarks[i].uv.x - prevNlfPrediction->landmarks[i].uv.x,
                                                     lastNlfPrediction->landmarks[i].uv.y - prevNlfPrediction->landmarks[i].uv.y};
                                        double l = std::hypot(imgVel.x, imgVel.y);
                                        if (l > 250.0 * dtInfer && l > 1e-9) { imgVel.x *= (250.0 * dtInfer / l); imgVel.y *= (250.0 * dtInfer / l); }
                                        pred.landmarks[i].uv.x += imgVel.x * ratio;
                                        pred.landmarks[i].uv.y += imgVel.y * ratio;
                                    }
                                }
                                samOverlay = kinectImageLabels(pred.landmarks);
                                auto evidence = bodyPoseEvidence(*frame, id, pred, camera);
                                keypoints = evidence.keypoints; learned = evidence.prior;
                            }
                        }
                    } else if(sam.ready()) {
                        auto camera=sam3dCamera(fitColorProjection(*frame));
                        if(camera.valid) {
                            if (shouldInfer || !lastSamPrediction || frame->host - lastInferenceHost > 0.12) {
                                auto crop=*lastCrop;crop.w=crop.h=std::max(crop.w,crop.h);
                                prevSamPrediction = lastSamPrediction;
                                prevInferenceHost = lastInferenceHost;
                                lastSamPrediction = sam.infer(*frame, crop, camera);
                                lastInferenceHost = frame->host;
                                inferenceMs = (now() - begin) * 1000;
                                lastMeasuredInferenceMs = inferenceMs;
                            }
                            if (lastSamPrediction) {
                                auto pred = *lastSamPrediction;
                                pred.host = frame->host;
                                double dtInfer = lastInferenceHost - prevInferenceHost;
                                double dtSkip = frame->host - lastInferenceHost;
                                if (!shouldInfer && prevSamPrediction && dtInfer > 0.015 && dtInfer < 0.15 && dtSkip > 0 && dtSkip < 0.12) {
                                    double ratio = std::clamp(dtSkip / dtInfer, 0.0, 1.0);
                                    for (int i = 0; i < 70; ++i) {
                                        V3 vel = lastSamPrediction->cameraPoints[i] - prevSamPrediction->cameraPoints[i];
                                        pred.cameraPoints[i] += bounded(vel, 3.5 * dtInfer) * ratio;
                                        V2 imgVel = {lastSamPrediction->imagePoints[i].x - prevSamPrediction->imagePoints[i].x,
                                                     lastSamPrediction->imagePoints[i].y - prevSamPrediction->imagePoints[i].y};
                                        double l = std::hypot(imgVel.x, imgVel.y);
                                        if (l > 250.0 * dtInfer && l > 1e-9) { imgVel.x *= (250.0 * dtInfer / l); imgVel.y *= (250.0 * dtInfer / l); }
                                        pred.imagePoints[i].x += imgVel.x * ratio;
                                        pred.imagePoints[i].y += imgVel.y * ratio;
                                    }
                                }
                                samOverlay=kinectImageLabels(sam3dLandmarks(pred));
                                auto evidence=sam3dEvidence(*frame, id, pred, camera);
                                keypoints=evidence.keypoints; learned=evidence.prior;
                            }
                        }
                    } else {
                        keypoints = kinectImageLabels(model.infer(*frame, *lastCrop));
                        inferenceMs = (now() - begin) * 1000;
                        lastMeasuredInferenceMs = inferenceMs;
                    }
                } catch (const std::exception &e) {
                    message(e.what());
                }
            }
            // Calibration consumes only the raw, currently depth-registered evidence.
            const auto calibrationPrior=learned;
            const double bodyStart=now();
            if(config.settings.baseline==0 && config.settings.inference && config.settings.depth) {
                auto stable=continuity.update(learned?*learned:PosePrior{},id,*frame,config.calibration,config.settings);
                learned=stable.valid?std::optional<PosePrior>{stable}:std::nullopt;
            }else continuity.reset();
            const double bodyMs=(now()-bodyStart)*1000;
            auto state = estimator.process(*frame, keypoints ? &*keypoints : nullptr, config.calibration,learned?&*learned:nullptr);
            state.fitMs+=bodyMs;
            auto rawBody = std::find_if(frame->bodies.begin(), frame->bodies.end(),
                                        [&](const Body &body) { return body.id == id; });
            if(collectingBody){std::lock_guard l(mutex_);collectingBody=view_.bodyCollecting;}
            if(collectingBody) {
                double captureStart;{std::lock_guard l(mutex_);captureStart=bodyCaptureStart_;}
                if(frame->host>=captureStart && frame->host<captureStart+4) {
                    if(sam.ready() || nlf.ready()) {
                        if(calibrationPrior && calibrationPrior->valid) {
                            Body sample;sample.id=id;
                            for(int j=0;j<J;++j)if(calibrationPrior->available[j])sample.joints[j]={calibrationPrior->points[j],.8,.1,5};
                            bodySamples.push_back(sample);
                        }
                    } else if(rawBody!=frame->bodies.end())bodySamples.push_back(*rawBody);
                }
                if(frame->host>=captureStart+4) {
                    bool accepted=bodySamples.size()>=30 && estimator.calibrateLengths(bodySamples);
                    if(accepted && (sam.ready() || nlf.ready()))continuity.setLengths(estimator.lengths());
                    collectingBody=false;
                    {std::lock_guard l(mutex_);view_.bodyCollecting=false;}
                    message(accepted?"Proportions captured. You can move normally.":"Not enough steady body data. Keep your body visible and try again.");
                }
            }
            std::lock_guard l(mutex_);
            view_.frame = frame;
            view_.state = state;
            view_.samOverlay=samOverlay;
            std::string cadenceDesc = cadence == 1 ? "Full (30 Hz)" :
                                      cadence == 2 ? "Balanced (20 Hz)" :
                                      cadence == 3 ? "Low GPU (15 Hz)" :
                                      (autoThrottleRemaining > 0 ? (autoThrottleLevel == 2 ? "Auto (throttled 15 Hz)" : "Auto (throttled 20 Hz)") : "Auto (30 Hz)");
            view_.cadenceStatus = cadenceDesc;
            view_.poseSource=(sam.ready() || nlf.ready())?(learned && std::any_of(state.learnedPosition.begin(),state.learnedPosition.end(),[](bool b){return b;})?
                (learned->vrAnchored?poseName+" body + gentle horizontal correction":learned->depthPredicted?poseName+" body / distance estimated ("+std::to_string(int(learned->depthAge*1000))+" ms)":
                poseName+" body + depth ("+std::to_string(learned->rootAnchors)+" supports)"):
                config.settings.baseline!=0?"SDK comparison mode":
                (!config.settings.inference || !config.settings.depth)?poseName+" pose disabled in settings":
                "SDK fallback: "+poseName+" depth anchor unavailable"):"";
            view_.lengths = estimator.lengths();
            view_.inferenceMs = lastMeasuredInferenceMs;
            view_.queueMs = config.replay ? 0 : std::max(0.0, (begin - frame->arrival) * 1000);
            view_.arrivalToEstimateMs = config.replay ? 0 : (now() - frame->arrival) * 1000;
            ++view_.frames;
            if(view_.collecting) {
                alignment_.add(*frame,id,view_.settings);
                view_.calibrationSamples=unsigned(alignment_.size());
                view_.calibrationDetail=alignment_.feedback();
                if(alignment_.done()) {
                    view_.collecting=false;view_.calibration=alignment_.result();
                    calibrationAccepted_=view_.calibration.valid;view_.notice=view_.calibration.reason;
                    if(view_.calibration.valid) {
                        view_.calibrationDetail=alignment_.agreement();
                        for(int d=1;d<=2;++d)view_.settings.deviceOffsets[d]=alignment_.offsets()[d];
                        view_.wristOffsetsReady=true;
                        saveCalibration(calibrationFile_,view_.calibration,view_.settings,true);
                    }
                }
            }
        }
    } catch (const std::exception &e) {
        message(std::string("Processing stopped: ") + e.what());
    }
}
void Engine::outputLoop() {
    timeBeginPeriod(1);
    OscOutput osc;SteamVrBridge bridge;
    HANDLE writer=CreateMutexW(nullptr,FALSE,L"Local\\KinectFBT_Writer_v1");bool claimed=false;
    while(run_) {
        auto s=view();bool wants=s.output && s.calibration.valid && !s.replay;
        std::string status;
        if(wants && s.steamVrOutput) {
            if(!claimed && writer){auto result=WaitForSingleObject(writer,0);claimed=result==WAIT_OBJECT_0 || result==WAIT_ABANDONED;}
            bool active=bridge.driverReady();
            if(claimed && s.frame) {
                auto packet=trackerPacket(s.state,s.calibration,s.frame->vr,now(),true);
                bool ok=bridge.publish(packet,active);
                if(!ok)status="SteamVR connection busy";
                else if(!active)status="SteamVR driver not running - restart SteamVR after installation";
                else if(!packet.enabled)status="Waiting for the SteamVR coordinate origin";
                else status="SteamVR hip and foot trackers connected";
            }else status="Another tracker app is using SteamVR output";
        }else {
            if(claimed){BridgePacket off;off.published=now();bool ready;bridge.publish(off,ready);ReleaseMutex(writer);claimed=false;}
            if(wants) {
                auto trackers=deliveryState(s.state,now()).trackers;
                const bool referenceReady=!s.calibration.rawReferenceValid ||
                    (s.frame && trackingReferenceValid(s.calibration,s.frame->vr));
                const auto transform=s.calibration.rawReferenceValid && referenceReady?
                    currentStandingCalibration(s.calibration,s.frame->vr):s.calibration.transform;
                auto bytes=referenceReady?oscBundle(trackers,transform):std::vector<uint8_t>{};
                if(!bytes.empty()) {
                    bool ok=osc.send(bytes);std::lock_guard l(mutex_);
                    if(ok)++view_.sent;else ++view_.sendErrors;
                }
                status="OSC output active";
            }else status=s.steamVrOutput?(bridge.driverReady()?"SteamVR driver ready - output paused":"SteamVR driver not running - restart SteamVR after installation"):"OSC output paused";
        }
        {std::lock_guard l(mutex_);view_.outputStatus=status;}
        if (overlay_) {
            OverlayState os;
            double nowTime = now();
            if (s.collecting) {
                lastCalibrationActive_ = nowTime;
                auto cue = alignment_.cue(nowTime);
                os.active = true;
                os.step = cue.step;
                os.secondsRemaining = cue.seconds;
                os.waitingForReady = cue.waitingForReady;
                os.collecting = cue.collecting;
                os.instruction = cue.instruction;
                os.feedback = s.calibrationDetail;
                if (s.frame) {
                    os.leftTracked = s.frame->vr.devices[1].valid;
                    os.rightTracked = s.frame->vr.devices[2].valid;
                    os.leftTrigger = s.frame->vr.triggerPressed[0];
                    os.rightTrigger = s.frame->vr.triggerPressed[1];
                }
                overlay_->update(os);
            } else if (lastCalibrationActive_ > 0 && nowTime - lastCalibrationActive_ < 3.5 && s.wristOffsetsReady) {
                os.active = true;
                os.done = true;
                os.success = true;
                os.agreement = s.calibrationDetail;
                overlay_->update(os);
            } else {
                lastCalibrationActive_ = 0;
                overlay_->hide();
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(8));
    }
    if(claimed){BridgePacket off;off.published=now();bool ready;bridge.publish(off,ready);ReleaseMutex(writer);}
    if(writer)CloseHandle(writer);
    if (overlay_) overlay_->hide();
    timeEndPeriod(1);
}
void Engine::recordLoop() {
    RecordingWriter writer;
    uint64_t generation = 0;
    while (run_) {
        if (!record_) {
            if (writer.active())
                writer.close();
            records_.pop(true);
            continue;
        }
        try {
            std::filesystem::path path;
            std::string meta;
            uint64_t gen;
            {
                std::lock_guard l(mutex_);
                path = recordPath_;
                meta = metadata_;
                gen = recordGeneration_;
            }
            if (gen != generation) {
                writer.open(path, meta);
                generation = gen;
            }
            if (auto frame = records_.pop())
                writer.write(**frame);
        } catch (const std::exception &e) {
            record_ = false;
            writer.close();
            std::lock_guard l(mutex_);
            view_.recording = false;
            view_.notice = e.what();
        }
    }
    writer.close();
}
void Engine::exportDiagnostics() {
    auto s = view();
    std::filesystem::create_directories(root_ / "diagnostics");
    std::ofstream f(root_ / "diagnostics/status.txt");
    f << "Kinect RGB-D 0.1.0 experimental\n"
      << s.sensor << '\n'
      << "exposure_preference=" << (s.prefer30?"prefer-30":"auto") << '\n'
      << "exposure_control=" << s.exposureStatus << '\n'
      << s.inference << '\n'
      << s.vr << "\noutput_target=" << (s.steamVrOutput?"steamvr":"osc")
      << "\noutput_enabled=" << s.output << "\noutput_status=" << s.outputStatus
      << "\nmode=" << modeName(s.state.mode) << "\nframes=" << s.frames
      << "\nqueue_drops=" << s.dropped << "\nrecord_drops=" << s.recordDrops << "\nOSC_sent=" << s.sent
      << "\nOSC_send_errors=" << s.sendErrors << "\ninference_ms=" << s.inferenceMs
      << "\nfit_ms=" << s.state.fitMs << "\nqueue_ms=" << s.queueMs
        << "\narrival_to_estimate_ms=" << s.arrivalToEstimateMs << "\ncalibration_rms_m=" << s.calibration.rms
      << "\npose_source=" << s.poseSource << "\nselected_model=" << (s.modelChoice==3?"SAM optimized":s.modelChoice==2?"SAM selective FP8":s.modelChoice==1?"NLF-S":"SAM FP16")
      << "\ncalibration_p95_m=" << s.calibration.p95 << "\ncalibration_result=" << s.calibration.reason
      << "\nwrist_offsets_ready=" << s.wristOffsetsReady
      << "\nalignment_routine=" << "guided_learn_offsets_with_holdout"
      << "\ncalibration_pacing=user_ready_then_3s_settle_and_3s_observed_hold"
      << "\nalignment_method=controllers_only_automatic_wrist_offsets"
      << "\nkinect_tilt_degrees=" << (s.tiltAngle?std::to_string(*s.tiltAngle):"unavailable")
      << "\nkinect_tilt_wait_s=" << s.tiltWait
      << "\ncalibration_rms_limit_m=" << calibrationMaxRms
      << "\ncalibration_min_consistent_fraction=" << calibrationMinInlierFraction
      << "\nNOTE: latest samples, not percentile latency; no motion-to-photon claim.\n";
    if(s.frame)f<<"\nsensor_version="<<s.frame->sensorVersion<<"\ncolor_dimensions="<<s.frame->width<<'x'<<s.frame->height
        <<"\ndepth_dimensions="<<s.frame->depthWidth<<'x'<<s.frame->depthHeight<<"\ncapture_ms="<<s.frame->captureMs
        <<"\ncolor_exposure_ms="<<s.frame->exposureMs<<"\ncolor_interval_ms="<<s.frame->colorIntervalMs<<'\n';
    if(s.frame && s.frame->floor.valid) {
        auto floor=s.frame->floor;
        f<<"floor_camera_height_m="<<floor.d<<"\nfloor_normal="<<floor.n.x<<','<<floor.n.y<<','<<floor.n.z<<'\n';
        if(s.calibration.valid)f<<"calibrated_floor_origin_height_m="<<s.calibration.transform.apply(floor.n*(-floor.d)).y
            <<"\nfloor_up_error_degrees="<<std::acos(std::clamp(dot(s.calibration.transform.q.rotate(floor.n),V3{0,1,0}),-1.,1.))*180/pi<<'\n';
        for(int side=0;side<2;++side){auto t=s.state.trackers[side+1];auto sole=t.p-t.q.rotate(s.settings.trackerOffsets[side+1]);
            f<<"foot_"<<side<<"_corrected_sole_height_m="<<floor.height(sole)<<'\n';}
    }
    f << "\n" << s.calibrationDetail << '\n';
    for(int d=0;d<3;++d) {
        auto offset=s.settings.deviceOffsets[d];
        f<<"device_"<<d<<"_offset_m="<<offset.x<<','<<offset.y<<','<<offset.z<<'\n';
    }
    f << "locked_body_id=" << selection_ << "\nestimated_body_id=" << s.state.body.id
      << "\norientation_ambiguous=" << s.state.ambiguous << "\nvisible_body_ids=";
    if (s.frame)
        for (const auto &body : s.frame->bodies)
            f << body.id << ' ';
    f << '\n';
    for (int i = 0; i < 3; ++i) {
        const auto &tracker = s.state.trackers[i];
        f << "tracker_" << i + 1 << "_valid=" << tracker.valid << "\ntracker_" << i + 1
          << "_position_m=" << tracker.p.x << ',' << tracker.p.y << ',' << tracker.p.z << "\ntracker_"
          << i+1 << "_SAM_position=" << s.state.learnedPosition[i] << "\ntracker_"
          << i + 1 << "_quaternion_wxyz=" << tracker.q.w << ',' << tracker.q.x << ',' << tracker.q.y << ','
          << tracker.q.z << "\ntracker_" << i + 1 << "_yaw_prior_sigma_rad=" << tracker.angularSigma.y
          << '\n';
    }
    {
        std::lock_guard l(mutex_);
        std::ofstream csv(root_ / "diagnostics/alignment-samples.csv");
        alignment_.writeCsv(csv);
        if (!csv)
            throw std::runtime_error("Cannot write alignment diagnostics");
    }
    message("Diagnostics exported: status.txt and alignment-samples.csv. Local 3D alignment samples only; no "
            "camera images.");
}
} // namespace kf
