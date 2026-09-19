#pragma once
#include "core.hpp"
#include <ostream>
namespace kf {
// An accepted transform survives temporary loss and cancelled calibration.
// Its live reference must not be rebound silently after a real VR origin change.
class SavedAlignment {
    Calibration saved_;
    bool accepted_{};
public:
    void remember(const Calibration& cal,bool accepted=true){saved_=cal;accepted_=accepted;}
    void invalidate(){accepted_=false;}
    bool available() const{return accepted_;}
    Calibration confirm(const VrSample& vr) const {
        auto out=saved_;out.valid=false;
        if(!accepted_)out.reason="No accepted alignment is saved for this Kinect. Run Align to VR.";
        else if(!finiteRigid(out.transform) || !std::isfinite(out.spread) || out.spread<.07 ||
                !std::isfinite(out.rms) || out.rms<0 || out.rms>=calibrationMaxRms)
            out.reason="The saved alignment failed its quality checks. Run Align to VR.";
        else if(!out.rawReferenceValid || !finiteRigid(out.standingToRaw))
            out.reason="This older alignment has no saved room reference. Run Align to VR once to save a reusable alignment.";
        else if(!vr.rawTransformValid || !finiteRigid(vr.standingToRaw))
            out.reason="Saved alignment is retained. Wake SteamVR and let tracking settle, then confirm again.";
        else if(!out.referenceSource || out.referenceSource!=vr.referenceSource)
            out.reason="The saved alignment belongs to a different or unknown VR tracking source. Use the original headset connection or run Align to VR.";
        else {
            // Explicit restoration assumes the camera and physical room still
            // match. Never replace the original standing-to-raw matrix: doing
            // so applies the old standing coordinates to a different origin.
            const bool restored=out.rawEpoch!=vr.epoch || out.referenceUniverse!=vr.referenceUniverse;
            out.rawEpoch=vr.epoch;out.referenceUniverse=vr.referenceUniverse;
            out.valid=true;
            out.reason=restored?"Saved alignment restored with its original room reference. Check tracker placement; if the Kinect or room setup changed, run Align to VR.":
                "Saved alignment restored; original room reference and playspace movement preserved.";
        }
        return out;
    }
};
class StableFloor {
    std::deque<std::pair<double,Plane>> samples_;
public:
    void add(double host,const Plane&);
    Plane value()const;
};
Rigid floorAligned(Rigid,const Plane&);
Calibration calibrateWithFloor(std::span<const Pair3>,const Plane&);
struct AlignmentObservation {
    double host{};
    int device{};
    V3 camera;
    DevicePose pose;
    V3 offset;
    Pair3 pair() const { return {camera, pose.p + pose.q.rotate(offset)}; }
};
struct AlignmentProgress {
    int percent{},secondsRemaining{int(calibrationHoldSeconds)};
    bool ready{};
};
AlignmentProgress alignmentProgress(std::span<const AlignmentObservation>,double time);
class AlignmentSession {
    bool automaticOffsets_{};
    bool controllersOnly_{};
    std::array<V3,3> fittedOffsets_{};
    std::array<std::deque<AlignmentObservation>, 3> windows_;
    std::array<double, 3> lastAccepted_{};
    std::vector<AlignmentObservation> samples_;
    std::array<std::string, 3> status_;
    StableFloor floor_;

  public:
    void reset(bool automaticOffsets=false,bool controllersOnly=false);
    void pause(){windows_={};}
    void add(const Frame &, uint32_t id, const Settings &, const PosePrior *prior=nullptr,const Calibration* reference=nullptr);
    Calibration finish();
    const auto &offsets() const { return fittedOffsets_; }
    std::string feedback() const;
    std::string agreement(const Calibration &) const;
    void writeCsv(std::ostream &, const Calibration &,bool header=true) const;
    size_t size() const { return samples_.size(); }
    const auto& samples()const{return samples_;}
    void applyOffsets(const std::array<V3,3>& offsets){fittedOffsets_=offsets;for(auto& s:samples_)s.offset=offsets[s.device];}
    int deviceSamples(int device) const {
        int n = 0;
        for (const auto& s : samples_) if (s.device == device) ++n;
        return n;
    }
    std::string deviceStatus(int device) const {
        return (device >= 0 && device < 3) ? status_[device] : "";
    }
};
Calibration calibrateControllers(std::span<const AlignmentObservation>, std::array<V3,3> &offsets,const Plane* floor=nullptr);
struct AlignmentCue {
    int step{}, seconds{}, retrySeconds{};
    int progressPercent{};
    bool collecting{},waitingForReady{};
    std::string instruction, speech;
};
std::string alignmentPoseTitle(int step);
std::string alignmentPoseInstruction(int step);
AlignmentCue alignmentCue(int step,double elapsed,bool done=false,bool waiting=false);
Calibration calibrateKnownOffsets(std::span<const AlignmentObservation> fit,
                                  std::span<const AlignmentObservation> validation={},const Plane* floor=nullptr);
class GuidedAlignment {
    std::array<AlignmentSession,5> sessions_;
    int stage_{};
    double stageStart_{};
    std::optional<double> captureStart_;
    std::array<bool,2> triggerReleased_{};
    std::string retryReason_;
    bool done_{};
    Calibration result_;
    Calibration candidate_;
    Calibration reference_;
    std::array<V3,3> offsets_{};
    StableFloor floor_;
public:
    void reset(double start,const VrSample* vr=nullptr);
    void capturePose(double time);
    void add(const Frame&,uint32_t,const Settings&,const PosePrior* prior=nullptr);
    AlignmentCue cue(double time)const;
    bool done()const{return done_;}
    int stage()const{return stage_;}
    int deviceSamples(int device)const{return stage_<5?sessions_[stage_].deviceSamples(device):0;}
    std::string deviceStatus(int device)const{return stage_<5?sessions_[stage_].deviceStatus(device):"";}
    const Calibration& result()const{return result_;}
    const auto& offsets()const{return offsets_;}
    std::string agreement()const{return sessions_[4].agreement(result_);}
    std::string feedback()const;
    bool retrying()const{return !retryReason_.empty() && !captureStart_;}
    std::string retryReason()const{return retryReason_;}
    size_t size()const;
    void writeCsv(std::ostream&)const;
};
} // namespace kf
