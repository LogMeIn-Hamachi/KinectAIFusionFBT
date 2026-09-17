#pragma once
#include "io.hpp"
#include "model.hpp"
#include "alignment.hpp"
#include "tilt.hpp"
#include "tracking_health.hpp"
#include <thread>
namespace kf {
struct View {
    std::shared_ptr<const Frame> frame;
    State state;
    std::optional<Keypoints> samOverlay;
    std::string poseSource;
    int modelChoice{}; // 0 SAM original, 1 NLF-S, 2 SAM FP8, 3 SAM optimized; stopped only.
    int cadenceChoice{}; // 0 Auto (worker-budget adaptive), 1 Full (30 Hz), 2 Balanced (20 Hz), 3 Low GPU (15 Hz)
    std::string cadenceStatus{"Auto (30 Hz)"};
    bool prefer30{false};
    std::string exposureStatus;
    Calibration calibration;
    Settings settings;
    std::array<double, bones.size()> lengths{};
    std::string modelHash;
    std::string sensor = "Stopped", inference = "Not loaded", vr = "SteamVR not connected",
                notice = "Start the sensor or open a local recording.";
    TrackingStats health;
    std::array<uint64_t,3> outputValidityLosses{};
    double inferenceMs{}, queueMs{}, arrivalToEstimateMs{};
    std::uint64_t frames{}, sent{}, sendErrors{}, dropped{}, recordDrops{};
    bool running{}, recording{}, replay{}, output{}, collecting{};
    unsigned calibrationSamples{};
    double calibrationSecondsRemaining{};
    std::string calibrationPrompt, calibrationDetail;
    std::string calibrationSpeech, bodyPrompt, bodySpeech;
    int calibrationStep{};
    int calibrationLeftSamples{};
    int calibrationRightSamples{};
    std::string calibrationLeftStatus;
    std::string calibrationRightStatus;
    bool bodyCollecting{};
    bool wristOffsetsReady{};
    bool calibrationWaiting{};
    bool calibrationRetrying{};
    std::string calibrationRetryReason;
    bool steamVrOutput{true};
    std::string outputStatus;
    double bodySecondsRemaining{};
    std::optional<int> tiltAngle;
    bool tiltPending{};
    double tiltWait{};
};
class VrOverlay;
class Engine {
    std::filesystem::path root_;
    std::filesystem::path calibrationFile_;
    mutable std::mutex mutex_;
    View view_;
    BoundedQueue<std::shared_ptr<Frame>> measurements_{2}, records_{4};
    std::thread capture_, process_, output_, recorder_;
    std::atomic_bool run_{}, record_{};
    std::atomic_uint32_t selection_{};
    std::atomic_bool calibrateBody_{};
    std::filesystem::path recordPath_;
    std::string metadata_;
    std::uint64_t recordGeneration_{};
    GuidedAlignment alignment_;
    double bodyCaptureStart_{};
    bool calibrationAccepted_{};
    bool useGpu_{true};
    TiltLimiter tiltLimiter_;
    std::optional<int> tiltTarget_;
    std::unique_ptr<VrOverlay> overlay_;
    double lastCalibrationActive_{0};
    void captureLoop(std::filesystem::path replay);
    void processLoop();
    void outputLoop();
    void recordLoop();
    void message(const std::string &);

  public:
    explicit Engine(std::filesystem::path root);
    ~Engine();
    View view() const;
    void start(const std::filesystem::path &replay = {});
    void stop();
    void select(uint32_t id);
    void toggleRecord();
    void output(bool enabled);
    void chooseOutput(bool steamVr);
    void beginCalibration();
    void captureAlignmentPose();
    void cancelCalibration();
    void tilt(int direction);
    void useSavedCalibration();
    void bodyCalibration();
    void settings(Settings);
    void chooseModel(int);
    void chooseCadence(int);
    void chooseExposure(bool prefer30);
    void exportDiagnostics();
    std::filesystem::path root() const { return root_; }
};
std::optional<Crop> playerCrop(const Frame &, uint32_t id);
} // namespace kf
