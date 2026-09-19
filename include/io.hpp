#pragma once
#include "core.hpp"
#include "exposure_startup.hpp"
#include <fstream>
#include <functional>
namespace kf {
class KinectV2Capture {
    struct Impl;
    std::unique_ptr<Impl> p_;
  public:
    KinectV2Capture();
    ~KinectV2Capture();
    bool open(); // false when no v2 is attached; errors after discovery are reported.
    void prioritize30(bool enabled);
    void close();
    bool healthy() const;
    std::shared_ptr<Frame> poll();
    std::string calibrationKey() const;
    std::uint64_t dropped{};
};
class KinectCapture {
    struct Impl;
    std::unique_ptr<Impl> p_;
    std::unique_ptr<KinectV2Capture> v2_;
    ExposureStartup exposureStartup_;

  public:
    KinectCapture();
    ~KinectCapture();
    KinectCapture(const KinectCapture &) = delete;
    void open(bool prefer30 = false);
    std::shared_ptr<Frame> poll();
    bool healthy() const;
    std::string calibrationKey() const;
    std::optional<int> elevation() const;
    void elevation(int degrees);
    void close();
    std::string status;
    std::string exposureStatus;
    std::uint64_t dropped{};
};
class VrInput {
    struct Impl;
    std::unique_ptr<Impl> p_;

  public:
    VrInput();
    ~VrInput();
    VrSample poll();
    std::string status;
    std::uint64_t epoch() const;
};
class OscOutput {
    std::uintptr_t socket_{~std::uintptr_t(0)};
    bool initialized_{};

  public:
    OscOutput();
    ~OscOutput();
    bool send(std::span<const std::uint8_t> bytes);
};
class RecordingWriter {
    std::ofstream stream_;

  public:
    void open(const std::filesystem::path &, const std::string &metadata);
    void write(const Frame &);
    void close();
    bool active() const { return stream_.is_open(); }
};
class RecordingReader {
    std::ifstream stream_;
    int version_{1};

  public:
    std::string metadata;
    void open(const std::filesystem::path &);
    std::shared_ptr<Frame> next();
    void close() { stream_.close(); }
};
void saveCalibration(const std::filesystem::path &, const Calibration &, const Settings &,bool learnedOffsets=false);
std::optional<std::string> trySaveCalibration(const std::filesystem::path &, const Calibration &, const Settings &,bool learnedOffsets=false);
bool loadCalibration(const std::filesystem::path &, Calibration &, Settings &,bool* learnedOffsets=nullptr);
void saveBgraBmp(const std::filesystem::path &, const Frame &);
} // namespace kf
