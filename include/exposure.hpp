#pragma once
#include <memory>
#include <string>
namespace kf {
class KinectExposure {
    struct Impl;
    std::unique_ptr<Impl> p_;
public:
    KinectExposure();
    ~KinectExposure();
    bool open(const std::wstring& sensorId);
    void prioritize30();
    void automatic();
};
}
