#pragma once
#include <mutex>
#include <shared_mutex>
namespace kf {
// Input and overlay may make API calls concurrently. Initialization/teardown
// must wait for every in-flight call, without serializing normal capture/render.
class RuntimeLifetime {
    std::shared_mutex mutex_;
public:
    auto read(){return std::shared_lock(mutex_);}
    auto write(){return std::unique_lock(mutex_);}
};
inline RuntimeLifetime openVrLifetime;
}
