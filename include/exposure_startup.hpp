#pragma once
namespace kf {
// Commands are attempted only after a live frame, never during sensor discovery.
// A finite retry budget avoids ongoing control work in the tracking loop.
struct ExposureStartup {
    static constexpr unsigned maxAttempts=3;
    bool prefer30{false}, complete{};
    unsigned attempts{};
    double nextAttempt{};
    void reset(bool prefer) { *this={};prefer30=prefer; }
    bool due(double time) const {return !complete && attempts<maxAttempts && time>=nextAttempt;}
    void begin() {++attempts;}
    void succeeded() {complete=true;}
    void failed(double time) {nextAttempt=time+1.0;}
    bool exhausted() const {return attempts>=maxAttempts;}
};
}
