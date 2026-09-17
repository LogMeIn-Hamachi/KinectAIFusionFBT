#pragma once
#include <memory>
#include <string>
#include <optional>
#include <mutex>

namespace kf {

struct OverlayState {
    bool active{false};
    int step{0};               // 0 to 4
    int retrySeconds{};
    int secondsRemaining{0};   // Settle or capture seconds
    bool waitingForReady{false};
    bool collecting{false};    // true = holding still during capture
    bool done{false};          // true = finished all poses
    bool success{false};       // true = validation passed
    bool isRetry{false};       // true = pose validation/tracking failed, waiting for user to squeeze trigger again
    std::string retryReason;   // specific reason for retry
    std::string instruction;   // Current pose instruction
    std::string feedback;      // Error / detail feedback
    std::string agreement;     // Final agreement stats
    int leftSamples{0};        // Samples recorded for left controller (0..12)
    int rightSamples{0};       // Samples recorded for right controller (0..12)
    std::string leftStatus;    // "collecting steady samples", "Kinect cannot see wrist", etc.
    std::string rightStatus;
    bool leftTrigger{false};
    bool rightTrigger{false};
    bool leftTracked{false};
    bool rightTracked{false};
    bool operator==(const OverlayState&) const = default;
};

// Commit only successfully displayed states. Retry failed uploads even when the
// calibration state is unchanged. Runtime recovery explicitly invalidates it.
class OverlayRefresh {
    std::optional<OverlayState> displayed_;
    double nextAttempt_{};
public:
    bool due(const OverlayState& state,double now) const {
        return now>=nextAttempt_ && (!displayed_ || *displayed_!=state);
    }
    void complete(const OverlayState& state,double now,bool success) {
        nextAttempt_=now+(success?.1:.25);
        if(success)displayed_=state;
        else displayed_.reset();
    }
    void reset(){*this={};}
};

class VrOverlay {
    mutable std::mutex mutex_;
    struct Impl;
    std::unique_ptr<Impl> impl_;

public:
    VrOverlay();
    ~VrOverlay();

    void update(const OverlayState& state);
    void hide();
    bool isVisible() const;
};

} // namespace kf
