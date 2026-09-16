#pragma once
#include <memory>
#include <string>

namespace kf {

struct OverlayState {
    bool active{false};
    int step{0};               // 0 to 4
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
};

class VrOverlay {
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
