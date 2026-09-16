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
    std::string instruction;   // Current pose instruction
    std::string feedback;      // Error / detail feedback
    std::string agreement;     // Final agreement stats
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
