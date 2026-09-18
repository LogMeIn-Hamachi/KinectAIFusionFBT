#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace kf {
constexpr double pi = 3.14159265358979323846;
inline double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct V2 {
    double x{}, y{};
};
struct V3 {
    double x{}, y{}, z{};
    V3 operator+(V3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    V3 operator-(V3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    V3 operator-() const { return {-x, -y, -z}; }
    V3 operator*(double s) const { return {x * s, y * s, z * s}; }
    V3 operator/(double s) const { return *this * (1 / s); }
    V3 &operator+=(V3 b) {
        *this = *this + b;
        return *this;
    }
    V3 &operator-=(V3 b) {
        *this = *this - b;
        return *this;
    }
    V3 &operator*=(double s) {
        x *= s; y *= s; z *= s;
        return *this;
    }
};
inline double dot(V3 a, V3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(V3 a) {
    return std::sqrt(dot(a, a));
}
inline bool finite(V3 a) {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}
inline V3 unit(V3 a, V3 fallback = {0, 1, 0}) {
    double n = norm(a);
    return n > 1e-9 && std::isfinite(n) ? a / n : fallback;
}
inline V3 bounded(V3 a, double n) {
    double l = norm(a);
    return l > n ? a * (n / l) : a;
}
inline V3 lerp(V3 a, V3 b, double t) {
    return a * (1 - t) + b * t;
}
struct Q {
    double w{1}, x{}, y{}, z{}; // scalar first internally; never memcpy to an external API
    Q operator-() const { return {-w, -x, -y, -z}; }
    Q operator*(Q b) const {
        return {w * b.w - x * b.x - y * b.y - z * b.z, w * b.x + x * b.w + y * b.z - z * b.y,
                w * b.y - x * b.z + y * b.w + z * b.x, w * b.z + x * b.y - y * b.x + z * b.w};
    }
    Q conjugate() const { return {w, -x, -y, -z}; }
    V3 rotate(V3 v) const {
        Q p = *this * Q{0, v.x, v.y, v.z} * conjugate();
        return {p.x, p.y, p.z};
    }
};
inline double dot(Q a, Q b) {
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Q normalized(Q q) {
    double n = std::sqrt(dot(q, q));
    return n > 1e-9 && std::isfinite(n) ? Q{q.w / n, q.x / n, q.y / n, q.z / n} : Q{};
}
inline Q continuous(Q q, Q previous) {
    q = normalized(q);
    return dot(q, previous) < 0 ? -q : q;
}
inline Q blend(Q a, Q b, double t) {
    b = continuous(b, a);
    return normalized(
        {a.w * (1 - t) + b.w * t, a.x * (1 - t) + b.x * t, a.y * (1 - t) + b.y * t, a.z * (1 - t) + b.z * t});
}
inline double angleBetween(Q a, Q b) {
    return 2.0 * std::acos(std::clamp(std::abs(dot(a, b)), 0.0, 1.0));
}
inline Q axisAngle(V3 axis, double angle) {
    axis = unit(axis);
    double s = std::sin(angle / 2);
    return {std::cos(angle / 2), axis.x * s, axis.y * s, axis.z * s};
}
struct M3 {
    double a[3][3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    V3 apply(V3 p) const {
        return {a[0][0] * p.x + a[0][1] * p.y + a[0][2] * p.z, a[1][0] * p.x + a[1][1] * p.y + a[1][2] * p.z,
                a[2][0] * p.x + a[2][1] * p.y + a[2][2] * p.z};
    }
};
M3 matrix(Q q);
Q quaternion(M3 m);
Q basis(V3 right, V3 up);
Q reflectZ(Q q);
inline V3 reflectZ(V3 v) {
    return {v.x, v.y, -v.z};
}
V3 eulerZXY(Q q);
Q fromEulerZXY(V3 degrees);
struct Rigid {
    Q q{};
    V3 t{};
    V3 apply(V3 v) const { return q.rotate(v) + t; }
    V3 inverse(V3 v) const { return q.conjugate().rotate(v - t); }
};
struct Plane {
    V3 n{0, 1, 0};
    double d{};
    bool valid{};
    double height(V3 v) const { return dot(n, v) + d; }
};
struct Pair3 {
    V3 camera, world;
};
inline constexpr double calibrationSettleSeconds = 3, calibrationHoldSeconds = 3;
inline constexpr int calibrationPoseCount = 5;
inline constexpr double calibrationMaxRms = .060;
inline constexpr double calibrationMinInlierFraction = .70;
inline constexpr double calibrationInlierDistance = .10;
struct Calibration {
    Rigid transform;
    double rms{}, p95{}, spread{};
    bool valid{};
    std::string reason;
    // Live reference captured at alignment/confirmation, never serialized into
    // old recordings or reused across a SteamVR restart without confirmation.
    Rigid standingToRaw{};
    std::uint64_t rawEpoch{};
    bool rawReferenceValid{};
};
Calibration calibrate(std::span<const Pair3> pairs);
Plane fitFloor(std::span<const V3> points);

struct ClockMap {
    bool ready{};
    std::int64_t last{};
    double offset{}, lastHost{};
    unsigned resets{};
    double map(std::int64_t milliseconds, double arrival);
};
template <class T> class BoundedQueue {
    std::deque<T> q_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    size_t capacity_;
    bool closed_{};

  public:
    std::atomic_uint64_t dropped{};
    explicit BoundedQueue(size_t n) : capacity_(std::max(size_t(1), n)) {}
    void push(T value) {
        std::lock_guard l(m_);
        if (closed_)
            return;
        while (q_.size() >= capacity_) {
            q_.pop_front();
            ++dropped;
        }
        q_.push_back(std::move(value));
        cv_.notify_one();
    }
    std::optional<T> pop(bool latest = false) {
        std::unique_lock l(m_);
        cv_.wait_for(l, std::chrono::milliseconds(30), [&] { return closed_ || !q_.empty(); });
        if (q_.empty())
            return {};
        if (latest)
            while (q_.size() > 1) {
                q_.pop_front();
                ++dropped;
            }
        T v = std::move(q_.front());
        q_.pop_front();
        return v;
    }
    void close() {
        std::lock_guard l(m_);
        closed_ = true;
        cv_.notify_all();
    }
    void reset() {
        std::lock_guard l(m_);
        q_.clear();
        closed_ = false;
        dropped = 0;
    }
};
template <class A, class B> class Pairer {
    std::deque<A> a_;
    std::deque<B> b_;

  public:
    std::uint64_t dropped{};
    void addA(A a) {
        a_.push_back(std::move(a));
        if (a_.size() > 4) {
            a_.pop_front();
            ++dropped;
        }
    }
    void addB(B b) {
        b_.push_back(std::move(b));
        if (b_.size() > 4) {
            b_.pop_front();
            ++dropped;
        }
    }
    std::optional<std::pair<A, B>> next() {
        while (!a_.empty() && !b_.empty()) {
            auto dt = a_.front().stamp - b_.front().stamp;
            if (std::abs(dt) <= 18) {
                auto p = std::make_pair(std::move(a_.front()), std::move(b_.front()));
                a_.pop_front();
                b_.pop_front();
                return p;
            }
            if (dt < 0)
                a_.pop_front();
            else
                b_.pop_front();
            ++dropped;
        }
        return {};
    }
    void clear() {
        a_.clear();
        b_.clear();
    }
};
constexpr int J = 26;
enum JointId {
    Nose = 0,
    LShoulder = 5,
    RShoulder = 6,
    LElbow = 7,
    RElbow = 8,
    LWrist = 9,
    RWrist = 10,
    LHip = 11,
    RHip = 12,
    LKnee = 13,
    RKnee = 14,
    LAnkle = 15,
    RAnkle = 16,
    Head = 17,
    Neck = 18,
    Hip = 19,
    LToe = 20,
    RToe = 21,
    LSmallToe = 22,
    RSmallToe = 23,
    LHeel = 24,
    RHeel = 25
};
inline constexpr std::array<std::pair<int, int>, 19> bones{{{Head, Neck},
                                                            {Neck, Hip},
                                                            {Neck, LShoulder},
                                                            {Neck, RShoulder},
                                                            {LShoulder, LElbow},
                                                            {LElbow, LWrist},
                                                            {RShoulder, RElbow},
                                                            {RElbow, RWrist},
                                                            {Hip, LHip},
                                                            {Hip, RHip},
                                                            {LHip, LKnee},
                                                            {LKnee, LAnkle},
                                                            {RHip, RKnee},
                                                            {RKnee, RAnkle},
                                                            {LAnkle, LHeel},
                                                            {LHeel, LToe},
                                                            {RAnkle, RHeel},
                                                            {RHeel, RToe},
                                                            {LHip, RHip}}};
struct Joint {
    V3 p{};
    double confidence{}, sigma{1};
    std::uint8_t source{};
}; // 1 SDK, 2 RGB-D, 3 VR, 4 temporal prediction, 5 depth-anchored learned pose
struct Body {
    std::uint32_t id{};
    std::uint8_t player{};
    std::array<Joint, J> joints{};
};
struct PixelMap {
    float x{}, y{}, z{};
    std::int32_t u{-1}, v{-1};
};
struct DevicePose {
    V3 p{};
    Q q{};
    bool valid{};
};
struct VrSample {
    double host{};
    std::array<DevicePose, 3> devices{};
    std::uint64_t epoch{};
    Rigid standingToRaw{}; // live-only; recording format remains unchanged.
    bool rawTransformValid{};
    // Live-only calibration controls; never stored in recordings.
    std::array<bool,2> triggerPressed{},triggerAvailable{};
};
inline bool finiteRigid(const Rigid& r) {
    return finite(r.t) && std::isfinite(dot(r.q,r.q)) && std::abs(dot(r.q,r.q)-1)<.001;
}
inline Rigid composeRigid(const Rigid& a,const Rigid& b) {
    return {normalized(a.q*b.q),a.apply(b.t)};
}
inline Rigid inverseRigid(const Rigid& a) {
    return {a.q.conjugate(),a.q.conjugate().rotate(-a.t)};
}
inline bool bindTrackingReference(Calibration& cal,const VrSample& vr) {
    if(!vr.rawTransformValid || !finiteRigid(vr.standingToRaw))return false;
    cal.standingToRaw=vr.standingToRaw;cal.rawEpoch=vr.epoch;cal.rawReferenceValid=true;return true;
}
inline bool trackingReferenceValid(const Calibration& cal,const VrSample& vr) {
    return cal.rawReferenceValid && vr.rawTransformValid && cal.rawEpoch==vr.epoch &&
        finiteRigid(cal.standingToRaw) && finiteRigid(vr.standingToRaw);
}
// Incoming VR devices use the same physical reference as the camera. A virtual
// space drag must not become an apparent controller/body measurement error.
inline VrSample calibrationVr(const VrSample& vr,const Calibration& cal) {
    auto out=vr;
    if(!cal.rawReferenceValid)return out; // legacy replay / non-live unit inputs
    if(!trackingReferenceValid(cal,vr)){for(auto& d:out.devices)d.valid=false;return out;}
    const auto toReference=composeRigid(inverseRigid(cal.standingToRaw),vr.standingToRaw);
    for(auto& d:out.devices)if(d.valid){d.p=toReference.apply(d.p);d.q=normalized(toReference.q*d.q);}
    return out;
}
inline Rigid currentStandingCalibration(const Calibration& cal,const VrSample& vr) {
    return composeRigid(inverseRigid(vr.standingToRaw),composeRigid(cal.standingToRaw,cal.transform));
}
class PoseHistory {
    std::deque<VrSample> poses_;

  public:
    void add(VrSample p);
    std::optional<VrSample> at(double host) const;
    void clear() { poses_.clear(); }
};
struct ReplayConfig;
struct ColorProjection;
struct Frame {
    std::int64_t rgbStamp{}, depthStamp{}, skeletonStamp{};
    std::uint32_t rgbId{}, depthId{}, skeletonId{}, epoch{};
    double host{}, arrival{};
    int width{640}, height{480};
    int depthWidth{640}, depthHeight{480}, sensorVersion{1};
    double captureMs{}; // host capture/conversion/mapping work, excludes waiting.
    double exposureMs{}, colorIntervalMs{};
    std::shared_ptr<const ColorProjection> colorProjection; // derived, not serialized.
    std::vector<std::uint8_t> bgra;
    std::vector<std::uint16_t> depth;     // millimetres << 3 | player 1..6; v2 repacked losslessly <=8191mm.
    std::vector<PixelMap> mapping;        // exact SDK depth-to-color and metric XYZ
    std::vector<std::int32_t> colorIndex; // derived z-buffer; not serialized
    std::vector<std::uint8_t> calibrationBlob;
    std::vector<std::uint8_t> sdkSkeletonBlob; // v1 raw frame; v2 tagged full 25-joint data and 64-bit native IDs.
    std::vector<Body> bodies;
    Plane floor;
    VrSample vr;
    std::shared_ptr<const ReplayConfig> runConfig;
};
struct Crop {
    double cx{}, cy{}, w{}, h{};
    V2 toImage(V2 p, int iw = 192, int ih = 256) const {
        return {p.x / iw * w + cx - w / 2, p.y / ih * h + cy - h / 2};
    }
    V2 toInput(V2 p, int iw = 192, int ih = 256) const {
        return {(p.x - cx + w / 2) / w * iw, (p.y - cy + h / 2) / h * ih};
    }
};
Crop cropBox(double x1, double y1, double x2, double y2);
std::optional<Crop> playerCrop(const Frame &, std::uint32_t id);
struct Keypoint {
    V2 uv{};
    double score{};
};
using Keypoints = std::array<Keypoint, J>;
struct DepthSupport {
    unsigned samples{}, playerSamples{};
    double spread{}, priorDelta{};
    int rejection{}; // 0 accepted, 1 keypoint, 2 support, 3 depth edge, 4 prior, 5 inliers
};
Keypoints unmirror(Keypoints k, int width);
Keypoints kinectImageLabels(Keypoints k);
std::optional<Joint> associateDepth(const Frame &frame, Keypoint kp, const Joint &prior, int joint,
                                    std::uint8_t player, double surfaceRadius = .045,
                                    DepthSupport *support = nullptr);
void indexRegistration(Frame &frame);
// Locally fitted color projection from SDK-registered 3D/2D correspondences.
// Valid only when held-out reprojection residual and conditioning checks pass.
struct ColorProjection {
    std::array<double, 11> p{};
    int width{640}, height{480};
    double rms{};
    bool valid{};
    V2 project(V3 point) const;
    V3 correction(V3 point, V2 target) const;
};
ColorProjection fitColorProjection(const Frame &frame);
bool stabilizeLegLabels(Body &observations, const Body &predicted);
enum class Contact { Air, Candidate, Planted, Sliding, Pivot };
struct FootContact {
    Contact state{Contact::Air};
    double dwell{};
    V3 anchor{};
    void update(double dt, double soleHeight, V3 velocity, bool visible, double tilt, V3 sole);
};
struct Tracker {
    V3 p{}, velocity{};
    Q q{};
    V3 angularVelocity{};
    V3 angularSigma{pi, pi, pi};
    double positionSigma{1};
    bool valid{};
    double observedHost{}; // per-tracker freshness, not serialized.
};
// Stable wire order: retain the original waist/feet IDs in every layout.
inline constexpr int trackerCount=8;
inline constexpr uint32_t baseTrackerMask=7, allTrackerMask=255;
inline constexpr const char* trackerNames[]{"Hips","Left foot","Right foot","Left knee","Right knee","Left elbow","Right elbow","Chest"};
inline constexpr uint32_t trackerMask(int extras) {
    return baseTrackerMask | ((extras&1)?24u:0u) | ((extras&2)?96u:0u) | ((extras&4)?128u:0u);
}
inline constexpr bool trackerEnabled(uint32_t mask,int role) {return role>=0 && role<trackerCount && (mask&(1u<<role));}
enum class Mode { Lost, RawSdk, FilteredSdk, Fused, Predicted, Degraded };
struct State {
    Body body;
    std::array<DepthSupport, J> depthSupport{};
    std::array<Tracker, trackerCount> trackers;
    uint32_t trackerMask{baseTrackerMask};
    std::array<FootContact, 2> contacts;
    Mode mode{Mode::Lost};
    double host{}, lastObserved{}, fitMs{};
    bool ambiguous{};
    std::array<bool,trackerCount> learnedDirection{};
    std::array<bool,trackerCount> learnedPosition{};
    std::array<bool,2> nativeFootDirection{};
};
// The body root is registered to selected-person depth. Articulated joint
// locations remain model estimates, including joints hidden from the camera.
struct PosePrior {
    std::array<V3,J> points{};
    std::array<bool,J> supported{};
    std::array<bool,J> available{};
    double host{}, registrationRms{};
    unsigned rootAnchors{};
    std::uint32_t selectedId{};
    bool valid{};
    // Articulation may be usable even when this frame lacks metric depth anchors.
    bool articulationValid{}, depthPredicted{};
    double depthAge{};
    V3 cameraOrigin{}, cameraForward{0,0,1};
    V3 imageRoot{};
    std::array<V3,9> depthRoots{};
    unsigned depthRootCount{};
    // Selected-person ankle surfaces in camera space, before shared body-root
    // registration/VR correction. Only used as independent foot motion evidence.
    std::array<V3,2> footSurfaces{};
    std::array<bool,2> footSurfaceValid{};
    bool bodyFitted{}, vrAnchored{};
    double rootSigma{};
    std::array<Q,2> footOrientations{};
    std::array<bool,2> footOrientationValid{};
    std::array<bool,J> resting{};
};
struct Settings {
    bool inference{true}, depth{true}, constraints{true}, contacts{true}, vrConstraints{true};
    int baseline{};
    int extraTrackers{}; // Output preference: bit 0 knees, bit 1 elbows, bit 2 chest.
    double soleOffset{0.075}, surfaceRadius{0.045};
    std::array<V3, 3> deviceOffsets{{{0, -0.10, 0.08}, {0, 0, 0}, {0, 0, 0}}};
    // Hip offset is from pelvis; feet are mounted above the estimated sole,
    // forward of the ankle, rather than 3.5 cm up the shin.
    std::array<V3, 3> trackerOffsets{{{0, 0, -0.08}, {0, 0.035, -0.06}, {0, 0.035, -0.06}}};
};
struct ReplayConfig {
    Settings settings;
    Calibration calibration;
    std::uint32_t selectedId{};
    std::array<double, bones.size()> lengths{};
    std::string modelHash;
};
// Confirm discontinuous observations, then allow recovery instead of rejecting forever.
struct RotationEvidence {
    Q value{}, reference{}, pending{};
    Q motionReference{};
    V3 motionTrend{};
    double motionActivity{};
    bool motionInitialized{};
    double since{}, lastHost{};
    bool initialized{}, trusted{}, waiting{};
    bool update(Q observation, bool visible, double host, double dwell, double jump, double rate, double tau,
                bool adaptive=false);
};
// Causal smoothing: alternating innovations cancel in the signed motion trend.
// A small soft deadband settles noise without locking a foot to the floor.
struct LearnedPositionFilter {
    V3 value{}, previousObservation{}, trend{};
    double activity{}, host{};
    bool initialized{}, resting{true};
    V3 update(V3 observation,double time,double restTau=.14);
};
struct SoleCorrection {
    double value{};
    double update(double target,double dt) {
        target=std::clamp(target,-.015,.015);
        double step=(target-value)*(1-std::exp(-dt/.08));
        value+=std::clamp(step,-.06*dt,.06*dt);
        return value;
    }
};
struct SourceTransition {
    V3 offset{};
    double since{};
    uint8_t source{};
    V3 update(V3 target,V3 previous,uint8_t next,double host) {
        if(next!=1 && next!=2 && next!=5)return target;
        if(source && next!=source && (source==5 || next==5)) {
            offset=previous-target;since=host;
        }
        source=next;
        const double age=host-since;
        if(age<0 || age>=.24){offset={};return target;}
        // Exact continuity on the switch, zero residual offset after 240 ms.
        const double x=std::clamp(age/.24,0.,1.);
        return target+offset*(1-x*x*(3-2*x));
    }
};
struct PlantStabilizer {
    V3 anchor{}, correction{};
    V3 surfaceAnchor{};
    Q direction{};
    double dwell{},unseenAge{},surfaceMotionAge{};
    bool planted{},surfaceReady{},surfaceAnchored{};
    V3 update(V3 sole,Q rotation,V3 velocity,const Plane& floor,bool observed,double dt,
              std::optional<V3> surface={}) {
        if(!floor.valid){*this={};return {};}
        if(surface && !finite(*surface))surface.reset();
        bool surfaceOffFloor=surface && (floor.height(*surface)<-.02 || floor.height(*surface)>.22);
        double height=floor.height(sole),vertical=dot(velocity,floor.n);
        unseenAge=(surface || (!surfaceAnchored && observed))?0:unseenAge+dt;
        double horizontal=norm(velocity-floor.n*vertical);
        double angle=2*std::acos(std::clamp(std::abs(dot(normalized(rotation),normalized(direction))),0.,1.));
        bool surfaceMoved=false;
        if(surface) {
            if(!surfaceReady){surfaceAnchor=*surface;surfaceReady=true;dwell=0;}
            const auto delta=*surface-surfaceAnchor;
            const double rise=dot(delta,floor.n),slide=norm(delta-floor.n*rise);
            // A single small noisy patch must not release a plant. Clear lifts
            // and large slides release immediately; smaller slides need 60 ms.
            surfaceMotionAge=slide>.025?surfaceMotionAge+dt:0;
            surfaceMoved=std::abs(rise)>.03 || slide>.06 || (surfaceMotionAge>=.06 && surfaceMotionAge>dt*1.5);
            if(!planted && norm(delta)>.018){surfaceAnchor=*surface;dwell=0;}
        }else {surfaceMotionAge=0;if(!planted)surfaceReady=false;}
        bool independent=planted && surfaceAnchored;
        bool release=surfaceOffFloor || unseenAge>.20 || std::abs(height)>.055 || std::abs(vertical)>.18 ||
            (independent?surfaceMoved:horizontal>.16) ||
            (planted && (std::abs(dot(sole-anchor,floor.n))>.045 ||
                         norm(sole-anchor)>(independent?.15:.07) || angle>.22));
        if(release){planted=false;dwell=0;surfaceAnchored=false;surfaceMotionAge=0;if(surface)surfaceAnchor=*surface;}
        else if(!planted) {
            if((surface || observed) && std::abs(height)<.03 && horizontal<.08 && std::abs(vertical)<.08)dwell+=dt;
            else dwell=0;
            if(dwell>=.18){planted=true;anchor=sole-floor.n*height;direction=rotation;surfaceAnchored=bool(surface);surfaceMotionAge=0;}
        }
        V3 target{};
        if(planted){target=bounded(anchor-sole,surfaceAnchored?.15:.05);}
        correction=lerp(correction,target,1-std::exp(-dt/(planted?.045:.035)));
        // Keep the planted horizontal position exact. Smoothing a correction
        // behind a moving body root itself makes a stationary foot skate.
        if(planted && surfaceAnchored)correction=target-floor.n*dot(target-correction,floor.n);
        return correction;
    }
};
// Velocity for presentation only: measure the final corrected tracker path.
// Preserve coherent slow motion; reject alternating noise without feeding a
// magnitude deadband back into the next velocity estimate.
struct PresentationVelocity {
    V3 previous{},trend{};
    double activity{},host{};
    bool initialized{};
    V3 update(V3 position,double time,bool valid) {
        if(!valid || !finite(position)){*this={};return {};}
        const double dt=time-host;
        if(!initialized || dt<=0 || dt>.15 || norm(position-previous)>.35) {
            *this={};previous=position;host=time;initialized=true;return {};
        }
        const V3 derivative=bounded((position-previous)/dt,4);
        const double alpha=1-std::exp(-dt/.07);
        trend=lerp(trend,derivative,alpha);
        activity+=(norm(derivative)-activity)*alpha;
        previous=position;host=time;
        const double coherence=norm(trend)/std::max(activity,1e-9);
        return trend*std::clamp((coherence-.50)/.35,0.,1.);
    }
};
class Estimator {
    bool smoothSourceChanges_{true};
    State state_;
    std::array<V3, J> velocity_{};
    std::array<PresentationVelocity,trackerCount> presentationVelocity_{};
    std::array<double, J> seen_{};
    std::array<double, bones.size()> lengths_{};
    std::uint32_t selected_{};
    unsigned initialized_{};
    std::array<RotationEvidence, trackerCount> rotations_{};
    std::array<V3, trackerCount> angularVelocity_{};
    std::array<Q, trackerCount> previousRotation_{};
    std::array<LearnedPositionFilter,J> learnedFilters_{};
    std::array<SourceTransition,J> sourceTransitions_{};
    std::array<SoleCorrection,2> soleCorrections_{};
    std::array<double,J> samSeen_{};
    std::array<PlantStabilizer,2> plants_{};
    double labelSwapSince_{}, rebindSince_{}, rebindLast_{};
    std::uint32_t rebindCandidate_{};

  public:
    explicit Estimator(bool smoothSourceChanges=true):smoothSourceChanges_(smoothSourceChanges){}
    Settings settings;
    void select(std::uint32_t id);
    // Returns the locked ID; changes it only after sustained, unique three-device agreement.
    std::uint32_t reconcileIdentity(const Frame &, const Calibration &);
    const State &state() const { return state_; }
    State process(const Frame &, const Keypoints *, const Calibration &, const PosePrior *prior=nullptr);
    State predict(double host) const;
    bool calibrateLengths(const Body &b);
    bool calibrateLengths(std::span<const Body> samples);
    const auto &lengths() const { return lengths_; }
    void restoreLengths(const std::array<double, bones.size()> &values) {
        for (double v : values)
            if (!std::isfinite(v) || v < 0 || v > .85)
                throw std::runtime_error("Invalid segment lengths");
        lengths_ = values;
    }
};
std::vector<std::uint8_t> oscBundle(std::span<const Tracker> trackers, const Rigid &cameraToVr);
std::string modeName(Mode mode);
std::string sha256(const std::filesystem::path &path);
} // namespace kf
