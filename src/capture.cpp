#include "io.hpp"
#include <Windows.h>
#include <objbase.h>
#include <dmo.h>
#include <NuiApi.h>
#include <dmo.h>
#include <objbase.h>
#include <sstream>
namespace kf {
struct ImagePart {
    std::int64_t stamp{};
    std::uint32_t id{};
    double arrival{};
    std::vector<std::uint8_t> bytes;
};
struct KinectCapture::Impl {
    HMODULE runtime{};
    INuiSensor *sensor{};
    INuiCoordinateMapper *mapper{};
    HANDLE rgb{}, depth{};
    ClockMap clock;
    Pairer<ImagePart, ImagePart> pairs;
    std::deque<NUI_SKELETON_FRAME> skeletons;
    double lastFrame{};
    std::vector<std::uint8_t> calibration;
    ~Impl() {
        if (mapper)
            mapper->Release();
        if (sensor) {
            sensor->NuiShutdown();
            sensor->Release();
        }
        if (runtime)
            FreeLibrary(runtime);
    }
};
static void checked(HRESULT hr, const char *name) {
    if (FAILED(hr)) {
        std::ostringstream s;
        s << name << " failed (0x" << std::hex << static_cast<unsigned long>(hr)
          << "). Check power, USB bandwidth, SDK 1.8 and whether another app owns the Kinect.";
        throw std::runtime_error(s.str());
    }
}
KinectCapture::KinectCapture() = default;
KinectCapture::~KinectCapture() = default;
void KinectCapture::close() {
    v2_.reset();
    p_.reset();
}
void KinectCapture::open(bool prefer30) {
    close();
    exposureStatus.clear();
    exposureStartup_.reset(prefer30);
    dropped=0;
    auto candidate=std::make_unique<KinectV2Capture>();
    if(candidate->open()) {
        v2_=std::move(candidate);
        status="Kinect v2: RGB 1920x1080 / depth 512x424, nominal 30 Hz";
        exposureStatus="Waiting for live colour frames before setting exposure";
        return;
    }
    p_ = std::make_unique<Impl>();
    p_->runtime = LoadLibraryExW(L"Kinect10.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!p_->runtime)
        throw std::runtime_error(
            "Kinect SDK 1.8 runtime missing. Install the full official SDK for Xbox 360 support.");
    using Count = HRESULT(WINAPI *)(int *);
    using Create = HRESULT(WINAPI *)(int, INuiSensor **);
    auto count = reinterpret_cast<Count>(GetProcAddress(p_->runtime, "NuiGetSensorCount"));
    auto create = reinterpret_cast<Create>(GetProcAddress(p_->runtime, "NuiCreateSensorByIndex"));
    if (!count || !create)
        throw std::runtime_error("Incompatible Kinect runtime");
    int n = 0;
    checked(count(&n), "Enumerate Kinect");
    for (int i = 0; i < n; ++i) {
        INuiSensor *s = nullptr;
        if (SUCCEEDED(create(i, &s)) && s) {
            if (SUCCEEDED(s->NuiStatus())) {
                p_->sensor = s;
                break;
            }
            s->Release();
        }
    }
    if (!p_->sensor)
        throw std::runtime_error(
            "No ready Kinect v1 or v2. Check its powered adapter and USB connection (v2 needs USB 3.0).");
    auto s = p_->sensor;
    checked(s->NuiInitialize(NUI_INITIALIZE_FLAG_USES_COLOR |
                             NUI_INITIALIZE_FLAG_USES_DEPTH_AND_PLAYER_INDEX |
                             NUI_INITIALIZE_FLAG_USES_SKELETON),
            "Initialize Kinect");
    checked(
        s->NuiImageStreamOpen(NUI_IMAGE_TYPE_COLOR, NUI_IMAGE_RESOLUTION_640x480, 0, 2, nullptr, &p_->rgb),
        "Open 640x480 RGB");
    checked(s->NuiImageStreamOpen(NUI_IMAGE_TYPE_DEPTH_AND_PLAYER_INDEX, NUI_IMAGE_RESOLUTION_640x480, 0, 2,
                                  nullptr, &p_->depth),
            "Open 640x480 depth/player");
    checked(s->NuiSkeletonTrackingEnable(nullptr, 0), "Enable skeleton");
    checked(s->NuiGetCoordinateMapper(&p_->mapper), "Coordinate mapper");
    ULONG bytes = 0;
    void *blob = nullptr;
    if (SUCCEEDED(p_->mapper->GetColorToDepthRelationalParameters(&bytes, &blob)) && blob) {
        auto data = static_cast<std::uint8_t *>(blob);
        p_->calibration.assign(data, data + bytes);
        CoTaskMemFree(blob);
    }
    p_->lastFrame = now();
    status = "Kinect v1 connected: RGB/depth 640x480, nominal 30 Hz";
}
bool KinectCapture::healthy() const {
    if(v2_)return v2_->healthy();
    return p_ && p_->sensor && SUCCEEDED(p_->sensor->NuiStatus()) && now() - p_->lastFrame < 3;
}
std::optional<int> KinectCapture::elevation() const {
    if(v2_)return {};
    LONG angle=0;
    if(!p_ || !p_->sensor || FAILED(p_->sensor->NuiCameraElevationGetAngle(&angle)))return {};
    return int(angle);
}
void KinectCapture::elevation(int degrees) {
    if(v2_)throw std::runtime_error("Kinect v2 has a manual tilt mount, no electric tilt motor.");
    if(!p_ || !p_->sensor)throw std::runtime_error("Start the Kinect before adjusting tilt.");
    if(degrees<NUI_CAMERA_ELEVATION_MINIMUM || degrees>NUI_CAMERA_ELEVATION_MAXIMUM)
        throw std::runtime_error("Kinect tilt must be between -27 and +27 degrees.");
    HRESULT hr=p_->sensor->NuiCameraElevationSetAngle(degrees);
    if(hr==HRESULT_FROM_WIN32(ERROR_RETRY) || hr==HRESULT_FROM_WIN32(ERROR_TOO_MANY_CMDS))
        throw std::runtime_error("Kinect motor is cooling down. Wait 20 seconds before trying again.");
    checked(hr,"Adjust Kinect motor");
}
std::shared_ptr<Frame> KinectCapture::poll() {
    if(v2_){
        auto frame=v2_->poll();dropped=v2_->dropped;
        if(frame && exposureStartup_.due(now())) {
            exposureStartup_.begin();
            try {
                v2_->prioritize30(exposureStartup_.prefer30);
                exposureStartup_.succeeded();
                exposureStatus=exposureStartup_.prefer30?"30 fps priority: exposure limited, gain automatic":"Automatic exposure";
            }catch(const std::exception& e) {
                exposureStartup_.failed(now());
                exposureStatus=exposureStartup_.exhausted()?"Exposure control unavailable: ":"Exposure control starting; retrying: ";
                exposureStatus+=e.what();
            }
            // Applying exposure can take 500 ms. Never send that pre-command frame
            // as fresh tracking data; the next poll acquires the latest frame.
            return {};
        }
        return frame;
    }
    if (!p_ || !p_->sensor)
        return {};
    auto &s = *p_;
    const double captureStart=now();
    auto read = [&](HANDLE stream, int pixelSize) -> std::optional<ImagePart> {
        NUI_IMAGE_FRAME f{};
        HRESULT hr = s.sensor->NuiImageStreamGetNextFrame(stream, 0, &f);
        if (FAILED(hr))
            return {};
        ImagePart part;
        part.stamp = f.liTimeStamp.QuadPart;
        part.id = f.dwFrameNumber;
        part.arrival = now();
        NUI_LOCKED_RECT rect{};
        hr = f.pFrameTexture->LockRect(0, &rect, nullptr, 0);
        if (SUCCEEDED(hr)) {
            if (rect.Pitch >= 640 * pixelSize) {
                part.bytes.resize(640 * 480 * pixelSize);
                for (int y = 0; y < 480; ++y)
                    std::memcpy(part.bytes.data() + y * 640 * pixelSize, rect.pBits + y * rect.Pitch,
                                640 * pixelSize);
            }
            f.pFrameTexture->UnlockRect(0);
        }
        s.sensor->NuiImageStreamReleaseFrame(stream, &f);
        if (part.bytes.empty())
            return {};
        return part;
    };
    if (auto a = read(s.rgb, 4))
        s.pairs.addA(std::move(*a));
    if (auto b = read(s.depth, 2))
        s.pairs.addB(std::move(*b));
    NUI_SKELETON_FRAME sk{};
    if (SUCCEEDED(s.sensor->NuiSkeletonGetNextFrame(0, &sk))) {
        s.skeletons.push_back(sk);
        while (s.skeletons.size() > 4)
            s.skeletons.pop_front();
    }
    auto pair = s.pairs.next();
    dropped = s.pairs.dropped;
    if (!pair)
        return {};
    auto f = std::make_shared<Frame>();
    f->rgbStamp = pair->first.stamp;
    f->depthStamp = pair->second.stamp;
    f->rgbId = pair->first.id;
    f->depthId = pair->second.id;
    f->arrival = std::max(pair->first.arrival, pair->second.arrival);
    f->host = s.clock.map(f->depthStamp, f->arrival);
    f->epoch = s.clock.resets;
    f->bgra = std::move(pair->first.bytes);
    f->depth.resize(640 * 480);
    std::memcpy(f->depth.data(), pair->second.bytes.data(), pair->second.bytes.size());
    f->calibrationBlob = s.calibration;
    std::vector<NUI_DEPTH_IMAGE_PIXEL> pixels(f->depth.size());
    std::vector<NUI_COLOR_IMAGE_POINT> colors(pixels.size());
    std::vector<Vector4> geometry(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        pixels[i].depth = f->depth[i] >> 3;
        pixels[i].playerIndex = f->depth[i] & 7;
    }
    checked(s.mapper->MapDepthFrameToColorFrame(
                NUI_IMAGE_RESOLUTION_640x480, DWORD(pixels.size()), pixels.data(), NUI_IMAGE_TYPE_COLOR,
                NUI_IMAGE_RESOLUTION_640x480, DWORD(colors.size()), colors.data()),
            "Register depth/color");
    checked(s.mapper->MapDepthFrameToSkeletonFrame(NUI_IMAGE_RESOLUTION_640x480, DWORD(pixels.size()),
                                                   pixels.data(), DWORD(geometry.size()), geometry.data()),
            "Map metric depth");
    f->mapping.resize(pixels.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        auto g = geometry[i];
        f->mapping[i] = {g.x, g.y, g.z, colors[i].x, colors[i].y};
    }
    const NUI_SKELETON_FRAME *nearest = nullptr;
    std::int64_t skew = 19;
    for (auto &a : s.skeletons) {
        auto dt = std::abs(a.liTimeStamp.QuadPart - f->depthStamp);
        if (dt < skew) {
            skew = dt;
            nearest = &a;
        }
    }
    if (nearest) {
        auto bytes = reinterpret_cast<const uint8_t *>(nearest);
        f->sdkSkeletonBlob.assign(bytes, bytes + sizeof(*nearest));
        f->skeletonStamp = nearest->liTimeStamp.QuadPart;
        f->skeletonId = nearest->dwFrameNumber;
        auto floor = nearest->vFloorClipPlane;
        double length = norm({floor.x, floor.y, floor.z});
        if (length > 0.9 && length < 1.1)
            f->floor = {{floor.x / length, floor.y / length, floor.z / length}, floor.w / length, true};
        constexpr std::array<std::pair<int, int>, 16> map{{{LShoulder, 4},
                                                           {RShoulder, 8},
                                                           {LElbow, 5},
                                                           {RElbow, 9},
                                                           {LWrist, 6},
                                                           {RWrist, 10},
                                                           {LHip, 12},
                                                           {RHip, 16},
                                                           {LKnee, 13},
                                                           {RKnee, 17},
                                                           {LAnkle, 14},
                                                           {RAnkle, 18},
                                                           {Head, 3},
                                                           {Neck, 2},
                                                           {Hip, 0},
                                                           {Nose, 3}}};
        for (int i = 0; i < NUI_SKELETON_COUNT; ++i) {
            const auto &b = nearest->SkeletonData[i];
            if (b.eTrackingState != NUI_SKELETON_TRACKED)
                continue;
            Body body;
            body.id = b.dwTrackingID;
            body.player = std::uint8_t(i + 1);
            for (auto [to, from] : map) {
                auto v = b.SkeletonPositions[from];
                auto state = b.eSkeletonPositionTrackingState[from];
                body.joints[to] = {{v.x, v.y, v.z},
                                   state == NUI_SKELETON_POSITION_TRACKED    ? 0.85
                                   : state == NUI_SKELETON_POSITION_INFERRED ? 0.2
                                                                             : 0,
                                   state == NUI_SKELETON_POSITION_TRACKED ? 0.035 : 0.15,
                                   1};
            }
            f->bodies.push_back(body);
        }
    }
    s.lastFrame = now();
    f->captureMs=(s.lastFrame-captureStart)*1000;
    return f;
}
std::string KinectCapture::calibrationKey() const {
    return v2_?v2_->calibrationKey():"";
}
} // namespace kf
