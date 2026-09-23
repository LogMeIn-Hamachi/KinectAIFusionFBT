#include "engine.hpp"
#include "build_version.hpp"
#include <Windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <future>
#include <iomanip>
#include <shellapi.h>
#include <sstream>
#include <sapi.h>
#include <windowsx.h>
using namespace kf;
namespace {
std::unique_ptr<Engine> engine;
HWND mainWindow{}, players{}, modeBox{}, modelBox{}, cadenceBox{};
HFONT font{}, titleFont{}, smallFont{};
ISpVoice *calibrationVoice{};
std::string lastSpokenPrompt;
bool wasCollecting{},wasBodyCollecting{};
std::string lastBodySpeech;
std::future<void> stopFuture;
bool closing{}, depthView{true}, advanced{};
std::vector<uint32_t> playerIds;
std::vector<int> modelChoices;
int setupStep{};
enum {
    Start = 101,
    Stop,
    Replay,
    Record,
    Output,
    Lock,
    Setup,
    BodyCal,
    Align,
    Export,
    Folder,
    ModeSelect,
    ModelSelect,
    CadenceSelect,
    Preview,
    Advanced,
    Saved,
    PlayerSelect,
    DepthToggle,
    ConstraintToggle,
    ContactToggle,
    VrToggle,
    SoleOffset,
    TiltDown,
    TiltUp,
    SteamVrToggle,
    LowEndCalibration,
    KneesToggle = 150,
    ElbowsToggle,
    ChestToggle,
    SaveOffsets = 130,
    OffsetBase = 140
};
std::wstring wide(const std::string &s) {
    if (s.empty())
        return {};
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), nullptr, 0);
    std::wstring out(n, L' ');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), int(s.size()), out.data(), n);
    return out;
}
void labelIfChanged(HWND h,const wchar_t* label) {
    wchar_t current[256];GetWindowTextW(h,current,256);
    if(std::wstring(current)!=label)SetWindowTextW(h,label);
}
void enableIfChanged(HWND h,bool enabled) {
    if(bool(IsWindowEnabled(h))!=enabled)EnableWindow(h,enabled);
}
void text(HDC dc, int x, int y, int w, int h, const std::wstring &s, COLORREF color = RGB(220, 228, 235),
          HFONT selected = nullptr) {
    SetTextColor(dc, color);
    SetBkMode(dc, TRANSPARENT);
    auto old = SelectObject(dc, selected ? selected : font);
    RECT rect{x, y, x + w, y + h};
    DrawTextW(dc, s.c_str(), -1, &rect, DT_LEFT | DT_WORDBREAK | DT_NOPREFIX);
    SelectObject(dc, old);
}
HWND control(const wchar_t *type, const wchar_t *label, int id, int x, int y, int w, int h, DWORD style = 0) {
    HWND c = CreateWindowExW(0, type, label, WS_CHILD | WS_VISIBLE | style, x, y, w, h, mainWindow,
                             reinterpret_cast<HMENU>(intptr_t(id)), nullptr, nullptr);
    SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
    return c;
}
void box(HDC dc, RECT r, COLORREF color) {
    auto b = CreateSolidBrush(color);
    FillRect(dc, &r, b);
    DeleteObject(b);
}
V2 project(const Frame &f, V3 p) {
    if(f.colorProjection && f.colorProjection->valid)return f.colorProjection->project(p);
    double best = 1e9;
    V2 uv{-100, -100}; // Use actual registration; nearest ray, not coincident RGB/depth pixels.
    V3 ray = unit(p);
    for (size_t i = 0; i < f.mapping.size(); i += 8) {
        auto m = f.mapping[i];
        if (m.z < .5 || m.u < 0 || m.u >= f.width || m.v < 0 || m.v >= f.height)
            continue;
        V3 point{m.x, m.y, m.z};
        double d = norm(unit(point) - ray) + std::abs(m.z - p.z) * .02;
        if (d < best) {
            best = d;
            uv = {double(m.u), double(m.v)};
        }
    }
    return uv;
}
void overlay(HDC dc, const Frame &frame, const Body &body, RECT r, COLORREF color, int thickness) {
    auto pen = CreatePen(PS_SOLID, thickness, color);
    auto old = SelectObject(dc, pen);
    std::array<V2, J> uv;
    for (int j = 0; j < J; ++j)
        if (body.joints[j].confidence > .15)
            uv[j] = project(frame, body.joints[j].p);
        else
            uv[j] = {-100, -100};
    for (auto [a, b] : bones) {
        if (uv[a].x < 0 || uv[b].x < 0)
            continue;
        int x = int(r.left + uv[a].x * (r.right - r.left) / frame.width),
            y = int(r.top + uv[a].y * (r.bottom - r.top) / frame.height);
        MoveToEx(dc, x, y, nullptr);
        LineTo(dc, int(r.left + uv[b].x * (r.right - r.left) / frame.width),
               int(r.top + uv[b].y * (r.bottom - r.top) / frame.height));
    }
    SelectObject(dc, old);
    DeleteObject(pen);
}
void bitmap(HDC dc, RECT r, int width, int height, const uint8_t *data) {
    BITMAPINFO info{};
    info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth = width;
    info.bmiHeader.biHeight = -height;
    info.bmiHeader.biPlanes = 1;
    info.bmiHeader.biBitCount = 32;
    info.bmiHeader.biCompression = BI_RGB;
    SetStretchBltMode(dc, COLORONCOLOR);
    StretchDIBits(dc, r.left, r.top, r.right - r.left, r.bottom - r.top, 0, 0, width, height, data, &info,
                  DIB_RGB_COLORS, SRCCOPY);
}
void samOverlay(HDC dc,const Keypoints &points,RECT r,int width,int height) {
    auto pen=CreatePen(PS_SOLID,1,RGB(238,112,235));auto old=SelectObject(dc,pen);
    for(auto [a,b]:bones) {
        if(a==Head || b==Head)continue; // MHR70 has no matching head-centre landmark.
        auto p=points[a].uv,q=points[b].uv;
        if(p.x<0 || p.x>=width || p.y<0 || p.y>=height || q.x<0 || q.x>=width || q.y<0 || q.y>=height)continue;
        MoveToEx(dc,int(r.left+p.x*(r.right-r.left)/width),int(r.top+p.y*(r.bottom-r.top)/height),nullptr);
        LineTo(dc,int(r.left+q.x*(r.right-r.left)/width),int(r.top+q.y*(r.bottom-r.top)/height));
    }
    SelectObject(dc,old);DeleteObject(pen);
}
void paint() {
    PAINTSTRUCT ps;
    HDC screen = BeginPaint(mainWindow, &ps);
    RECT client;
    GetClientRect(mainWindow, &client);
    HDC dc = CreateCompatibleDC(screen);
    HBITMAP canvas = CreateCompatibleBitmap(screen, client.right, client.bottom);
    auto old = SelectObject(dc, canvas);
    box(dc, client, RGB(15, 21, 29));
    auto s = engine->view();
    text(dc, 24, 18, 800, 38, L"Kinect RGB-D  /  Full-body tracking", RGB(237, 246, 250), titleFont);
    text(dc, 26, 59, 1150, 24,
         L"Kinect v1 / v2  •  native Windows  •  experimental fusion  •  all camera data stays on this PC",
         RGB(139, 161, 179), smallFont);
    std::wstring tiltLabel=L"Kinect tilt: "+(s.tiltAngle?std::to_wstring(*s.tiltAngle)+L" deg":L"unavailable");
    if(s.frame && s.frame->sensorVersion==2)tiltLabel=L"Kinect v2: adjust tilt by hand";
    if(s.tiltPending)tiltLabel+=L" (moving)";
    else if(s.tiltWait>0)tiltLabel+=L" ("+std::to_wstring(int(std::ceil(s.tiltWait)))+L"s)";
    text(dc,800,25,260,28,tiltLabel,RGB(199,215,226),smallFont);
    RECT rgb{24, 200, 664, 680};
    box(dc, rgb, RGB(5, 10, 16));
    if (!advanced && s.frame && s.frame->bgra.size() == size_t(s.frame->width) * s.frame->height * 4) {
        // Letterbox rather than stretching v2's wider image or cropping the feet.
        int h=std::min(480,640*s.frame->height/s.frame->width);
        int w=std::min(640,480*s.frame->width/s.frame->height);
        rgb.left+=(640-w)/2;rgb.top+=(480-h)/2;rgb.right=rgb.left+w;rgb.bottom=rgb.top+h;
        if(s.frame->sensorVersion==2) {
            std::wostringstream camera;
            camera<<L"Colour camera: ";
            if(s.frame->colorIntervalMs>0)camera<<std::fixed<<std::setprecision(0)<<1000/s.frame->colorIntervalMs<<L" fps";
            else camera<<L"timing unavailable";
            camera<<L" / "<<std::fixed<<std::setprecision(1)<<s.frame->exposureMs<<L" ms exposure\n"<<wide(s.exposureStatus);
            text(dc,40,204,610,55,camera.str(),RGB(164,185,202),smallFont);
        }
        bitmap(dc, rgb, s.frame->width, s.frame->height, s.frame->bgra.data());
        for (auto &raw : s.frame->bodies)
            if (raw.id == s.state.body.id)
                overlay(dc, *s.frame, raw, rgb, RGB(255, 173, 73), 1);
        overlay(dc, *s.frame, s.state.body, rgb, RGB(76, 229, 203), 2);
        if(s.samOverlay)samOverlay(dc,*s.samOverlay,rgb,s.frame->width,s.frame->height);
    } else
        text(dc, 110, 390, 480, 140,
             L"No camera frame\n\nConnect and power the Kinect, then Start.\nOr open a local recording to "
             L"inspect a session.",
             RGB(153, 174, 191));
    text(dc, 24, 176, 640, 22, s.samOverlay?L"Orange = SDK  /  pink = AI estimate  /  mint = output":
         L"RGB  •  orange = SDK  /  mint = processed", RGB(164, 185, 202), smallFont);
    RECT depth{684, 200, 1004, 440};
    box(dc, depth, RGB(5, 10, 16));
    if (!advanced && depthView && s.frame && s.frame->depth.size() == size_t(s.frame->depthWidth) * s.frame->depthHeight) {
        std::vector<uint8_t> pixels(320 * 240 * 4);
        for (int y = 0; y < 240; ++y)
            for (int x = 0; x < 320; ++x) {
                uint16_t packed = s.frame->depth[(y*s.frame->depthHeight/240)*s.frame->depthWidth+x*s.frame->depthWidth/320];
                int mm = packed >> 3;
                size_t i = (y * 320 + x) * 4;
                uint8_t value = mm >= 800 && mm <= 4000 ? uint8_t(255 - (mm - 800) * 220 / 3200) : 0;
                pixels[i] = value;
                pixels[i + 1] = uint8_t(value * .8);
                pixels[i + 2] = (packed & 7) ? value : 0;
            }
        int dw=std::min(320,240*s.frame->depthWidth/s.frame->depthHeight);
        int dh=std::min(240,320*s.frame->depthHeight/s.frame->depthWidth);
        depth.left+=(320-dw)/2;depth.top+=(240-dh)/2;depth.right=depth.left+dw;depth.bottom=depth.top+dh;
        bitmap(dc, depth, 320, 240, pixels.data());
    }
    text(dc, 684, 176, 450, 22, L"Original metric depth  •  holes remain visible", RGB(164, 185, 202),
         smallFont);
    std::wostringstream d;
    d << wide(modeName(s.state.mode)) << L"\n" << wide(s.poseSource) << L"\n"
      << wide(s.sensor) << L"\n\n"
      << wide(s.inference) << L"\n"
      << L"Neural  " << wide(s.cadenceStatus) << L" / " << std::fixed << std::setprecision(1) << s.health.neuralHz << L" actual Hz\n\n"
      << wide(s.vr) << L"\n\n"
      << std::fixed << std::setprecision(1) << L"Frames  " << s.frames << L"    Dropped  " << s.dropped
      << L"\nInference  " << s.inferenceMs << L" ms    Fit  " << s.state.fitMs << L" ms\nQueue age  "
      << s.queueMs << L" ms\nArrival → estimate  " << s.arrivalToEstimateMs << L" ms\nOSC bundles  " << s.sent
      << L"    Errors  " << s.sendErrors << L"\nRecord drops  " << s.recordDrops << L"\n";
    text(dc, 684, 456, 470, 278, d.str(), RGB(199, 215, 226), smallFont);
    box(dc, {24, 704, 664, 824}, RGB(24, 34, 44));
    text(dc, 40, 715, 606, 96, wide(s.notice), RGB(207, 227, 237));
    // A segment-attached tracker schematic accompanies the camera overlays; it is not a synthetic pose feed.
    int tx = 1020, ty = 222, row=0;
    const bool compact=s.settings.extraTrackers!=0;
    for (int i = 0; i < trackerCount; ++i) {
        if(!trackerEnabled(trackerMask(s.settings.extraTrackers),i))continue;
        const int y=ty+row++*(compact?27:72);
        auto t = s.state.trackers[i];
        auto brush = CreateSolidBrush(!t.valid                ? RGB(82, 95, 108)
                                      : t.angularSigma.y >= 1 ? RGB(236, 181, 75)
                                      : s.state.learnedDirection[i] ? RGB(113,172,243)
                                                              : RGB(72, 216, 181));
        auto ob = SelectObject(dc, brush);
        Ellipse(dc, tx, y, tx + 18, y + 18);
        SelectObject(dc, ob);
        DeleteObject(brush);
        if(compact) {text(dc,tx+26,y-3,150,25,wide(trackerNames[i]),RGB(173,193,209),smallFont);continue;}
        std::wostringstream tr;
        tr << (i == 0   ? L"Hips"
               : i == 1 ? L"Left foot"
                        : L"Right foot")
           << L"\n"
           << (s.state.learnedPosition[i]?(s.modelChoice==1?L"NLF-S estimate ":L"SAM estimate "):L"SDK/depth ")
           << std::fixed << std::setprecision(0) << t.positionSigma*100 << L" cm σ\n"
           << (t.angularSigma.y >= 1 ? L"Direction held" : s.state.learnedDirection[i] ? L"Direction inferred" : L"Direction observed");
        text(dc, tx + 26, ty - 4 + i * 72, 150, 68, tr.str(), RGB(173, 193, 209), smallFont);
    }
    if (s.collecting) {
        std::wostringstream line;
        line << L"Alignment: follow the current pose and countdown";
        line << L"\nSamples: " << s.calibrationSamples;
        text(dc, 684, 748, 490, 60, line.str(), RGB(97, 225, 194));
    } else {
        std::wostringstream line;
        line << L"Alignment: " << (s.calibration.valid ? L"accepted" : L"required") << L"   residual "
             << std::fixed << std::setprecision(1) << s.calibration.rms * 1000 << L" mm";
        text(dc, 684, 752, 500, 52, line.str(), RGB(151, 178, 196), smallFont);
    }
    text(dc,684,724,510,36,wide(s.outputStatus),RGB(151,178,196),smallFont);
    if (s.bodyCollecting || s.collecting || (!s.output && !s.calibrationDetail.empty())) {
        // Keep the camera preview visible while replacing the timing panel with actionable alignment
        // feedback.
        box(dc, {684, 456, 1200, 736}, RGB(15, 21, 29));
        text(dc, 684, 460, 500, 152,
             s.bodyCollecting?wide(s.bodyPrompt):s.collecting ? wide(s.calibrationPrompt) : L"Alignment agreement by device", RGB(97, 225, 194));
        text(dc, 684, 614, 510, 120, s.bodyCollecting?L"Stand naturally. Keep feet apart and arms relaxed. The capture uses four seconds of steady data.":wide(s.calibrationDetail), RGB(220, 228, 235), smallFont);
    }
    if (advanced) {
        box(dc, {24, 200, 1200, 680}, RGB(26, 37, 49));
        text(dc, 40, 218, 1100, 70,
             L"Advanced: device-to-joint offsets in metres. Controller wrist offsets are learned by Align to VR.\n"
             L"The headset offset is only a tracking assumption; the new alignment does not use head samples.",
             RGB(218, 233, 240));
        text(dc, 42, 317, 130, 28, L"HMD");
        text(dc, 42, 355, 130, 28, L"Left hand");
        text(dc, 42, 393, 130, 28, L"Right hand");
        text(dc, 184, 289, 650, 22, L"X                         Y                         Z",
             RGB(164, 185, 202));
        text(dc, 880, 289, 290, 24, L"Ankle-to-sole distance (metres)", RGB(164, 185, 202), smallFont);
        text(dc, 40, 572, 320, 24, L"AI model (change while stopped)",RGB(164,185,202),smallFont);
        text(dc, 385, 572, 400, 24, L"Tracking mode / diagnostic comparison",RGB(164,185,202),smallFont);
        text(dc, 805, 572, 360, 24, L"GPU cadence / performance",RGB(164,185,202),smallFont);
        text(dc, 40, 641, 1100, 28,
             L"Use RGB-D fusion for normal tracking. Sole distance is measured below the ankle; default 0.075 m.",
             RGB(164,185,202),smallFont);
        text(dc,40,532,1110,38,s.replay?
             L"Replay uses recorded geometry settings. Only comparison mode and extra tracker choices can be changed; AI runs on every frame.":
             L"Low-end PC collects calibration samples at uneven frame rates; final alignment checks stay unchanged.",
             RGB(164,185,202),smallFont);
    }
    BitBlt(screen, 0, 0, client.right, client.bottom, dc, 0, 0, SRCCOPY);
    SelectObject(dc, old);
    DeleteObject(canvas);
    DeleteDC(dc);
    EndPaint(mainWindow, &ps);
}
void advancedControls(bool show) {
    auto s = engine->view();
    for (int i = 0; i < 9; ++i) {
        auto h = GetDlgItem(mainWindow, OffsetBase + i);
        if (!h) {
            int row = i / 3, col = i % 3;
            h = control(L"EDIT", L"", OffsetBase + i, 178 + col * 155, 317 + row * 38, 135, 27,
                        WS_BORDER | ES_AUTOHSCROLL);
        }
        double values[]{s.settings.deviceOffsets[i / 3].x, s.settings.deviceOffsets[i / 3].y,
                        s.settings.deviceOffsets[i / 3].z};
        std::wostringstream t;
        t << values[i % 3];
        SetWindowTextW(h, t.str().c_str());
        ShowWindow(h, show ? SW_SHOW : SW_HIDE);
    }
    if (!GetDlgItem(mainWindow, SoleOffset))
        control(L"EDIT", L"", SoleOffset, 880, 317, 135, 27, WS_BORDER | ES_AUTOHSCROLL);
    std::wostringstream sole;
    sole << s.settings.soleOffset;
    SetDlgItemTextW(mainWindow, SoleOffset, sole.str().c_str());
    SendDlgItemMessageW(mainWindow,SteamVrToggle,BM_SETCHECK,s.steamVrOutput?BST_CHECKED:BST_UNCHECKED,0);
    SendDlgItemMessageW(mainWindow,LowEndCalibration,BM_SETCHECK,s.lowEndCalibration?BST_CHECKED:BST_UNCHECKED,0);
    SendMessageW(cadenceBox, CB_SETCURSEL, s.cadenceChoice, 0);
    for (int id : {DepthToggle, ConstraintToggle, ContactToggle, VrToggle, SaveOffsets, SoleOffset, SteamVrToggle, LowEndCalibration, ModelSelect, ModeSelect, CadenceSelect})
        ShowWindow(GetDlgItem(mainWindow, id), show ? SW_SHOW : SW_HIDE);
}
void setup() {
    static const wchar_t *steps[]{
        L"1 / Placement\n\nPlace the Kinect on a stable surface with feet, floor and wrists visible. Your head may be outside the image. "
        L"Kinect v1: use the minus/plus motor buttons. Kinect v2: adjust its mount by hand. Set tilt before alignment. "
        L"Start with the normal 30 Hz mode. Avoid sunlight and depth holes. Keep feet within the depth "
        L"image.",
        L"2 / Player\n\nPress Start. Choose your body ID and click Lock player. A second person will never "
        L"be selected automatically. If your identity is lost and replaced, select yourself explicitly "
        L"again.",
        L"3 / Body and floor\n\nPress Capture proportions. You get eight seconds to position yourself, followed by a four-second capture. Stand naturally with feet visible. Check the "
        L"orange SDK and mint processed overlays. The SDK floor plane is used for sole-height contact "
        L"constraints; do not move the camera afterward.",
        L"4 / SteamVR alignment\n\nReset OVR space drag and rotation before aligning or confirming a saved alignment after an app restart. "
        L"Use a headset with two positional controllers exposed through SteamVR; no headset-specific preset is needed. "
        L"Align to VR uses one guided routine. Take as long as you need on each pose; squeeze either controller trigger when ready, or click Capture pose. "
        L"A three-second settling countdown follows your ready signal. Keep the same normal grip and point controllers as shown; no exact palm twists. "
        L"Keep wrists in view and clear of your torso. Use comfortable positions with wrists clear of your torso. "
        L"The app learns wrist offsets automatically and does not use head samples. Headset-only or untracked controllers are not supported.",
        L"5 / VRChat\n\nAfter alignment, click Start trackers. SteamVR receives Waist, Left Foot and Right Foot trackers. Use VRChat Calibrate "
        L"FBT. Check hip and foot tracker spheres, height and facing direction. OVR space drag and rotation now move the trackers with your playspace. "
        L"Repeat alignment after physically moving the camera or changing your room setup. After restarting this app, reset OVR offsets and confirm saved alignment. If SteamVR restarts while tracking, Align to VR again."};
    MessageBoxW(mainWindow, steps[setupStep], L"Guided setup", MB_OK | MB_ICONINFORMATION);
    setupStep = (setupStep + 1) % 5;
}
void stopAsync() {
    if (stopFuture.valid())
        return;
    stopFuture = std::async(std::launch::async, [] { engine->stop(); });
}
LRESULT CALLBACK procedure(HWND window, UINT msg, WPARAM wp, LPARAM lp) {
    try {
        switch (msg) {
        case WM_CREATE: {
            mainWindow = window;
            font =
                CreateFontW(-17, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                            CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
            titleFont = CreateFontW(-28, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                    OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH,
                                    L"Segoe UI");
            smallFont =
                CreateFontW(-14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
                            CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH, L"Segoe UI");
            control(L"BUTTON", L"Start", Start, 24, 96, 86, 34);
            control(L"BUTTON", L"Stop", Stop, 118, 96, 86, 34);
            control(L"BUTTON", L"Open replay", Replay, 212, 96, 126, 34);
            control(L"BUTTON", L"Record locally", Record, 346, 96, 138, 34);
            control(L"BUTTON", L"Start trackers", Output, 492, 96, 134, 34);
            control(L"BUTTON", L"Setup guide", Setup, 634, 96, 126, 34);
            control(L"BUTTON", L"Diagnostics", Export, 768, 96, 122, 34);
            control(L"BUTTON", L"Open folder", Folder, 898, 96, 126, 34);
            control(L"BUTTON", L"Advanced", Advanced, 1032, 96, 128, 34);
            control(L"BUTTON", L"-", TiltDown, 1064, 18, 44, 32);
            control(L"BUTTON", L"+", TiltUp, 1116, 18, 44, 32);
            modelBox=control(WC_COMBOBOXW,L"",ModelSelect,40,598,320,160,CBS_DROPDOWNLIST);
            {
                wchar_t exe[32768];GetModuleFileNameW(nullptr,exe,32768);
                auto root=std::filesystem::path(exe).parent_path();
                const wchar_t* names[]{L"SAM - original",L"NLF-S - lighter body model",L"SAM - faster",L"SAM - optimized"};
                const char* graphs[]{"assets/sam3d/backbone.onnx","assets/nlf/pose.onnx","assets/sam3d-fp8/backbone.onnx","assets/sam3d-optimized/backbone.onnx"};
                for(int choice:{3,0,2,1})if(std::filesystem::exists(root/graphs[choice])) {
                    SendMessageW(modelBox,CB_ADDSTRING,0,reinterpret_cast<LPARAM>(names[choice]));
                    if(choice==engine->view().modelChoice)SendMessageW(modelBox,CB_SETCURSEL,modelChoices.size(),0);
                    modelChoices.push_back(choice);
                }
            }
            players =
                control(WC_COMBOBOXW, L"", PlayerSelect, 24, 141, 150, 200, CBS_DROPDOWNLIST | WS_VSCROLL);
            control(L"BUTTON", L"Lock player", Lock, 182, 140, 122, 30);
            control(L"BUTTON", L"Capture proportions", BodyCal, 312, 140, 184, 30);
            control(L"BUTTON", L"Align to VR", Align, 504, 140, 126, 30);
            control(L"BUTTON", L"Confirm saved alignment", Saved, 638, 140, 210, 30);
            control(L"STATIC",L"Add:",149,864,146,40,24);
            control(L"BUTTON",L"Knees",KneesToggle,905,140,88,30,BS_AUTOCHECKBOX);
            control(L"BUTTON",L"Elbows",ElbowsToggle,997,140,94,30,BS_AUTOCHECKBOX);
            control(L"BUTTON",L"Chest",ChestToggle,1095,140,88,30,BS_AUTOCHECKBOX);
            for(int i=0;i<3;++i)SendDlgItemMessageW(window,KneesToggle+i,BM_SETCHECK,
                (engine->view().settings.extraTrackers&(1<<i))?BST_CHECKED:BST_UNCHECKED,0);
            modeBox = control(WC_COMBOBOXW, L"", ModeSelect, 385, 598,400,180,CBS_DROPDOWNLIST);
            for (auto name : {L"RGB-D fusion (experimental)", L"Raw SDK baseline", L"Filtered SDK baseline"})
                SendMessageW(modeBox, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(name));
            SendMessageW(modeBox, CB_SETCURSEL, 0, 0);
            cadenceBox = control(WC_COMBOBOXW, L"", CadenceSelect, 805, 598, 360, 180, CBS_DROPDOWNLIST);
            for (auto name : {L"Auto (GPU adaptive)", L"30 Hz (Full AI)", L"20 Hz (Balanced)", L"15 Hz (Low GPU / Heavy VRChat)"})
                SendMessageW(cadenceBox, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(name));
            SendMessageW(cadenceBox, CB_SETCURSEL, engine->view().cadenceChoice, 0);
            control(L"BUTTON", L"Depth / SAM body", DepthToggle, 40, 452, 210, 30, BS_AUTOCHECKBOX);
            control(L"BUTTON", L"SDK bone fit", ConstraintToggle, 265, 452, 210, 30, BS_AUTOCHECKBOX);
            control(L"BUTTON", L"Foot contact", ContactToggle, 490, 452, 180, 30, BS_AUTOCHECKBOX);
            control(L"BUTTON", L"VR constraints", VrToggle, 685, 452, 190, 30, BS_AUTOCHECKBOX);
            control(L"BUTTON", L"Apply offsets", SaveOffsets, 675, 317, 160, 32);
            control(L"BUTTON", L"SteamVR trackers (instead of OSC)", SteamVrToggle, 40, 489, 460, 30, BS_AUTOCHECKBOX);
            control(L"BUTTON", L"Low-end PC (calibration)", LowEndCalibration, 520, 489, 400, 30, BS_AUTOCHECKBOX);
            for (int id : {DepthToggle, ConstraintToggle, ContactToggle, VrToggle})
                SendDlgItemMessageW(window, id, BM_SETCHECK, BST_CHECKED, 0);
            advancedControls(false);
            SetTimer(window, 1, 100, nullptr);
            return 0;
        }
        case WM_TIMER: {
            if (stopFuture.valid() &&
                stopFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                stopFuture.get();
                if (closing) {
                    DestroyWindow(window);
                    return 0;
                }
            }
            auto s = engine->view();
            for(auto [controlId,checked]:{std::pair{DepthToggle,s.settings.depth},
                    {ConstraintToggle,s.settings.constraints},{ContactToggle,s.settings.contacts},{VrToggle,s.settings.vrConstraints}}) {
                const auto h=GetDlgItem(window,controlId);
                const auto mark=checked?BST_CHECKED:BST_UNCHECKED;
                if(SendMessageW(h,BM_GETCHECK,0,0)!=mark)SendMessageW(h,BM_SETCHECK,mark,0);
                enableIfChanged(h,!s.replay && !s.collecting && !s.bodyCollecting);
            }
            enableIfChanged(cadenceBox,!s.replay);
            enableIfChanged(GetDlgItem(window,LowEndCalibration),!s.replay && !s.collecting);
            enableIfChanged(GetDlgItem(window,SaveOffsets),!s.replay && !s.collecting && !s.bodyCollecting);
            for(int i=0;i<9;++i)enableIfChanged(GetDlgItem(window,OffsetBase+i),!s.replay && !s.collecting && !s.bodyCollecting);
            enableIfChanged(GetDlgItem(window,SoleOffset),!s.replay && !s.collecting && !s.bodyCollecting);
            enableIfChanged(modeBox,!s.collecting && !s.bodyCollecting);
            enableIfChanged(GetDlgItem(window,Output),s.running && !s.replay && !s.collecting && !s.bodyCollecting);
            if (s.collecting && s.calibrationSpeech != lastSpokenPrompt) {
                lastSpokenPrompt = s.calibrationSpeech;
                if (calibrationVoice)
                    calibrationVoice->Speak(wide(lastSpokenPrompt).c_str(),
                                            SPF_ASYNC | SPF_PURGEBEFORESPEAK | SPF_IS_NOT_XML, nullptr);
            } else if (wasCollecting && !s.collecting) {
                lastSpokenPrompt.clear();
                if (calibrationVoice)
                    calibrationVoice->Speak(s.calibration.valid
                                                ? L"Alignment accepted."
                                                : L"Alignment stopped. Check the device results on screen.",
                                            SPF_ASYNC | SPF_PURGEBEFORESPEAK | SPF_IS_NOT_XML, nullptr);
            }
            wasCollecting = s.collecting;
            if(s.bodyCollecting && s.bodySpeech!=lastBodySpeech) {
                lastBodySpeech=s.bodySpeech;
                if(calibrationVoice)calibrationVoice->Speak(wide(lastBodySpeech).c_str(),SPF_ASYNC|SPF_PURGEBEFORESPEAK|SPF_IS_NOT_XML,nullptr);
            } else if(wasBodyCollecting && !s.bodyCollecting) {
                lastBodySpeech.clear();
                if(calibrationVoice)calibrationVoice->Speak(wide(s.notice).c_str(),SPF_ASYNC|SPF_PURGEBEFORESPEAK|SPF_IS_NOT_XML,nullptr);
            }
            wasBodyCollecting=s.bodyCollecting;
            enableIfChanged(GetDlgItem(window,SteamVrToggle),!s.output);
            for(int i=0;i<3;++i) {
                auto h=GetDlgItem(window,KneesToggle+i);
                enableIfChanged(h,!s.output && !s.collecting && !s.bodyCollecting);
                const auto checked=(s.settings.extraTrackers&(1<<i))?BST_CHECKED:BST_UNCHECKED;
                if(SendMessageW(h,BM_GETCHECK,0,0)!=checked)SendMessageW(h,BM_SETCHECK,checked,0);
            }
            enableIfChanged(GetDlgItem(window,BodyCal),s.running && !s.collecting && !s.bodyCollecting);
            labelIfChanged(GetDlgItem(window,Saved),s.collecting?L"Capture pose":L"Confirm saved alignment");
            enableIfChanged(GetDlgItem(window,Saved),!s.bodyCollecting && (!s.collecting || s.calibrationWaiting));
            labelIfChanged(GetDlgItem(window,Align),s.collecting?L"Cancel align":L"Align to VR");
            enableIfChanged(GetDlgItem(window,Align),s.running && !s.bodyCollecting);
            enableIfChanged(GetDlgItem(window, Start), !s.running && !stopFuture.valid());
            enableIfChanged(modelBox,!s.running && !stopFuture.valid());
            enableIfChanged(GetDlgItem(window, Replay), !s.running && !stopFuture.valid());
            bool canTilt=s.running && !s.replay && !s.recording && s.tiltAngle.has_value() && !s.tiltPending && s.tiltWait<=0 && !stopFuture.valid();
            enableIfChanged(GetDlgItem(window,TiltDown),canTilt && *s.tiltAngle>-27);
            enableIfChanged(GetDlgItem(window,TiltUp),canTilt && *s.tiltAngle<27);
            labelIfChanged(GetDlgItem(window, Record), s.recording ? L"Stop recording" : L"Record locally");
            labelIfChanged(GetDlgItem(window, Output), s.output ? L"Pause output" : s.steamVrOutput?L"Start trackers":L"Enable OSC");
            std::vector<uint32_t> ids;
            if (s.frame)
                for (auto b : s.frame->bodies)
                    ids.push_back(b.id);
            if (ids != playerIds) {
                playerIds = ids;
                SendMessageW(players, CB_RESETCONTENT, 0, 0);
                for (auto id : ids) {
                    auto label = L"Body " + std::to_wstring(id);
                    SendMessageW(players, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(label.c_str()));
                }
                if (!ids.empty())
                    SendMessageW(players, CB_SETCURSEL, 0, 0);
            }
            if (!IsIconic(window))
                InvalidateRect(window, nullptr, FALSE);
            return 0;
        }
        case WM_COMMAND: {
            int id = LOWORD(wp);
            if(id==ModelSelect && HIWORD(wp)==CBN_SELCHANGE) {
                const auto selected=int(SendMessageW(modelBox,CB_GETCURSEL,0,0));
                if(selected>=0 && selected<int(modelChoices.size()))engine->chooseModel(modelChoices[selected]);
                for(size_t i=0;i<modelChoices.size();++i)if(modelChoices[i]==engine->view().modelChoice)
                    SendMessageW(modelBox,CB_SETCURSEL,i,0);
                return 0;
            }
            if (id == ModeSelect && HIWORD(wp) == CBN_SELCHANGE) {
                auto s = engine->view().settings;
                s.baseline = int(SendMessageW(modeBox, CB_GETCURSEL, 0, 0));
                engine->settings(s);
                return 0;
            }
            if (id == CadenceSelect && HIWORD(wp) == CBN_SELCHANGE) {
                const auto selected = int(SendMessageW(cadenceBox, CB_GETCURSEL, 0, 0));
                if (selected >= 0 && selected <= 3) engine->chooseCadence(selected);
                return 0;
            }
            if (HIWORD(wp) != BN_CLICKED)
                return 0;
            switch (id) {
            case KneesToggle:
            case ElbowsToggle:
            case ChestToggle: {
                int extras=0;
                for(int i=0;i<3;++i)if(SendDlgItemMessageW(window,KneesToggle+i,BM_GETCHECK,0,0)==BST_CHECKED)extras|=1<<i;
                engine->chooseTrackers(extras);
                break;
            }
            case LowEndCalibration:
                engine->chooseLowEndCalibration(SendDlgItemMessageW(window,LowEndCalibration,BM_GETCHECK,0,0)==BST_CHECKED);
                break;
            case Start:
                engine->start();
                break;
            case Stop:
                stopAsync();
                break;
            case Replay: {
                wchar_t path[32768]{};
                OPENFILENAMEW dialog{sizeof dialog};
                dialog.hwndOwner = window;
                dialog.lpstrFilter = L"Kinect RGB-D recording\0*.kfr\0\0";
                dialog.lpstrFile = path;
                dialog.nMaxFile = 32768;
                dialog.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
                if (GetOpenFileNameW(&dialog))
                    engine->start(path);
                break;
            }
            case Record:
                engine->toggleRecord();
                break;
            case Output:
                engine->output(!engine->view().output);
                break;
            case Lock: {
                int index = int(SendMessageW(players, CB_GETCURSEL, 0, 0));
                if (index >= 0 && index < int(playerIds.size()))
                    engine->select(playerIds[index]);
                break;
            }
            case Setup:
                setup();
                break;
            case SteamVrToggle:
                engine->chooseOutput(SendDlgItemMessageW(window,SteamVrToggle,BM_GETCHECK,0,0)==BST_CHECKED);break;
            case BodyCal:
                engine->bodyCalibration();
                break;
            case Align:
                if(engine->view().collecting)engine->cancelCalibration();else engine->beginCalibration();
                break;
            case TiltDown:engine->tilt(-1);break;
            case TiltUp:engine->tilt(1);break;
            case Saved:
                if(engine->view().collecting)engine->captureAlignmentPose();else engine->useSavedCalibration();
                break;
            case Export:
                engine->exportDiagnostics();
                break;
            case Folder:
                ShellExecuteW(window, L"open", engine->root().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
                break;
            case Advanced:
                advanced = !advanced;
                advancedControls(advanced);
                break;
            case SaveOffsets: {
                auto s = engine->view().settings;
                wchar_t soleText[64]{};
                GetDlgItemTextW(window, SoleOffset, soleText, 64);
                wchar_t *soleEnd;
                s.soleOffset = wcstod(soleText, &soleEnd);
                if (soleEnd == soleText || *soleEnd || !std::isfinite(s.soleOffset) || s.soleOffset < .02 ||
                    s.soleOffset > .2)
                    throw std::runtime_error("Sole distance must be between 0.02 and 0.20 metres");
                for (int row = 0; row < 3; ++row) {
                    double v[3];
                    for (int col = 0; col < 3; ++col) {
                        wchar_t b[64];
                        GetDlgItemTextW(window, OffsetBase + row * 3 + col, b, 64);
                        wchar_t *end;
                        v[col] = wcstod(b, &end);
                        if (end == b || *end || !std::isfinite(v[col]) || std::abs(v[col]) > .5)
                            throw std::runtime_error("Offsets must be finite metres between -0.5 and 0.5");
                    }
                    s.deviceOffsets[row] = {v[0], v[1], v[2]};
                }
                engine->settings(s);
                engine->output(false);
                advanced = false;
                advancedControls(false);
                break;
            }
            case DepthToggle:
            case ConstraintToggle:
            case ContactToggle:
            case VrToggle: {
                auto s = engine->view().settings;
                s.depth = SendDlgItemMessageW(window, DepthToggle, BM_GETCHECK, 0, 0) == BST_CHECKED;
                s.constraints =
                    SendDlgItemMessageW(window, ConstraintToggle, BM_GETCHECK, 0, 0) == BST_CHECKED;
                s.contacts = SendDlgItemMessageW(window, ContactToggle, BM_GETCHECK, 0, 0) == BST_CHECKED;
                s.vrConstraints = SendDlgItemMessageW(window, VrToggle, BM_GETCHECK, 0, 0) == BST_CHECKED;
                engine->settings(s);
                break;
            }
            }
            InvalidateRect(window, nullptr, FALSE);
            return 0;
        }
        case WM_PAINT:
            paint();
            return 0;
        case WM_ERASEBKGND:
            return 1;
        case WM_CLOSE:
            closing = true;
            stopAsync();
            return 0;
        case WM_DESTROY:
            KillTimer(window, 1);
            DeleteObject(font);
            DeleteObject(titleFont);
            DeleteObject(smallFont);
            PostQuitMessage(0);
            return 0;
        }
    } catch (const std::exception &e) {
        MessageBoxW(window, wide(e.what()).c_str(), L"Kinect RGB-D", MB_OK | MB_ICONERROR);
    }
    return DefWindowProcW(window, msg, wp, lp);
}
} // namespace
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE, PWSTR, int show) {
    HRESULT comStatus = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (SUCCEEDED(comStatus))
        CoCreateInstance(__uuidof(SpVoice), nullptr, CLSCTX_INPROC_SERVER, __uuidof(ISpVoice),
                         reinterpret_cast<void **>(&calibrationVoice));
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    SetProcessDPIAware();
    wchar_t path[32768];
    GetModuleFileNameW(nullptr, path, 32768);
    engine = std::make_unique<Engine>(std::filesystem::path(path).parent_path());
    INITCOMMONCONTROLSEX controls{sizeof controls, ICC_STANDARD_CLASSES};
    InitCommonControlsEx(&controls);
    WNDCLASSW wc{};
    wc.lpfnWndProc = procedure;
    wc.hInstance = instance;
    wc.lpszClassName = L"KinectRGBDWindow";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    RegisterClassW(&wc);
    RECT r{0, 0, 1224, 850};
    AdjustWindowRect(&r, WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX, FALSE);
    auto window =
        CreateWindowW(wc.lpszClassName, wide("Kinect RGB-D - Experimental full-body tracking (" KF_VERSION ")").c_str(),
                      WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_CLIPCHILDREN, CW_USEDEFAULT, CW_USEDEFAULT,
                      r.right - r.left, r.bottom - r.top, nullptr, nullptr, instance, nullptr);
    if (!window)
        return 1;
    ShowWindow(window, show);
    MSG message;
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        if (!IsDialogMessageW(window, &message)) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
    engine.reset();
    if (calibrationVoice) {
        calibrationVoice->Speak(nullptr, SPF_PURGEBEFORESPEAK, nullptr);
        calibrationVoice->Release();
    }
    if (SUCCEEDED(comStatus))
        CoUninitialize();
    return 0;
}
