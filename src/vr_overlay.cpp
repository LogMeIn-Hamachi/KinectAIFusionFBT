#include "vr_overlay.hpp"
#include "alignment.hpp"
#include "runtime_lifetime.hpp"
#include <chrono>
#include <Windows.h>
#include <objidl.h>
#include <gdiplus.h>
#include <openvr.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "msimg32.lib")

namespace kf {

namespace {
constexpr int OverlayWidth = 1024;
constexpr int OverlayHeight = 640;

void drawRoundedRect(Gdiplus::Graphics& g, const Gdiplus::Pen& pen, const Gdiplus::Brush& brush,
                     float x, float y, float w, float h, float r) {
    Gdiplus::GraphicsPath path;
    path.AddArc(x, y, r + r, r + r, 180, 90);
    path.AddArc(x + w - r - r, y, r + r, r + r, 270, 90);
    path.AddArc(x + w - r - r, y + h - r - r, r + r, r + r, 0, 90);
    path.AddArc(x, y + h - r - r, r + r, r + r, 90, 90);
    path.CloseFigure();
    g.FillPath(&brush, &path);
    if (pen.GetWidth() > 0)
        g.DrawPath(&pen, &path);
}

void drawArrow(Gdiplus::Graphics& g, const Gdiplus::Pen& pen, const Gdiplus::Brush& brush,
               float startX, float startY, float endX, float endY, float headSize) {
    g.DrawLine(&pen, startX, startY, endX, endY);
    float angle = std::atan2(endY - startY, endX - startX);
    float arrowAngle1 = angle + 2.5f;
    float arrowAngle2 = angle - 2.5f;
    Gdiplus::PointF points[3] = {
        Gdiplus::PointF(endX, endY),
        Gdiplus::PointF(endX + std::cos(arrowAngle1) * headSize, endY + std::sin(arrowAngle1) * headSize),
        Gdiplus::PointF(endX + std::cos(arrowAngle2) * headSize, endY + std::sin(arrowAngle2) * headSize)
    };
    g.FillPolygon(&brush, points, 3);
}

} // namespace

struct VrOverlay::Impl {
    ULONG_PTR gdiplusToken_{0};
    vr::VROverlayHandle_t handle_{vr::k_ulOverlayHandleInvalid};
    std::vector<uint8_t> pixelBuffer_;
    bool visible_{false};
    bool hidePending_{};
    OverlayRefresh refresh_;
    double nextHealthCheck_{};
    uint32_t runtimeToken_{};
    const std::string key_="kinect_fbt.calibration_guide."+std::to_string(GetCurrentProcessId());
    std::string lastError_="none";
    unsigned submissions_{}, failures_{};
    int displayedStep_{-1};
    bool gpuResult(HRESULT result,const char* operation) {
        if(SUCCEEDED(result))return true;
        lastError_=std::string(operation)+": HRESULT "+std::to_string(result);return false;
    }
    bool overlayResult(vr::EVROverlayError result,const char* operation) {
        if(result==vr::VROverlayError_None)return true;
        lastError_=std::string(operation)+": OpenVR error "+std::to_string(int(result));return false;
    }
    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture_;

    bool ensureTexture() {
        if(device_ && FAILED(device_->GetDeviceRemovedReason())) {
            texture_.Reset();context_.Reset();device_.Reset();
        }
        if(texture_)return true;
        // Use the compositor's adapter, including PCs with an integrated GPU.
        int adapterIndex=-1;vr::VRSystem()->GetDXGIOutputInfo(&adapterIndex);
        Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if(adapterIndex<0){lastError_="SteamVR compositor adapter unavailable";return false;}
        if(!gpuResult(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())),"CreateDXGIFactory1") ||
           !gpuResult(factory->EnumAdapters1(UINT(adapterIndex),adapter.GetAddressOf()),"EnumAdapters1"))return false;
        if(!device_ && !gpuResult(D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,
            device_.GetAddressOf(),nullptr,context_.GetAddressOf()),"D3D11CreateDevice"))return false;
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width=OverlayWidth;desc.Height=OverlayHeight;desc.MipLevels=1;desc.ArraySize=1;
        desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.SampleDesc.Count=1;
        desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        return gpuResult(device_->CreateTexture2D(&desc,nullptr,texture_.GetAddressOf()),"CreateTexture2D");
    }


    Impl() {
        Gdiplus::GdiplusStartupInput gdiplusStartupInput;
        Gdiplus::GdiplusStartup(&gdiplusToken_, &gdiplusStartupInput, nullptr);
        pixelBuffer_.resize(OverlayWidth * OverlayHeight * 4);
    }

    ~Impl() {
        hide();
        if (handle_ != vr::k_ulOverlayHandleInvalid && vr::VROverlay()) {
            vr::VROverlay()->DestroyOverlay(handle_);
            handle_ = vr::k_ulOverlayHandleInvalid;
        }
        if (gdiplusToken_) {
            Gdiplus::GdiplusShutdown(gdiplusToken_);
            gdiplusToken_ = 0;
        }
    }

    void syncRuntime() {
        const auto token=vr::VR_GetInitToken();
        if(token!=runtimeToken_) {
            runtimeToken_=token;handle_=vr::k_ulOverlayHandleInvalid;visible_=false;
            refresh_.reset();texture_.Reset();context_.Reset();device_.Reset();
        }
    }
    bool ensureOverlay() {
        syncRuntime();
        if (!vr::VRSystem() || !vr::VROverlay()) {lastError_="SteamVR interfaces unavailable";handle_=vr::k_ulOverlayHandleInvalid;visible_=false;return false;}
        if (handle_ != vr::k_ulOverlayHandleInvalid) {
            vr::EVROverlayError status=vr::VROverlayError_None;
            char key[128]{};
            vr::VROverlay()->GetOverlayKey(handle_,key,sizeof(key),&status);
            if(status==vr::VROverlayError_None && std::string(key)==key_)return true;
            handle_=vr::k_ulOverlayHandleInvalid;visible_=false;refresh_.reset();
        }

        vr::EVROverlayError err = vr::VROverlay()->FindOverlay(key_.c_str(), &handle_);
        if (err != vr::EVROverlayError::VROverlayError_None) {
            err = vr::VROverlay()->CreateOverlay(key_.c_str(), "Kinect FBT Calibration Guide", &handle_);
        }
        if (err == vr::EVROverlayError::VROverlayError_None) {
            if(!overlayResult(vr::VROverlay()->SetOverlayWidthInMeters(handle_,1.15f),"SetOverlayWidth") ||
               !overlayResult(vr::VROverlay()->SetOverlayAlpha(handle_,.96f),"SetOverlayAlpha")) {
                vr::VROverlay()->DestroyOverlay(handle_);handle_=vr::k_ulOverlayHandleInvalid;return false;
            }
            vr::HmdMatrix34_t mat = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, -0.05f,
                0.0f, 0.0f, 1.0f, -1.25f
            };
            if(!overlayResult(vr::VROverlay()->SetOverlayTransformTrackedDeviceRelative(handle_,vr::k_unTrackedDeviceIndex_Hmd,&mat),"SetOverlayTransform")) {
                vr::VROverlay()->DestroyOverlay(handle_);handle_=vr::k_ulOverlayHandleInvalid;return false;
            }
            refresh_.reset();return true;
        }
        overlayResult(err,"CreateOverlay");return false;
    }

    void hide() {
        syncRuntime();
        if (hidePending_ && handle_ != vr::k_ulOverlayHandleInvalid && vr::VROverlay()) {
            vr::VROverlay()->HideOverlay(handle_);
        }
        visible_=false;hidePending_=false;refresh_.reset();
    }

    void render(const OverlayState& state) {
        Gdiplus::Bitmap bitmap(OverlayWidth, OverlayHeight, OverlayWidth * 4,
                               PixelFormat32bppARGB, pixelBuffer_.data());
        Gdiplus::Graphics g(&bitmap);
        g.Clear(Gdiplus::Color(0,0,0,0));
        g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
        g.SetTextRenderingHint(Gdiplus::TextRenderingHintAntiAliasGridFit);

        // 1. Base card background
        Gdiplus::SolidBrush bgBrush(Gdiplus::Color(245, 14, 20, 30));
        Gdiplus::Pen borderPen(Gdiplus::Color(255, 36, 75, 88), 3.0f);
        drawRoundedRect(g, borderPen, bgBrush, 12, 12, OverlayWidth - 24, OverlayHeight - 24, 24);

        // Subtle accent top line
        Gdiplus::Pen accentPen(Gdiplus::Color(255, 97, 225, 194), 2.5f);
        g.DrawLine(&accentPen, 36.0f, 22.0f, OverlayWidth - 36.0f, 22.0f);

        // Typography helpers
        Gdiplus::FontFamily fontFamily(L"Segoe UI");
        Gdiplus::Font headerFont(&fontFamily, 15, Gdiplus::FontStyleBold, Gdiplus::UnitPixel);
        Gdiplus::Font stepBadgeFont(&fontFamily, 14, Gdiplus::FontStyleBold, Gdiplus::UnitPixel);
        Gdiplus::Font titleFont(&fontFamily, 26, Gdiplus::FontStyleBold, Gdiplus::UnitPixel);
        Gdiplus::Font descFont(&fontFamily, 19, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
        Gdiplus::Font detailFont(&fontFamily, 15, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
        Gdiplus::Font bannerTitleFont(&fontFamily, 22, Gdiplus::FontStyleBold, Gdiplus::UnitPixel);
        Gdiplus::Font bannerSubFont(&fontFamily, 16, Gdiplus::FontStyleRegular, Gdiplus::UnitPixel);
        Gdiplus::Font diagramFont(&fontFamily, 13, Gdiplus::FontStyleBold, Gdiplus::UnitPixel);

        Gdiplus::SolidBrush tealBrush(Gdiplus::Color(255, 97, 225, 194));
        Gdiplus::SolidBrush whiteBrush(Gdiplus::Color(255, 255, 255, 255));
        Gdiplus::SolidBrush grayBrush(Gdiplus::Color(255, 164, 185, 202));
        Gdiplus::SolidBrush mutedBrush(Gdiplus::Color(255, 110, 130, 150));

        // 2. Header
        g.DrawString(L"KINECT FBT  \u2022  ALIGN TO VR", -1, &headerFont, Gdiplus::PointF(40, 36), &tealBrush);

        // Step pill badge
        std::wstring stepBadge;
        Gdiplus::Color badgeBgColor(255, 24, 48, 60);
        Gdiplus::Color badgeBorderColor(255, 60, 120, 135);
        Gdiplus::Color badgeTextColor(255, 97, 225, 194);

        if (state.done) {
            stepBadge = L"COMPLETE";
        } else if (state.isRetry) {
            stepBadge = L"RETRY REQUIRED (" + std::to_wstring(state.step + 1) + L"/5)";
            badgeBgColor = Gdiplus::Color(255, 54, 28, 14);
            badgeBorderColor = Gdiplus::Color(255, 180, 80, 30);
            badgeTextColor = Gdiplus::Color(255, 255, 195, 60);
        } else if (state.step == 4) {
            stepBadge = L"VALIDATION CHECK (5/5)";
        } else {
            stepBadge = L"STEP " + std::to_wstring(state.step + 1) + L" OF 5";
        }
        Gdiplus::SolidBrush badgeBg(badgeBgColor);
        Gdiplus::Pen badgeBorder(badgeBorderColor, 1.5f);
        drawRoundedRect(g, badgeBorder, badgeBg, OverlayWidth - 260, 32, 218, 32, 8);
        Gdiplus::SolidBrush badgeTextBrush(badgeTextColor);
        Gdiplus::StringFormat centerFormat;
        centerFormat.SetAlignment(Gdiplus::StringAlignmentCenter);
        centerFormat.SetLineAlignment(Gdiplus::StringAlignmentCenter);
        g.DrawString(stepBadge.c_str(), -1, &stepBadgeFont,
                     Gdiplus::RectF(OverlayWidth - 260, 32, 218, 32), &centerFormat, &badgeTextBrush);

        // 3. Left Column: Pose Title and Description
        int step = std::clamp(state.step, 0, 4);
        const auto title=alignmentPoseTitle(step);
        const std::wstring titleW(title.begin(),title.end());
        if (!state.done) {
            g.DrawString(titleW.c_str(), -1, &titleFont, Gdiplus::PointF(40, 88), &whiteBrush);
            const auto text=alignmentPoseInstruction(step);
            const std::wstring description(text.begin(),text.end());
            g.DrawString(description.c_str(), -1, &descFont, Gdiplus::RectF(40, 130, 560, 100), nullptr, &grayBrush);

            std::wstring tip = L"Keep your normal grip. Move your arms; keep wrists comfortable.";
            g.DrawString(tip.c_str(), -1, &detailFont, Gdiplus::PointF(40, 245), &mutedBrush);
        } else {
            g.DrawString(L"Alignment Complete!", -1, &titleFont, Gdiplus::PointF(40, 88), &whiteBrush);
            std::wstring doneMsg = L"Your Kinect camera and SteamVR tracking spaces are now aligned.\nTrackers are ready for use in VRChat and SteamVR.";
            g.DrawString(doneMsg.c_str(), -1, &descFont, Gdiplus::RectF(40, 130, 560, 100), nullptr, &grayBrush);
            if (!state.agreement.empty()) {
                std::wstring agreeW(state.agreement.begin(), state.agreement.end());
                g.DrawString(agreeW.c_str(), -1, &detailFont, Gdiplus::RectF(40, 240, 560, 80), nullptr, &tealBrush);
            }
        }

        // 4. Right Column: Visual Diagram Box (340x210)
        float diagX = OverlayWidth - 380;
        float diagY = 88;
        float diagW = 340;
        float diagH = 210;
        Gdiplus::SolidBrush diagBg(Gdiplus::Color(255, 11, 16, 24));
        Gdiplus::Pen diagBorder(Gdiplus::Color(255, 30, 45, 60), 1.5f);
        drawRoundedRect(g, diagBorder, diagBg, diagX, diagY, diagW, diagH, 14);

        // Hand-position guide, drawn from the user's perspective (left is left).
        Gdiplus::SolidBrush diagHeaderBrush(Gdiplus::Color(255,130,155,175));
        g.DrawString(L"YOUR LEFT                 YOUR RIGHT",-1,&diagramFont,
                     Gdiplus::PointF(diagX+25,diagY+12),&diagHeaderBrush);
        const float cx=diagX+170, shoulderY=diagY+64, waistY=diagY+157;
        Gdiplus::Pen bodyPen(Gdiplus::Color(255,82,102,122),3.f);
        Gdiplus::Pen armPen(Gdiplus::Color(255,83,217,255),4.f);
        g.DrawEllipse(&bodyPen,cx-15,diagY+28,30.f,27.f);
        g.DrawLine(&bodyPen,cx-39,shoulderY,cx+39,shoulderY);
        g.DrawLine(&bodyPen,cx-39,shoulderY,cx-28,waistY);
        g.DrawLine(&bodyPen,cx+39,shoulderY,cx+28,waistY);
        g.DrawLine(&bodyPen,cx-28,waistY,cx+28,waistY);
        const float spread=step==1?112.f:76.f;
        const float leftY=diagY+(step==0||step==1?139.f:88.f);
        const float rightY=step==4?diagY+139:leftY;
        auto arm=[&](float side,float handY) {
            const float elbowX=cx+side*51,elbowY=diagY+119,handX=cx+side*spread;
            g.DrawLine(&armPen,cx+side*39,shoulderY,elbowX,elbowY);
            g.DrawLine(&armPen,elbowX,elbowY,handX,handY);
            g.FillEllipse(&tealBrush,handX-9,handY-9,18.f,18.f);
        };
        arm(-1,leftY);arm(1,rightY);
        if(step==3) {
            // Forward reach cannot be represented by height alone.
            Gdiplus::SolidBrush arrowBrush(Gdiplus::Color(255,83,217,255));
            drawArrow(g,armPen,arrowBrush,cx-76,leftY,cx-91,leftY-22,8);
            drawArrow(g,armPen,arrowBrush,cx+76,rightY,cx+91,rightY-22,8);
        }
        const wchar_t* hints[]{L"FOREARMS FORWARD",L"SHALLOW V / HANDS STILL FORWARD",
            L"ELBOWS LOW / HANDS OFF CHEST",L"REACH TOWARDS THE CAMERA",L"LEFT AT CHEST / RIGHT AT WAIST"};
        g.DrawString(hints[step],-1,&diagramFont,Gdiplus::RectF(diagX,diagY+180,diagW,24),&centerFormat,&diagHeaderBrush);
        bool leftOccluded=(state.leftStatus=="Kinect cannot see wrist");
        bool rightOccluded=(state.rightStatus=="Kinect cannot see wrist");
        std::wstring counts=L"Left "+std::to_wstring(std::min(state.leftSamples,12))+L"/12    Right "+std::to_wstring(std::min(state.rightSamples,12))+L"/12";
        g.DrawString(counts.c_str(),-1,&diagramFont,Gdiplus::RectF(diagX,diagY+158,diagW,21),&centerFormat,&diagHeaderBrush);

        // 5. Bottom Status / Progress Banner
        float bannerX = 36;
        float bannerY = 320;
        float bannerW = OverlayWidth - 72;
        float bannerH = OverlayHeight - bannerY - 36;

        if (state.done) {
            // Completed state
            Gdiplus::SolidBrush doneBg(Gdiplus::Color(255, 14, 48, 30));
            Gdiplus::Pen donePen(Gdiplus::Color(255, 40, 130, 80), 2.0f);
            drawRoundedRect(g, donePen, doneBg, bannerX, bannerY, bannerW, bannerH, 16);

            Gdiplus::SolidBrush doneText(Gdiplus::Color(255, 78, 240, 143));
            g.DrawString(L"\u2713 ALIGNMENT SUCCESSFUL", -1, &bannerTitleFont,
                         Gdiplus::RectF(bannerX, bannerY + 45, bannerW, 35), &centerFormat, &doneText);
            g.DrawString(L"Alignment saved. Use the app to enable tracker output when ready.", -1, &bannerSubFont,
                         Gdiplus::RectF(bannerX, bannerY + 90, bannerW, 25), &centerFormat, &grayBrush);
        } else if (state.isRetry && state.waitingForReady) {
            // Dedicated Amber / Coral "TRY AGAIN" Banner
            Gdiplus::SolidBrush retryBg(Gdiplus::Color(255, 48, 26, 14));
            Gdiplus::Pen retryPen(Gdiplus::Color(255, 180, 80, 30), 2.0f);
            drawRoundedRect(g, retryPen, retryBg, bannerX, bannerY, bannerW, bannerH, 16);

            Gdiplus::SolidBrush retryTitleBrush(Gdiplus::Color(255, 255, 195, 60));
            std::wstring retryTitle = L"\u26A0 POSE DID NOT PASS \u2014 SQUEEZE TRIGGER TO RETRY";
            g.DrawString(retryTitle.c_str(), -1, &bannerTitleFont,
                         Gdiplus::RectF(bannerX, bannerY + 28, bannerW, 35), &centerFormat, &retryTitleBrush);

            std::wstring reasonW;
            if (!state.retryReason.empty()) {
                reasonW = std::wstring(state.retryReason.begin(), state.retryReason.end());
            } else if (!state.feedback.empty()) {
                reasonW = std::wstring(state.feedback.begin(), state.feedback.end());
            } else {
                reasonW = L"Wrists were occluded or not held steady. Check your position and try again.";
            }

            g.DrawString(reasonW.c_str(), -1, &bannerSubFont,
                         Gdiplus::RectF(bannerX + 30, bannerY + 68, bannerW - 60, 80), &centerFormat, &whiteBrush);

            Gdiplus::SolidBrush actionBrush(Gdiplus::Color(255, 255, 215, 90));
            std::wstring actionText = L">> RELEASE & SQUEEZE EITHER TRIGGER TO RETRY <<   \u2022   Earlier poses kept";
            g.DrawString(actionText.c_str(), -1, &diagramFont,
                         Gdiplus::RectF(bannerX, bannerY + 164, bannerW, 25), &centerFormat, &actionBrush);
        } else if (state.waitingForReady) {
            // Waiting for trigger state
            Gdiplus::SolidBrush waitBg(Gdiplus::Color(255, 18, 38, 54));
            Gdiplus::Pen waitPen(Gdiplus::Color(255, 45, 95, 130), 2.0f);
            drawRoundedRect(g, waitPen, waitBg, bannerX, bannerY, bannerW, bannerH, 16);

            Gdiplus::SolidBrush waitText(Gdiplus::Color(255, 83, 217, 255));
            std::wstring waitTitle;
            if (step > 0 && state.feedback.find("Could not see") == std::string::npos && state.feedback.find("did not agree") == std::string::npos) {
                waitTitle = L"\u2713 POSE " + std::to_wstring(step) + L" COMPLETE! SQUEEZE TRIGGER FOR POSE " + std::to_wstring(step + 1);
            } else {
                waitTitle = L">> SQUEEZE EITHER TRIGGER WHEN READY <<";
            }
            g.DrawString(waitTitle.c_str(), -1, &bannerTitleFont,
                         Gdiplus::RectF(bannerX, bannerY + 38, bannerW, 35), &centerFormat, &waitText);

            std::wstring waitSub;
            if (!state.feedback.empty() && state.feedback.find("Could not see") != std::string::npos) {
                waitSub = L"Wrists were occluded. Adjust your stance and squeeze trigger to retry this pose (earlier poses kept).";
            } else if (step > 0) {
                waitSub = L"Position controllers as shown above for " + titleW + L", then squeeze trigger.";
            } else {
                waitSub = L"Take your time getting into position. Capture will not start until you squeeze a trigger.";
            }
            g.DrawString(waitSub.c_str(), -1, &bannerSubFont,
                         Gdiplus::RectF(bannerX + 30, bannerY + 82, bannerW - 60, 45), &centerFormat, &grayBrush);
        } else if (state.collecting) {
            // Actively capturing wrists (holding still)
            bool moving = (state.leftStatus.find("moving") != std::string::npos ||
                           state.rightStatus.find("moving") != std::string::npos);

            float sampleFraction = std::clamp(float(state.progressPercent) / 100.0f, 0.0f, 1.0f);

            bool alert = (leftOccluded || rightOccluded || moving);
            Gdiplus::Color bgColor = alert ? Gdiplus::Color(255, 48, 32, 14) : Gdiplus::Color(255, 14, 46, 26);
            Gdiplus::Color penColor = alert ? Gdiplus::Color(255, 140, 95, 30) : Gdiplus::Color(255, 38, 120, 68);
            Gdiplus::Color textColor = alert ? Gdiplus::Color(255, 255, 195, 60) : Gdiplus::Color(255, 78, 240, 143);

            Gdiplus::SolidBrush captBg(bgColor);
            Gdiplus::Pen captPen(penColor, 2.0f);
            drawRoundedRect(g, captPen, captBg, bannerX, bannerY, bannerW, bannerH, 16);

            Gdiplus::SolidBrush captText(textColor);
            std::wstring captTitle;
            std::wstring captSub;

            if (leftOccluded || rightOccluded) {
                captTitle = L"\u26A0 KINECT CANNOT SEE WRISTS \u2014 ADJUST POSITION";
                captSub = L"Hold controllers slightly forward and clear of your body so the Kinect camera can see wrists.";
            } else if (moving) {
                captTitle = L"\u26A0 MOVEMENT DETECTED \u2014 HOLD STILL";
                captSub = L"Pause and hold hands completely steady in position.";
            } else if (state.secondsRemaining <= 0) {
                captTitle = L"KEEP BOTH WRISTS VISIBLE \u2014 FINAL CHECK";
                captSub = L"Waiting for fresh, steady measurements from both wrists.";
            } else {
                captTitle = L"HOLD STILL \u2014 CAPTURING WRISTS";
                captSub = L"About " + std::to_wstring(state.secondsRemaining) + L" seconds of steady observations remaining.";
            }

            g.DrawString(captTitle.c_str(), -1, &bannerTitleFont,
                         Gdiplus::RectF(bannerX, bannerY + 30, bannerW, 35), &centerFormat, &captText);
            g.DrawString(captSub.c_str(), -1, &bannerSubFont,
                         Gdiplus::RectF(bannerX, bannerY + 70, bannerW, 25), &centerFormat, &grayBrush);

            const std::string detail="Left: "+state.leftStatus+" | Right: "+state.rightStatus;
            const std::wstring detailW(detail.begin(),detail.end());
            g.DrawString(detailW.c_str(),-1,&bannerSubFont,
                         Gdiplus::RectF(bannerX+25,bannerY+165,bannerW-50,42),&centerFormat,&grayBrush);
            const std::wstring timeout=L"If this pose cannot be captured, retry instructions appear in "+std::to_wstring(state.retrySeconds)+L"s.";
            g.DrawString(timeout.c_str(),-1,&diagramFont,
                         Gdiplus::RectF(bannerX+25,bannerY+220,bannerW-50,24),&centerFormat,&grayBrush);

            // Progress bar
            float barX = bannerX + 60;
            float barY = bannerY + 115;
            float barW = bannerW - 120;
            float barH = 18;
            Gdiplus::SolidBrush barBg(Gdiplus::Color(255, 10, 28, 18));
            drawRoundedRect(g, Gdiplus::Pen(Gdiplus::Color(0,0,0,0)), barBg, barX, barY, barW, barH, 9);

            Gdiplus::SolidBrush fillBrush(textColor);
            if(sampleFraction>0)
                drawRoundedRect(g, Gdiplus::Pen(Gdiplus::Color(0,0,0,0)), fillBrush, barX, barY, barW * sampleFraction, barH,
                                std::min(9.f,barW*sampleFraction/2));
        } else {
            // Settling state (3s countdown)
            Gdiplus::SolidBrush settleBg(Gdiplus::Color(255, 48, 38, 14));
            Gdiplus::Pen settlePen(Gdiplus::Color(255, 125, 98, 36), 2.0f);
            drawRoundedRect(g, settlePen, settleBg, bannerX, bannerY, bannerW, bannerH, 16);

            Gdiplus::SolidBrush settleText(Gdiplus::Color(255, 255, 205, 60));
            std::wstring settleTitle = L"GET READY \u2014 SETTLING IN POSITION (" + std::to_wstring(std::max(0, state.secondsRemaining)) + L"s)";
            g.DrawString(settleTitle.c_str(), -1, &bannerTitleFont,
                         Gdiplus::RectF(bannerX, bannerY + 30, bannerW, 35), &centerFormat, &settleText);

            g.DrawString(L"Settle into the pose. Capturing will start automatically...", -1, &bannerSubFont,
                         Gdiplus::RectF(bannerX, bannerY + 70, bannerW, 25), &centerFormat, &grayBrush);

            // Progress bar
            float barX = bannerX + 60;
            float barY = bannerY + 115;
            float barW = bannerW - 120;
            float barH = 18;
            Gdiplus::SolidBrush barBg(Gdiplus::Color(255, 28, 22, 8));
            drawRoundedRect(g, Gdiplus::Pen(Gdiplus::Color(0,0,0,0)), barBg, barX, barY, barW, barH, 9);

            float fraction = std::clamp((3.0f - float(state.secondsRemaining)) / 3.0f, 0.05f, 1.0f);
            Gdiplus::SolidBrush fillBrush(Gdiplus::Color(255, 255, 205, 60));
            drawRoundedRect(g, Gdiplus::Pen(Gdiplus::Color(0,0,0,0)), fillBrush, barX, barY, barW * fraction, barH, 9);
        }
    }

    void update(const OverlayState& state) {
        syncRuntime();
        if (!state.active) {
            hide();
            return;
        }

        const double now=std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
        // Validate without resubmitting an unchanged texture. Drain this
        // overlay's events so long sessions do not accumulate image/UI events.
        if(now>=nextHealthCheck_) {
            nextHealthCheck_=now+1.;
            if(!ensureOverlay()){refresh_.complete(state,now,false);return;}
            if(!vr::VROverlay()->IsOverlayVisible(handle_)){visible_=false;refresh_.reset();}
        }
        if(handle_!=vr::k_ulOverlayHandleInvalid && vr::VROverlay()) {
            vr::VREvent_t event{};
            for(int i=0;i<64 && vr::VROverlay()->PollNextOverlayEvent(handle_,&event,sizeof(event));++i){}
        }
        if(!refresh_.due(state,now))return;
        if(!ensureOverlay() || !ensureTexture()){++failures_;refresh_.complete(state,now,false);return;}
        hidePending_=true;
        render(state);
        vr::Texture_t image{texture_.Get(),vr::TextureType_DirectX,vr::ColorSpace_Gamma};
        auto* overlay=vr::VROverlay();
        const bool success=presentOverlayFrame(visible_,
            [&]{context_->UpdateSubresource(texture_.Get(),0,nullptr,pixelBuffer_.data(),OverlayWidth*4,0);},
            [&]{return overlay && overlayResult(overlay->SetOverlayTexture(handle_,&image),"SetOverlayTexture");},
            [&]{context_->Flush();},
            [&]{return overlayResult(overlay->ShowOverlay(handle_),"ShowOverlay");});
        if(success){visible_=true;++submissions_;displayedStep_=state.done?5:state.step;}
        else ++failures_;
        refresh_.complete(state,now,success);
        // A failed update leaves the last visible texture in place until retry.

    }
};

VrOverlay::VrOverlay() : impl_(std::make_unique<Impl>()) {}
VrOverlay::~VrOverlay() {
    auto lifetime=openVrLifetime.read();
    impl_.reset();
}

void VrOverlay::update(const OverlayState& state) {
    std::lock_guard lock(mutex_);
    auto lifetime=openVrLifetime.read();
    if (impl_) impl_->update(state);
}

void VrOverlay::hide() {
    std::lock_guard lock(mutex_);
    auto lifetime=openVrLifetime.read();
    if (impl_) impl_->hide();
}

bool VrOverlay::isVisible() const {
    std::lock_guard lock(mutex_);
    return impl_ && impl_->visible_;
}

std::string VrOverlay::diagnostics() const {
    std::lock_guard lock(mutex_);
    if(!impl_)return "overlay=unavailable\n";
    return "overlay_visible="+std::to_string(impl_->visible_)+
        "\noverlay_submissions="+std::to_string(impl_->submissions_)+
        "\noverlay_failures="+std::to_string(impl_->failures_)+
        "\noverlay_displayed_step="+std::to_string(impl_->displayedStep_)+
        "\noverlay_last_error="+impl_->lastError_+"\n";
}
} // namespace kf
