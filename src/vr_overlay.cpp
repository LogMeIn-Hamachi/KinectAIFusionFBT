#include "vr_overlay.hpp"
#include <Windows.h>
#include <objidl.h>
#include <gdiplus.h>
#include <openvr.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

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
    double doneTime_{0};
    int lastStep_{-1};
    int lastSeconds_{-1};
    bool lastCollecting_{false};
    bool lastWaiting_{false};
    bool lastDone_{false};
    bool lastSuccess_{false};
    int lastLeftSamples_{-1};
    int lastRightSamples_{-1};
    std::string lastLeftStatus_;
    std::string lastRightStatus_;
    bool lastLeftTracked_{false};
    bool lastRightTracked_{false};

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

    bool ensureOverlay() {
        if (!vr::VRSystem() || !vr::VROverlay()) return false;
        if (handle_ != vr::k_ulOverlayHandleInvalid) return true;

        vr::EVROverlayError err = vr::VROverlay()->FindOverlay("kinect_fbt.calibration_guide", &handle_);
        if (err != vr::EVROverlayError::VROverlayError_None) {
            err = vr::VROverlay()->CreateOverlay("kinect_fbt.calibration_guide", "Kinect FBT Calibration Guide", &handle_);
        }
        if (err == vr::EVROverlayError::VROverlayError_None) {
            vr::VROverlay()->SetOverlayWidthInMeters(handle_, 1.15f);
            vr::VROverlay()->SetOverlayAlpha(handle_, 0.96f);
            vr::HmdMatrix34_t mat = {
                1.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 1.0f, 0.0f, -0.05f,
                0.0f, 0.0f, 1.0f, -1.25f
            };
            vr::VROverlay()->SetOverlayTransformTrackedDeviceRelative(handle_, vr::k_unTrackedDeviceIndex_Hmd, &mat);
            return true;
        }
        return false;
    }

    void hide() {
        if (visible_ && handle_ != vr::k_ulOverlayHandleInvalid && vr::VROverlay()) {
            vr::VROverlay()->HideOverlay(handle_);
            visible_ = false;
            lastStep_ = -1;
        }
    }

    void render(const OverlayState& state) {
        Gdiplus::Bitmap bitmap(OverlayWidth, OverlayHeight, OverlayWidth * 4,
                               PixelFormat32bppARGB, pixelBuffer_.data());
        Gdiplus::Graphics g(&bitmap);
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
        if (state.done) {
            stepBadge = L"COMPLETE";
        } else if (state.step == 4) {
            stepBadge = L"VALIDATION CHECK (5/5)";
        } else {
            stepBadge = L"STEP " + std::to_wstring(state.step + 1) + L" OF 5";
        }
        Gdiplus::SolidBrush badgeBg(Gdiplus::Color(255, 24, 48, 60));
        Gdiplus::Pen badgeBorder(Gdiplus::Color(255, 60, 120, 135), 1.5f);
        drawRoundedRect(g, badgeBorder, badgeBg, OverlayWidth - 260, 32, 218, 32, 8);
        Gdiplus::StringFormat centerFormat;
        centerFormat.SetAlignment(Gdiplus::StringAlignmentCenter);
        centerFormat.SetLineAlignment(Gdiplus::StringAlignmentCenter);
        g.DrawString(stepBadge.c_str(), -1, &stepBadgeFont,
                     Gdiplus::RectF(OverlayWidth - 260, 32, 218, 32), &centerFormat, &tealBrush);

        // 3. Left Column: Pose Title and Description
        std::wstring poseTitles[5] = {
            L"Pose 1: Hands Low (Point Down)",
            L"Pose 2: Hands Forward (Chest Height)",
            L"Pose 3: Hands Apart (Diagonal Out)",
            L"Pose 4: Hands Up (In Front of Chest)",
            L"Pose 5: Check Pose (Hands Forward)"
        };
        std::wstring poseDescriptions[5] = {
            L"Hold both controllers low in front of your waist.\nPoint both controllers straight down towards the floor.\nKeep hands slightly forward away from your body so Kinect sees wrists.",
            L"Hold controllers forward at chest height.\nPoint both controllers straight forward towards the camera.",
            L"Hold hands comfortably apart to each side.\nPoint controllers diagonally outward to each side.",
            L"Hold controllers in front of your chest.\nPoint both controllers straight up towards the ceiling.",
            L"Validation check: hold hands forward between waist and chest.\nPoint controllers straight forward."
        };

        int step = std::clamp(state.step, 0, 4);
        if (!state.done) {
            g.DrawString(poseTitles[step].c_str(), -1, &titleFont, Gdiplus::PointF(40, 88), &whiteBrush);
            g.DrawString(poseDescriptions[step].c_str(), -1, &descFont, Gdiplus::RectF(40, 130, 560, 100), nullptr, &grayBrush);

            std::wstring tip = L"Keep your wrists clearly visible to the Kinect sensor. Stand naturally.";
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

        // Header for diagram
        Gdiplus::SolidBrush diagHeaderBrush(Gdiplus::Color(255, 130, 155, 175));
        g.DrawString(L"CONTROLLER ORIENTATION", -1, &diagramFont, Gdiplus::PointF(diagX + 16, diagY + 12), &diagHeaderBrush);

        // Draw Left & Right controller representations
        float ctrlLeftX = diagX + 75;
        float ctrlRightX = diagX + 225;
        float ctrlCenterY = diagY + 110;

        // Controller grips (capsules)
        Gdiplus::SolidBrush gripBrush(Gdiplus::Color(255, 45, 60, 78));
        Gdiplus::Pen gripBorder(Gdiplus::Color(255, 80, 105, 130), 2.0f);
        drawRoundedRect(g, gripBorder, gripBrush, ctrlLeftX - 22, ctrlCenterY - 32, 44, 64, 10);
        drawRoundedRect(g, gripBorder, gripBrush, ctrlRightX - 22, ctrlCenterY - 32, 44, 64, 10);

        // Controller labels & sample count
        std::wstring leftLabel = L"LEFT\n" + std::to_wstring(std::clamp(state.leftSamples, 0, 12)) + L"/12";
        std::wstring rightLabel = L"RIGHT\n" + std::to_wstring(std::clamp(state.rightSamples, 0, 12)) + L"/12";
        g.DrawString(leftLabel.c_str(), -1, &diagramFont, Gdiplus::RectF(ctrlLeftX - 35, ctrlCenterY + 36, 70, 32), &centerFormat, &diagHeaderBrush);
        g.DrawString(rightLabel.c_str(), -1, &diagramFont, Gdiplus::RectF(ctrlRightX - 35, ctrlCenterY + 36, 70, 32), &centerFormat, &diagHeaderBrush);

        // Controller tracking dots (Kinect sight + VR connection)
        bool leftOccluded = (state.leftStatus == "Kinect cannot see wrist");
        bool rightOccluded = (state.rightStatus == "Kinect cannot see wrist");

        Gdiplus::SolidBrush trackedDot(Gdiplus::Color(255, 78, 240, 143));
        Gdiplus::SolidBrush occludedDot(Gdiplus::Color(255, 255, 195, 60));
        Gdiplus::SolidBrush untrackedDot(Gdiplus::Color(255, 240, 78, 78));

        auto getDot = [&](bool tracked, bool occluded) -> Gdiplus::Brush* {
            if (!tracked) return &untrackedDot;
            if (occluded) return &occludedDot;
            return &trackedDot;
        };

        g.FillEllipse(getDot(state.leftTracked, leftOccluded), ctrlLeftX - 5.0f, ctrlCenterY - 26.0f, 10.0f, 10.0f);
        g.FillEllipse(getDot(state.rightTracked, rightOccluded), ctrlRightX - 5.0f, ctrlCenterY - 26.0f, 10.0f, 10.0f);

        // Directional arrows for current pose
        Gdiplus::Pen arrowPen(Gdiplus::Color(255, 83, 217, 255), 4.5f);
        Gdiplus::SolidBrush arrowBrush(Gdiplus::Color(255, 83, 217, 255));
        Gdiplus::SolidBrush arrowTextBrush(Gdiplus::Color(255, 83, 217, 255));

        std::wstring arrowLabels[5] = { L"POINT DOWN", L"POINT FORWARD", L"POINT OUTWARD", L"POINT UP", L"CHECK POSE" };
        g.DrawString(arrowLabels[step].c_str(), -1, &diagramFont, Gdiplus::RectF(diagX, diagY + 185, diagW, 20), &centerFormat, &arrowTextBrush);

        if (step == 0) {
            // Point Down
            drawArrow(g, arrowPen, arrowBrush, ctrlLeftX, ctrlCenterY - 10.0f, ctrlLeftX, ctrlCenterY + 28.0f, 12.0f);
            drawArrow(g, arrowPen, arrowBrush, ctrlRightX, ctrlCenterY - 10.0f, ctrlRightX, ctrlCenterY + 28.0f, 12.0f);
        } else if (step == 1 || step == 4) {
            // Point Forward (represented as 3D perspective / upward-forward ring)
            drawArrow(g, arrowPen, arrowBrush, ctrlLeftX, ctrlCenterY + 12.0f, ctrlLeftX, ctrlCenterY - 26.0f, 12.0f);
            drawArrow(g, arrowPen, arrowBrush, ctrlRightX, ctrlCenterY + 12.0f, ctrlRightX, ctrlCenterY - 26.0f, 12.0f);
            Gdiplus::Pen ringPen(Gdiplus::Color(255, 83, 217, 255), 2.5f);
            g.DrawEllipse(&ringPen, ctrlLeftX - 12.0f, ctrlCenterY - 22.0f, 24.0f, 14.0f);
            g.DrawEllipse(&ringPen, ctrlRightX - 12.0f, ctrlCenterY - 22.0f, 24.0f, 14.0f);
        } else if (step == 2) {
            // Point Diagonally Outward
            drawArrow(g, arrowPen, arrowBrush, ctrlLeftX + 5.0f, ctrlCenterY + 5.0f, ctrlLeftX - 25.0f, ctrlCenterY - 22.0f, 12.0f);
            drawArrow(g, arrowPen, arrowBrush, ctrlRightX - 5.0f, ctrlCenterY + 5.0f, ctrlRightX + 25.0f, ctrlCenterY - 22.0f, 12.0f);
        } else if (step == 3) {
            // Point Straight Up
            drawArrow(g, arrowPen, arrowBrush, ctrlLeftX, ctrlCenterY + 16.0f, ctrlLeftX, ctrlCenterY - 28.0f, 12.0f);
            drawArrow(g, arrowPen, arrowBrush, ctrlRightX, ctrlCenterY + 16.0f, ctrlRightX, ctrlCenterY - 28.0f, 12.0f);
        }

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
            g.DrawString(L"Closing overlay and resuming live tracker output in SteamVR...", -1, &bannerSubFont,
                         Gdiplus::RectF(bannerX, bannerY + 90, bannerW, 25), &centerFormat, &grayBrush);
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
                waitSub = L"Position controllers as shown above for " + poseTitles[step] + L", then squeeze trigger.";
            } else {
                waitSub = L"Take your time getting into position. Capture will not start until you squeeze a trigger.";
            }
            g.DrawString(waitSub.c_str(), -1, &bannerSubFont,
                         Gdiplus::RectF(bannerX + 30, bannerY + 82, bannerW - 60, 45), &centerFormat, &grayBrush);
        } else if (state.collecting) {
            // Actively capturing wrists (holding still)
            bool moving = (state.leftStatus.find("moving") != std::string::npos ||
                           state.rightStatus.find("moving") != std::string::npos);

            int minSamples = std::min(state.leftSamples, state.rightSamples);
            float sampleFraction = std::clamp(float(minSamples) / 12.0f, 0.05f, 1.0f);

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
                captTitle = L"\u26A0 MOVEMENT DETECTED \u2014 HOLD STILL (" + std::to_wstring(minSamples) + L"/12 samples)";
                captSub = L"Pause and hold hands completely steady in position.";
            } else if (state.secondsRemaining <= 0 && minSamples < 12) {
                captTitle = L"KEEP HOLDING STILL \u2014 FINALIZING SAMPLES (" + std::to_wstring(minSamples) + L"/12)";
                captSub = L"Almost done! Keep controllers steady until step advances automatically.";
            } else {
                captTitle = L"HOLD STILL \u2014 CAPTURING WRISTS (" + std::to_wstring(minSamples) + L"/12 samples)";
                captSub = L"Left: " + std::to_wstring(state.leftSamples) + L"/12 \u2022 Right: " + std::to_wstring(state.rightSamples) + L"/12 steady samples recorded.";
            }

            g.DrawString(captTitle.c_str(), -1, &bannerTitleFont,
                         Gdiplus::RectF(bannerX, bannerY + 30, bannerW, 35), &centerFormat, &captText);
            g.DrawString(captSub.c_str(), -1, &bannerSubFont,
                         Gdiplus::RectF(bannerX, bannerY + 70, bannerW, 25), &centerFormat, &grayBrush);

            // Progress bar
            float barX = bannerX + 60;
            float barY = bannerY + 115;
            float barW = bannerW - 120;
            float barH = 18;
            Gdiplus::SolidBrush barBg(Gdiplus::Color(255, 10, 28, 18));
            drawRoundedRect(g, Gdiplus::Pen(Gdiplus::Color(0,0,0,0)), barBg, barX, barY, barW, barH, 9);

            Gdiplus::SolidBrush fillBrush(textColor);
            drawRoundedRect(g, Gdiplus::Pen(Gdiplus::Color(0,0,0,0)), fillBrush, barX, barY, barW * sampleFraction, barH, 9);
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
        if (!state.active) {
            hide();
            return;
        }

        if (!ensureOverlay()) return;

        bool changed = (!visible_ ||
                        lastStep_ != state.step ||
                        lastSeconds_ != state.secondsRemaining ||
                        lastWaiting_ != state.waitingForReady ||
                        lastCollecting_ != state.collecting ||
                        lastDone_ != state.done ||
                        lastSuccess_ != state.success ||
                        lastLeftSamples_ != state.leftSamples ||
                        lastRightSamples_ != state.rightSamples ||
                        lastLeftStatus_ != state.leftStatus ||
                        lastRightStatus_ != state.rightStatus ||
                        lastLeftTracked_ != state.leftTracked ||
                        lastRightTracked_ != state.rightTracked);

        if (!changed) return;

        lastStep_ = state.step;
        lastSeconds_ = state.secondsRemaining;
        lastWaiting_ = state.waitingForReady;
        lastCollecting_ = state.collecting;
        lastDone_ = state.done;
        lastSuccess_ = state.success;
        lastLeftSamples_ = state.leftSamples;
        lastRightSamples_ = state.rightSamples;
        lastLeftStatus_ = state.leftStatus;
        lastRightStatus_ = state.rightStatus;
        lastLeftTracked_ = state.leftTracked;
        lastRightTracked_ = state.rightTracked;

        // Render the card into pixelBuffer_
        render(state);

        // Submit raw 32-bit ARGB buffer to OpenVR compositor
        vr::EVROverlayError err = vr::VROverlay()->SetOverlayRaw(handle_, pixelBuffer_.data(),
                                                                 OverlayWidth, OverlayHeight, 4);
        if (err == vr::EVROverlayError::VROverlayError_None) {
            vr::VROverlay()->ShowOverlay(handle_);
            visible_ = true;
        }
    }
};

VrOverlay::VrOverlay() : impl_(std::make_unique<Impl>()) {}
VrOverlay::~VrOverlay() = default;

void VrOverlay::update(const OverlayState& state) {
    if (impl_) impl_->update(state);
}

void VrOverlay::hide() {
    if (impl_) impl_->hide();
}

bool VrOverlay::isVisible() const {
    return impl_ && impl_->visible_;
}

} // namespace kf
