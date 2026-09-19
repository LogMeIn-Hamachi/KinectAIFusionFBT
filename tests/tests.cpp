#include "io.hpp"
#include "alignment.hpp"
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
using namespace kf;
static int count = 0;
static void check(bool value, const char *name) {
    ++count;
    if (!value)
        throw std::runtime_error(name);
}
static Body body() {
    Body b;
    b.id = 7;
    b.player = 1;
    for (auto &j : b.joints)
        j = {{0, 1, 2}, 0.85, 0.035, 1};
    b.joints[Hip].p = {0, 1, 2};
    b.joints[Neck].p = {0, 1.5, 2};
    b.joints[Head].p = {0, 1.75, 2};
    for (int side = 0; side < 2; ++side) {
        double x = side ? 0.15 : -0.15;
        b.joints[LHip + side].p = {x, 1, 2};
        b.joints[LKnee + side].p = {x, 0.55, 2};
        b.joints[LAnkle + side].p = {x, 0.1, 2};
        b.joints[LHeel + side] = {{x, 0.05, 2.05}, 0, 1, 0};
        b.joints[LToe + side] = {{x, 0.05, 1.85}, 0, 1, 0};
        b.joints[LShoulder + side].p = {x * 1.5, 1.5, 2};
        b.joints[LElbow + side].p = {x * 2, 1.25, 2};
        b.joints[LWrist + side].p = {x * 2, 1.05, 2};
    }
    return b;
}
int main() {
    try {
        for (V3 e : std::array<V3, 6>{{{0, 0, 0},
                                       {10, 40, -20},
                                       {89.999, 170, 50},
                                       {-90, 70, 0},
                                       {90, 0, 80},
                                       {-35, -179, 175}}}) {
            Q q = fromEulerZXY(e);
            check(std::abs(dot(q, fromEulerZXY(eulerZXY(q)))) > 0.999999, "Euler ZXY roundtrip");
            check(std::abs(dot(q, quaternion(matrix(q)))) > 0.999999, "Matrix quaternion roundtrip");
            V3 p{0.2, 1, 2};
            check(norm(reflectZ(q).rotate(reflectZ(p)) - reflectZ(q.rotate(p))) < 1e-8,
                  "Handedness reflection");
            check(dot(continuous(-q, q), q) > 0.9999, "Quaternion sign continuity");
        }
        ClockMap clock;
        check(clock.map(1000, 12) == 12, "Clock origin");
        check(clock.map(1033, 12.05) <= 12.05, "Clock arrival bound");
        clock.map(4, 13);
        check(clock.resets == 1, "Clock reset detection");
        clock.map(0xffffffff, 14);
        clock.map(4, 15);
        check(clock.resets == 3, "Clock wrap discontinuity");
        struct T {
            int64_t stamp;
            int id;
        };
        Pairer<T, T> pair;
        pair.addA({0, 1});
        pair.addA({33, 2});
        pair.addB({31, 3});
        auto p = pair.next();
        check(p && p->first.id == 2 && pair.dropped == 1, "Pairing rejects stale RGB");
        check(!pair.next(), "No duplicate pair");
        for (int i = 0; i < 8; ++i)
            pair.addA({i * 33, i});
        check(pair.dropped >= 5, "Bounded unmatched streams");
        BoundedQueue<int> queue(2);
        queue.push(1);
        queue.push(2);
        queue.push(3);
        check(queue.pop(true) == 3 && queue.dropped == 2, "Freshness queue");
        queue.close();
        queue.reset();
        queue.push(9);
        check(queue.pop() == 9, "Queue lifecycle restart");
        Crop c = cropBox(120, 40, 500, 440);
        V2 uv{199.5, 322.5}, round = c.toImage(c.toInput(uv));
        check(std::hypot(round.x - uv.x, round.y - uv.y) < 1e-8, "Inverse crop transform");
        Keypoints k;
        for (int i = 0; i < J; ++i)
            k[i] = {{double(i * 10), double(i)}, 0.9};
        auto mirrored = unmirror(unmirror(k, 640), 640);
        check(mirrored[20].uv.x == k[20].uv.x && mirrored[24].uv.y == 24, "Mirror labels and pixels");
        auto anatomical = kinectImageLabels(k);
        check(anatomical[LAnkle].uv.x == k[RAnkle].uv.x && anatomical[LHeel].uv.x == k[RHeel].uv.x &&
                  anatomical[LToe].uv.x == k[RToe].uv.x && anatomical[LWrist].uv.y == k[RWrist].uv.y &&
                  anatomical[Head].uv.x == k[Head].uv.x,
              "Kinect RGB anatomical conversion changes paired labels without moving registered pixels");
        std::vector<Pair3> samples;
        Rigid truth{fromEulerZXY({12, 65, -8}), {1, 0.2, -1}};
        for (int i = 0; i < 60; ++i) {
            V3 a{std::sin(i * .6) * .6, std::cos(i * .4) * .4 + 1, 2 + std::sin(i * .2) * .5};
            samples.push_back({a, truth.apply(a)});
        }
        auto cal = calibrate(samples);
        check(cal.valid && cal.rms < 1e-8, "Rigid calibration correctness");
        check(norm(cal.transform.apply({1, 2, 3}) - truth.apply({1, 2, 3})) < 1e-7,
              "Rigid orientation convention");
        samples[2].world += V3{1, 0, 0};
        samples[5].world += V3{-1, 0, 1};
        cal = calibrate(samples);
        check(cal.valid && cal.rms < 0.005, "Robust calibration outliers");
        auto noisyCalibration = [&](double noise, int outliers) {
            std::vector<Pair3> observations;
            for (int i = 0; i < 12; ++i) {
                V3 point{std::sin(i * .6) * .6, std::cos(i * .4) * .4 + 1, 2 + std::sin(i * .2) * .5};
                for (auto delta : std::array<V3, 6>{{{noise, 0, 0},
                                                     {-noise, 0, 0},
                                                     {0, noise, 0},
                                                     {0, -noise, 0},
                                                     {0, 0, noise},
                                                     {0, 0, -noise}}})
                    observations.push_back({point, truth.apply(point) + delta});
                for (int n = 0; n < outliers; ++n)
                    observations.push_back({point, truth.apply(point) + V3{n % 2 ? -.4 : .4, 0, 0}});
            }
            return calibrate(observations);
        };
        auto forgiving = noisyCalibration(.05, 2);
        check(forgiving.valid && forgiving.rms > .045 && forgiving.rms < .06,
              "Accept 5 cm noise with 75 percent consistent samples");
        auto inconsistent = noisyCalibration(.04, 4);
        check(!inconsistent.valid && inconsistent.rms < .06 &&
                  inconsistent.reason.find("Too many inconsistent") != std::string::npos,
              "Explain rejected outlier fraction even below RMS limit");
        auto inaccurate = noisyCalibration(.075, 2);
        check(!inaccurate.valid && inaccurate.reason.find("Position mismatch") != std::string::npos,
              "Reject excessive RMS despite sufficient inliers");
        check(forgiving.reason.find("75%") != std::string::npos &&
                  forgiving.reason.find("6.0") != std::string::npos,
              "Calibration result displays both acceptance checks");
        auto heldSession = [&](bool missingRight, bool moving) {
            AlignmentSession session;
            session.reset();
            Settings settings;
            settings.deviceOffsets = {};
            for (int pose = 0; pose < 5; ++pose)
                for (int tick = 0; tick < 45; ++tick) {
                    Frame frame;
                    frame.host = 100 + (pose * 45 + tick) / 30.0;
                    frame.vr.host = frame.host;
                    auto person = body();
                    person.joints[LWrist].p.y += pose * .08;
                    person.joints[RWrist].p.z += pose * .07;
                    if (moving)
                        for (auto &joint : person.joints)
                            joint.p.x += tick * .02;
                    frame.bodies.push_back(person);
                    for (int device = 0; device < 3; ++device) {
                        int joint = device == 0 ? Head : device == 1 ? LWrist : RWrist;
                        frame.vr.devices[device] = {truth.apply(person.joints[joint].p), truth.q,
                                                    !(missingRight && device == 2)};
                    }
                    session.add(frame, person.id, settings);
                }
            return session;
        };
        auto held = heldSession(false, false);
        auto heldResult = held.finish();
        check(heldResult.valid && heldResult.rms < 1e-8 && held.size() > 30,
              "Guided held-pose calibration recovers known transform");
        auto missing = heldSession(true, false).finish();
        check(!missing.valid && missing.reason.find("Right controller") != std::string::npos,
              "Name missing device instead of generic residual error");
        check(heldSession(false, true).size() == 0, "Skip continuously moving calibration observations");
        check(held.agreement(heldResult).find("Headset: median 0.0 cm; 100.0%") != std::string::npos &&
                  held.agreement(heldResult).find("Right controller") != std::string::npos,
              "Report each device including all its samples");
        std::ostringstream alignmentCsv;
        held.writeCsv(alignmentCsv, heldResult);
        check(alignmentCsv.str().find("vr_qw") != std::string::npos &&
                  alignmentCsv.str().find("Left controller") != std::string::npos,
              "Export alignment positions and device rotations for diagnosis");
        for (int i = 0; i < 60; ++i)
            samples[i] = {{double(i) / 10, 0, 0}, {double(i) / 10, 0, 0}};
        check(!calibrate(samples).valid, "Reject collinear calibration");
        std::vector<V3> floorPoints;
        for (int i = 0; i < 300; ++i)
            floorPoints.push_back({double(i % 20) * .1, -1, double(i / 20) * .1});
        auto plane = fitFloor(floorPoints);
        check(plane.valid && std::abs(plane.height({0, -1, 0})) < 1e-7, "Floor geometry");
        FootContact contact;
        for (int i = 0; i < 5; ++i)
            contact.update(.033, .01, {}, true, 0, {});
        check(contact.state == Contact::Planted, "Foot contact hysteresis");
        contact.update(.033, .01, {.3, 0, 0}, true, 0, {});
        check(contact.state == Contact::Sliding, "Foot sliding");
        contact.update(.033, .01, {}, true, .5, {});
        check(contact.state == Contact::Pivot, "Foot pivot");
        contact.update(.033, .12, {0, .3, 0}, true, 0, {});
        check(contact.state == Contact::Air, "Raised foot releases");
        Frame f;
        f.host = 10;
        f.arrival = 10.02;
        f.bodies = {body()};
        f.floor = {{0, 1, 0}, 0, true};
        Estimator est;
        est.select(7);
        auto state = est.process(f, nullptr, {});
        check(state.trackers[0].valid, "SDK baseline produces hip");
        check(state.mode != Mode::Fused, "Baseline never mislabeled advanced");
        auto bytes = oscBundle(state.trackers, {});
        check(!bytes.empty() && bytes[0] == '#' && bytes[15] == 1, "OSC immediate bundle");
        check(bytes.size() < 1400, "OSC single datagram");
        std::array<Tracker, 3> wireTrackers{};
        for (auto &t : wireTrackers) {
            t.valid = true;
            t.p = {1.25, 2.5, 3.75};
            t.q = axisAngle({0, 1, 0}, pi / 2);
        }
        auto packet = oscBundle(wireTrackers, {});
        size_t cursor = 16;
        auto readWord = [&]() {
            if (cursor + 4 > packet.size())
                throw std::runtime_error("Truncated OSC packet");
            uint32_t word = 0;
            for (int k = 0; k < 4; ++k)
                word = (word << 8) | packet[cursor++];
            return word;
        };
        auto readString = [&]() {
            std::string value;
            while (cursor < packet.size() && packet[cursor])
                value += char(packet[cursor++]);
            if (cursor == packet.size())
                throw std::runtime_error("Unterminated OSC string");
            ++cursor;
            while (cursor % 4) {
                if (packet.at(cursor++) != 0)
                    throw std::runtime_error("OSC padding");
            }
            return value;
        };
        for (int i = 0; i < 6; ++i) {
            auto length = readWord();
            size_t end = cursor + length;
            check(readString() ==
                      "/tracking/trackers/" + std::to_string(i / 2 + 1) + (i % 2 ? "/rotation" : "/position"),
                  "OSC stable tracker address");
            check(readString() == ",fff", "OSC float type tags");
            float xyz[3];
            for (auto &v : xyz)
                v = std::bit_cast<float>(readWord());
            check(cursor == end, "OSC element length and padding");
            if (i % 2)
                check(std::abs(xyz[0]) < 1e-5 && std::abs(xyz[1] + 90) < 1e-4 && std::abs(xyz[2]) < 1e-5,
                      "OSC reflected yaw degrees");
            else
                check(xyz[0] == 1.25f && xyz[1] == 2.5f && xyz[2] == -3.75f,
                      "OSC big-endian metric Unity position");
        }
        check(cursor == packet.size(), "OSC exactly six coherent messages");
        wireTrackers[1].valid = false;
        wireTrackers[1].p.x = NAN;
        auto partial = oscBundle(wireTrackers, {});
        std::string partialText(partial.begin(), partial.end());
        check(partial.size() == (packet.size() - 16) * 2 / 3 + 16 &&
                  partialText.find("/tracking/trackers/1/position") != std::string::npos &&
                  partialText.find("/tracking/trackers/3/rotation") != std::string::npos &&
                  partialText.find("/tracking/trackers/2/") == std::string::npos,
              "Lost left foot preserves hip and right foot output with original IDs");
        wireTrackers[0].valid = wireTrackers[2].valid = false;
        check(oscBundle(wireTrackers, {}).empty(), "All missing trackers produce no bundle");
        for (auto &tracker : wireTrackers) {
            tracker.valid = true;
            tracker.p = {1.25, 2.5, 3.75};
        }
        Rigid invalidTransform;
        invalidTransform.t.x = NAN;
        check(oscBundle(wireTrackers, invalidTransform).empty(), "Reject invalid OSC calibration");
        state.trackers[0].p.x = NAN;
        check(oscBundle(state.trackers, {}).empty(), "Reject NaN output");
        check(est.predict(10.4).mode == Mode::Degraded, "Bounded occlusion prediction");
        f.host += .033;
        f.bodies[0].id = 8;
        state = est.process(f, nullptr, {});
        check(state.body.id == 7, "Bystander identity never adopted");
        f.host += .4;
        state = est.process(f, nullptr, {});
        check(!state.trackers[0].valid, "Stale joint invalidation");
        f.host += .033;
        f.bodies = {body()};
        state = est.process(f, nullptr, {});
        check(state.trackers[0].valid && finite(state.trackers[0].p), "Reacquisition finite");
        RotationEvidence rotation;
        rotation.update({}, true, 30, 0, .65, 8, .045);
        Q reversed = axisAngle({0, 1, 0}, pi);
        check(!rotation.update(reversed, true, 30.033, .1, .65, 8, .045) &&
                  std::abs(dot(rotation.value, Q{})) > .999,
              "Isolated reversed foot evidence is held for confirmation");
        for (int i = 2; i <= 35; ++i)
            rotation.update(reversed, true, 30 + i / 30.0, .1, .65, 8, .045);
        check(std::abs(dot(rotation.value, reversed)) > .999,
              "Sustained half turn recovers instead of permanent rotation rejection");
        auto heldRotation = rotation.value;
        check(!rotation.update({}, false, 31.2, .1, .65, 8, .045) &&
                  std::abs(dot(rotation.value, heldRotation)) > .99999,
              "Unobserved direction is held and marked uncertain");
        for (int i = 0; i < 20; ++i)
            rotation.update(i % 2 ? Q{} : reversed, true, 31.24 + i / 30.0, .1, .65, 8, .045);
        check(!rotation.trusted && std::abs(dot(rotation.value, heldRotation)) > .99999,
              "Alternating toe directions cannot pass reacquisition dwell");
        Estimator feet;
        feet.settings.constraints = false;
        feet.select(7);
        Frame footFrame;
        footFrame.host = 35;
        footFrame.bodies = {body()};
        for (int joint : {LHeel, RHeel, LToe, RToe}) {
            footFrame.bodies[0].joints[joint].confidence = .9;
            footFrame.bodies[0].joints[joint].source = 2;
        }
        auto footStep = [&] {
            footFrame.host += 1.0 / 30;
            return feet.process(footFrame, nullptr, {});
        };
        auto acquiringFoot = footStep();
        check(acquiringFoot.trackers[1].angularSigma.y == pi,
              "First heel/toe observation remains uncertain until confirmed");
        for (int i = 0; i < 20; ++i)
            state = footStep();
        auto stableFoot = state.trackers[1].q;
        check(state.trackers[1].angularSigma.y < .5 && state.trackers[1].valid,
              "Consistent plausible heel/toe geometry acquires foot orientation");
        check(dot(reflectZ(state.trackers[1].q).rotate({0, 0, 1}),
                  reflectZ(unit(footFrame.bodies[0].joints[LToe].p - footFrame.bodies[0].joints[LHeel].p))) >
                  .999,
              "Exported Unity foot forward points toward the toes");
        check(std::abs(dot(state.trackers[1].q, state.trackers[0].q)) > .999,
              "Forward-facing foot observation and pelvis fallback share the same local axes");
        std::swap(footFrame.bodies[0].joints[LHeel].p, footFrame.bodies[0].joints[LToe].p);
        auto toeOutlier = footStep();
        check(std::abs(dot(toeOutlier.trackers[1].q, stableFoot)) > .99,
              "Single reversed heel/toe frame does not reverse output foot");
        for (int i = 0; i < 40; ++i)
            state = footStep();
        check(std::abs(dot(state.trackers[1].q, stableFoot)) < .1 && state.trackers[1].angularSigma.y < .5,
              "Confirmed new foot direction recovers through production geometry gate");
        footFrame.bodies[0].joints[LToe].confidence = 0;
        for (int i = 0; i < 6; ++i)
            state = footStep();
        check(state.trackers[1].angularSigma.y == pi && state.trackers[0].valid,
              "Missing toe raises yaw uncertainty without discarding a visible hip");
        Estimator turning;
        turning.settings.constraints = false;
        turning.select(7);
        Frame turn;
        turn.host = 40;
        turn.bodies = {body()};
        auto facing = turning.process(turn, nullptr, {});
        auto turnedBody = body();
        for (auto &joint : turnedBody.joints)
            joint.p = V3{0, 1, 2} + reversed.rotate(joint.p - V3{0, 1, 2});
        turn.bodies = {turnedBody};
        turn.host += 1.0 / 30;
        auto firstTurn = turning.process(turn, nullptr, {});
        check(std::abs(dot(firstTurn.trackers[0].q, facing.trackers[0].q)) > .99,
              "Single bilateral label reversal does not flip the pelvis");
        for (int i = 2; i <= 65; ++i) {
            turn.host = 40 + i / 30.0;
            state = turning.process(turn, nullptr, {});
        }
        check(std::abs(dot(state.trackers[0].q, reversed * facing.trackers[0].q)) > .99 &&
                  state.trackers[0].valid && !state.ambiguous,
              "Production estimator escapes both label and pelvis half-turn latches");
        Estimator identity;
        identity.select(7);
        identity.settings.deviceOffsets = {};
        Calibration identityCal;
        identityCal.valid = true;
        Frame identityFrame;
        identityFrame.host = identityFrame.vr.host = 50;
        auto newBody = body();
        newBody.id = 99;
        identityFrame.bodies = {newBody};
        for (int device = 0; device < 3; ++device)
            identityFrame.vr.devices[device] = {
                newBody.joints[device == 0   ? Head
                               : device == 1 ? LWrist
                                             : RWrist]
                    .p,
                {},
                true};
        check(identity.reconcileIdentity(identityFrame, identityCal) == 7,
              "Identity recovery requires dwell rather than one matching frame");
        for (int i = 1; i <= 12; ++i) {
            identityFrame.host = identityFrame.vr.host = 50 + i / 30.0;
            identity.reconcileIdentity(identityFrame, identityCal);
        }
        check(identity.state().body.id == 99, "Sustained unique three-device match recovers a new SDK ID");
        for (int scenario = 0; scenario < 5; ++scenario) {
            identity.select(7);
            auto rejectedFrame = identityFrame;
            if (scenario == 0)
                rejectedFrame.vr.devices[2].valid = false;
            if (scenario == 1) {
                auto bystander = newBody;
                bystander.id = 101;
                rejectedFrame.bodies.push_back(bystander);
            }
            if (scenario == 2)
                rejectedFrame.bodies.push_back(body());
            if (scenario == 3)
                rejectedFrame.vr.devices[0].p.x += .5;
            auto rejectedCal = identityCal;
            if (scenario == 4)
                rejectedCal.valid = false;
            for (int i = 0; i < 20; ++i) {
                rejectedFrame.host = rejectedFrame.vr.host = 60 + i / 30.0;
                identity.reconcileIdentity(rejectedFrame, rejectedCal);
            }
            check(identity.state().body.id == 7, "Refuse identity recovery with missing devices, ambiguity, "
                                                 "existing lock, mismatch or no alignment");
        }
        Frame depth;
        depth.mapping.resize(20);
        depth.depth.resize(20, 16001);
        for (auto &m : depth.mapping)
            m = {0, 1, 2, 100, 100};
        auto d = associateDepth(depth, {{100, 100}, .9}, {{0, 1, 2.045}, .8, .03, 1}, LKnee, 1);
        check(d && d->p.z > 2, "Surface not anatomical center");
        check(!associateDepth(depth, {{100, 100}, .9}, {{0, 1, 3}, .8, .03, 1}, LKnee, 1),
              "Occluded depth rejected");
        for (int i = 0; i < 10; ++i)
            depth.mapping[i].z = 3;
        check(!associateDepth(depth, {{100, 100}, .9}, {}, LKnee, 1), "Depth edge rejected");
        Frame footDepth;
        for (int i = 0; i < 20; ++i) {
            footDepth.mapping.push_back({0, 0, i < 8 ? 2.0f : 2.2f, 100, 100});
            footDepth.depth.push_back(std::uint16_t((i < 8 ? 16000 : 17600) | (i < 8 ? 1 : 0)));
        }
        auto shoe = associateDepth(footDepth, {{100, 100}, .9}, {}, LToe, 1);
        check(shoe && shoe->p.z < 2.05, "Foot depth uses the selected shoe instead of surrounding carpet");
        for (auto &word : footDepth.depth)
            word &= ~7;
        check(!associateDepth(footDepth, {{100, 100}, .9}, {}, LToe, 1),
              "Unlabelled floor alone cannot produce a measured toe");
        Frame mirroredFrame;
        Keypoints modelPoints{};
        modelPoints[LAnkle] = {{360, 200}, .9};
        modelPoints[RAnkle] = {{280, 200}, .9};
        for (int i = 0; i < 16; ++i) {
            bool left = i < 8;
            mirroredFrame.mapping.push_back({left ? -.15f : .15f, 0, 2, left ? 280 : 360, 200});
            mirroredFrame.depth.push_back(16001);
        }
        auto converted = kinectImageLabels(modelPoints);
        auto leftAnkle = associateDepth(mirroredFrame, converted[LAnkle], {}, LAnkle, 1);
        check(leftAnkle && leftAnkle->p.x < -.14,
              "Mirrored RGB left ankle associates with the SDK left leg, not the opposite foot");
        PoseHistory history;
        VrSample a;
        a.host = 1;
        a.devices[0] = {{0, 0, 0}, {}, true};
        VrSample b = a;
        b.host = 1.02;
        b.devices[0].p = {2, 0, 0};
        history.add(a);
        history.add(b);
        check(std::abs(history.at(1.01)->devices[0].p.x - 1) < 1e-8, "Capture-time pose interpolation");
        check(!history.at(.9), "No arbitrary VR pose extrapolation");
        auto path = std::filesystem::temp_directory_path() / "kinect_rgbd_unit_test.kfr";
        RecordingWriter writer;
        writer.open(path, "synthetic unit fixture, not sensor evidence");
        auto runConfig = std::make_shared<ReplayConfig>();
        runConfig->selectedId = 7;
        runConfig->settings.soleOffset = .08;
        runConfig->calibration.transform.t = {.1, .2, .3};
        runConfig->calibration.valid=true;
        VrSample origin;origin.epoch=3;origin.rawTransformValid=true;
        origin.referenceSource=42;origin.referenceUniverse=19;origin.referenceEvents=VrStandingReset;
        origin.standingToRaw={axisAngle({0,1,0},.25),{.1,0,.2}};
        bindTrackingReference(runConfig->calibration,origin);
        f.vr=origin;f.vr.standingToRaw.t.x+=1.;
        f.vr.devices[1]={{-.5,1,2},{},true};
        const auto liveController=calibrationVr(f.vr,runConfig->calibration).devices[1].p;
        runConfig->lengths[0]=.22;runConfig->lengthsCaptured=true;
        f.enqueuedHost=12345;
        f.runConfig = runConfig;
        f.rgbId = 42;
        writer.write(f);
        f.host += .2;
        f.rgbId = 48;
        writer.write(f);
        writer.close();
        RecordingReader reader;
        reader.open(path);
        auto first = reader.next(), second = reader.next();
        check(first->vr.referenceSource==0 && first->runConfig->calibration.referenceSource==0 &&
              first->vr.referenceEvents==0,"Recording persisted live identity metadata inconsistently");
        check(first->vr.rawTransformValid && first->runConfig->calibration.rawReferenceValid &&
                  first->runConfig->calibration.rawEpoch==3 && first->runConfig->lengthsCaptured &&
                  first->runConfig->lengths[0]==.22 && first->enqueuedHost==0 &&
                  norm(calibrationVr(first->vr,first->runConfig->calibration).devices[1].p-liveController)<1e-10,
              "Replay lost physical reference, OVR movement or captured proportions");
        check(first->runConfig && first->runConfig->selectedId == 7 &&
                  first->runConfig->settings.soleOffset == .08 &&
                  first->runConfig->calibration.transform.t.z == .3,
              "Replay restores configuration and metric alignment");
        check(first->rgbId == 42 && second->rgbId == 48 && std::abs(second->host - first->host - .2) < 1e-8,
              "Replay preserves frame gaps and clock");
        check(!reader.next(), "Replay end of file");
        {
            std::fstream corrupt(path, std::ios::binary | std::ios::in | std::ios::out);
            corrupt.seekp(-1, std::ios::end);
            char byte = 127;
            corrupt.write(&byte, 1);
        }
        bool rejected = false;
        try {
            reader.open(path);
            reader.next();
            reader.next();
        } catch (...) {
            rejected = true;
        }
        check(rejected, "Recording corruption rejected");
        reader.close();
        std::filesystem::remove(path);
        Frame projected;
        auto temporal = body(), swapped = temporal;
        for (auto [a, b] :
             std::array<std::pair<int, int>, 3>{{{LHip, RHip}, {LKnee, RKnee}, {LAnkle, RAnkle}}})
            std::swap(swapped.joints[a], swapped.joints[b]);
        check(stabilizeLegLabels(swapped, temporal) &&
                  norm(swapped.joints[LAnkle].p - temporal.joints[LAnkle].p) < 1e-9,
              "Bilateral label swap rejected by temporal association");
        auto crossing = temporal;
        std::swap(crossing.joints[LAnkle], crossing.joints[RAnkle]);
        check(!stabilizeLegLabels(crossing, temporal),
              "Isolated leg crossing does not flip whole-leg identity");
        for (int i = 0; i < 2400; ++i) {
            double z = 1.1 + (i % 17) * .12, x = std::sin(i * .7) * .45, y = std::cos(i * .31) * .35;
            projected.mapping.push_back(
                {float(x), float(y), float(z), int(320 - 530 * x / z), int(240 - 530 * y / z)});
        }
        auto camera = fitColorProjection(projected);
        check(camera.valid && camera.rms < 1, "Color projection recovers SDK-like registration");
        V3 point{.2, .15, 2};
        auto before = camera.project(point);
        V2 target{before.x + 8, before.y - 6};
        auto after = camera.project(point + camera.correction(point, target));
        check(std::hypot(after.x - target.x, after.y - target.y) < 10,
              "RGB residual reduces reprojection error");
        Frame occluder;
        occluder.mapping = {{0, 0, 3, 80, 80}, {0, 0, 1, 80, 80}};
        indexRegistration(occluder);
        check(occluder.colorIndex[80 * 640 + 80] == 1, "Registration z-buffer picks visible surface");
        Frame boneFrame;
        boneFrame.bodies = {body()};
        boneFrame.host = 20;
        Estimator boneEstimator;
        boneEstimator.select(7);
        boneEstimator.process(boneFrame, nullptr, {});
        boneFrame.host += .033;
        boneFrame.bodies[0].joints[LKnee].p.x += .16;
        auto constrained = boneEstimator.process(boneFrame, nullptr, {});
        double rawLength = norm(boneFrame.bodies[0].joints[LKnee].p - boneFrame.bodies[0].joints[LHip].p);
        double fitLength = norm(constrained.body.joints[LKnee].p - constrained.body.joints[LHip].p);
        check(std::abs(fitLength - .45) < std::abs(rawLength - .45),
              "Articulated fit reduces bone-length violation");
        Estimator rawEstimator;
        rawEstimator.settings.baseline = 1;
        rawEstimator.select(7);
        auto rawState = rawEstimator.process(boneFrame, nullptr, {});
        check(norm(rawState.body.joints[Hip].p - boneFrame.bodies[0].joints[Hip].p) < 1e-9,
              "Raw baseline preserves SDK hip");
        auto calibrationPath = std::filesystem::temp_directory_path() / "kf_bad_calibration.txt";
        {
            std::ofstream invalid(calibrationPath);
            invalid << "KF_CALIBRATION_1\n1 nan 0 0 0 0 0 0 0 0 1\n";
        }
        Calibration invalid;
        Settings cfg;
        check(!loadCalibration(calibrationPath, invalid, cfg), "Corrupt calibration rejected");
        std::filesystem::remove(calibrationPath);
        std::cout << count << " invariant checks passed. Synthetic tests are not live tracking validation.\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "CHECK FAILED after " << count << ": " << e.what() << '\n';
        return 1;
    }
}
