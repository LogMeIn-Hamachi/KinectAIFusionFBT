#include "core.hpp"
#include "body_tracker.hpp"
namespace kf {
V3 LearnedPositionFilter::update(V3 observation,double time,double restTau) {
    if(!finite(observation) || !std::isfinite(time))return value;
    double dt=time-host;
    if(!initialized || dt>.15) {
        value=previousObservation=observation;trend={};activity=0;host=time;initialized=true;resting=true;
        return value;
    }
    if(dt<=0)return value;
    V3 derivative=(observation-previousObservation)/dt;
    double a=1-std::exp(-dt/.18);
    trend=lerp(trend,derivative,a);activity+=(norm(derivative)-activity)*a;
    previousObservation=observation;host=time;
    double coherence=norm(trend)/std::max(activity,1e-9);
    double speed=norm(trend)*std::clamp((coherence-.20)/.4,0.,1.);
    double tau=std::clamp(.14/(1+10*std::max(0.,speed-.10))+
        (restTau-.14)*std::clamp(1-speed/.15,0.,1.),.015,restTau);
    resting=speed<.10;
    V3 innovation=observation-value;
    double length=norm(innovation);
    // Soft 4 mm deadband, reduced to 1 mm during coherent motion.
    double deadband=resting?.004:.001;
    if(length>deadband)value+=innovation*((1-std::exp(-dt/tau))*(1-deadband/length));
    return value;
}
bool RotationEvidence::update(Q observation, bool visible, double host, double dwell, double jump,
                              double rate, double tau,bool adaptive) {
    double dt = lastHost > 0 ? host - lastHost : 1.0 / 30;
    if (dt <= 0)
        return false;
    if (dt > .12) {
        trusted = false;
        waiting = false;
        motionInitialized=false;
    }
    lastHost = host;
    if (!visible || !std::isfinite(dot(observation, observation))) {
        trusted = false;
        waiting = false;
        motionInitialized=false;
        return false;
    }
    observation = normalized(observation);
    if(adaptive) {
        if(motionInitialized) {
            Q delta=continuous(observation,motionReference)*motionReference.conjugate();
            V3 vector{delta.x,delta.y,delta.z};
            double angle=2*std::atan2(norm(vector),std::max(0.,delta.w));
            V3 velocity=norm(vector)>1e-9?unit(vector)*(angle/dt):V3{};
            double a=1-std::exp(-dt/.18);
            motionTrend=lerp(motionTrend,velocity,a);motionActivity+=(norm(velocity)-motionActivity)*a;
        } else {motionTrend={};motionActivity=0;motionInitialized=true;}
        motionReference=observation;
        double coherence=norm(motionTrend)/std::max(motionActivity,1e-9);
        double speed=norm(motionTrend)*std::clamp((coherence-.20)/.4,0.,1.);
        // At full speed retain the previous rotation response; do not make a
        // deliberate pivot sharper merely because adaptive smoothing is enabled.
        tau=std::clamp(.14/(1+3*std::max(0.,speed-.35)),std::max(.015,tau),.14);
    } else motionInitialized=false;
    bool discontinuity =
        initialized && angleBetween(value, observation) > jump && angleBetween(reference, observation) > jump;
    if (!trusted || discontinuity || waiting) {
        if (!waiting || angleBetween(pending, observation) > .35) {
            pending = observation;
            since = host;
            waiting = true;
        }
        if (host - since + 1e-9 < dwell)
            return false;
        waiting = false;
    }
    trusted = true;
    reference = observation;
    if (!initialized) {
        value = observation;
        initialized = true;
    } else {
        observation = continuous(observation, value);
        double angle = angleBetween(value, observation);
        double fraction = std::min(1 - std::exp(-std::min(dt, .05) / tau),
                                   rate * std::min(dt, .05) / std::max(angle, 1e-9));
        if(adaptive)fraction*=std::max(0.,1-.018/std::max(angle,1e-9)); // roughly one degree, never a floor lock
        // Spherical interpolation gives a real angular-speed bound, including near 180 degrees.
        double half = angle / 2, denominator = std::sin(half);
        if (denominator > 1e-6) {
            double a = std::sin((1 - fraction) * half) / denominator;
            double b = std::sin(fraction * half) / denominator;
            value = normalized({value.w * a + observation.w * b, value.x * a + observation.x * b,
                                value.y * a + observation.y * b, value.z * a + observation.z * b});
        } else
            value = blend(value, observation, fraction);
    }
    return true;
}
std::uint32_t Estimator::reconcileIdentity(const Frame &frame, const Calibration &cal) {
    const auto vr=calibrationVr(frame.vr,cal);
    auto reset = [&] {
        rebindCandidate_ = 0;
        rebindSince_ = rebindLast_ = 0;
    };
    if (!selected_ || !cal.valid || std::abs(vr.host - frame.host) > .025 ||
        std::any_of(frame.bodies.begin(), frame.bodies.end(),
                    [&](const Body &b) { return b.id == selected_; })) {
        reset();
        return selected_;
    }
    std::uint32_t candidate = 0;
    for (const auto &body : frame.bodies) {
        bool matches = true;
        for (int device = 0; device < 3; ++device) {
            const auto &pose = vr.devices[device];
            const auto &joint = body.joints[device == 0 ? Head : device == 1 ? LWrist : RWrist];
            V3 target = pose.p + (device==0?settings.deviceOffsets[device]:pose.q.rotate(settings.deviceOffsets[device]));
            double distance = norm(cal.transform.apply(joint.p) - target);
            if (!pose.valid || joint.confidence < .7 || !std::isfinite(distance) ||
                distance > (device == 0 ? .22 : .28))
                matches = false;
        }
        if (matches && body.id) {
            if (candidate) { // Two plausible people: do not guess.
                reset();
                return selected_;
            }
            candidate = body.id;
        }
    }
    if (!candidate) {
        reset();
        return selected_;
    }
    if (candidate != rebindCandidate_ || frame.host - rebindLast_ > .12 || frame.host <= rebindLast_) {
        rebindCandidate_ = candidate;
        rebindSince_ = frame.host;
    }
    rebindLast_ = frame.host;
    if (frame.host - rebindSince_ >= .3) {
        selected_ = candidate;
        state_.body.id = candidate;
        velocity_ = {}; // Preserve proportions, but never carry occlusion velocity into a new ID.
        angularVelocity_ = {};
        previousRotation_ = {};
        learnedFilters_={};
        sourceTransitions_={};
        soleCorrections_={};
        samSeen_={};plants_={};
        reset();
    }
    return selected_;
}
bool stabilizeLegLabels(Body &observations, const Body &predicted) {
    constexpr std::array<std::pair<int, int>, 6> pairs{{{LHip, RHip},
                                                        {LKnee, RKnee},
                                                        {LAnkle, RAnkle},
                                                        {LHeel, RHeel},
                                                        {LToe, RToe},
                                                        {LSmallToe, RSmallToe}}};
    double direct = 0, swapped = 0;
    unsigned good = 0;
    for (auto [a, b] : pairs) {
        if (observations.joints[a].confidence < .4 || observations.joints[b].confidence < .4 ||
            predicted.joints[a].confidence < .3 || predicted.joints[b].confidence < .3)
            continue;
        direct += norm(observations.joints[a].p - predicted.joints[a].p) +
                  norm(observations.joints[b].p - predicted.joints[b].p);
        swapped += norm(observations.joints[b].p - predicted.joints[a].p) +
                   norm(observations.joints[a].p - predicted.joints[b].p);
        ++good;
    }
    // Require strong bilateral evidence across the whole leg, not an isolated foot crossing.
    if (good < 3 || direct < .45 || swapped > .35 * direct || swapped / good > .10)
        return false;
    for (auto [a, b] : pairs)
        std::swap(observations.joints[a], observations.joints[b]);
    return true;
}
bool Estimator::calibrateLengths(std::span<const Body> samples) {
    if (samples.size() < 20)
        return false;
    auto next = lengths_;
    unsigned good = 0;
    for (size_t i = 0; i < bones.size(); ++i) {
        auto [a, b] = bones[i];
        std::vector<double> values;
        for (auto &body : samples)
            if (body.joints[a].confidence > .65 && body.joints[b].confidence > .65) {
                double d = norm(body.joints[a].p - body.joints[b].p);
                if (d > .035 && d < .85)
                    values.push_back(d);
            }
        if (values.size() < samples.size() / 2)
            continue;
        std::sort(values.begin(), values.end());
        if (values[values.size() * 9 / 10] - values[values.size() / 10] > .06)
            continue;
        next[i] = values[values.size() / 2];
        ++good;
    }
    if (good < 10)
        return false;
    lengths_ = next;
    return true;
}
void Estimator::select(std::uint32_t id) {
    selected_ = id;
    state_ = {};
    state_.body.id = id;
    velocity_ = {};
    angularVelocity_ = {};
    previousRotation_ = {};
    seen_ = {};
    lengths_ = {};
    initialized_ = 0;
    rotations_ = {};
    learnedFilters_={};
    sourceTransitions_={};
    soleCorrections_={};
        samSeen_={};plants_={};
    labelSwapSince_ = rebindSince_ = rebindLast_ = 0;
    rebindCandidate_ = 0;
}
bool Estimator::calibrateLengths(const Body &b) {
    auto next = lengths_;
    for (size_t i = 0; i < bones.size(); ++i) {
        auto [a, c] = bones[i];
        if (b.joints[a].confidence < 0.65 || b.joints[c].confidence < 0.65)
            continue;
        double d = norm(b.joints[a].p - b.joints[c].p);
        if (d < 0.035 || d > 0.85)
            return false;
        next[i] = d;
    }
    lengths_ = next;
    return true;
}
State Estimator::process(const Frame &f, const Keypoints *rgb, const Calibration &cal,const PosePrior *learned) {
    const auto vr=calibrationVr(f.vr,cal);
    if(learned && (!learned->valid || learned->selectedId!=selected_ ||
       (learned->bodyFitted?(!learned->articulationValid || !std::isfinite(learned->rootSigma) || learned->rootSigma<0 || learned->rootSigma>.5 || !std::isfinite(learned->depthAge) || learned->depthAge<0):learned->depthPredicted?(!learned->articulationValid || !std::isfinite(learned->depthAge) || learned->depthAge<=0 || learned->depthAge>.4 || learned->rootAnchors!=0):learned->rootAnchors<3) ||
       !std::isfinite(learned->host) || !std::isfinite(learned->registrationRms) || learned->registrationRms<0 ||
       std::abs(learned->host-f.host)>.001 || settings.baseline!=0 || !settings.inference || !settings.depth)) learned=nullptr;
    double start = now();
    double dt = state_.host > 0 ? std::clamp(f.host - state_.host, 0.001, 0.15) : 1.0 / 30;
    if (state_.host > 0 && f.host <= state_.host)
        return state_;
    state_.learnedDirection={};
    state_.learnedPosition={};
    state_.nativeFootDirection={};
    const Body *raw = nullptr;
    for (auto &b : f.bodies)
        if (b.id == selected_)
            raw = &b;
    if(!raw)learned=nullptr;
    if (!selected_) {
        state_.mode = Mode::Lost;
        state_.host = f.host;
        return state_;
    }
    if (!initialized_ && !raw) {
        state_.mode = Mode::Lost;
        state_.host = f.host;
        return state_;
    }
    if (!initialized_ && raw) {
        state_.body = *raw;
        if (std::none_of(lengths_.begin(), lengths_.end(), [](double d) { return d > 0; }))
            calibrateLengths(*raw);
        ++initialized_;
    }
    auto previous = state_;
    ColorProjection projection;
    if (rgb && settings.baseline == 0)
        projection = fitColorProjection(f);
    std::array<Joint, J> measurements{};
    state_.depthSupport = {};
    unsigned measured = 0, advanced = 0;
    Body evidence;
    Body predicted = state_.body;
    for (int j = 0; j < J; ++j) {
        predicted.joints[j].p += velocity_[j] * std::min(dt, .05);
        Joint observation = raw ? raw->joints[j] : Joint{};
        if(learned && learned->available[j]) {
            // Model owns articulation once its root has selected-player metric
            // support. Do not pull it back onto SDK joints or skin-surface depth.
            double sigma=learned->bodyFitted?(learned->supported[j]?.04:.12)+learned->rootSigma:(learned->supported[j]?.08:.18)+learned->registrationRms;
            observation={learned->points[j],learned->supported[j]?.7:.45,sigma,5};
            samSeen_[j]=f.host;
        } else if (settings.baseline == 0 && rgb && settings.depth) {
            auto depth =
                associateDepth(f, (*rgb)[j], predicted.joints[j], j, raw ? raw->player : state_.body.player,
                               settings.surfaceRadius, &state_.depthSupport[j]);
            if (depth && (observation.confidence < .7 ||
                          ((*rgb)[j].score > .55 && norm(observation.p - depth->p) < .16)))
                observation = *depth;
        }
        // A briefly missing image pose is not evidence that SDK's conflicting
        // articulation has become correct. Hold this joint, then blend fallback.
        if(observation.source!=5 && samSeen_[j]>0 && f.host-samSeen_[j]<.22)observation.confidence=0;
        evidence.joints[j] = observation;
    }
    Body originalEvidence = evidence;
    bool correctedLabels =
        !learned && settings.baseline == 0 && previous.host > 0 && dt < .1 && stabilizeLegLabels(evidence, predicted);
    if (correctedLabels) {
        if (!labelSwapSince_)
            labelSwapSince_ = f.host;
        // A temporal label prior is not proof of facing direction. Persistent new observations
        // must eventually replace it, or a real half-turn can be "corrected" forever.
        if (f.host - labelSwapSince_ >= .25) {
            evidence = originalEvidence;
            correctedLabels = false;
        }
    } else
        labelSwapSince_ = 0;
    for (int j = 0; j < J; ++j) {
        Joint prior = state_.body.joints[j];
        prior.p += velocity_[j] * std::min(dt, 0.05);
        Joint obs = evidence.joints[j];
        if (obs.confidence > 0 && (!finite(obs.p) || obs.p.z < 0.4 || obs.p.z > 5))
            obs.confidence = 0;
        // SDK and registered depth share sensor noise: choose one metric observation, not two independent
        // votes.
        if (obs.source == 2 || obs.source == 5)
            ++advanced;
        if (!(learned && learned->bodyFitted) && settings.baseline == 0 && settings.vrConstraints && cal.valid &&
            std::abs(vr.host - f.host) < 0.025) {
            int device = j == Head ? 0 : j == LWrist ? 1 : j == RWrist ? 2 : -1;
            if (device >= 0 && vr.devices[device].valid) {
                auto d = vr.devices[device];
                V3 p = device==0?cal.transform.inverse(d.p)+
                    (rotations_[0].initialized?rotations_[0].value:cal.transform.q.conjugate()).rotate(settings.deviceOffsets[device]):
                    cal.transform.inverse(d.p + d.q.rotate(settings.deviceOffsets[device]));
                if (norm(p - prior.p) < 0.4) {
                    obs = {p, 0.95, 0.02, 3};
                }
            }
        }
        double jump = norm(obs.p - prior.p);
        if (seen_[j] > 0 && f.host - seen_[j] < 0.2 && jump > 0.65)
            obs.confidence = 0;
        if (obs.confidence >= 0.25) {
            ++measured;
            seen_[j] = f.host;
            measurements[j] = obs;
            double alpha = settings.baseline == 1   ? 1.0
                           : settings.baseline == 2 ? 1 - std::exp(-dt / 0.055)
                                                    : std::clamp(dt / (dt + obs.sigma * 0.2), 0.35, 0.95);
            if (previous.body.joints[j].confidence <= 0.1)
                alpha = 1;
            if(obs.source==5) {
                if(!learned->bodyFitted)obs.p=learnedFilters_[j].update(obs.p,f.host);
                alpha=1;
            } else {
                learnedFilters_[j]={};
            }
            state_.body.joints[j] = obs;
            state_.body.joints[j].p = lerp(prior.p, obs.p, alpha);
        } else {
            double age = seen_[j] > 0 ? f.host - seen_[j] : 100;
            auto &out = state_.body.joints[j];
            out.p += velocity_[j] * (age <= 0.12 ? dt : 0);
            out.sigma = std::min(2.0, out.sigma + dt * 0.8);
            out.confidence = age <= outputHoldSeconds ? std::max(0.0, out.confidence - dt * .5) : 0;
            out.source = 4;
            velocity_[j] = velocity_[j] * std::exp(-dt * 8);
        }
    }
    // Alternate robust measurement proximal steps and articulated length projections, strictly past-only.
    if (settings.constraints && settings.baseline == 0) {
        for (int iter = 0; iter < 6; ++iter) {
            for (int j = 0; j < J; ++j) {
                auto &out = state_.body.joints[j];
                const auto &m = measurements[j];
                if (m.confidence > 0.25 && m.source!=5) {
                    V3 residual = m.p - out.p;
                    double weight = std::min(1.0, 0.06 / std::max(1e-9, norm(residual)));
                    out.p += residual * (0.22 * weight);
                }
                if (out.source!=5 && rgb && !correctedLabels && projection.valid && (*rgb)[j].score > .5 && out.p.z > .8 &&
                    out.p.z < 4) {
                    // RGB constrains the image plane even with an invalid local depth sample.
                    // Position uncertainty and missing-depth age are intentionally not reset here.
                    out.p += projection.correction(out.p, (*rgb)[j].uv) * .18;
                }
            }
            for (size_t i = 0; i < bones.size(); ++i) {
                if (lengths_[i] <= 0)
                    continue;
                auto [a, b] = bones[i];
                auto &x = state_.body.joints[a];
                auto &y = state_.body.joints[b];
                // SAM's own articulated body is already coherent. SDK segment
                // definitions/proportions must not reshape it back into SDK pose.
                if(x.source==5 && y.source==5)continue;
                if (x.confidence < 0.05 || y.confidence < 0.05)
                    continue;
                V3 d = y.p - x.p;
                double l = norm(d);
                if (l < 1e-6)
                    continue;
                V3 correction = bounded(d * ((l - lengths_[i]) / l), 0.04);
                double sx = std::max(x.sigma, 0.015), sy = std::max(y.sigma, 0.015);
                x.p += correction * (sx / (sx + sy) * 0.65);
                y.p += correction * (-sy / (sx + sy) * 0.65);
            }
        }
    }
    for (int j = 0; j < J; ++j) {
        auto &out = state_.body.joints[j];
        if(settings.baseline==0 && smoothSourceChanges_)
            out.p=sourceTransitions_[j].update(out.p,previous.body.joints[j].p,out.source,f.host);
        if (!finite(out.p)) {
            out = previous.body.joints[j];
            out.confidence = 0;
        }
        if (measurements[j].confidence > 0.25)
            velocity_[j] = lerp(velocity_[j], bounded((out.p - previous.body.joints[j].p) / dt, 5), 0.4);
        if(out.source==5 && ((!(learned && learned->bodyFitted) && learnedFilters_[j].resting) || previous.body.joints[j].source!=5))
            velocity_[j]={}; // Do not extrapolate measurement wobble into fresh OSC positions.
        if(learned && learned->bodyFitted) {
            double speed = norm(velocity_[j]);
            double deadband = learned->resting[j] ? 0.035 : 0.015;
            if (speed <= deadband) {
                velocity_[j] = {};
            } else if (speed < deadband + 0.035) {
                velocity_[j] *= (speed - deadband) / 0.035;
            }
        }
    }
    if (raw)
        state_.body.player = raw->player;
    auto &b = state_.body.joints;
    if (!(learned && learned->bodyFitted) && settings.baseline == 0 && b[LHip].confidence > 0.3 && b[RHip].confidence > 0.3) {
        b[Hip].p = (b[LHip].p + b[RHip].p) / 2;
        b[Hip].confidence = std::min(b[LHip].confidence, b[RHip].confidence);
        if(b[LHip].source==5 && b[RHip].source==5) {
            b[Hip].source=5;b[Hip].sigma=std::max(b[LHip].sigma,b[RHip].sigma);
        }
        seen_[Hip] = std::min(seen_[LHip], seen_[RHip]);
    }
    if(!(learned && learned->bodyFitted))velocity_[Hip] = (velocity_[LHip] + velocity_[RHip]) / 2;
    V3 up = unit(b[Neck].p - b[Hip].p), right = unit(b[RHip].p - b[LHip].p, {1, 0, 0});
    bool pelvisEvidence = !correctedLabels && measurements[LHip].confidence > .4 &&
                          measurements[RHip].confidence > .4 && measurements[Neck].confidence > .4 &&
                          norm(b[RHip].p - b[LHip].p) > .12 && norm(b[Neck].p - b[Hip].p) > .15 &&
                          norm(cross(right, up)) > .4;
    Q pelvisObservation=basis(right,up);
    bool learnedPelvis=learned && learned->available[LHip] && learned->available[RHip] &&
                       learned->available[Neck] && measurements[Hip].confidence>.4;
    if(learnedPelvis) {
        auto r=learned->points[RHip]-learned->points[LHip];
        auto u=learned->points[Neck]-learned->points[Hip];
        learnedPelvis=norm(r)>.12 && norm(u)>.15 && norm(cross(unit(r),unit(u)))>.4;
        if(learnedPelvis){pelvisObservation=basis(r,u);pelvisEvidence=true;}
    }
    bool pelvisAccepted = rotations_[0].update(pelvisObservation, pelvisEvidence, f.host,
                                               previous.host > 0 ? .20 : 0, 1.0, 10, .025,learnedPelvis);
    Q pelvis = rotations_[0].value;
    state_.ambiguous = !pelvisAccepted;
    auto &hip = state_.trackers[0];
    hip.q = continuous(pelvis, hip.q);
    hip.p = b[Hip].p + hip.q.rotate(settings.trackerOffsets[0]);
    hip.velocity = velocity_[Hip];
    hip.positionSigma = b[Hip].sigma;
    hip.angularSigma = state_.ambiguous ? V3{0.5, pi, 0.5} : V3{0.25, 0.4, 0.25};
    state_.learnedDirection[0]=pelvisAccepted&&learnedPelvis;
    if(state_.learnedDirection[0])hip.angularSigma={.4,.6,.4};
    hip.valid = b[Hip].confidence > 0.2 && f.host - seen_[Hip] <= outputHoldSeconds;
    hip.observedHost=seen_[Hip];
    state_.learnedPosition[0]=hip.valid && b[Hip].source==5;
    for (int side = 0; side < 2; ++side) {
        int ankle = LAnkle + side, heel = LHeel + side, toe = LToe + side, small = LSmallToe + side;
        auto &t = state_.trackers[side + 1];
        Q q = t.q;
        double footLength = norm(b[toe].p - b[heel].p);
        bool footEvidence = measurements[heel].source == 2 && measurements[toe].source == 2 &&
                            measurements[heel].confidence > .4 && measurements[toe].confidence > .4 &&
                            measurements[ankle].confidence > .4 && footLength > .09 && footLength < .38 &&
                            norm(b[heel].p - b[ankle].p) < .25 && norm(b[toe].p - b[ankle].p) < .38;
        bool learnedFoot=learned && learned->available[ankle] && learned->available[heel] &&
                         learned->available[toe] && measurements[ankle].source==5 && measurements[ankle].confidence>.4;
        if(learnedFoot) {
            V3 forward=learned->points[heel]-learned->points[toe];
            V3 floorUp=f.floor.valid?f.floor.n:V3{0,1,0};
            V3 across=(learned->points[small]-learned->points[toe])*(side==0?-1.:1.);
            // The inferred foot triangle supplies roll as well as yaw/pitch.
            // A lifted or vertical foot must not be forced into the floor frame.
            bool triangle=learned->available[small] && norm(across)>.025 && norm(across)<.16 &&
                          norm(cross(unit(across),unit(forward)))>.4;
            learnedFoot=norm(forward)>.09 && norm(forward)<.38 &&
                        (triangle || norm(cross(floorUp,unit(forward)))>.35);
            if(learnedFoot) {
                auto r=triangle?unit(across-unit(forward)*dot(across,unit(forward))):
                                unit(cross(floorUp,unit(forward)),right);
                q=basis(r,cross(unit(forward),r));
                // Use the anatomical rig rotation only when it agrees with the
                // current fitted foot. This keeps corrupt/mismapped native axes
                // from creating a new hard orientation handoff.
                if(learned->bodyFitted && learned->footOrientationValid[side] &&
                   angleBetween(q,learned->footOrientations[side])<.45) {
                    const double weight=.75*std::clamp((.45-angleBetween(q,learned->footOrientations[side]))/.20,0.,1.);
                    q=blend(q,learned->footOrientations[side],weight);
                    state_.nativeFootDirection[side]=true;
                }
                footEvidence=true;
            }
        }
        if (footEvidence && !learnedFoot) {
            // Core +Z is anatomical back; the OSC S*R*S conversion makes Unity +Z forward.
            // Match the pelvis fallback frame so accepting a foot does not add a half-turn.
            V3 forward = unit(b[heel].p - b[toe].p);
            V3 floorUp = f.floor.valid ? f.floor.n : V3{0, 1, 0};
            footEvidence = norm(cross(floorUp, forward)) > .35;
            V3 r = unit(cross(floorUp, forward), right);
            if (b[small].source == 2 && b[small].confidence > 0.5) {
                V3 across = (b[small].p - b[toe].p) * (side == 0 ? -1.0 : 1.0);
                if (norm(across) > .035 && norm(across) < .15 && dot(unit(across), r) > .7)
                    r = unit(lerp(r, unit(across), .25), r);
            }
            q = basis(r, cross(forward, r));
        } else if(!learnedFoot) {
            q = t.q;
        }
        auto &rotation = rotations_[side + 1];
        if (!rotation.initialized)
            rotation.value = pelvis;
        bool accepted = rotation.update(q, footEvidence, f.host, .10, .65, 8, .045,learnedFoot);
        t.angularSigma = accepted ? V3{.25, .3, .45} : V3{.7, pi, .9};
        state_.learnedDirection[side+1]=accepted&&learnedFoot;
        if(state_.learnedDirection[side+1])t.angularSigma={.5,.65,.6};
        t.q = continuous(rotation.value, t.q);
        t.p = b[ankle].p + t.q.rotate(settings.trackerOffsets[side + 1]-V3{0,settings.soleOffset,0});
        t.velocity = velocity_[ankle];
        t.positionSigma = b[ankle].sigma;
        t.valid = b[ankle].confidence > 0.2 && f.host - seen_[ankle] <= outputHoldSeconds;
        t.observedHost=seen_[ankle];
        state_.learnedPosition[side+1]=t.valid && b[ankle].source==5;
        if (f.floor.valid && settings.contacts && settings.baseline == 0) {
            V3 sole = b[ankle].p - t.q.rotate({0, settings.soleOffset, 0});
            double h = f.floor.height(sole);
            double tilt = std::acos(std::clamp(dot(t.q.rotate({0, 1, 0}), f.floor.n), -1.0, 1.0));
            state_.contacts[side].update(dt, h, t.velocity, b[ankle].confidence > 0.5 && b[ankle].source != 4 &&
                                         (b[ankle].source!=5 || (learned && learned->supported[ankle])),
                                         tilt, sole);
            auto contact = state_.contacts[side].state;
            double target=0;
            if(contact == Contact::Planted || contact == Contact::Pivot || contact == Contact::Sliding)
                target=-std::clamp(h,-.015,.015)*std::clamp((.055-std::abs(h))/.020,0.,1.);
            // The old binary correction could jump 1.5 cm at a contact/height
            // boundary. Fade in/out; never anchor horizontal motion to the floor.
            if(learned && learned->bodyFitted) {
                bool reliable=learned->supported[ankle] && !learned->depthPredicted && learned->rootSigma<.09;
                std::optional<V3> surface;
                if(learned->footSurfaceValid[side])surface=learned->footSurfaces[side];
                t.p+=plants_[side].update(sole,t.q,t.velocity,f.floor,reliable,dt,surface);
                if(plants_[side].planted)t.velocity={};
                soleCorrections_[side]={}; // plant correction already includes floor height.
            }else {plants_[side]={};t.p+=f.floor.n*soleCorrections_[side].update(target,dt);}
        } else {soleCorrections_[side]={};plants_[side]={};}
    }
    for (int i = 0; i < 3; ++i) {
        auto &t = state_.trackers[i];
        if (t.valid && previous.host > 0 && dt > 0.005 && std::isfinite(dot(previousRotation_[i], previousRotation_[i])) && dot(previousRotation_[i], previousRotation_[i]) > 0.5) {
            Q delta = continuous(t.q, previousRotation_[i]) * previousRotation_[i].conjugate();
            V3 axis = {delta.x, delta.y, delta.z};
            double normAxis = norm(axis);
            double angle = 2 * std::atan2(normAxis, std::max(0.0, delta.w));
            V3 rawAngVel = normAxis > 1e-6 ? (axis / normAxis) * (angle / dt) : V3{};
            angularVelocity_[i] = lerp(angularVelocity_[i], bounded(rawAngVel, 20), 0.45);
            double speed = norm(angularVelocity_[i]);
            double deadband = 0.04;
            if (speed <= deadband) {
                angularVelocity_[i] = {};
            } else if (speed < deadband + 0.08) {
                angularVelocity_[i] *= (speed - deadband) / 0.08;
            }
        } else if (!t.valid) {
            angularVelocity_[i] = {};
        }
        t.angularVelocity = angularVelocity_[i];
        previousRotation_[i] = t.q;
    }
    state_.host = f.host;
    if (measured >= 6)
        state_.lastObserved = f.host;
    state_.mode = measured < 6 ? (f.host - state_.lastObserved <= 0.12 ? Mode::Predicted : Mode::Degraded)
                  : settings.baseline == 1 ? Mode::RawSdk
                  : settings.baseline == 2 ? Mode::FilteredSdk
                  : advanced > 0           ? Mode::Fused
                                           : Mode::FilteredSdk;
    state_.fitMs = (now() - start) * 1000;
    return state_;
}
State Estimator::predict(double host) const {
    return deliveryState(state_,host);
}

static void u32(std::vector<std::uint8_t> &b, std::uint32_t v) {
    for (int i = 3; i >= 0; --i)
        b.push_back(static_cast<std::uint8_t>(v >> (i * 8)));
}
static void oscString(std::vector<std::uint8_t> &b, const std::string &s) {
    b.insert(b.end(), s.begin(), s.end());
    b.push_back(0);
    while (b.size() % 4)
        b.push_back(0);
}
static void oscVec(std::vector<std::uint8_t> &b, const std::string &address, V3 v) {
    std::vector<std::uint8_t> m;
    oscString(m, address);
    oscString(m, ",fff");
    for (double x : {v.x, v.y, v.z})
        u32(m, std::bit_cast<std::uint32_t>(static_cast<float>(x)));
    u32(b, static_cast<std::uint32_t>(m.size()));
    b.insert(b.end(), m.begin(), m.end());
}
std::vector<std::uint8_t> oscBundle(const std::array<Tracker, 3> &t, const Rigid &transform) {
    if (!finite(transform.t) || !std::isfinite(dot(transform.q, transform.q)) ||
        std::abs(dot(transform.q, transform.q) - 1) > .001)
        return {};
    if (std::none_of(t.begin(), t.end(), [](const Tracker &a) { return a.valid; }))
        return {};
    for (auto &a : t) {
        if (!a.valid)
            continue;
        if (!finite(a.p) || norm(a.p) > 20 || !std::isfinite(dot(a.q, a.q)) ||
            std::abs(dot(a.q, a.q) - 1) > .001 || !finite(transform.apply(a.p)) ||
            norm(transform.apply(a.p)) > 50)
            return {};
    }
    std::vector<std::uint8_t> b;
    oscString(b, "#bundle");
    u32(b, 0);
    u32(b, 1);
    for (int i = 0; i < 3; ++i) {
        if (!t[i].valid)
            continue;
        V3 p = reflectZ(transform.apply(t[i].p));
        Q q = reflectZ(transform.q * t[i].q);
        std::string path = "/tracking/trackers/" + std::to_string(i + 1);
        oscVec(b, path + "/position", p);
        oscVec(b, path + "/rotation", eulerZXY(q));
    }
    return b;
}
} // namespace kf
