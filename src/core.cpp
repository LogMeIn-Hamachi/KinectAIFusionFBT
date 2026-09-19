#include "core.hpp"
#include <numeric>
#include <sstream>
#include <iomanip>

namespace kf {
M3 matrix(Q q) {
    q = normalized(q);
    M3 m;
    double w = q.w, x = q.x, y = q.y, z = q.z;
    m.a[0][0] = 1 - 2 * (y * y + z * z);
    m.a[0][1] = 2 * (x * y - z * w);
    m.a[0][2] = 2 * (x * z + y * w);
    m.a[1][0] = 2 * (x * y + z * w);
    m.a[1][1] = 1 - 2 * (x * x + z * z);
    m.a[1][2] = 2 * (y * z - x * w);
    m.a[2][0] = 2 * (x * z - y * w);
    m.a[2][1] = 2 * (y * z + x * w);
    m.a[2][2] = 1 - 2 * (x * x + y * y);
    return m;
}
Q quaternion(M3 m) {
    auto &a = m.a;
    Q q;
    double t = a[0][0] + a[1][1] + a[2][2];
    if (t > 0) {
        double s = std::sqrt(t + 1) * 2;
        q = {s / 4, (a[2][1] - a[1][2]) / s, (a[0][2] - a[2][0]) / s, (a[1][0] - a[0][1]) / s};
    } else {
        int i = a[1][1] > a[0][0] ? 1 : 0;
        if (a[2][2] > a[i][i])
            i = 2;
        int j = (i + 1) % 3, k = (j + 1) % 3;
        double s = std::sqrt(std::max(0.0, 1 + a[i][i] - a[j][j] - a[k][k])) * 2;
        double v[3]{};
        if (s < 1e-9)
            return {};
        v[i] = s / 4;
        v[j] = (a[j][i] + a[i][j]) / s;
        v[k] = (a[k][i] + a[i][k]) / s;
        q = {(a[k][j] - a[j][k]) / s, v[0], v[1], v[2]};
    }
    return normalized(q);
}
Q basis(V3 right, V3 up) {
    right = unit(right, {1, 0, 0});
    V3 forward = unit(cross(right, up), {0, 0, 1});
    up = unit(cross(forward, right));
    M3 m;
    V3 rows[3]{{right.x, up.x, forward.x}, {right.y, up.y, forward.y}, {right.z, up.z, forward.z}};
    for (int i = 0; i < 3; ++i) {
        m.a[i][0] = rows[i].x;
        m.a[i][1] = rows[i].y;
        m.a[i][2] = rows[i].z;
    }
    return quaternion(m);
}
Q reflectZ(Q q) {
    M3 m = matrix(q);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            m.a[i][j] *= (i == 2 ? -1 : 1) * (j == 2 ? -1 : 1);
    return quaternion(m);
}
V3 eulerZXY(Q q) {
    M3 m = matrix(q);
    double x = std::asin(std::clamp(-m.a[1][2], -1.0, 1.0)), y, z;
    if (std::abs(std::cos(x)) > 1e-7) {
        z = std::atan2(m.a[1][0], m.a[1][1]);
        y = std::atan2(m.a[0][2], m.a[2][2]);
    } else {
        z = 0;
        y = std::atan2(-m.a[2][0], m.a[0][0]);
    }
    return V3{x, y, z} * (180 / pi);
}
Q fromEulerZXY(V3 d) {
    d = d * (pi / 180);
    return axisAngle({0, 1, 0}, d.y) * axisAngle({1, 0, 0}, d.x) * axisAngle({0, 0, 1}, d.z);
}
// Symmetric Jacobi eigenvectors. Small fixed matrices avoid a heavyweight math dependency.
template <size_t N> static void eigen(double (&a)[N][N], double (&v)[N][N]) {
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            v[i][j] = i == j ? 1 : 0;
    for (int sweep = 0; sweep < 80; ++sweep) {
        size_t p = 0, q = 1;
        double biggest = 0;
        for (size_t i = 0; i < N; ++i)
            for (size_t j = i + 1; j < N; ++j)
                if (std::abs(a[i][j]) > biggest) {
                    biggest = std::abs(a[i][j]);
                    p = i;
                    q = j;
                }
        if (biggest < 1e-12)
            break;
        double angle = 0.5 * std::atan2(2 * a[p][q], a[q][q] - a[p][p]), c = std::cos(angle),
               s = std::sin(angle);
        double pp = c * c * a[p][p] - 2 * s * c * a[p][q] + s * s * a[q][q],
               qq = s * s * a[p][p] + 2 * s * c * a[p][q] + c * c * a[q][q];
        for (size_t k = 0; k < N; ++k)
            if (k != p && k != q) {
                double kp = c * a[k][p] - s * a[k][q], kq = s * a[k][p] + c * a[k][q];
                a[k][p] = a[p][k] = kp;
                a[k][q] = a[q][k] = kq;
            }
        a[p][p] = pp;
        a[q][q] = qq;
        a[p][q] = a[q][p] = 0;
        for (size_t k = 0; k < N; ++k) {
            double kp = c * v[k][p] - s * v[k][q];
            v[k][q] = s * v[k][p] + c * v[k][q];
            v[k][p] = kp;
        }
    }
}
Calibration calibrate(std::span<const Pair3> p) {
    Calibration out;
    if (p.size() < 18) {
        out.reason = "Collect at least 18 varied samples";
        return out;
    }
    for (auto a : p)
        if (!finite(a.camera) || !finite(a.world)) {
            out.reason = "Nonfinite sample";
            return out;
        }
    std::vector<double> w(p.size(), 1);
    Rigid r;
    double spread = 0;
    for (int iter = 0; iter < 8; ++iter) {
        V3 ac{}, bc{};
        double total = 0;
        for (size_t i = 0; i < p.size(); ++i) {
            ac += p[i].camera * w[i];
            bc += p[i].world * w[i];
            total += w[i];
        }
        ac = ac / total;
        bc = bc / total;
        double s[3][3]{}, cov[3][3]{};
        for (size_t k = 0; k < p.size(); ++k) {
            V3 a = p[k].camera - ac, b = p[k].world - bc;
            double av[]{a.x, a.y, a.z}, bv[]{b.x, b.y, b.z};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    s[i][j] += w[k] * av[i] * bv[j];
                    cov[i][j] += w[k] * av[i] * av[j] / total;
                }
        }
        double ev[3][3];
        eigen(cov, ev);
        std::array<double, 3> vals{cov[0][0], cov[1][1], cov[2][2]};
        std::sort(vals.begin(), vals.end());
        spread = std::sqrt(std::max(0.0, vals[1]));
        if (spread < 0.07) {
            out.reason =
                "Samples are too concentrated or collinear; move hands across height, width and depth";
            return out;
        }
        double tr = s[0][0] + s[1][1] + s[2][2];
        double n[4][4]{
            {tr, s[1][2] - s[2][1], s[2][0] - s[0][2], s[0][1] - s[1][0]},
            {s[1][2] - s[2][1], s[0][0] - s[1][1] - s[2][2], s[0][1] + s[1][0], s[2][0] + s[0][2]},
            {s[2][0] - s[0][2], s[0][1] + s[1][0], -s[0][0] + s[1][1] - s[2][2], s[1][2] + s[2][1]},
            {s[0][1] - s[1][0], s[2][0] + s[0][2], s[1][2] + s[2][1], -s[0][0] - s[1][1] + s[2][2]}};
        double v[4][4];
        eigen(n, v);
        int best = 0;
        for (int i = 1; i < 4; ++i)
            if (n[i][i] > n[best][best])
                best = i;
        r.q = normalized({v[0][best], v[1][best], v[2][best], v[3][best]});
        r.t = bc - r.q.rotate(ac);
        for (size_t i = 0; i < p.size(); ++i) {
            double e = norm(r.apply(p[i].camera) - p[i].world);
            w[i] = e <= 0.035 ? 1 : 0.035 / e;
        }
    }
    std::vector<double> errors;
    double sum = 0;
    size_t accepted = 0;
    for (auto a : p) {
        double e = norm(r.apply(a.camera) - a.world);
        errors.push_back(e);
        if (e < calibrationInlierDistance) {
            sum += e * e;
            ++accepted;
        }
    }
    std::sort(errors.begin(), errors.end());
    out.rms = accepted ? std::sqrt(sum / accepted) : 1;
    out.p95 = errors[static_cast<size_t>((errors.size() - 1) * 0.95)];
    out.spread = spread;
    out.transform = r;
    const bool enoughConsistent = double(accepted) / p.size() >= calibrationMinInlierFraction;
    const bool closeEnough = out.rms < calibrationMaxRms;
    out.valid = enoughConsistent && closeEnough;
    std::ostringstream message;
    message << (out.valid           ? "Alignment accepted. "
                : !enoughConsistent ? "Too many inconsistent samples. "
                                    : "Position mismatch too large. ")
            << std::fixed << std::setprecision(1) << "RMS " << out.rms * 100 << " cm (must be below "
            << calibrationMaxRms * 100 << "); " << std::setprecision(0) << double(accepted) * 100 / p.size()
            << "% within " << calibrationInlierDistance * 100 << " cm (need at least "
            << calibrationMinInlierFraction * 100 << "%).";
    if (!out.valid)
        message << " Keep hands visible, move slowly, and check device offsets.";
    out.reason = message.str();
    return out;
}
Plane fitFloor(std::span<const V3> p) {
    Plane best;
    if (p.size() < 40)
        return best;
    size_t bestCount = 0;
    for (size_t i = 0; i < 96; ++i) {
        V3 a = p[(i * 47) % p.size()], b = p[(i * 109 + 7) % p.size()], c = p[(i * 233 + 31) % p.size()];
        V3 n = unit(cross(b - a, c - a), {0, 0, 0});
        if (n.y < 0)
            n = -n;
        if (n.y < 0.75)
            continue;
        double d = -dot(n, a);
        size_t count = 0;
        for (V3 x : p)
            if (finite(x) && std::abs(dot(n, x) + d) < 0.025)
                ++count;
        if (count > bestCount) {
            bestCount = count;
            best = {n, d, true};
        }
    }
    if (bestCount < p.size() / 4)
        best.valid = false;
    return best;
}
double ClockMap::map(std::int64_t ms, double arrival) {
    bool reset = ready && (ms < last || ms - last > 3000 || arrival < lastHost);
    if (!ready || reset) {
        offset = arrival - ms * 0.001;
        ready = true;
        if (reset)
            ++resets;
    } else {
        double sample = arrival - ms * 0.001;
        offset = std::min(offset + 0.000005, sample);
    }
    last = ms;
    lastHost = arrival;
    return std::min(arrival, ms * 0.001 + offset);
}
void PoseHistory::add(VrSample p) {
    if (!poses_.empty() && (p.host <= poses_.back().host || p.epoch != poses_.back().epoch ||
                           p.referenceSource != poses_.back().referenceSource))
        poses_.clear();
    poses_.push_back(p);
    while (poses_.size() > 300 || (!poses_.empty() && p.host - poses_.front().host > 3))
        poses_.pop_front();
}
std::optional<VrSample> PoseHistory::at(double host) const {
    if (poses_.empty() || host < poses_.front().host || host > poses_.back().host + 0.015)
        return {};
    if (host >= poses_.back().host)
        return poses_.back();
    for (size_t i = 1; i < poses_.size(); ++i)
        if (poses_[i].host >= host) {
            auto a = poses_[i - 1], b = poses_[i];
            if (b.host - a.host > 0.04 || a.epoch!=b.epoch || a.referenceSource!=b.referenceSource)
                return {};
            double t = (host - a.host) / (b.host - a.host);
            a.host = host;
            // Interpolate in one physical reference. Mixing standing poses from
            // opposite sides of a space-drag update invents controller movement.
            a.rawTransformValid=a.rawTransformValid && b.rawTransformValid &&
                finiteRigid(a.standingToRaw) && finiteRigid(b.standingToRaw);
            if(a.rawTransformValid) {
                const auto bToA=composeRigid(inverseRigid(a.standingToRaw),b.standingToRaw);
                for(auto& d:b.devices)if(d.valid){d.p=bToA.apply(d.p);d.q=normalized(bToA.q*d.q);}
            }
            for (int j = 0; j < 3; ++j) {
                a.devices[j].valid = a.devices[j].valid && b.devices[j].valid;
                a.devices[j].p = lerp(a.devices[j].p, b.devices[j].p, t);
                a.devices[j].q = blend(a.devices[j].q, b.devices[j].q, t);
            }
            return a;
        }
    return {};
}
Crop cropBox(double x1, double y1, double x2, double y2) {
    double w = std::max(20.0, x2 - x1) * 1.25, h = std::max(20.0, y2 - y1) * 1.25;
    if (w / h > 0.75)
        h = w / 0.75;
    else
        w = h * 0.75;
    return {(x1 + x2) / 2, (y1 + y2) / 2, w, h};
}
Keypoints kinectImageLabels(Keypoints k) {
    constexpr std::array<int, J> swap{0,  2,  1,  4,  3,  6,  5,  8,  7,  10, 9,  12, 11,
                                      14, 13, 16, 15, 17, 18, 19, 21, 20, 23, 22, 25, 24};
    Keypoints o;
    for (int i = 0; i < J; ++i) {
        o[i] = k[swap[i]];
    }
    return o;
}
Keypoints unmirror(Keypoints k, int width) {
    auto o = kinectImageLabels(k);
    for (auto &point : o)
        point.uv.x = width - 1 - point.uv.x;
    return o;
}
void indexRegistration(Frame &f) {
    f.colorIndex.assign(size_t(f.width) * f.height, -1);
    for (size_t i = 0; i < f.mapping.size(); ++i) {
        auto m = f.mapping[i];
        if (!std::isfinite(m.x) || !std::isfinite(m.y) || !std::isfinite(m.z) || m.z < (f.sensorVersion==2?.5:.8) || m.z > (f.sensorVersion==2?4.5:4.) ||
            m.u < 0 || m.v < 0 || m.u >= f.width || m.v >= f.height)
            continue;
        auto &index = f.colorIndex[size_t(m.v) * f.width + m.u];
        if (index < 0 || m.z < f.mapping[index].z)
            index = int32_t(i);
    }
}
std::optional<Joint> associateDepth(const Frame &f, Keypoint kp, const Joint &prior, int joint,
                                    std::uint8_t player, double surfaceRadius, DepthSupport *support) {
    DepthSupport diagnostic;
    if (!support)
        support = &diagnostic;
    *support = {};
    support->rejection = 1;
    if (kp.score < 0.35 || !std::isfinite(kp.uv.x) || !std::isfinite(kp.uv.y) || kp.uv.x < 0 ||
        kp.uv.x >= f.width || kp.uv.y < 0 || kp.uv.y >= f.height)
        return {};
    std::vector<V3> points;
    points.reserve(121);
    std::vector<V3> footPlayerPoints, footInnerPoints;
    // Feet occupy few depth pixels. A broad patch straddles toe/ankle depth edges.
    bool foot = joint >= LToe;
    int pixelRadius = 5 * std::max(1,f.width/640);
    auto add = [&](size_t i) {
        const auto &m = f.mapping[i];
        if (!std::isfinite(m.z) || m.z < (f.sensorVersion==2?.5f:.8f) || m.z > (f.sensorVersion==2?4.5f:4.f) || std::abs(m.u - kp.uv.x) > pixelRadius ||
            std::abs(m.v - kp.uv.y) > pixelRadius)
            return;
        if (player && i < f.depth.size() && (f.depth[i] & 7) != 0 && (f.depth[i] & 7) != player)
            return;
        V3 point{m.x, m.y, m.z};
        points.push_back(point);
        if (player && i < f.depth.size() && (f.depth[i] & 7) == player) {
            ++support->playerSamples;
            if (foot) {
                footPlayerPoints.push_back(point);
                if (std::abs(m.u - kp.uv.x) <= 2*std::max(1,f.width/640) && std::abs(m.v - kp.uv.y) <= 2*std::max(1,f.width/640))
                    footInnerPoints.push_back(point);
            }
        }
    };
    if (f.colorIndex.size() == size_t(f.width) * f.height) {
        int cx = int(std::round(kp.uv.x)), cy = int(std::round(kp.uv.y));
        for (int y = std::max(0, cy - pixelRadius); y <= std::min(f.height - 1, cy + pixelRadius); ++y)
            for (int x = std::max(0, cx - pixelRadius); x <= std::min(f.width - 1, cx + pixelRadius); ++x) {
                int i = f.colorIndex[size_t(y) * f.width + x];
                if (i >= 0)
                    add(size_t(i));
            }
    } else
        for (size_t i = 0; i < f.mapping.size(); ++i)
            add(i);
    if (foot && player) {
        // Use the tight patch when supported; otherwise use nearby selected-player pixels.
        // Unlabelled carpet is not evidence of the toe, even if its depth is very stable.
        points = footInnerPoints.size() >= 8 ? std::move(footInnerPoints) : std::move(footPlayerPoints);
    }
    support->samples = unsigned(points.size());
    support->rejection = 2;
    if (points.size() < 8)
        return {};
    std::sort(points.begin(), points.end(), [](V3 a, V3 b) { return a.z < b.z; });
    double median = points[points.size() / 2].z,
           spread = points[points.size() * 9 / 10].z - points[points.size() / 10].z;
    support->spread = spread;
    support->rejection = 3;
    if (spread > 0.085)
        return {};
    double radius = (joint == LHip || joint == RHip || joint == Hip) ? surfaceRadius * 2
                                                                     : (joint >= 20 ? 0.025 : surfaceRadius);
    support->priorDelta = std::abs(prior.p.z - (median + radius));
    support->rejection = 4;
    if (prior.confidence > 0.1 && support->priorDelta > 0.18)
        return {};
    V3 sum{};
    int count = 0;
    for (V3 p : points)
        if (std::abs(p.z - median) < 0.035) {
            sum += p;
            ++count;
        }
    support->rejection = 5;
    if (count < 8)
        return {};
    V3 surface = sum / count;
    V3 center = surface + unit(surface) * radius;
    support->rejection = 0;
    return Joint{center, std::min(0.85, kp.score), 0.035 + spread + radius * 0.5, 2};
}
void FootContact::update(double dt, double h, V3 v, bool visible, double tilt, V3 sole) {
    double horizontal = std::hypot(v.x, v.z);
    if (!visible || h > 0.055 || std::abs(v.y) > 0.22) {
        state = Contact::Air;
        dwell = 0;
        return;
    }
    if (h < 0.025 && std::abs(v.y) < 0.1) {
        dwell += dt;
        if (state == Contact::Air)
            state = Contact::Candidate;
        if (dwell >= 0.1) {
            if (horizontal > 0.13)
                state = Contact::Sliding;
            else if (tilt > 0.15)
                state = Contact::Pivot;
            else
                state = Contact::Planted;
            if (state != Contact::Planted || dwell - dt < 0.1)
                anchor = sole;
        }
    } else if (state == Contact::Candidate) {
        state = Contact::Air;
        dwell = 0;
    }
}
std::string modeName(Mode m) {
    switch (m) {
    case Mode::RawSdk:
        return "Raw SDK baseline";
    case Mode::FilteredSdk:
        return "Filtered SDK baseline";
    case Mode::Fused:
        return "RGB-D fusion (experimental)";
    case Mode::Predicted:
        return "Short prediction";
    case Mode::Degraded:
        return "Degraded - output paused";
    default:
        return "No selected player";
    }
}
std::optional<Crop> playerCrop(const Frame &f, uint32_t id) {
    auto it = std::find_if(f.bodies.begin(), f.bodies.end(), [&](auto &b) { return b.id == id; });
    if (it == f.bodies.end())
        return {};
    double x0 = f.width, y0 = f.height, x1 = 0, y1 = 0;
    int count = 0;
    for (size_t i = 0; i < f.mapping.size(); i += 2)
        if (i < f.depth.size() && (f.depth[i] & 7) == it->player) {
            auto p = f.mapping[i];
            if (p.u >= 0 && p.u < f.width && p.v >= 0 && p.v < f.height && p.z > (f.sensorVersion==2?.5f:.8f) && p.z < (f.sensorVersion==2?4.5f:4.f)) {
                x0 = std::min(x0, double(p.u));
                x1 = std::max(x1, double(p.u));
                y0 = std::min(y0, double(p.v));
                y1 = std::max(y1, double(p.v));
                ++count;
            }
        }
    if (count < 100 || x1 - x0 < 20 || y1 - y0 < 40)
        return {};
    return cropBox(x0, y0, x1, y1);
}
} // namespace kf
