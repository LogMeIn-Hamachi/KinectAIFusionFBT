#include "io.hpp"
#include <Windows.h>
#include <bcrypt.h>
#include <cstring>
#include <iomanip>
#include <sstream>
namespace kf {
namespace {
struct Bytes {
    std::vector<std::uint8_t> b;
    size_t pos{};
    template <class T> void put(T value) {
        static_assert(std::is_arithmetic_v<T>);
        auto *p = reinterpret_cast<const std::uint8_t *>(&value);
        b.insert(b.end(), p, p + sizeof value);
    }
    template <class T> T get() {
        if (pos + sizeof(T) > b.size())
            throw std::runtime_error("Truncated recording record");
        T value;
        std::memcpy(&value, b.data() + pos, sizeof value);
        pos += sizeof value;
        return value;
    }
    void vec(V3 v) {
        put(v.x);
        put(v.y);
        put(v.z);
    }
    V3 vec() { return {get<double>(), get<double>(), get<double>()}; }
    void quat(Q q) {
        put(q.w);
        put(q.x);
        put(q.y);
        put(q.z);
    }
    Q quat() { return {get<double>(), get<double>(), get<double>(), get<double>()}; }
    template <class T> void array(const std::vector<T> &v) {
        put<uint32_t>(uint32_t(v.size()));
        auto *p = reinterpret_cast<const uint8_t *>(v.data());
        if (!v.empty())
            b.insert(b.end(), p, p + v.size() * sizeof(T));
    }
    template <class T> std::vector<T> array(size_t limit) {
        auto n = get<uint32_t>();
        if (n > limit || pos + size_t(n) * sizeof(T) > b.size())
            throw std::runtime_error("Invalid recording array length");
        std::vector<T> v(n);
        if (n)
            std::memcpy(v.data(), b.data() + pos, size_t(n) * sizeof(T));
        pos += size_t(n) * sizeof(T);
        return v;
    }
};
uint32_t crc(std::span<const uint8_t> b) {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k)
                c = (c >> 1) ^ (0xedb88320u & -(int(c & 1)));
            t[i] = c;
        }
        return t;
    }();
    uint32_t c = ~0u;
    for (auto v : b)
        c = table[(c ^ v) & 255] ^ (c >> 8);
    return ~c;
}
void writeRecord(std::ofstream &s, const Bytes &b) {
    uint32_t n = uint32_t(b.b.size()), c = crc(b.b);
    s.write(reinterpret_cast<char *>(&n), 4);
    s.write(reinterpret_cast<char *>(&c), 4);
    s.write(reinterpret_cast<const char *>(b.b.data()), n);
    if (!s)
        throw std::runtime_error("Recording write failed; check free disk space");
}
std::optional<Bytes> readRecord(std::ifstream &s) {
    uint32_t n = 0, c = 0;
    s.read(reinterpret_cast<char *>(&n), 4);
    if (s.eof() && s.gcount() == 0)
        return {};
    if (!s || n > 32 * 1024 * 1024 || n < 4)
        throw std::runtime_error("Invalid recording record size");
    s.read(reinterpret_cast<char *>(&c), 4);
    Bytes b;
    b.b.resize(n);
    s.read(reinterpret_cast<char *>(b.b.data()), n);
    if (!s || crc(b.b) != c)
        throw std::runtime_error("Truncated or corrupted recording (CRC mismatch)");
    return b;
}
} // namespace
std::string sha256(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file for integrity check: " + path.string());
    BCRYPT_ALG_HANDLE alg{};
    BCRYPT_HASH_HANDLE hash{};
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        throw std::runtime_error("SHA256 provider unavailable");
    DWORD size = 0, read = 0;
    BCryptGetProperty(alg, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&size), sizeof size, &read, 0);
    std::vector<UCHAR> object(size);
    if (BCryptCreateHash(alg, &hash, object.data(), size, nullptr, 0, 0) < 0) {
        BCryptCloseAlgorithmProvider(alg, 0);
        throw std::runtime_error("SHA256 initialization failed");
    }
    std::array<char, 65536> buf;
    while (file) {
        file.read(buf.data(), buf.size());
        if (file.gcount() > 0)
            BCryptHashData(hash, reinterpret_cast<PUCHAR>(buf.data()), ULONG(file.gcount()), 0);
    }
    std::array<UCHAR, 32> result{};
    BCryptFinishHash(hash, result.data(), 32, 0);
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(alg, 0);
    std::ostringstream out;
    for (auto c : result)
        out << std::hex << std::setw(2) << std::setfill('0') << unsigned(c);
    return out.str();
}
void RecordingWriter::open(const std::filesystem::path &p, const std::string &metadata) {
    close();
    stream_.open(p, std::ios::binary | std::ios::trunc);
    if (!stream_)
        throw std::runtime_error("Cannot create recording");
    stream_.write("KFRGBD03", 8);
    Bytes b;
    b.array(std::vector<uint8_t>(metadata.begin(), metadata.end()));
    writeRecord(stream_, b);
}
void RecordingWriter::write(const Frame &f) {
    if (!active())
        return;
    Bytes b;
    b.put(f.rgbStamp);
    b.put(f.depthStamp);
    b.put(f.skeletonStamp);
    b.put(f.rgbId);
    b.put(f.depthId);
    b.put(f.skeletonId);
    b.put(f.epoch);
    b.put(f.host);
    b.put(f.arrival);
    b.put<int32_t>(f.width);
    b.put<int32_t>(f.height);
    b.put<int32_t>(f.depthWidth);
    b.put<int32_t>(f.depthHeight);
    b.put<int32_t>(f.sensorVersion);
    b.put(f.captureMs);
    b.put(f.exposureMs);b.put(f.colorIntervalMs);
    b.array(f.bgra);
    b.array(f.depth);
    b.put<uint32_t>(uint32_t(f.mapping.size()));
    for (auto p : f.mapping) {
        b.put(p.x);
        b.put(p.y);
        b.put(p.z);
        b.put(p.u);
        b.put(p.v);
    }
    b.array(f.calibrationBlob);
    b.array(f.sdkSkeletonBlob);
    b.put<uint32_t>(uint32_t(f.bodies.size()));
    for (auto &body : f.bodies) {
        b.put(body.id);
        b.put(body.player);
        for (auto j : body.joints) {
            b.vec(j.p);
            b.put(j.confidence);
            b.put(j.sigma);
            b.put(j.source);
        }
    }
    b.vec(f.floor.n);
    b.put(f.floor.d);
    b.put<uint8_t>(f.floor.valid);
    b.put(f.vr.host);
    b.put(f.vr.epoch);
    for (auto d : f.vr.devices) {
        b.vec(d.p);
        b.quat(d.q);
        b.put<uint8_t>(d.valid);
    }
    b.quat(f.vr.standingToRaw.q);b.vec(f.vr.standingToRaw.t);
    b.put<uint8_t>(f.vr.rawTransformValid);
    b.put<uint8_t>(bool(f.runConfig));
    if (f.runConfig) {
        const auto &c = *f.runConfig;
        const auto &s = c.settings;
        for (bool flag : {s.inference, s.depth, s.constraints, s.contacts, s.vrConstraints})
            b.put<uint8_t>(flag);
        b.put<int32_t>(s.baseline);
        b.put(s.soleOffset);
        b.put(s.surfaceRadius);
        for (auto p : s.deviceOffsets)
            b.vec(p);
        for (auto p : s.trackerOffsets)
            b.vec(p);
        b.quat(c.calibration.transform.q);
        b.vec(c.calibration.transform.t);
        b.put<uint8_t>(c.calibration.valid);
        b.put(c.calibration.rms);
        b.put(c.calibration.p95);
        b.put(c.calibration.spread);
        b.quat(c.calibration.standingToRaw.q);b.vec(c.calibration.standingToRaw.t);
        b.put(c.calibration.rawEpoch);b.put<uint8_t>(c.calibration.rawReferenceValid);
        b.put(c.selectedId);
        for (auto length : c.lengths)
            b.put(length);
        b.put<uint8_t>(c.lengthsCaptured);
        b.array(std::vector<uint8_t>(c.modelHash.begin(), c.modelHash.end()));
    }
    writeRecord(stream_, b);
}
void RecordingWriter::close() {
    if (stream_.is_open()) {
        stream_.flush();
        stream_.close();
    }
}
void RecordingReader::open(const std::filesystem::path &p) {
    stream_.close();
    stream_.clear();
    stream_.open(p, std::ios::binary);
    char magic[8]{};
    stream_.read(magic, 8);
    if (!stream_ || std::memcmp(magic, "KFRGBD0", 7) || magic[7]<'1' || magic[7]>'3')
        throw std::runtime_error("Not a supported Kinect RGB-D recording");
    version_=magic[7]-'0';
    auto b = readRecord(stream_);
    if (!b)
        throw std::runtime_error("Missing recording metadata");
    auto s = b->array<uint8_t>(1024 * 1024);
    metadata.assign(s.begin(), s.end());
}
std::shared_ptr<Frame> RecordingReader::next() {
    auto record = readRecord(stream_);
    if (!record)
        return {};
    auto &b = *record;
    auto f = std::make_shared<Frame>();
    f->rgbStamp = b.get<int64_t>();
    f->depthStamp = b.get<int64_t>();
    f->skeletonStamp = b.get<int64_t>();
    f->rgbId = b.get<uint32_t>();
    f->depthId = b.get<uint32_t>();
    f->skeletonId = b.get<uint32_t>();
    f->epoch = b.get<uint32_t>();
    f->host = b.get<double>();
    f->arrival = b.get<double>();
    f->width = b.get<int32_t>();
    f->height = b.get<int32_t>();
    if(version_>=2) {
        f->depthWidth=b.get<int32_t>();f->depthHeight=b.get<int32_t>();
        f->sensorVersion=b.get<int32_t>();f->captureMs=b.get<double>();
        f->exposureMs=b.get<double>();f->colorIntervalMs=b.get<double>();
    }
    bool v1=f->sensorVersion==1 && f->width==640 && f->height==480 && f->depthWidth==640 && f->depthHeight==480;
    bool v2=f->sensorVersion==2 && f->width==1920 && f->height==1080 && f->depthWidth==512 && f->depthHeight==424;
    if ((!v1 && !v2) || !std::isfinite(f->host) || !std::isfinite(f->arrival) || !std::isfinite(f->captureMs) || f->captureMs<0 ||
        !std::isfinite(f->exposureMs) || f->exposureMs<0 || !std::isfinite(f->colorIntervalMs) || f->colorIntervalMs<0)
        throw std::runtime_error("Unsupported recording dimensions/timestamp");
    f->bgra = b.array<uint8_t>(size_t(f->width) * f->height * 4);
    f->depth = b.array<uint16_t>(size_t(f->depthWidth) * f->depthHeight);
    auto n = b.get<uint32_t>();
    if (n > unsigned(f->depthWidth * f->depthHeight))
        throw std::runtime_error("Invalid mapping length");
    f->mapping.resize(n);
    for (auto &p : f->mapping)
        p = {b.get<float>(), b.get<float>(), b.get<float>(), b.get<int32_t>(), b.get<int32_t>()};
    if(v2 && (f->bgra.size()!=1920*1080*4 || f->depth.size()!=512*424 || f->mapping.size()!=512*424))
        throw std::runtime_error("Incomplete Kinect v2 recording frame");
    f->calibrationBlob = b.array<uint8_t>(1024 * 1024);
    f->sdkSkeletonBlob = b.array<uint8_t>(65536);
    n = b.get<uint32_t>();
    if (n > 6)
        throw std::runtime_error("Invalid body count");
    f->bodies.resize(n);
    for (auto &body : f->bodies) {
        body.id = b.get<uint32_t>();
        body.player = b.get<uint8_t>();
        for (auto &j : body.joints) {
            j.p = b.vec();
            j.confidence = b.get<double>();
            j.sigma = b.get<double>();
            j.source = b.get<uint8_t>();
            if (!finite(j.p) || !std::isfinite(j.confidence) || j.confidence < 0 || j.confidence > 1 ||
                !std::isfinite(j.sigma) || j.sigma < 0)
                throw std::runtime_error("Invalid joint in recording");
        }
    }
    f->floor.n = b.vec();
    f->floor.d = b.get<double>();
    f->floor.valid = b.get<uint8_t>() != 0;
    f->vr.host = b.get<double>();
    f->vr.epoch = b.get<uint64_t>();
    for (auto &d : f->vr.devices) {
        d.p = b.vec();
        d.q = b.quat();
        d.valid = b.get<uint8_t>() != 0;
        if (!finite(d.p) || !std::isfinite(dot(d.q, d.q)))
            throw std::runtime_error("Invalid VR pose");
    }
    if(version_>=3) {
        f->vr.standingToRaw.q=b.quat();f->vr.standingToRaw.t=b.vec();
        f->vr.rawTransformValid=b.get<uint8_t>()!=0;
        if(!finiteRigid(f->vr.standingToRaw))throw std::runtime_error("Invalid recorded VR reference");
    }
    if (b.get<uint8_t>()) {
        auto c = std::make_shared<ReplayConfig>();
        auto &s = c->settings;
        s.inference = b.get<uint8_t>() != 0;
        s.depth = b.get<uint8_t>() != 0;
        s.constraints = b.get<uint8_t>() != 0;
        s.contacts = b.get<uint8_t>() != 0;
        s.vrConstraints = b.get<uint8_t>() != 0;
        s.baseline = b.get<int32_t>();
        s.soleOffset = b.get<double>();
        s.surfaceRadius = b.get<double>();
        for (auto &p : s.deviceOffsets)
            p = b.vec();
        for (auto &p : s.trackerOffsets)
            p = b.vec();
        c->calibration.transform.q = b.quat();
        c->calibration.transform.t = b.vec();
        c->calibration.valid = b.get<uint8_t>() != 0;
        c->calibration.rms = b.get<double>();
        c->calibration.p95 = b.get<double>();
        c->calibration.spread = b.get<double>();
        if(version_>=3) {
            c->calibration.standingToRaw.q=b.quat();c->calibration.standingToRaw.t=b.vec();
            c->calibration.rawEpoch=b.get<uint64_t>();c->calibration.rawReferenceValid=b.get<uint8_t>()!=0;
            if(!finiteRigid(c->calibration.standingToRaw))throw std::runtime_error("Invalid recorded calibration reference");
        }
        c->selectedId = b.get<uint32_t>();
        for (auto &length : c->lengths)
            length = b.get<double>();
        if(version_>=3)c->lengthsCaptured=b.get<uint8_t>()!=0;
        auto hash = b.array<uint8_t>(64);
        c->modelHash.assign(hash.begin(), hash.end());
        if (s.baseline < 0 || s.baseline > 2 || !std::isfinite(s.soleOffset) || s.soleOffset < .02 ||
            s.soleOffset > .2 || !std::isfinite(s.surfaceRadius) || s.surfaceRadius < .005 ||
            s.surfaceRadius > .15 || !finite(c->calibration.transform.t) ||
            !std::isfinite(dot(c->calibration.transform.q, c->calibration.transform.q)) ||
            std::abs(dot(c->calibration.transform.q, c->calibration.transform.q) - 1) > .001)
            throw std::runtime_error("Invalid recorded configuration");
        for (auto v : s.deviceOffsets)
            if (!finite(v) || norm(v) > .5)
                throw std::runtime_error("Invalid recorded offsets");
        for (auto v : s.trackerOffsets)
            if (!finite(v) || norm(v) > .5)
                throw std::runtime_error("Invalid tracker offsets");
        for (double v : c->lengths)
            if (!std::isfinite(v) || v < 0 || v > .85)
                throw std::runtime_error("Invalid recorded body proportions");
        f->runConfig = c;
    }
    if (b.pos != b.b.size())
        throw std::runtime_error("Unexpected trailing record data");
    return f;
}
void saveCalibration(const std::filesystem::path &p, const Calibration &c, const Settings &s,bool learnedOffsets) {
    std::ofstream f(p.string() + ".tmp");
    f << std::setprecision(17) << "KF_CALIBRATION_2\n"
      << c.valid << ' ' << c.transform.q.w << ' ' << c.transform.q.x << ' ' << c.transform.q.y << ' '
      << c.transform.q.z << ' ' << c.transform.t.x << ' ' << c.transform.t.y << ' ' << c.transform.t.z << ' '
      << c.rms << ' ' << c.p95 << ' ' << c.spread << '\n';
    for (auto v : s.deviceOffsets)
        f << v.x << ' ' << v.y << ' ' << v.z << '\n';
    f << s.soleOffset << '\n';
    f << "WRIST_OFFSETS " << learnedOffsets << '\n';
    f << "REFERENCE " << c.rawReferenceValid << ' ' << c.standingToRaw.q.w << ' '
      << c.standingToRaw.q.x << ' ' << c.standingToRaw.q.y << ' ' << c.standingToRaw.q.z << ' '
      << c.standingToRaw.t.x << ' ' << c.standingToRaw.t.y << ' ' << c.standingToRaw.t.z << ' '
      << c.referenceSource << ' ' << c.referenceUniverse << '\n';
    f.close();
    if (!f)
        throw std::runtime_error("Cannot save calibration");
    if (!MoveFileExW(std::filesystem::path(p.string() + ".tmp").c_str(), p.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Cannot replace calibration");
}
std::optional<std::string> trySaveCalibration(const std::filesystem::path& p,const Calibration& c,const Settings& s,bool learnedOffsets) {
    try {saveCalibration(p,c,s,learnedOffsets);return std::nullopt;}
    catch(const std::exception& e){return std::string(e.what());}
}
bool loadCalibration(const std::filesystem::path &p, Calibration &c, Settings &s,bool* learnedOffsets) {
    if(learnedOffsets)*learnedOffsets=false;
    std::ifstream f(p);
    std::string magic;
    f >> magic;
    Calibration a;
    Settings next = s;
    f >> a.valid >> a.transform.q.w >> a.transform.q.x >> a.transform.q.y >> a.transform.q.z >>
        a.transform.t.x >> a.transform.t.y >> a.transform.t.z >> a.rms >> a.p95 >> a.spread;
    for (auto &v : next.deviceOffsets)
        f >> v.x >> v.y >> v.z;
    f >> next.soleOffset;
    if (!f || (magic != "KF_CALIBRATION_1" && magic != "KF_CALIBRATION_2") || !finite(a.transform.t) || norm(a.transform.t) > 20 ||
        std::abs(dot(a.transform.q, a.transform.q) - 1) > 0.001 ||
        !std::isfinite(dot(a.transform.q, a.transform.q)) || !std::isfinite(next.soleOffset) ||
        next.soleOffset < 0.02 || next.soleOffset > 0.2 || !std::isfinite(a.rms) || a.rms < 0 || a.rms > 1 ||
        !std::isfinite(a.p95) || a.p95 < 0 || a.p95 > 2 || !std::isfinite(a.spread) || a.spread < 0 ||
        a.spread > 10)
        return false;
    for (auto v : next.deviceOffsets)
        if (!finite(v) || norm(v) > 0.5)
            return false;
    bool accepted=a.valid;
    bool known=accepted && a.spread>=.07 && a.rms<calibrationMaxRms && norm(next.deviceOffsets[1])>.001 && norm(next.deviceOffsets[2])>.001;
    std::string marker;bool savedKnown{};
    const bool hasOffsets=bool(f>>marker>>savedKnown);
    if(hasOffsets)known=marker=="WRIST_OFFSETS" && savedKnown;
    if(magic=="KF_CALIBRATION_2") {
        if(!hasOffsets || marker!="WRIST_OFFSETS")return false;
        f>>marker>>a.rawReferenceValid>>a.standingToRaw.q.w>>a.standingToRaw.q.x
         >>a.standingToRaw.q.y>>a.standingToRaw.q.z>>a.standingToRaw.t.x>>a.standingToRaw.t.y
         >>a.standingToRaw.t.z>>a.referenceSource>>a.referenceUniverse;
        if(!f || marker!="REFERENCE" || !finiteRigid(a.standingToRaw) || norm(a.standingToRaw.t)>100)return false;
        // Runtime counters are never durable room identifiers. Loading always
        // requires explicit restoration, retaining the serialized reference.
        a.rawEpoch=0;
    }
    if(learnedOffsets)*learnedOffsets=known && norm(next.deviceOffsets[1])<=.30 && norm(next.deviceOffsets[2])<=.30;
    a.valid = false;
    a.reason = "Saved transform loaded; verify current sensor and VR origin before enabling output";
    c = a;
    s = next;
    return accepted;
}
void saveBgraBmp(const std::filesystem::path &p, const Frame &f) {
    if (f.bgra.size() != size_t(f.width) * f.height * 4)
        throw std::runtime_error("No RGB image");
    BITMAPFILEHEADER file{};
    BITMAPINFOHEADER info{};
    file.bfType = 0x4d42;
    file.bfOffBits = sizeof file + sizeof info;
    file.bfSize = file.bfOffBits + DWORD(f.bgra.size());
    info.biSize = sizeof info;
    info.biWidth = f.width;
    info.biHeight = -f.height;
    info.biPlanes = 1;
    info.biBitCount = 32;
    std::ofstream out(p, std::ios::binary);
    out.write(reinterpret_cast<char *>(&file), sizeof file);
    out.write(reinterpret_cast<char *>(&info), sizeof info);
    out.write(reinterpret_cast<const char *>(f.bgra.data()), f.bgra.size());
    if (!out)
        throw std::runtime_error("Cannot write BMP");
}
} // namespace kf
