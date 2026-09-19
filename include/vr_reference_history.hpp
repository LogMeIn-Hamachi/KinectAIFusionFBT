#pragma once
#include "core.hpp"
#include <ostream>

namespace kf {
// Bounded, numbers-only history. No disk writes until Diagnostics is requested.
// Keep it across capture restarts so the old and new references remain visible.
class VrReferenceHistory {
    struct Entry {
        VrSample vr;
        Rigid cameraToStanding, savedStandingToRaw;
        std::uint64_t savedEpoch{},savedSource{},savedUniverse{};
        bool accepted{},savedReference{},output{};
    };
    std::deque<Entry> samples_;
    static bool different(const Rigid& a,const Rigid& b) {
        return norm(a.t-b.t)>1e-6 || std::abs(dot(a.q,b.q))<1-1e-10;
    }
public:
    void add(const VrSample& vr,const Calibration& cal,bool output) {
        if(!std::isfinite(vr.host))return;
        if(!samples_.empty()) {
            const auto& last=samples_.back();
            bool changed=vr.referenceEvents || vr.epoch!=last.vr.epoch ||
                vr.referenceSource!=last.vr.referenceSource || vr.referenceUniverse!=last.vr.referenceUniverse ||
                vr.rawTransformValid!=last.vr.rawTransformValid || cal.valid!=last.accepted ||
                cal.rawReferenceValid!=last.savedReference || cal.rawEpoch!=last.savedEpoch ||
                output!=last.output || different(vr.standingToRaw,last.vr.standingToRaw) ||
                different(cal.standingToRaw,last.savedStandingToRaw) || different(cal.transform,last.cameraToStanding);
            for(int i=0;i<3;++i)changed|=vr.devices[i].valid!=last.vr.devices[i].valid;
            if(!changed && vr.host>=last.vr.host && vr.host-last.vr.host<.25)return;
        }
        samples_.push_back({vr,cal.transform,cal.standingToRaw,cal.rawEpoch,cal.referenceSource,
                            cal.referenceUniverse,cal.valid,cal.rawReferenceValid,output});
        if(samples_.size()>600)samples_.pop_front();
    }
    size_t size()const{return samples_.size();}
    void write(std::ostream& out)const {
        out<<"elapsed_s,epoch,source_key,universe,events,reference_valid,alignment_valid,output_enabled,saved_epoch,saved_source_key,saved_universe,saved_reference_valid";
        for(const char* label:{"standing_to_raw","saved_standing_to_raw","camera_to_saved_standing"})
            for(const char* part:{"qw","qx","qy","qz","x","y","z"})out<<','<<label<<'_'<<part;
        for(int i=0;i<3;++i)for(const char* part:{"valid","qw","qx","qy","qz","x","y","z"})out<<",device"<<i<<'_'<<part;
        out<<'\n';
        auto rigid=[&](const Rigid& r){out<<','<<r.q.w<<','<<r.q.x<<','<<r.q.y<<','<<r.q.z<<','<<r.t.x<<','<<r.t.y<<','<<r.t.z;};
        for(const auto& s:samples_) {
            const auto& v=s.vr;
            out<<v.host-samples_.front().vr.host<<','<<v.epoch<<','<<v.referenceSource<<','<<v.referenceUniverse
               <<','<<v.referenceEvents<<','<<v.rawTransformValid<<','<<s.accepted<<','<<s.output
               <<','<<s.savedEpoch<<','<<s.savedSource<<','<<s.savedUniverse<<','<<s.savedReference;
            rigid(v.standingToRaw);rigid(s.savedStandingToRaw);rigid(s.cameraToStanding);
            for(const auto& d:v.devices){out<<','<<d.valid;rigid({d.q,d.p});}
            out<<'\n';
        }
    }
};
} // namespace kf
