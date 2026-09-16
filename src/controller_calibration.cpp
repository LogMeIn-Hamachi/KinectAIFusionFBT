#include "alignment.hpp"
namespace kf {
namespace {
constexpr int N=12; // camera rotation, translation, left wrist offset, right wrist offset
using Matrix=std::array<std::array<double,N>,N>;
using Vector=std::array<double,N>;
bool solve(Matrix a,Vector b,Vector &x) {
    Vector scale{};
    for(int i=0;i<N;++i) {
        if(!std::isfinite(a[i][i]) || a[i][i]<1e-9)return false;
        scale[i]=std::sqrt(a[i][i]);
    }
    for(int i=0;i<N;++i){b[i]/=scale[i];for(int j=0;j<N;++j)a[i][j]/=scale[i]*scale[j];}
    for(int k=0;k<N;++k) {
        int p=k;for(int i=k+1;i<N;++i)if(std::abs(a[i][k])>std::abs(a[p][k]))p=i;
        // Reject unobservable translations/offsets, including all controllers
        // held in one orientation or rotations about only one shared axis.
        if(std::abs(a[p][k])<.001)return false;
        std::swap(a[p],a[k]);std::swap(b[p],b[k]);
        double d=a[k][k];for(int j=k;j<N;++j)a[k][j]/=d;b[k]/=d;
        for(int i=0;i<N;++i)if(i!=k){double f=a[i][k];for(int j=k;j<N;++j)a[i][j]-=f*a[k][j];b[i]-=f*b[k];}
    }
    for(int i=0;i<N;++i){x[i]=b[i]/scale[i];if(!std::isfinite(x[i]))return false;}
    return true;
}
double cost(std::span<const AlignmentObservation> samples,const Rigid &r,const std::array<V3,3> &offsets) {
    double total=0;
    for(auto &s:samples) {
        double e=norm(r.apply(s.camera)-s.pose.p-s.pose.q.rotate(offsets[s.device]));
        total+=e<.035?e*e:2*.035*e-.035*.035;
    }
    return total;
}
void normal(std::span<const AlignmentObservation> samples,const Rigid &r,const std::array<V3,3> &offsets,Matrix &h,Vector &b) {
    const V3 axes[]{{1,0,0},{0,1,0},{0,0,1}};
    for(auto &s:samples) {
        V3 c=r.q.rotate(s.camera),e=c+r.t-s.pose.p-s.pose.q.rotate(offsets[s.device]);
        double w=std::min(1.,.035/std::max(norm(e),1e-9));
        std::array<V3,N> j{};
        for(int k=0;k<3;++k){j[k]=cross(axes[k],c);j[3+k]=axes[k];j[6+(s.device-1)*3+k]=-s.pose.q.rotate(axes[k]);}
        for(int a=0;a<N;++a){b[a]-=w*dot(j[a],e);for(int k=0;k<N;++k)h[a][k]+=w*dot(j[a],j[k]);}
    }
}
}
Calibration calibrateControllers(std::span<const AlignmentObservation> samples,std::array<V3,3> &offsets,const Plane* floor) {
    offsets={};
    Calibration result;
    std::array<int,3> counts{};
    std::vector<Pair3> pairs;
    for(auto &s:samples) {
        if(s.device<1 || s.device>2 || !s.pose.valid || !finite(s.camera) || !finite(s.pose.p) ||
           !std::isfinite(dot(s.pose.q,s.pose.q)) || std::abs(dot(s.pose.q,s.pose.q)-1)>.001) {
            result.reason="Invalid controller calibration sample.";return result;
        }
        ++counts[s.device];pairs.push_back({s.camera,s.pose.p});
    }
    for(int d=1;d<=2;++d)if(counts[d]<18) {
        result.reason=std::string(d==1?"Left":"Right")+" controller needs more steady samples. Keep that wrist visible and pause at each pose. Head visibility is not required.";
        return result;
    }
    auto initial=calibrate(pairs);
    if(initial.spread<.07)return initial;
    Rigid r=initial.transform;
    if(floor && floor->valid)r=floorAligned(r,*floor);
    std::array<V3,3> candidate{};
    for(int iteration=0;iteration<60;++iteration) {
        Matrix h{};Vector b{},step{};normal(samples,r,candidate,h,b);
        if(floor && floor->valid)for(int fixed:{0,2,4}) {
            // Floor fixes world roll/pitch and height; only yaw/X/Z remain.
            for(int j=0;j<N;++j)h[fixed][j]=h[j][fixed]=0;
            h[fixed][fixed]=1;b[fixed]=0;
        }
        if(!solve(h,b,step)) {
            result.reason="Point controllers in different directions: forwards, out to the sides, then up. Keep the same grip and wrists visible. Exact angles and head visibility are not required.";
            return result;
        }
        V3 rotation{step[0],step[1],step[2]};
        double before=cost(samples,r,candidate),factor=std::min(1.,.3/std::max(norm(rotation),1e-9));
        bool improved=false;
        for(int trial=0;trial<10;++trial,factor*=.5) {
            auto next=r;auto nextOffsets=candidate;
            if(norm(rotation)>1e-12)next.q=normalized(axisAngle(unit(rotation),norm(rotation)*factor)*r.q);
            next.t+=V3{step[3],step[4],step[5]}*factor;
            for(int d=1;d<=2;++d)nextOffsets[d]+=V3{step[6+(d-1)*3],step[7+(d-1)*3],step[8+(d-1)*3]}*factor;
            double after=cost(samples,next,nextOffsets);
            if(after<=before+1e-14){r=next;candidate=nextOffsets;improved=true;break;}
        }
        double size=0;for(double v:step)size+=v*v;
        if(!improved || size*factor*factor<1e-14)break;
    }
    for(int d=1;d<=2;++d)if(!finite(candidate[d]) || norm(candidate[d])>.30) {
        result.reason="Controller-to-wrist estimate is implausible. Keep wrists clear of the torso and maintain a normal grip through all poses.";return result;
    }
    pairs.clear();for(auto &s:samples)pairs.push_back({s.camera,s.pose.p+s.pose.q.rotate(candidate[s.device])});
    offsets=candidate; // Diagnostic candidate only; callers apply it only after acceptance.
    result=floor && floor->valid?calibrateWithFloor(pairs,*floor):calibrate(pairs);
    if(!result.valid) {
        auto guidance=result.reason.find(" Keep hands visible");
        if(guidance!=std::string::npos)result.reason.resize(guidance);
        result.reason="Controller-only alignment did not agree. Keep wrists visible, hold each pose, and turn both controllers as directed by the palm instructions. "+result.reason;
        return result;
    }
    // A good hand must not hide a bad hand in the combined acceptance fraction.
    for(int d=1;d<=2;++d) {
        int inliers=0;double squared=0;
        for(size_t i=0;i<samples.size();++i)if(samples[i].device==d) {
            double error=norm(result.transform.apply(pairs[i].camera)-pairs[i].world);
            if(error<calibrationInlierDistance){++inliers;squared+=error*error;}
        }
        if(double(inliers)/counts[d]<calibrationMinInlierFraction || !inliers || std::sqrt(squared/inliers)>=calibrationMaxRms) {
            result.valid=false;result.reason=std::string(d==1?"Left":"Right")+" wrist samples disagree. Keep this wrist visible, away from your torso, and repeat the guided poses.";return result;
        }
    }
    result.reason="Controller-only alignment accepted; wrist offsets learned automatically. Your head did not need to be visible.";
    if(floor && floor->valid)result.reason+=" Floor height and tilt matched to SteamVR.";
    return result;
}
}
