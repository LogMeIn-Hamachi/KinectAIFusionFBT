#include "sam3d_geometry.hpp"

namespace kf {
V3 Sam3dCamera::toSdk(V3 p) const {
    p = p - translation;
    return rows[0] * p.x + rows[1] * p.y + rows[2] * p.z;
}
Sam3dCamera sam3dCamera(const ColorProjection &projection) {
    Sam3dCamera out;
    if (!projection.valid) return out;
    const auto &p=projection.p;
    V3 a{p[0]*projection.width,p[1]*projection.width,p[2]*projection.width}, b{p[4]*projection.height,p[5]*projection.height,p[6]*projection.height}, c{p[8],p[9],1};
    double scale=norm(c);
    auto z=c/scale;
    double cy=dot(b/scale,z);
    auto y=b/scale-z*cy;
    double fy=norm(y);
    if(fy<100) return out;
    y=y/fy;
    double cx=dot(a/scale,z), skew=dot(a/scale,y);
    auto x=a/scale-z*cx-y*skew;
    double fx=norm(x);
    if(fx<100 || fx>6000 || fy>6000 || cx<0 || cx>projection.width || cy<0 || cy>projection.height || std::abs(skew)>2*std::max(1.,projection.width/640.))
        return out;
    x=x/fx;
    double tz=p[10]/scale;
    double ty=(p[7]*projection.height/scale-cy*tz)/fy;
    double tx=(p[3]*projection.width/scale-cx*tz-skew*ty)/fx;
    out.intrinsics={float(fx),0,float(cx),0,float(fy),float(cy),0,0,1};
    out.rows={x,y,z};out.translation={tx,ty,tz};
    out.valid=finite(out.translation) && norm(out.translation)<.15;
    return out;
}
Crop sam3dCrop(double x1, double y1, double x2, double y2) {
    for (double v : {x1, y1, x2, y2})
        if (!std::isfinite(v) || std::abs(v) > 100000)
            throw std::runtime_error("Invalid SAM 3D bounding box");
    if (x2 <= x1 || y2 <= y1)
        throw std::runtime_error("Empty SAM 3D bounding box");
    double w = (x2 - x1) * 1.25, h = (y2 - y1) * 1.25;
    h = std::max(h, w / .75);
    w = std::max(w, h * .75);
    double side = std::max(w, h);
    return {(x1 + x2) / 2, (y1 + y2) / 2, side, side};
}
std::vector<float> sam3dImage(const Frame &f, Crop c) {
    if (f.width <= 0 || f.height <= 0 || f.bgra.size() != size_t(f.width) * f.height * 4 ||
        !std::isfinite(c.cx) || !std::isfinite(c.cy) || !std::isfinite(c.w) ||
        !std::isfinite(c.h) || c.w <= 0 || c.h <= 0 || c.w > 400000 ||
        c.h > 400000 || std::abs(c.cx) > 100000 || std::abs(c.cy) > 100000)
        throw std::runtime_error("Invalid SAM 3D image/crop");
    std::vector<float> out(3 * 512 * 512);
    // Integer interpolation exactly matches the 5-bit OpenCV table, including
    // half-up output rounding and black borders, without per-channel float math.
    std::array<int,512> fixedX;
    for(int x=0;x<512;++x)
        fixedX[x]=(int(std::lrint((c.cx-c.w/2)*1024))+16+int(std::lrint(c.w/512*x*1024)))>>5;
    for(int y=0;y<512;++y) {
        int iy=(int(std::lrint((c.cy-c.h/2+c.h/512*y)*1024))+16)>>5;
        int y0=iy>>5,fy=iy&31;
        for(int x=0;x<512;++x) {
            int x0=fixedX[x]>>5,fx=fixedX[x]&31;
            const uint8_t black[4]{};
            const uint8_t *pixels[4]{black,black,black,black};
            for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx)
                if(x0+dx>=0 && x0+dx<f.width && y0+dy>=0 && y0+dy<f.height)
                    pixels[dy*2+dx]=f.bgra.data()+(size_t(y0+dy)*f.width+x0+dx)*4;
            const int w[]{(32-fx)*(32-fy),fx*(32-fy),(32-fx)*fy,fx*fy};
            for(int ch=0;ch<3;++ch) {
                int sum=0;
                for(int k=0;k<4;++k)sum+=pixels[k][2-ch]*w[k];
                out[(ch*512+y)*512+x]=float((sum+512)>>10)/255.f;
            }
        }
    }
    return out;
}
Keypoints sam3dLandmarks(const Sam3dPrediction &p) {
    // Official mhr70.py order -> internal Halpe26 order; -1 has no equivalent.
    constexpr std::array<int, J> index{0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41,
                                       9, 10, 11, 12, 13, 14, -1, 69, -1,
                                       15, 18, 16, 19, 17, 20};
    Keypoints out{};
    for (int j = 0; j < J; ++j)
        if (index[j] >= 0) {
            auto v = p.imagePoints[index[j]];
            if (!std::isfinite(v.x) || !std::isfinite(v.y))
                throw std::runtime_error("Nonfinite SAM 3D landmark");
            out[j].uv = v;
        }
    // These points are estimates; no fabricated 2D detector confidence.
    out[Hip].uv = {(out[LHip].uv.x + out[RHip].uv.x) / 2,
                   (out[LHip].uv.y + out[RHip].uv.y) / 2};
    return out;
}
std::array<std::optional<Q>,2> sam3dFootOrientations(const Sam3dPrediction& prediction,const Sam3dCamera& camera) {
    std::array<std::optional<Q>,2> out;
    if(prediction.hasFootRotations && camera.valid) {
        // Anatomical sole axes exported from the supplied neutral MHR rig and
        // official MHR70 mapping (export_foot_frames.py). Internal left/right
        // follow Kinect image labels, hence MHR right (23), left (7).
        constexpr V3 forward[2]{{-.976598799,-.211356401,-.039790560},{.976266026,.212391555,.042360999}};
        constexpr V3 right[2]{{-.013364970,-.125014797,.992064834},{-.018040622,-.115163431,.993182778}};
        auto vector=[&](V3 v){v={v.x,-v.y,-v.z};return camera.rows[0]*v.x+camera.rows[1]*v.y+camera.rows[2]*v.z;};
        for(int side=0;side<2;++side) {
            auto r=vector(prediction.footRotations[side].apply(right[side]));
            auto z=vector(prediction.footRotations[side].apply(forward[side]));
            if(!finite(r) || !finite(z) || std::abs(norm(r)-1)>.03 || std::abs(norm(z)-1)>.03 || std::abs(dot(r,z))>.03)continue;
            out[side]=basis(r,cross(z,r));
        }
    }
    return out;
}
Sam3dEvidence sam3dEvidence(const Frame &f,std::uint32_t id,const Sam3dPrediction &prediction,const Sam3dCamera &camera) {
    BodyPrediction generic;generic.host=prediction.host;
    generic.landmarks=sam3dLandmarks(prediction);
    constexpr std::array<int,J> map{0,1,2,3,4,5,6,7,8,62,41,9,10,11,12,13,14,-1,69,-1,15,18,16,19,17,20};
    for(int j=0;j<J;++j) if(map[j]>=0) {
        generic.cameraPoints[j]=prediction.cameraPoints[map[j]];generic.available[j]=true;
    }
    auto out=bodyPoseEvidence(f,id,generic,camera);
    if(out.prior.articulationValid) {
        auto feet=sam3dFootOrientations(prediction,camera);
        for(int side=0;side<2;++side)if(feet[side]) {
            out.prior.footOrientations[side]=*feet[side];out.prior.footOrientationValid[side]=true;
        }
    }
    return out;
}
Sam3dEvidence bodyPoseEvidence(const Frame &f,std::uint32_t id,const BodyPrediction &prediction,const Sam3dCamera &camera) {
    Sam3dEvidence out;
    auto body=std::find_if(f.bodies.begin(),f.bodies.end(),[&](const Body &b){return b.id==id;});
    if(!id || body==f.bodies.end() || !body->player || !camera.valid ||
       !std::isfinite(prediction.host) || !std::isfinite(f.host) ||
       std::abs(f.host-prediction.host)>.001 || f.colorIndex.size()!=size_t(f.width)*f.height) return out;
    out.keypoints=kinectImageLabels(prediction.landmarks);
    constexpr std::array<int,J> index{0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,17,18,19,21,20,23,22,25,24};
    auto &prior=out.prior;prior.host=f.host;prior.selectedId=id;
    for(int j=0;j<J;++j) if(prediction.available[index[j]]) {
        if(!finite(prediction.cameraPoints[index[j]])) return {};
        prior.points[j]=camera.toSdk(prediction.cameraPoints[index[j]]);
        prior.available[j]=true;
    }
    prior.points[Hip]=(prior.points[LHip]+prior.points[RHip])/2;
    prior.available[Hip]=true;
    // Reject implausible articulated predictions before allowing them to drive
    // positions. SDK limb coordinates/confidence are deliberately not consulted.
    for(auto [a,b]:std::array<std::pair<int,int>,8>{{{LShoulder,RShoulder},{LHip,RHip},
            {LHip,LKnee},{LKnee,LAnkle},{RHip,RKnee},{RKnee,RAnkle},
            {LShoulder,LElbow},{RShoulder,RElbow}}}) {
        if(!prior.available[a] || !prior.available[b])return out;
        double length=norm(prior.points[a]-prior.points[b]);
        if(length<.08 || length>.75)return out;
    }
    if(norm(prior.points[Neck]-prior.points[Hip])<.2 ||
       norm(prior.points[Neck]-prior.points[Hip])>.85)return out;
    prior.articulationValid=true;
    prior.cameraOrigin=camera.toSdk({});prior.cameraForward=camera.rows[2];
    prior.imageRoot=prior.points[Hip];
    auto median=[](std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];};
    const int pixelRadius=4*std::max(1,f.width/640);
    auto surface=[&](int j)->std::optional<V3> {
        if(!prior.available[j])return {};
        auto uv=out.keypoints[j].uv;
        if(uv.x<pixelRadius || uv.y<pixelRadius || uv.x>=f.width-pixelRadius || uv.y>=f.height-pixelRadius)return {};
        std::vector<double> x,y,z;
        for(int dy=-pixelRadius;dy<=pixelRadius;++dy)for(int dx=-pixelRadius;dx<=pixelRadius;++dx) {
            int u=int(std::lrint(uv.x))+dx,v=int(std::lrint(uv.y))+dy;
            if(u<0 || v<0 || u>=f.width || v>=f.height)continue;
            auto k=f.colorIndex[size_t(v)*f.width+u];
            if(k<0 || size_t(k)>=f.depth.size() || size_t(k)>=f.mapping.size() ||
               (f.depth[k]&7)!=body->player)continue;
            auto p=f.mapping[k];
            if(!finite(V3{p.x,p.y,p.z}) || p.z<.5 || p.z>4.5)continue;
            x.push_back(p.x);y.push_back(p.y);z.push_back(p.z);
        }
        if(z.size()<12)return {};
        std::sort(z.begin(),z.end());
        if(z[z.size()*9/10]-z[z.size()/10]>.08)return {};
        return V3{median(x),median(y),median(z)};
    };
    // Register one rigid translation from actual selected-person depth, not SDK
    // shoulders/knees/ankles. This allows SAM to disagree with wrong SDK limbs.
    // Radius is a surface-to-joint approximation, not an independent observation.
    std::vector<V3> offsets;
    std::vector<V3> anchors;
    for(int j:{Neck,LShoulder,RShoulder,LHip,RHip,LKnee,RKnee,LAnkle,RAnkle})
        if(auto p=surface(j)) {
            if(j==LAnkle || j==RAnkle) {
                prior.footSurfaces[j-LAnkle]=*p;
                prior.footSurfaceValid[j-LAnkle]=true;
            }
            double radius=(j==LHip || j==RHip)?.09:j==Neck?.06:.04;
            auto target=*p+unit(*p)*radius;
            offsets.push_back(target-prior.points[j]);anchors.push_back(target);
            prior.depthRoots[prior.depthRootCount++]=prior.imageRoot+target-prior.points[j];
        }
    if(offsets.size()<3)return out;
    // Max-consensus translation tolerates a hidden landmark projecting onto a
    // different part of the same body. Require spatially separated support.
    V3 shift;unsigned best=0;double bestError=1e9;
    for(auto candidate:offsets) {
        unsigned count=0;double error=0;
        for(auto d:offsets)if(norm(d-candidate)<.16){++count;error+=dot(d-candidate,d-candidate);}
        if(count>best || (count==best && error<bestError)){shift=candidate;best=count;bestError=error;}
    }
    if(best<3 || best*2<offsets.size())return out;
    std::vector<double> x,y,z;
    for(auto d:offsets)if(norm(d-shift)<.16){x.push_back(d.x);y.push_back(d.y);z.push_back(d.z);}
    shift={median(x),median(y),median(z)};
    // Refine the discrete median with a robust weighted mean. A small depth
    // change must not jump the whole body to whichever anchor wins a rank tie.
    for(int iteration=0;iteration<4;++iteration) {
        V3 sum{};double weight=0;
        for(auto offset:offsets) {
            double distance=norm(offset-shift);
            if(distance>=.16)continue;
            double w=std::min(1.,.04/std::max(distance,1e-9));
            sum+=offset*w;weight+=w;
        }
        if(weight>0)shift=sum/weight;
    }
    double squared=0,span=0;unsigned inliers=0;
    for(size_t i=0;i<offsets.size();++i)if(norm(offsets[i]-shift)<.16) {
        squared+=dot(offsets[i]-shift,offsets[i]-shift);++inliers;
        for(size_t k=0;k<i;++k)if(norm(offsets[k]-shift)<.16)span=std::max(span,norm(anchors[i]-anchors[k]));
    }
    if(inliers<3 || span<.25)return out;
    prior.rootAnchors=inliers;prior.registrationRms=std::sqrt(squared/inliers);
    if(prior.registrationRms>.10 || norm(shift)>.8)return out;
    for(auto &p:prior.points)p+=shift;
    for(int j=0;j<J;++j) {
        if(!prediction.available[index[j]]) continue;
        auto uv=out.keypoints[j].uv;unsigned nearby=0;
        if(uv.x<0 || uv.x>=f.width || uv.y<0 || uv.y>=f.height) continue;
        for(int dy=-pixelRadius;dy<=pixelRadius;++dy) for(int dx=-pixelRadius;dx<=pixelRadius;++dx) {
            int u=int(std::lrint(uv.x))+dx,v=int(std::lrint(uv.y))+dy;
            if(u<0 || v<0 || u>=f.width || v>=f.height)continue;
            auto k=f.colorIndex[size_t(v)*f.width+u];
            if(k<0 || size_t(k)>=f.depth.size() || size_t(k)>=f.mapping.size() || (f.depth[k]&7)!=body->player)continue;
            auto m=f.mapping[k];
            if(finite(V3{m.x,m.y,m.z}) && m.z>.5 && std::abs(m.z-prior.points[j].z)<.16)++nearby;
        }
        prior.supported[j]=nearby>=12;
        // A geometric support gate, not the network's confidence probability.
        if(prior.supported[j])out.keypoints[j].score=.8;
    }
    prior.supported[Hip]=prior.supported[LHip]&&prior.supported[RHip];
    prior.valid=true;
    return out;
}
} // namespace kf
