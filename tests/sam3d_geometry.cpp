#include "sam3d_geometry.hpp"
#include <iostream>
using namespace kf;
int main() {
    try {
        {
            Frame image;image.width=8;image.height=6;image.bgra.resize(8*6*4);
            for(size_t i=0;i<image.bgra.size();++i)image.bgra[i]=uint8_t(i*37);
            std::vector<float> reused;sam3dImage(image,{4,3,8,8},reused);
            const auto storage=reused.data();
            for(Crop crop: {Crop{4,3,8,8},Crop{0,0,16,16},Crop{40,40,8,8}}) {
                auto expected=sam3dImage(image,crop);
                std::fill(reused.begin(),reused.end(),-123.f);
                sam3dImage(image,crop,reused);
                if(reused!=expected || reused.data()!=storage)
                    throw std::runtime_error("Reused SAM crop changed pixels, border padding or buffer ownership");
            }
        }
        ColorProjection projection;
        projection.valid=true;
        // Reflected Y camera, 2 cm horizontal baseline. A proper rotation
        // decomposition would silently reverse one axis for this camera.
        projection.p={500./640,0,320./640,10./640,0,-510./480,240./480,0,0,0,0};
        auto camera=sam3dCamera(projection);
        if(!camera.valid) throw std::runtime_error("Reflected camera rejected");
        V3 sdk{.3,.7,2};
        V3 imageCamera{sdk.x+.02,-sdk.y,sdk.z};
        if(norm(camera.toSdk(imageCamera)-sdk)>1e-9)
            throw std::runtime_error("Reflected camera inverse");
        auto uv=projection.project(sdk);
        if(std::abs(uv.x-(500*imageCamera.x/imageCamera.z+320))>1e-8 ||
           std::abs(uv.y-(510*imageCamera.y/imageCamera.z+240))>1e-8)
            throw std::runtime_error("Camera reprojection");
        projection.valid=false;
        if(sam3dCamera(projection).valid) throw std::runtime_error("Invalid camera accepted");
        Sam3dPrediction prediction;
        for (int i = 0; i < 70; ++i) prediction.imagePoints[i] = {double(i), double(i * 2)};
        auto mapped = sam3dLandmarks(prediction);
        if (mapped[LWrist].uv.x != 62 || mapped[RWrist].uv.x != 41 ||
            mapped[LHeel].uv.x != 17 || mapped[RHeel].uv.x != 20 ||
            mapped[LToe].uv.x != 15 || mapped[RToe].uv.x != 18 ||
            mapped[LSmallToe].uv.x != 16 || mapped[RSmallToe].uv.x != 19 ||
            mapped[Hip].uv.x != 9.5 || mapped[Neck].uv.x != 69)
            throw std::runtime_error("MHR70 anatomical mapping");
        for (auto &p : mapped)
            if (p.score != 0) throw std::runtime_error("Inferred keypoint falsely marked observed");
        Frame frame;frame.host=1;frame.colorIndex.assign(640*480,-1);
        Body selected;selected.id=12;selected.player=1;
        // Deliberately unusable SDK joints: registration must depend on depth.
        for(auto &joint:selected.joints)joint={{1,0,3},0,.5,1};
        frame.bodies.push_back(selected);
        Sam3dCamera registered;registered.valid=true;registered.rows={V3{1,0,0},V3{0,1,0},V3{0,0,1}};
        Sam3dPrediction learned;learned.host=1;
        for(auto &point:learned.cameraPoints)point={0,.8,2};
        constexpr std::array<int,J> indices{0,2,1,4,3,6,5,8,7,41,62,10,9,12,11,14,13,-1,69,-1,18,15,19,16,20,17};
        std::array<V3,J> expected;
        for(auto &p:expected)p={0,.8,2};
        expected[Neck]={0,.65,2};
        for(int side=0;side<2;++side) {
            double sign=side?1:-1;
            expected[LShoulder+side]={sign*.2,.55,2};
            expected[LElbow+side]={sign*.3,.3,2};
            expected[LWrist+side]={sign*.3,.05,2};
            expected[LHip+side]={sign*.12,0,2};
            expected[LKnee+side]={sign*.12,-.45,2};
            expected[LAnkle+side]={sign*.12,-.86,2};
            expected[LHeel+side]={sign*.12,-.9,2.06};
            expected[LToe+side]={sign*.12,-.9,1.85};
            expected[LSmallToe+side]={sign*.16,-.9,1.87};
        }
        for(int j=0;j<J;++j)if(indices[j]>=0) {
            auto p=expected[j];int u=int(320+200*p.x),v=int(240-200*p.y);
            learned.cameraPoints[indices[j]]=p;learned.imagePoints[indices[j]]={double(u),double(v)};
            double radius=(j==LHip || j==RHip)?.09:j==Neck?.06:.04;
            V3 surface=p-unit(p)*radius;
            for(int dy=-4;dy<=4;++dy)for(int dx=-4;dx<=4;++dx) {
                frame.colorIndex[(v+dy)*640+u+dx]=int(frame.mapping.size());
                frame.mapping.push_back({float(surface.x),float(surface.y),float(surface.z),u+dx,v+dy});
                frame.depth.push_back(uint16_t(2000*8+2));
            }
        }
        auto evidence=sam3dEvidence(frame,12,learned,registered);
        if(evidence.prior.valid || evidence.keypoints[LToe].score!=0)
            throw std::runtime_error("Bystander depth accepted for learned pose");
        for(auto &depth:frame.depth)depth=uint16_t(2000*8+1);
        evidence=sam3dEvidence(frame,12,learned,registered);
        if(!evidence.prior.valid || !evidence.prior.supported[LShoulder] || evidence.prior.rootAnchors<3)
            throw std::runtime_error("Selected-player depth anchor rejected without SDK joints");
        auto registeredPose=evidence.prior;
        if(!registeredPose.footSurfaceValid[0] || !registeredPose.footSurfaceValid[1])
            throw std::runtime_error("Independent foot surfaces missing");
        {
            auto shifted=learned;
            for(auto& p:shifted.cameraPoints)p.x+=.12;
            auto result=sam3dEvidence(frame,12,shifted,registered);
            if(!result.prior.footSurfaceValid[0] ||
               norm(result.prior.footSurfaces[0]-registeredPose.footSurfaces[0])>1e-9)
                throw std::runtime_error("Shared model translation changed independent foot observation");
        }
        {
            Frame hd=frame;hd.width=1920;hd.height=1080;hd.depthWidth=512;hd.depthHeight=424;hd.sensorVersion=2;
            auto predictionHd=learned;
            for(auto &p:predictionHd.imagePoints){p.x*=3;p.y*=2.25;}
            for(auto &m:hd.mapping){m.u*=3;m.v=int(std::lround(m.v*2.25));}
            indexRegistration(hd);
            auto result=sam3dEvidence(hd,12,predictionHd,registered);
            if(!result.prior.valid || !result.prior.supported[LAnkle] || !result.prior.supported[RAnkle] ||
                norm(result.prior.points[LAnkle]-registeredPose.points[LAnkle])>1e-5)
                throw std::runtime_error("Full-HD sparse depth registration changed metric articulation");
            // An isolated sample is still not enough; never splat one measurement
            // into enough colour pixels to manufacture the required support count.
            for(size_t k=0;k<hd.depth.size();++k)if(k%81)hd.depth[k]&=~7;
            if(sam3dEvidence(hd,12,predictionHd,registered).prior.valid)
                throw std::runtime_error("Sparse v2 depth falsely counted as dense observations");
        }
        for(auto &joint:frame.bodies[0].joints)joint={{-.8,.3,3.5},.85,.04,1};
        evidence=sam3dEvidence(frame,12,learned,registered);
        if(!evidence.prior.valid || norm(evidence.prior.points[LAnkle]-registeredPose.points[LAnkle])>1e-9)
            throw std::runtime_error("Wrong confident SDK limbs changed SAM registration");
        // Occluded lower legs remain explicitly inferred, never invented depth.
        for(size_t k=0;k<frame.depth.size();++k)if(frame.mapping[k].v>380)frame.depth[k]&=~7;
        evidence=sam3dEvidence(frame,12,learned,registered);
        if(!evidence.prior.valid || evidence.prior.supported[LAnkle] || !evidence.prior.available[LAnkle] ||
           evidence.keypoints[LAnkle].score!=0 || evidence.prior.footSurfaceValid[0])
            throw std::runtime_error("Hidden ankle mislabeled as observed or entire pose suppressed");
        Estimator estimator;estimator.select(12);Calibration calibration;
        State state;
        for(int n=0;n<12;++n) {
            frame.host=learned.host=1+n/30.;
            evidence=sam3dEvidence(frame,12,learned,registered);
            state=estimator.process(frame,&evidence.keypoints,calibration,&evidence.prior);
        }
        if(state.body.joints[LAnkle].source!=5 || !state.learnedPosition[1] || !state.learnedDirection[1] ||
           norm(state.body.joints[LAnkle].p-evidence.prior.points[LAnkle])>.001)
            throw std::runtime_error("SAM articulation did not replace conflicting SDK ankle");
        if(state.contacts[0].state!=Contact::Air || state.trackers[1].positionSigma<.18)
            throw std::runtime_error("Hidden model foot falsely treated as measured contact");
        auto tilted=evidence.prior;
        auto tilt=axisAngle({1,0,0},pi/2);
        for(int j:{LHeel,LToe,LSmallToe})
            tilted.points[j]=tilted.points[LAnkle]+tilt.rotate(tilted.points[j]-tilted.points[LAnkle]);
        for(int n=0;n<24;++n) {
            frame.host+=1/30.;tilted.host=frame.host;
            state=estimator.process(frame,nullptr,calibration,&tilted);
        }
        auto direction=unit(tilted.points[LHeel]-tilted.points[LToe]);
        if(!state.learnedDirection[1] || dot(state.trackers[1].q.rotate({0,0,1}),direction)<.995)
            throw std::runtime_error("Vertical learned foot forced into floor orientation");
        Estimator resting;resting.select(12);
        for(int n=0;n<120;++n) {
            frame.host+=1/30.;auto noisy=tilted;noisy.host=frame.host;
            noisy.points[LAnkle].x+=(n%2?.01:-.01);
            auto quiet=resting.process(frame,nullptr,calibration,&noisy);
            if(n>60 && (norm(quiet.trackers[1].velocity)>1e-9 ||
               norm(resting.predict(frame.host+.04).trackers[1].p-quiet.trackers[1].p)>1e-9))
                throw std::runtime_error("Resting SAM noise extrapolated into outgoing position");
        }
        tilted.host=frame.host;
        for(int baseline:{0,1,2}) {
            Estimator comparison;comparison.select(12);comparison.settings.baseline=baseline;
            auto mismatched=tilted;
            if(baseline==0)mismatched.selectedId=99;
            auto result=comparison.process(frame,nullptr,calibration,&mismatched);
            if(result.learnedPosition[1] || result.body.joints[LAnkle].source==5)
                throw std::runtime_error("SAM bypassed identity or SDK comparison mode");
        }
        // Identity, freshness and mode gates must also hold at estimator boundary.
        auto stale=tilted;frame.host+=.2;
        for(auto &joint:frame.bodies[0].joints)joint.confidence=0;
        state=estimator.process(frame,nullptr,calibration,&stale);
        if(state.learnedPosition[1] || state.trackers[1].valid)
            throw std::runtime_error("Stale SAM pose continued output");
        learned.host=frame.host-.1;
        if(sam3dEvidence(frame,12,learned,registered).prior.valid)
            throw std::runtime_error("Stale learned pose accepted");
        learned.host=frame.host;frame.bodies.clear();
        if(sam3dEvidence(frame,12,learned,registered).prior.valid)
            throw std::runtime_error("Missing selected identity guessed from model");
        auto crop = sam3dCrop(0, 0, 300, 400);
        if (crop.cx != 150 || crop.cy != 200 || crop.w != 500 || crop.h != 500)
            throw std::runtime_error("SAM crop aspect/padding");
        bool rejected = false;
        try { sam3dCrop(20, 30, 10, 40); } catch (...) { rejected = true; }
        if (!rejected) throw std::runtime_error("Invalid bbox accepted");
        prediction.imagePoints[17].x = std::numeric_limits<double>::quiet_NaN();
        rejected = false;
        try { sam3dLandmarks(prediction); } catch (...) { rejected = true; }
        if (!rejected) throw std::runtime_error("Nonfinite landmark accepted");
        std::cout << "SAM geometry: landmark mapping, no invented confidence, bbox and finite checks passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
