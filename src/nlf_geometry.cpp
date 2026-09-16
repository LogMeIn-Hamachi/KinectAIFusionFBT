#include "nlf_geometry.hpp"
namespace kf {
NlfWarp nlfWarp(Crop c,const Sam3dCamera &camera) {
    if(!camera.valid || !std::isfinite(c.cx) || !std::isfinite(c.cy) || !std::isfinite(c.w) ||
       !std::isfinite(c.h) || c.w<=1 || c.h<=1 || c.w>4000 || c.h>4000 ||
       std::abs(c.cx)>4000 || std::abs(c.cy)>4000)throw std::runtime_error("Invalid NLF camera/crop");
    const auto &k=camera.intrinsics;
    const double fx=k[0],fy=k[4],cx=k[2],cy=k[5];
    if(fx<=0 || fy<=0)throw std::runtime_error("Invalid NLF focal length");
    auto ray=[&](V2 p){return V3{(p.x-cx)/fx,(p.y-cy)/fy,1};};
    auto z=unit(ray({c.cx,c.cy})), x=unit(cross(z,{0,-1,0})), y=cross(z,x);
    NlfWarp out;out.rotation={x,y,z};
    auto project=[&](V2 p){auto r=ray(p);double d=dot(z,r);return V2{fx*dot(x,r)/d+cx,fy*dot(y,r)/d+cy};};
    auto top=project({c.cx,c.cy-c.h/2}),bottom=project({c.cx,c.cy+c.h/2});
    auto left=project({c.cx-c.w/2,c.cy}),right=project({c.cx+c.w/2,c.cy});
    double scale=256/std::max(std::hypot(top.x-bottom.x,top.y-bottom.y),std::hypot(left.x-right.x,left.y-right.y));
    out.pyramidLevel=std::clamp(int(std::floor(-std::log2(scale))),0,2);
    out.intrinsics={float(fx*scale),0,128,0,float(fy*scale),128,0,0,1};
    // K_original R_transpose K_crop_inverse maps crop pixel to original pixel.
    const auto &nk=out.intrinsics;
    V3 columns[]{x/nk[0],y/nk[4],z-x*(128/nk[0])-y*(128/nk[4])};
    for(int row=0;row<3;++row) {
        V3 r=row==0?V3{fx,0,cx}:row==1?V3{0,fy,cy}:V3{0,0,1};
        out.sourceProjection[row]={dot(r,columns[0]),dot(r,columns[1]),dot(r,columns[2])};
    }
    return out;
}
std::vector<float> nlfImage(const Frame &f,const NlfWarp &warp) {
    if(f.width<4 || f.height<4 || f.width>1920 || f.height>1080 ||
       f.bgra.size()!=size_t(f.width)*f.height*4)throw std::runtime_error("Invalid NLF RGB image");
    static const auto linear=[] {std::array<double,256> v{};for(int i=0;i<256;++i)v[i]=std::pow(i/255.,2.2);return v;}();
    const int factor=1<<warp.pyramidLevel,width=f.width/factor,height=f.height/factor;
    std::vector<float> pyramid;
    if(factor>1) {
        pyramid.resize(width*height*3);
        for(int y=0;y<height;++y)for(int x=0;x<width;++x)for(int ch=0;ch<3;++ch) {
            double sum=0;for(int dy=0;dy<factor;++dy)for(int dx=0;dx<factor;++dx)
                sum+=linear[f.bgra[((y*factor+dy)*f.width+x*factor+dx)*4+2-ch]];
            pyramid[(y*width+x)*3+ch]=float(sum/(factor*factor));
        }
    }
    std::vector<float> out(3*256*256);
    for(int y=0;y<256;++y)for(int x=0;x<256;++x) {
        V3 p{double(x),double(y),1};double d=dot(warp.sourceProjection[2],p);
        double u=dot(warp.sourceProjection[0],p)/d,v=dot(warp.sourceProjection[1],p)/d;
        u=(u+.5)/factor-.5;v=(v+.5)/factor-.5;
        if(!std::isfinite(u) || !std::isfinite(v) || std::abs(u)>100000 || std::abs(v)>100000)
            throw std::runtime_error("Invalid NLF image warp");
        int ix=int(std::floor(u)),iy=int(std::floor(v));double wx=u-ix,wy=v-iy;
        for(int ch=0;ch<3;++ch) {
            double value=0;
            for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx) {
                int xx=ix+dx,yy=iy+dy;
                double sample=1.;
                if(xx>=0 && xx<width && yy>=0 && yy<height)
                    sample=factor>1?pyramid[(yy*width+xx)*3+ch]:linear[f.bgra[(yy*f.width+xx)*4+2-ch]];
                value+=sample*(dx?wx:1-wx)*(dy?wy:1-wy);
            }
            // Gamma conversion runs on the GPU with the network.
            out[(ch*256+y)*256+x]=float(std::clamp(value,0.,1.));
        }
    }
    return out;
}
BodyPrediction nlfDecode(std::span<const float> data,const NlfWarp &warp,const Sam3dCamera &camera,double host) {
    if(data.size()!=J*6 || !camera.valid || !std::isfinite(host))throw std::runtime_error("Invalid NLF output");
    for(float v:data)if(!std::isfinite(v))throw std::runtime_error("Nonfinite NLF output");
    const auto &k=warp.intrinsics;
    std::array<V2,J> rays{},rhs{};std::array<V3,J> relative{};std::array<bool,J> valid{};
    double sum2=0,sumB=0,count=0;
    for(int j=0;j<J;++j) {
        auto p=data.subspan(j*6,6);
        rays[j]={(p[0]-128)/k[0],(p[1]-128)/k[4]};relative[j]={p[2],p[3],p[4]};
        rhs[j]={rays[j].x*p[4]-p[2],rays[j].y*p[4]-p[3]};
        valid[j]=p[0]>=32 && p[0]<=224 && p[1]>=32 && p[1]<=224 && p[5]<.3;
        if(valid[j]){sum2+=rays[j].x*rays[j].x+rays[j].y*rays[j].y;sumB+=rhs[j].x*rhs[j].x+rhs[j].y*rhs[j].y;++count;}
    }
    if(count<3)throw std::runtime_error("NLF needs more reliable visible body points");
    // Same normalized weighted least squares and L2 regularization as NLF.
    double scale2=std::sqrt(sum2/count+1e-10),scaleB=std::sqrt(sumB/count+1e-10);
    double system[3][4]{};for(int a=0;a<3;++a)system[a][a]=1e-4;
    for(int j=0;j<J;++j)for(int axis=0;axis<2;++axis) {
        double a[]{axis==0?1.:0.,axis==1?1.:0.,-(axis==0?rays[j].x:rays[j].y)/scale2};
        double b=(axis==0?rhs[j].x:rhs[j].y)/scaleB,w=valid[j]?1.+1e-8:1e-8;
        for(int row=0;row<3;++row){for(int col=0;col<3;++col)system[row][col]+=w*a[row]*a[col];system[row][3]+=w*a[row]*b;}
    }
    for(int col=0;col<3;++col) {
        int pivot=col;for(int row=col+1;row<3;++row)if(std::abs(system[row][col])>std::abs(system[pivot][col]))pivot=row;
        if(std::abs(system[pivot][col])<1e-12)throw std::runtime_error("Degenerate NLF perspective solve");
        for(int c=col;c<4;++c)std::swap(system[col][c],system[pivot][c]);
        double divisor=system[col][col];for(int c=col;c<4;++c)system[col][c]/=divisor;
        for(int row=0;row<3;++row)if(row!=col){double factor=system[row][col];for(int c=col;c<4;++c)system[row][c]-=factor*system[col][c];}
    }
    V3 ref{system[0][3]*scaleB,system[1][3]*scaleB,system[2][3]*scaleB/scale2};
    BodyPrediction out;out.host=host;
    for(int j=0;j<J;++j) {
        V3 p=relative[j]+ref;double z=std::max(.1,p.z);
        double u=k[0]*p.x/z+128,v=k[4]*p.y/z+128;
        if(p.z>.001 && u>=19.2 && u<=236.8 && v>=19.2 && v<=236.8)
            p=(p+V3{rays[j].x*p.z,rays[j].y*p.z,p.z})*.5;
        p=warp.rotation[0]*p.x+warp.rotation[1]*p.y+warp.rotation[2]*p.z;
        if(!finite(p) || p.z<=.1 || norm(p)>15)throw std::runtime_error("Implausible NLF camera pose");
        out.cameraPoints[j]=p;out.available[j]=data[j*6+5]<.3;
        out.landmarks[j].uv={camera.intrinsics[0]*p.x/p.z+camera.intrinsics[2],camera.intrinsics[4]*p.y/p.z+camera.intrinsics[5]};
    }
    return out;
}
}
