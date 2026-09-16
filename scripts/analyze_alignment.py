"""Development-only diagnosis of exported 3D alignment samples; never edits application settings."""
import csv, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'third_party/dev_python'))
import numpy as np

def rotation(q):
    w,x,y,z=q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)], [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)], [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
def exp(v):
    a=np.linalg.norm(v)
    if a<1e-12:return np.eye(3)
    s=skew(v/a)
    return np.eye(3)+np.sin(a)*s+(1-np.cos(a))*(s@s)

rows=list(csv.DictReader(open(sys.argv[1],newline='')))
names=['Headset','Left controller','Right controller']
dev=np.array([names.index(r['device']) for r in rows])
def cols(keys):return np.array([[float(r[k]) for k in keys] for r in rows])
x=cols(['camera_x','camera_y','camera_z']);p=cols(['vr_x','vr_y','vr_z'])
q=np.array([rotation(v) for v in cols(['vr_qw','vr_qx','vr_qy','vr_qz'])])
o=cols(['offset_x','offset_y','offset_z'])
host=cols(['host']).ravel(); phase=np.minimum(4,((host-host.min())/5).astype(int))

def fit(mask, offsets=False):
    xx,pp,qq,dd,oo=x[mask],p[mask],q[mask],dev[mask],o[mask]
    target=pp+np.einsum('nij,nj->ni',qq,oo)
    u,s,v=np.linalg.svd((xx-xx.mean(0)).T@(target-target.mean(0)))
    r=v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T;t=target.mean(0)-r@xx.mean(0)
    extra=np.zeros((3,3));dim=12 if offsets else 6
    for iteration in range(100):
        rx=xx@r.T
        residual=rx+t-pp-np.einsum('nij,nj->ni',qq,oo+extra[dd])
        errors=np.linalg.norm(residual,axis=1);weight=np.sqrt(np.minimum(1,.04/np.maximum(errors,1e-12)))
        j=np.zeros((len(xx),3,dim))
        for n in range(len(xx)):
            j[n,:,:3]=-skew(rx[n]);j[n,:,3:6]=np.eye(3)
            if offsets and dd[n]>0:j[n,:,3+dd[n]*3:6+dd[n]*3]=-qq[n]
        a=(j*weight[:,None,None]).reshape(-1,dim);b=(-residual*weight[:,None]).ravel()
        step=np.linalg.lstsq(a,b,rcond=None)[0]
        # Trust region for initial large rotations/offset compensation.
        scale=min(1,.1/max(1e-12,np.linalg.norm(step[:3])),.1/max(1e-12,np.linalg.norm(step[3:])))
        step*=scale;r=exp(step[:3])@r;t+=step[3:6]
        if offsets:extra[1]+=step[6:9];extra[2]+=step[9:12]
        if np.linalg.norm(step)<1e-9:break
    singular=np.linalg.svd(a,compute_uv=False)
    return r,t,extra,dict(condition=float(singular[0]/singular[-1]),iterations=iteration+1)

def metrics(model,mask):
    r,t,extra,info=model
    e=np.linalg.norm(x@r.T+t-p-np.einsum('nij,nj->ni',q,o+extra[dev]),axis=1)
    return {name:{'n':int(sum(mask&(dev==d))),'median_cm':float(np.median(e[mask&(dev==d)]))*100,'p95_cm':float(np.percentile(e[mask&(dev==d)],95))*100,'within_10cm_pct':float(np.mean(e[mask&(dev==d)]<.1))*100} for d,name in enumerate(names) if any(mask&(dev==d))}
allmask=np.ones(len(rows),dtype=bool)
report={'duration_s':float(host.max()-host.min()),'phase_counts':{int(i):int(sum(phase==i)) for i in range(5)}}
for mode in [False,True]:
    model=fit(allmask,mode)
    report['fit_offsets' if mode else 'fixed_offsets']={'metrics':metrics(model,allmask),'offset_delta_m':model[2].tolist(), 'rotation':model[0].tolist(),'translation':model[1].tolist(),**model[3]}
report['leave_one_phase_out']=[]
for i in range(5):
    train=phase!=i;test=phase==i
    if sum(test)<3:continue
    model=fit(train,True)
    report['leave_one_phase_out'].append({'phase':i,'offset_delta_m':model[2].tolist(),'metrics':metrics(model,test),**model[3]})
report['shared_local_z_hypotheses']=[]
original_o=o.copy()
for z in [.097,.15,.20,.25,.30]:
    o[dev>0,2]=z
    model=fit(allmask,False)
    report['shared_local_z_hypotheses'].append({'local_z_m':z,'metrics':metrics(model,allmask)})
o[:]=original_o
report['head_height_hypotheses']=[]
for head_y in [-.1,0,.05,.1]:
    for hand_z in [.097,.1491,.20]:
        o[:]=original_o;o[dev==0,1]=head_y;o[dev>0,2]=hand_z
        model=fit(allmask,False)
        r,t,extra,_=model
        errs=np.linalg.norm(x@r.T+t-p-np.einsum('nij,nj->ni',q,o),axis=1)
        report['head_height_hypotheses'].append({'head_y':head_y,'hand_z':hand_z,'inlier_rms_cm':float(np.sqrt(np.mean(errs[errs<.1]**2)))*100,'inlier_fraction':float(np.mean(errs<.1)),'metrics':metrics(model,allmask)})
o[:]=original_o
def production_fit(offsets):
    target=p+np.einsum('nij,nj->ni',q,offsets)
    weight=np.ones(len(x))
    for _ in range(8):
        ac=np.average(x,axis=0,weights=weight);bc=np.average(target,axis=0,weights=weight)
        u,sv,v=np.linalg.svd(((x-ac)*weight[:,None]).T@(target-bc))
        r=v.T@np.diag([1,1,np.linalg.det(v.T@u.T)])@u.T;t=bc-r@ac
        e=np.linalg.norm(x@r.T+t-target,axis=1);weight=np.minimum(1,.035/np.maximum(e,1e-12))
    return (r,t,np.zeros((3,3)),{}),e
report['production_reproduction']={}
for name in ['original','model_handgrip']:
    o[:]=original_o
    if name=='model_handgrip':o[dev>0]=[0,.003,.097]
    model,err=production_fit(o)
    rms=float(np.sqrt(np.mean(err[err<.1]**2)))
    fraction=float(np.mean(err<.1))
    report['production_reproduction'][name]={'rms_cm':rms*100,'consistent_percent':fraction*100,'accepted':rms<.06 and fraction>=.7,'metrics':metrics(model,allmask),'max_difference_exported_error_m':float(np.max(np.abs(err-cols(['error_m']).ravel()))) if name=='original' else None}
o[:]=original_o
print(json.dumps(report,indent=2))
