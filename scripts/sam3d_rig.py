"""Exportable MHR math specialized to the vertices used by MHR70.

Equations follow Meta Momentum (MIT), commit
11172f05996c1009cae6c5ae8e01654d4f2e270b. Assets remain under their original
licenses. No weights are approximated; unused vertex rows are removed.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def multiply(a,b):
    x,y,z,w=a.unbind(-1);X,Y,Z,W=b.unbind(-1)
    return torch.stack((w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z),dim=-1)


def xyz_quat(e):
    x,y,z=(e*.5).unbind(-1)
    sx,sy,sz=torch.sin(x),torch.sin(y),torch.sin(z)
    cx,cy,cz=torch.cos(x),torch.cos(y),torch.cos(z)
    return torch.stack((sx*cy*cz-cx*sy*sz,cx*sy*cz+sx*cy*sz,cx*cy*sz-sx*sy*cz,cx*cy*cz+sx*sy*sz),-1)


def rotate(q,v):
    q=F.normalize(q,dim=-1)
    a=q[...,:3]
    uv=torch.linalg.cross(a.expand_as(v),v)
    return v+2*(q[...,3:4]*uv+torch.linalg.cross(a.expand_as(uv),uv))


def compose(a,b):
    aq=F.normalize(a[...,3:7],dim=-1);bq=F.normalize(b[...,3:7],dim=-1)
    return torch.cat((a[...,:3]+rotate(aq,b[...,:3]*a[...,7:8]),multiply(aq,bq),a[...,7:8]*b[...,7:8]),-1)


class LandmarkRig(nn.Module):
    def __init__(self,original,mapping):
        super().__init__()
        state=original.state_dict()
        ids=torch.nonzero(mapping[:70,:18439].abs().sum(0),as_tuple=False).flatten().cpu()
        self.vertex_count=len(ids)
        self.register_buffer("vertex_ids",ids)
        def buf(name,key,select=None):
            data=state[key].detach().cpu()
            if select is not None:data=select(data)
            self.register_buffer(name,data.clone())
        buf("parameter_transform","character_torch.parameter_transform.parameter_transform")
        buf("offsets","character_torch.skeleton.joint_translation_offsets")
        buf("prerotations","character_torch.skeleton.joint_prerotations")
        buf("inverse_bind","character_torch.linear_blend_skinning.inverse_bind_pose")
        buf("base","character_torch.blend_shape.base_shape",lambda x:x[ids])
        buf("shape_vectors","character_torch.blend_shape.shape_vectors",lambda x:x[:,ids].reshape(45,-1))
        buf("face_vectors","face_expressions_model.shape_vectors",lambda x:x[:,ids].reshape(72,-1))
        sparse=getattr(original.pose_correctives_model.pose_dirs_predictor,"0")
        dense=torch.sparse_coo_tensor(sparse.sparse_indices.cpu(),sparse.sparse_weight.cpu(),tuple(sparse.sparse_shape)).to_dense()
        self.register_buffer("corrective_hidden",dense)
        buf("corrective_output","pose_correctives_model.pose_dirs_predictor.2.weight",lambda x:x.reshape(18439,3,-1)[ids].reshape(len(ids)*3,-1))
        # Fixed prefix multiplication groups preserve the released rig's FK order.
        groups=torch.split(state['character_torch.skeleton.pmi'].cpu(),list(original.character_torch.skeleton._pmi_buffer_sizes),dim=1)
        self.group_count=len(groups)
        for i,g in enumerate(groups):
            self.register_buffer(f"sources_{i}",g[0].clone())
            self.register_buffer(f"parents_{i}",g[1].clone())
        remap={int(v):i for i,v in enumerate(ids)}
        influences=[[] for _ in ids]
        v=state['character_torch.linear_blend_skinning.vert_indices_flattened'].cpu().tolist()
        j=state['character_torch.linear_blend_skinning.skin_indices_flattened'].cpu().tolist()
        w=state['character_torch.linear_blend_skinning.skin_weights_flattened'].cpu().tolist()
        for vi,ji,wi in zip(v,j,w):
            if vi in remap:influences[remap[vi]].append((ji,wi))
        width=max(map(len,influences))
        skin_index=torch.zeros(len(ids),width,dtype=torch.int64)
        skin_weight=torch.zeros(len(ids),width,dtype=torch.float32)
        for i,rows in enumerate(influences):
            for k,(ji,wi) in enumerate(rows):skin_index[i,k]=ji;skin_weight[i,k]=wi
        self.register_buffer("skin_index",skin_index)
        self.register_buffer("skin_weight",skin_weight)
        self.register_buffer("mapping",torch.cat((mapping[:70,ids.to(mapping.device)].cpu(),mapping[:70,18439:].cpu()),dim=1).float())

    def forward(self,shape,pose,expression=None,apply_correctives=True):
        parameters=torch.cat((pose,torch.zeros_like(shape)),dim=1)
        joint=(parameters@self.parameter_transform.T).reshape(-1,127,7)
        q=multiply(self.prerotations[None],xyz_quat(joint[...,3:6]))
        skel=torch.cat((joint[...,:3]+self.offsets[None],q,torch.exp(joint[...,6:7]*0.69314718246459961)),dim=-1)
        for i in range(self.group_count):
            sources=getattr(self,f"sources_{i}");parents=getattr(self,f"parents_{i}")
            merged=compose(skel[:,parents],skel[:,sources])
            skel=skel.index_copy(1,sources,merged)
        unposed=(shape@self.shape_vectors).reshape(-1,self.vertex_count,3)+self.base[None]
        if expression is not None:
            unposed=unposed+(expression@self.face_vectors).reshape(-1,self.vertex_count,3)
        if apply_correctives:
            x,y,z,w=xyz_quat(joint[:,2:,3:6]).unbind(-1)
            # First two columns of XYZ rotation, column-major, minus identity.
            features=torch.stack((-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y),
                                  2*(x*y-w*z),-2*(x*x+z*z),2*(y*z+w*x)),dim=-1).flatten(1)
            hidden=F.relu(features@self.corrective_hidden.T)
            unposed=unposed+(hidden@self.corrective_output.T).reshape(-1,self.vertex_count,3)
        skin=compose(skel,self.inverse_bind[None])[:,self.skin_index]
        template=unposed[:,:,None,:].expand(-1,-1,self.skin_index.shape[1],-1)
        transformed=skin[...,:3]+rotate(skin[...,3:7],template*skin[...,7:8])
        vertices=(transformed*self.skin_weight[None,:,:,None]).sum(dim=2)
        return vertices,skel


def validate_rig(original,reduced):
    device=next(reduced.buffers()).device
    generator=torch.Generator(device=device).manual_seed(419)
    reports=[]
    with torch.no_grad():
        for i in range(5):
            shape=torch.randn(1,45,generator=generator,device=device)*(.1 if i else 0)
            pose=torch.randn(1,204,generator=generator,device=device)*(.15 if i else 0)
            expr=torch.randn(1,72,generator=generator,device=device)*.02
            for correction in (False,True):
                v,s=original(shape,pose,expr,correction)
                a,b=reduced(shape,pose,expr,correction)
                ve=(a-v[:,reduced.vertex_ids]).abs().max().item()
                se=(b-s).abs().max().item()
                reports.append({"pose":i,"correctives":correction,"max_vertex_error_cm":ve,"max_skeleton_component_error":se})
                if ve>.003 or se>.003:raise RuntimeError(f"Rig mismatch {reports[-1]}")
    return reports
