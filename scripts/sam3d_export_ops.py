"""Equivalent index-copy assemblies for the legacy ONNX exporter.

The pinned upstream code uses advanced-index writes, which Torch 2.11's legacy
exporter emits with missing operands. Keep vendor sources unchanged.
"""
import torch
from sam_3d_body.models.modules import mhr_utils as u
from sam_3d_body.models.heads import mhr_head as h


def body(cont):
    threes=u._batchXYZfrom6D_jit(cont[..., :138].unflatten(-1,(-1,6))).flatten(-2,-1)
    ones=cont[...,138:254].unflatten(-1,(-1,2))
    ones=torch.atan2(ones[...,0],ones[...,1])
    idx=u._get_body_cached_idx(cont.device)
    out=cont.new_zeros(*cont.shape[:-1],133)
    out=out.index_copy(-1,idx['3dof_flat'],threes)
    out=out.index_copy(-1,idx['1dof_rot'],ones)
    return out.index_copy(-1,idx['1dof_trans'],cont[...,254:])


def hand(cont):
    idx=u._get_cached_idx(cont.device)
    threes=u._batchXYZfrom6D_jit(cont.index_select(-1,idx['cont_3dof']).unflatten(-1,(-1,6))).flatten(-2,-1)
    ones=cont.index_select(-1,idx['cont_1dof']).unflatten(-1,(-1,2))
    ones=torch.atan2(ones[...,0],ones[...,1])
    out=cont.new_zeros(*cont.shape[:-1],27).index_copy(-1,idx['param_3dof'],threes)
    return out.index_copy(-1,idx['param_1dof'],ones)


def install():
    # Check the replacements against the original before monkey-patching.
    generator=torch.Generator().manual_seed(719)
    for count,original,replacement in [(260,h.compact_cont_to_model_params_body_fast,body),
                                       (54,h.compact_cont_to_model_params_hand_fast,hand)]:
        x=torch.randn(5,count,generator=generator)
        torch.testing.assert_close(original(x),replacement(x),rtol=0,atol=0)
    h.compact_cont_to_model_params_body_fast=body
    h.compact_cont_to_model_params_hand_fast=hand
