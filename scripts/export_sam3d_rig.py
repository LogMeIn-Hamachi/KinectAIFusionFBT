"""Export the separately supplied Meta MHR TorchScript rig, development only."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/"artifacts/sam3d/rig-fp32"
out.mkdir(exist_ok=True)
rig=torch.jit.load(str(ROOT/"mhr_model.pt"),map_location="cpu").eval()
(out/"forward.txt").write_text(rig.code)
print(rig.code,flush=True)
args=(torch.zeros(1,45),torch.zeros(1,204),torch.zeros(1,72))
class FixedRig(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rig=rig
    def forward(self,shape,pose,expression):
        return self.rig(shape,pose,expression,True)
with torch.no_grad():
    wrapper=FixedRig().eval()
    result=wrapper(*args)
    print([x.shape for x in result],flush=True)
    traced=torch.jit.trace(wrapper,args,check_trace=False)
    print("Rig traced",flush=True)
    from torch._export.converter import TS2EPConverter
    program=TS2EPConverter(traced,args).convert()
    print("Rig converted to ExportedProgram",flush=True)
    torch.onnx.export(program,(),str(out/"rig.onnx"),input_names=["shape","pose","expression"],
                      output_names=["vertices_cm","skeleton_state_cm"],opset_version=18,dynamo=True,external_data=True)
print("Rig exported",flush=True)
