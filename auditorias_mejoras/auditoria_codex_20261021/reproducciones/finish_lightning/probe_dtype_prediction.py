import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch
from graph2mat import BasisConfiguration
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.bindings.torch.load import load_from_lit_ckpt
ROOT=Path(__file__).parent
checkpoint=ROOT.parent/'lightning'/'float64.ckpt'
state=torch.load(checkpoint,weights_only=False)['state_dict']
torch.set_default_dtype(torch.float64)
exact,processor=load_from_lit_ckpt(checkpoint)
assert all(torch.equal(v,exact.state_dict()[k]) for k,v in state.items())
torch.set_default_dtype(torch.float32)
rounded,_=load_from_lit_ckpt(checkpoint)
rounded.double()
torch.set_default_dtype(torch.float64)
cfg=BasisConfiguration(point_types=['A','A'],positions=np.array([[0.,0.,0.],[.7,0.,0.]]),basis=processor.basis_table,cell=np.eye(3)*10,pbc=(False,False,False),matrix=np.array([[2.,3.],[3.,4.]]))
data=TorchBasisMatrixData.new(cfg,processor)
batch=Batch.from_data_list([data])
exact.eval();rounded.eval()
with torch.no_grad():
 p0=exact(batch);p1=rounded(batch)
result={key:float((p0[key]-p1[key]).abs().max()) for key in ('node_labels','edge_labels')}
result['saved_floating_dtypes']=sorted({str(v.dtype) for v in state.values() if v.is_floating_point()})
result['all_float64_reload_matches_saved_state']=True
result['rounded_then_double_matches_saved_state']=all(torch.equal(v,rounded.state_dict()[k]) for k,v in state.items())
print(json.dumps(result))
(ROOT/'dtype_prediction_result.json').write_text(json.dumps(result,indent=2))
