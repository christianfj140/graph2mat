import warnings
warnings.filterwarnings('ignore')
import torch
from graph2mat.tools.lightning.tests.test_long_range import _model_and_batch
from graph2mat.models.long_range import ewald_potentials, equilibrate_charges

torch.set_default_dtype(torch.float32)
for mode in (False,True,'qeq'):
    model,batch,_,_=_model_and_batch(mode)
    for fp32 in (False,True):
        model.readout_fp32=fp32
        try:
            with torch.autocast('cpu',dtype=torch.bfloat16): out=model(batch)
            print('model autocast',mode,'readout_fp32',fp32,'success',out['node_labels'].dtype)
        except Exception as exc:
            print('model autocast',mode,'readout_fp32',fp32,type(exc).__name__,str(exc).splitlines()[0])
    if mode:
        feats=torch.rand(len(batch.point_types),model.charge_readout.irreps_in.dim)
        outs=dict(node_labels=torch.zeros(len(batch.point_labels)),edge_labels=torch.zeros(len(batch.edge_labels)))
        try:
            with torch.autocast('cpu',dtype=torch.bfloat16): out=model._long_range(batch,feats,outs)
            print('direct_long_range',mode,'success')
        except Exception as exc:
            print('direct_long_range',mode,type(exc).__name__,str(exc).splitlines()[0])

model,batch,_,_=_model_and_batch(False)
model.double(); batch.apply(lambda v:v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
for fp32 in (False,True):
    model.readout_fp32=fp32
    try: print('float64',fp32,'success',model(batch)['node_labels'].dtype)
    except Exception as exc: print('float64',fp32,type(exc).__name__,str(exc).splitlines()[0])
