from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from graph2mat import BasisConfiguration,BasisTableWithEdges,PointBasis
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.bindings.torch.load import load_from_lit_ckpt
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
ROOT=Path(__file__).parent
pl.seed_everything(3,workers=True)
torch.set_default_dtype(torch.float64)
table=BasisTableWithEdges([PointBasis('A',R=1,basis=[1])])
cfg=BasisConfiguration(point_types=['A','A'],positions=np.array([[0.,0.,0.],[.7,0.,0.]]),basis=table,cell=np.eye(3)*10,pbc=(False,False,False),matrix=np.array([[2.,3.],[3.,4.]]))
dm=MatrixDataModule(basis_table=table,symmetric_matrix=True,sub_point_matrix=False,train_runs=[cfg],val_runs=[],loader_threads=0)
m=LitMACEMatrixModel(basis_table=table,symmetric_matrix=True,hidden_irreps='2x0e+2x1o',edge_hidden_irreps='2x0e+2x1o',num_bessel=2,max_ell=1,optim_lr=0.)
t=pl.Trainer(accelerator='cpu',devices=1,max_steps=1,max_epochs=1,logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,default_root_dir=ROOT)
t.fit(m,datamodule=dm)
ck=ROOT/'float64.ckpt';t.save_checkpoint(ck)
state=torch.load(ck,weights_only=False)['state_dict']
print('SAVED_DTYPES',sorted({str(x.dtype) for x in state.values() if x.is_floating_point()}))
torch.set_default_dtype(torch.float32)
restored,processor=load_from_lit_ckpt(ck)
print('LOADED_DTYPES',sorted({str(x.dtype) for x in restored.state_dict().values() if x.is_floating_point()}))
key=next(k for k,v in state.items() if v.is_floating_point() and torch.any(v != restored.state_dict()[k].to(v)))
print('FIRST_LOSSY_KEY',key,'MAX_ABS_DIFF',float((state[key]-restored.state_dict()[key].to(state[key])).abs().max()))
# Requesting double afterward cannot recover bits already discarded by the loader.
restored.double(); print('AFTER_DOUBLE_MAX_ABS_DIFF',float((state[key]-restored.state_dict()[key]).abs().max()))
dm.teardown(None)
