from pathlib import Path
import json
import numpy as np
import pytorch_lightning as pl
import torch
from graph2mat import BasisConfiguration,BasisTableWithEdges,PointBasis
from graph2mat.bindings.torch.load import load_from_lit_ckpt
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
root=Path(__file__).parent
pl.seed_everything(19,workers=True)
table=BasisTableWithEdges([PointBasis('A',R=1.,basis=[1])])
cfg=BasisConfiguration(point_types=['A','A'],positions=np.array([[0.,0.,0.],[.7,0.,0.]]),basis=table,cell=np.eye(3)*10,pbc=(False,False,False),matrix=np.array([[4.,3.],[3.,5.]]))
dm=MatrixDataModule(basis_table=table,symmetric_matrix=True,sub_point_matrix=False,train_runs=[cfg],val_runs=[cfg],loader_threads=0)
m=LitMACEMatrixModel(basis_table=table,symmetric_matrix=True,hidden_irreps='2x0e+2x1o',edge_hidden_irreps='2x0e+2x1o',num_bessel=2,max_ell=1,optim_lr=0.,training_stages=[{'name':'mae_stage','max_epochs':1,'loss':'elementwise_mae'}])
def trainer():return pl.Trainer(accelerator='cpu',devices=1,max_steps=1,max_epochs=1,logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,default_root_dir=root)
t=trainer();t.fit(m,datamodule=dm)
ck=root/'stage.ckpt';t.save_checkpoint(ck)
meta=torch.load(ck,weights_only=False)
restored,p=load_from_lit_ckpt(ck)
print('SAVED_ACTIVE_STAGE',meta['active_training_stage'],'SAVED_HPARAM_DEFAULT_LOSS',meta['hyper_parameters']['loss'])
print('LIVE_LOSS',type(m.loss_fn).__name__,'RESTORED_LOSS',type(restored.loss_fn).__name__)
m.eval();restored.eval();b=next(iter(dm.val_dataloader()))
with torch.no_grad():
 old,new=m(b),restored(b)
 print('PREDICTIONS_EQUAL',all(torch.equal(old[k],new[k]) for k in ('node_labels','edge_labels')))
results={}
for name,model in [('live',m),('restored',restored)]:
 results[name]=trainer().validate(model,datamodule=dm,verbose=False)[0]['val_loss']
print('VALIDATION_RESULTS',json.dumps(results))
(root/'checkpoint_stage_result.json').write_text(json.dumps(results,indent=2))
dm.teardown(None)
