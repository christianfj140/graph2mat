import json
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from graph2mat import BasisConfiguration,BasisTableWithEdges,PointBasis
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
ROOT=Path(__file__).parent
class Recorder(pl.Callback):
 def __init__(self):self.rows=[]
 def on_validation_epoch_end(self,t,m):
  batch=next(iter(t.datamodule.val_dataloader()))
  with torch.no_grad():actual=float((m(batch)['edge_labels']-batch.edge_labels).square().mean().sqrt())
  self.rows.append({'epoch':t.current_epoch,'loss_fn':type(m.loss_fn).__name__,'actual_edge_rmse':actual,'edge_count':float(m._epoch_rmse['val_edge_rmse'].count),'metrics':{k:float(v) for k,v in t.callback_metrics.items() if k in ['hp_metric','val_matrix_rmse','val_edge_rmse','val_loss']}})
pl.seed_everything(20,workers=True)
table=BasisTableWithEdges([PointBasis('A',R=1,basis=[1])])
cfg=BasisConfiguration(point_types=['A','A'],positions=np.array([[0.,0.,0.],[.7,0.,0.]]),basis=table,cell=np.eye(3)*10,pbc=(False,False,False),matrix=np.array([[4.,3.],[3.,5.]]))
dm=MatrixDataModule(basis_table=table,symmetric_matrix=True,sub_point_matrix=False,train_runs=[cfg],val_runs=[cfg],loader_threads=0)
m=LitMACEMatrixModel(basis_table=table,symmetric_matrix=True,hidden_irreps='2x0e+2x1o',edge_hidden_irreps='2x0e+2x1o',num_bessel=2,max_ell=1,optim_lr=.002,selection_metric='val_edge_rmse',training_stages=[{'name':'mse','max_epochs':1},{'name':'mae','max_epochs':1,'loss':'block_type_mae'}])
r=Recorder()
checkpoint=pl.callbacks.ModelCheckpoint(dirpath=ROOT/'selection_ckpt',monitor='val_edge_rmse',mode='min',save_top_k=1,filename='{epoch}')
t=pl.Trainer(accelerator='cpu',devices=1,max_epochs=2,logger=False,enable_checkpointing=True,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0,default_root_dir=ROOT,callbacks=[r,checkpoint])
t.fit(m,datamodule=dm)
print('SELECTION_RESULTS',json.dumps(r.rows))
print('BEST_MODEL',checkpoint.best_model_path,'BEST_SCORE',float(checkpoint.best_model_score))
(ROOT/'selected_checkpoint.json').write_text(json.dumps({'best_model_path':checkpoint.best_model_path,'best_model_score':float(checkpoint.best_model_score),'checkpoint_epoch':torch.load(checkpoint.best_model_path,weights_only=False)['epoch']},indent=2))
(ROOT/'stale_selection_result.json').write_text(json.dumps(r.rows,indent=2))
dm.teardown(None)
