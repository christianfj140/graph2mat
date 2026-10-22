import json
from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from graph2mat import BasisConfiguration, BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.core.data.metrics import elementwise_mse
from graph2mat.tools.lightning.model import LitBasisMatrixModel
from graph2mat.tools.lightning.tests.test_epoch_metrics import _PredictionModel
from graph2mat.tools.lightning.callbacks import SamplewiseMetricsLogger
ROOT=Path(__file__).parent
TABLE=BasisTableWithEdges([PointBasis('A',R=1,basis=[1])])
class Module(pl.LightningDataModule):
 def __init__(self):
  super().__init__();self.basis_table=TABLE;self.out_matrix='density_matrix';self.symmetric_matrix=True
 def val_dataloader(self):
  p=MatrixDataProcessor(TABLE,sub_point_matrix=False,symmetric_matrix=True)
  samples=[]
  for i in range(4):
   cfg=BasisConfiguration(point_types=['A'],positions=np.zeros((1,3)),basis=TABLE,cell=np.eye(3)*10,pbc=(False,False,False),matrix=np.zeros((1,1)),metadata={'path':str(ROOT/f'sample_{i}'/'RUN.fdf')})
   d=TorchBasisMatrixData.new(cfg,p);d.node_prediction=torch.tensor([float(i+1)]);d.edge_prediction=torch.empty(0);samples.append(d)
  return DataLoader(samples,batch_size=1,shuffle=False)
class Recorder(pl.Callback):
 def on_validation_epoch_end(self,t,m):
  (ROOT/f'ddp_callback_rank{t.global_rank}.json').write_text(json.dumps({'rank':t.global_rank,'global_loss':float(t.callback_metrics['val_loss'])}))
if __name__=='__main__':
 torch.set_num_threads(2)
 model=LitBasisMatrixModel(model_cls=_PredictionModel,basis_table=TABLE);model.init_model()
 callback=SamplewiseMetricsLogger(metrics=[elementwise_mse],splits='val',output_file=ROOT/'ddp_samples.csv')
 trainer=pl.Trainer(accelerator='cpu',devices=2,strategy='ddp',logger=False,enable_checkpointing=False,enable_progress_bar=False,enable_model_summary=False,default_root_dir=ROOT,callbacks=[callback,Recorder()])
 trainer.validate(model,datamodule=Module(),verbose=False)
 if trainer.is_global_zero:print('DDP_CALLBACK_VALIDATION_DONE',flush=True)
