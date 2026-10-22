from pathlib import Path
from types import SimpleNamespace
import traceback
import numpy as np
import sisl
from torch_geometric.data import Batch
from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
from graph2mat.bindings.torch import TorchBasisMatrixData
from graph2mat.tools.lightning.callbacks import MatrixWriter
root=Path(__file__).parent
atom=sisl.Atom('H',orbitals=[sisl.AtomicOrbital('1s',R=.8)])
g=sisl.Geometry([[0.,0.,0.],[1.,0.,0.]],atoms=atom,lattice=[10.,10.,10.])
h=sisl.Hamiltonian(g,orthogonal=False)
h[0,0,:],h[1,1,:],h[0,1,:],h[1,0,:]=[1.,1.],[2.,1.],[.25,.1],[.25,.1]
for policy,components in [('h_only',1),('h_and_overlap',2)]:
 p=MatrixDataProcessor(AtomicTableWithEdges([atom]),symmetric_matrix=True,sub_point_matrix=False,out_matrix='hamiltonian',matrix_component_policy=policy,n_matrix_components=components)
 d=TorchBasisMatrixData.new(h,p);b=Batch.from_data_list([d])
 pred={'node_labels':b.point_labels,'edge_labels':b.edge_labels}
 trainer=SimpleNamespace(datamodule=SimpleNamespace(data_processor=p,out_matrix='hamiltonian'))
 for extension in ['.nc','.HSX']:
  path=root/f'writer_{policy}{extension}'
  print('TRY',policy,extension,flush=True)
  try:
   MatrixWriter(str(path))._on_batch_end('predict',trainer,None,pred,b,0,0)
   written=sisl.get_sile(path).read_hamiltonian()
   print('WROTE',path,'H',written.tocsr(0).toarray(),flush=True)
  except Exception:traceback.print_exc()
