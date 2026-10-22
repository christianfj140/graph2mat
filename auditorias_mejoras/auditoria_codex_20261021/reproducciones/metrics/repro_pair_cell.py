from types import SimpleNamespace
import numpy as np
import torch
from graph2mat.core.data.metrics import find_displacement_pairs, pair_difference_mse

class Table:
    basis_size=np.array([1]); edge_block_size=np.array([1])
    @staticmethod
    def point_block_pointer(types): return np.arange(len(types)+1)

# det(M)=1: these cells represent the same physical square lattice.
orthogonal=np.diag([1.,1.,10.])
unimodular=np.array([[100,1,0],[99,1,0],[0,0,1]])
skew=unimodular@orthogonal
for name,cell in [('orthogonal',orthogonal),('same_lattice_skew',skew)]:
    batch=SimpleNamespace(positions=torch.tensor([[0.,-.02,0.],[0.,.02,0.]],dtype=torch.float64),cell=torch.tensor(np.stack([cell,cell])),batch=torch.tensor([0,1]),num_graphs=2,point_types=torch.zeros(2,dtype=torch.long),edge_types=torch.empty(0,dtype=torch.long),edge_index=torch.empty((2,0),dtype=torch.long))
    refs=torch.zeros(2,dtype=torch.float64); edges=torch.empty(0,dtype=torch.float64)
    kw=dict(nodes_pred=torch.tensor([1.,-1.],dtype=torch.float64),nodes_ref=refs,edges_pred=edges,edges_ref=edges,batch=batch,basis_table=Table())
    val,stats=pair_difference_mse(**kw)
    print(name,'pairs =',find_displacement_pairs(batch,.05),'loss =',val.item(),'stats =',stats)
    if name=='same_lattice_skew':
        try: pair_difference_mse(**kw,expected_pairs=1)
        except ValueError as exc: print('expected_pairs=1 error =',str(exc))
print('same physical nearest-image separation = 0.04 Ang; expected both [(0,1)], loss5')
