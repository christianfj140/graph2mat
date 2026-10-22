from pathlib import Path
from types import SimpleNamespace
import numpy as np
import sisl
import torch
from scipy.sparse import csr_array
from graph2mat.core.data.metrics import gauge_invariant_mse
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR
from graph2mat.tools.spectral_targets import null_space_columns

root=Path(__file__).parent
U=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],float)/2
S=U@np.diag([1.5,2.4948,.005,.0002])@U.T
geom=sisl.Geometry([[0,0,0]],atoms=sisl.Atom('H',orbitals=[sisl.AtomicOrbital('1s',R=1) for _ in range(4)]),lattice=[10,10,10])
h=sisl.Hamiltonian.fromsp(geom,csr_array(.5*S),S=csr_array(S))
path=root/'null_reference.nc'
h.write(str(path))
C=null_space_columns(path)
rows,cols=np.divmod(np.arange(16),4)
lay=dict(node_row=torch.tensor(rows),node_col=torch.tensor(cols),edge_row=torch.empty(0,dtype=torch.long),edge_col=torch.empty(0,dtype=torch.long),orb_struct=torch.zeros(4,dtype=torch.long),orb_local=torch.arange(4),no_max=4,symmetric=False,node_struct=torch.zeros(16,dtype=torch.long),edge_struct=torch.empty(0,dtype=torch.long),n_structures=1)
batch=SimpleNamespace(spectral_C=torch.tensor(C))
setattr(batch,LAYOUT_ATTR,lay)
rn=torch.tensor(np.column_stack([.5*S.ravel(),S.ravel()]))
re=torch.empty((0,2),dtype=torch.float64)
pn=(rn[:,0]+rn[:,1]).clone().requires_grad_()
pe=torch.empty(0,dtype=torch.float64,requires_grad=True)
base,bstats=gauge_invariant_mse(nodes_pred=pn,nodes_ref=rn,edges_pred=pe,edges_ref=re,batch=batch,spectral_weight=0.)
actual,stats=gauge_invariant_mse(nodes_pred=pn,nodes_ref=rn,edges_pred=pe,edges_ref=re,batch=batch,spectral_weight=1.)
print('diag_S =',np.diag(S).tolist())
print('C.T S C =',np.round(C.T@S@C,9).tolist())
print('pure_gauge_base_mse =',base.detach().item())
print('pure_gauge_total_mse =',actual.detach().item(),'spectral_rmse =',stats['spectral_rmse'].item(),'expected = 0')
print('max_abs_gradient =',torch.autograd.grad(actual,pn)[0].abs().max().item())
assert base.detach().item()<1e-28 and abs(actual.detach().item()-.08)<1e-10
