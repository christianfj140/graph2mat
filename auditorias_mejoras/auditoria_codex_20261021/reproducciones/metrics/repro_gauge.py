from types import SimpleNamespace
import numpy as np
import torch
from graph2mat.core.data.metrics import gauge_invariant_mse, _spectral_window_loss
from graph2mat.core.modules.graph2mat import LAYOUT_ATTR

# One orbital and two collinear channels. Common scalar gauge cannot change splitting.
batch = SimpleNamespace()
setattr(batch, LAYOUT_ATTR, dict(node_struct=torch.tensor([0]), edge_struct=torch.empty(0,dtype=torch.long),n_structures=1))
rn = torch.tensor([[0.,0.,1.]],dtype=torch.float64)
re = torch.empty((0,3),dtype=torch.float64)
pn = torch.tensor([[1.,-1.]],dtype=torch.float64,requires_grad=True)
pe = torch.empty((0,2),dtype=torch.float64,requires_grad=True)
loss, stats = gauge_invariant_mse(nodes_pred=pn,nodes_ref=rn,edges_pred=pe,edges_ref=re,batch=batch)
grad = torch.autograd.grad(loss,pn)[0]
print('spin_ref_levels =',rn[0,:2].tolist())
print('spin_pred_levels =',pn[0].tolist())
print('ref_splitting_eV = 0; pred_splitting_eV = 2')
print('gauge_mse =',float(loss),'gradient =',grad.tolist(),'common_gauge_oracle_mse = 1')
assert float(loss)==0 and torch.equal(grad,torch.zeros_like(grad))

# S below null_space_columns floor: its columns are not S-orthonormal.
S=np.diag([.005,.0002])
w,v=np.linalg.eigh(S)
C=v/np.sqrt(np.maximum(w,.001))
rows,cols=np.divmod(np.arange(4),2)
lay=dict(node_row=torch.tensor(rows),node_col=torch.tensor(cols),edge_row=torch.empty(0,dtype=torch.long),edge_col=torch.empty(0,dtype=torch.long),orb_struct=torch.zeros(2,dtype=torch.long),orb_local=torch.arange(2),no_max=2,symmetric=False)
batch=SimpleNamespace(spectral_C=torch.tensor(C))
setattr(batch,LAYOUT_ATTR,lay)
d=torch.tensor(S.ravel(),requires_grad=True)
value,count=_spectral_window_loss(d,torch.empty(0,dtype=torch.float64),torch.zeros(4,dtype=torch.long),torch.empty(0,dtype=torch.long),1,batch,None,None)
print('C.T S C =',np.round(C.T@S@C,8).tolist())
print('nullspace_pure_cS_spectral_mse =',float(value),'count =',float(count),'expected = 0')
print('nullspace_gradient =',torch.autograd.grad(value,d)[0].tolist())
assert abs(float(value)-.08)<1e-13
