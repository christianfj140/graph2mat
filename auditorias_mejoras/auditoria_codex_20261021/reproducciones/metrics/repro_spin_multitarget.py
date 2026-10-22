import torch
import numpy as np
from scipy.linalg import eigh
from graph2mat.bindings.torch.tests.test_multitarget_loss import sample, objective, evaluate
from graph2mat.core.data.tests.physical_fixtures import analytic_matrices

torch.set_default_dtype(torch.float64)
batch,p=sample(spin=2)
n,e=batch.point_labels.clone(),batch.edge_labels.clone()
scol=p.matrix_target_layout.slices['overlap'].start
n[:,0]+=n[:,scol]; n[:,1]-=n[:,scol]
e[:,0]+=e[:,scol]; e[:,1]-=e[:,scol]
n.requires_grad_(); e.requires_grad_()
fn=objective(p,'gauge_invariant_mse',weights={'hamiltonian':1.,'overlap':0.,'density_matrix':0.,'energy_density_matrix':0.})
value,stats=evaluate(fn,batch,p,n,e)
grad=torch.autograd.grad(value,(n,e))
m=analytic_matrices(spin=2)
S=m['overlap'].Sk(format='array'); H=m['hamiltonian'].Hk(spin=0,format='array')
ref=eigh(H,S,eigvals_only=True)
up=eigh(H+S,S,eigvals_only=True); down=eigh(H-S,S,eigvals_only=True)
print('reference_eigenvalues_both_spins_eV =',ref.tolist())
print('pred_up_eV =',up.tolist(),'pred_down_eV =',down.tolist())
print('pred_spin_splitting_eV =',(up-down).tolist())
print('multitarget_H_gauge_mse =',value.detach().item(),'max_abs_gradient =',max(g.abs().max().item() for g in grad))
assert np.allclose(up-down,2.) and value.detach().item()<1e-28
