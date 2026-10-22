import torch
from graph2mat.models.long_range import _kvectors, ewald_potentials, ewald_matrix, equilibrate_charges
from graph2mat.tools.lightning.tests.test_long_range import _model_and_batch
from graph2mat.models.mace import MatrixMACE
from e3nn import o3

# Same lattice under unimodular basis change; fields must be identical.
torch.set_default_dtype(torch.float64)
cell=torch.diag(torch.tensor([10.,10.,10.]))
M=torch.tensor([[10.,1.,0.],[9.,1.,0.],[0.,0.,1.]])
skew=M@cell
q=torch.tensor([1.,-1.]); r=torch.tensor([[0.,0.,0.],[2.,1.,.5]])
ids=torch.zeros(2,dtype=torch.long)
for name,c in [('orthogonal',cell),('same_lattice_skew',skew)]:
    v=ewald_potentials(q,r,c,ids,smearing=2.)
    a=ewald_matrix(r,c,smearing=2.)
    print(name,'nk =',len(_kvectors(c,4.291932052578694)),'potential_eV =',v.tolist(),'matrix_eV =',a.tolist())

# Public MatrixMACE: two H channels, overlap supplied in labels' last component.
torch.set_default_dtype(torch.float32)
plain,batch,processor,table=_model_and_batch(False,n_components=3)
model=MatrixMACE(mace=plain.mace,unique_basis=table,symmetric=True,edge_hidden_irreps=o3.Irreps('2x0e+2x1o'),n_matrix_components=2,long_range=True,long_range_smearing=1.)
torch.nn.init.normal_(model.charge_readout.weight)
with torch.no_grad():
    out=model(batch)
    saved=model.charge_readout
    model.charge_readout=None
    base=model(batch)
    model.charge_readout=saved
    diff=out['node_labels']-base['node_labels']
    print('collinear dH_up max =',diff[:,0].abs().max().item(),'dH_down max =',diff[:,1].abs().max().item())
    print('charge max =',out['node_charges'].abs().max().item(),'potential max =',out['node_potential'].abs().max().item())

for mode in (True,'qeq'):
    model,batch,_,_=_model_and_batch(mode)
    try:
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out=model(batch)
        print('autocast',mode,'success',out['node_labels'].dtype)
    except Exception as exc:
        print('autocast',mode,type(exc).__name__,str(exc).splitlines()[0])

plain,batch,_,_=_model_and_batch(False)
plain.double(); batch.apply(lambda v: v.double() if torch.is_tensor(v) and v.is_floating_point() else v)
plain.readout_fp32=True
try:
    plain(batch)
    print('readout_fp32 with float64 success')
except Exception as exc:
    print('readout_fp32 with float64',type(exc).__name__,str(exc).splitlines()[0])
