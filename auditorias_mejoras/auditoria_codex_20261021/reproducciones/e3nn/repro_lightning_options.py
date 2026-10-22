import warnings
warnings.filterwarnings('ignore')
import torch
from graph2mat.tools.lightning.tests.test_long_range import _model_and_batch
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
from graph2mat.core.data.metrics import gauge_invariant_mse
from graph2mat.models.long_range import label_element_atoms

torch.set_default_dtype(torch.float32)
_,batch,_,table=_model_and_batch(False,n_components=3)
# Positive definite scalar S=I, zero H_up/H_down on represented support.
batch.point_labels.zero_(); batch.edge_labels.zero_()
start=0
for point_type in batch.point_types.tolist():
    n=int(table.basis_size[point_type])
    batch.point_labels[start:start+n*n,-1]=torch.eye(n).reshape(-1)
    start+=n*n
opts=dict(basis_table=table,symmetric_matrix=True,hidden_irreps='2x0e+1x1o',edge_hidden_irreps='2x0e+1x1o',num_bessel=3,max_ell=1,num_interactions=1,correlation=1,lean_mace=True)
lit=LitMACEMatrixModel(**opts,n_matrix_components=3,n_output_components=2,long_range=True,long_range_smearing=1.,loss=gauge_invariant_mse).eval()
model=lit.model
torch.manual_seed(3); torch.nn.init.normal_(model.charge_readout.weight)
with torch.no_grad():
    actual=lit(batch)
    charge_head=model.charge_readout; model.charge_readout=None
    base=lit(batch); model.charge_readout=charge_head
    difference=actual['node_labels']-base['node_labels']
    owners,_,_=label_element_atoms(batch.point_types,batch.edge_types,batch.edge_index,len(batch.edge_labels),table)
    expected=actual['node_potential'][owners]*batch.point_labels[:,-1]
    print('LitMACEMatrixModel accepted long_range=True, n_matrix_components3, n_output_components2, gauge loss')
    print('S diagonal ones, offdiagonal zero; expected same electrostatic correction on both H spins')
    print('up_correction_max =',difference[:,0].abs().max().item(),'down_correction_max =',difference[:,1].abs().max().item())
    print('up_error_vs_VS =',(difference[:,0]-expected).abs().max().item(),'down_error_vs_VS =',(difference[:,1]-expected).abs().max().item())
    assert expected.abs().max()>1e-4 and difference[:,1].abs().max()==0

lit=LitMACEMatrixModel(**opts,gradient_checkpointing=True,training_stages=[{'max_epochs':1,'freeze':['mace'],'train':['readout']}])
lit._apply_training_stage_for_epoch(0)
batch.point_labels=batch.point_labels[:,0]; batch.edge_labels=batch.edge_labels[:,0]
print('Lightning training stage accepted; all MACE frozen =',all(not p.requires_grad for p in lit.model.mace.parameters()))
for checkpoint in (False,True):
    lit.model.gradient_checkpointing=checkpoint
    lit.zero_grad(set_to_none=True)
    out=lit(batch)
    loss,_=lit.loss_fn(nodes_pred=out['node_labels'],nodes_ref=batch.point_labels,edges_pred=out['edge_labels'],edges_ref=batch.edge_labels)
    print('checkpoint =',checkpoint,'loss =',loss.detach().item(),'loss_requires_grad =',loss.requires_grad)
    try:
        loss.backward()
        print('readout_nonzero_gradient_tensors =',sum(p.grad is not None and p.grad.abs().max().item()>0 for p in lit.model.matrix_readouts.parameters()))
    except RuntimeError as exc: print('backward error =',str(exc))
