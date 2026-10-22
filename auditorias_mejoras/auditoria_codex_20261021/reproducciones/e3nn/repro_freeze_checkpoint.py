import warnings
warnings.filterwarnings('ignore')
import torch
from graph2mat.tools.lightning.tests.test_long_range import _model_and_batch

torch.set_default_dtype(torch.float32)
model,batch,_,_=_model_and_batch(False)
model.lean_mace=True
for p in model.mace.parameters(): p.requires_grad_(False)
for checkpoint in (False,True):
    model.gradient_checkpointing=checkpoint
    model.zero_grad(set_to_none=True)
    out=model(batch)
    loss=out['node_labels'].square().mean()+out['edge_labels'].square().mean()
    print('frozen MACE; gradient_checkpointing =',checkpoint,'loss requires_grad =',loss.requires_grad,'loss =',loss.detach().item())
    try:
        loss.backward()
        print('readout_nonzero_grad_tensors =',sum(p.grad is not None and p.grad.abs().max().item()>0 for p in model.matrix_readouts.parameters()))
    except RuntimeError as exc: print('backward error =',str(exc))
