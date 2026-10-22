"""CPU lattice and autograd controls for the corrected reciprocal enumeration."""
import math
import time

import torch

from graph2mat.models.long_range import _kvectors, equilibrate_charges, ewald_potentials


torch.set_default_dtype(torch.float64)
cell = torch.eye(3) * 10
transform = torch.tensor([[10.0, 1.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 0.5]])
batch = torch.zeros(2, dtype=torch.long)
smearing = 2.0
kmax = 2 * math.sqrt(-math.log(1e-8)) / smearing
results = []
for name, lattice in (("cubic", cell), ("same_lattice_skew", transform @ cell)):
    start = time.perf_counter()
    vectors = _kvectors(lattice, kmax)
    elapsed = time.perf_counter() - start
    bounds = torch.ceil(kmax * lattice.norm(dim=1) / (2 * math.pi)).long()
    candidates = int((2 * bounds + 1).prod())
    charges = torch.tensor([1.0, -1.0], requires_grad=True)
    potential = ewald_potentials(charges, positions, lattice, batch, smearing)
    gradient = torch.autograd.grad(potential.square().sum(), charges)[0]
    chi = torch.tensor([-0.4, 0.7], requires_grad=True)
    hardness = torch.tensor([9.0, 11.0], requires_grad=True)
    qeq = equilibrate_charges(chi, hardness, positions, lattice, batch, torch.tensor([0.0]), smearing)
    qeq_grad = torch.autograd.grad(qeq.square().sum(), (chi, hardness))
    values = (potential, gradient, qeq, *qeq_grad)
    assert all(torch.isfinite(value).all() for value in values)
    assert len(vectors) == 1356
    assert abs(qeq.sum().item()) < 1e-14
    results.append(values)
    print(name, "returned", len(vectors), "candidates", candidates, "enumeration_seconds", elapsed,
          "potential_eV", potential.detach().tolist(), "charge_gradient", gradient.tolist(),
          "qeq_charges", qeq.detach().tolist())
for original, transformed in zip(*results):
    torch.testing.assert_close(original, transformed, rtol=1e-11, atol=1e-11)
print("PASS: potential, charge gradients, Qeq charges and chi/hardness gradients are lattice-invariant")
