"""Independent complete reciprocal enumeration; original source remains unchanged."""
import math

import torch

from graph2mat.models.long_range import COULOMB, _kvectors, ewald_matrix, ewald_potentials


torch.set_default_dtype(torch.float64)
cell = torch.eye(3) * 10
positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 1.0, 0.5]])
charges = torch.tensor([1.0, -1.0])
batch = torch.zeros(2, dtype=torch.long)
smearing = 2.0
kmax = 2 * math.sqrt(-math.log(1e-8)) / smearing


def complete_vectors(cell):
    # k.a_i = 2*pi*n_i, hence |n_i| <= kmax*|a_i|/(2*pi).
    bounds = torch.ceil(kmax * cell.norm(dim=1) / (2 * math.pi)).to(torch.int64)
    grids = [torch.arange(-n.item(), n.item() + 1) for n in bounds]
    integer = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).reshape(-1, 3)
    vectors = integer.to(cell.dtype) @ (2 * math.pi * torch.linalg.inv(cell).T)
    squared = vectors.square().sum(-1)
    return vectors[(squared > 0) & (squared <= kmax**2)]


def complete_matrix(cell):
    vectors = complete_vectors(cell)
    squared = vectors.square().sum(-1)
    weights = 4 * math.pi / cell.det().abs() * torch.exp(-squared * smearing**2 / 4) / squared
    phase = positions @ vectors.T
    cosine, sine = phase.cos(), phase.sin()
    return COULOMB * ((cosine * weights) @ cosine.T + (sine * weights) @ sine.T)


reference = complete_matrix(cell)
for name, transform in (
    ("identity", torch.eye(3)),
    ("moderate_shear", torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
    ("strong_shear", torch.tensor([[10.0, 1.0, 0.0], [9.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
):
    assert torch.isclose(transform.det(), torch.tensor(1.0))
    transformed = transform @ cell
    expected = complete_matrix(transformed)
    actual = ewald_matrix(positions, transformed, smearing)
    potential = ewald_potentials(charges, positions, transformed, batch, smearing)
    expected_potential = expected @ charges - COULOMB * charges * 2 / (smearing * math.sqrt(math.pi))
    torch.testing.assert_close(expected, reference, rtol=1e-12, atol=1e-12)
    assert len(complete_vectors(transformed)) == 1356
    print(name, "current_vectors", len(_kvectors(transformed, kmax)), "complete_vectors", len(complete_vectors(transformed)),
          "matrix_error_eV", (actual - expected).abs().max().item(), "potential_eV", potential.tolist(),
          "potential_error_eV", (potential - expected_potential).abs().max().item())

print("oracle_passed: same lattice, same full reciprocal sphere, same analytic matrix")
