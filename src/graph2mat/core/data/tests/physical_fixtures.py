"""Independent occupied-state and real-space fields for physical objectives."""

import numpy as np
import sisl
from scipy.sparse import csr_array


def analytic_matrices(spin=1, fractional=False):
    """Two 1s atoms: DS populations 1/atom and HD=ES, without a loss oracle."""
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.8, q0=1.)])
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=[atom, atom], lattice=[20., 20., 20.])
    geometry.set_nsc([1, 1, 1])
    overlap = csr_array([[1., .2], [.2, 1.]])
    hamiltonian = csr_array([[.5, -.3], [-.3, .5]])
    scale = .6 if fractional else 1.
    density = csr_array(np.full((2, 2), scale * 5 / 6))
    energy_density = csr_array(np.full((2, 2), scale * 5 / 36))
    parts = (.7, .3) if fractional else (.5, .5)
    polarized = spin == 2
    spin_name = "polarized" if polarized else "unpolarized"
    return dict(
        overlap=sisl.Overlap.fromsp(geometry, overlap),
        density=sisl.DensityMatrix.fromsp(geometry, [density*p for p in parts] if polarized else density,
                                        S=overlap, spin=spin_name),
        hamiltonian=sisl.Hamiltonian.fromsp(geometry, [hamiltonian, hamiltonian] if polarized else hamiltonian,
                                          S=overlap, spin=spin_name),
        energy_density=sisl.EnergyDensityMatrix.fromsp(geometry, [energy_density*p for p in parts] if polarized else energy_density,
                                                      S=overlap, spin=spin_name),
        kpoints=np.zeros((1, 3)), kweights=np.ones(1),
    )


def periodic_matrices():
    """One atom with +/- images: Tr(DS)=.84, unlike Gamma product 1.4."""
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2., q0=1.)])
    geometry = sisl.Geometry([[0., 0., 0.]], atoms=[atom], lattice=[3., 20., 20.])
    geometry.set_nsc([3, 1, 1])
    overlap = csr_array([[1. if not np.any(r) else .2 for r in geometry.sc_off]])
    density = csr_array([[.8 if not np.any(r) else .1 for r in geometry.sc_off]])
    return dict(
        overlap=sisl.Overlap.fromsp(geometry, overlap),
        density=sisl.DensityMatrix.fromsp(geometry, density, S=overlap),
        hamiltonian=sisl.Hamiltonian.fromsp(geometry, .5*overlap, S=overlap),
        energy_density=sisl.EnergyDensityMatrix.fromsp(geometry, .5*density, S=overlap),
        kpoints=np.column_stack((np.arange(8)/8, np.zeros(8), np.zeros(8))),
        kweights=np.full(8, 1/8),
    )
