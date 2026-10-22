"""Same minimum-image counterexample, with hopping stored at physical +/-3 images."""
import json
import tempfile
from pathlib import Path

import numpy as np
import sisl

from graph2mat.tools.analysis.breakdown import min_image_distances, structure_breakdown
from graph2mat.tools.conftest import write_fdf

atom = sisl.Atom(6, orbitals=[sisl.AtomicOrbital('2s', R=2., q0=1)])
geometry = sisl.Geometry([[0., 0., 0.], [31., 0., 0.]], atoms=[atom, atom],
                         lattice=sisl.Lattice([10., 10., 10.], nsc=[7, 1, 1]))
geometry.lattice.pbc = [True, False, False]
ref = sisl.Hamiltonian(geometry, orthogonal=False)
ref[0, 0] = (1., 1.)
ref[1, 1] = (2., 1.)
ref[0, 1, (-3, 0, 0)] = (.2, 0.)
ref[1, 0, (3, 0, 0)] = (.2, 0.)
pred = ref.copy()
pred[0, 1, (-3, 0, 0)] = (.3, 0.)
pred[1, 0, (3, 0, 0)] = (.3, 0.)

distance = float(np.linalg.norm(geometry.axyz(1, isc=[-3, 0, 0]) - geometry.xyz[0]))
assert distance == 1.
for k in ([0., 0., 0.], [.23, 0., 0.]):
    hk = ref.Hk(k=k, format='array')
    np.testing.assert_allclose(hk, hk.conj().T)
with tempfile.TemporaryDirectory(prefix='t8-hopping-', dir=Path(__file__).parent) as tmp:
    directory = Path(tmp)
    ref.write(str(directory / 'ref.TSHS'))
    pred.write(str(directory / 'pred.TSHS'))
    fdf = write_fdf(directory / 'fdf', geometry)
    reread = sisl.get_sile(str(directory / 'ref.TSHS')).read_hamiltonian()
    assert reread.geometry.xyz[1, 0] == 31.
    row = structure_breakdown(directory / 'pred.TSHS', directory / 'ref.TSHS', fdf, [atom])
    assert abs(row['total_sum'] - .2) < 1e-12
    assert row['beyond_range_sum'] == row['total_sum']
    assert row['shell_6.0-99.0_sum'] == row['total_sum']
    assert row['shell_0.1-1.6_sum'] == 0.
    out = dict(physical_hopping_image=[-3, 0, 0], physical_hopping_distance_Ang=distance,
               actual_distance_Ang=float(min_image_distances(reread.geometry)[0, 1]),
               orbital_R_Ang=2., cutoff_sum_Ang=4.,
               classification={k: row[k] for k in ['total_sum', 'beyond_range_sum',
                                                  'shell_6.0-99.0_sum', 'shell_0.1-1.6_sum']})
print(json.dumps(out, indent=2))
