import pytest
import numpy as np

from graph2mat import BasisConfiguration, OrbitalConfiguration, PointBasis


@pytest.mark.parametrize("config_cls", [BasisConfiguration, OrbitalConfiguration])
def test_init_configuration(config_cls):
    # The basis
    point_1 = PointBasis("A", R=2, basis=[1], basis_convention="spherical")
    point_2 = PointBasis("B", R=5, basis=[2, 1], basis_convention="spherical")

    positions = np.array([[0, 0, 0], [6.0, 0, 0], [9, 0, 0]])

    # Initialize configuration
    config_cls(
        point_types=["A", "B", "A"],
        positions=positions,
        basis=[point_1, point_2],
        cell=np.eye(3) * 100,
        pbc=(False, False, False),
    )


def test_external_geometry_must_have_the_label_atoms(tmp_path):
    """Roadmap 3.11: a geometry file read in place of the run's must describe the same atoms
    as the matrix (same count and species order), or labels land on the wrong atoms."""
    sisl = pytest.importorskip("sisl")

    atom = sisl.Atom(6, R=1.5)
    geometry = sisl.Geometry([[0, 0, 0], [1.4, 0, 0]], atom, lattice=[10, 10, 10])
    H = sisl.Hamiltonian(geometry)
    H.construct([[0.1, 1.5], [0.0, -1.0]])
    geometry.write(tmp_path / "RUN.fdf")
    with open(tmp_path / "RUN.fdf", "a") as fdf:
        fdf.write("SystemLabel g\n")
    H.write(tmp_path / "g.TSHS")

    moved = geometry.move([0.1, 0, 0])
    moved.write(tmp_path / "moved.xyz")
    config = OrbitalConfiguration.from_run(tmp_path / "RUN.fdf", out_matrix="hamiltonian",
                                           geometry_path=tmp_path / "moved.xyz")
    assert np.allclose(config.positions, moved.xyz)

    geometry.add(sisl.Geometry([[5, 5, 5]], sisl.Atom(8, R=1.5))).write(tmp_path / "other.xyz")
    with pytest.raises(ValueError, match="geometry_path"):
        OrbitalConfiguration.from_run(tmp_path / "RUN.fdf", out_matrix="hamiltonian",
                                      geometry_path=tmp_path / "other.xyz")
