"""Experimental module for defining node features.

You probably don't care about this module unless you have come
here to find it because you want to play with node features.

Usually node features/embeddings should be defined by the atomic
environment descriptor of choice, but I was playing with these to
incorporate the total dipole of water as a global descriptor into
MACE.
"""
import numpy as np
from typing import Dict, Type


class NodeFeature:
    def __new__(cls, config, data_processor):
        return cls.get_feature(config, data_processor)

    registry: Dict[str, Type["NodeFeature"]] = {}

    def __init_subclass__(cls) -> None:
        NodeFeature.registry[cls.__name__] = cls

    @staticmethod
    def get_feature(config: dict, data_processor) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_e3nn_irreps(data_processor):
        raise NotImplementedError


class OneHotZ(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps([(len(basis_table), (0, 1))])

    @staticmethod
    def get_feature(config, data_processor):
        indices = data_processor.get_point_types(config)
        return data_processor.one_hot_encode(indices)


class WaterDipole(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps("1x1o")

    @staticmethod
    def get_feature(config, data_processor):
        # The whole vector (it used only z, which is not a 1o irrep: MATH_EQUIV-22).
        dipole = data_processor.cartesian_to_basis(_water_dipole(config)) / 30
        return np.tile(dipole, (len(config.positions), 1))


class WaterDipoleInv(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps("1x0e")

    @staticmethod
    def get_feature(config, data_processor):
        # The norm (its z component was not invariant under rotations: MATH_EQUIV-22).
        dipole = np.linalg.norm(_water_dipole(config)) / 30
        return np.full((len(config.positions), 1), dipole)


def _water_dipole(config) -> np.ndarray:
    """Point-charge dipole with O = -2 and H = +1."""
    charges = np.select([np.asarray(config.point_types) == 8, np.asarray(config.point_types) == 1], [-2.0, 1.0], 0.0)
    return charges @ np.asarray(config.positions, dtype=float)


class Nothing(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps("1x0e")

    @staticmethod
    def get_feature(config, data_processor):
        n_atoms = len(config.positions)

        z_dipole = np.array([0.0])

        z_dipole = np.tile(z_dipole, n_atoms).reshape(n_atoms, 1)

        return z_dipole


class NothingVector(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps("1x1o")

    @staticmethod
    def get_feature(config, data_processor):
        n_atoms = len(config.positions)

        z_dipole = np.array([0.0, 0.0, 0.0])

        z_dipole = np.tile(z_dipole, n_atoms).reshape(n_atoms, 3)

        return z_dipole


class One(NodeFeature):
    @staticmethod
    def get_e3nn_irreps(basis_table):
        from e3nn import o3

        return o3.Irreps("1x0e")

    @staticmethod
    def get_feature(config, data_processor):
        n_atoms = len(config.positions)

        z_dipole = np.array([1.0])

        z_dipole = np.tile(z_dipole, n_atoms).reshape(n_atoms, 1)

        return z_dipole
