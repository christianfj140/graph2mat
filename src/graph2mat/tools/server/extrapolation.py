"""Functionality for matrix extrapolation from a time series."""

from typing import Union, Type, Tuple, Any, Dict, Callable, Optional

from copy import copy
from pathlib import Path

from abc import ABC, abstractmethod

import numpy as np

import sisl

from collections import defaultdict, deque

from e3nn import o3

from graph2mat import (
    AtomicTableWithEdges,
    MatrixDataProcessor,
    OrbitalConfiguration,
    BasisMatrixData,
)
from graph2mat.core.data.matrices import get_matrix_cls
from graph2mat.tools.server._serialization import write_prediction


class DescriptorManager(ABC):
    @classmethod
    @abstractmethod
    def get_descriptors(
        cls, data, configs, hist_b_dict, missing_block_steps, data_processor, **kwargs
    ) -> Tuple[dict, Tuple[list, list]]:
        pass

    @classmethod
    def predict_from_descriptors(
        cls, history_descriptors, history_labels, block_kwargs
    ):
        predicted_labels = {}

        for k, block_desc in history_descriptors.items():
            predicted_labels[k] = cls.predict_block(
                block_desc,
                history_labels[k],
                **{key: v[k] for key, v in block_kwargs.items()},
            )

        return predicted_labels

    @staticmethod
    def predict_block(
        block_history_descriptors, block_history_labels, rcond=1e-8, **kwargs
    ):
        def predict_next(history_descriptors, history_labels):
            history_descriptors = np.asarray(history_descriptors)
            history_labels = np.asarray(history_labels)
            if len(history_descriptors) == 0:
                raise ValueError("Need at least one descriptor for the target.")
            if len(history_descriptors) != len(history_labels) + 1:
                raise ValueError("Need one more descriptor than historical labels.")
            if history_descriptors.shape[0] == 1:
                return np.zeros(history_labels.shape[1:], dtype=float)

            labels = history_labels.reshape(history_labels.shape[0], -1)
            if np.isfinite(labels).all():
                x_lstsq = DescriptorManager.block_coefficients(history_descriptors, rcond=rcond)
                predicted_labels = labels.T @ x_lstsq.reshape(-1)
            else:
                # Each matrix component can have different missing observations.
                predicted_labels = np.zeros(labels.shape[1], dtype=np.result_type(
                    labels.dtype, history_descriptors.dtype, np.float32,
                ))
                for column in range(labels.shape[1]):
                    valid = np.isfinite(labels[:, column])
                    if not valid.any():
                        continue
                    descriptors = np.concatenate([
                        history_descriptors[:-1][valid], history_descriptors[-1:],
                    ])
                    weights = DescriptorManager.block_coefficients(descriptors, rcond=rcond)
                    predicted_labels[column] = labels[valid, column] @ weights

            return predicted_labels.reshape(*history_labels.shape[1:])

        if "descriptors_irreps" in kwargs:
            desc_irreps = o3.Irreps(kwargs["descriptors_irreps"])

            rtp = kwargs["rtp"]
            op = kwargs["op"]

            block_history_descriptors = op(block_history_descriptors)
            irreps_out = op.irreps_out

            raveled_change_of_basis = rtp.change_of_basis.reshape(
                rtp.change_of_basis.shape[0], -1
            ).numpy()

            irreps_labels = block_history_labels @ raveled_change_of_basis.T

            predicted_labels = irreps_labels[-1].copy()

            for desc_slice, labels_slice, x, y in zip(
                irreps_out.slices(), rtp.irreps_out.slices(), irreps_out, rtp.irreps_out
            ):
                # print(x, y)

                descriptors = block_history_descriptors[:, desc_slice]
                labels = irreps_labels[:, labels_slice]

                if descriptors.shape[-1] == 0:
                    predicted_labels[labels_slice] = labels[-1]
                else:
                    predicted_labels[labels_slice] = predict_next(descriptors, labels)

            predicted_labels = predicted_labels @ raveled_change_of_basis
        else:
            predicted_labels = predict_next(
                block_history_descriptors, block_history_labels
            )

        return predicted_labels

    @classmethod
    def coefficients_from_descriptors(cls, history_descriptors, block_kwargs):
        predicted_coefficients = {}

        for k, block_desc in history_descriptors.items():
            predicted_coefficients[k] = cls.block_coefficients(
                block_desc, **{key: v[k] for key, v in block_kwargs.items()}
            )

        return predicted_coefficients

    @staticmethod
    def block_coefficients(block_history_descriptors, rcond=1e-8, **kwargs):
        block_history_descriptors = np.asarray(block_history_descriptors)
        if len(block_history_descriptors) == 0:
            raise ValueError("Need at least one descriptor for the target.")
        A = block_history_descriptors[:-1].T
        b = block_history_descriptors[-1]

        if np.isfinite(block_history_descriptors).all():
            # Keep the complete-data objective, dtype and numerical path unchanged.
            x_lstsq, r, rank, sv = np.linalg.lstsq(A, b, rcond=rcond)
            return x_lstsq

        features = np.isfinite(b)
        valid = np.isfinite(A[features]).all(axis=0)
        coefficients = np.zeros(len(block_history_descriptors) - 1,
                                dtype=np.result_type(block_history_descriptors.dtype, np.float32))
        if valid.any():
            coefficients[valid] = np.linalg.lstsq(A[features][:, valid], b[features], rcond=rcond)[0]
        return coefficients


class HistorySOAPs(DescriptorManager):
    @classmethod
    def get_descriptors(
        cls,
        data,
        configs,
        hist_b_dict,
        missing_block_steps,
        data_processor,
        r_cut=3,
        n_max=1,
        l_max=1,
        rbf="gto",
    ):
        """Computes SOAP descriptors for all atoms in a data batch"""
        try:
            from dscribe.descriptors import SOAP
        except ModuleNotFoundError as exc:
            if exc.name != "dscribe":
                raise
            raise ModuleNotFoundError(
                "SOAP descriptors require graph2mat[extrapolation].", name="dscribe"
            ) from exc

        species = sorted({at.Z for at in data_processor.basis_table.atoms})
        pbc = [
            config.pbc if config.pbc is not None else config.metadata["geometry"].lattice.pbc
            for config in configs
        ]

        soap = SOAP(
            species=species,
            periodic=bool(np.any(pbc)),
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            rbf=rbf,
        )

        na = data[0].num_nodes

        target_geom = configs[-1].metadata["geometry"]
        target_data_arrays = data[-1].numpy_arrays()
        unique_edges = target_data_arrays.edge_index[:, ::2]
        edge_shifts = target_geom.sc_off[target_data_arrays.neigh_isc[::2]]

        atom_neighs = []
        atom_shifts = []
        for at in range(na):
            has_this_atom = (unique_edges == at).any(axis=0)
            atom_edges = unique_edges[:, has_this_atom]

            shifts = edge_shifts.copy()[has_this_atom]
            shifts[atom_edges[0] != at] *= -1

            neighs = atom_edges[0].copy()
            neighs[neighs == at] = atom_edges[1][neighs == at]

            atom_neighs.append(neighs)
            atom_shifts.append(shifts)

        all_descriptors = {k: [] for k in hist_b_dict}
        for i_molec, config in enumerate(configs):
            geom = config.metadata["geometry"]
            atoms = geom.to.ase()
            atoms.set_pbc(pbc[i_molec])

            atom_soaps = soap.create(atoms)

            # Build all atom descriptors, we do this before edges so that
            # we can get the atom descriptors to add them to edge descriptors.
            atom_descriptors = []
            for i in range(na):
                atom_descriptors.append(
                    [
                        *atom_soaps[i],
                        *cls.compute_atom_descriptor(
                            geom,
                            i,
                            atom_neighs[i],
                            atom_shifts[i] @ geom.cell,
                            processor=data_processor,
                        ),
                    ]
                )
                key = (i, i, 0)
                if key in all_descriptors and i_molec not in missing_block_steps.get(key, []):
                    all_descriptors[key].append(atom_descriptors[-1])

            for k in all_descriptors:
                i, j, isc = k

                if i == j and isc == 0:
                    # This is a node block, we have already built its descriptors
                    continue
                elif False:
                    pass
                    # Handle edge not being present in geometry
                else:
                    if i_molec in missing_block_steps[i, j, isc]:
                        continue

                    shift = target_geom.sc_off[isc] @ geom.cell

                    edge_sender_descriptors = atom_descriptors[i]
                    edge_receiver_descriptors = atom_descriptors[j]

                    edge_center = (geom.xyz[i] + geom.xyz[j] + shift) / 2
                    edge_soap = soap.create(atoms, [edge_center])

                    all_descriptors[i, j, isc].append(
                        [
                            *edge_soap.ravel(),
                            *cls.compute_edge_descriptors(
                                geom, (i, j), shift, processor=data_processor
                            ).ravel(),
                            *edge_sender_descriptors,
                            *edge_receiver_descriptors,
                        ]
                    )

        all_descriptors = {
            k: np.array(v) for k, v in all_descriptors.items() if len(v) > 0
        }

        atom_desc_irreps = [
            # (atom_soaps.shape[-1] + atom_neighs[0].shape[0], (0,1)), # Invariant (scalar) quantities
            # (atom_neighs[0].shape[0], (1, -1)) # Vectors
        ]

        edge_desc_irreps = [
            # (edge_soaps.shape[-1] + 1, (0,1)), # Invariant (scalar) quantities
            # (1, (1,-1)), # Edge vector
            # # Quantities coming from the two atoms that form the edge.
            # (atom_soaps.shape[-1] * 2, (0,1)), # Invariant (scalar) quantities
            # #((atom_soaps.shape[-1] + atom_neighs[0].shape[0]) * 2, (0,1)), # Invariant (scalar) quantities
            # #(atom_neighs[0].shape[0] * 2, (1, -1)) # Vectors
        ]

        return all_descriptors, (atom_desc_irreps, edge_desc_irreps)

    @staticmethod
    def compute_atom_descriptor(geom, at, neighs, shifts, processor):
        rs = geom[neighs] + shifts - geom.xyz[at]
        dists = np.linalg.norm(rs, axis=-1)

        r_u = rs / dists.reshape(-1, 1)

        orbs = geom.atoms[at].orbitals
        orb_values = []
        for orb in orbs:
            orb_values.extend(orb.psi(rs))

        r_u = processor.cartesian_to_basis(r_u)
        return np.array([*dists, *r_u.ravel(), *orb_values])

    @staticmethod
    def compute_edge_descriptors(geom, edge_index, shifts, processor):
        xyz = geom.xyz[list(edge_index)]

        edge_vecs = xyz[1] + shifts - xyz[0]
        edge_vecs = processor.cartesian_to_basis(edge_vecs)
        edge_dists = np.linalg.norm(edge_vecs, axis=-1).reshape(-1, 1)

        # orb_values = np.zeros((edge_index.shape[1], geom.orbitals.max() * 2))
        # for i_edge, (edge, edge_vec) in enumerate( zip(edge_index.T, edge_vecs) ):
        #     edge_orb_values = []
        #     for sign, atom in zip((-1, 1), edge):
        #         orbs = geom.atoms[atom].orbitals
        #         for orb in orbs:
        #             edge_orb_values.append(orb.psi(sign * edge_vec))
        #     orb_values[i_edge, :len(edge_orb_values)] = edge_orb_values

        return np.concatenate([edge_dists, edge_vecs / edge_dists], axis=-1)


class HistoryCopy(DescriptorManager):
    @classmethod
    def get_descriptors(
        cls,
        data,
        configs,
        hist_b_dict,
        missing_block_steps,
        data_processor,
        **kwargs,
    ):
        values = {k: v.reshape(len(v), -1) for k, v in hist_b_dict.items() if len(v)}
        if not values:
            return {}, None
        history_len = (len(configs) - 1 if configs is not None else
                       max(len(v) + len(missing_block_steps.get(k, [])) for k, v in values.items()))
        all_descriptors = np.full(
            (history_len, sum(v.shape[1] for v in values.values())), np.nan,
            dtype=np.result_type(np.float32, *(v.dtype for v in values.values())),
        )
        rows = {}
        offset = 0
        for key, value in values.items():
            present = np.ones(history_len, dtype=bool)
            present[missing_block_steps.get(key, [])] = False
            if present.sum() != len(value):
                raise ValueError("Missing block steps do not align with historical labels.")
            rows[key] = present
            all_descriptors[present, offset:offset + value.shape[1]] = value
            offset += value.shape[1]
        # Keep the global matrix descriptor, selecting the same available frames
        # as each block's compacted labels, and no columns containing missing data.
        descriptors = {}
        for key, present in rows.items():
            value = all_descriptors[present]
            descriptors[key] = value[:, np.isfinite(value).all(axis=0)]

        return descriptors, None

    @classmethod
    def predict_from_descriptors(
        cls, history_descriptors, history_labels, block_kwargs
    ):
        history_labels = {k: v[1:] for k, v in history_labels.items()}

        return super().predict_from_descriptors(
            history_descriptors, history_labels, block_kwargs
        )


def get_history_block_dicts(configs, target_data):
    history_len = len(configs) - 1

    # Get the history block dict
    target_config = configs[-1]
    history_configs = configs[:-1]

    # Geometry
    target_geom = target_config.metadata["geometry"]

    arrays = target_data.numpy_arrays()
    stride = 2 if target_data.metadata["data_processor"].symmetric_matrix else 1
    target_edges = arrays.edge_index[:, ::stride]
    target_isc = arrays.neigh_isc[::stride]

    # Initialize the history block dict
    hist_b_dict = {(i, j, isc): [] for (i, j), isc in zip(target_edges.T, target_isc)}
    for i in range(target_data.num_nodes):
        hist_b_dict[i, i, 0] = []

    missing_block_steps = defaultdict(list)

    for i_step, step_config in enumerate(history_configs):
        step_geom = step_config.metadata["geometry"]

        for i, j, isc in hist_b_dict:
            if isc != 0:
                target_sc_off = target_geom.sc_off[isc]
                if np.any(np.abs(target_sc_off) > step_geom.nsc // 2):
                    missing_block_steps[i, j, isc].append(i_step)
                    continue
                new_isc = step_geom.isc_off[tuple(target_sc_off)]
            else:
                new_isc = 0

            block = (step_config.matrix._get_block(i, j, new_isc)
                     if step_config.matrix is not None else None)
            if block is None or not np.isfinite(block).any():
                missing_block_steps[i, j, isc].append(i_step)
            else:
                hist_b_dict[i, j, isc].append(block)

    hist_b_dict = {k: np.array(v) for k, v in hist_b_dict.items() if v}

    return hist_b_dict, missing_block_steps


def _check_reference_layout(matrix, reference):
    if (type(matrix) is not type(reference) or matrix.spin != reference.spin
            or matrix.shape[:2] != reference.shape[:2]
            or not np.array_equal(matrix.geometry.orbitals, reference.geometry.orbitals)
            or not np.array_equal(matrix.geometry.sc_off, reference.geometry.sc_off)):
        raise ValueError("Matrix reference must have the same type, spin, orbitals and image layout.")


def _matrix_delta(matrix, reference, include_overlap=False):
    _check_reference_layout(matrix, reference)
    if not include_overlap:
        matrix = matrix.transform(orthogonal=True) if not matrix.orthogonal else matrix
        reference = reference.transform(orthogonal=True) if not reference.orthogonal else reference
    elif matrix.orthogonal != reference.orthogonal:
        raise ValueError("Explicit overlap components require the same reference overlap layout.")
    return matrix - reference


def _restore_matrix_reference(matrix, reference):
    _check_reference_layout(matrix, reference)
    if isinstance(matrix, sisl.Hamiltonian) and matrix.orthogonal and not reference.orthogonal:
        if (not np.allclose(matrix.geometry.xyz, reference.geometry.xyz)
                or not np.allclose(matrix.geometry.cell, reference.geometry.cell)):
            raise ValueError("Reference overlap must belong to the predicted geometry.")
        channels = [matrix.tocsr(i) + reference.tocsr(i) for i in range(matrix.spin.size(matrix.dtype))]
        return sisl.Hamiltonian.fromsp(matrix.geometry, channels, spin=matrix.spin,
                                      S=reference.tocsr(reference.S_idx))
    if matrix.orthogonal != reference.orthogonal:
        matrix = matrix.transform(orthogonal=True) if not matrix.orthogonal else matrix
        reference = reference.transform(orthogonal=True) if not reference.orthogonal else reference
    return matrix + reference


class MatrixTimeSeriesState:
    processor: Union[MatrixDataProcessor, None]

    configs: deque
    last_matrix_ref: Any

    descriptors: deque

    def __init__(self, history_len, processor=None):
        self.processor = processor

        self.configs = deque(maxlen=history_len + 1)
        self.descriptors = deque(maxlen=history_len + 1)

        self.last_matrix_ref = None
        self.required_matrix_type = None

    def add_processor(self, processor):
        self.processor = processor

    def add_next_geometry(self, geometry):
        next_config = OrbitalConfiguration.from_geometry(
            geometry, metadata={"geometry": geometry}
        )
        self.add_next_config(next_config)

    def add_next_config(self, config):
        self.configs.append(config)

    def add_last_matrix_ref(self, matrix_ref):
        self.last_matrix_ref = matrix_ref

    def add_last_matrix(self, matrix, all_dims=False):
        last_config = self.configs[-1]

        if self.last_matrix_ref is not None:
            include_overlap = all_dims or (self.processor is not None
                                          and self.processor.out_matrix == "hamiltonian"
                                          and self.processor.matrix_component_policy in ("h_and_overlap", "raw_components"))
            matrix = _matrix_delta(matrix, self.last_matrix_ref, include_overlap)

        kwargs = self.processor.get_config_kwargs(matrix) if self.processor else {}
        if all_dims:
            kwargs["matrix_component_policy"] = "raw_components"
        # The regular converter already preserves channels and the target's
        # missing-entry policy, without using matrix entries as row indices.
        last_config.matrix = OrbitalConfiguration.from_matrix(matrix, **kwargs).matrix


def extrapolate_from_series(
    series: MatrixTimeSeriesState,
    descriptor_manager: Type[DescriptorManager] = HistorySOAPs,
    soap_l_max: int = 1,
    descriptor_order: int = 2,
    r_cut: float = 6,
    n_max: int = 1,
    node_rcond: float = 1e-6,
    edge_rcond: float = 1e-6,
):
    if len(series.configs) < 2 or not any(config.matrix is not None for config in list(series.configs)[:-1]):
        raise ValueError("Extrapolation needs at least one historical matrix.")
    if series.processor is None:
        if len(series.configs) == 2:
            return {key: np.nan_to_num(value, nan=0., posinf=0., neginf=0.)
                    for key, value in series.configs[-2].matrix.block_dict.items()}
        raise ValueError("Extrapolation needs a data processor.")

    dataset = [
        BasisMatrixData.from_config(
            config, series.processor, nsc=config.metadata["geometry"].nsc
        )
        for config in series.configs
    ]
    data = list(dataset)

    # data[-1].neigh_isc = np.zeros(data.edge_index.shape[1], dtype=int)

    configs = list(series.configs)
    configs[-1] = copy(configs[-1])

    # geom = configs[-1].metadata["geometry"].copy()
    # geom.set_lattice(geom.lattice.copy())
    # geom.lattice.set_nsc(data[-1].nsc[0])

    # configs[-1].metadata["geometry"] = geom

    hist_b_dict, missing_steps = get_history_block_dicts(configs, data[-1])
    if not hist_b_dict:
        raise ValueError("Extrapolation needs an observed historical matrix block.")
    if len(configs) == 2:
        return {key: np.nan_to_num(value[0], nan=0., posinf=0., neginf=0.)
                for key, value in hist_b_dict.items()}

    descriptors, descriptors_irreps = descriptor_manager.get_descriptors(
        data,
        configs,
        hist_b_dict,
        missing_steps,
        series.processor,
        r_cut=r_cut,
        n_max=n_max,
        l_max=soap_l_max,
        rbf="gto",
    )

    # descriptors_irreps = [descriptors_irreps[0]] * n_atoms + [descriptors_irreps[1]] * edge_index.shape[1]

    # Modify current descriptor. Add a constant descriptor and create higher
    # order descriptors.
    for k, block_desc in descriptors.items():
        ones = np.ones((block_desc.shape[0], 1))

        if descriptor_order == 2:
            order2_descriptors = np.einsum(
                "hx, hy -> hxy", block_desc, block_desc
            ).reshape(block_desc.shape[0], -1)
            descriptors[k] = np.concatenate([ones, order2_descriptors], axis=-1)
        else:
            descriptors[k] = np.concatenate([ones, descriptors[k]], axis=-1)

    block_rcond = {
        (i, j, isc): node_rcond if i == j and isc == 0 else edge_rcond
        for i, j, isc in descriptors
    }

    predicted_labels = descriptor_manager.predict_from_descriptors(
        descriptors, hist_b_dict, block_kwargs={"rcond": block_rcond}
    )

    return predicted_labels


def create_extrapolation_app(
    time_series: Union[Dict[Any, MatrixTimeSeriesState], None] = None,
    matrix_refs: Union[Dict[str, Callable], None] = None,
    data_processors: Union[Dict[str, MatrixDataProcessor], None] = None,
):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse

    if time_series is None:
        time_series = {}
    if matrix_refs is None:
        matrix_refs = {}
    if data_processors is None:
        data_processors = {}

    app = FastAPI()

    @app.exception_handler(ValueError)
    async def invalid_series_input(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/init")
    def init_time_series(history_len: int, data_processor: Optional[str] = None, matrix_type: Optional[str] = None):
        if data_processor is not None:
            data_processor = data_processors[data_processor]
            if matrix_type is not None and data_processor.out_matrix != matrix_type:
                raise HTTPException(status_code=422, detail=f"This workflow requires {matrix_type}, not {data_processor.out_matrix}.")

        time_series[0] = MatrixTimeSeriesState(history_len, processor=data_processor)
        time_series[0].required_matrix_type = matrix_type

    @app.get("/setup_processor")
    def setup_processor(basis_dir: str, matrix: str, series: int = 0):
        this_series = time_series[series]
        if this_series.required_matrix_type is not None and matrix != this_series.required_matrix_type:
            raise HTTPException(status_code=422, detail=f"This workflow requires {this_series.required_matrix_type}.")

        # The basis table.
        basis_ext = (
            "ion.nc" if len(list(Path(basis_dir).glob("*ion.nc"))) > 0 else "ion.xml"
        )
        table = AtomicTableWithEdges.from_basis_dir(basis_dir, basis_ext)

        # The data processor.
        processor = MatrixDataProcessor(
            basis_table=table,
            symmetric_matrix=True,
            sub_point_matrix=False,
            out_matrix=matrix,
        )

        this_series.add_processor(processor)

    @app.get("/extrapolate")
    def write_extrapolate_from_series(
        out: str,
        series: int = 0,
        soap_l_max: int = 1,
        r_cut: float = 6.0,
        n_max: int = 1,
        descriptor_order: int = 2,
        node_rcond: float = 1e-6,
        edge_rcond: float = 1e-6,
        m_0: Optional[str] = None,
        fermi_level: Optional[float] = None,
        orthogonal_basis: bool = False,
    ):
        this_series = time_series[series]

        if len(this_series.configs) < 2 and this_series.last_matrix_ref is None:
            raise HTTPException(status_code=422, detail="Extrapolation needs a historical matrix or prediction reference.")
        if len(this_series.configs) == 1:
            m = this_series.last_matrix_ref
            if m is None:
                raise ValueError(
                    "History depth is 0 and there is no matrix reference (prediction)."
                )
        else:
            predicted = extrapolate_from_series(
                this_series,
                soap_l_max=soap_l_max,
                r_cut=r_cut,
                n_max=n_max,
                descriptor_order=descriptor_order,
                node_rcond=node_rcond,
                edge_rcond=edge_rcond,
            )

            geom = this_series.configs[-1].metadata["geometry"]

            nsc = this_series.configs[-1].metadata["geometry"].nsc
            basis_count = this_series.configs[-1].metadata["geometry"].orbitals

            c = copy(this_series.configs[-1])
            m = get_matrix_cls(this_series.processor.out_matrix)(
                predicted, nsc, basis_count
            )
            c.matrix = m

            m = BasisMatrixData.from_config(c, this_series.processor)
            m = m.convert_to(this_series.processor.default_out_format)

            if this_series.last_matrix_ref is not None:
                m = _restore_matrix_reference(m, this_series.last_matrix_ref)

        if m_0 is not None:
            mat_0 = getattr(
                sisl.get_sile(m_0), f"read_{this_series.processor.out_matrix}"
            )(geometry=m.geometry)
            m = _restore_matrix_reference(m, mat_0)

        write_prediction(m, out, fermi_level=fermi_level, orthogonal_basis=orthogonal_basis)

    @app.get("/add_step")
    def add_geometry(path: str, series: int = 0, matrix_ref: Union[str, None] = None):
        config = OrbitalConfiguration.from_run(path)
        time_series[series].add_next_config(config)

        if matrix_ref is not None:
            time_series[series].add_last_matrix_ref(
                matrix_refs[matrix_ref](config.metadata["geometry"])
            )

        # geometry_xv = sisl.get_sile(Path(path).parent / "siesta.XV").read_geometry()

        # print("XV", np.allclose(geometry.xyz, geometry_xv.xyz))

    @app.get("/add_matrix")
    def add_matrix(path: str, series: int = 0, m_0: Optional[str] = None):
        this_series = time_series[series]

        sile = sisl.get_sile(path)
        matrix = getattr(sile, f"read_{this_series.processor.out_matrix}")()

        if m_0 is not None:
            mat_0 = getattr(
                sisl.get_sile(m_0), f"read_{this_series.processor.out_matrix}"
            )()
            include_overlap = (this_series.processor.out_matrix == "hamiltonian"
                               and this_series.processor.matrix_component_policy in ("h_and_overlap", "raw_components"))
            matrix = _matrix_delta(matrix, mat_0, include_overlap)

        this_series.add_last_matrix(matrix)

    return app
