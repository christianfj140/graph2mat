"""Fixed semantic layout and references for joint H/S/DM/EDM prediction."""

from dataclasses import dataclass
from copy import copy

import numpy as np
import sisl

from .configuration import BasisConfiguration
from .formats import Formats
from .matrices import OrbitalMatrix, get_matrix_cls
from .processing import BasisMatrixData
from .sparse import csr_to_block_dict


MATRIX_TARGETS = ("hamiltonian", "overlap", "density_matrix", "energy_density_matrix")
_UNITS = {"hamiltonian": "eV", "overlap": "dimensionless",
          "density_matrix": "sisl_occupation", "energy_density_matrix": "eV"}


@dataclass(frozen=True)
class MatrixTargetLayout:
    targets: tuple[str, ...]
    spin_components: int

    def __post_init__(self):
        if (isinstance(self.targets, str) or not self.targets
                or any(t not in MATRIX_TARGETS for t in self.targets)
                or len(set(self.targets)) != len(self.targets)
                or not {"hamiltonian", "overlap"}.issubset(self.targets)):
            raise ValueError("matrix_targets must select H and S and optional DM/EDM without duplicates")
        if (not isinstance(self.spin_components, (int, np.integer))
                or isinstance(self.spin_components, (bool, np.bool_)) or self.spin_components not in (1, 2)):
            raise ValueError("matrix_target_spin must be 1 or 2 (real unpolarized/collinear)")
        object.__setattr__(self, "targets", tuple(t for t in MATRIX_TARGETS if t in self.targets))
        object.__setattr__(self, "spin_components", int(self.spin_components))

    @property
    def slices(self):
        result, start = {}, 0
        for name in self.targets:
            stop = start + (1 if name == "overlap" else self.spin_components)
            result[name] = slice(start, stop)
            start = stop
        return result

    @property
    def n_components(self):
        return next(reversed(self.slices.values())).stop

    @property
    def signature(self):
        return np.array([[1, self.spin_components, *(int(t in self.targets) for t in MATRIX_TARGETS)]],
                        dtype=np.int64)

    def to_dict(self):
        return {"version": 1, "targets": list(self.targets), "spin_components": self.spin_components,
                "n_components": self.n_components,
                "slices": {name: [s.start, s.stop] for name, s in self.slices.items()},
                "units": {name: _UNITS[name] for name in self.targets}}

    @classmethod
    def from_dict(cls, value):
        try:
            layout = cls(tuple(value["targets"]), value["spin_components"])
            if value != layout.to_dict():
                raise ValueError("multitarget descriptor version/layout does not match its semantics")
            return layout
        except (KeyError, TypeError) as error:
            raise ValueError("invalid multitarget descriptor") from error


def _set_array(data, name, array):
    value = data.process_input_array(name, np.ascontiguousarray(array))
    setattr(data, name, value)
    if "_data" in vars(data):
        data._data[name] = value


def _flatten_on_graph(matrix, channels, data, *, symmetric, fill_value):
    arrays = data.numpy_arrays()
    blocks = csr_to_block_dict(matrix._csr, matrix.atoms, nsc=matrix.nsc,
        matrix_cls=OrbitalMatrix if isinstance(matrix, sisl.Overlap) else get_matrix_cls(type(matrix)),
        fill_value=fill_value, component_indices=range(channels))
    values = blocks.to_flat_nodes_and_edges(
        edge_index=arrays.edge_index[:, ::2] if symmetric else arrays.edge_index,
        edge_sc_shifts=arrays.neigh_isc[::2] if symmetric else arrays.neigh_isc)
    return tuple(value.reshape(-1, channels) for value in values)


def normalize_multitarget_physical_labels(labels, targets):
    """Reference preparation by target; the existing list syntax has fixed semantics."""
    from .physical import normalize_physical_labels

    if not labels:
        return {}
    if not isinstance(labels, dict):
        requested = normalize_physical_labels(labels)
        if "charge" in requested and "density_matrix" not in targets:
            raise ValueError("charge physical_labels require selecting density_matrix")
        labels = {target: tuple(label for label in requested
                    if label == "energy" or target == "density_matrix")
                  for target in targets if target != "overlap"}
    result = {}
    for target, requested in labels.items():
        if target not in targets or target == "overlap":
            raise ValueError(f"{target}: physical labels require a selected H/DM/EDM target")
        requested = normalize_physical_labels(requested)
        if "charge" in requested and target != "density_matrix":
            raise ValueError(f"{target}: charge physical labels require density_matrix")
        if requested:
            result[target] = requested
    return result


def prepare_multitarget_data(
    processor, *, hamiltonian, overlap, density=None, energy_density=None,
    observation_masks=None, kpoints=None, kweights=None, physical_labels=None,
    data_cls=BasisMatrixData,
):
    """Prepare direct matrix labels on one graph; missing DM/EDM entries stay unknown."""
    from .physical import _observed_sparse, _physical_channels, _validate_geometry, _validate_hermitian

    layout = processor.matrix_target_layout
    if layout is None:
        raise ValueError("prepare_multitarget_data requires matrix_targets")
    matrices = dict(hamiltonian=hamiltonian, overlap=overlap, density_matrix=density,
                    energy_density_matrix=energy_density)
    classes = dict(hamiltonian=sisl.Hamiltonian, overlap=sisl.Overlap,
                   density_matrix=sisl.DensityMatrix, energy_density_matrix=sisl.EnergyDensityMatrix)
    if not isinstance(overlap, sisl.Overlap):
        raise ValueError("overlap reference is absent or has wrong matrix type")
    for name in layout.targets:
        matrix = matrices[name]
        if not isinstance(matrix, classes[name]):
            raise ValueError(f"{name} reference is absent or has wrong matrix type")
        _validate_geometry(overlap.geometry, matrix, name)
        channels = _physical_channels(matrix, name)
        if channels != (1 if name == "overlap" else layout.spin_components):
            raise ValueError(f"{name} reference spin does not match multitarget layout")
        _validate_hermitian(matrix, channels, name)
        if name in ("density_matrix", "energy_density_matrix"):
            transposed = matrix.transpose()
            for component in range(channels):
                if (_observed_sparse(matrix, component) != _observed_sparse(transposed, component)).nnz:
                    raise ValueError(f"{name} observation support is not Hermitian across opposite images")
        if name != "overlap" and not matrix.orthogonal:
            difference = matrix.tocsr(matrix.S_idx) - overlap.tocsr(0)
            if difference.nnz and (not np.isfinite(difference.data).all()
                                   or np.max(np.abs(difference.data)) > 1e-10):
                raise ValueError(f"{name} embedded overlap does not match reference overlap")

    base_processor = processor.copy(matrix_targets=None, matrix_target_spin=None,
        n_matrix_components=layout.spin_components, out_matrix="hamiltonian",
        matrix_component_policy="spin_h_only" if layout.spin_components == 2 else "h_only")
    config = BasisConfiguration.new(hamiltonian,
                                   matrix_component_policy=base_processor.matrix_component_policy)
    data = data_cls.from_config(config, base_processor)
    data.metadata["data_processor"] = processor
    data.metadata["matrix_target_descriptor"] = layout.to_dict()
    node_parts, edge_parts = [], []
    observation_masks = observation_masks or {}
    if any(t not in layout.targets or t in ("hamiltonian", "overlap") for t in observation_masks):
        raise ValueError("observation masks may restrict selected DM/EDM targets only")
    for name in layout.targets:
        channels = 1 if name == "overlap" else layout.spin_components
        nodes, edges = _flatten_on_graph(matrices[name], channels, data,
            symmetric=processor.symmetric_matrix,
            fill_value=np.nan if name in ("density_matrix", "energy_density_matrix") else 0.)
        if name in observation_masks:
            masks = observation_masks[name]
            if len(masks) != channels:
                raise ValueError(f"{name} observation mask spin does not match")
            for component, mask in enumerate(masks):
                if mask.shape != matrices[name].shape[:2] or mask.dtype != np.dtype(bool):
                    raise ValueError(f"{name} observation mask must have matching shape and boolean dtype")
                support = matrices[name].tocsr(component).copy()
                support.data = np.ones(support.nnz, dtype=bool)
                if (mask.astype(np.int8) - mask.multiply(support).astype(np.int8)).nnz:
                    raise ValueError(f"{name} observation mask declares an absent entry observed")
                mask_matrix = sisl.Overlap.fromsp(overlap.geometry, mask.astype(float))
                _validate_hermitian(mask_matrix, 1, f"{name} observation mask")
                mn, me = _flatten_on_graph(mask_matrix, 1, data,
                    symmetric=processor.symmetric_matrix, fill_value=0.)
                nodes[:, component] = np.where(mn[:, 0] != 0, nodes[:, component], np.nan)
                edges[:, component] = np.where(me[:, 0] != 0, edges[:, component], np.nan)
        node_parts.append(nodes)
        edge_parts.append(edges)
    for name, value in (("point_labels", np.concatenate(node_parts, axis=1)),
                        ("edge_labels", np.concatenate(edge_parts, axis=1))):
        _set_array(data, name, value)
    _set_array(data, "matrix_known_nodes", np.isfinite(np.concatenate(node_parts, axis=1)))
    _set_array(data, "matrix_known_edges", np.isfinite(np.concatenate(edge_parts, axis=1)))
    _set_array(data, "matrix_target_signature", layout.signature)
    requested_by_target = normalize_multitarget_physical_labels(physical_labels, layout.targets)
    for target, requested in requested_by_target.items():
        from .physical import prepare_physical_data, physical_loss_terms

        view = target_data_view(data, layout, target)
        target_processor = base_processor.copy(out_matrix=target)
        view = prepare_physical_data(target_processor, overlap=overlap, density=density,
            hamiltonian=hamiltonian, energy_density=energy_density, kpoints=kpoints,
            kweights=kweights, requested=requested, data_cls=data_cls, _data=view)
        columns = layout.slices[target]
        for side, packed_mask in (("nodes", data.matrix_known_nodes), ("edges", data.matrix_known_edges)):
            _set_array(view, f"physical_reference_known_{side}", view.ensure_numpy(packed_mask)[:, columns])
        physical_loss_terms(view.point_labels, view.edge_labels, batch=view,
            basis_table=processor.basis_table, electron_weight=float("charge" in requested),
            energy_weight=float("energy" in requested))
        for name, value in view._data.items():
            if name.startswith("physical_") and "__" not in name:
                _set_array(data, f"physical_{target}__{name}", view.ensure_numpy(value))
    return data


def target_data_view(data, layout, target):
    """Shallow differentiable view of one target, never a NumPy copy of Torch labels."""
    if target not in layout.targets:
        raise ValueError(f"matrix target {target!r} is not selected")
    view = copy(data)
    if "_data" in vars(data):
        view._data = dict(data._data)
    component_slice = layout.slices[target]
    for name in ("point_labels", "edge_labels", "matrix_known_nodes", "matrix_known_edges"):
        if name.startswith("matrix_known_") and data._data.get(name) is None:
            continue  # Geometry-only predictions have no reference observation masks.
        values = data[name][..., component_slice]
        setattr(view, name, values)
        if "_data" in vars(view):
            view._data[name] = values
    prefix = f"physical_{target}__"
    for name, value in data._data.items():
        if name.startswith(prefix):
            field = name[len(prefix):]
            setattr(view, field, value)
            if "_data" in vars(view):
                view._data[field] = value
    return view


def load_multitarget_data(path, processor, *, physical_labels=None, data_cls=BasisMatrixData):
    """Read local FDF references selected by a semantic layout, never ZIP placeholders."""
    from .physical import _read_reference_bundle

    layout = processor.matrix_target_layout
    if layout is None:
        raise ValueError("load_multitarget_data requires matrix_targets")
    physical_labels = normalize_multitarget_physical_labels(physical_labels, layout.targets)
    energy = any("energy" in requested for requested in physical_labels.values())
    bundle, metadata = _read_reference_bundle(path,
        need_density="density_matrix" in layout.targets or bool(physical_labels),
        need_energy_density="energy_density_matrix" in layout.targets or energy,
        need_hamiltonian=True, require_mesh=bool(physical_labels), processor=processor)
    data = prepare_multitarget_data(processor, **bundle,
                                   physical_labels=physical_labels, data_cls=data_cls)
    data.metadata["physical_reference"] = metadata
    data.metadata["path"] = str(path)
    return data


def reconstruct_target(processor, data, target, *, threshold=None, out_format=None):
    """Construct a selected physical matrix, explicitly carrying predicted S."""
    layout = processor.matrix_target_layout
    if target not in layout.targets:
        raise ValueError("select matrix_target from the configured matrix targets before conversion")
    expected_format = {"hamiltonian": Formats.SISL_H, "overlap": None,
                       "density_matrix": Formats.SISL_DM, "energy_density_matrix": Formats.SISL_EDM}[target]
    if out_format not in (None, Formats.SISL, expected_format):
        raise ValueError(f"{target}: incompatible physical output format {out_format!r}")
    arrays = data.numpy_arrays()
    signature = arrays.matrix_target_signature
    if (signature.dtype.kind not in "iu" or signature.shape != (1, 6)
            or not np.array_equal(signature, layout.signature)):
        raise ValueError("physical reconstruction semantic layout signature mismatch")
    for name in ("point_labels", "edge_labels"):
        values = arrays[name]
        if values.ndim != 2 or values.shape[1] != layout.n_components:
            raise ValueError("physical reconstruction requires the complete semantic packing")
        for component_slice in (layout.slices[target], layout.slices["overlap"]):
            if not np.isfinite(values[:, component_slice]).all():
                raise ValueError(f"{target}: reconstruction requires finite selected values and predicted overlap")
    scalar = processor.copy(matrix_targets=None, matrix_target_spin=None, n_matrix_components=1,
                            out_matrix=None, matrix_component_policy="h_only")
    overlap_view = target_data_view(data, layout, "overlap")
    for name in ("point_labels", "edge_labels"):
        values = overlap_view[name][:, 0]
        setattr(overlap_view, name, values)
        if "_data" in vars(overlap_view):
            overlap_view._data[name] = values
    overlap = scalar.labels_to(Formats.SISL, overlap_view.numpy_arrays(), threshold=None,
                               sp_class=sisl.Overlap)
    if target == "overlap":
        return overlap
    selected = target_data_view(data, layout, target)
    csrs = []
    for component in range(layout.spin_components):
        one = copy(selected)
        if "_data" in vars(selected):
            one._data = dict(selected._data)
        for name in ("point_labels", "edge_labels"):
            values = selected[name][:, component]
            setattr(one, name, values)
            if "_data" in vars(one):
                one._data[name] = values
        csr = scalar.labels_to(Formats.SCIPY_CSR, one.numpy_arrays(), threshold=threshold)
        csr.indices = csr.indices.astype(np.int32)
        csr.indptr = csr.indptr.astype(np.int32)
        csrs.append(csr)
    cls = {"hamiltonian": sisl.Hamiltonian, "density_matrix": sisl.DensityMatrix,
           "energy_density_matrix": sisl.EnergyDensityMatrix}[target]
    return cls.fromsp(overlap.geometry, csrs[0] if layout.spin_components == 1 else csrs,
                      S=overlap.tocsr(0), spin="unpolarized" if layout.spin_components == 1 else "polarized")
