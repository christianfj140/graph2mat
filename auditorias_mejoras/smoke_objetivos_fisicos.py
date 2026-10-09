"""Read-only scientific references and two-step CPU physical-loss smokes.

Run with CUDA_VISIBLE_DEVICES='' and this checkout's absolute src on PYTHONPATH.
This proves integration and gradients, not accuracy or convergence.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile


CASES = (("density_matrix", "electron_weight"), ("density_matrix", "mulliken_weight"),
    ("density_matrix", "energy_weight"), ("hamiltonian", "energy_weight"),
    ("energy_density_matrix", "energy_weight"))


def validate_paths(fdf, output):
    fdf, output = Path(fdf).resolve(), Path(output).resolve()
    if not fdf.is_file() or fdf.suffix.lower() != ".fdf":
        raise ValueError("an explicit existing FDF source is required")
    dataset = next((p.parent for p in fdf.parents if p.name == "MD_steps"), fdf.parent)
    if output == dataset or dataset in output.parents:
        raise ValueError("output must be outside the source dataset")
    if output.exists():
        raise ValueError("output exists; refusing to overwrite")
    return fdf, output


def source_hashes(fdf):
    paths = [Path(fdf)] + sorted(p for p in Path(fdf).parent.iterdir()
        if p.suffix.upper() in (".TSHS", ".TSDE", ".HSX"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def self_check():
    import numpy as np
    import sisl
    from scipy.sparse import csr_array
    from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
    from graph2mat.core.data.physical import prepare_physical_data, physical_loss_terms
    from graph2mat.core.data.metrics import elementwise_mse

    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.8, q0=1.)])
    g = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=atom, lattice=[20.]*3)
    g.set_nsc([1, 1, 1])
    s = csr_array([[1., .2], [.2, 1.]])
    table = AtomicTableWithEdges([atom])
    processor = MatrixDataProcessor(basis_table=table, out_matrix="density_matrix",
        symmetric_matrix=True, sub_point_matrix=False)
    data = prepare_physical_data(processor, overlap=sisl.Overlap.fromsp(g, s),
        density=sisl.DensityMatrix.fromsp(g, csr_array(np.full((2, 2), 5/6))),
        hamiltonian=sisl.Hamiltonian.fromsp(g, csr_array([[.5, -.3], [-.3, .5]])),
        energy_density=sisl.EnergyDensityMatrix.fromsp(g, csr_array(np.full((2, 2), 5/36))),
        kpoints=np.zeros((1, 3)), kweights=np.ones(1), requested=("charge", "energy"))
    nodes = data.point_labels.copy()
    nodes[0] += .1
    args = dict(batch=data, basis_table=table)
    total, _ = elementwise_mse(electron_weight=2., mulliken_weight=3.)(nodes_pred=nodes,
        nodes_ref=data.point_labels, edges_pred=data.edge_labels, edges_ref=data.edge_labels, **args)
    energy, _ = physical_loss_terms(nodes, data.edge_labels, energy_weight=1., **args)
    assert np.isclose(total, 7/300, rtol=0, atol=1e-14)
    assert np.isclose(energy, 1/800, rtol=0, atol=1e-14)


def processor_for(fdf, target):
    import sisl
    from graph2mat import AtomicTableWithEdges, MatrixDataProcessor

    table = AtomicTableWithEdges(sisl.get_sile(fdf).read_overlap(order=["TSHS"]).geometry.atoms.atom)
    return MatrixDataProcessor(basis_table=table, out_matrix=target,
        symmetric_matrix=True, sub_point_matrix=False, n_matrix_components=1)


def load_sample(fdf, processor, term=None):
    """Matched one-channel labels; control skips physical certification/auxiliaries."""
    import sisl
    from graph2mat.bindings.torch import TorchBasisMatrixData
    from graph2mat.core.data.physical import load_physical_data

    if term is not None:
        return load_physical_data(fdf, processor, requested=("energy",) if term == "energy_weight"
            else ("charge",), data_cls=TorchBasisMatrixData)
    reader = sisl.get_sile(fdf)
    geometry = reader.read_overlap(order=["TSHS"]).geometry
    if processor.out_matrix == "hamiltonian":
        matrix = reader.read_hamiltonian(order=["TSHS"], geometry=geometry)
    else:
        tsde = sisl.get_sile(reader.dir_file(reader.get("SystemLabel", default="siesta") + ".TSDE"))
        method = (tsde.read_density_matrix if processor.out_matrix == "density_matrix"
            else tsde.read_energy_density_matrix)
        raw = method(geometry=geometry)
        if not raw.spin.is_unpolarized:
            raise ValueError("this real-data CPU evidence script supports unpolarized sources only")
        matrix = type(raw).fromsp(raw.geometry, raw.tocsr(0))
    return TorchBasisMatrixData.new(matrix, processor)


def model_for(table, weights):
    import torch
    from graph2mat.core.data.metrics import elementwise_mse
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

    torch.manual_seed(13)
    return LitMACEMatrixModel(basis_table=table, symmetric_matrix=True,
        hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o",
        num_bessel=2, max_ell=1, num_interactions=2, correlation=1,
        n_matrix_components=1, coefficient_training=False,
        loss=elementwise_mse, loss_kwargs=weights).double()


def loader_for(data, table):
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    from graph2mat.tools.lightning.data import TypeLayoutCollate

    return DataLoader([data], batch_size=1,
        collate_fn=TypeLayoutCollate(Batch.from_data_list, table, symmetric=True))


def fit_steps(model, data, table, root, steps=2):
    import pytorch_lightning as pl

    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=steps,
        limit_train_batches=1, limit_val_batches=0, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, default_root_dir=root)
    trainer.fit(model, train_dataloaders=loader_for(data, table))
    assert trainer.global_step == steps
    return {"steps": trainer.global_step, "loss": float(trainer.callback_metrics["train_loss_epoch"])}


def image_trace(x, y, geometry, *, unknown_x=False, unknown_y=False):
    """Independent scalar-channel oracle, joining actual opposite image entries."""
    import numpy as np

    a, b = x.tocoo(), y.tocoo()
    left = {(int(i), int(j)): v for i, j, v in zip(a.row, a.col, a.data)}
    right = {}
    for i, j, value in zip(b.row, b.col, b.data):
        image, orbital = divmod(int(j), geometry.no)
        reverse = int(geometry.sc_index(-geometry.sc_off[image]))
        right[(orbital, reverse*geometry.no + int(i))] = value
    population = np.zeros(geometry.na)
    for key in left.keys() | right.keys():
        a = left.get(key, np.nan if unknown_x else 0.)
        b = right.get(key, np.nan if unknown_y else 0.)
        if a == 0 or b == 0:
            continue
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError("oracle: unknown coefficient with a nonzero opposite partner")
        population[geometry.o2a(key[0])] += a*b
    return population


def reference_oracles(fdf):
    import numpy as np
    import sisl
    from graph2mat.bindings.torch import TorchBasisMatrixData
    from graph2mat.core.data.physical import load_physical_data, physical_contractions, physical_layout

    reader = sisl.get_sile(fdf)
    s = reader.read_overlap(order=["TSHS"])
    g = s.geometry
    tsde = sisl.get_sile(reader.dir_file(reader.get("SystemLabel", default="siesta") + ".TSDE"))
    d = tsde.read_density_matrix(geometry=g)
    e = tsde.read_energy_density_matrix(geometry=g)
    h = reader.read_hamiltonian(order=["TSHS"], geometry=g)
    population = image_trace(d.tocsr(0), s.tocsr(), g, unknown_x=True)
    residual = image_trace(h.tocsr(0), d.tocsr(0), g, unknown_y=True) - image_trace(e.tocsr(0), s.tocsr(), g, unknown_x=True)
    # sisl Mulliken needs genuine S, not the dummy overlap stored by raw TSDE reads.
    with_s = sisl.DensityMatrix.fromsp(g, d.tocsr(0), S=s.tocsr())
    np.testing.assert_allclose(population, with_s.mulliken("atom"), rtol=0, atol=1e-8)
    processor = processor_for(fdf, "density_matrix")
    data = load_physical_data(fdf, processor, requested=("charge", "energy"), data_cls=TorchBasisMatrixData)
    np.testing.assert_allclose(data.physical_population_reference.numpy(), population, rtol=0, atol=1e-8)
    np.testing.assert_allclose(data.physical_energy_reference_residual.numpy(), residual, rtol=0, atol=1e-8)
    # The writer may use a smaller image box than the archive; remap its columns.
    represented = processor.matrix_from_data(data, out_format="sisl_DM", threshold=0.)
    coo = represented.tocsr(0).tocoo()
    from scipy.sparse import csr_array
    columns = np.array([int(g.sc_index(represented.geometry.sc_off[int(j)//g.no]))*g.no + int(j)%g.no for j in coo.col])
    matrix_g = csr_array((coo.data, (coo.row, columns)), shape=d.tocsr(0).shape)
    represented_population = image_trace(matrix_g, s.tocsr(), g)
    layout = physical_layout(data, processor.basis_table)
    contracted = physical_contractions(data.physical_reference_nodes, data.physical_reference_edges,
        data.physical_overlap_nodes, data.physical_overlap_edges, layout=layout,
        reverse_partner_edges=data.physical_overlap_reverse_edges)
    np.testing.assert_allclose(contracted.numpy(), represented_population, rtol=0, atol=1e-8)
    return {"population_full": population.tolist(), "population_G": represented_population.tolist(),
        "population_tail_outside_G": (population-represented_population).tolist(),
        "energy_reference_residual_eV_atom": residual.tolist(), "sisl_mulliken_checked": True,
        "opposite_image_oracle_checked": True}


def smoke_case(fdf, target, term, root):
    import torch
    from graph2mat.core.data.metrics import elementwise_mse

    processor = processor_for(fdf, target)
    # Unsupported references are explicit preparation failures, never fake labels.
    try:
        data = load_sample(fdf, processor, term)
    except ValueError as error:
        if target == "hamiltonian" and "unknown/unobserved density partner where predicted H can vary" in str(error):
            return {"target": target, "term": term, "status": "blocked", "reason": str(error)}
        raise
    control = load_sample(fdf, processor)
    for key in ("point_labels", "edge_labels"):
        torch.testing.assert_close(data[key], control[key], rtol=0, atol=0, equal_nan=True)
    model = model_for(processor.basis_table, {term: .3})
    initial = deepcopy(model.state_dict())
    out = model(data)
    args = dict(nodes_pred=out["node_labels"], nodes_ref=data.point_labels,
        edges_pred=out["edge_labels"], edges_ref=data.edge_labels, batch=data, basis_table=processor.basis_table)
    total, _ = model.loss_fn(**args)
    base, _ = elementwise_mse()(**args)
    physical = total-base
    gradients = torch.autograd.grad(physical, tuple(model.parameters()), allow_unused=True)
    used = [g for g in gradients if g is not None]
    assert used and all(torch.isfinite(g).all() for g in used)
    norm = sum(float(g.abs().sum()) for g in used)
    assert norm > 0 and torch.isfinite(physical) and float(physical.detach()) > 0
    model.load_state_dict(initial)
    candidate_fit = fit_steps(model, data, processor.basis_table, root)
    baseline = model_for(processor.basis_table, {})
    baseline.load_state_dict(initial)
    control_fit = fit_steps(baseline, control, processor.basis_table, root)
    return {"target": target, "term": term, "status": "verified_cpu", "weight": .3,
        "physical_loss_initial": float(physical.detach()), "physical_gradient_l1": norm,
        "candidate_fit": candidate_fit, "control_fit": control_fit}


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    fdf, output = validate_paths(args.fdf, args.output)
    if torch.cuda.is_available():
        parser.error("disable CUDA; this evidence is CPU-only")
    torch.set_default_dtype(torch.float64)
    self_check()
    before = source_hashes(fdf)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="g2m-physical-smoke-", dir=output.parent) as root:
            cases = [smoke_case(fdf, target, term, root) for target, term in CASES]
        result = {"fdf": str(fdf), "source_hashes": before, "cases": cases,
            "oracles": reference_oracles(fdf), "scope": "two-step CPU integration, not accuracy",
            "model_limit": "tiny MACE does not span every high-l coefficient; this is a gradient smoke"}
    finally:
        assert before == source_hashes(fdf), "scientific sources changed during smoke"
    result["source_hashes_unchanged"] = True
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"CPU smoke written: {output}")


if __name__ == "__main__":
    main()
