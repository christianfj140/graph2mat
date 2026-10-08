from io import BytesIO
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import sisl
import numpy as np

from graph2mat import AtomicTableWithEdges, BasisTableWithEdges, MatrixDataProcessor, PointBasis
from graph2mat.core.data.node_feats import OneHotZ
from graph2mat.tools.server.server_app import create_server_app


def _extra_node_feature(config, processor):
    return np.ones((len(config.positions), 1))


@pytest.mark.parametrize("atomic", [False, True])
@pytest.mark.parametrize("array_radius", [False, True])
def test_model_info_serializes_real_processor_as_json(atomic, array_radius):
    radius = np.array([2., 3., 3., 3.]) if array_radius else np.float64(2.)
    if atomic:
        atom = sisl.Atom("C", orbitals=[sisl.AtomicOrbital(l=0, R=2.), *[
            sisl.AtomicOrbital(l=1, m=m, R=3. if array_radius else 2.) for m in (-1, 0, 1)
        ]])
        table = AtomicTableWithEdges([atom])
        expected_radius = [2., 3., 3., 3.] if array_radius else [2., 2., 2., 2.]
        convention = "siesta_spherical"
    else:
        table = BasisTableWithEdges([PointBasis(np.int64(6), R=radius, basis=[1, 1], basis_convention="cartesian")])
        expected_radius = [2., 3., 3., 3.] if array_radius else 2.
        convention = "cartesian"
    processor = MatrixDataProcessor(
        basis_table=table, symmetric_matrix=True, sub_point_matrix=False,
        out_matrix="density_matrix", n_matrix_components=np.int64(2),
        matrix_component_policy="raw_components", node_attr_getters=[OneHotZ, _extra_node_feature],
    )
    app = create_server_app({"dm": {
        "data_processor": processor, "prediction_function": lambda data: data,
        "description": "Two-component density matrix", "files": {},
        "test_metrics_summary": "",
    }})
    response = TestClient(app).get("/api/models/dm/info")

    assert response.status_code == 200
    assert response.json() == {
        "basis_table": {"basis": [{"type": 6, "R": expected_radius,
                                   "basis": [[1, 0, 1], [1, 1, -1]], "basis_convention": convention}]},
        "symmetric_matrix": True, "sub_point_matrix": False, "out_matrix": "density_matrix",
        "n_matrix_components": 2, "matrix_component_policy": "raw_components",
        "node_attr_getters": ["graph2mat.core.data.node_feats.OneHotZ", __name__ + "._extra_node_feature"],
        "description": "Two-component density matrix",
    }
    assert processor.basis_table is table
    assert processor.node_attr_getters == [OneHotZ, _extra_node_feature]


@pytest.mark.parametrize("spec_kind", ["checkpoint", "minimal_spec", "described_spec"])
def test_filesystem_checkpoint_info_accepts_optional_description(tmp_path, spec_kind):
    import pytorch_lightning as pl
    import yaml
    from graph2mat import BasisConfiguration
    from graph2mat.tools.lightning import MatrixDataModule
    from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel
    from graph2mat.tools.server.server_app import create_server_app_from_filesystem

    pl.seed_everything(0, workers=True)
    basis = AtomicTableWithEdges([sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.)])])
    config = BasisConfiguration(point_types=[1, 1], positions=np.array([[0., 0., 0.], [1., 0., 0.]]),
                                basis=basis, cell=np.eye(3) * 10, pbc=(False, False, False), matrix=np.ones((2, 2)))
    module = MatrixDataModule(basis_table=basis, symmetric_matrix=True, sub_point_matrix=False,
                              out_matrix="density_matrix", train_runs=[config], val_runs=[], loader_threads=0)
    model = LitMACEMatrixModel(basis_table=basis, symmetric_matrix=True, hidden_irreps="2x0e+2x1o",
                              edge_hidden_irreps="2x0e+2x1o", num_bessel=2, max_ell=1)
    trainer = pl.Trainer(accelerator="cpu", devices=1, fast_dev_run=True, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, datamodule=module)
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    path = checkpoint
    description = "A described density checkpoint" if spec_kind == "described_spec" else ""
    if spec_kind != "checkpoint":
        spec = {"files": {"ckpt": "model.ckpt"}}
        if description:
            spec["description"] = description
        path = tmp_path / "spec.yaml"
        path.write_text(yaml.safe_dump(spec))
    app = create_server_app_from_filesystem({"dm": str(path)}, cpu=True)
    response = TestClient(app, raise_server_exceptions=False).get("/api/models/dm/info")
    assert response.status_code == 200, response.text
    assert response.json()["description"] == description
    assert response.json()["out_matrix"] == "density_matrix"
    assert response.json()["n_matrix_components"] == 1
    module.teardown(None)


def test_frontend_route_names_and_operation_ids_are_preserved():
    app = create_server_app({})
    paths = ["/models/{model_name}/{model_action}", "/models/{model_name}", "/models", "/about", "/"]
    routes = [route for route in app.routes if route.path in paths]
    assert [route.name for route in routes] == ["get"] * 5
    assert [app.openapi()["paths"][path]["get"]["operationId"] for path in paths] == [
        "get_models__model_name___model_action__get", "get_models__model_name__get",
        "get_models_get", "get_about_get", "get__get",
    ]


@pytest.mark.parametrize("path", ["/", "/models", "/about", "/models/dm", "/models/dm/predict", "/models/dm/test"])
def test_frontend_pages_render_with_real_processor(path):
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges([sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.)])]),
        out_matrix="density_matrix", symmetric_matrix=True, sub_point_matrix=False,
    )
    app = create_server_app({"dm": {
        "data_processor": processor, "prediction_function": lambda data: data,
        "description": "Density model", "authors": [], "files": {}, "test_metrics_summary": "",
    }})
    response = TestClient(app, raise_server_exceptions=False).get(path)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/html")
    assert "<!doctype html>" in response.text


class _DummyMatrix:
    def write(self, path, **kwargs):
        with open(path, "wb") as fd:
            fd.write(b"dummy")


class _DummyProcessor:
    out_matrix = "hamiltonian"

    def torch_predict(self, prediction_function, geometry):
        return _DummyMatrix()


def test_predict_uses_hamiltonian_suffix(monkeypatch):
    class FakeSile:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read_geometry(self):
            return object()

    monkeypatch.setattr(sisl, "get_sile_class", lambda _: FakeSile)

    app = create_server_app(
        {
            "h_model": {
                "prediction_function": lambda data: data,
                "data_processor": _DummyProcessor(),
                "description": "",
                "authors": [],
                "files": {},
                "root_dir": ".",
                "test_metrics_summary": "",
            }
        }
    )
    client = TestClient(app)

    response = client.post(
        "/api/models/h_model/predict",
        files={"geometry_file": ("geom.xyz", BytesIO(b"X"), "text/plain")},
    )

    assert response.status_code == 200
    assert ".nc" in response.headers["content-disposition"]


@pytest.mark.parametrize("output_path", [None, "prediction.TSHS"])
def test_local_prediction_resolves_output_beside_geometry(tmp_path, monkeypatch, output_path):
    class GeometrySile:
        def read_geometry(self):
            return object()

    monkeypatch.setattr(sisl, "get_sile", lambda path: GeometrySile())
    current_directory = tmp_path / "cwd"
    current_directory.mkdir()
    monkeypatch.chdir(current_directory)
    app = create_server_app({"h_model": {
        "prediction_function": lambda data: data, "data_processor": _DummyProcessor(),
        "files": {}, "test_metrics_summary": "",
    }}, local=True)
    geometry = tmp_path / "geometry.xyz"
    params = {"geometry_path": str(geometry)}
    if output_path is not None:
        params["output_path"] = output_path

    response = TestClient(app).get("/api/models/h_model/local_write_predict", params=params)

    assert response.status_code == 200
    expected = tmp_path / (output_path or "ML_prediction.nc")
    assert Path(response.json()) == expected
    assert expected.read_bytes() == b"dummy"


@pytest.mark.parametrize("allow_overwrite", [False, True])
def test_local_prediction_respects_overwrite_permission(tmp_path, monkeypatch, allow_overwrite):
    class GeometrySile:
        def read_geometry(self):
            return object()

    monkeypatch.setattr(sisl, "get_sile", lambda path: GeometrySile())
    app = create_server_app({"h_model": {
        "prediction_function": lambda data: data, "data_processor": _DummyProcessor(),
        "files": {}, "test_metrics_summary": "",
    }}, local=True)
    output = tmp_path / "existing.TSHS"
    output.write_bytes(b"existing matrix")

    response = TestClient(app).get("/api/models/h_model/local_write_predict", params={
        "geometry_path": str(tmp_path / "geometry.xyz"), "output_path": str(output),
        "allow_overwrite": allow_overwrite,
    })

    assert response.status_code == (200 if allow_overwrite else 409)
    assert output.read_bytes() == (b"dummy" if allow_overwrite else b"existing matrix")


@pytest.mark.parametrize("pooled", [False, True])
def test_uploaded_matrix_is_evaluated_through_real_data_processor(tmp_path, pooled):
    import torch
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])
    geometry = sisl.Geometry([[0, 0, 0], [1, 0, 0]], atoms=atom,
                             lattice=sisl.Lattice([10, 10, 10], nsc=[1, 1, 1]))
    target = sisl.DensityMatrix(geometry)
    target[0, 0], target[1, 1] = 1, 2
    target[0, 1], target[1, 0] = 0.3, 0.3
    geometry_file, matrix_file = tmp_path / "geometry.xyz", tmp_path / "target.DM"
    geometry.write(geometry_file)
    target.write(matrix_file)
    processor = MatrixDataProcessor(basis_table=AtomicTableWithEdges([atom]),
                                    out_matrix="density_matrix", symmetric_matrix=True,
                                    sub_point_matrix=False)
    def predict(data):
        if pooled:
            torch.testing.assert_close(data.ptr, torch.tensor([0, 2]))
            totals = torch.zeros(1).scatter_add(0, data.batch, torch.ones(2))
            nodes = data.point_labels + totals[data.batch]
        else:
            nodes = data.point_labels
        return {"node_labels": nodes, "edge_labels": data.edge_labels}

    app = create_server_app({"dm_model": {
        "prediction_function": predict,
        "data_processor": processor, "files": {}, "test_metrics_summary": "",
    }})

    response = TestClient(app).post("/api/models/dm_model/test", files={
        "geometry_file": ("geometry.xyz", geometry_file.read_bytes()),
        "output_matrix_file": ("target.DM", matrix_file.read_bytes()),
    })

    assert response.status_code == 200
    assert response.json()["elementwise_mse"] == pytest.approx(8 / 3 if pooled else 0)
    assert response.json()["elementwise_mae"] == pytest.approx(4 / 3 if pooled else 0)


@pytest.mark.parametrize("prediction_components", ["h_flat", "h_column", "hs"])
def test_uploaded_hamiltonian_evaluates_only_predicted_components(tmp_path, prediction_components):
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])
    geometry = sisl.Geometry([[0, 0, 0], [1, 0, 0], [0, 1, 0]], atoms=atom,
                             lattice=sisl.Lattice([10, 10, 10], nsc=[1, 1, 1]))
    target = sisl.Hamiltonian(geometry, orthogonal=False)
    for i in range(3):
        for j in range(3):
            target[i, j] = [-3 + i if i == j else 0.1, 1 if i == j else 0.05]
    geometry_file, matrix_file = tmp_path / "geometry.xyz", tmp_path / "target.TSHS"
    geometry.write(geometry_file)
    target.write(matrix_file)
    processor = MatrixDataProcessor(basis_table=AtomicTableWithEdges([atom]),
                                    out_matrix="hamiltonian", symmetric_matrix=True,
                                    sub_point_matrix=False, n_matrix_components=2,
                                    matrix_component_policy="h_and_overlap")

    def predict(data):
        result = {}
        for name, labels in (("node_labels", data.point_labels), ("edge_labels", data.edge_labels)):
            values = labels.clone()
            values[:, 0] += 0.25
            if prediction_components == "h_flat":
                values = values[:, 0]
            elif prediction_components == "h_column":
                values = values[:, :1]
            result[name] = values
        return result

    app = create_server_app({"h_model": {
        "prediction_function": predict, "data_processor": processor,
        "files": {}, "test_metrics_summary": "",
    }})
    response = TestClient(app).post("/api/models/h_model/test", files={
        "geometry_file": ("geometry.xyz", geometry_file.read_bytes()),
        "output_matrix_file": ("target.TSHS", matrix_file.read_bytes()),
    })

    assert response.status_code == 200
    # Omitted S is not part of the denominator; explicitly predicted S is.
    denominator = 2 if prediction_components == "hs" else 1
    assert response.json()["elementwise_mse"] == pytest.approx(0.25 ** 2 / denominator)
    assert response.json()["elementwise_mae"] == pytest.approx(0.25 / denominator)
