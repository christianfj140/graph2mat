from io import BytesIO
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import sisl

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
from graph2mat.tools.server.server_app import create_server_app


class _DummyMatrix:
    def write(self, path):
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
    assert ".TSHS" in response.headers["content-disposition"]


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
    expected = tmp_path / (output_path or "ML_prediction.TSHS")
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


def test_uploaded_matrix_is_evaluated_through_real_data_processor(tmp_path):
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
    app = create_server_app({"dm_model": {
        "prediction_function": lambda data: {"node_labels": data.point_labels,
                                               "edge_labels": data.edge_labels},
        "data_processor": processor, "files": {}, "test_metrics_summary": "",
    }})

    response = TestClient(app).post("/api/models/dm_model/test", files={
        "geometry_file": ("geometry.xyz", geometry_file.read_bytes()),
        "output_matrix_file": ("target.DM", matrix_file.read_bytes()),
    })

    assert response.status_code == 200
    assert response.json()["elementwise_mse"] == 0
    assert response.json()["elementwise_mae"] == 0


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
