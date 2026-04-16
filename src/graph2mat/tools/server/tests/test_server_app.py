from io import BytesIO

from fastapi.testclient import TestClient

import sisl

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
