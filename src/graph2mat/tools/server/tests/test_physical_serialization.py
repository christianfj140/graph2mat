"""Real HTTP/writer oracles: matrix type, overlap, gauge and safe outputs."""

import numpy as np
import pytest
import sisl

pytest.importorskip("fastapi")

import torch
import requests
import socket
import threading
import time
from contextlib import contextmanager
from fastapi.testclient import TestClient

from graph2mat import AtomicTableWithEdges, MatrixDataProcessor
from graph2mat.tools.server.extrapolation import MatrixTimeSeriesState, create_extrapolation_app
from graph2mat.tools.server.server_app import create_server_app
from graph2mat.tools.server.api_client import ServerClient
from graph2mat.tools.server._serialization import write_prediction


def _model(tmp_path, kind="hamiltonian", spin=False, **options):
    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=atom, lattice=[10.]*3)
    geometry_path = tmp_path / "geometry.xyz"
    geometry.write(geometry_path)
    if kind == "hamiltonian":
        nodes = [[-2., -4., 1.], [-3., -6., 1.]] if spin else [[-2., 1.], [-3., 1.]]
        edges = [[.5, 1., .25]] if spin else [[.5, .25]]
        policy = "h_and_overlap"
    else:
        nodes = [[2., 4.], [3., 6.]] if spin else [[2.], [3.]]
        edges = [[.5, 1.]] if spin else [[.5]]
        policy = "h_only"
    if options.pop("h_only", False):
        nodes, edges, policy = [[-2.], [-3.]], [[.5]], "h_only"
    if options.pop("invalid_overlap", False):
        nodes[0][-1] = 0.
    if options.pop("invalid_prediction", False):
        nodes[0][0] = float("nan")
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges([atom]), out_matrix=kind,
        symmetric_matrix=True, sub_point_matrix=False, n_matrix_components=len(nodes[0]),
        matrix_component_policy=policy,
    )

    def predict(data):
        node_values = torch.tensor(nodes, dtype=torch.float64)
        edge_values = torch.tensor(edges, dtype=torch.float64)
        if len(nodes[0]) == 1:
            node_values, edge_values = node_values[:, 0], edge_values[:, 0]
        return {"node_labels": node_values, "edge_labels": edge_values}

    model = {"data_processor": processor, "prediction_function": predict,
             "files": {}, "test_metrics_summary": "", **options}
    return geometry, geometry_path, model


def _request(client, geometry_path, output, local, **params):
    if local:
        return client.get("/api/models/m/local_write_predict", params={
            "geometry_path": str(geometry_path), "output_path": str(output), **params,
        })
    return client.post("/api/models/m/predict", params=params, files={
        "geometry_file": ("geometry.xyz", geometry_path.read_bytes()),
    })


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("spin", [False, True])
@pytest.mark.parametrize("kind", ["hamiltonian", "density_matrix", "energy_density_matrix"])
def test_http_prediction_roundtrips_real_matrix_spin_overlap_and_fermi(tmp_path, local, spin, kind):
    geometry, geometry_path, model = _model(tmp_path, kind, spin, fermi_level=.5)
    suffix = ".DM" if kind == "density_matrix" else ".nc"
    output = tmp_path / ("prediction" + suffix)
    # Standalone EDM is stored in its supplied gauge, without an invented DM.
    if kind == "energy_density_matrix":
        model.pop("fermi_level")
    client = TestClient(create_server_app({"m": model}, local=True), raise_server_exceptions=False)
    response = _request(client, geometry_path, output, local)
    assert response.status_code == 200, response.text
    if not local:
        assert suffix in response.headers["content-disposition"]
        output.write_bytes(response.content)
    with sisl.get_sile(output) as reader:
        matrix = getattr(reader, "read_" + kind)(geometry=geometry)
        if kind == "hamiltonian":
            assert reader.read_fermi_level() == pytest.approx(.5)
    assert matrix.spin.is_polarized == spin
    if kind == "hamiltonian":
        assert not matrix.orthogonal
        np.testing.assert_allclose(matrix.tocsr(matrix.S_idx).toarray(), [[1., .25], [.25, 1.]], atol=1e-12)
        # sisl's nc reader returns H - Ef*S. No extra shift by Graph2Mat.
        wanted = [[-2.5, .375], [.375, -3.5]]
        wanted_down = [[-4.5, .875], [.875, -6.5]]
    else:
        wanted, wanted_down = [[2., .5], [.5, 3.]], [[4., 1.], [1., 6.]]
    np.testing.assert_allclose(matrix.tocsr(0).toarray(), wanted, atol=1e-10)
    if spin:
        np.testing.assert_allclose(matrix.tocsr(1).toarray(), wanted_down, atol=1e-10)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("invalid", ["missing_fermi", "nan_fermi", "h_only", "invalid_overlap", "invalid_prediction"])
def test_invalid_physical_hamiltonian_is_rejected_without_overwriting(tmp_path, local, invalid):
    options = {"fermi_level": 0.}
    if invalid == "missing_fermi":
        options.pop("fermi_level")
    elif invalid == "nan_fermi":
        options["fermi_level"] = float("nan")
    else:
        options[invalid] = True
    _, geometry_path, model = _model(tmp_path, **options)
    output = tmp_path / "existing.nc"
    output.write_bytes(b"original")
    client = TestClient(create_server_app({"m": model}, local=True), raise_server_exceptions=False)
    response = _request(client, geometry_path, output, local, allow_overwrite=True if local else False)
    assert response.status_code == 422, response.text
    assert output.read_bytes() == b"original"
    assert "detail" in response.json()


@pytest.mark.parametrize("kind,suffix,fermi", [
    ("hamiltonian", ".TSHS", 1.), ("hamiltonian", ".DM", 0.),
    ("energy_density_matrix", ".EDM", 0.), ("energy_density_matrix", ".TSDE", 0.),
])
@pytest.mark.parametrize("local", [False, True])
def test_incompatible_physical_format_is_rejected_before_writing(tmp_path, kind, suffix, fermi, local):
    _, geometry_path, model = _model(tmp_path, kind, fermi_level=fermi)
    output = tmp_path / ("prediction" + suffix)
    client = TestClient(create_server_app({"m": model}, local=True), raise_server_exceptions=False)
    response = _request(client, geometry_path, output, local, **({} if local else {"output_suffix": suffix}))
    assert response.status_code == 422, response.text
    assert not output.exists()


def test_explicitly_orthogonal_h_and_zero_fermi_tshs_remain_supported(tmp_path):
    geometry, geometry_path, model = _model(tmp_path, h_only=True, orthogonal_basis=True, fermi_level=0.)
    client = TestClient(create_server_app({"m": model}, local=True))
    output = tmp_path / "prediction.TSHS"
    response = _request(client, geometry_path, output, True)
    assert response.status_code == 200, response.text
    matrix = sisl.get_sile(output).read_hamiltonian(geometry=geometry)
    np.testing.assert_allclose(matrix.tocsr(0).toarray(), [[-2., .5], [.5, -3.]], atol=1e-12)
    assert matrix.orthogonal


@pytest.mark.parametrize("kind", ["density_matrix", "hamiltonian", "energy_density_matrix"])
def test_lua_dm_init_declares_and_checks_the_matrix_type(tmp_path, kind):
    _, _, model = _model(tmp_path, kind)
    state = {}
    client = TestClient(create_extrapolation_app(time_series=state, data_processors={"m": model["data_processor"]}))
    response = client.get("/init", params={"history_len": 2, "data_processor": "m", "matrix_type": "density_matrix"})
    assert response.status_code == (200 if kind == "density_matrix" else 422), response.text
    assert (0 in state) == (kind == "density_matrix")


def test_rendered_lua_aborts_when_the_dm_contract_is_rejected(tmp_path):
    from graph2mat.tools.siesta.md import setup

    setup(ml="1", work_dir=tmp_path, inplace=True)
    content = (tmp_path / "graph2mat.lua").read_text()
    assert "matrix_type=density_matrix" in content
    assert "curl --fail" in content
    assert "success ~= true and success ~= 0" in content
    assert 'error("Graph2Mat request failed' in content
    assert 'predict_next(work_dir .. "/siesta.DM")' in content
    assert '&allow_overwrite=true' in content


def test_dm_contract_cannot_be_bypassed_by_later_processor_setup(tmp_path):
    state = {}
    client = TestClient(create_extrapolation_app(time_series=state))
    assert client.get("/init", params={"history_len": 1, "matrix_type": "density_matrix"}).status_code == 200
    response = client.get("/setup_processor", params={"basis_dir": str(tmp_path), "matrix": "hamiltonian"})
    assert response.status_code == 422
    assert state[0].processor is None


def test_standalone_extrapolation_requires_a_physical_output_contract(tmp_path):
    geometry, _, model = _model(tmp_path, fermi_level=.5)
    series = MatrixTimeSeriesState(1, model["data_processor"])
    series.add_next_geometry(geometry.copy())
    series.add_last_matrix(model["data_processor"].torch_predict(model["prediction_function"], geometry))
    series.add_next_geometry(geometry.copy())
    client = TestClient(create_extrapolation_app(time_series={0: series}), raise_server_exceptions=False)
    output = tmp_path / "extrapolated.nc"
    invalid = client.get("/extrapolate", params={"out": str(output)})
    assert invalid.status_code == 422
    assert not output.exists()
    response = client.get("/extrapolate", params={"out": str(output), "fermi_level": .5})
    assert response.status_code == 200, response.text
    matrix = sisl.get_sile(output).read_hamiltonian()
    np.testing.assert_allclose(matrix.tocsr(0).toarray(), [[-2.5, .375], [.375, -3.5]], atol=1e-10)


@pytest.mark.parametrize("endpoint,params", [
    ("setup_processor", {"matrix": "density_matrix"}),
    ("add_step", {}), ("add_matrix", {}), ("extrapolate", {"allow_overwrite": True}),
])
def test_remote_server_forbids_extrapolation_filesystem_access(tmp_path, endpoint, params):
    _, geometry_path, model = _model(tmp_path, kind="density_matrix")
    output = tmp_path / "existing.DM"
    output.write_bytes(b"preserve existing content")
    client = TestClient(create_server_app({"m": model}, local=False), raise_server_exceptions=False)
    client.get("/api/extrapolation/init", params={"history_len": 0, "data_processor": "m"})
    if endpoint == "extrapolate":
        client.get("/api/extrapolation/add_step", params={"path": str(geometry_path), "matrix_ref": "m"})
    response = client.get(f"/api/extrapolation/{endpoint}", params={
        "path": str(geometry_path), "out": str(output), "basis_dir": str(tmp_path), "matrix_ref": "m", **params,
    })
    assert response.status_code == 403
    assert output.read_bytes() == b"preserve existing content"
    assert not list(tmp_path.glob(".g2m-*"))


@pytest.mark.parametrize("allow_overwrite", [None, False, True])
def test_local_extrapolation_requires_explicit_overwrite_permission(tmp_path, allow_overwrite):
    geometry, _, model = _model(tmp_path, kind="density_matrix")
    series = MatrixTimeSeriesState(0, model["data_processor"])
    series.add_next_geometry(geometry)
    series.add_last_matrix_ref(model["data_processor"].torch_predict(model["prediction_function"], geometry))
    output = tmp_path / "existing.DM"
    output.write_bytes(b"preserve existing content")
    client = TestClient(create_extrapolation_app(time_series={0: series}))
    params = {"out": str(output)}
    if allow_overwrite is not None:
        params["allow_overwrite"] = allow_overwrite
    response = client.get("/extrapolate", params=params)
    assert response.status_code == (200 if allow_overwrite else 409)
    if allow_overwrite:
        written = sisl.get_sile(output).read_density_matrix(geometry=geometry)
        np.testing.assert_allclose(written.tocsr().toarray(), [[2., .5], [.5, 3.]])
    else:
        assert output.read_bytes() == b"preserve existing content"
    assert not list(tmp_path.glob(".g2m-*"))


@pytest.mark.parametrize("components", [1, 2])
@pytest.mark.parametrize("with_reference", [False, True])
@pytest.mark.parametrize("delayed_processor", [False, True])
def test_http_density_history_reads_multi_orbital_binary_matrices_with_series_geometry(tmp_path, components, with_reference, delayed_processor):
    radial = np.linspace(0., 1.4, 50)
    atom = sisl.Atom("H", orbitals=[
        sisl.AtomicOrbital("1s", spherical=(radial, np.exp(-radial)), R=1.4),
        sisl.AtomicOrbital("2s", spherical=(radial, (1. - radial / 2) * np.exp(-radial / 2)), R=1.4),
    ])
    geometry = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=atom, lattice=[10.] * 3)
    geometry.set_nsc([1, 1, 1])
    matrix = sisl.DensityMatrix(geometry, spin="polarized" if components == 2 else "unpolarized")
    expected = np.fromfunction(lambda i, j: 1. + i + j, (4, 4))
    for i in range(4):
        for j in range(4):
            matrix[i, j] = expected[i, j] * np.arange(1, components + 1)
    path, reference = tmp_path / "history.DM", tmp_path / "reference.DM"
    matrix.write(path)
    (matrix * .25).write(reference)
    original = {file: file.read_bytes() for file in (path, reference)}
    processor = MatrixDataProcessor(basis_table=AtomicTableWithEdges([atom]), out_matrix="density_matrix",
                                    symmetric_matrix=True, sub_point_matrix=False, n_matrix_components=components)
    geometry_path = tmp_path / "geometry.xyz"
    geometry.write(geometry_path)
    original[geometry_path] = geometry_path.read_bytes()
    states = {}
    client = TestClient(create_extrapolation_app(time_series=states, data_processors={"dm": processor}),
                        raise_server_exceptions=False)
    params = {"history_len": 2}
    if not delayed_processor:
        params["data_processor"] = "dm"
    assert client.get("/init", params=params).status_code == 200
    assert client.get("/add_step", params={"path": str(geometry_path)}).status_code == 200
    series = states[0]
    if delayed_processor:
        original_config = series.configs[-1]
        # /setup_processor delegates here after reading the basis from ion files.
        series.add_processor(processor)
        np.testing.assert_array_equal(original_config.metadata["geometry"].orbitals, [1, 1])
    stored_geometry = series.configs[-1].metadata["geometry"]
    np.testing.assert_array_equal(stored_geometry.orbitals, [2, 2])
    np.testing.assert_allclose(stored_geometry.xyz, [[0., 0., 0.], [1., 0., 0.]])
    np.testing.assert_allclose(stored_geometry.cell, np.eye(3) * 10)
    np.testing.assert_array_equal(stored_geometry.nsc, [1, 1, 1])
    assert series.configs[-1].metadata["path"] == str(geometry_path)
    params = {"path": str(path)}
    if with_reference:
        params["m_0"] = str(reference)
    response = client.get("/add_matrix", params=params)
    assert response.status_code == 200, response.text
    np.testing.assert_array_equal(series.configs[-1].matrix.basis_count, [2, 2])
    assert series.configs[-1].matrix._n_components == components
    assert client.get("/add_step", params={"path": str(geometry_path)}).status_code == 200
    params = {"path": str(path)}
    if with_reference:
        params["m_0"] = str(reference)
    assert client.get("/add_matrix", params=params).status_code == 200
    assert client.get("/add_step", params={"path": str(geometry_path)}).status_code == 200
    output = tmp_path / "predicted.DM"
    params = {"out": str(output)}
    if with_reference:
        params["m_0"] = str(reference)
    response = client.get("/extrapolate", params=params)
    assert response.status_code == 200, response.text
    written = sisl.get_sile(output).read_density_matrix(geometry=geometry)
    assert written.spin == matrix.spin
    for channel in range(components):
        np.testing.assert_allclose(written.tocsr(channel).toarray(), expected * (channel + 1))
    assert all(file.read_bytes() == value for file, value in original.items())


@contextmanager
def _http_server(app):
    import uvicorn

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(.01)
            assert server.started
            yield f"http://127.0.0.1:{port}"
        finally:
            server.should_exit = True
            thread.join(timeout=5)
            assert not thread.is_alive()


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("kind,cls", [
    ("hamiltonian", sisl.Hamiltonian), ("density_matrix", sisl.DensityMatrix),
    ("energy_density_matrix", sisl.EnergyDensityMatrix),
])
def test_real_requests_client_reads_the_matching_physical_format(tmp_path, local, kind, cls):
    geometry, _, model = _model(tmp_path, kind, spin=True, fermi_level=0.)
    with _http_server(create_server_app({"m": model}, local=True)) as url:
        matrix = ServerClient(url=url).predict(geometry, cls, "m", local=local)
    assert isinstance(matrix, cls)
    assert matrix.spin.is_polarized
    expected = [[-2., .5], [.5, -3.]] if kind == "hamiltonian" else [[2., .5], [.5, 3.]]
    np.testing.assert_allclose(matrix.tocsr(0).toarray(), expected, atol=1e-12)


@pytest.mark.parametrize("local", [False, True])
def test_requests_client_cleans_owned_files_after_rejected_output(tmp_path, monkeypatch, local):
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    geometry, _, model = _model(tmp_path)  # Missing fermi_level must be rejected.
    with _http_server(create_server_app({"m": model}, local=True)) as url:
        before = set(tmp_path.iterdir())
        with pytest.raises(requests.HTTPError):
            ServerClient(url=url).predict(geometry, sisl.Hamiltonian, "m", local=local)
        assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("depth", [0, 1])
def test_extrapolation_without_observations_is_a_client_error_not_http_500(tmp_path, depth):
    geometry, _, model = _model(tmp_path, "density_matrix")
    series = MatrixTimeSeriesState(2, model["data_processor"])
    for _ in range(depth):
        series.add_next_geometry(geometry.copy())
    client = TestClient(create_extrapolation_app(time_series={0: series}), raise_server_exceptions=False)
    output = tmp_path / "unknown.DM"
    response = client.get("/extrapolate", params={"out": str(output)})
    assert response.status_code == 422, response.text
    assert not output.exists()


@pytest.mark.parametrize("kind,cls", [
    ("hamiltonian", sisl.Hamiltonian), ("density_matrix", sisl.DensityMatrix),
    ("energy_density_matrix", sisl.EnergyDensityMatrix),
])
@pytest.mark.parametrize("spin", [False, True])
def test_delta_history_does_not_mix_overlap_into_unpredicted_components(tmp_path, kind, cls, spin):
    geometry, _, _ = _model(tmp_path, kind)
    components = 2 if spin else 1
    matrix = cls(geometry, spin="polarized" if spin else "unpolarized", orthogonal=False)
    reference = cls(geometry, spin=matrix.spin, orthogonal=True)
    for i in range(2):
        for j in range(2):
            matrix[i, j] = [9., 18.][:components] + [1. if i == j else .25]
            reference[i, j] = [4., 8.][:components]
    matrix.finalize()
    reference.finalize()
    before, before_reference = matrix._csr.data.copy(), reference._csr.data.copy()
    processor = MatrixDataProcessor(
        basis_table=AtomicTableWithEdges(geometry.atoms.atom), out_matrix=kind,
        symmetric_matrix=True, sub_point_matrix=False, n_matrix_components=components,
        matrix_component_policy="spin_h_only" if kind == "hamiltonian" and spin else "h_only",
    )
    series = MatrixTimeSeriesState(1, processor)
    series.add_next_geometry(geometry)
    series.add_last_matrix_ref(reference)
    series.add_last_matrix(matrix)
    for block in series.configs[-1].matrix.block_dict.values():
        wanted = np.array([5., 10.][:components])
        wanted = wanted.reshape(1, 1, components) if spin else wanted.reshape(1, 1)
        np.testing.assert_array_equal(block, wanted)
    np.testing.assert_array_equal(matrix._csr.data, before)
    np.testing.assert_array_equal(reference._csr.data, before_reference)


def test_extrapolated_h_only_delta_restores_explicit_reference_overlap(tmp_path):
    geometry, _, model = _model(tmp_path, h_only=True)
    reference = sisl.Hamiltonian(geometry, orthogonal=False)
    for i in range(2):
        for j in range(2):
            reference[i, j] = [-10. if i == j else 1., 1. if i == j else .25]
    reference.finalize()
    reference_path = tmp_path / "reference.TSHS"
    reference.write(reference_path)
    before = reference._csr.data.copy()
    series = MatrixTimeSeriesState(1, model["data_processor"])
    series.add_next_geometry(geometry.copy())
    correction = model["data_processor"].torch_predict(model["prediction_function"], geometry)
    series.add_last_matrix(correction)
    series.add_next_geometry(geometry.copy())
    client = TestClient(create_extrapolation_app(time_series={0: series}), raise_server_exceptions=False)
    output = tmp_path / "restored.nc"
    response = client.get("/extrapolate", params={"out": str(output), "m_0": str(reference_path), "fermi_level": 0.})
    assert response.status_code == 200, response.text
    matrix = sisl.get_sile(output).read_hamiltonian()
    np.testing.assert_allclose(matrix.tocsr(0).toarray(), [[-12., 1.5], [1.5, -13.]], atol=1e-10)
    np.testing.assert_allclose(matrix.tocsr(matrix.S_idx).toarray(), [[1., .25], [.25, 1.]], atol=1e-12)
    np.testing.assert_array_equal(reference._csr.data, before)


@pytest.mark.parametrize("spin", [False, True])
def test_h_overlap_delta_and_reference_recover_s_without_doubling_it(tmp_path, spin):
    geometry, _, model = _model(tmp_path, spin=spin)
    matrix = model["data_processor"].torch_predict(model["prediction_function"], geometry)
    reference = matrix.copy()
    for channel in range(matrix.spin.size(matrix.dtype)):
        reference._csr.data[:, channel] *= .5
    before, before_reference = matrix._csr.data.copy(), reference._csr.data.copy()
    series = MatrixTimeSeriesState(1, model["data_processor"])
    series.add_next_geometry(geometry.copy())
    series.add_last_matrix_ref(reference)
    series.add_last_matrix(matrix)
    series.add_next_geometry(geometry.copy())
    client = TestClient(create_extrapolation_app(time_series={0: series}))
    output = tmp_path / "restored.nc"
    response = client.get("/extrapolate", params={"out": str(output), "fermi_level": 0.})
    assert response.status_code == 200
    restored = sisl.get_sile(output).read_hamiltonian()
    for channel in range(matrix.dim):
        np.testing.assert_allclose(restored.tocsr(channel).toarray(), matrix.tocsr(channel).toarray(), atol=1e-10)
    np.testing.assert_array_equal(matrix._csr.data, before)
    np.testing.assert_array_equal(reference._csr.data, before_reference)


def test_atomic_writer_keeps_existing_output_and_cleans_partial_file(tmp_path, monkeypatch):
    geometry, _, model = _model(tmp_path)
    matrix = model["data_processor"].torch_predict(model["prediction_function"], geometry)
    output = tmp_path / "existing.nc"
    output.write_bytes(b"original")
    before = set(tmp_path.iterdir())

    def broken_write(self, path, **kwargs):
        path.write_bytes(b"partial")
        raise OSError("writer failed")

    monkeypatch.setattr(sisl.Hamiltonian, "write", broken_write)
    with pytest.raises(OSError, match="writer failed"):
        write_prediction(matrix, output, fermi_level=0.)
    assert set(tmp_path.iterdir()) == before
    assert output.read_bytes() == b"original"


def test_local_no_clobber_is_atomic_if_destination_appears_during_prediction(tmp_path):
    _, geometry_path, model = _model(tmp_path, "density_matrix")
    output = tmp_path / "late.DM"
    prediction = model["prediction_function"]

    def create_destination(data):
        output.write_bytes(b"created concurrently")
        return prediction(data)

    model["prediction_function"] = create_destination
    client = TestClient(create_server_app({"m": model}, local=True))
    response = _request(client, geometry_path, output, True)
    assert response.status_code == 409
    assert output.read_bytes() == b"created concurrently"
    assert not list(tmp_path.glob(".g2m-*"))
