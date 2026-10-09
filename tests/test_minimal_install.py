"""Core import/NumPy regressions, outside the Torch-dependent package conftest."""

import subprocess
import sys

import pytest


def test_server_extra_declares_checkpoint_and_api_dependencies():
    from pathlib import Path
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    dependencies = {canonicalize_name(Requirement(value).name)
                    for value in project["project"]["optional-dependencies"]["server"]}
    required = {"torch", "torch-geometric", "e3nn", "pytorch-lightning", "mace-torch", "jsonargparse"}
    assert required <= dependencies, f"Missing server imports/checkpoint dependencies: {required - dependencies}"


@pytest.mark.parametrize("extra", ["server", "tools", "docs"])
def test_server_extras_reject_legacy_template_api(extra):
    from pathlib import Path
    from packaging.requirements import Requirement
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    requirement = next(Requirement(value) for value in project["project"]["optional-dependencies"][extra]
                       if Requirement(value).name == "fastapi")
    assert not requirement.specifier.contains("0.107.0")
    assert requirement.specifier.contains("0.108.0")
    assert requirement.specifier.contains("0.142.4")


@pytest.mark.parametrize("missing", ["fastapi", "starlette"])
def test_server_missing_dependency_preserves_cause(missing):
    code = r'''
import importlib.abc
import sys

missing = sys.argv[1]
class MissingFastAPI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] == "fastapi":
            raise ModuleNotFoundError("simulated missing: " + missing, name=missing)

sys.meta_path.insert(0, MissingFastAPI())
try:
    from graph2mat.tools.server.server_app import create_server_app
    create_server_app({})
except ImportError as exc:
    if missing == "fastapi":
        assert "graph2mat[server]" in str(exc)
        assert isinstance(exc.__cause__, ModuleNotFoundError)
        assert exc.__cause__.name == "fastapi"
    else:
        assert isinstance(exc, ModuleNotFoundError)
        assert exc.name == "starlette"
else:
    raise AssertionError("The server cannot start without its dependency")
'''
    result = subprocess.run([sys.executable, "-c", code, missing], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


SMOKE = r'''
import importlib.abc
import sys

mode = sys.argv[1]
blocked = {"torch_geometric"}
if mode != "torch_only":
    blocked |= {"torch", "e3nn", "mace", "pytorch_lightning"}

class MissingML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in blocked:
            name = "sympy" if mode == "broken_torch" and root == "torch" else root
            raise ModuleNotFoundError("simulated missing dependency: " + name, name=name)

sys.meta_path.insert(0, MissingML())

import numpy as np
import graph2mat as g
from graph2mat.core.data.metrics import elementwise_mse, block_type_mape
from graph2mat.core.modules._labels_resort import get_labels_resorting_array

indices = get_labels_resorting_array(
    np.array([1, 0, -1], dtype=np.int64),
    np.array([[1, 2], [1, 3]], dtype=np.int64),
    transpose_neg=True,
)
np.testing.assert_array_equal(indices, [1, 2, 3, 4, 5, 6, 0, 7, 10, 8, 11, 9, 12])

basis = g.BasisTableWithEdges([
    g.PointBasis("A", R=2, basis=[1], basis_convention="cartesian")
])
processor = g.MatrixDataProcessor(
    basis_table=basis, symmetric_matrix=True,
    sub_point_matrix=False, node_attr_getters=[],
)
config = g.BasisConfiguration(
    point_types=["A", "A"], positions=np.array([[0., 0., 0.], [1., 0., 0.]]),
    basis=basis, cell=np.eye(3)*10, pbc=(False, False, False),
    matrix=np.array([[1., .25], [.25, 1.]]),
)
data = g.BasisMatrixData.new(config, processor)
assert len(data.point_labels) == 2 and len(data.edge_labels) == 1
prediction = processor.matrix_from_data(data, predictions={
    "node_labels": data.point_labels + 1, "edge_labels": data.edge_labels + .5,
}, out_format="numpy")
np.testing.assert_array_equal(prediction, [[2., .75], [.75, 2.]])
np.testing.assert_array_equal(data.point_labels, [1., 1.])
np.testing.assert_array_equal(data.edge_labels, [.25])
loss, _ = elementwise_mse(
    nodes_pred=data.point_labels+1, nodes_ref=data.point_labels,
    edges_pred=data.edge_labels+1, edges_ref=data.edge_labels,
)
assert np.isclose(loss, 1)
import sisl
from scipy.sparse import csr_array
from graph2mat.core.data.physical import prepare_physical_data, physical_loss_terms

atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1., q0=1.)])
geometry = sisl.Geometry([[0., 0., 0.]], atoms=atom, lattice=[10., 10., 10.])
geometry.set_nsc([1, 1, 1])
scalar = csr_array([[1.]])
overlap = sisl.Overlap.fromsp(geometry, scalar)
density = sisl.DensityMatrix.fromsp(geometry, scalar, S=scalar)
atomic_table = g.AtomicTableWithEdges(geometry.atoms.atom)
physical_processor = g.MatrixDataProcessor(basis_table=atomic_table,
    out_matrix="density_matrix", sub_point_matrix=False, node_attr_getters=[])
physical_data = prepare_physical_data(physical_processor, overlap=overlap,
    density=density, kpoints=np.zeros((1, 3)), kweights=np.ones(1), requested=("charge",))
physical_loss, physical_stats = physical_loss_terms(physical_data.point_labels + .1,
    physical_data.edge_labels, batch=physical_data, basis_table=atomic_table,
    electron_weight=1., mulliken_weight=1.)
assert np.isclose(physical_loss, .02)
assert np.isclose(physical_stats["physical_delta_n_mean"], .1)
if mode == "base":
    assert "torch" not in sys.modules
loss, _ = block_type_mape(
    nodes_pred=np.array([0., 1.]), nodes_ref=np.array([0., 1.]),
    edges_pred=np.ones(1), edges_ref=np.ones(1),
)
assert loss == 0
loss, _ = elementwise_mse(
    nodes_pred=np.ones((2, 2)), nodes_ref=np.zeros((2, 2)),
    edges_pred=np.array([[1., 1.], [99., 99.]]),
    edges_ref=np.array([[0., 0.], [np.nan, np.nan]]),
)
assert np.isclose(loss, 1)
'''


@pytest.mark.parametrize("mode", ["base", "torch_only", "broken_torch"])
def test_core_optional_torch(mode):
    result = subprocess.run(
        [sys.executable, "-c", SMOKE, mode], capture_output=True, text=True,
    )
    if mode == "broken_torch":
        assert result.returncode != 0
        assert "simulated missing dependency: sympy" in result.stderr
    else:
        assert result.returncode == 0, result.stderr


CLI_SMOKE = r'''
import importlib.abc
import importlib.util
import sys

blocked = {"torch", "torch_geometric", "e3nn", "mace", "pytorch_lightning",
           "requests", "yaml", "jinja2", "fastapi", "uvicorn"}
# pandas is a transitive base dependency of sisl[viz], not a missing CLI extra.

class MissingOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in blocked:
            raise ModuleNotFoundError("simulated missing dependency: " + root, name=root)

sys.meta_path.insert(0, MissingOptional())
# Optional availability probes should see absence, just as in a clean profile.
original_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **kw: None if name.partition(".")[0] in blocked else original_find_spec(name, *a, **kw)
from typer.testing import CliRunner
from graph2mat.tools.cli.cli import app
result = CliRunner().invoke(app, [*sys.argv[1:], "--help"])
assert result.exit_code == 0, (result.output, repr(result.exception))
assert "Usage:" in result.output
assert not blocked.intersection(sys.modules)
'''


@pytest.mark.parametrize("command", [
    [], ["request"], ["request", "predict"], ["request", "avail-models"],
    ["serve"], ["utils"], ["models"], ["siesta"], ["siesta", "md"],
    ["siesta", "md", "setup"], ["siesta", "md", "setup-store"],
    ["siesta", "md", "analyze"],
])
def test_base_cli_help_does_not_import_optional_command_dependencies(command):
    result = subprocess.run([sys.executable, "-c", CLI_SMOKE, *command],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_importing_graph2mat_cli_does_not_reformat_other_typer_apps():
    code = r'''
import os
os.environ["FORCE_COLOR"] = "1"
import typer
from typer.testing import CliRunner
other = typer.Typer(rich_markup_mode="markdown")
@other.command()
def main():
    """First paragraph.

    Second paragraph with a **bold** word.
    """
runner = CliRunner()
before = runner.invoke(other, ["--help"], color=True)
assert before.exit_code == 0
import graph2mat.tools.cli.cli
after = runner.invoke(other, ["--help"], color=True)
assert after.exit_code == 0
assert after.output == before.output
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_yaml_parser_preserves_options():
    from graph2mat.tools.cli._typer import get_dict_param_kwargs

    parser = get_dict_param_kwargs((str, int))["parser"]
    options = {"steps": 2}
    assert parser(options) is options
    assert parser("{steps: 2, enabled: true}") == {"steps": 2, "enabled": True}


@pytest.mark.parametrize("missing", ["yaml", "jinja2"])
def test_base_optional_command_reports_required_extra(missing):
    code = r'''
import importlib.abc
import importlib.util
import sys
import tempfile
from pathlib import Path

missing = sys.argv[1]
class MissingOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] == missing:
            raise ModuleNotFoundError("simulated missing dependency", name=missing)
sys.meta_path.insert(0, MissingOptional())
original_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **kw: None if name.partition(".")[0] == missing else original_find_spec(name, *a, **kw)

if missing == "yaml":
    import typer
    from graph2mat.tools.cli._typer import get_dict_param_kwargs
    parser = get_dict_param_kwargs((str, int))["parser"]
    options = {"steps": 2}
    assert parser(options) is options
    try:
        parser("{steps: 2}")
    except typer.BadParameter as exc:
        assert "graph2mat[siesta]" in str(exc)
    else:
        raise AssertionError("missing YAML must produce a usage error")
else:
    from graph2mat.tools.siesta.md import setup_store
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "store.lua"
        try:
            setup_store(out=output)
        except ModuleNotFoundError as exc:
            assert exc.name == "jinja2"
            assert "graph2mat[siesta]" in str(exc)
        else:
            raise AssertionError("missing template dependency must be explicit")
        assert not output.exists()
'''
    result = subprocess.run([sys.executable, "-c", code, missing],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_setup_store_renders_with_siesta_extra(tmp_path):
    from typer.testing import CliRunner
    from graph2mat.tools.cli.cli import app

    output = tmp_path / "store.lua"
    result = CliRunner().invoke(app, [
        "siesta", "md", "setup-store", "--dir", "frames", "--interval", "3",
        "--step-prefix", "sample_", "--files", "*XV *DM", "--out", str(output),
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    script = output.read_text()
    assert 'local store_dir = "frames"' in script
    assert "local store_interval = 3" in script
    assert 'local store_step_prefix = "sample_"' in script
    assert 'local store_files = "*XV *DM"' in script
    assert "function store_step(istep)" in script
    assert "server_address" not in script


def test_http_client_does_not_import_server_or_ml_dependencies():
    code = r'''
import importlib.abc
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

blocked = {"torch", "torch_geometric", "e3nn", "mace", "pytorch_lightning",
           "fastapi", "uvicorn", "yaml", "jinja2"}
class MissingServer(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.partition(".")[0]
        if root in blocked:
            raise ModuleNotFoundError("simulated missing server dependency", name=root)
sys.meta_path.insert(0, MissingServer())
original_find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **kw: None if name.partition(".")[0] in blocked else original_find_spec(name, *a, **kw)
from graph2mat.tools.server import ServerClient

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        assert self.path == "/api/avail_models"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(["density"]).encode())
    def log_message(self, *args):
        pass

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
try:
    client = ServerClient(url=f"http://127.0.0.1:{server.server_port}")
    assert client.avail_models() == ["density"]
finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
assert not thread.is_alive()
assert not blocked.intersection(sys.modules)
assert "graph2mat.tools.server.server_app" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
