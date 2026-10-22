"""Options of LitMACEMatrixModel without other tests (roadmap 0.8 guards, 6.9 cueq)."""
import pytest
import pytorch_lightning as pl
import sisl

from graph2mat import AtomicTableWithEdges
from graph2mat.tools.lightning import MatrixDataModule
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

ATOM = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=1.4)])
BASIS = AtomicTableWithEdges([ATOM])
SMALL = dict(basis_table=BASIS, symmetric_matrix=True, hidden_irreps="4x0e+4x1o",
             edge_hidden_irreps="2x0e+2x1o", num_bessel=2, max_ell=1)


@pytest.mark.parametrize("bad, match", [({"version": "old"}, "version='new'"),
                                        ({"initial_node_feats": "OneHotZ Extra"}, "initial_node_feats")])
def test_unsupported_options_are_rejected(bad, match):
    """Roadmap 0.8: these options are accepted by the signature but not implemented."""
    with pytest.raises(ValueError, match=match):
        LitMACEMatrixModel(**SMALL, **bad)


def test_cueq_model_trains_and_uses_cuequivariance(tmp_path):
    """Roadmap 6.9: cueq=True builds MACE with cuequivariance operations and trains."""
    from mace.modules.wrapper_ops import CUET_AVAILABLE

    if not CUET_AVAILABLE:
        pytest.skip("cuequivariance-torch is not installed")
    g = sisl.Geometry([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], atoms=ATOM, lattice=[10.0] * 3)
    m = sisl.DensityMatrix(g)
    m[0, 0], m[1, 1], m[0, 1], m[1, 0] = 1.0, 2.0, 0.25, 0.25
    m.write(tmp_path / "s.nc")
    pl.seed_everything(0)
    data = MatrixDataModule(out_matrix="density_matrix", root_dir=str(tmp_path), basis_table=BASIS,
                            symmetric_matrix=True, sub_point_matrix=False, loader_threads=0,
                            train_runs="s.nc", val_runs="s.nc")
    model = LitMACEMatrixModel(**SMALL, cueq=True)
    assert any("cuequivariance" in type(module).__module__ for module in model.modules())
    trainer = pl.Trainer(accelerator="cpu", fast_dev_run=True, logger=False, enable_checkpointing=False,
                         enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, datamodule=data)
    assert all(p.isfinite().all() for p in model.parameters())
