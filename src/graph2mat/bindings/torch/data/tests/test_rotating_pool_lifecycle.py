"""Owned pool resources must stop; seeded initial sampling must repeat."""

import multiprocessing

import numpy as np
import pytest
import torch

from graph2mat import BasisConfiguration, BasisTableWithEdges, PointBasis
from graph2mat.bindings.torch import RotatingPoolData
from graph2mat.tools.lightning import MatrixDataModule


class _TensorDataset:
    def __init__(self):
        self.read_indices = []

    def __len__(self):
        return 100

    def __getitem__(self, index):
        self.read_indices.append(index)
        return torch.tensor([index], dtype=torch.long)


def _exercise_pool(connection, operation):
    pools = []
    try:
        if operation == "close":
            pool = RotatingPoolData(_TensorDataset(), 3)
            pools.append(pool)
            sample = pool[0]
            pool.close()
            pool.close()
            connection.send({
                "sample_shape": tuple(sample.shape), "sample_dtype": str(sample.dtype),
                "reader_alive": pool.loader_process.is_alive(),
                "thread_alive": pool.transfer_thread.is_alive(),
                "manager_alive": pool.manager._process.is_alive(),
            })
        elif operation == "seed":
            indices = []
            for _ in range(2):
                np.random.seed(19)
                data = _TensorDataset()
                pool = RotatingPoolData(data, 32)
                pools.append(pool)
                indices.append(data.read_indices.copy())
            connection.send({"indices": indices})
        elif operation == "datamodule":
            basis = BasisTableWithEdges([PointBasis("A", R=2, basis=[1])])
            config = BasisConfiguration(
                point_types=["A", "A"], positions=np.array([[0., 0., 0.], [1., 0., 0.]]),
                basis=basis, cell=np.eye(3)*10, pbc=(False, False, False),
                matrix=np.ones((2, 2)),
            )
            module = MatrixDataModule(
                basis_table=basis, symmetric_matrix=True, sub_point_matrix=False,
                train_runs=[config, config], val_runs=[config],
                rotating_pool_size=2, loader_threads=0,
            )
            module.setup("validate")
            assert module.train_dataset is None
            module.setup("fit")
            original = module.train_dataset
            pools.append(original)
            module.setup("fit")
            reused = module.train_dataset is original
            module.teardown("fit")
            stopped = not any((original.loader_process.is_alive(),
                               original.transfer_thread.is_alive(),
                               original.manager._process.is_alive()))
            restored = module.train_dataset is original.parent_data
            module.setup("validate")
            validation_did_not_restart = module.train_dataset is original.parent_data
            module.setup("fit")
            replacement = module.train_dataset
            if replacement is not original:
                pools.append(replacement)
            module.teardown(None)
            connection.send({
                "reused": reused, "stopped": stopped, "restored": restored,
                "validation_did_not_restart": validation_did_not_restart,
                "restarted": replacement is not original,
                "final_stopped": not any((replacement.loader_process.is_alive(),
                                          replacement.transfer_thread.is_alive(),
                                          replacement.manager._process.is_alive())),
            })
        else:
            try:
                pool = RotatingPoolData([] if operation == "empty" else _TensorDataset(),
                                        3 if operation == "empty" else 0)
            except ValueError:
                connection.send({"rejected": True})
            else:
                pools.append(pool)
                connection.send({"rejected": False})
    except Exception as error:
        connection.send({"error": repr(error)})
    finally:
        for pool in pools:
            if hasattr(pool, "close"):
                pool.close()
            else:
                # The old implementation has no way to stop its thread. Isolate this
                # RED check in a subprocess so its daemon thread cannot leak to pytest.
                pool.loader_process.terminate()
                pool.loader_process.join(5)
                pool.manager.shutdown()
        connection.close()


@pytest.mark.parametrize("operation", ["close", "seed", "empty", "zero", "datamodule"])
def test_rotating_pool_lifecycle_and_seed(operation):
    context = multiprocessing.get_context()
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_exercise_pool, args=(sender, operation))
    process.start()
    sender.close()
    try:
        assert receiver.poll(15), "Pool lifecycle child did not respond"
        result = receiver.recv()
        process.join(10)
        assert process.exitcode == 0
        assert "error" not in result, result
        if operation == "close":
            assert result["sample_shape"] == (1,)
            assert result["sample_dtype"] == "torch.int64"
            assert not result["reader_alive"]
            assert not result["thread_alive"]
            assert not result["manager_alive"]
        elif operation == "seed":
            assert len(result["indices"][0]) == 32
            assert result["indices"][0] == result["indices"][1]
        elif operation == "datamodule":
            assert all(result.values()), result
        else:
            assert result["rejected"]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        receiver.close()
