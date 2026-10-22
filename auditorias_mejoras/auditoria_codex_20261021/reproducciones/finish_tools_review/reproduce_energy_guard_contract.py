"""Exercise original launch gate with synthetic telemetry; no subprocess or GPU."""
import ast
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

source = Path('/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/source/auditorias_mejoras/run_energy_loss_experiment.py')
nodes = [n for n in ast.parse(source.read_text()).body
         if isinstance(n, ast.FunctionDef) and n.name in ('_room', 'train_all')]


class Stopped(Exception):
    pass


def sleep(_seconds):
    raise Stopped()


out = []
for case, gpu, cpu in [
    ('gpu_temperature', (81, 0, 24000), 30),
    ('cpu_temperature', (30, 0, 24000), 90),
    ('insufficient_vram', (30, 22000, 24000), 30),
]:
    starts, room_calls = [], []

    def start(root, job):
        starts.append(job)
        return SimpleNamespace(pid=123), root / 'not-created'

    autorun = SimpleNamespace(gpu=lambda: gpu, cpu_temp=lambda: cpu,
                             LIMIT_C=85, GPU_PAUSE_C=80,
                             VRAM_MARGIN_MIB=1000, MIN_FREE_VRAM_FRAC=0.2)
    namespace = dict(sys=SimpleNamespace(path=SimpleNamespace(insert=lambda *args: None)),
                     rc=SimpleNamespace(SCRIPTS=source.parent),
                     PARALLEL=3, VRAM_PER_TRAINING_MIB=7000,
                     _parallel=lambda root: 3, start=start,
                     time=SimpleNamespace(sleep=sleep))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    actual_room = namespace['_room']

    def observed_room(*args):
        room_calls.append(list(args))
        return actual_room(*args)

    namespace['_room'] = observed_room
    with patch.dict(sys.modules, {'g2m_autorun': autorun}):
        room_would_allow = actual_room(0, 3)
        assert room_would_allow is False
        with tempfile.TemporaryDirectory(prefix='guard-contract-', dir=Path(__file__).parent) as tmp:
            try:
                namespace['train_all'](Path(tmp), [dict(arm='control', seed=0)])
            except Stopped:
                pass
    assert len(starts) == 1 and room_calls == []
    out.append(dict(case=case, room_would_allow=room_would_allow,
                    room_calls=room_calls, starts=starts))
print(json.dumps(out, indent=2))
