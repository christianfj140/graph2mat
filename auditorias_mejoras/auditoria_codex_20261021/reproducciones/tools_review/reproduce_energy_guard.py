import ast,json,tempfile
from pathlib import Path
from types import SimpleNamespace
source=Path('/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/source/auditorias_mejoras/run_energy_loss_experiment.py')
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='train_all')
class Stopped(Exception): pass
room_calls=[];starts=[]
def room(*args):room_calls.append(args);return False
def start(root,job):starts.append(job);return SimpleNamespace(pid=123),root/'fake'
def sleep(seconds):raise Stopped()
ns=dict(_parallel=lambda root:3,_room=room,start=start,time=SimpleNamespace(sleep=sleep),Path=Path)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
with tempfile.TemporaryDirectory(prefix='energy-',dir=Path(__file__).parent) as tmp:
 try:ns['train_all'](Path(tmp),[dict(arm='control',seed=0)])
 except Stopped:pass
result=dict(room_would_allow=False,room_calls=room_calls,starts=starts)
assert len(starts)==1 and room_calls==[]
print(json.dumps(result,indent=2))
