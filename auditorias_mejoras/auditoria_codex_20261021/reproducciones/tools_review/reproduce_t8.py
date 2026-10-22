from pathlib import Path
import json,tempfile
import numpy as np
import sisl
from graph2mat.tools.analysis.breakdown import min_image_distances, structure_breakdown
from graph2mat.tools.conftest import write_fdf

atom=sisl.Atom(6,orbitals=[sisl.AtomicOrbital('2s', R=2.,q0=1)])
g=sisl.Geometry([[0.,0.,0.],[31.,0.,0.]],atoms=[atom,atom],lattice=sisl.Lattice([10.,10.,10.],nsc=[1,1,1]))
g.lattice.pbc=[True,False,False]
wrapped=g.copy();wrapped.xyz[1,0]=1.
actual=float(min_image_distances(g)[0,1]);expected=float(np.linalg.norm(g.axyz(1,isc=[-3,0,0])-g.xyz[0]))
out={'unwrapped_distance_Ang':actual,'oracle_sisl_image_minus3_Ang':expected,'wrapped_distance_Ang':float(min_image_distances(wrapped)[0,1])}
assert actual==21. and expected==1.
with tempfile.TemporaryDirectory(prefix='t8-',dir=Path(__file__).parent) as tmp:
 p=Path(tmp);ref=sisl.Hamiltonian(g,orthogonal=False);ref[0,0]=(1.,1.);ref[1,1]=(2.,1.);ref[0,1]=(.2,0.);ref[1,0]=(.2,0.)
 pred=ref.copy();pred[0,1]=(.3,0.);pred[1,0]=(.3,0.);ref.write(str(p/'ref.TSHS'));pred.write(str(p/'pred.TSHS'));fdf=write_fdf(p/'fdf',g)
 row=structure_breakdown(p/'pred.TSHS',p/'ref.TSHS',fdf,[atom]);out['wrong_classification']={key:row[key] for key in ['total_sum','beyond_range_sum','beyond_range_n','shell_6.0-99.0_sum','shell_0.1-1.6_sum']}
 assert row['beyond_range_sum']>0.19 and row['shell_0.1-1.6_sum']==0.
print(json.dumps(out,indent=2))
