import traceback
import numpy as np
import sisl
from graph2mat import AtomicTableWithEdges, MatrixDataProcessor, OrbitalConfiguration, BasisMatrixData
for norb in (1,2,4):
    orbs=[sisl.AtomicOrbital(n=2,l=0,m=0,R=.8,q0=2.)]
    if norb==2:orbs += [sisl.AtomicOrbital(n=3,l=0,m=0,R=.8,q0=1.)]
    if norb==4:orbs += [sisl.AtomicOrbital(n=2,l=1,m=m,R=.8,q0=1.) for m in [-1,0,1]]
    atom=sisl.Atom('C',orbitals=orbs)
    g=sisl.Geometry([[0.,0.,0.]],atoms=atom,lattice=[10.,10.,10.])
    dm=sisl.DensityMatrix(g,spin='polarized')
    for i in range(norb):dm[i,i,:]=[.4+i,.6+i]
    processor=MatrixDataProcessor(AtomicTableWithEdges([atom]),out_matrix='density_matrix',n_matrix_components=2)
    print('\nnorb',norb,'baseline',processor.basis_table.point_matrix)
    try:
        d=BasisMatrixData.new(dm,processor)
        print('labels',d.point_labels)
        roundtrip=d.convert_to('sisl_DM')
        print('ref',np.stack([dm.tocsr(i).toarray() for i in range(2)],-1))
        print('roundtrip',np.stack([roundtrip.tocsr(i).toarray() for i in range(2)],-1))
    except Exception:traceback.print_exc()
