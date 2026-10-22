import warnings, traceback
from pathlib import Path
import numpy as np
import sisl
from graph2mat import AtomicTableWithEdges, BasisTableWithEdges, BasisMatrixData, MatrixDataProcessor, OrbitalConfiguration, PointBasis
from graph2mat.bindings.torch import TorchBasisMatrixDataset, TorchBasisMatrixData
import graph2mat.core.data._sparse as cy_sparse
import graph2mat.core.modules._labels_resort as cy_labels
print('EXTENSIONS', cy_sparse.__file__, cy_labels.__file__)

def run(name, f):
    print('\nPROBE', name)
    try: f()
    except Exception: traceback.print_exc()

def complex_single():
    atom=sisl.Atom('H', orbitals=[sisl.AtomicOrbital('1s',R=.8)])
    g=sisl.Geometry([[0.,0.,0.],[1.,0.,0.]],atoms=atom,lattice=[10.,10.,10.])
    h=sisl.Hamiltonian(g,dtype=np.complex128)
    h[0,0],h[1,1],h[0,1],h[1,0]=1.,2.,.25+.5j,.25-.5j
    p=MatrixDataProcessor(AtomicTableWithEdges([atom]),sub_point_matrix=False,out_matrix='hamiltonian')
    print('reference',h.tocsr(0).toarray())
    cfg=OrbitalConfiguration.from_matrix(h)
    d=BasisMatrixData.from_config(cfg,p)
    print('labels',d.point_labels,d.edge_labels)
    print('roundtrip',d.convert_to('sisl_H').tocsr(0).toarray())

def nsc_override():
    atom=sisl.Atom('H', orbitals=[sisl.AtomicOrbital('1s',R=1.1)])
    g=sisl.Geometry([[0.,0.,0.]],atoms=atom,lattice=sisl.Lattice([2.,10.,10.],nsc=[3,1,1]))
    h=sisl.Hamiltonian(g)
    h[0,0]=10.
    for shift,value in [(-1,3.),(1,7.)]:
        h[0,g.sc_index([shift,0,0])]=value
    p=MatrixDataProcessor(AtomicTableWithEdges([atom]),sub_point_matrix=False,out_matrix='hamiltonian')
    c=OrbitalConfiguration.from_matrix(h)
    for nsc in [None,[5,1,1]]:
        d=BasisMatrixData.from_config(c,p,nsc=nsc)
        print('override',nsc,'isc',d.neigh_isc,'shifts',d.shifts,'labels',d.edge_labels)

def same_size_basis():
    three_s=sisl.Atom('C',orbitals=[sisl.AtomicOrbital(n=n,l=0,m=0,R=.8) for n in [2,3,4]],tag='C_3s')
    one_p=sisl.Atom('C',orbitals=[sisl.AtomicOrbital(n=2,l=1,m=m,R=.8) for m in [-1,0,1]],tag='C_1p')
    g=sisl.Geometry([[0.,0.,0.],[1.,0.,0.]],atoms=[three_s,one_p],lattice=[10.,10.,10.])
    h=sisl.Hamiltonian(g)
    for i in range(6): h[i,i]=i+1.
    p=MatrixDataProcessor(BasisTableWithEdges([PointBasis.from_sisl_atom(three_s)]),sub_point_matrix=False,out_matrix='hamiltonian')
    d=BasisMatrixData.from_config(OrbitalConfiguration.from_matrix(h),p)
    print('actual bases',[PointBasis.from_sisl_atom(a).basis for a in g.atoms.atom])
    print('interpreted type ids',d.point_types,'table basis',p.basis_table.basis[0].basis,'accepted labels',d.point_labels)
    print('roundtrip atom bases',[PointBasis.from_sisl_atom(a).basis for a in d.convert_to('sisl_H').atoms.atom])

def labels_off():
    atom=sisl.Atom('H',orbitals=[sisl.AtomicOrbital('1s',R=.8)])
    g=sisl.Geometry([[0.,0.,0.]],atoms=atom,lattice=[10.,10.,10.])
    h=sisl.Hamiltonian(g); h[0,0]=42.
    p=MatrixDataProcessor(AtomicTableWithEdges([atom]),sub_point_matrix=False,out_matrix='hamiltonian')
    cfg=OrbitalConfiguration.from_matrix(h)
    for x in [cfg,TorchBasisMatrixData.new(cfg,p)]:
        ds=TorchBasisMatrixDataset([x],p,load_labels=False)
        got=ds[0]
        print(type(x).__name__,'load_labels=False labels',got.point_labels, 'alias',got is x)

def included_charge():
    directory=Path('/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/data/include_case');directory.mkdir(exist_ok=True)
    (directory/'charge.fdf').write_text('NetCharge 2.0\n')
    f=directory/'RUN.fdf'
    f.write_text('''SystemLabel test
NumberOfAtoms 1
NumberOfSpecies 1
LatticeConstant 1.0 Ang
%block LatticeVectors
10 0 0
0 10 0
0 0 10
%endblock LatticeVectors
%block ChemicalSpeciesLabel
1 1 H
%endblock ChemicalSpeciesLabel
AtomicCoordinatesFormat Ang
%block AtomicCoordinatesAndAtomicSpecies
0 0 0 1
%endblock AtomicCoordinatesAndAtomicSpecies
%include charge.fdf
''')
    atom=sisl.Atom('H',orbitals=[sisl.AtomicOrbital('1s',R=.8)])
    p=MatrixDataProcessor(AtomicTableWithEdges([atom]),sub_point_matrix=False)
    print('sisl NetCharge',sisl.get_sile(f).get('NetCharge'))
    ds=TorchBasisMatrixDataset([f],p,load_labels=False)
    print('dataset total_charge',ds[0].total_charge)

for name, f in [('complex_single',complex_single),('nsc_override',nsc_override),('same_size_basis',same_size_basis),('labels_off',labels_off),('included_charge',included_charge)]:run(name,f)
