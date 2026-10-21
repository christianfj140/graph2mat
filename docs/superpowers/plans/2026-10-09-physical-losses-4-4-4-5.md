# Physical Losses 4.4–4.5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrar y verificar en CPU pérdidas opcionales de número de electrones, población Mulliken y consistencia de perturbaciones energéticas, usando referencias físicas reales y sin cambiar los defaults.

**Architecture:** Ampliar `elementwise_mse` y `gauge_invariant_mse`, no crear otro framework de pérdidas. Un módulo NumPy prepara y valida referencias sparse; las contracciones diferenciables usan el orden de etiquetas y el layout existente. Dataset/collate transportan auxiliares exclusivos de la pérdida; Lightning valida todas las etapas y agrega cada término con su denominador propio.

**Tech Stack:** Python >=3.9, NumPy, SciPy/sisl, Torch/PyG, Lightning y los gates Ruff/mypy existentes; ninguna dependencia nueva.

**Spec:** [Diseño aprobado](../specs/2026-10-09-physical-losses-4-4-4-5-design.md), aprobación explícita del usuario el 2026-10-09. Evidencia: [inventario](../../../auditorias_mejoras/inventario_objetivos_fisicos.md). Plan aprobado explícitamente; método Native elegido por el usuario (2026-10-09).

## Global Constraints

- Repositorio `<graph2mat>`, rama `g2m-v2`, mismo checkout; sin worktree, push ni merge. Preservar cambios ajenos y no usar staging global.
- `electron_weight=0.0`, `mulliken_weight=0.0`, `energy_weight=0.0`; pesos finitos y no negativos. `physical_labels=None` conserva la ruta actual, sin lecturas auxiliares.
- Charge necesita DM y S reales; energy necesita H/S/DM/EDM reales. No inventar DM/EDM, ocupaciones ni etiquetas cero a partir de H.
- Matrices reales no polarizadas o colineales; sin factor de spin adicional. Rechazar SOC, no colinealidad y coeficientes complejos.
- DM residual no polarizada restaura D0 antes del observable; rechazar DM colineal con `sub_point_matrix=True`.
- Referencia energética: residuo máximo <=1e−6 eV por átomo. S(k), float64 en la malla declarada: rechazar lambda_min<=0 o lambda_min/lambda_max<=1e−12. Sin regularización ni cambio implícito de gauge.
- Carga compara la predicción realmente representada con la población de referencia completa. Energía conserva la referencia fuera de G: no es la energía total exacta del writer.
- Medias físicas por estructura; contracciones periódicas por átomo, nunca producto de matrices plegadas a Γ. Media arista aporta a ambos extremos, también a imágenes del mismo átomo.
- Preparación CPU: diagonalización de S permitida; paso de pérdida: productos/scatter, sin reconstrucción orbital densa ni diagonalización.
- Campos `physical_*` sin `index`; referencias no entran en features/readout. Importación NumPy sin Torch y checkpoints legacy preservados.
- Datos externos autorizados solo en lectura: `$MDVS_ROOT/Comparison/datasets`. No escribir allí ni importar scripts de ese repositorio.
- Sin fonones, 7.4, autorun, colas, GPU, AMP CUDA, entrenamientos comparativos largos, instalaciones ni cambios de entornos compartidos. No detener procesos ajenos.
- No atribuir mejoras de precisión, velocidad o VRAM a una implementación correcta. Los auxiliares pueden aumentar RAM/transferencia/VRAM.

## Review Focus

1. Entrada FDF que declara otra malla o carece de ella: no validar silenciosamente solo Γ; prueba `test_fdf_requires_supported_explicit_grid` en Task 2.
2. Componente DM ausente frente a un cero almacenado explícitamente: no inferir observación por valor numérico; prueba `test_unknown_partner_is_not_an_observed_zero` en Task 2.
3. Checkpoint con etapa futura que activa física mientras la etapa inicial la apaga: fallar antes de actualizar pesos; prueba `test_future_physics_fails_before_optimizer_step` en Task 4.
4. Batch no simétrico o periódico con autoimagen: no perder layout ni duplicar incorrectamente contribuciones; prueba `test_half_edge_and_directed_self_image_agree` en Task 3.
5. Época particionada en batches con distintos números de átomos/elementos: agregar SSE/count correctos, no medias de batches; prueba `test_physical_epoch_is_partition_invariant` en Task 4.

---

## Mapa de archivos y contratos

Todos los paths siguientes son relativos a Graph2Mat. No modificar MACE ni readouts.

| Archivo | Responsabilidad |
|---|---|
| `src/graph2mat/core/data/physical.py` (nuevo) | Preparación/validación y contracciones NumPy/Torch opcionales, con importación Torch diferida. |
| `src/graph2mat/core/data/matrices/__init__.py` | Registro EDM, sin baseline inventada. |
| `src/graph2mat/core/data/metrics.py` | Tres kwargs y suma de términos; reutilizar la alineación gauge existente. |
| `src/graph2mat/bindings/torch/data/dataset.py` | `physical_labels`, carga FDF opt-in y conservación del camino legacy. |
| `src/graph2mat/tools/lightning/data.py` | Propagar opción y preparar ownership CPU incluso para matrices no simétricas. |
| `src/graph2mat/tools/lightning/model.py` | Preflight, etapas/checkpoints, fallback matricial y denominadores de época. |
| `src/graph2mat/core/data/tests/physical_fixtures.py` (nuevo) | Referencias analíticas independientes reutilizadas solo por tests. |
| `src/graph2mat/core/data/tests/test_physical_references.py` (nuevo) | Validez de referencias, convenciones y oráculos completos. |
| `src/graph2mat/core/data/tests/test_physical_losses.py` (nuevo) | Contracciones, valores literales, invariancias y gradientes. |
| `src/graph2mat/bindings/torch/data/tests/test_physical_dataset.py` (nuevo) | Preparación, componentes, máscaras, FDF, batching y caché. |
| `src/graph2mat/tools/lightning/tests/test_physical_training.py` (nuevo) | Modelo real, etapas, checkpoints, epochs y DDP CPU. |
| `auditorias_mejoras/smoke_objetivos_fisicos.py` (nuevo) | Smoke real reproducible, sin escritura de fuentes externas. |
| `auditorias_mejoras/bench_objetivos_fisicos.py` (nuevo) | Costes CPU pareados y separados por fase. |
| `auditorias_mejoras/informe_objetivos_fisicos_cpu.md` (nuevo) | Evidencia real, commits, comandos, límites y hallazgos. |
| `auditorias_mejoras/protocolo_precision_4_4_4_5.md` (nuevo) | Evaluación posterior, sin lanzarla. |
| `auditorias_mejoras/resultados_objetivos_fisicos_cpu.json` (nuevo) | Resultados generados exclusivamente por el benchmark propio. |

Modificar `processing.py` únicamente si una prueba demuestra pérdida de auxiliares en una conversión pública necesaria; no cambiar su selección legacy de componentes. Los tests existentes se amplían solo cuando poseen el comportamiento protegido.

### Interfaces compartidas fijadas

En `physical.py`, con anotaciones compatibles con Python 3.9 (`Optional`, `Sequence`, `Tuple`, `Dict`, `Any`):

```python
normalize_physical_labels(labels: Optional[Sequence[str]]) -> Tuple[str, ...]
validate_physical_weights(electron_weight: float = 0.0,
                          mulliken_weight: float = 0.0,
                          energy_weight: float = 0.0) -> Tuple[float, float, float]
physical_reference_targets(overlap: sisl.Overlap, density: sisl.DensityMatrix, *,
                           hamiltonian: Optional[sisl.Hamiltonian] = None,
                           energy_density: Optional[sisl.EnergyDensityMatrix] = None,
                           kpoints: np.ndarray, kweights: np.ndarray,
                           requested: Sequence[str]) -> Dict[str, np.ndarray]
prepare_physical_data(processor: MatrixDataProcessor, *,
                      overlap: sisl.Overlap, density: sisl.DensityMatrix,
                      hamiltonian: Optional[sisl.Hamiltonian] = None,
                      energy_density: Optional[sisl.EnergyDensityMatrix] = None,
                      kpoints: np.ndarray, kweights: np.ndarray,
                      requested: Sequence[str],
                      data_cls: Type[BasisMatrixDataBase] = BasisMatrixData
                      ) -> BasisMatrixDataBase
load_physical_data(path: Union[str, Path], processor: MatrixDataProcessor, *,
                   requested: Sequence[str],
                   data_cls: Type[BasisMatrixDataBase] = BasisMatrixData
                   ) -> BasisMatrixDataBase
physical_layout(batch: Any, basis_table: BasisTableWithEdges) -> Dict[str, Any]
physical_contractions(nodes: Any, edges: Any, partner_nodes: Any,
                      partner_edges: Any, *, layout: Dict[str, Any],
                      reverse_partner_edges: Optional[Any] = None) -> Any
physical_loss_terms(nodes_pred: Any, edges_pred: Any, *, batch: Any,
                    basis_table: BasisTableWithEdges,
                    electron_weight: float = 0.0, mulliken_weight: float = 0.0,
                    energy_weight: float = 0.0,
                    aligned_nodes: Optional[Any] = None,
                    aligned_edges: Optional[Any] = None
                    ) -> Tuple[Any, Dict[str, Any]]
```

`Any` para arrays duales no permite omitir comprobaciones de forma/dtype. Los
helpers privados nuevos se mantienen en ese módulo; no se exporta una jerarquía
de objetos ni se añade un registro de pérdidas. El entrypoint directo público es
`prepare_physical_data`, importable desde `graph2mat.core.data.physical`.

### Contrato de datos preparado

`physical_reference_targets` devuelve `population` `(n_atoms,)` cuando charge
está solicitado, `energy_residual` `(n_atoms,)` cuando energy está solicitado,
`spin_components` `(1,)`, y `capabilities` `(1,)` como arrays. Charge=1,
energy=2, ambos=3. Referencias completas, no limitadas a G.

Auxiliares de cada muestra, concatenables por PyG:

| Campo | Forma y significado |
|---|---|
| `physical_version`, `physical_kind`, `physical_spin`, `physical_capabilities` | `(1,)`, enteros; versión 1, H=0/DM=1/EDM=2, spin 1/2, bits anteriores. |
| `physical_reference_nodes/edges` | `(n_label_rows, n_spin)`, target físico de referencia sin S y sin resta D0. |
| `physical_overlap_nodes/edges` | `(n_label_rows, 1)`, S de referencia. |
| `physical_overlap_reverse_edges` | `(n_edge_rows, 1)`, partner de la otra orientación de media arista. |
| `physical_partner_nodes/edges` | Solo energy: H→D, DM→H, EDM→S; canales físicos o S escalar. |
| `physical_partner_reverse_edges`, `physical_partner_known_reverse_edges` | Solo energy: partner y máscara de la otra orientación, misma forma de edge. |
| `physical_reference_known_nodes/edges`, `physical_partner_known_nodes/edges` | Máscaras bool con la misma forma que sus arrays, no inferidas de `value!=0`. |
| `physical_population_reference` | Solo charge: `(n_atoms,)`, población completa por átomo. |
| `physical_energy_reference_residual` | Solo energy: `(n_atoms,)`, H–D menos E–S completo. |
| `physical_baseline_nodes` | DM: `(n_node_rows, n_spin)`, D0 no polarizada o ceros si la baseline está apagada. |

No almacenar arrays auxiliares de H/EDM en el modo charge. El layout vive fuera
del store PyG, bajo `LAYOUT_ATTR`, reutilizando `layout_to`:
`physical_node_owner`, `physical_edge_source`, `physical_edge_target`,
`physical_atom_structure`, `physical_atoms_per_structure` y
`physical_symmetric`. Owners son enteros y se calculan en CPU. Las claves son
planas y la transferencia reutiliza `layout_to` sin cambios.

## Preflight de ejecución, antes del primer RED

- [x] Releer spec/instrucciones, comprobar rama/status y registrar HEAD. No ejecutar producto hasta que el usuario haya revisado este plan y elegido método.

```bash
git status --short --branch
git log -3 --oneline
```

- [x] Fijar entornos existentes y confirmar que importan este checkout. Si desaparecieron, informar el recurso; no instalar paquetes ni sustituir silenciosamente versiones.

```bash
export G2M_PHYS_PY=/tmp/g2m-python312-full.LHyni1/venv/bin/python
export G2M_PHYS_PY39=/tmp/g2m-python39.D7TJAp/full-venv/bin/python
export G2M_PHYS_RUFF=/tmp/g2m-static-gates.YpKMAH/venv/bin/ruff
export G2M_PHYS_MYPY=/tmp/g2m-static-gates.YpKMAH/venv/bin/mypy
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$G2M_PHYS_PY" -c 'import sys,graph2mat,sisl,torch,pytorch_lightning; print(sys.version,graph2mat.__file__,sisl.__version__,torch.__version__,pytorch_lightning.__version__); assert graph2mat.__file__.startswith("src/"); assert not torch.cuda.is_available()'
```

- [x] Registrar baseline enfocada, salida completa y exit code; ningún test nuevo debe contar como RED por un fallo previo de entorno.

```bash
"$G2M_PHYS_PY" -m pytest src/graph2mat/core/data/tests/test_periodic_hermiticity_oracles.py src/graph2mat/tools/lightning/tests/test_gauge_epoch_metrics.py src/graph2mat/tools/lightning/tests/test_training_stages.py src/graph2mat/tools/lightning/tests/test_datamodule_stages.py src/graph2mat/tools/lightning/tests/test_coefficient_matrix_training.py -q
```

La suite histórica 2251/2 no es evidencia nueva de este bloque.

### Verificación preparatoria ejecutada (2026-10-09)

- Estado comprobado: HEAD `1e09715`, rama `g2m-v2`, producto sin cambios;
  untracked ajenos conservados. Estado preparatorio histórico; plan/método
  aprobados después explícitamente por el usuario.
- Python 3.12.3, sisl 0.16.4, Torch 2.11.0+cpu, Lightning 2.6.1; importación
  desde `src/graph2mat/__init__.py`,
  CUDA no disponible con la variable de entorno vacía.
- Comando de baseline de cinco módulos arriba, con PYTHONPATH absoluto y
  OMP/OPENBLAS/MKL=2: **138 passed, 4992 warnings, 70.74 s, exit code 0**.
  Sesión de ejecución propia `7149`, terminada; no hay un test activo que esperar.
  Los warnings incluyen deprecaciones de dependencias, avisos de fixtures y
  configuración de Lightning; no se ocultaron ni se consideran fallos nuevos.
- Python 3.9.25/Torch 2.4.1+cpu: importación de este checkout y ausencia de CUDA
  confirmadas, exit code 0. Esto **no** es todavía el gate de tests Python 3.9.
- No se implementó producto, no se escribió ninguna referencia externa y no
  se validaron las pérdidas nuevas. Este resultado prueba la línea base
  existente, no el cierre 4.4–4.5 ni una mejora de precisión/rendimiento.

### Task 1: Referencias físicas completas y oráculos independientes

**Files:** Create `core/data/physical.py`, `core/data/tests/physical_fixtures.py`, `core/data/tests/test_physical_references.py`; modify `core/data/matrices/__init__.py`. Todos bajo `src/graph2mat/`.

**Interfaces:** Consumes `sisl.*.tocsr(component)`, `Overlap.Sk`, `geometry.o2a`; produces `normalize_physical_labels`, `validate_physical_weights`, `physical_reference_targets` y fixture `analytic_matrices(spin=1, fractional=False) -> dict`.

- [x] **1.1 RED: crear fixture analítica, sin usar implementación nueva para el esperado.**

```python
def analytic_matrices(spin=1, fractional=False):
    import numpy as np
    import sisl
    from scipy.sparse import csr_array

    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2.8, q0=1.)])
    g = sisl.Geometry([[0., 0., 0.], [1., 0., 0.]], atoms=[atom, atom], lattice=[20., 20., 20.])
    g.set_nsc([1, 1, 1])
    s = csr_array([[1., .2], [.2, 1.]])
    h = csr_array([[.5, -.3], [-.3, .5]])
    scale = .6 if fractional else 1.
    d = csr_array(np.full((2, 2), scale * 5 / 6))
    e = csr_array(np.full((2, 2), scale * 5 / 36))
    polarized = spin == 2
    parts = (.5, .5) if not fractional else (.7, .3)
    spin_name = "polarized" if polarized else "unpolarized"
    return dict(
        overlap=sisl.Overlap.fromsp(g, s),
        density=sisl.DensityMatrix.fromsp(g, [d * p for p in parts] if polarized else d, S=s, spin=spin_name),
        hamiltonian=sisl.Hamiltonian.fromsp(g, [h, h] if polarized else h, S=s, spin=spin_name),
        energy_density=sisl.EnergyDensityMatrix.fromsp(g, [e * p for p in parts] if polarized else e, S=s, spin=spin_name),
        kpoints=np.zeros((1, 3)), kweights=np.ones(1),
    )
```

```python
@pytest.mark.parametrize("spin", [1, 2])
@pytest.mark.parametrize("fractional", [False, True])
def test_reference_population_and_energy_are_independent(spin, fractional):
    matrices = analytic_matrices(spin, fractional)
    actual = physical_reference_targets(**matrices, requested=("charge", "energy"))
    np.testing.assert_allclose(actual["population"], [.6, .6] if fractional else [1., 1.], atol=1e-14)
    np.testing.assert_allclose(actual["energy_residual"], [0., 0.], atol=1e-14)
    assert actual["spin_components"].tolist() == [spin]
    if fractional and spin == 1:
        d = matrices["density"].tocsr(0).toarray()
        s = matrices["overlap"].tocsr(0).toarray()
        assert not np.allclose(d @ s @ d, 2 * d)
```

Run: `"$G2M_PHYS_PY" -m pytest src/graph2mat/core/data/tests/test_physical_references.py -q`.
Expected RED: import/helper absent, not a fixture-construction error. Run fixture-only checks first if construction fails.

- [x] **1.2 Implementar selección/pesos y registro EDM.**

```python
def normalize_physical_labels(labels=None):
    if labels is None:
        return ()
    if isinstance(labels, str) or any(x not in ("charge", "energy") for x in labels):
        raise ValueError("physical_labels must contain charge and/or energy")
    return tuple(x for x in ("charge", "energy") if x in labels)

def validate_physical_weights(electron_weight=0., mulliken_weight=0., energy_weight=0.):
    values = tuple(float(x) for x in (electron_weight, mulliken_weight, energy_weight))
    if any(not np.isfinite(x) or x < 0 for x in values):
        raise ValueError("physical weights must be finite and nonnegative")
    return values
```

Añadir dos entradas a `_KEY_TO_MATRIX_CLS`:

```python
"energy_density_matrix": OrbitalMatrix,
sisl.EnergyDensityMatrix: OrbitalMatrix,
```

Probar pesos `-1,nan,inf`; lista/string inválida; las entradas EDM no emiten fallback ni adquieren `atomic_DM`.

- [x] **1.3 RED de frontera y validación de referencias; después implementar en `physical_reference_targets`.**

```python
@pytest.mark.parametrize("small", [0., -1e-3, 1e-13])
def test_bad_overlap_is_rejected_without_regularization(small):
    from scipy.sparse import csr_array
    import sisl
    m = analytic_matrices()
    m["overlap"] = sisl.Overlap.fromsp(m["overlap"].geometry, csr_array(np.diag([1., small])))
    with pytest.raises(ValueError, match="singular|ill-conditioned|positive"):
        physical_reference_targets(**m, requested=("charge", "energy"))

def test_energy_reference_does_not_shift_itself_to_pass():
    m = analytic_matrices()
    m["energy_density"][0, 0, 0] += .01
    with pytest.raises(ValueError, match="energy|gauge|residual"):
        physical_reference_targets(**m, requested=("energy",))
```

Validar primero presencia, real/spin, geometría (`xyz`, cell, nsc, sc_off), base
(orbitales/radios/convención). Validar condicionamiento de S antes de contrastar
la S embebida de H/D/E, para que la prueba de singularidad no pase por un error
incidental de incompatibilidad. Contrastar S embebida después de ese gate.
Comparar imágenes por vector R, no asumir offsets equivalentes por longitud.
Validar k `(n,3)`, pesos `(n,)` finitos no negativos, suma 1 con tolerancia 1e-12
y al menos un peso positivo. No normalizarlos ni fabricar Γ. Luego:

```python
for k in kpoints:
    sk = np.asarray(overlap.Sk(k, format="array"), dtype=np.complex128)
    if not np.isfinite(sk).all() or not np.allclose(sk, sk.conj().T, rtol=0, atol=1e-12):
        raise ValueError("reference overlap must be finite and Hermitian")
    eigenvalues = np.linalg.eigvalsh(sk)
    if eigenvalues.size and (eigenvalues[0] <= 0 or eigenvalues[0] / eigenvalues[-1] <= 1e-12):
        raise ValueError("reference overlap is singular or ill-conditioned")
```

Usar sparse por componente, no `Tr(D(Γ)S(Γ))`. Para referencia real hermítica,
`D.tocsr(c).multiply(S.tocsr(0)).sum(axis=1)` da las poblaciones orbitales;
sumar canales y agrupar con `np.add.at(population, geometry.o2a(arange(no)), rows)`.
Energía: mismo producto H·D menos E·S, agrupar por átomo **antes** de tolerancia.
Comprobar Hermiticidad por imágenes y máscaras antes de contraer. Unknown-D/E
con contraparte no nula es error; no usar `nan_to_num`. Residuo <=1e-6.

- [x] **1.4 RED periódico independiente, y GREEN de referencias.** En la fixture de una cadena de un átomo/1s, cell `(3,20,20)`, R=2, nsc `(3,1,1)`, construir por `geometry.sc_off` los campos:

```python
fields = {
    "S": {(0, 0, 0): 1., (1, 0, 0): .2, (-1, 0, 0): .2},
    "D": {(0, 0, 0): .8, (1, 0, 0): .1, (-1, 0, 0): .1},
}
# H=.5*S, E=.5*D; no obtener E desde una predicción.
expected_population = .8 + .1 * .2 + .1 * .2
assert expected_population == pytest.approx(.84)
assert (sum(fields["D"].values()) * sum(fields["S"].values())) == pytest.approx(1.4)
ks = np.column_stack((np.arange(8) / 8, np.zeros(8), np.zeros(8)))
bloch = lambda name, k: sum(v * np.exp(2j * np.pi * np.dot(k, r)) for r, v in fields[name].items())
assert np.mean([bloch("D", k) * bloch("S", k) for k in ks]).real == pytest.approx(.84)
```

Definir `periodic_matrices()` en `physical_fixtures.py`:

```python
def periodic_matrices():
    import numpy as np
    import sisl
    from scipy.sparse import csr_array

    atom = sisl.Atom("H", orbitals=[sisl.AtomicOrbital("1s", R=2., q0=1.)])
    g = sisl.Geometry([[0., 0., 0.]], atoms=[atom], lattice=[3., 20., 20.])
    g.set_nsc([3, 1, 1])
    s = csr_array([[1. if not np.any(r) else .2 for r in g.sc_off]])
    d = csr_array([[.8 if not np.any(r) else .1 for r in g.sc_off]])
    return dict(overlap=sisl.Overlap.fromsp(g, s),
        density=sisl.DensityMatrix.fromsp(g, d, S=s),
        hamiltonian=sisl.Hamiltonian.fromsp(g, .5*s, S=s),
        energy_density=sisl.EnergyDensityMatrix.fromsp(g, .5*d, S=s),
        kpoints=np.column_stack((np.arange(8)/8, np.zeros(8), np.zeros(8))),
        kweights=np.full(8, 1/8))
```

Probar `.84` y residuo 0 con el helper.
Añadir rechazo de matriz ausente, compleja, SOC/no colineal, geometría/base/celda
distinta y malla inválida mediante copias de estas referencias, nunca fixtures
random que ya incumplan la identidad energética.

Run: `"$G2M_PHYS_PY" -m pytest src/graph2mat/core/data/tests/test_physical_references.py src/graph2mat/core/data/tests/test_periodic_hermiticity_oracles.py -q`.
Expected GREEN: todos pasan, sin regularización y con test fraccionario aceptado.

- [x] **1.5 Commit propio.**

```bash
git add src/graph2mat/core/data/physical.py src/graph2mat/core/data/matrices/__init__.py src/graph2mat/core/data/tests/physical_fixtures.py src/graph2mat/core/data/tests/test_physical_references.py
git commit -m "feat: validate physical matrix references and periodic targets"
```

### Task 2: Preparación opt-in, procedencia y collate

**Files:** Modify `physical.py`, `bindings/torch/data/dataset.py`, `tools/lightning/data.py`, `core/data/tests/physical_fixtures.py`; create `bindings/torch/data/tests/test_physical_dataset.py`.

**Interfaces:** Consumes Task 1 and `csr_to_block_dict(..., component_indices=...)`, `MatrixDataProcessor.get_labels_from_types_and_edges`, `BasisMatrixDataBase.from_config`, `_label_edge_mask`, `layout_to`. Produces `prepare_physical_data`, `load_physical_data`, `physical_layout`; appends `physical_labels=None` to dataset/DataModule constructors. Test helper `prepared_sample(target="density_matrix", spin=1, baseline=False, symmetric=True, requested=("charge", "energy"), data_cls=BasisMatrixData, fractional=False, gauge=False) -> (data, table)`.

- [x] **2.1 RED: componentes, baseline, aislamiento y unknown frente a cero observado.**

```python
@pytest.mark.parametrize("spin", [1, 2])
def test_preparation_excludes_embedded_overlap(spin):
    data, table = prepared_sample(spin=spin)
    assert data.physical_spin.tolist() == [spin]
    assert data.physical_reference_nodes.shape[1] == spin
    assert data.point_labels.ndim == (1 if spin == 1 else 2)
    np.testing.assert_allclose(data.physical_population_reference, [1., 1.], atol=1e-14)

def test_collinear_baseline_is_not_broadcast_implicitly():
    with pytest.raises(ValueError, match="collinear|baseline"):
        prepared_sample(spin=2, baseline=True)

def test_unknown_partner_is_not_an_observed_zero():
    from scipy.sparse import csr_array
    import sisl
    m = analytic_matrices()
    # Remove one D coefficient, retaining nonzero H and S counterpart.
    d = m["density"].tocsr(0).tolil()
    d[0, 1] = 0.
    d[1, 0] = 0.
    # No embedded S: adding S here would union sparse supports and insert zeros.
    m["density"] = sisl.DensityMatrix.fromsp(m["density"].geometry, csr_array(d))
    processor = MatrixDataProcessor(basis_table=AtomicTableWithEdges(m["density"].geometry.atoms.atom), out_matrix="hamiltonian", symmetric_matrix=True, sub_point_matrix=False)
    with pytest.raises(ValueError, match="unknown|observed"):
        prepare_physical_data(processor, **m, requested=("energy",))
```

Companion de cero explícito: crear CSR con `data=[5/6,0.,0.,5/6]`,
`indices=[0,1,0,1]`, `indptr=[0,2,4]`; importar sin S embebida y comprobar que
sisl preserva nnz=4. Verificado en sisl 0.16.4: no se eliminan esos ceros.
Usarlo para charge sin H/E: no es unknown, población `[5/6,5/6]`. La prueba
no puede pasar omitiendo esa distinción. Añadir un caso con NaN almacenado.

Run nuevo módulo: expected RED helpers ausentes.

- [x] **2.2 Implementar preparación con conversores existentes y máscaras de observación.**

```python
# Dentro de prepare_physical_data, antes de construir el grafo:
requested = normalize_physical_labels(requested)
targets = physical_reference_targets(overlap, density, hamiltonian=hamiltonian,
    energy_density=energy_density, kpoints=kpoints, kweights=kweights, requested=requested)
target = {"hamiltonian": hamiltonian, "density_matrix": density,
          "energy_density_matrix": energy_density}[processor.out_matrix]
spin = int(targets["spin_components"][0])
if target is None:
    raise ValueError("requested prediction target is absent")
if processor.out_matrix == "density_matrix" and spin == 2 and processor.sub_point_matrix:
    raise ValueError("collinear density baseline is not certified")
```

Crear configuración usando `.new` y `dataclasses.replace(matrix=...)` con
`csr_to_block_dict` y componentes físicos `range(spin)`. Para H gauge,
conservar etiquetas `(H,S)` según la política existente; auxiliares reference
siempre H físico. No cambiar globalmente `_matrix_component_indices`.
Pasar configuración a `data_cls.from_config`; aplanar S, referencia, partner y
máscaras con **las mismas** `point_types/edge_index/neigh_isc` y la misma
selección simétrica. No llamar a un segundo generador de vecinos.

Máscara por soporte CSR, no por contenido:

```python
observed = density.tocsr(0).copy()
observed.data = np.ones(observed.nnz, dtype=bool)
# Aplanar observed con fill_value=False. H/S ausentes son ceros estructurales;
# DM/EDM ausentes son unknown. NaN almacenado sigue siendo unknown.
```

Verificar por componente cuando sus soportes difieran. Un desconocido solo se
excluye si su contraparte es **exactamente** cero. En energy-H, D debe ser
conocida donde H predicha puede variar, aunque H_ref sea cero. El hecho de que
H_ref/S_ref sean cero en una fila no certifica D=0. Guardar masks y rechazar
configuración inválida antes de entrenamiento.

Adjuntar arrays contiguos mediante `data.process_input_array`/`setattr`; también
actualizar `_data` para el contenedor NumPy. D0 usar `DensityMatrix`/tabla
existentes, solo on-site, sin reconstruirlo desde la población objetivo.
`prepared_sample` usa especies únicas (`geometry.atoms.atom`, no todos los
átomos, que duplican tags). Mantener float64 antes de la conversión Torch para
que los esperados literales no prueben coeficientes previamente redondeados:

```python
def prepared_sample(target="density_matrix", spin=1, baseline=False, symmetric=True,
                    requested=("charge", "energy"), data_cls=BasisMatrixData,
                    fractional=False, gauge=False):
    matrices = analytic_matrices(spin=spin, fractional=fractional)
    table = AtomicTableWithEdges(matrices["density"].geometry.atoms.atom)
    processor = MatrixDataProcessor(basis_table=table, out_matrix=target,
        symmetric_matrix=symmetric, sub_point_matrix=baseline,
        n_matrix_components=spin+int(gauge),
        matrix_component_policy="h_and_overlap" if gauge else "h_only")
    if data_cls is BasisMatrixData:
        return prepare_physical_data(processor, **matrices, requested=requested, data_cls=data_cls), table
    import torch
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        data = prepare_physical_data(processor, **matrices, requested=requested, data_cls=data_cls)
    finally:
        torch.set_default_dtype(previous_dtype)
    return data, table
```

Imports de test: `BasisMatrixData`, `AtomicTableWithEdges`, `MatrixDataProcessor`
desde graph2mat y `prepare_physical_data` desde su módulo. Ningún helper genera
esperados mediante el producto que comprueba.

- [x] **2.3 RED del loader FDF y modo apagado; implementar carga mínima explícita.**

```python
@pytest.mark.parametrize("block", [None, ["2 1 0 0", "0 2 0 0", "0 0 1 0"],
                                      ["2 0 0 .5", "0 2 0 0", "0 0 1 0"]])
def test_fdf_requires_supported_explicit_grid(tmp_path, monkeypatch, block):
    from types import SimpleNamespace
    import graph2mat.core.data.physical as physical
    fdf = tmp_path / "input.fdf"
    fdf.write_text("SystemLabel analytic\n")
    fake = SimpleNamespace(get=lambda name, default=None: block if "monkhorst" in name.lower() else default)
    monkeypatch.setattr(physical.sisl, "get_sile", lambda path: fake)
    _, table = prepared_sample()
    processor = MatrixDataProcessor(basis_table=table, out_matrix="density_matrix", sub_point_matrix=False)
    with pytest.raises(ValueError, match="grid|mesh"):
        load_physical_data(fdf, processor, requested=("charge",))
```

Primer loader: FDF local con `kgrid_Monkhorst_Pack` diagonal positiva y offsets
cero, como las fuentes inventariadas. Parsear con `fdf.get`, no regex duplicada
de bloques/includes. No existe `fdf.read_brillouinzone` en sisl 0.16.4.
Comprobar el bloque de malla antes de leer matrices, también para los errores
de entrada del test con sile mínimo.
Usar `sisl.MonkhorstPack(geometry, diag, centered=True, trs=False)`; probar los
puntos contra una lista explícita de una malla 2×1×1 y la malla archivada 20×20×1.
Mallas no soportadas, offsets no cero o ausencia del bloque: error explícito;
API directa acepta arrays k/pesos explícitos y validados. Este límite es de
lectura FDF, no de la fórmula física. No asumir Γ molecular por omisión.

Leer `read_overlap` y `read_density_matrix`; solo energy lee además
`read_hamiltonian` y `read_energy_density_matrix`. Verificar fuentes/cero de
energía (TSHS/TSDE, misma referencia) y conservar paths en metadata existente.
Si falta una matriz o no se puede certificar el gauge, error indicando archivo
y recurso. No buscar arbitrariamente un archivo alternativo para hacer pasar.

Dataset opt-in llama `load_physical_data` únicamente con labels=True; la API
directa entrega objetos preparados por `prepare_physical_data`. Rechazar ZIP
opt-in con labels=True. Con opción apagada conservar la secuencia antigua,
incluyendo ZIP, espectral, NetCharge y workers. Predict sin etiquetas no lee
referencias aunque train las utilice. Testear charge con espías que hacen
fallar cualquier lectura H/EDM, y modo off con espías sobre todas las lecturas
auxiliares. No desactivar una petición física de entrenamiento por tratarla
como predict.

- [x] **2.4 Implementar ownership CPU, batching y transferencia; probar GREEN.**

```python
sizes = basis_table.basis_size[point_types]
node_owner = np.repeat(np.arange(len(point_types)), sizes * sizes)
labelled = _label_edge_mask(batch.edge_labels, edge_types, edge_index, point_types, basis_table)
sources, targets = edge_index[:, labelled]
rows = sizes[sources] * sizes[targets]
edge_source = np.repeat(sources, rows)
edge_target = np.repeat(targets, rows)
```

`physical_layout` obtiene point/edge arrays y node_graph con helpers existentes
en `metrics.py`, importados de forma diferida para evitar ciclos. NumPy devuelve
owners NumPy; Torch collate los convierte a long en CPU. `physical_symmetric`
proviene de selección real de etiquetas, no de contar signos de edge_types.
Guardar atom_structure y bincount de átomos por estructura. Conservar átomos de
base vacía en los denominadores. Rechazar estructura completamente vacía.
`TypeLayoutCollate` crea estos campos antes de su retorno no simétrico, sin
activar el layout de readout simétrico donde no aplica.

Probar dos muestras de tamaños distintos, spin1/2 por separado, get_example,
deepcopy/InMemoryData y workers: todos los auxiliares recuperados son iguales y
sin offsets automáticos. Mutar copia no altera snapshot. La copia NumPy pública
usada por métricas conserva referencias o falla explícitamente; si hoy pierde
campos, añadir regresión antes de un cambio mínimo en `processing.py`.

Run: nuevo módulo más `test_datamodule_stages.py`, tests existentes de caché,
ZIP y spin. Expected GREEN; sin cambiar salidas de muestras legacy.

- [x] **2.5 Commit únicamente archivos propios modificados en esta tarea.**

```bash
git add src/graph2mat/core/data/physical.py src/graph2mat/core/data/tests/physical_fixtures.py src/graph2mat/bindings/torch/data/dataset.py src/graph2mat/tools/lightning/data.py src/graph2mat/bindings/torch/data/tests/test_physical_dataset.py
git commit -m "feat: prepare optional physical labels and atomic layouts"
```

### Task 3: Pérdidas diferenciables con valores y gradientes independientes

**Files:** Modify `physical.py`, `metrics.py`, `physical_fixtures.py`; create `core/data/tests/test_physical_losses.py`.

**Interfaces:** Consumes prepared contract/layout from Task 2. Produces `physical_contractions` and `physical_loss_terms` with signatures above. Existing metrics accept the three final optional kwargs before `**kwargs`; gauge passes `aligned_nodes/aligned_edges` expanded to original rows, never recalculates c.

- [x] **3.1 RED con valor y derivada literales.**

```python
@pytest.mark.parametrize("baseline", [False, True])
def test_dm_terms_have_literal_loss_and_gradient(baseline):
    data, table = prepared_sample(baseline=baseline, data_cls=TorchBasisMatrixData)
    n = data.point_labels.double().clone()
    n[0] += .1
    n.requires_grad_()
    e = data.edge_labels.double().clone().requires_grad_()
    value, stats = elementwise_mse(electron_weight=2., mulliken_weight=3.)(
        nodes_pred=n, nodes_ref=data.point_labels.double(), edges_pred=e,
        edges_ref=data.edge_labels.double(), batch=data, basis_table=table)
    assert float(value) == pytest.approx(7 / 300, abs=1e-14)
    assert float(torch.autograd.grad(value, n)[0][0]) == pytest.approx(7 / 15, abs=1e-13)
    assert float(stats["physical_electron_mse"]) == pytest.approx(1 / 400)
    assert float(stats["physical_mulliken_mse"]) == pytest.approx(1 / 200)
```

Run module: expected RED metric ignores physical kwargs or helper absent.

- [x] **3.2 Implementar scatter compartido, sin desconectar autograd.**

```python
# Después de validar shapes, masks y convertir auxiliares al dtype/device:
node_values = (nodes * partner_nodes).sum(axis=-1)
edge_values = (edges * partner_edges).sum(axis=-1)
if _is_torch_tensor(nodes):
    result = nodes.new_zeros(len(layout["physical_atom_structure"]))
    result = result.index_add(0, layout["physical_node_owner"], node_values)
    result = result.index_add(0, layout["physical_edge_source"], edge_values)
    if layout["physical_symmetric"]:
        result = result.index_add(0, layout["physical_edge_target"], edge_values)
else:
    result = np.zeros(len(layout["physical_atom_structure"]), dtype=nodes.dtype)
    np.add.at(result, layout["physical_node_owner"], node_values)
    np.add.at(result, layout["physical_edge_source"], edge_values)
    if layout["physical_symmetric"]:
        np.add.at(result, layout["physical_edge_target"], edge_values)
```

`_is_torch_tensor` ya existe en metrics; importarlo diferido o reutilizar su
patrón sin import Torch obligatorio. Validar broadcasting explícito S1 frente
a D1/D2. No usar `edge_weight` en este scatter. Unknown y contraparte cero se
filtran **antes** del producto con selección/where independiente de la predicción;
unknown con contraparte no cero es error. Predicciones no finitas en términos
activos tampoco se convierten en ceros.

- [x] **3.3 Implementar términos y conectar ambas métricas conservando zero-off.**

Charge, con nodos/edges normalizados a 2D físicos:

```python
population = physical_contractions(nodes_pred + batch.physical_baseline_nodes,
    edges_pred, batch.physical_overlap_nodes, batch.physical_overlap_edges, layout=layout)
delta = population - batch.physical_population_reference
atom_structure = layout["physical_atom_structure"]
atoms_per_structure = layout["physical_atoms_per_structure"]
structures = len(atoms_per_structure)
electron_s = (_sum_by_structure(delta, atom_structure, structures) / atoms_per_structure)**2
mulliken_s = _sum_by_structure(delta**2, atom_structure, structures) / atoms_per_structure
```

Helper privado acotado, con owners ya en el backend/device correcto:

```python
def _sum_by_structure(values, owners, count):
    if _is_torch_tensor(values):
        return values.new_zeros(count).index_add(0, owners, values)
    result = np.zeros(count, dtype=values.dtype)
    np.add.at(result, owners, values)
    return result
```

Energy: tomar diferencias contra `physical_reference_nodes/edges` sin baseline
duplicada. DM residual restaura D0 antes de la diferencia. H gauge recibe
`aligned_nodes/aligned_edges` ya calculados por la pérdida base; raw MSE usa
delta-H en gauge fijo. EDM cambia el signo:

```python
perturbation = physical_contractions(delta_nodes, delta_edges,
    batch.physical_partner_nodes, batch.physical_partner_edges, layout=layout)
sign = -1. if kind == 2 else 1.
residual = batch.physical_energy_reference_residual + sign * perturbation
energy_s = _sum_by_structure(residual**2, atom_structure, structures) / atoms_per_structure
```

Devolver suma ponderada de medias por estructura y stats **detached**:
`physical_electron_mse`, `physical_mulliken_mse`, `physical_energy_mse`,
`physical_delta_n_mean`, `physical_delta_n_rmse`, `physical_structures`.
Publicar únicamente stats pertinentes, pero esquema constante entre ranks de
la misma configuración. Electron/Mulliken sobre otro target: error. Validar
capabilities/versión/spin, máscaras y shapes, no aceptar arrays sueltos que
carezcan del contrato preparado. Toda llamada valida pesos, incluso apagados.

```python
weights = validate_physical_weights(electron_weight, mulliken_weight, energy_weight)
if any(weights):
    physical, physical_stats = physical_loss_terms(nodes_pred, edges_pred,
        batch=kwargs.get("batch"), basis_table=kwargs.get("basis_table"),
        electron_weight=weights[0], mulliken_weight=weights[1], energy_weight=weights[2])
    loss = loss + physical
    stats.update(physical_stats)
```

En gauge, conservar las predicciones originales antes de filtrar filas y pasar
el delta **alineado**, no `d_nodes/d_edges` sin alinear. La función actual
filtra filas NaN: extender aligned a las filas originales, conservando máscaras
y rechazando un unknown con partner no nulo. En filas excluidas y partner cero,
la extensión cero evita NaN*0; no afirma que la referencia desconocida era cero.

```python
aligned_nodes_full = aligned_nodes.new_zeros((physical_nodes_pred.shape[0], n_h))
aligned_edges_full = aligned_edges.new_zeros((physical_edges_pred.shape[0], n_h))
aligned_nodes_full = aligned_nodes_full.index_copy(0, valid_nodes.nonzero(as_tuple=True)[0], aligned_nodes)
aligned_edges_full = aligned_edges_full.index_copy(0, valid_edges.nonzero(as_tuple=True)[0], aligned_edges)
physical, physical_stats = physical_loss_terms(physical_nodes_pred, physical_edges_pred,
    batch=batch, basis_table=basis_table, electron_weight=electron_weight,
    mulliken_weight=mulliken_weight, energy_weight=energy_weight,
    aligned_nodes=aligned_nodes_full, aligned_edges=aligned_edges_full)
```

Definir `physical_nodes_pred, physical_edges_pred = nodes_pred, edges_pred`
antes de `valid_nodes/valid_edges`. Crear estas extensiones solo con pesos
activos. No modificar denominador H/S/espectral, c ni stats legacy cuando los
pesos son cero. Testear explícitamente una fila gauge descartada por NaN con
partner cero y otra con partner no cero.

- [x] **3.4 RED/GREEN del factor de aristas, energía y gradcheck.**

```python
def test_half_edge_and_directed_self_image_agree():
    m = periodic_matrices()
    answers = []
    for symmetric in (False, True):
        processor = MatrixDataProcessor(basis_table=AtomicTableWithEdges(m["density"].geometry.atoms.atom),
            out_matrix="density_matrix", symmetric_matrix=symmetric, sub_point_matrix=False)
        data = prepare_physical_data(processor, **m, requested=("charge",))
        layout = physical_layout(data, processor.basis_table)
        pop = physical_contractions(data.physical_reference_nodes, data.physical_reference_edges,
            data.physical_overlap_nodes, data.physical_overlap_edges, layout=layout)
        answers.append(pop)
        np.testing.assert_allclose(pop, [.84], atol=1e-14)
    np.testing.assert_allclose(*answers, atol=1e-14)

@pytest.mark.parametrize("target,expected", [("hamiltonian", 1 / 288),
                                           ("energy_density_matrix", 1 / 200),
                                           ("density_matrix", 1 / 800)])
def test_energy_perturbation_is_not_tautological(target, expected):
    data, table = prepared_sample(target=target, data_cls=TorchBasisMatrixData)
    n = data.point_labels.double().clone()
    n[0] += .1
    n.requires_grad_()
    e = data.edge_labels.double().clone().requires_grad_()
    value, stats = physical_loss_terms(n, e, batch=data, basis_table=table, energy_weight=1.)
    assert float(value) == pytest.approx(expected, abs=1e-14)
    assert torch.autograd.grad(value, n, retain_graph=True)[0].abs().sum() > 0
    assert torch.autograd.gradcheck(lambda a, b: physical_loss_terms(a, b, batch=data,
        basis_table=table, energy_weight=1.)[0], (n, e), eps=1e-6, atol=1e-7)
```

Añadir diferencias finitas centrales de un parámetro del modelo, no solo de
labels. Testear carga faltante fuera de G: referencia completa `[1,1]`, aporte
representado menor; perturbación cero no implica charge=0 cuando la matriz
writer está truncada. Para energy, `prediction=reference_G` deja exactamente
`mean(r_ref**2)`, no el residuo truncado G ni un cero impuesto artificialmente.

- [x] **3.5 Invariancias y regresiones; no sustituirlas por roundtrips.**

En fixture de dos átomos con dos orbitales cada uno definir S SPD y matrices
compatibles por solución generalizada independiente:

```python
rng = np.random.default_rng(44)
b = rng.normal(size=(4, 4))
s = b.T @ b + np.eye(4)
q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
c = np.linalg.solve(np.linalg.cholesky(s).T, q)
occupation, epsilon = np.array([1.6, .9, .3, .1]), np.array([-.7, -.2, .3, .9])
h = s @ c @ np.diag(epsilon) @ c.T @ s
d = c @ np.diag(occupation) @ c.T
e = c @ np.diag(occupation * epsilon) @ c.T
angles = (.31, -.52)
u = np.zeros((4, 4))
for atom, angle in enumerate(angles):
    block = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    u[2*atom:2*atom+2, 2*atom:2*atom+2] = block
expected_population = np.array([np.diag(d @ s)[:2].sum(), np.diag(d @ s)[2:].sum()])
rotated = np.diag((u.T @ d @ u) @ (u.T @ s @ u))
np.testing.assert_allclose([rotated[:2].sum(), rotated[2:].sum()], expected_population, atol=1e-13)
```

Convertir esos campos a sisl con dos orbitales 1s/2s por átomo y repetir a
través de preparación/pérdidas con perturbación transformada `u.T@delta@u`.
Probar gauge por `H_ref/H_pred += a*S`, `E_ref += a*D`; y pred-only+aS usando
el c real de gauge. Comparar con oráculo denso independiente de c, usando la
misma ponderación matricial y aristas; energía raw no es gauge-invariante ante
pred-only shift. Comprobar intercambio de extremos/permutación atómica.

Matriz de regresiones: NumPy/Torch, spin1/2, float32/64, simétrico/dirigido,
aristas vacías, átomos con base vacía parcial, batch heterogéneo, máscaras con
NaN y contraparte cero/no cero, pesos cero idénticos al comportamiento previo.
Con monkeypatch hacer fallar cualquier ensamblado denso/eigensolver durante
`physical_loss_terms`; debe seguir funcionando. Importación NumPy en proceso
que bloquea Torch: preparación/contracción NumPy siguen importables.

Run: nuevos tests core y regresiones gauge/espectral/NumPy/DM. Expected GREEN,
gradcheck y FD sin relajar tolerancias de oráculos previos.

- [x] **3.6 Commit propio.**

```bash
git add src/graph2mat/core/data/physical.py src/graph2mat/core/data/metrics.py src/graph2mat/core/data/tests/physical_fixtures.py src/graph2mat/core/data/tests/test_physical_losses.py
git commit -m "feat: add optional charge Mulliken and energy perturbation losses"
```

### Task 4: Integración Lightning, preflight, épocas y checkpoints

**Files:** Modify `tools/lightning/model.py`, `tools/lightning/data.py`; create `tools/lightning/tests/test_physical_training.py`. Extend existing stage/coefficient/checkpoint tests only at their owning behavior.

**Interfaces:** Consumes prepared contract and detached stats Task 3, `_EpochWeightedLoss.update(means, counts, weights, stds=None)`, existing validated stages. Produces early validation of default/all future objectives, physical epoch logs, matrix fallback and unchanged checkpoint schema for model parameters.

- [x] **4.1 Crear helpers de test y RED de integración real.**

```python
def physical_model(table, loss_kwargs, *, stages=None):
    return LitMACEMatrixModel(basis_table=table, symmetric_matrix=True,
        hidden_irreps="2x0e+2x1o", edge_hidden_irreps="2x0e+2x1o",
        num_bessel=2, max_ell=1, num_interactions=2, correlation=1,
        n_matrix_components=1, coefficient_training=True,
        loss=elementwise_mse, loss_kwargs=loss_kwargs, training_stages=stages).double()

def fit_physical(model, samples, table, tmp_path, max_epochs=1):
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    collate = TypeLayoutCollate(Batch.from_data_list, table, symmetric=True)
    loader = DataLoader(samples, batch_size=2, collate_fn=collate)
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=max_epochs,
        limit_train_batches=1, limit_val_batches=0, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        default_root_dir=tmp_path)
    trainer.fit(model, train_dataloaders=loader)
    return trainer
```

Data fuera de DataModule también debe validar todos los contratos antes de
`optimizer.step`; no asumir que solo existe el loader propio del proyecto.
No añadir un argumento target ignorado en producto: el target sale del dato.

```python
@pytest.mark.parametrize("term", ["electron_weight", "mulliken_weight", "energy_weight"])
def test_each_physical_term_changes_real_model_gradient(term):
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    model = physical_model(table, {term: .3})
    batch = TypeLayoutCollate(Batch.from_data_list, table)([data])
    out = model(batch)
    total, _ = model.loss_fn(nodes_pred=out["node_labels"], nodes_ref=batch.point_labels,
        edges_pred=out["edge_labels"], edges_ref=batch.edge_labels, batch=batch, basis_table=table)
    base, _ = elementwise_mse()(nodes_pred=out["node_labels"], nodes_ref=batch.point_labels,
        edges_pred=out["edge_labels"], edges_ref=batch.edge_labels)
    parameters = list(model.parameters())
    gradients = torch.autograd.grad(total-base, parameters, allow_unused=True)
    actual = [g for g in gradients if g is not None]
    assert actual and all(torch.isfinite(g).all() for g in actual)
    assert sum(float(g.abs().sum()) for g in actual) > 0
    assert model._coefficient_training_reason(batch) == "physical objectives require matrix predictions"
```

Expected RED: coefficient certificate accepts elementwise with nonzero physical
weights or preflight/logging absent. No afirmación de éxito por una pérdida base
con gradiente no nulo: se mide `total-base`.

- [x] **4.2 Implementar guard y validación previa de todas las etapas.**

```python
weights = validate_physical_weights(**{
    name: self.loss_fn.metric_kwargs.get(name, 0.)
    for name in ("electron_weight", "mulliken_weight", "energy_weight")})
if any(weights):
    return "physical objectives require matrix predictions"
```

Ubicar guard antes de cualquier backbone/forward certificado. No suprimir el
término para mantener coefficient-only. Con pesos cero el certificado sigue
intacto. Validar pesos al construir las pérdidas y validar los loss_kwargs de
todas `_validated_training_stages`, incluidos defaults heredados y checkpoints.
En `on_fit_start`, comprobar unión de capacidades necesarias y preflight de
los datos que se usarán para train/val; usar snapshots cacheados si existen,
iterar on-demand en CPU si no (coste documentado). Para loaders directos,
validar su contrato antes del primer paso; datos no inspeccionables exigen un
error claro, no permiso implícito. No consumir irreversiblemente generadores.

```python
def test_future_physics_fails_before_optimizer_step(tmp_path, monkeypatch):
    data, table = prepared_sample(requested=("charge",), data_cls=TorchBasisMatrixData)
    stages = [{"max_epochs": 1, "loss": "elementwise_mse", "loss_kwargs": {}},
              {"max_epochs": 1, "loss": "elementwise_mse", "loss_kwargs": {"energy_weight": 1.}}]
    model = physical_model(table, {}, stages=stages)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    def forbidden_step(*args, **kwargs):
        raise AssertionError("optimizer.step ran before physical preflight")
    monkeypatch.setattr(torch.optim.Adam, "step", forbidden_step)
    with pytest.raises(ValueError, match="energy|reference|physical"):
        fit_physical(model, [data], table, tmp_path, max_epochs=2)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
```

`max_epochs` es el esquema existente; `LitMACEMatrixModel.configure_optimizers`
devuelve Adam. El test debe comprobar ese tipo antes de espiarlo; si cambia el
producto, actualizar el espía junto al test, no aceptar un espía sin llamadas.
Comprobar también target incorrecto, contract/version/spin inválidos, physical
flags ausentes y etapas sin física con datos legacy. El error identifica etapa
y recurso, no llega tras una actualización.

- [x] **4.3 RED/GREEN de agregación por época con denominadores separados.**

Con física activa, componer términos base y físicos usando `_EpochWeightedLoss`.
Elementwise base: count=node_count+edge_weight*edge_count; gauge base conserva
H/S/window y sus counts. Física: count=n_structures para cada media. No reutilizar
counts matriciales ni promediar RMSEs. Delta-N medio y RMSE usan stats propios
por estructura, unidades electrones, no error normalizado por átomo.

```python
def test_physical_epoch_is_partition_invariant():
    from graph2mat.tools.lightning.model import _EpochWeightedLoss
    for partitions in (((1., 3., 2., 4.),), ((1.,), (3., 2.), (4.,))):
        metric = _EpochWeightedLoss(2)
        for values in partitions:
            values = torch.tensor(values, dtype=torch.float64)
            metric.update(torch.stack([values.square().mean(), values.mean()]),
                          torch.tensor([len(values), len(values)]), (1., .5))
        assert float(metric.compute()) == pytest.approx(8.75)
        metric.reset()
        assert metric.counts.tolist() == [0., 0.]
```

Este test del reductor se complementa con validación real de `_log_loss` y
`_log_stats`: muestras con 1/2/3 átomos y perturbaciones conocidas; particiones
1/2/1 frente a batch completo deben dar mismo `val_loss` y stats físicos.
Confirmar que base matricial mantiene ponderación por elementos, a diferencia
de cada término físico. DDP CPU dos ranks, mismas muestras que serial, suma de
SSE/count y gradientes equivalentes; incluir un rank sin aristas. Lanzar con
PYTHONPATH absoluto y comprobar exit code, no solo texto del log.

- [x] **4.4 Checkpoints, stages y separación features/targets.**

Entrenar dos pasos CPU, guardar checkpoint propio en tmp_path y cargar con
`LitMACEMatrixModel.load_from_checkpoint`. Misma predicción/kwargs/stage al
reanudar. Cargar checkpoint legacy sin kwargs ni auxiliares y comparar pesos y
predicciones exactas; no introducir parámetros en `state_dict`. Activación de
física en segunda etapa funciona con datos completos, sin contaminar la etapa
anterior ni conservar estadísticas de pesos distintos.

```python
def test_auxiliary_targets_do_not_change_model_features():
    from copy import deepcopy
    data, table = prepared_sample(data_cls=TorchBasisMatrixData)
    model = physical_model(table, {"electron_weight": 1.}).eval()
    changed = deepcopy(data)
    changed.physical_population_reference += 17.
    changed.physical_reference_nodes += 23.
    with torch.no_grad():
        before, after = model(data), model(changed)
    for key in ("node_labels", "edge_labels"):
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
```

Alterar asimismo partner/EDM en una copia y comprobar predicciones iguales pero
pérdida distinta/rechazo de contrato. Ninguna referencia se usa para generar
features. Run nuevo módulo más stages/checkpoint/coefficient/gauge epoch/DDP
existentes. Expected GREEN; pesos cero siguen usando ruta rápida certificada.

- [x] **4.5 Commit propio.**

```bash
git add src/graph2mat/tools/lightning/model.py src/graph2mat/tools/lightning/data.py src/graph2mat/tools/lightning/tests/test_physical_training.py
git commit -m "feat: integrate physical objectives with Lightning stages and metrics"
```

### Task 5: Evidencia real, costes, gates y handoff experimental

**Files:** Create smoke/benchmark/informe/resultados/protocolo named in map; modify `auditorias_mejoras/inventario_objetivos_fisicos.md`, `auditorias_mejoras/roadmap.md`, `auditorias_mejoras/roadmap_precision_rendimiento.md`; update this plan with actual checkboxes/evidence.

**Interfaces:** Consumes local FDF loader, direct preparation, MACE model and all three metric kwargs. Produces JSON benchmark and CPU evidence; no entrenamiento experimental ni promoción de defaults.

- [x] **5.1 Smoke real, con autoverificaciones del script antes de medir.**

CLI exacta: `smoke_objetivos_fisicos.py --fdf PATH --output PATH`. Solo aceptar
FDF explícito y output fuera de dataset. Guardar hashes antes/después de
TSHS/TSDE/HSX/FDF y no sobrescribir referencias. Inicialmente usar `MD_steps/0`
de grafeno; hBN para charge/EDM cuando sus masks permitan el objetivo. Bicapa
diagnóstica separada, nunca declararla benchmark-ready.

Para cada término por separado, preparar labels apropiadas, construir MACE
pequeño con semilla 13, mismo checkpoint inicial para control/candidato, y fit
dos pasos CPU. Capturar gradientes del término, no de la pérdida total:

```python
physical = total_loss - base_loss
gradients = torch.autograd.grad(physical, tuple(model.parameters()), allow_unused=True, retain_graph=True)
used = [g for g in gradients if g is not None]
assert used and all(torch.isfinite(g).all() for g in used)
assert sum(float(g.abs().sum()) for g in used) > 0
assert torch.isfinite(physical) and float(physical) > 0
```

Probar N y Mulliken sobre DM, energy sobre H y EDM; DM energy es el mismo
contrato y debe tener smoke o un bloqueo concreto, no desaparecer del informe.
Comparar población preparada con `DensityMatrix.mulliken("atom")`; identidad
full y observables G con cálculo independiente por imágenes. Registrar tails
fuera de G. **hBN tiene DM/EDM unknown en filas del grafo: H-energy no puede
rellenarlas a cero para pasar.** Documentar aplicabilidad real por target/muestra,
no extrapolar tres muestras a 1560 snapshots.

El script rechaza datasets/output incompatibles y exit code no cero si una
prueba anunciada como soportada falla; un objetivo no aplicable se distingue
explícitamente y no cuenta como validado. Ejecutar primero self-check con la
fixture analítica, con esperado literal Task 3, después datos reales.

- [x] **5.2 Benchmark CPU pareado con trabajo comparable.**

CLI: `bench_objetivos_fisicos.py --fdf PATH --warmup 3 --rounds 12 --output PATH`.
Fases independientes: carga/preparación/validación, collate, loss+backward y
paso completo Lightning. Baseline y física usan iguales batches, predicciones,
pesos iniciales/dtype, seeds y threads. Alternar orden AB/BA y limpiar solo
gradientes propios entre pasos. No simular el coste de carga con datos precargados.

```python
from statistics import median
from time import perf_counter

def summarize_seconds(values):
    center = median(values)
    return {"median_s": center, "mad_s": median(abs(x-center) for x in values), "samples_s": values}

def paired_times(baseline, candidate, warmup=3, rounds=12):
    for _ in range(warmup):
        baseline()
        candidate()
    a, b = [], []
    for turn in range(rounds):
        for function, samples in ((baseline, a), (candidate, b)) if turn % 2 == 0 else ((candidate, b), (baseline, a)):
            start = perf_counter()
            function()
            samples.append(perf_counter()-start)
    return {"baseline": summarize_seconds(a), "physical": summarize_seconds(b)}
```

JSON: git SHA/diff hash, fuentes/hashes, versiones, CPU/threads/carga del host,
dtype/spin/nátomos/labels, pesos, muestras crudas, mediana/MAD y ratio. Incluir
bytes de auxiliares por muestra/batch y RSS con alcance declarado; sin inferir
VRAM desde RSS. Si cache reutiliza validación, medir también first-load versus
cached-access; coste on-demand no se oculta. Self-check `summarize_seconds([1,2,3])`
da mediana2/MAD1 y número exacto de llamadas warmup+rounds de cada función.

- [x] **5.3 Gates enfocados y completos, con evidencia nueva.**

```bash
"$G2M_PHYS_PY" -m pytest src/graph2mat/core/data/tests/test_physical_references.py src/graph2mat/core/data/tests/test_physical_losses.py src/graph2mat/bindings/torch/data/tests/test_physical_dataset.py src/graph2mat/tools/lightning/tests/test_physical_training.py -q
"$G2M_PHYS_PY" -m pytest src/graph2mat tests -q
"$G2M_PHYS_RUFF" check src tests auditorias_mejoras/smoke_objetivos_fisicos.py auditorias_mejoras/bench_objetivos_fisicos.py
"$G2M_PHYS_MYPY" --config-file pyproject.toml
"$G2M_PHYS_PY39" -m pytest src/graph2mat/core/data/tests/test_physical_references.py src/graph2mat/core/data/tests/test_physical_losses.py src/graph2mat/bindings/torch/data/tests/test_physical_dataset.py src/graph2mat/tools/lightning/tests/test_physical_training.py -q
"$G2M_PHYS_PY39" -m compileall -q src/graph2mat/core/data/physical.py src/graph2mat/core/data/metrics.py src/graph2mat/bindings/torch/data/dataset.py src/graph2mat/tools/lightning/data.py src/graph2mat/tools/lightning/model.py auditorias_mejoras/smoke_objetivos_fisicos.py auditorias_mejoras/bench_objetivos_fisicos.py
git diff --check
```

Usar sesiones/logs propios para suite larga, informar progreso y verificar exit
code final. Añadir import-only NumPy sin Torch a gate mínimo existente. No
reclasificar un fallo como skip ni limitar mypy excluyendo el módulo nuevo.
Si un fix cambia producto después de los gates, repetir los gates afectados y
suite completa final; guardar resultados por commit, no mezclar antecedentes.

- [x] **5.4 Protocolo posterior de precisión, sin ejecutarlo.** Escribir cuatro brazos antes de combinaciones: control, N, Mulliken y energy; para cada target aplicable, semillas 0/1/2 pareadas. Congelar lista/hash de muestras y splits de manifiesto existentes, no repartir frames excluidos ni usar bicapa no-ready como benchmark. Mismo modelo, inicialización, dtype, optimizer, steps, scheduler y stopping; pesos elegidos solo con train/validation. Registrar tiempo y bytes de auxiliares, no afirmar ahorro.

Medir error matricial raw/alineado, delta-N total, poblaciones por átomo, energía
de perturbación y observables físicos de la matriz writer **completa**, bandas,
gap/OOD pertinentes. Desglosar error dentro/fuera de G y controles de gauge.
Referencia preservada; informe por semilla y dispersión, no solo mejor corrida.
No recomendar un término si empeora OOD aunque baje su propio residuo. McWeeny
queda no aplicable como identidad exacta para ocupaciones fraccionarias; no
habilitarlo en este plan. El experimento largo/GPU sigue aplazado.

- [x] **5.5 Revisión final y cierre honesto.** Leer diff completo conectado con código existente, revisar cinco Review Focus y cobertura de spec. Si el usuario eligió Native, solicitar una revisión independiente final según la skill de review; si eligió subagentes, además conservar revisiones por tarea. No cambiar modelo del revisor salvo instrucción aplicable. Corregir hallazgos importantes y repetir evidencia afectada.

Actualizar ambos roadmaps separando `implementado/verificado CPU`,
`no aplicable`, `bloqueado por recurso concreto` y `experimental pendiente`.
No marcar todo 4.4 o toda ola 4 como cerrado: el bloque espectral previo y el
resto del roadmap conservan sus estados. Informe incluye comandos/exit codes,
resultados, commits propios, masks/colas, coste y límites. Este plan aprobado
por sí solo no satisface el goal de implementación.

- [x] **5.6 Commit explícito de evidencia propia.**

```bash
git add auditorias_mejoras/smoke_objetivos_fisicos.py auditorias_mejoras/bench_objetivos_fisicos.py auditorias_mejoras/resultados_objetivos_fisicos_cpu.json auditorias_mejoras/informe_objetivos_fisicos_cpu.md auditorias_mejoras/protocolo_precision_4_4_4_5.md auditorias_mejoras/inventario_objetivos_fisicos.md auditorias_mejoras/roadmap.md auditorias_mejoras/roadmap_precision_rendimiento.md docs/superpowers/plans/2026-10-09-physical-losses-4-4-4-5.md
git commit -m "docs: record physical loss CPU evidence and precision protocol"
git status --short --branch
```

## Handoff y condición de salida

Recomendación: **Native**, cinco tareas con contratos muy acoplados; ejecución
inline RED→GREEN y una revisión independiente al final evita releer el mismo
contexto físico en cada tarea. Alternativa: subagentes con revisión por tarea.
El usuario revisa este documento y elige/confirma método antes de producto.

El goal de implementación solo se cierra tras integración, referencias reales,
gates CPU, costes, revisión y protocolo. Si un recurso impide un objetivo,
identificar exactamente cuál y completar las partes independientes; no llamar
validación real a fixtures sintéticas. Precisión/bandas/OOD y VRAM GPU quedan
pendientes hasta los experimentos expresamente aplazados.

## Auto-revisión del plan (2026-10-09)

- Spec §§1–3: Tasks 1/3/5 conservan convenciones, observable writer frente a
  perturbación energética y límites fraccionarios. §4.1: Task 2; §§4.2–4.3:
  Tasks 3–4. §5: Tasks 1–5; §6: Task 5; §7: mapa de archivos y gates finales.
- Los cinco Review Focus tienen prueba nombrada y task propietaria. Se revisaron
  firmas, shapes, nombres de stats, keys de etapas y selección de componentes.
- Corregidos durante revisión: `max_epochs` real; delta gauge alineado y filas
  filtradas; especies únicas en tabla; float64 anterior a conversión Torch;
  soporte CSR compartido con S que puede insertar ceros; test de S singular
  que no debe pasar por incompatibilidad incidental.
- Escaneo sin marcadores de trabajo sin definir. Los ejemplos son instrucciones
  de implementación y pruebas futuras, no evidencia de tests ya ejecutados.
- Comprobación documental ejecutada con Python CPU 3.12/sisl 0.16.4: fixtures
  analíticas spin1/spin2, ocupaciones enteras/fraccionarias y cadena periódica
  generan los esperados independientes; fences Markdown balanceados. Esto no
  sustituye el RED→GREEN ni los gates de producto pendientes.
- Límite adicional explícito de loader: solo malla FDF diagonal sin offset en
  primera versión; otras mallas requieren API directa con k/pesos declarados.
- Ruling de ejecución por evidencia real: H/S presentan asimetría numérica
  de 1e-8/1e-9. Política y cota explícitas en spec y ledger; las contracciones
  usan verdaderas imágenes contrapuestas y partners forward/reverse, sin
  alterar datos. Esto sustituye el atajo de reutilizar un mismo partner para
  ambos extremos en los snippets anteriores cuando la referencia no es exacta.

## Evidencia de ejecución Native (2026-10-09)

Tasks 1–5 completas según ledger, commits propios y gates terminales.
Suite Task4: 2466 passed/2 skipped, 710.02 s, exit0. Una revisión independiente
identificó un hueco Important de preflight, corregido con tres regresiones RED→GREEN
en `371f5a2`; ningún Critical/Minor. Gate físico posterior: 226 passed en
Python 3.12 y 3.9, exit0; Ruff/mypy/compileall pasan. Ambos smokes y benchmarks
repetidos sobre el fix: nueve casos válidos, H-energy/hBN bloqueado sin ceros
inventados, hashes científicos intactos. Suite completa posterior sobre `2759800`:
2477 passed / 2 skipped / 30182 warnings, 748.84 s, exit0.
Única pasada correctora terminada; no Critical/Important abiertos ni menores aplazados.
Goal acotado CPU cerrado, no evaluación experimental ni el resto del roadmap.
Informe, costes, límites y protocolo: [informe CPU](../../../auditorias_mejoras/informe_objetivos_fisicos_cpu.md).
No se declaran mejoras de precisión, bandas, OOD ni GPU/VRAM.
