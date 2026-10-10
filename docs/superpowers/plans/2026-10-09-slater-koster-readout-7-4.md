# Slater–Koster neuronal con residual — 7.4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Ejecución Native ya elegida; no agentes por tarea. Una revisión independiente final del cambio completo.

**Goal:** Añadir el readout opcional SK+residual aprobado, verificar corrección e integración CPU y medir coste CPU; preparar, sin ejecutar, evaluación GPU/VRAM/precisión.

**Architecture:** Una operación produce integrales SK condicionadas por invariantes y las lleva a coeficientes con armónicos esféricos y mapas constantes por par de capas. Suma el residual `E3nnLowRankEdgeBlock` existente. RTP, reconstrucción, simetría, layouts y selectores se reutilizan; ningún default se modifica.

**Tech Stack:** Python >=3.9, Torch/e3nn/MACE/Lightning/NumPy/sisl ya instalados. Sin dependencias nuevas. Desarrollo y pruebas CPU.

**Spec:** `docs/superpowers/specs/2026-10-09-slater-koster-readout-7-4-design.md`, aprobado por el usuario: «Apruebo el diseño 7.4: redacta el plan».

## Global Constraints

- `/home/christian/repositorios/graph2mat`, rama `g2m-v2`, mismo checkout, sin worktree; preservar cambios ajenos.
- Python >=3.9; sin dependencias nuevas, GPU, entrenamientos comparativos largos, fonones, autorun, colas, otros repositorios, push/merge ni procesos ajenos.
- Readout actual, defaults, APIs anteriores y checkpoints legacy se conservan. No se cambia el backbone MACE ni se reabre 4.4–4.5 o 7.1.
- Una variante: SK neuronal con residual de 7.1; no segunda arquitectura SO(2).
- Fixtures propias; no hacen falta etiquetas científicas ni datos externos para este goal CPU.
- Sin beneficios ML/GPU/VRAM asumidos. No cerrar por diseño, benchmarks aislados o tests parciales.
- Plan aprobado explícitamente por el usuario: «Apruebo». Ejecución Native
  autorizada; cierre CPU documentado el 2026-10-10, sin extender a GPU/precisión.

## Review Focus

1. Bases con capas radiales repetidas: identidad de cada par/instancia/componente preservada, no mezcla por multiplicidad agregada (Task 1).
2. Direcciones degeneradas, casi polares y longitudes positivas pequeñas: errores para cero/no finito, continuidad sin clamp ni eje transversal de producción (Task 2).
3. Configuración/custom readout que elude selector Lightning: rechazo de contexto no envuelto y shifts de arista, antes de fit o de devolver predicciones inválidas (Tasks 2–3).
4. Estado serializado, dtype y orden de kwargs: mapas SK coherentes con Q, reload estricto, fuentes correctamente identificadas, defaults sin claves nuevas (Tasks 1–3).
5. Coefficient-only, agrupación max y tipos ausentes: mapas/orientaciones/canales correctos; fallback actual si no certificable, nunca una pérdida aparentemente verde pero distinta (Task 3).

## Archivos y responsabilidades

Rutas siguientes relativas a `/home/christian/repositorios/graph2mat`.

| Archivo | Acción y responsabilidad |
| --- | --- |
| `src/graph2mat/bindings/e3nn/modules/slater_koster.py` | Crear: calibración angular privada y operación opcional |
| `src/graph2mat/bindings/e3nn/modules/__init__.py` | Exportar operación; binding superior ya reexporta módulos |
| `src/graph2mat/bindings/e3nn/modules/matrixblock.py` | Metadatos de constructor solicitados, warning angular y guard de shifts para nueva operación |
| `src/graph2mat/models/mace.py` | Publicar vectores dirigidos existentes; conservar/ensanchar guard de contexto |
| `src/graph2mat/tools/lightning/models/mace.py` | Documentar selección; cambiar producto solo si una prueba descubre un hueco real |
| `src/graph2mat/bindings/e3nn/modules/tests/test_slater_koster_readout.py` | Crear: oráculos SK, O(3), contrato, cutoff y derivadas |
| `src/graph2mat/tools/lightning/tests/test_slater_koster_model.py` | Crear: modelos/config/checkpoints/periódicas/coeficientes |
| `tests/test_slater_koster_benchmark.py` | Crear: autocomprobación del benchmark sin medición larga |
| `auditorias_mejoras/bench_slater_koster_cpu.py` | Crear: benchmark de tres variantes, resultados crudos y memoria host |
| `auditorias_mejoras/resultados_slater_koster_cpu.json` | Crear al medir: evidencia real, no plantilla con cifras |
| `auditorias_mejoras/informe_slater_koster_cpu.md` | Crear al ejecutar: comandos, resultados, revisiones y límites |
| `auditorias_mejoras/protocolo_gpu_precision_7_4.md` | Crear: contrato experimental posterior, sin ejecutarlo |
| `auditorias_mejoras/roadmap.md`, `auditorias_mejoras/roadmap_precision_rendimiento.md` | Actualizar estados al final, separando CPU y experimental |

No modificar core de datos, pérdidas, `_stored_rtps`, backbone ni benchmark 7.1
sin una regresión que demuestre una necesidad dentro de este alcance.

## Comandos y evidencia

Antes de ejecutar, validar que existen los entornos y registrar sus versiones;
si no existen, buscar entornos ya disponibles sin instalar ni tocar otros repos.
Si faltan recursos imprescindibles, informar el bloqueo concreto.

```bash
cd /home/christian/repositorios/graph2mat
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH=/home/christian/repositorios/graph2mat/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
G2M_PY=/tmp/g2m-python312-full.LHyni1/venv/bin/python
G2M_PY39=/tmp/g2m-python39.D7TJAp/full-venv/bin/python
G2M_RUFF=/tmp/g2m-static-gates.YpKMAH/venv/bin/ruff
G2M_MYPY=/tmp/g2m-static-gates.YpKMAH/venv/bin/mypy
export G2M_EVIDENCE=$(mktemp -d /tmp/g2m-sk-evidence.XXXXXX)
test -x "$G2M_PY"
test -x "$G2M_PY39"
test -x "$G2M_RUFF"
test -x "$G2M_MYPY"
git status --short
git rev-parse HEAD
"$G2M_PY" -m pip check
```

Variables de shell anteriores solo sirven para esta tarea. Guardar logs y exit
codes con el mecanismo de ejecución; no aceptar el exit del último comando de
un pipeline como prueba de pytest. No excluir warnings/fallos desconocidos.

Antes del primer cambio de producto, capturar referencia legacy propia:

```bash
"$G2M_PY" - <<'PY'
import os
from pathlib import Path
import torch
import graph2mat
from graph2mat.tools.lightning.tests.test_coefficient_matrix_training import training_inputs
torch.set_default_dtype(torch.float64)
model, batch, _ = training_inputs(coefficient_training=False)
model.eval()
out = model(batch.clone())
torch.save({"state_dict": model.state_dict(), "batch": batch,
            "predictions": {key: out[key].detach() for key in ("node_labels", "edge_labels")}},
           Path(os.environ["G2M_EVIDENCE"]) / "legacy-default.pt")
PY
```

Es un artefacto de prueba propio, no un checkpoint ajeno. Task 3 cargará este
estado con `weights_only=False`, comparará claves y predicciones exactas.

### Task 1: mapas SK por capa y oráculos independientes

**Files:** Crear `slater_koster.py` y `test_slater_koster_readout.py` en los
directorios indicados. No integrar todavía en MACE ni exportar operación incompleta.

**Interfaces:**

```python
def _sk_couplings(i_basis, j_basis, single_irreps_out, change_of_basis):
    # list[(segments, transform)] en orden de pares de capas expandidas.
    # segments: tuple[(coefficient_start, coefficient_stop, L), ...]
    # transform: Tensor[len(segments), min(la, lb)+1], sin gradiente.
    # Q de UN componente; los componentes se duplican en Task 2.
```

1. **RED — prueba axial y rotada, sin reutilizar calibración como oráculo.**

- [x] Crear test parametrizado sobre los 16 pares la/lb=0..3 y cada q admitido.
  Construir Q con `ReducedTensorProducts('ij', ...)`, bases con una capa y el
  bloque axial literal; importar `graph2mat` antes de e3nn para usar el bootstrap
  de serialización ya existente. Núcleo del oráculo:

```python
axis = torch.zeros(2 * la + 1, 2 * lb + 1, dtype=torch.float64)
axis[la - q, lb - q] = 1
axis[la + q, lb + q] = 1
R = o3.angles_to_matrix(*(torch.tensor(x, dtype=torch.float64)
                         for x in (0.31, 0.73, -0.29)))
n = R @ torch.tensor([0., 1., 0.], dtype=torch.float64)
expected = (o3.Irrep(la, (-1)**la).D_from_matrix(R) @ axis
            @ o3.Irrep(lb, (-1)**lb).D_from_matrix(R).T)
groups = _sk_couplings(i_basis, j_basis, rtp.irreps_out, rtp.change_of_basis)
segments, transform = groups[0]
coeff = torch.zeros(rtp.irreps_out.dim, dtype=torch.float64)
for k, (start, stop, L) in enumerate(segments):
    coeff[start:stop] = transform[k, q] * o3.spherical_harmonics(
        L, n, normalize=True, normalization="component")
actual = torch.einsum("k,kij->ij", coeff, rtp.change_of_basis)
torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
assert torch.linalg.matrix_rank(transform) == min(la, lb) + 1
```

- [x] Ejecutar `"$G2M_PY" -m pytest src/graph2mat/bindings/e3nn/modules/tests/test_slater_koster_readout.py -q --tb=short`; verificar fallo por helper ausente.

2. **GREEN — calibración mínima, sin construir marcos en forward.**

- [x] Expandir cada multiplicidad en capas radiales separadas. Localizar cada
  instancia de irrep con su soporte en Q y sus offsets, no solo con l/dimensión.
  Para cada capa y q, proyectar el axial literal con Q y comprobar el soporte
  esperado. Fórmula de amplitud por segmento:

```python
y0 = o3.spherical_harmonics(L, n0, normalize=True, normalization="component")
projected = torch.einsum("kij,ij->k", Q_segment, axial)
amplitude = (projected * y0).sum() / y0.square().sum()
torch.testing.assert_close(projected, amplitude * y0, **calibration_tolerances)
```

- [x] Comprobar L/paridad permitidos, rango, reconstrucción completa y
  dimensiones/Q finito; rechazar Q incompatible y bases no naturales/l>3.
  No persistir copia completa adicional de Q en operación ni matriz KxK.
  Calibrar en dtype de Q; conservar precisión generando Q float64 cuando la
  prueba lo exige, no convirtiendo Q float32 y pretendiendo recuperarla.

3. **Regresiones de identidad.**

- [x] Añadir caso `2x0e+2x1o+1x2e+1x3o` frente a `1x0e+2x1o+1x3o`;
  activar una única integral de un único par y verificar bloque cero fuera de
  ese par. Repetir todos los pares, no solo la primera capa. Alterar Q con una
  permutación no certificada y comprobar error de calibración, no éxito aparente.
- [x] Repetir tests con default float32 y float64 restaurando dtype al terminar.
  Presupuestos iniciales: float64 2e-12, float32 2e-5 para oráculo angular;
  cualquier ajuste exige identificar su fuente numérica, no tapar un fallo.
- [x] Guardar RED/GREEN en evidencia y commit local solo de estos dos archivos.

### Task 2: operación SK+residual, contrato, cutoff y derivadas

**Files:** `slater_koster.py`, export en `modules/__init__.py`, cambios acotados
en `matrixblock.py`, tests `test_slater_koster_readout.py`.

**Interfaces:** Operación pública seleccionable por mecanismos existentes:

```python
class E3nnSlaterKosterEdgeBlock(torch.nn.Module):
    _data_get_edge_args = ("edge_feats", "edge_vectors")
    requires_zero_edge_shift = True

    def __init__(self, irreps_in, irreps_out, edge_feats_irreps,
                 i_basis, j_basis, single_irreps_out, change_of_basis,
                 symmetry="ij", n_matrix_components=1, rank=4):
        # buffers pequeños sk_transform_0, ...; metadatos _sk_groups.
        # submódulos públicos de estado: sk_net y residual.
        # sk_irreps: irreps naturales realmente cubiertas por rama angular.

    def forward(self, edge_feats=None, edge_vectors=None, **tuple_kwargs):
        # Tensor[n_edges, irreps_out.dim], componentes en orden RTP existente.

    def _sk_coefficients(self, integrals, vectors):
        # integrals: [n, C, sum_ab(min(la,lb)+1)]; vectors: [n,3].
        # Privado; permite oráculos con V literal, sin desactivar gates públicos.
```

1. **RED — rama SK real y op opcional ausente.**

- [x] Test de export y construcción mediante `E3nnIrrepsMatrixBlock`; helper
  local de test `sk_block(i_basis, j_basis, components=1, **kwargs)` usa
  `irreps={'node_feats_irreps': o3.Irreps('2x0e+1x1o'),
  'edge_feats_irreps': o3.Irreps('2x0e')}` y `symmetry='ij'`.
- [x] Fijar V en `_sk_coefficients`, reconstruir mediante el wrapper y probar
  fórmulas analíticas s-s, s-p, p-p, por C1/C2 y dirección genérica. p-p:

```python
n = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
expected = (v_pi[:, None, None] * torch.eye(3, dtype=n.dtype)
            + (v_sigma-v_pi)[:, None, None] * n[:, :, None] * n[:, None, :])
torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
```

- [x] Ejecutar archivo nuevo; observar fallos por operación/export/metadatos
  ausentes, no por imports de terceros. Guardar fallo observado.

2. **GREEN — operación e inyección mínima.**

- [x] En `matrixblock.py`, guardar/entregar i_basis/j_basis, single_irreps_out,
  Q, symmetry y C únicamente a parámetros explícitos de constructor solicitados
  por firma. Sin mutar dict `irreps` recibido ni cambiar llamadas legacy.
- [x] Construir `residual=E3nnLowRankEdgeBlock(...)` sin modificar 7.1. Reutilizar
  `_ordered_inputs`, `_scalar_slices`/`_extract_scalars` existentes para routing
  y conditioning 0e; no asumir escalares al principio de features.
- [x] Registrar transformaciones pequeñas por par como buffers; no registrar
  Q otra vez. MLP `Linear(condition_dim,32) -> SiLU -> Linear(32,C*n_integrals)`.
  Radiales promediadas y escalares de ambos extremos en orden estable; fuente
  node y edge distinguida aunque kwargs se inviertan. Núcleo de forward:

```python
radial = (edge_feats[0] + edge_feats[1]) * 0.5
gate = torch.tanh(radial.square().sum(-1, keepdim=True))
integrals = (self.sk_net(condition) * gate).reshape(n, C, n_integrals)
return (self._sk_coefficients(integrals, edge_vectors[0])
        + self.residual(edge_feats=edge_feats, **tuple_kwargs))
```

- [x] Validar fuentes/tuple/shape/C/rank/dtype/device, radiales 0e, finitud,
  vectores opuestos y longitud no cero. Rechazar C distinto de 1/2, complejo,
  paridad no natural/l>3 y symmetry distinta de `ij` al construir cuando sea
  identificable. No confundir C2 real con SOC. Preservar lote n=0 sin `max()`.
- [x] Normalizar vectores con escalado por máximo absoluto antes de su norma
  cuando sea necesario evitar underflow/overflow; no convertir longitud pequeña
  válida en cero ni clamplear dirección. Esta estabilidad no amplía el dominio
  a vectores realmente cero.
- [x] Warning de alcance: eliminar solo irreps faltantes cubiertas por
  `sk_irreps`. Test con inputs escalares y f-f confirma L=6e angular y mantiene
  aviso de canales no naturales que el residual escalar no puede producir.
- [x] Guard específico de `requires_zero_edge_shift`: `set_output_scaleshift`
  rechaza `shift_scalars=True`; forward rechaza shift no cero/no finito si
  scaleshift habilitado, incluidos buffers cargados manualmente. Escalado
  por irrep y `shift_scalars=False` siguen soportados y probados.

3. **O(3), libertad de marco y residual.**

- [x] Repetir transporte de Task 1 con C1/C2, capas repetidas, ejes exactos,
  direcciones próximas a polos y rotaciones impropias. Verificar
  `B(Rn)=Di(R) B(n) Dj(R).T` y coeficientes con sus irreps.
- [x] Test de gauge del marco usando `R` y `R @ Ry(phi)` con al menos dos phi;
  comparar ambos transportes y el forward sin cartas. Ningún Euler en producto.
- [x] Apagar residual poniendo sus parámetros a cero; demostrar SK no nula.
  Apagar `sk_net` y controlar pesos del residual; demostrar un bloque que no
  queda axial en el marco del enlace. Mantener ambos sumandos entrenables.

4. **Gradientes, negativos y corte.**

- [x] Gradcheck float64 para vectores y features no triviales. Incluir parámetros
  mediante functional_call, sin reasignar `.data`; patrón:

```python
from torch.func import functional_call
names, values = zip(*operation.named_parameters())
params = tuple(p.detach().clone().requires_grad_() for p in values)
def f(*new_params):
    return functional_call(operation, dict(zip(names, new_params)), (), kwargs)
assert torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-5, rtol=1e-4)
```

- [x] Cubrir 0o en radial, rank bool/0, formas incorrectas, NaN/Inf, vectores
  iguales en ambas direcciones, cero, n=0, palabras clave invertidas, features
  repetidas/no contiguas y longitudes pequeñas representables (p. ej. 1e-20
  float32). Comprobar mensaje de error y no mutación de entradas.
- [x] Con nodos fijos no nulos y el envelope radial existente, comprobar valor
  exactamente cero en/fuera del corte, límite interior y derivada radial cero.
  Incluir escalado de salida ajustado; shifts no nulos deben fallar.
- [x] Ejecutar tests nuevos + `test_low_rank_edge_readout.py`,
  `test_coefficient_only_block.py`, `test_edge_pairing_and_safety.py`.
  Guardar evidencia y commit propio de Task 2.

### Task 3: MatrixMACE, configuración, periódicas y checkpoint

**Files:** `models/mace.py`, documentación del selector en
`tools/lightning/models/mace.py`, nuevo `test_slater_koster_model.py`.

**Interfaces:** Misma API actual: `edge_block_readout=E3nnSlaterKosterEdgeBlock`,
`edge_block_readout_kwargs={'rank': 2}` para fixtures pequeñas. MACE publica
`data_for_readout['edge_vectors']=vectors` ya calculado con shifts; Graph2Mat
parea ese campo mediante `_data_get_edge_args`. No segunda conversión de base.

1. **RED — integración mínima.**

- [x] Reutilizar `training_inputs` y `data_module_for` de
  `tools/lightning/tests/test_coefficient_matrix_training.py`; definir
  `sk_inputs(**kwargs)` reenviando selector y rank. Test parametrizado sobre
  C=1/2, grouping=point_type/basis_shape/max y mode=single/sum/mean, con
  reutilización de nodos activada/desactivada. Verificar fallo por vectores
  no publicados antes de cambiar MACE.

```python
model, batch, _ = sk_inputs(components=components, grouping=grouping, mode=mode)
out = model(batch.clone())
expected = ((out['node_labels']-batch.point_labels).square().sum()
            + 2*(out['edge_labels']-batch.edge_labels).square().sum())
expected /= batch.point_labels.numel() + 2*batch.edge_labels.numel()
loss = model.training_step(batch.clone(), 0)['loss']
torch.testing.assert_close(loss, expected, rtol=1e-10, atol=1e-12)
loss.backward()
```

2. **GREEN — plomería y guard.**

- [x] Publicar vectores sin recalcular vecinos/shift ni modificar datos de
  entrada. Guard de `hamiltonian_local_context` inspecciona operaciones reales;
  conservar guard low-rank existente. El nuevo residual contenido debe quedar
  cubierto sin añadir un catálogo de clases si el guard actual ya lo encuentra.
- [x] Test directo de `MatrixMACE` con subclase personalizada de `E3nnGraph2Mat`
  que selecciona SK; contexto prohibido falla al construir, no tras fit.
  Con `mp_cutoff`, readout conserva todas las aristas de matriz, no solo las MP.
- [x] Comparar forward con/sin layout y coefficient-only con pérdidas/gradientes
  de matrices. Invalidar mapa certificado y comprobar fallback actual, no
  forzar ruta rápida. Cubrir un tipo ausente y datos sin edge de ese tipo.

3. **Periódicas, convenciones y posición.**

- [x] Reutilizar patrón del oráculo periódico de `test_low_rank_edge_model.py`:
  sisl con s/p/d/f y conversiones de base reales; no importar su test como
  sustituto. Incluir homo/hetero, self-image de shift no cero, intercambio de
  extremos, permutación y supercelda equivalente. Probar al menos convenciones
  cartesian/spherical/siesta_spherical y las rotaciones con inversión.
- [x] Diferencia finita de posiciones con topología congelada, lejos de cortes
  y cambio de vecinos; incluir shift periódico para que no desaparezca el
  gradiente geométrico. Comparar con autograd en componentes concretas.

4. **Dos pasos, serialización y defaults.**

- [x] Test `pl.Trainer(accelerator='cpu', devices=1, max_steps=2, max_epochs=2,
  logger=False, enable_checkpointing=False, enable_progress_bar=False,
  enable_model_summary=False, num_sanity_val_steps=0)` y fixture de datos.
  Confirmar `global_step==2`, gradientes finitos y cambio efectivo en `sk_net`
  y residual de operaciones utilizadas. No exigir gradiente en tipos ausentes.
- [x] `trainer.save_checkpoint(tmp_path/'sk.ckpt')`, reload estricto con
  `LitMACEMatrixModel.load_from_checkpoint(..., map_location='cpu',
  weights_only=False)` de checkpoint propio. Predicciones iguales y mapas
  no entrenables, rank/clase/C preservados. Repetir legacy quitando kwargs
  opcionales del checkpoint como hace test low-rank existente.
- [x] Parsear class path con el mecanismo CLI/jsonargparse ya usado en
  `test_models.py`; sin nueva ruta de selección. Ejemplo esperado:

```yaml
model:
  edge_block_readout: graph2mat.bindings.e3nn.E3nnSlaterKosterEdgeBlock
  edge_block_readout_kwargs:
    rank: 4
```

- [x] Comparar lista exacta de claves/defaults legacy con estado de base
  `8c8dc3e` mediante evidencia/fixture local obtenida antes de cambiar producto,
  no mediante otro worktree ni git checkout. Verificar ausencia de `sk_net`,
  `sk_transform` y `residual` nuevos en arquitectura default; comparar salidas
  exactas cargando los mismos pesos. Conservar límites de checkpointing/compile/
  AMP y documentar combinaciones no certificadas, sin ejecutarlas en CUDA.
- [x] Ejecutar tests SK de modelo + low-rank + coefficient_matrix_training +
  hamiltonian_context + modelos/CLI existentes. Commit solo de Task 3.

### Task 4: benchmark CPU emparejado y protocolo de precisión/GPU

**Files:** nuevo `bench_slater_koster_cpu.py`, `test_slater_koster_benchmark.py`,
`protocolo_gpu_precision_7_4.md`; JSON solo generado por medición real.

**Interfaces:** CLI `--points 72 --rank 4 --rounds 12 --warmup 3`; salida JSON
stdout. Variantes `baseline`, `low_rank`, `sk_residual`; CPU float32 con C2 y
s/p/d/f. Reutilizar `dataset_for` de `bench_cache_tensores_cpu.py` y
`digest_tensors`, `parameters`, `summary` de `bench_readout_rango_bajo.py`.
No modificar los resultados ni la semántica del benchmark histórico.

1. **RED — contrato de benchmark sin medida larga.**

- [x] Test por subprocess con `--points 4 --rounds 1 --warmup 1 --rank 1`,
  CUDA vacío. JSON contiene tres variantes, medidas crudas no vacías,
  median/MAD recalculables, hashes, `device=cpu` y scopes explícitos. Valorar
  resultado numérico finito, no un tiempo mínimo obligatorio. Código de check:

```python
assert set(result['measure']['full_model']) == {'baseline', 'low_rank', 'sk_residual'}
for phases in result['measure']['full_model'].values():
    for record in phases.values():
        assert len(record['raw_ms']) == result['rounds']
        assert record['median'] == statistics.median(record['raw_ms'])
assert result['device'] == 'cpu'
assert 'vram_bytes' not in result
```

2. **GREEN — medición de igual trabajo, no igual arquitectura.**

- [x] Tres modelos con backbone/onsite idénticos por carga estricta de estado,
  fixture congelada, dtype/hilos idénticos y semillas registradas. Comparación
  aislada con wrappers del MISMO par s/p/d/f, Q y entradas; no comparar SK de
  orbitales completos con otra operación de irreps recortadas.
- [x] Medir forward de entrenamiento, backward y paso completo Adam; separar
  construcción/calentamiento. Warmup inicializa optimizador antes de cronometrar
  pasos. Rotar orden de las tres variantes y alternar reverso para que ninguna
  quede siempre en medio. Patrón de orden:

```python
names = ['baseline', 'low_rank', 'sk_residual']
order = names[index % 3:] + names[:index % 3]
if (index // 3) % 2:
    order.reverse()
```

- [x] Guardar raw_ms por fase/variante y resúmenes median/MAD/min/max. Fuera
  de timing comprobar loss/gradientes finitos, formas, ramas SK y residual
  efectivas y hash del batch inalterado. Registrar política de actualizaciones
  del optimizador, copias/loss incluidas y carga host no aislada.
- [x] Memoria host en subprocess fresco por variante, misma fixture/secuencia:
  `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024` en Linux, etiquetado
  `process_peak_rss_bytes`. Incluye imports/construcción/ejecución; no atribuirlo
  solo al readout ni medir variantes acumuladas en el mismo proceso. Guardar
  también bytes de parámetros/buffers con alcance diferenciado; ninguna cifra
  se denomina VRAM. Manejar plataforma no Linux con error explícito del modo
  de memoria, no unidad incorrecta.
  Implementar helper privado `_memory_worker(variant, points, rank, connection)`
  con `multiprocessing.get_context('spawn')` y `Pipe`: el hijo construye UNA
  variante, ejecuta la secuencia fija y envía el récord RSS/scopes. El padre
  comprueba exit code/errores y usa el mismo handle si tarda, sin reiniciarlo
  por timeout de observación. No memoria medida en proceso padre con tres modelos.
- [x] Ejecutar autocomprobación y luego benchmark principal. Guardar stdout
  real como `resultados_slater_koster_cpu.json` mediante el runner de evidencia;
  no inventar cifras ni escribirlo manualmente. Incluir SHA Git y hashes de
  archivos producto/script, versiones, fixture, dtype e hilos. Probar JSON
  también con Python 3.9 mediante smoke pequeño, no repetir benchmark largo.

3. **Protocolo posterior, no ejecución.**

- [x] Documentar baseline/7.1/SK+residual, seeds 0/1/2 emparejadas, splits
  congelados y budget por pasos/muestras junto a tiempo para no esconder coste.
  Congelar material/split/scheduler/selección de checkpoint antes de lanzar.
  Dar un preflight que liste los artefactos científicos necesarios; la ausencia
  de dataset/split concreto significa experimento no listo, no completar ese
  campo con un identificador inventado.
- [x] Métrica primaria predefinida: MAE H alineada al gauge en test; secundarias
  H raw, bloques por especie/distancia, bandas/gap y OOD cuando haya referencias.
  Promoción requiere razón geométrica pareada de MAE primaria <1 y ninguna
  degradación OOD pareada por encima del 5%; si dispersión/intervalos no permiten
  concluir, resultado inconcluso, no promoción. Mismo H0/gauge/pérdidas entre
  variantes, sin modificar física 4.4–4.5 durante la comparación.
- [x] GPU: batch fijo y máximo batch separadamente; forward/backward/paso,
  sincronización, warmup, median/MAD y peak allocated/reserved tras reset.
  Registrar GPU/software/dtype y AMP solo si se autoriza y valida aparte.
  Precisión o VRAM no evaluadas se mantienen pendientes. No ejecutar comandos
  GPU ni entrenamientos al redactar este protocolo.
- [x] Commit scripts/tests/protocolo/resultados reales; informar cualquier
  regresión de tiempo o memoria, no promover defaults.

### Task 5: gates finales CPU, revisión y cierre auditable

**Files:** informe CPU, resultados/gates, roadmaps y registros de este plan/spec;
producto/tests únicamente para corregir hallazgos dentro del alcance aprobado.

**Interfaces:** Matriz de aceptación con cada requisito de spec §§3–6,
evidencia/commit/comando/exit/conteos y estado probado/fallido/pendiente. JSON
con benchmark y protocolo no sustituye logs de pruebas.

1. **Gates posteriores al último cambio de producto.**

- [x] Ejecutar y registrar:

```bash
"$G2M_PY" -m pytest src/graph2mat/bindings/e3nn/modules/tests/test_slater_koster_readout.py src/graph2mat/tools/lightning/tests/test_slater_koster_model.py tests/test_slater_koster_benchmark.py -q --tb=short
"$G2M_PY39" -m pytest src/graph2mat/bindings/e3nn/modules/tests/test_slater_koster_readout.py src/graph2mat/tools/lightning/tests/test_slater_koster_model.py tests/test_slater_koster_benchmark.py -q --tb=short
"$G2M_PY" -m pytest src/graph2mat tests -q --tb=short
"$G2M_RUFF" check src tests
"$G2M_RUFF" check --no-respect-gitignore --config 'include=["*.py"]' auditorias_mejoras/bench_slater_koster_cpu.py
"$G2M_MYPY" --config-file pyproject.toml
"$G2M_PY39" -m compileall -q src/graph2mat/bindings/e3nn/modules/slater_koster.py src/graph2mat/models/mace.py src/graph2mat/tools/lightning/models/mace.py auditorias_mejoras/bench_slater_koster_cpu.py
git diff --check
```

- [x] No considerar 2477/2 histórico como prueba actual. No omitir tests nuevos
  por faltar feature ni relajar oráculos para obtener verde. Entornos ausentes
  se informan; no instalar ni declarar el gate satisfecho.

2. **Revisión independiente final única (método Native).**

- [x] Tras commits propios, pedir un revisor nuevo para todo el rango desde
  `8c8dc3e`, spec/plan/fixtures y benchmark. Solo revisión, sin agentes de
  implementación por tarea. Revisar en particular los cinco Review Focus y
  que SK no sea una rama siempre nula ni una reducción silenciosa de alcance.
- [x] Por cada hallazgo importante: test RED de causa, fix mínimo, GREEN;
  repetir gates afectados y full CPU después del último fix de producto.
  No dejar Critical/Important abiertos ni convertirlos en skips. No reenviar
  revisiones repetitivas para acumular coste sin una nueva pregunta concreta.

3. **Informe y actualización precisa.**

- [x] Registrar versiones, base/final SHA, comandos, RED/GREEN, counts/warnings,
  tolerancias justificadas, resultados crudos, tiempos/params/RSS con scopes,
  revisión y limitaciones. Tabla de aceptación basada en evidencia nueva.
- [x] Separar explícitamente implementado/verificado CPU, fallido y pendiente.
  La suite de corrección no demuestra precisión científica. Reportar coste
  adicional frente a 7.1 si lo hay, además de comparación frente a baseline.
- [x] Actualizar 7.4 y Task 4: código CPU solo si pasa todo; GPU/VRAM/precisión,
  bandas/OOD permanecen abiertos. No cerrar todo Task 4 ni tocar otros puntos.
- [x] Commit local de documentación propia; comprobar estado de trabajo y
  preservar los untracked previos. Sin push, merge, borrados ni nueva rama.
- [x] Auditar cada requisito y artefacto antes de marcar goal complete. Si
  cualquier prueba obligatoria falla/falta, continuar o explicar bloqueo real.
  No cerrar por presupuesto, plan escrito ni artefactos de intención.

## Evidencia de planificación (no implementación)

- Base de producto `8c8dc3e`; diseño `8d49845`; sin cambios de producto al
  redactar este plan. El usuario aprobó diseño, no una implementación existente.
- Probe CPU efímero, sin archivos de producto, el 2026-10-09: 16 pares la/lb
  y 30 integrales axiales unitarias; proyección sobre Y_L(n0) con L/paridad
  naturales da error máximo `1.3322676295501878e-15`, exit 0, float64.
  No probó rotaciones/gradientes/integración; Tasks 1–3 siguen obligatorias.
- Primera invocación del probe falló al importar e3nn antes de graph2mat:
  `torch.load` rechazó `slice` en constantes de e3nn 0.4.4. El bootstrap existente
  de `graph2mat/__init__.py` ya allowlistea `slice`. Se corrigió el orden de
  imports del probe, sin parche de entorno, sin desactivar weights_only y sin
  cambiar producto. Los ejemplos nuevos deben respetar ese orden.
- APIs gradcheck/functional_call contrastadas con documentación primaria de
  [PyTorch 2.11](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/notes/extending.rst)
  y [torch.func](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/func.api.md)
  mediante Context7; pruebas reales en Torch 2.4.1 del piso siguen pendientes.
- Self-review: spec §§1–2 → Constraints/arquitectura; §3 → Tasks 1–2; §4 →
  Tasks 2–3; §5 → Tasks 1–3/5; §6 → Task 4; §7 → Task 5. Cinco Review Focus
  asignados; ningún test, benchmark ni beneficio de implementación declarado.

## Handoff

Plan aprobado y ejecutado con método **Native** y una revisión independiente
final (0 Critical / 0 Important / 0 Minor). Tasks 1–5 satisfechas en CPU:
2597 passed / 2 skipped existentes; focused120 en Python3.9 y3.12. Ruff,
mypy-core, compileall3.9, pip check y diff check en verde. No fixes de producto
posteriores a la revisión ni menores diferidos.

Base producto `8c8dc3e`, código/tests final `8a5cc7c`, benchmark medido `f0acdad`;
commits posteriores solo documentación/evidencia. RED/GREEN, tolerancias,
tradeoffs, decisiones y logs en el
[informe CPU](../../../auditorias_mejoras/informe_slater_koster_cpu.md).
El protocolo GPU/precisión está preparado, **no ejecutado**. 7.4 global y
Task4 experimental siguen abiertos; no se alteran defaults.
