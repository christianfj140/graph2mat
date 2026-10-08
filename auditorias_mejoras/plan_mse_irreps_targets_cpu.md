# Targets CPU y MSE físico en irreps — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL `superpowers:executing-plans`;
> ejecución directa, TDD y revisión independiente al cierre del goal amplio.

**Goal:** Preparar targets CPU reutilizables y una pérdida sin reconstrucción
que conserve el MSE por elemento físico, no la media por operación de coeficientes.

**Architecture:** Snapshot CPU de Q y metadatos, sin guardar el modelo en workers.
El RTP estándar agrupa coeficientes por par de copias angulares: sus soportes
matriciales son disjuntos. Cada grupo observado usa QR local de Q efectivo,
targets proyectados y el residuo ortogonal del target. El hot path evalúa
`||(c @ R.T) - (Y @ U)||² + ||Y - Y @ U @ U.T||²`, sin bloques ni Q global.
Así se conserva también Q float32 casteado a float64. No crear Gram K×K.

**Tech Stack:** Python >=3.9, Torch/NumPy/e3nn y pytest existentes.

**Spec:** `roadmap_precision_rendimiento.md`, Task 1, incluidos sus límites.
Este subproyecto entrega el objetivo/targets; Graph2Mat/MACE/collate/Lightning
se integran después, sin dar por completo 5.3 al acabar este plan.

## Global Constraints

- Solo `/home/christian/repositorios/graph2mat`, `g2m-v2`, mismo checkout.
- APIs, defaults y checkpoints preservados; metadatos nuevos no persistentes.
- Entornos propios; no GPU ocupada ni cambios de procesos/colas/datos originales.
- `edge_weight` cuenta elementos tanto en numerador como denominador.
- Componentes explícitas, sin ignorar S ni promediar operaciones por igual.
- NaN por bloque se omite con gradiente cero; NaN parciales y fórmulas no
  certificadas solicitan fallback explícito, no targets cero.
- Snapshot válido para el Q efectivo: reconstruirlo si Q cambia; no guardar
  módulos, parámetros ni buffers CUDA en el projector serializable.
- No pérdida MAE/Huber/gauge ni claim de rendimiento experimental nuevo.

## Review Focus

1. Padding max y radiales repetidas: contar solo etiquetas físicas, no padding.
2. Targets no representables (ruido antisimétrico): conservar su residuo en SSE.
3. Cast Q32→64: QR con Q real, no supuesto de ortogonalidad exacta.
4. Ghost/vacíos/NaN: no evaluar NaN multiplicado por cero; validar formas.
5. Workers/readouts por interacción: pickle CPU y coordenadas efectivas comunes.

### Task 1: snapshot de targets y objetivo equivalente

**Files:**
- Modify: `src/graph2mat/bindings/e3nn/irreps_tools.py`.
- Modify: `src/graph2mat/bindings/e3nn/modules/matrixblock.py`.
- Create: `src/graph2mat/bindings/torch/data/coefficient_targets.py`.
- Create: `src/graph2mat/core/data/tests/test_coefficient_matrix_mse.py`.
- Modify: `docs/data_loading.rst`, este plan y el registro del goal.

**Interfaces:**
- Consume el orden de coeficientes de `E3nnIrrepsMatrixBlock`, tablas/filtros y
  mappings existentes, `batch.point_types/edge_types/edge_index`, labels CPU.
- Produce `operation._coefficient_groups`: tuplas de índices por par angular;
  `CoefficientMatrixMSEProjector(readout).__call__(batch,nodes_ref,edges_ref)`
  devuelve diccionarios node/edge con shapes, términos CPU, SSE residual y
  conteos. `coefficient_matrix_mse(predictions, targets, edge_weight=1.0)` devuelve
  `(loss, stats)` con node/edge SSE, conteos y RMSE. Sin exports de alto nivel aún.
- Shapes de predicción `(n_blocks,n_components*K)`; términos locales guardan
  filas/columnas, factor QR y targets comprimidos; preserve component-major.

- [x] Step 1: escribir RED parametrizado con readout e3nn real. Referencia
  independiente: bloques `einsum(c,Q)` recortados por el forward existente,
  `get_predictions_error` y SSE física literal. Casos point_type/basis_shape/max,
  simétrico/dirigido, 1/2 componentes, edge_weight 1/2/0 y float32/64/cast32→64.
  Target aleatorio no simétrico, orientación heteronuclear y padding. Parchear
  reconstrucción para que falle solo durante projector/loss y comparar gradientes.
  Expected: assertion por projector inexistente, no fallo de imports.
- [x] Step 2: añadir pruebas de NaN completo, NaN parcial solicitado fallback,
  ghost y ausencia de edges, shape/componente incorrectos, pickle CPU sin
  módulos y ModuleList compatible/incompatible. Ejecutar fichero nuevo.
  Expected: RED real para funciones nuevas; controles existentes no se cambian.
- [x] Step 3: añadir grupos locales al RTP al colocar slices sorted y copiar
  solo las tuplas a MatrixBlock. Sin buffers persistentes ni duplicar Q.
  Snapshot CPU clona Q/metadatos; verifica cobertura y soportes locales disjuntos.
  Por tipo físico construye máscara, QR local `q.T=U@R` y factor `R.T`.
  Para targets finitos proyecta `Y@U`, acumula residuo de `Y-(Y@U)@U.T`.
  Agrupar bloques por tipo físico para compartir factor; excluir bloque NaN.
  Validar finitud, shapes, conteos y claves; fallback explícito si máscara corta
  un soporte local o hay NaN parciales. Expected: ningún tensor target con grad.
- [x] Step 4: pérdida conecta outputs vacíos con `prediction[:0].sum()`, sin
  `NaN*0`; SSE suma términos locales y residuo. MSE divide por
  `node_count + edge_weight*edge_count`, con denominador 1 si vacío. Comparar
  valores/gradientes a 1e-12 float64 y 5e-6 float32. Expected: GREEN.
- [x] Step 5: ejecutar regresiones de coeficientes, RTP, dense readout y pérdidas;
  después suite source CPU completa, gates Ruff/mypy y floor Python3.9.
  Comando común: env CUDA_VISIBLE_DEVICES= PYTHONPATH=<repo>/src
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
  /tmp/g2m-python312-full.LHyni1/venv/bin/python -m pytest -q
  --disable-warnings --tb=short (añadir selección para el gate enfocado).
  Expected: exit0; registrar fallos reales y avisos. No GPU requerida.
- [ ] Step 6: documentar contrato, fallback e integración pendiente; commit
  explícito y task-done con el fichero nuevo más orientación/compatibilidad.
  Expected: ledger complete del subproyecto, no de Task1 amplia ni del goal.

## Diseño contrastado y decisión

- Referencia kernel verificada: 1730 pasan/2 omitidos; no se repite implementación.
- Probe de 30 máscaras confirma soportes observables completos en ij/nodos ij=ji.
- Alternativa descartada: identidad Gram aproximada pierde equivalencia en
  cast32→64. Alternativa descartada: Gram global K×K aumenta memoria/init.
- QR por par angular aprovecha el RTP actual, dimensiones <=49 para pares f/f,
  conserva redondeo y residuo. Init/RAM/throughput se medirán tras integración;
  no hay promesa de que operaciones pequeñas sean más rápidas en toda GPU.
- Ruling de ejecución: autorización del goal y ejecución directa incluyen
  concretar este subdiseño sin nueva ronda de aprobación entre tareas.
- Pre-flight: una tarea; interfaces de kernel conocidas, sin conflictos.

Evidencia final: 91 tests específicos; 147 en Python3.9.25/Torch2.4.1cpu;
suite source CPU completa1821 pasan/2 omitidos,17156warnings,409.36s,exit0.
Ruff/mypy-core/compileall39/diff verdes. QR con AMP externo protegido después
de RED2 (error relativo3.68e-4); entrada f32/f64 conserva precisión del objetivo.
