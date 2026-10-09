# 7.1 — Readout de aristas de rango bajo: plan de implementación

> Ejecución directa con `superpowers:executing-plans`, sin agentes por tarea.

**Goal:** Implementación opcional integrada y verificada en CPU de 7.1;
protocolo GPU/precisión preparado, sin ejecutar GPU ni declarar beneficios sin medir.

**Spec:** objetivo 7.1 aprobado en esta conversación y Task 3 de
`auditorias_mejoras/roadmap_precision_rendimiento.md`.

**Architecture:** `E3nnLowRankEdgeBlock` utiliza dos proyecciones equivariantes
independientes por fuente de features a `rank` canales de cada irrep presente.
Acopla todos los pares angulares permitidos hacia las irreps de salida mediante
TP `uuu`, con pesos externos por arista. Una red radial por operación/par de
especies, sin biases, produce gates `tanh` por canal/camino y un gate escalar
para el término lineal de las features originales. Un `o3.Linear` mezcla los
caminos hacia las multiplicidades de salida. Se suman las fuentes node/edge
como en el readout actual; no se cruzan fuentes distintas.

**Tech Stack:** Torch/e3nn/MACE/Lightning ya instalados; Python >=3.9.

## Decisiones y límites

- Se elige rango bajo, no una segunda arquitectura canal a canal independiente.
  El TP interno es canal a canal **después** de las proyecciones: factorización
  del mapa bilineal, no restricción a los canales originales.
- `rank=4` y ancho radial 32 son valores iniciales experimentales; `rank` es
  configurable. Todos los caminos angulares alcanzables siguen representados,
  pero compartir proyecciones reduce la capacidad frente al TP completo.
- No se exige igualdad de salida/gradientes con el readout anterior. Sí se
  exige O(3), corte, simetría del ensamblado y backward finito.
- `edge_feats_irreps` debe contener escalares **0e**, y se requieren radiales
  pareadas. Se promedian ambas direcciones antes de los gates; radial cero
  implica bloque cero incluso al reutilizar features de nodos no nulas.
- `MatrixMACE` rechaza la combinación con `HamiltonianLocalContext`: concatena
  contexto de especies/nodos después del envelope y no garantiza radial cero.
  El guard inspecciona módulos reales, incluidos readouts personalizados.
- La simetría de intercambio corresponde al `E3nnIrrepsMatrixBlock` existente,
  no a imponer un bloque heteronuclear internamente simétrico.
- Selección por `edge_block_readout` existente; añadir al final de la firma
  Lightning `edge_block_readout_kwargs=None` para pasar `rank` al mecanismo
  `edge_operation_kwargs` ya existente. Defaults y checkpoints viejos intactos.
- Objetivos/parámetros/tiempos CPU no prueban precisión ni ahorro de VRAM.

## Global Constraints

- `/home/christian/repositorios/graph2mat`, `g2m-v2`, checkout actual.
- No otro worktree, dependencias nuevas, GPU, fonones, autorun, colas, datasets
  originales, entrenamientos ajenos, push o merge. Cambios ajenos preservados.
- APIs previas, checkpoints y Python >=3.9 preservados. Commits locales propios.
- La orden de ejecutar el prompt delega la elección de variante y el plan:
  se documentan aquí antes del código, sin solicitar otra aprobación del alcance.

## Review Focus

1. Paridad impar en radiales: rechazar 0o antes de romper O(3).
2. Múltiples fuentes/orden de kwargs: node y edge con irreps distintas.
3. Empty batches, multiplicidades repetidas, outputs inalcanzables: errores
   explícitos o ceros estructurales, sin reshape ambiguo ni truncamiento.
4. Gate radial cero con nodos no nulos: no introducir colas constantes.
5. Checkpoint Lightning/default anterior y coeficientes C1/C2: restauración
   estricta y reconstrucción, sin claves nuevas en la arquitectura por defecto.

### Task 1: operación, selección y contratos CPU

**Files:** `bindings/e3nn/modules/edge_operations.py`,
`models/mace.py`, `tools/lightning/models/mace.py` dentro de `src/graph2mat/`;
`bindings/e3nn/modules/tests/test_low_rank_edge_readout.py` y
`tools/lightning/tests/test_low_rank_edge_model.py`; este plan/informe y roadmaps.

**Interfaces:** consume `irreps_in`, `irreps_out`, `edge_feats_irreps` y tuples
`node_feats`/`edge_messages`/`edge_feats`; produce coeficientes `(n, irreps_out.dim)`.
`_data_get_edge_args=("edge_feats",)` alimenta radiales por la ruta existente.

- [x] Escribir tests RED de export/contrato antes del código, incluyendo un
  caso escalar con pesos fijados a mano y gates radiales independientes.
  Ejemplo: `assert hasattr(e3nn, "E3nnLowRankEdgeBlock")` falla antes del cambio;
  una vez implementado, el caso escalar contrasta una fórmula literal, no el TP.
- [x] Ejecutar tests nuevos con CPU forzada y observar fallos de feature ausente.
- [x] Implementar proyecciones, instrucciones `uuu`, gates y selección opcional;
  validar rank entero positivo, radiales 0e y dimensiones/fuentes/tuples.
- [x] Ejecutar contratos O(3), inversión, cortes/continuidad, intercambio,
  periódicos homo/hetero s/p/d/f, C1/C2, layouts y backward; ampliar RED→GREEN
  para integración MatrixMACE, checkpoints y defaults.
- [x] Medir parámetros y forward/backward/paso CPU pareados de la operación y
  modelo completo usando fixtures propias; warmup, mediana y dispersión, hashes,
  versiones y carga no aislada explícitos. No entrenar datos del usuario.
- [x] Escribir protocolo GPU y precisión con métrica primaria previa, semillas
  0/1/2, splits congelados y presupuesto idéntico; sin ejecutar GPU.
- [x] Ejecutar suite CPU completa (`python -m pytest -q`), tests nuevos en
  Python 3.9, Ruff/mypy-core, compilación py39 y `git diff --check`.
  Expected: exit 0; registrar conteos reales, warnings y omisiones.
- [x] Commit local propio; `task-done` repite los tests enfocados; revisión
  independiente única del rango completo de este goal y correcciones RED→GREEN
  si aparecen problemas importantes. No merge/push.

## Registro

- Base: `7225ca4`. Worktree solo con untracked ajenos, preservados.
- Diseño/contexto inspeccionados; referencia enfocada en ejecución.
- Ruling: ejecutar el plan localmente aunque el servicio del goal figure limitado
  por cuota; no cambiar su objetivo ni confundir implementación CPU con evaluación.
- Source final: 61 enfocadas en Python 3.12/3.9 pasan; full CPU final
  2251 pasan/2 omitidos/28815 warnings/751.75 s, exit 0. Ruff/mypy-core,
  compilación py39 y diff checks verdes. Revisión: un Important corregido
  mediante guard de contexto y cinco regresiones RED→GREEN; ningún Minor aplazado.
- Producto/tests `5d607b3`. `task-done` real, rango `7225ca4..5d607b3`:
  42 pasan/2715 warnings/37.99 s, exit 0. Documentación y resultados en commit
  separado. Subplan CPU cerrado; GPU/AMP CUDA/precisión científica pendientes.
