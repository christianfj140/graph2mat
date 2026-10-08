# Targets irreps en collate CPU — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL `superpowers:executing-plans`;
> TDD directo; revisión independiente solo al final del goal general.

**Goal:** Preparar targets equivalentes en los workers del loader de training,
sin copiar modelos y sin alterar batches/validación existentes.

**Architecture:** `TypeLayoutCollate` admite un snapshot CPU opcional y simetría
explícita. El datamodule recibe el snapshot por un setter runtime (no hparam);
solo train lo consume. El resultado vive fuera del store PyG, igual que layout,
y Lightning lo mueve con el helper recursivo existente, incluyendo listas.
Un string en ese atributo comunica fallback certificado no disponible.

**Tech Stack:** Python>=3.9, Torch/e3nn/PyG/Lightning existentes.

**Spec:** `roadmap_precision_rendimiento.md` Task1 y `plan_forward_coeficientes.md`.
No conexión del training_step en este lote ni medida GPU/precisión.

## Global Constraints

- Solo Graph2Mat/g2m-v2/mismo checkout, defaults/checkpoints intactos.
- Snapshot reconstruido por el caller tras restaurar Q/dtype; no modelo en workers.
- No CUDA ocupado ni entornos/datos/procesos ajenos; CPU aislada.
- Fallback solo `CoefficientMSEUnsupported`; entradas malformadas siguen fallando.
- Factores compartidos de snapshot son de solo lectura; no caché de datasets aquí.

## Review Focus

- No aplicar el layout simétrico a grafos dirigidos.
- Workers spawn/persistentes no retienen ni serializan módulos/parametrizaciones.
- PyG get_example conserva el contrato original y no concatena targets de módulos.
- Targets parciales NaN/complex solicitan fallback, nunca pierden información.
- Transferencia llega a tensores dentro de listas de términos, sin CUDA en workers.

### Task 1: collate y transferencia

**Files:** Modify `bindings/torch/data/coefficient_targets.py`,
`bindings/torch/modules/graph2mat.py`, `tools/lightning/data.py`,
`tools/lightning/model.py` bajo src/graph2mat; Create
`src/graph2mat/tools/lightning/tests/test_coefficient_target_collate.py`;
Modify docs/data_loading.rst y registros del goal.

**Interfaces:** `CoefficientMatrixMSEProjector(readout)(batch,nodes_ref,edges_ref)`;
`TypeLayoutCollate(collate,basis_table,coefficient_projector=None,symmetric=True)`;
`MatrixDataModule.set_coefficient_mse_projector(projector)`;
`COEFFICIENT_TARGETS_ATTR='_graph2mat_coefficient_mse_targets'` en módulo privado.
El atributo externo contiene dict node/edge o string de fallback.

- [x] Step 1: RED con datos PyG reales de processor, readout agrupado real y
  loaders workers0/2 spawn: sym/dir, point_type/basis_shape/max, C1/2. Calcular
  referencia con matrices reales y `elementwise_mse(edge_weight=2)`.
  En iteraciones repetidas, comparar pérdida/gradiente con targets del collate.
  ```python
  loader.collate_fn = TypeLayoutCollate(loader.collate_fn, table,
      coefficient_projector=CoefficientMatrixMSEProjector(readout), symmetric=symmetric)
  batch = next(iter(loader))
  loss, stats = coefficient_matrix_mse(readout(batch, node_feats=features,
      coefficients_only=True), getattr(batch, TARGETS_ATTR), edge_weight=2)
  ```
  Expected: TypeError por nuevo keyword; no GPU.
- [x] Step 2: implementar delegación después del collate; layout solo symmetric;
  catch Unsupported y guardar reason string, no capturar ValueError general.
  ```python
  try:
      targets = self.coefficient_projector(batch, batch.point_labels, batch.edge_labels)
  except CoefficientMSEUnsupported as error:
      targets = str(error)
  object.__setattr__(batch, COEFFICIENT_TARGETS_ATTR, targets)
  ```
  Expected: GREEN incluido NaN parcial y error de componentes no ocultado.
- [x] Step 3: RED setter train-only/default y transferencia profunda a dispositivo
  meta (sin GPU). Extender `_map_layout` a listas conservando su tipo y mover
  atributo aparte en `transfer_batch_to_device`, como layout existente.
  Expected: todos los terms/factors/targets trasladados; get_example original.
- [x] Step 4: RED complex labels/Q; rechazar con Unsupported antes de double,
  no cast que descarta imaginarios. Expected: fallback explícito, real sin cambios.
- [x] Step 5: selección nueva+MSE+orientation+workers previos, floor3.9, suite
  CPU completa, Ruff/mypy/compile39/diff. Expected: exit0 con outputs leídos.
- [ ] Step 6: documentar runtime/ownership/límites, commit explícito y task-done:
  `python -m pytest -q src/graph2mat/tools/lightning/tests/test_coefficient_target_collate.py`.
  Expected: exit0; sin cerrar5.3/goal ni atribuir ganancia GPU.

## Decisiones

- El goal autoriza diseño/ejecución sin preguntas rutinarias: este lote concreta
  la integración ya aprobada, no amplía su propósito ni añade dependencia.
- No nueva caché, hparam de datos ni export público; el caller training fija el
  snapshot después del restore. Estrategias restore-after-setup requieren fallback.
- El string de fallback evita exceptions de workers para casos no certificados,
  manteniendo las validaciones de shape/finite como errores reales.
