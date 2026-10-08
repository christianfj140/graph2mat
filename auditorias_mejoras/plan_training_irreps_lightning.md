# MSE irreps opt-in en Lightning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL `superpowers:executing-plans`;
> ejecución directa TDD; una revisión independiente al final del goal amplio.

**Goal:** Entrenar MatrixMACE sin reconstruir matrices en los batches certificados,
preservando pérdida/gradientes, logging global, callbacks y restauración.

**Architecture:** `coefficient_training=False` explícito en base Lit y LitMACE.
En on_fit_start, tras restauración y scaleshift, snapshot CPU runtime para el
MatrixDataModule. training_step elige ruta antes del backbone, no doble forward.
La ruta nueva entrega solo loss; los callers que precisan labels hacen fallback.
Las métricas TorchMetrics existentes reciben SSE/counts físicos precomputados
mediante su update envuelto, manteniendo cache/reset/reducciones DDP.

**Tech Stack:** Python>=3.9, Torch/e3nn/MACE/PyG/Lightning/TorchMetrics existentes.

**Spec:** `roadmap_precision_rendimiento.md` Task1; prerrequisitos kernel41a03f4,
targets48a9f0f, forwarded975eb y `plan_collate_irreps_cpu.md`.
No objetivos científicos nuevos ni campaña GPU/precisión en este lote.

## Global Constraints

- Solo Graph2Mat/g2m-v2/mismo checkout; no datos/colas/entornos ajenos ni fonones.
- Default False, mismo state_dict y checkpoint antiguo sin flag sigue False.
- Certificado real fp32/fp64 y elementwise_mse, edge_weight en SSE y count.
- MAE/Huber/gauge, GC/LR, verbosidad matricial, AMP reducido sin readout_fp32,
  datos/callbacks/transforms no certificados vuelven a matrices antes del backbone.
- Nunca generar targets desde CUDA dentro de training_step.
- Snapshot fijo durante fit, rebuild tras restore; estrategias restore-after-setup
  fallback. Si el usuario cambia Q/labels después del collate no es certificado.
- Factores targets readonly. No nueva caché/sampler/dependencia/export público.
- CUDA ocupado: no lanzamiento propio; corrección CPU no implica ganancia.

## Review Focus

- Batches/ranks sin operaciones/observaciones: cero conectado y schema idéntico.
- Loss por etapas y callbacks añadidos: reevaluar elegibilidad antes del backbone.
- No tomar snapshot de Q anterior al checkpoint/dtype efectivo.
- TorchMetrics cache/reset: actualizar con update envuelto, no nuevo método lateral.
- Callbacks/custom batch transforms no deben leer outputs incompletos ni targets viejos.

### Task 1: ruta de training y pruebas completas de contrato

**Files:** Modify `tools/lightning/model.py`, `tools/lightning/models/mace.py`,
`bindings/torch/data/coefficient_targets.py` bajo src/graph2mat.
Create `src/graph2mat/tools/lightning/tests/test_coefficient_matrix_training.py`;
Modify `test_ddp_cpu.py`, docs/data_loading.rst y roadmaps de este goal.

**Interfaces:** consume setter runtime/COEFFICIENT_TARGETS_ATTR, CPU projector,
MatrixMACE.forward(coefficients_only=True), coefficient_matrix_mse.
Produce flag de training único, losses/stats físicos consumidos por logging.
Extiende `_EpochMatrixLoss.update(..., matrix_stats=None)` y `_log_loss` /
`_log_stats(..., matrix_stats=None)` sin alterar llamadas/defaults anteriores.

- [ ] Step 1: RED training_step realMACE, graph periódico heterogéneo s/p/d/f,
  C1/2, group3, single/sum/mean, dtypef32/64. Referencia matrices reales antes
  de prohibir coefficients_to_block durante el paso nuevo. No usar targets
  antiguos reproyectados. Código central:
  ```python
  targets = CoefficientMatrixMSEProjector(model.model.matrix_readouts)
  batch = TypeLayoutCollate(loader.collate_fn, table, targets, symmetric)(samples)
  expected = ((out['node_labels']-batch.point_labels).square().sum()
      + 2*(out['edge_labels']-batch.edge_labels).square().sum()) / physical_count
  result = model.training_step(batch, 0)
  torch.testing.assert_close(result['loss'], expected, rtol=tol, atol=tol)
  ```
  Expected: constructor rechaza nuevo flag o training aún reconstruye.
- [ ] Step 2: flag explícito/base guarda hparams, LitMACE propaga sin kwargs
  arquitectónicos. Eligibilidad antes de model: exact elementwise_mse/float32–64,
  no GC/LR, diccionario CPU certificado transferido, no callback matricial/
  transform custom; AMP requiere readout_fp32. Aviso una vez por razón.
  ```python
  out = self.model(batch, coefficients_only=True)
  loss, physical = coefficient_matrix_mse({'node':out['node_coefficients'],
      'edge':out['edge_coefficients']}, targets, edge_weight)
  stats = {key:physical[key] for key in ('node_rmse','edge_rmse')}
  ```
  Expected: GREEN loss/grad, salida nueva solo loss, defaults con labels.
- [ ] Step 3: RED fallback MAE/Huber/gauge/NaN parcial/no targets/GC/LR/
  callback train y custom, AMP con/sin readout_fp32. Un spy sobre forward real
  mide una llamada, no hace mock de predicciones. Certificar mapas simetrizados
  contra Q efectivo; mapas ausentes/incompatibles solicitan fallback sin bloques.
  Expected: una sola llamada correcta, matrices en fallback, nuevas pruebas verdes.
- [ ] Step 4: RED métricas precomputadas con SSE literals [3,8],counts[3,2],
  siguiente batch [9,48],counts[1,3],edge_weight2 => loss124/14, node_rmse√3,
  edge_rmse√(56/5). Compute antes y después de update y reset, cache no vieja.
  Extender update y logging existentes sin materializar errores dense.
  Expected: misma agregación y esquema de entrenamiento sin SSE internos expuestos.
- [ ] Step 5: RED fit real MatrixDataModule, workers0/2, 2epochs, validación
  matricial y callbacks val; entrenamiento sin reconst probado dentro training_step.
  Snapshot after checkpoint probado restaurando Q fp32→fp64 distinto de init;
  estrategia restore-after-setup fallback, default checkpoint antiguo False.
  Epochs reset y loss por etapas vuelven a matriz cuando cambia el objetivo.
  Expected: optimizer avanza, callbacks conservados y losses iguales referencia.
- [ ] Step 6: RED/ GREEN dos ranks Gloo reales, incluidas especies ausentes y
  NaN por bloque; sin find_unused_parameters, todas las gradientes DDP conectadas.
  Contrastar pesos/pérdidas base vs candidato con semillas/datos idénticos,
  epoch global SSE/counts y reinicio. Expected: procesos exit0, no GPU.
- [ ] Step 7: focused, floorPython3.9/Torch2.4.1, suite source CPU completa,
  Ruff/mypy/compile39/diff. Expected: exit0, leer outputs; fallos no ocultos.
- [ ] Step 8: docs/registro, commit explícito y task-done con fichero nuevo y
  `test_ddp_cpu.py -k 'coefficient_matrix_training or irreps_training'`.
  Expected: exit0; Task1 general no cerrada sin medida GPU y revisión final.

## Diseño y alternativas

- Ruta integrada al training_step existente, no entrenador paralelo ni loss nueva
  pública. Reusar Metric.update permite cache/reset/DDP sin APIs duplicadas.
- No reconstrucción lazy para callbacks: añade complejidad y retención de tensors;
  fallback completo conocido antes de ejecutar el backbone es más pequeño.
- El goal permite concretar/executar subplanes sin preguntas rutinarias. No se
  extiende alcance a física/arquitecturas ni se promete aceleración no medida.
- on_fit_start precede restore en estrategias restore-after-setup: caso rechazado
  para la ruta rápida, comprobado en código instalado y documentación oficial.
