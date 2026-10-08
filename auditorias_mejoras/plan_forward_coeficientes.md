# Forward coefficient-only Graph2Mat/MACE — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL `superpowers:executing-plans`;
> ejecución directa con TDD; revisión independiente al cierre del goal amplio.

**Goal:** Exponer una ruta opt-in realmente sin labels/reconstrucciones en
Graph2Mat y MatrixMACE, consumible por el objetivo CPU ya verificado.

**Architecture:** `forward(..., coefficients_only=False)` conserva exactamente
las salidas actuales por defecto. En modo True Graph2Mat delega en
`forward_coefficients`, conserva orden de módulos/filas y omite flatten, merge
de labels y resorting. Devuelve el diccionario node/edge/metadata. MatrixMACE
propaga el flag por llamada y agrega solo coeficientes (sum/mean), sin modificar
configuración ni state_dict. GC reentrante/long_range solicitan ruta matricial.

**Tech Stack:** Python>=3.9, Torch/e3nn/MACE/PyG y pytest existentes.

**Spec:** `roadmap_precision_rendimiento.md`, Task1; prerrequisitos `41a03f4`
y `48a9f0f`. Esta ruta no completa la integración collate/Lightning ni la medida GPU.

## Global Constraints

- Solo Graph2Mat, `g2m-v2`, mismo checkout; no datos/colas/procesos/entornos ajenos.
- Default `coefficients_only=False`; return_coefficients anterior inalterado.
- No buffers persistentes/opciones de arquitectura nuevas; checkpoints intactos.
- Mismas matrices/gradientes, componentes y orientación; no promesa bitwise.
- No GPU ocupada; tests CPU aislados. Sin revisión por tarea ni cierre del goal.

## Review Focus

- Layout fast path y fallback agrupado: mismo orden sin resortar matrices.
- Homonucleares dirigidos y heteronucleares signed: coeficientes de cada módulo.
- Ghost/vacíos/varias componentes: no inventar operaciones ni perder el eje.
- ModuleList: sum/mean conserva el modelo, no promedio de pérdidas de readouts.
- Checkpointing/long_range: error explícito en ruta directa, fallback en futuro caller.

### Task 1: forward integrado sin matrices

**Files:**
- Modify: `src/graph2mat/core/modules/graph2mat.py`.
- Modify: `src/graph2mat/models/mace.py`.
- Create: `src/graph2mat/core/data/tests/test_coefficient_only_forward.py`.
- Modify: `src/graph2mat/tools/lightning/tests/test_models.py`.
- Modify: `docs/data_loading.rst` y registro/roadmap de este goal.

**Interfaces:** consume `forward_coefficients`, metadatos/Q/grupos existentes,
`CoefficientMatrixMSEProjector`, `coefficient_matrix_mse` y layout CPU de collate.
Graph2Mat produce dict node/edge/metadata si coefficients_only; MatrixMACE produce
node_coefficients/edge_coefficients/coefficient_metadata, sin node/edge_labels.

- [x] Step 1: RED Graph2Mat real para point_type/basis_shape/max, sym/dir,
  1/2 componentes, float32/64/cast, layout/no layout. Referencia `nodes,edges =
  readout(data,node_feats=features)`; reconstruir coeficientes fuera del tramo
  certificado y comparar loss/grad a 5e-6/1e-12. En tramo certificado parchear
  coefficients_to_block y resorters para fallar. Expected: TypeError por nuevo
  keyword coefficients_only, no alterar tests para aceptar reconstrucción.
- [x] Step 2: propagar flag a las dos rutas internas y _evaluate_block; este
  devuelve `(None,func.forward_coefficients(**kwargs))` en el nuevo modo.
  Saltar inmediatamente tratamiento matricial tras cada evaluación; devolver
  dicts antes de concatenación/resorting, incluidos vacíos. Añadir dict a las
  anotaciones de retorno. Expected: tests Graph2Mat GREEN y mypy-core verde.
- [x] Step 3: RED MatrixMACE real con fixtures de test_models (same_basis/DZP/f,
  molécula/cuadrado/cadena periódica), single/ModuleList sum/mean, 1/2 componentes.
  Comparar el objetivo CPU nuevo con MSE literal del forward matricial y
  gradientes de parámetros; prohibir reconstrucción durante el forward nuevo.
  Expected: flag llega incorrectamente a MACE o aún reconstruye.
- [x] Step 4: parámetro por llamada en MatrixMACE.forward, _checkpointed_readout
  y _readout. `need_coefficients = self.return_coefficients or coefficients_only`;
  en ModuleList añadir coeficientes y no labels, agregar con helpers existentes.
  En modo solo-coeficientes output conserva mace_out/context y omite labels.
  Rechazar GC/long_range antes del backbone. Expected: GREEN y default sigue
  devolviendo matrices con las mismas claves, strict loading sin keys nuevas.
- [x] Step 5: gates ampliados y floor3.9, luego suite source CPU completa,
  Ruff/mypy/compileall39/diff. Comando CPU exacto de plan_mse_irreps_targets_cpu
  con selección de nuevos tests y controles de coeficientes. Expected: exit0.
- [ ] Step 6: documentar ruta y límites de Lightning/fallback; commit explícito
  y task-done real con tests nuevos Graph2Mat y MACE. No cerrar 5.3 ni goal.

## Pre-flight y decisiones

- Una tarea acoplada: MACE consume exactamente el dict producido por Graph2Mat.
- No añadir forward independiente ni extra config de constructor: flag por
  llamada evita cambiar defaults/checkpoints y permite al caller elegir fallback.
- MSE referencia usa labels reales del forward anterior, no coeficientes
  antiguos reproyectados, que contienen redondeo adicional de Q32→64.
- NaN parcial/fórmula sin cert/objetivo no MSE/callback que exige labels se
  resolverán en el caller de training, no mediante outputs silenciosamente incompletos.
