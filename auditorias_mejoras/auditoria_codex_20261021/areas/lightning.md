# Auditoría de Lightning, checkpoints y callbacks — 2026-10-21, cierre 2026-10-22

## Alcance y resultado

La rama integra objetivos por etapas, métricas agregadas por época, selección de checkpoints, entrenamiento en coeficientes y callbacks para matrices y errores por estructura. Esta revisión comprueba que las métricas y los checkpoints conservan su significado al cambiar de etapa, distribuir las muestras y volver a cargar el modelo.

El objeto auditado es exclusivamente HEAD `a66b46cb0955ca344693d50737c5e9952fc257fc`, con base `origin/g2m-v2=a5e81345c8494d6cc1c7f9c917b46467507d3b97` y sus 267 commits. El cierre usa la copia congelada `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/source`. No incorpora los commits ni los cambios sin guardar que aparecieron después en el checkout compartido. Las ubicaciones de código que siguen corresponden al HEAD original.

Se confirman **tres defectos operativos P2**: dos introducidos dentro de los 267 commits y uno heredado de la base. Se registra además una limitación heredada de precisión al cargar checkpoints float64; no se contabiliza como regresión ni como incumplimiento de un contrato adicional de Graph2Mat que no se ha encontrado.

La revisión es de solo lectura. Las reproducciones finales escriben únicamente en `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/finish_lightning/`. No se modificaron fuentes, tests, configuraciones, colas, instalaciones ni datasets. No se usó GPU ni se lanzaron entrenamientos reales: solo tres pasos CPU en total sobre una estructura sintética, validación/test y una validación distribuida CPU de cuatro estructuras.

## L1 — P2, heredado: el checkpoint no restaura la pérdida de la etapa activa

**Ubicación:** `src/graph2mat/tools/lightning/model.py:1137`, `:1144`, `:926` y `:1074`; consumidor público `src/graph2mat/bindings/torch/load.py:64`.

**Qué hace y escenario:** `training_stages` puede sustituir la pérdida y sus parámetros durante el fit. `on_save_checkpoint` guarda tanto la receta como `active_training_stage`. Tras entrenar una etapa con `elementwise_mae`, cargar ese checkpoint mediante `load_from_lit_ckpt` y llamar a `Trainer.validate` o `Trainer.test` es un flujo admitido.

**Problema:** el constructor inicializa `loss_fn` con la pérdida predeterminada de los hiperparámetros. `on_load_checkpoint` no aplica `active_training_stage`; los hooks que aplican etapas solo se ejecutan durante fit. La evaluación independiente del checkpoint vuelve a `elementwise_mse`, aunque la etapa guardada usase MAE.

**Reproducción y oráculo:** `finish_lightning/probe_checkpoint_stage.py` hace un paso CPU de un MACE pequeño, con LR cero, dos átomos de un orbital s, etiquetas simétricas y sin baseline. Guarda el checkpoint y compara el modelo vivo con el recargado. Ambos `state_dict` y ambas predicciones son exactamente iguales. El resultado esperado es conservar la función y el valor de la pérdida de la etapa guardada:

```text
checkpoint.active_training_stage = mae_stage
checkpoint.hyper_parameters.loss = elementwise_mse
modelo vivo:      elementwise_mae, val_loss = test_loss = 4.054138978322347
modelo recargado: elementwise_mse, val_loss = test_loss = 17.018317178497075
state_dicts iguales = True; predicciones iguales = True
```

Los resultados completos están en `finish_lightning/checkpoint_stage_result.json` y `checkpoint_stage.log`. La reproducción previa, limitada a validate, también queda conservada en `lightning/probe_checkpoint_stage.py`.

**Origen:** heredado de `origin/g2m-v2`. `1a131f1c21109fb928b356a1748834c32853c477` introduce las etapas, la escritura de su nombre y su aplicación exclusiva en fit; ya es ancestro de la base. No pertenece a los 267 commits. Los cambios posteriores no corrigen esa omisión.

**Consecuencia:** `val_loss` y `test_loss` después de recargar no son comparables con los registrados durante la etapa. Cambian tanto la definición como las unidades de MAE a MSE sin que cambien las predicciones. Este caso no demuestra corrupción de pesos ni de la inferencia de matrices.

**Propuesta mínima:** al cargar, restaurar la pérdida y `loss_kwargs` correspondientes al nombre de etapa guardado; mantener el recálculo de etapas por época al reanudar fit. La restauración para evaluación no necesita modificar optimizadores. Añadir una regresión de ida y vuelta que compare pérdida, validate y test; los tests actuales de etapas comprueban la aplicación manual, pero no esta recarga.

## L2 — P2, introducido: el CSV por estructura descarta las muestras de los demás rangos DDP

**Ubicación:** `src/graph2mat/tools/lightning/callbacks.py:343` y `:344`; contrato del callback en `:252` y `docs/tutorials/cli/siesta_MDaccel.rst:374`.

**Qué hace y escenario:** `SamplewiseMetricsLogger` promete una fila de métricas por estructura. En DDP cada proceso recibe su partición del dataloader. El escenario probado usa dos procesos CPU, cuatro muestras, sin shuffle, sin padding y con una muestra por batch.

**Problema:** `_on_batch_end` retorna en todo rango distinto de cero antes de calcular o recoger sus filas. No hay recopilación entre rangos. Evitar escrituras concurrentes deja un CSV válido pero incompleto, sin aviso de que contiene solo una partición.

**Reproducción y oráculo:** `finish_lightning/probe_ddp_callbacks.py` usa el `Trainer` real con Gloo y dos rangos. Las referencias escalares son cero y las predicciones son 1, 2, 3 y 4. Ambos rangos registran correctamente la pérdida global `(1+4+9+16)/4 = 7.5`; por tanto, el fallo del CSV no se confunde con un fallo de reducción de la pérdida.

```text
Esperado: sample_0, sample_1, sample_2, sample_3, cuatro filas.
Real:     sample_0, sample_2, dos filas, errores 1.0 y 9.0.
val_loss en rank0 = val_loss en rank1 = 7.5.
```

Evidencia: `finish_lightning/ddp_samples.csv`, `ddp_callback_rank0.json`, `ddp_callback_rank1.json` y `ddp_callbacks.log`. No hay muestras duplicadas por el sampler que puedan explicar las ausentes.

**Origen:** introducido en `b7c612dea9ba4dfccca71989a3af797095ae4448`, dentro de los 267 commits; confirmado mediante blame de las líneas exactas. `0f3529f` añade avisos cuando falla una métrica, pero no introduce este retorno por rango.

**Consecuencia:** los análisis por muestra omiten silenciosamente `1−1/world_size` de las estructuras en una partición equilibrada. Aquí se pierde la mitad; su media de errores es 5 en vez de 7.5. La métrica global del Trainer permanece correcta en este control.

**Propuesta mínima:** recoger las filas calculadas por todos los rangos antes de escribir una sola vez en el rango cero, conservando la identidad de cada muestra. Si el callback aún no admite esa operación, rechazar expresamente DDP evita producir un informe incompleto. Una regresión con un número de muestras divisible por los rangos debe exigir las cuatro identidades. Esta corrección de software no requiere abrir la campaña multi-GPU 3.13, excluida por el usuario.

## L4 — P2, introducido: cambiar de pérdida permite seleccionar checkpoints con una métrica de la época anterior

**Ubicación:** `src/graph2mat/tools/lightning/model.py:648`, especialmente `:654`; invocación en `:1095` y `:1100`; traslado del monitor del checkpoint en `src/graph2mat/tools/lightning/cli.py:25`.

**Qué hace y escenario:** `selection_metric="val_edge_rmse"` permite seleccionar modelos mediante una métrica de validación. La configuración está aceptada y cubierta por `test_selection_metric.py:92`. La reproducción usa dos etapas consecutivas: `elementwise_mse` y después `block_type_mae`, una época por etapa.

**Problema:** la primera pérdida produce `edge_rmse`, que crea un objeto persistente en `_epoch_rmse`. La segunda no produce esa estadística. `_log_selection_metric` comprueba solo que la clave exista en el diccionario, sin comprobar que se haya actualizado en esta validación. Registra `hp_metric=0` desde el objeto reiniciado, mientras `callback_metrics["val_edge_rmse"]` conserva el valor anterior. `ModelCheckpoint` consume ese valor obsoleto.

**Reproducción y oráculo:** `finish_lightning/probe_stale_selection.py` ejecuta dos pasos CPU sobre la misma estructura sintética y mide también el RMSE directamente a partir de las predicciones de cada época. Incluye un `ModelCheckpoint(monitor="val_edge_rmse", mode="min", save_top_k=1)` real:

| Época | Pérdida activa | RMSE de edges medido directamente | `val_edge_rmse` publicado | `hp_metric` | Conteo del acumulador |
|---|---|---:|---:|---:|---:|
| 0 | `elementwise_mse` | 2.3665695190 | 2.3665695190 | 2.3665695190 | 1 |
| 1 | `block_type_mae` | 2.0252730846 | **2.3665695190** | **0** | **0** |

**Esperado:** seleccionar época 1, que tiene menor RMSE, o detenerse con un error claro si esa métrica deja de estar disponible. **Real:** `best_model_path` conserva `epoch=0.ckpt` y `best_model_score=2.3665695190429688`. El checkpoint guardado confirma `epoch=0`. También mejora `val_matrix_rmse` de 7.93525 a 6.97874, por lo que no se trata de un empate de predicciones.

Evidencia: `finish_lightning/stale_selection_result.json`, `selected_checkpoint.json`, `selection_ckpt/epoch=0.ckpt` y `stale_selection.log`. La reproducción original mejorada permanece en `lightning/probe_stale_selection.py`; `probe_stale_selection_v1.log`, que usaba LR cero, no se utiliza como prueba de una selección incorrecta.

**Origen:** `1a0160f8c1fe3370592c1680053137346920d458`, dentro de los 267 commits, introduce la selección comprobando presencia en estos diccionarios. El soporte de etapas ya existía en la base; su combinación con este selector vuelve alcanzable el fallo.

**Consecuencia:** el entrenamiento puede elegir un checkpoint peor y publicar un cero de ajuste de hiperparámetros que no mide el error actual. No se necesita DDP ni un target físico especial.

**Propuesta mínima:** comprobar que la métrica seleccionada se haya actualizado en la validación actual y fallar si desaparece. Para mantener la selección por RMSE independiente de la pérdida entre etapas, calcular explícitamente el RMSE elegido desde las predicciones, como ya se hace para `val_matrix_rmse`. Cubrir el cambio de pérdida y exigir que el checkpoint seleccionado coincida con el oráculo medido directamente.

## L3 — Limitación heredada: cargar float64 bajo el dtype predeterminado float32 descarta precisión

**Ubicación:** `src/graph2mat/bindings/torch/load.py:64`. La misma llamada a `LitMACEMatrixModel.load_from_checkpoint` ya existe en `origin/g2m-v2`; los cambios dentro de los 267 commits no añaden una política de dtype al loader.

`lightning/probe_checkpoint_dtype.py` guarda un modelo construido en float64. Después cambia el dtype predeterminado a float32 y llama al loader público: todos los tensores flotantes del modelo recargado quedan en float32. El primer peso diferente tiene error máximo `1.6084631959678575e-08`. Aplicar `.double()` después conserva ese mismo error: los bits perdidos al cargar ya no se recuperan.

La comprobación final `finish_lightning/probe_dtype_prediction.py` carga el mismo checkpoint en float64 antes de construir el modelo y recupera exactamente todo su `state_dict`. Frente a cargar en float32 y convertir después a double, las predicciones difieren como máximo en `1.4600481890170158e-07` para nodos y `6.214526027115141e-08` para edges. Los resultados están en `dtype_prediction_result.json` y `dtype_prediction.log`.

Esto coincide con el mecanismo de Lightning 2.6.1 instalado: `pytorch_lightning/core/saving.py:167` construye la clase y `:189` ejecuta `obj.load_state_dict(...)` sobre ese objeto. No se ha encontrado en la API ni en la documentación de Graph2Mat un compromiso adicional de reconstruir automáticamente el dtype del checkpoint. Por eso se informa como límite reproducible de la carga, no como cuarto defecto introducido ni como prueba de error físico significativo. El contrato de escritura de matrices float64 no establece por sí mismo ese compromiso para los pesos del modelo.

Para reproducir pesos float64, el dtype de construcción debe coincidir antes de cargar; convertir solo después no basta. No se han medido bandas, autovalores ni observables físicos afectados por esta diferencia, y no se asignan unidades físicas a estos ejemplos sintéticos.

## Cobertura y límites

La revisión completa del área, iniciada antes de la interrupción de cuota, siguió el flujo de `tools/lightning/model.py`, `data.py`, `callbacks.py`, `cli.py`, `_helpers.py`, `split_guardrail.py`, el wrapper MACE y `bindings/torch/load.py`, con su diff y sus consumidores. El diff principal leído está conservado en `lightning/reviewed_main_diff.patch`. El cierre volvió a contrastar las líneas de carga, etapas, selección y callbacks contra la fuente congelada y su historia Git; `finish_lightning/origins.json` registra la pertenencia exacta de los tres commits de origen al rango de 267.

Se inspeccionaron especialmente las pruebas de etapas, selección, agregación de métricas, DDP CPU, procedencia del checkpoint, escritura de matrices, objetivos en coeficientes y multitarget. Las pruebas de etapas y selección existentes cubren sus rutas aisladas, pero no los casos L1 y L4. La validación DDP con pérdida global correcta no comprueba la integridad del CSV por muestra de L2. Las reproducciones son sondas de auditoría, no cambios incorporados a la suite.

Se descartó la hipótesis de que `MatrixWriter` fallase al escribir H a NetCDF/HSX por ausencia de Ef: `lightning/probe_writer.py` escribe y vuelve a leer los cuatro casos `h_only`/`h_and_overlap` × `.nc`/`.HSX` con el H esperado. Ese control no certifica todos los observables ni formatos. Tampoco se presenta como defecto la combinación `output_scaleshift` con un readout `ModuleList` que la API rechaza expresamente. Los problemas de gauge por spin y de proyección casi nula de S se documentan en `areas/metrics.md`; no se duplican aquí. La preparación y lectura de datos se documenta en `areas/data.md`.

Los cuatro scripts finales salieron con código 0 usando el wrapper `audit-python`: Torch `2.11.0+cu130`, Lightning `2.6.1`, CUDA oculta y OMP/BLAS con dos hilos. `finish_lightning/runtime.json` registra la ruta importada desde la copia congelada. No se ha repetido la suite completa, que pertenece a la evidencia general de la auditoría. No se evaluaron GPU, rendimiento, convergencia, SCF, independencia estadística de datasets ni precisión de bandas. Las matrices usadas aquí mantienen estructura, orden y base constantes; son controles de coherencia del software, no validación científica de un material.

Comandos de reproducción del cierre, ejecutados desde su directorio temporal:

```bash
/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/audit-python probe_checkpoint_stage.py
/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/audit-python probe_ddp_callbacks.py
/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/audit-python probe_stale_selection.py
/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/audit-python probe_dtype_prediction.py
```

**Dictamen:** corregir L2 y L4 antes de dar por fiables esos informes y esa selección en las configuraciones admitidas. Corregir también L1 para hacer comparable la evaluación independiente de modelos por etapas, señalando que el defecto es heredado. Estos resultados no establecen que las campañas registradas hayan usado las configuraciones afectadas ni autorizan relanzarlas.
