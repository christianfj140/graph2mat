# Revisión de los fixes de Claude: Lightning y métricas — 2026-10-22

## Objeto y procedencia

Se revisaron las correcciones posteriores al HEAD original `a66b46cb0955ca344693d50737c5e9952fc257fc`, separándolas de la auditoría original ya cerrada. El punto de partida de la revisión fue `17e246f` más los ocho archivos con cambios de Claude que el agente coordinador congeló. El trabajo se realizó exclusivamente en `/tmp/graph2mat-codex-review-20261022-f80m50ec/source`, mediante `review-python`, con CUDA oculta y dos hilos OMP/BLAS. No se editaron fuentes del checkout compartido ni se ejecutaron GPU, SCF, campañas, instalaciones o commits.

También se inspeccionó el commit posterior `2248f520dc8fe03326dac1e5d51550373ac74a2b`, porque modifica la misma recarga de etapas. Sus hooks de evaluación se reprodujeron temporalmente en la copia para comprobar la nueva regresión y después se eliminaron. Se incorporó su guardia de error de `output_scaleshift` sin datamodule. La guardia DDP de ese commit ya estaba presente entre los cambios iniciales congelados y se conservó.

Los cambios finales de esta área están congelados en nueve archivos. El directorio de evidencia es `/tmp/graph2mat-codex-review-20261022-f80m50ec/lightning_metrics/`; contiene `before_hashes.json`, `changed-files.json` y `changes-from-review-snapshot.patch`. El último parche se calcula contra la copia inicial, no contra el checkout original que Claude siguió modificando.

## Dictamen sobre las correcciones revisadas

| Cambio de Claude | Dictamen y comprobación |
|---|---|
| `81b18ca`: scheduler sin optimizador declarado | Correcto para el caso reproducido. La CLI conecta el scheduler al optimizador del modelo y conserva sus grupos MACE. Los tests verifican cosine hasta LR cero, LR constante sin scheduler, rechazo de `optim_lr` por etapas incompatible y rechazo de reanudación histórica sin estado de scheduler. Ahora una configuración antes ignorada sí modifica el entrenamiento: sus curvas antiguas no deben reinterpretarse como entrenamientos cosine. |
| `2caeb69`: globs ordenados | Correcto en el helper y sus consumidores de split. Se verificaron directorios y ZIP con orden de creación distinto. La semilla y la fracción predeterminadas se conservan; ordenar entradas puede cambiar la pertenencia al split frente al antiguo orden arbitrario del filesystem. No se afirma identidad de splits históricos. La comprobación de base en el wrapper MACE corresponde a la revisión del otro propietario de ese archivo. |
| `6509fad`: writer y métricas con predicción H-only y etiquetas H/S | Correcto en los casos comprobados. Los tests reales del Trainer verifican S de referencia en la matriz escrita, metadata explícita, rechazo de colisión de archivos y métricas H-only finitas. No se extrapola esta comprobación a todos los formatos, políticas y spins. |
| `75e75d2`: gauge espectral con columnas no S-ortonormales | Corrige M2. El código usa `G=CᵀSC` real. La reproducción pública con autovalores a ambos lados del floor da pérdida total `8.106631183039924e-28`, frente a `0.08` original; gradiente máximo `7.804105207100576e-12`, frente a `60` original. El floor permanece igual. |
| `b52c73b`: CSV por muestra en DDP | Corrige L2 mediante un archivo por rango. Se comprobó con dos procesos CPU reales: la unión de `ddp_samples.rank0.csv` y `ddp_samples.rank1.csv` contiene exactamente las cuatro identidades y errores `[1,4,9,16]`; ambos rangos mantienen pérdida global `7.5`. Los consumidores deben reunir esos archivos. No es un único CSV global. |
| Guardia DDP inicial y `2248f52` | Se conserva el rechazo explícito de etapas que congelan parámetros sin `find_unused_parameters=True`. La guardia se prueba para configuración permitida, rechazada y un solo proceso. Los cuatro controles DDP numéricos finales comprueban agregación de gauge; no certifican entrenamiento completo con congelación bajo DDP. |
| Hooks de evaluación añadidos por `2248f52` | Incorrectos: fuerzan la última etapa de la receta en lugar de la etapa del checkpoint. También sustituyen la pérdida predeterminada antes de haber entrenado. Se eliminan ambos hooks y se restaura la pérdida guardada durante `on_load_checkpoint`. La guardia de scaleshift sin datamodule se conserva y se prueba. |

## Defectos que seguían presentes y correcciones finales

### M1 — P1: un único gauge de energía para ambos spins

**Antes:** `gauge_invariant_mse` ajustaba un desplazamiento por canal H. Los nuevos oráculos CPU dieron pérdida cero para errores `(+S,-S)`, aunque introducen una separación física entre spins. La ruta multitarget también dio cero donde el oráculo exige `0.68`.

**Cambio:** `core/data/metrics.py` suma el numerador sobre los canales H y divide por `n_h` veces la norma ponderada de S. Así obtiene un único escalar por estructura. El mismo escalar se resta a ambos spins; no se añade ninguna opción ni se conserva la alineación independiente como gauge físico. La pérdida, sus gradientes y el término energético reutilizan los deltas corregidos.

`tools/lightning/model.py` cuenta `gauge_shift_mean_abs` por estructura, sin multiplicar por el número de spins. Las fixtures de época y DDP ahora usan un desplazamiento común y errores de canal distintos. Se actualizó además el oráculo de una fila ausente: errores de spin `[1,2]` se alinean a `[-0.5,0.5]`, no desaparecen. La semántica de estructuras totalmente NaN permanece intacta.

**Prueba:** `M1-red.log` registra los dos fallos esperados; los tests nuevos comprueban pérdida y gradiente para separación opuesta, desplazamiento común y la ruta multitarget. Los controles de épocas, física y cuatro casos DDP pasan en los gates posteriores. El valor esperado del shift DDP con dos estructuras es `15`, incluida la variante spin2.

### M3 — P2: imagen mínima válida para celdas no reducidas

**Antes:** redondear fracciones pierde una pareja separada `0.04 Å` tras el cambio unimodular `[[100,1,0],[99,1,0],[0,0,1]]`, aunque la red y las posiciones cartesianas son idénticas. `M3-red.log` conserva el fallo `[] != [(0,1)]`.

**Cambio:** se usa `ase.geometry.geometry.general_find_mic`, con la misma periodicidad completa que aplicaba la implementación anterior cuando la celda era invertible. Esta ruta reduce la celda mediante Minkowski. No se añade ninguna dependencia: ASE ya es obligatoria.

La llamada habitual `ase.geometry.find_mic` también falló en este contraejemplo porque su atajo naive aceptó un vector de longitud aproximada `4 Å` en la base no reducida. Esa tentativa y su fallo están registrados en `M1-M3-green.log`; el parche final usa expresamente la ruta general. La regresión exige la misma pareja y el mismo rechazo por umbral en ambas representaciones de la red. No cambia la política histórica de periodicidad ni la detección de celdas variables.

### L1 — P2: restaurar la pérdida de la etapa guardada sin alterar el registro DDP

**Antes:** los checkpoints perdían la función de pérdida activa al recargarse. La corrección posterior de Claude aplicaba siempre la última etapa, lo cual falla para checkpoints tempranos. Las pruebas numéricas con esos hooks dieron MSE `24.353024500801457` donde el MAE de la etapa guardada era `4.606046676635742`. Antes del fit, una evaluación que debía dar MSE por bloques `13` devolvía Huber `4`.

**Cambio:** `on_load_checkpoint` recupera la pérdida y sus kwargs a partir del nombre de la etapa guardada. No congela, descongela ni modifica optimizadores. Deja el índice activo en `None` para que el hook de fit aplique de nuevo la política de parámetros y el LR después del registro DDP. Esto también cubre cargar sobre una instancia ya utilizada.

Los hooks `on_validation_start` y `on_test_start` que imponían `stages[-1]` quedan eliminados. Un modelo que todavía no entrenó conserva su pérdida predeterminada. Un checkpoint que declara una etapa ausente o ambigua en su receta se rechaza con un error claro. El fixture que fabrica un checkpoint anterior al soporte de etapas ahora elimina también `active_training_stage`; conservar ese campo mientras se borraba la receta producía metadata contradictoria, no un checkpoint legacy fiel.

**Prueba:** `L1-red.log`, `L1-reuse-red.log`, `2248-evaluation-red.log` y `L1-missing-stage-red.log` registran los fallos previos correspondientes. Los tests finales usan `Trainer.validate` y `Trainer.test` reales sobre checkpoints tempranos y finales, contrastan la pérdida con un oráculo numérico obtenido de las predicciones y comparan pesos exactamente. También comprueban preservación de `requires_grad` al cargar, reaplicación posterior de políticas, receta contradictoria, evaluación antes del fit y reanudación del checkpoint físico.

### L4 — P2: impedir selección mediante métricas obsoletas

**Antes:** después de cambiar de MSE a `block_type_mae`, el diccionario seguía conteniendo los acumuladores `val_node_rmse` y `val_edge_rmse`, aunque ya no se actualizaban. El selector aceptaba esos objetos reiniciados y podía publicar `hp_metric=0` o conservar un monitor de la época anterior.

**Cambio:** el selector exige `Metric.update_called` en la validación actual. Si la pérdida deja de producir la métrica seleccionada, lanza `ValueError` y enumera las métricas disponibles. Se conserva el cálculo independiente existente de `val_matrix_rmse`; no se introduce cómputo adicional ni se cambia la selección predeterminada.

**Prueba:** `L4-red.log` contiene dos casos que no lanzaban el error esperado. Los tests finales pasan para `val_node_rmse` y `val_edge_rmse`, preservan la igualdad entre `hp_metric` y la selección cuando la métrica existe y conservan los tests de batching y de monitor de CLI.

## Archivos y verificación

Archivos modificados respecto a la copia inicial:

- `src/graph2mat/core/data/metrics.py`.
- `src/graph2mat/core/data/tests/test_metrics.py`.
- `src/graph2mat/bindings/torch/tests/test_multitarget_loss.py`.
- `src/graph2mat/tools/lightning/model.py`.
- `src/graph2mat/tools/lightning/tests/test_training_stages.py`.
- `src/graph2mat/tools/lightning/tests/test_selection_metric.py`.
- `src/graph2mat/tools/lightning/tests/test_gauge_epoch_metrics.py`.
- `src/graph2mat/tools/lightning/tests/test_ddp_cpu.py`.
- `src/graph2mat/tools/lightning/tests/test_physical_training.py`.

Los logs registran los resultados y las advertencias sin ocultarlas:

| Gate | Resultado |
|---|---|
| Gate amplio de 19 archivos de métricas, física, multitarget, Lightning, scheduler, helper, split y callbacks | `798 passed, 2 skipped, 2835 warnings`, `38.39 s`, salida 0; `focal-verified.log`. Fue anterior al último endurecimiento de metadata contradictoria y su test específico. |
| Estado final: etapas, procedencia de checkpoint, selección y regresión de checkpoint físico/legacy | `44 passed, 2 skipped, 723 warnings`, `6.98 s`, salida 0; `L1-L4-final.log`. |
| Agregación gauge/espectral con dos rangos CPU | `4 passed, 52 warnings`, `20.82 s`, salida 0; `DDP-gauge-final.log`. |
| Sonda de CSV DDP real | Salida 0; cuatro muestras exactas en la unión de dos archivos; `ddp_callback_result.json` y `ddp_callbacks.log`. |
| Sonda pública de proyección espectral con floor | Salida 0; `spectral_pipeline.log`. |
| `git diff --check` | Correcto. |
| Ruff en el wrapper | No disponible: `No module named ruff`; `ruff.log`. No se instaló ningún paquete. |

La primera ejecución amplia tuvo `793 passed`, `2 skipped` y un fallo en el fixture legacy contradictorio; se conserva en `focal-final.log`. Su causa, la corrección del fixture y el rechazo explícito final están descritos en L1. Los logs intermedios RED no se presentan como resultados finales verdes. Las dos omisiones del gate de selección corresponden a la combinación expresamente no soportada de `block_normalized_huber` con etiquetas de dos componentes.

No se ejecutó la suite general desde esta área; corresponde al coordinador después de integrar todos los propietarios. No se modificaron `models/mace.py`, el wrapper Lightning MACE, herramientas de análisis/SIESTA ni las fuentes del checkout original. No se corrigió L3 dtype: sigue siendo la limitación estándar de construcción/carga ya documentada. Tampoco se modificó la semántica descartada de per-structure/NaN.

## Instrucción de integración

El merge automático de tres versiones puede volver a introducir los hooks de `2248f52`, porque no existían en la base inicial de esta revisión. **Deben eliminarse ambos hooks que fuerzan `self._validated_training_stages[-1]` y los tres tests de Claude que fijan esa conducta.** Deben conservarse la guardia DDP, la guardia de scaleshift y los tests numéricos nuevos de checkpoint temprano/final y evaluación previa al fit. Los hashes de `changed-files.json` identifican el estado final de esta copia; no se editarán más fuentes de esta área después de su entrega.
