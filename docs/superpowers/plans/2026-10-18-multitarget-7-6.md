# Multitarget H/S/DM/EDM 7.6 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Ejecución Native ya elegida: implementación aquí y una revisión independiente final; sin commits ni worktrees.

**Goal:** Implementar objetivos conjuntos H/S/DM/EDM sobre un backbone, con semántica, pérdidas, máscaras y reconstrucción explícitas, preservando la ruta legacy.

**Architecture:** Reutilizar el readout multicomponente de MatrixMACE y un descriptor fijo versionado. Un único grafo aporta etiquetas empaquetadas y máscaras; vistas diferenciables separan los objetivos para calcular pérdidas y exportar matrices físicas. La física opcional reutiliza referencias certificadas de 4.4–4.5, no productos entre predicciones.

**Tech Stack:** Python >=3.9, NumPy/SciPy, sisl, Torch/PyG/Lightning, MACE/e3nn y pytest existentes. Sin dependencias nuevas.

**Spec:** `docs/superpowers/specs/2026-10-18-multitarget-7-6-design.md`, aprobado por el usuario («Apruebo», 2026-10-18).

## Global Constraints

- Solo `/home/christian/repositorios/graph2mat`, rama `g2m-v2`, checkout actual; preservar cambios anteriores y artefactos congelados.
- Solo CPU. Un paso sintético de Trainer está autorizado; no entrenamientos científicos, GPU, cola/autorun, nuevos datasets, otros proyectos ni fonones.
- Sin commits, cambios de rama, worktrees, instalaciones ni cambios de defaults legacy.
- `matrix_targets=None` conserva ruta, retornos y claves de pesos legacy; no exige nuevas claves en checkpoints antiguos.
- Nombres: `hamiltonian`, `overlap`, `density_matrix`, `energy_density_matrix`, orden H,S,DM,EDM. H y S obligatorios; DM/EDM opcionales. Versión 1; spin real 1/2, S escalar; conjunto completo de 4/7 columnas.
- Matrices directas, `sub_point_matrix=False`; sin H0/DM0 implícitas. H/EDM en eV, S adimensional, DM con ocupación sisl, sin factor2 inventado ni McWeeny.
- Geometría/base/orden orbital/celda/imágenes/spin/gauge compatibles. TSDE no suministra S: usar S real coincidente.
- Selección exige cada matriz incluso con peso0; coeficientes desconocidos no son ceros observados. Seleccionar máscara antes de aritmética; no NaN*0.
- Pesos explícitos finitos/no negativos y al menos uno positivo. Target activo sin observaciones en una muestra: error contextual.
- Pérdida H gauge y física usan S_ref. Reconstrucciones autocontenidas usan S_pred; no reparar SPD, recortar S ni recuperar coeficientes fuera del grafo.
- Rechazar ZIP nuevo-modo, long_range, hamiltonian_local_context, coefficient-only y etapas con pérdida single-target incompatible. Legacy conserva esas opciones.
- Sin endpoints HTTP multitarea nuevos ni afirmaciones de precisión, velocidad GPU o VRAM por tests sintéticos.

## Review Focus

- Dos layouts con igual anchura y distinta semántica: rechazar en batch, checkpoint y CLI (Tasks 1/2/4).
- Solo un spin DM desconocido: excluir esa entrada, no perder el otro spin; ceros explícitos siguen observados (Tasks 1/3).
- Target con peso0 y salida NaN: no evaluar su término ni contaminar gradientes activos (Task 3).
- Inferencia sin referencias y S_pred distinta de S_ref: no leer etiquetas ni reconstruir con S_ref a escondidas (Tasks 2/4/5).
- Callback legacy o formato que supone un Hamiltoniano: no emitir métricas de unidades mezcladas ni escribir parcialmente el batch (Tasks 4/5).

## Mapa de archivos y reutilización

- Crear `src/graph2mat/core/data/multitarget.py`: descriptor fijo, preparación/carga y selección de vistas de datos; NumPy/sisl, sin importar Torch.
- Modificar `core/data/processing.py`: parámetros opt-in y selección explícita de reconstrucción. No tocar los cambios previos de `sparse.py`; reutilizar CSR y construir clases sisl explícitas.
- Modificar `core/data/physical.py` solo para extraer/reutilizar lectura y auxiliares existentes, sin cambiar su contrato single-target.
- Modificar `bindings/torch/data/dataset.py`, `tools/lightning/data.py`, `tools/lightning/cli.py`: preparación, batching, inferencia de layout y enlace semántico.
- Modificar `models/mace.py`, `tools/lightning/models/mace.py`: una pasada de backbone y vistas nombradas; hparams y rechazos tempranos.
- Crear `bindings/torch/multitarget.py`: pérdidas y estadísticos por objetivo, reutilizando `core/data/metrics.py` y `physical_loss_terms`; no registro general de tareas.
- Modificar `tools/lightning/model.py`: pérdida opt-in, preflight, logging/denominadores y checkpoint; `bindings/torch/load.py`: restaurar descriptor en procesador.
- Modificar `tools/lightning/callbacks.py` y `tools/server/server_app.py`: selección/guards de consumo, sin ampliar endpoints.
- Crear tests vecinos indicados por tarea; reutilizar `core/data/tests/physical_fixtures.py`, `test_physical_dataset.py` y helpers de modelos/Trainer existentes.
- Documentación: `docs/data_loading.rst`, `docs/server.rst`, informe nuevo `auditorias_mejoras/informe_multitarget_7_6_cpu_20261018.md`, roadmap/traspaso y registros de este Goal.

Rutas de producto abreviadas arriba son relativas a `src/graph2mat/`. No extraer ni reestructurar subsistemas ajenos.

## Interfaces acordadas

- `MatrixTargetLayout(targets: tuple[str, ...], spin_components: int)`, dataclass inmutable en `core/data/multitarget.py`. Normaliza nombres; propiedades `n_components`, `slices: dict[str, slice]`; métodos `to_dict() -> dict`, `from_dict(value: dict) -> MatrixTargetLayout`. Serializa `version=1`, targets, spin, componentes/slices y unidades; valida todos al leer, no confiar en slices recibidos.
- Processor/DataModule/LitMACEMatrixModel añaden `matrix_targets: Optional[list[str]]=None`, `matrix_target_spin: Optional[int]=None`. Con targets activos el spin debe resolverse a 1/2 antes del readout; API directa lo declara, CLI de fit puede inferirlo de una referencia seleccionada. Predict sin etiquetas exige config/checkpoint, nunca infiere del test.
- El descriptor determina el total: `n_matrix_components=1` conserva el sentinel/default actual; en modo nuevo se resuelve al total. Otro valor explícito solo se acepta si coincide. `n_output_components` solo None o total; nunca omitir heads silenciosamente.
- `prepare_multitarget_data(processor, *, hamiltonian, overlap, density=None, energy_density=None, observation_masks=None, kpoints=None, kweights=None, physical_labels=None, data_cls=BasisMatrixData) -> BasisMatrixDataBase` y `load_multitarget_data(path, processor, *, physical_labels=None, data_cls=BasisMatrixData) -> BasisMatrixDataBase`. `observation_masks` opcional mapea nombre de target a una secuencia de CSR booleanas, una por componente físico y con forma orbital/imágenes de su matriz; solo restringe el soporte archivado, nunca declara observado un coeficiente ausente.
- `target_data_view(data, layout, target) -> BasisMatrixDataBase`: copia superficial para cambiar etiquetas/auxiliares sin mutar ni separar autograd; para H gauge aporta H_ref+S_ref. Máscaras numéricas por componente en `matrix_known_nodes`/`matrix_known_edges`, mismas formas que labels packed; referencias físicas target-specific en atributos planos prefijados por target. No diccionarios de tensores anidados cuya concatenación PyG sea ambigua.
- Datos llevan `matrix_target_signature` entero de forma `(1, 6)`: `[version, spin, H_present, S_present, DM_present, EDM_present]`. Metadatos conservan descriptor completo; collate contrasta firmas antes de concatenar y conserva el descriptor canónico. El descriptor de modelo se valida también en forward/preflight; la firma no sustituye la validación de checkpoint.
- MatrixMACE devuelve claves legacy packed más `matrix_predictions[target] = {"node_labels": tensor_view, "edge_labels": tensor_view}`. Añadir vistas después de todas las rutas de forward, incluido checkpointing; no nuevos parámetros por envolver módulos.
- `MultitargetMatrixLoss(layout, *, weights: dict[str, float], losses: dict[str, dict])` en `bindings/torch/multitarget.py`; llamada con la firma keyword de pérdidas actual retorna `(loss, stats)`. Config `losses[target] = {"name": "elementwise_mse" | "gauge_invariant_mse", "kwargs": {...}}`; gauge solo H. Exigir claves exactas de targets en weights/losses. S/DM/EDM usan elementwise_mse inicialmente; H elige explícitamente entre ambas.
- LitMACEMatrixModel añade `matrix_target_weights: Optional[dict]=None`, `matrix_target_losses: Optional[dict]=None`, y usa ese loss solo al activar targets. No reutilizar silenciosamente loss/loss_kwargs globales no triviales; rechazar una configuración global contradictoria. Los parámetros nuevos no son requisitos de carga legacy.
- `matrix_from_data(..., *, matrix_target: Optional[str]=None)`, `yield_from_batch(..., *, matrix_target: Optional[str]=None)` y `torch_predict(..., *, matrix_target: Optional[str]=None)` preservan argumentos posicionales actuales. Modo nuevo exige target para reconstruir/conversión, permite packing interno al separar sin conversión. DM/EDM se reconstruyen explícitamente con S_pred.
- `MatrixWriter(..., matrix_target: Optional[str]=None)` selecciona objetivo. SamplewiseMetricsLogger/PlotMatrixError obtienen el mismo selector opcional; modo nuevo sin selector falla antes de abrir/escribir salidas. No es un logger automático de todas las combinaciones.

## Entorno y ciclo de verificación

Ejecutar desde graph2mat. No instalar si faltan temporales: informar del recurso exacto.

```bash
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=/tmp/g2m-server-tests.IcPjmO:/tmp/g2m-http-runtime.LOcBGk:/tmp/g2m-soap-deps.o0Kjat
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
G2M_PY=/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python
```

El intérprete es una herramienta existente; no acceder al código/datos del proyecto que lo contiene. Verificar import de graph2mat en este checkout. Evidencia previa 4.1: suite CPU2732passed/20skip, no asumir que sigue pasando después de 7.6.

Cada tarea: añadir tests, ejecutar el comando y registrar fallo por funcionalidad ausente; implementar, repetir el mismo comando y vecinos. No fabricar fallos para comportamientos ya presentes. Guardar logs y avance en `.superpowers/sdd/2026-10-18-multitarget-7-6/` usando las herramientas permitidas. Sin commits por restricción expresa.

## Task 1: Descriptor, etiquetas coincidentes y lector

**Files:** nuevo helper core, processing/physical y `core/data/tests/test_multitarget.py`; ampliar fixtures solo cuando no basten las actuales.

**Interfaces:** produce MatrixTargetLayout, prepare/load_multitarget_data y target_data_view definidos arriba; consume conversiones y validadores físicos existentes.

- [x] **1.1 Tests RED de descriptor.** `test_layout_canonical_spin_slices`: entrada desordenada completa produce orden H/S/DM/EDM; spin1 slices `(0:1,1:2,2:3,3:4)`, spin2 `(0:2,2:3,3:5,5:7)`, round-trip exacto. `test_layout_rejects_invalid_semantics`: duplicado, nombre desconocido, sin H/S, spin4, versión ajena o slices falseados → ValueError. H+S y H+S+solo DM/EDM también válidos.
- [x] **1.2 Ejecutar** `$G2M_PY -m pytest src/graph2mat/core/data/tests/test_multitarget.py -q --tb=short`; exigir fallos por interfaz ausente, no por entorno.
- [x] **1.3 Implementar descriptor/preparación.** Un grafo canónico de geometría/base; aplanar cada matriz sobre exactamente sus bloques/aristas existentes, comprobar correspondencia. H/S ceros estructurales; DM/EDM máscaras desde soporte archivado, cero explícito observado, ausencia no observada. Cuando el contenedor sisl comparta soporte entre spins, no inventar un soporte original por componente: usar ese soporte común o máscaras explícitas validadas del bundle. NaN/Inf archivado arbitrario produce error, no una máscara fabricada; NaN interno solo representa ausencia demostrada. Reutilizar validación geométrica, spin y hermiticidad; baselines nuevas rechazadas, no desactivadas silenciosamente.
- [x] **1.4 Tests RED de datos.** `test_preparation_matches_independent_channels` compara cada canal con arrays analíticos independientes para spin1/2, dirigido/media arista y periódico. `test_component_support_is_not_row_support`: máscara explícita excluye solo un componente DM, conserva el otro y el espejo periódico coherente; cero almacenado sigue observado. Máscara con forma/spin erróneos, soporte inventado o espejo inconsistente falla. `test_incompatible_bundle_rejected` cambia orbital/especie/celda/imagen/spin/overlap incrustada; matriz seleccionada ausente y NaN/Inf fallan con nombre/contexto. No mutación del bundle. Sin pérdidas físicas, no imponer HD=ES ni requerir matrices no seleccionadas.
- [x] **1.5 Implementar lectura FDF.** Reutilizar lecturas TSHS/TSDE y comprobación de gauge; factorizar el mínimo código de physical.py en vez de copiarlas. Leer solo targets/partners necesarios, con S desde TSHS y geometría/base común. EDM exige certificar cero energético común H/EDM incluso sin energy loss; no exigir HD=ES hasta solicitar esa pérdida. La malla explícita positiva diagonal sin offset se exige para preflight físico, no para aprender H/S solamente. Metadata conserva procedencia/unidades/convención de ocupación; no buscar archivos alternativos en otros árboles.
- [x] **1.6 Tests GREEN de lectura.** `test_real_fdf_tshs_tsde_roundtrip`: temporales propios con geometría/base y SystemLabel, TSHS real y `write_density_matrices(DM, EDM)` real; leer mediante FDF real, sin sustituir el parser completo por fake. Comparar canales/unidades/gauge con rtol/atol1e-10; probar spin1/2 y H/S sin TSDE. `test_fdf_missing_or_shifted_partner` elimina TSDE o cambia Ef → error cuando corresponde; H/S no lee DM/EDM. ZIP nuevo falla antes de leer. Repetir tests core nuevos y `test_physical_references.py`.

## Task 2: Dataset, batch, CLI y preparación sin fuga

**Files:** dataset.py, data.py, cli.py; nuevos `bindings/torch/data/tests/test_multitarget_dataset.py` y pruebas en `tools/lightning/tests/test_cli.py`.

**Interfaces:** consume Task1; propaga matrix_targets/spin y descriptor, entrega batch packed más máscaras/auxiliares planos.

- [x] **2.1 Tests RED.** `test_multitarget_batch_heterogeneous_ownership` combina 2/3 átomos y especies compatibles, spin1/2 por separado: labels/máscaras/partners round-trip por muestra, firma `(B,6)`, ninguna reasignación de átomos/imágenes. `test_collate_rejects_semantic_mismatch` mezcla igual anchura pero target distinto y mezcla spin/layout → error previo al modelo. `test_cache_does_not_alias_labels` muta una muestra sin cambiar originales/caché.
- [x] **2.2 Ejecutar** `$G2M_PY -m pytest src/graph2mat/bindings/torch/data/tests/test_multitarget_dataset.py -q --tb=short`; confirmar RED de comportamiento.
- [x] **2.3 Integrar preparación.** Dataset elige nuevo loader/bundle/preparado solo con modo activo y load_labels=True; load_labels=False crea únicamente geometría/features y firma del descriptor configurado. DataModule valida sub_point_matrix y layout al setup, enlaza targets/spin/total con modelo en CLI; desajuste semántico explícito falla, no se pisa. Preservar type-layout/cost batching, copia/device y ZIP legacy. physical_labels solo carga referencias solicitadas y no entra en features.
- [x] **2.4 Tests de configuración/inferencia.** `test_cli_multitarget_fit_resolves_layout` infiere spin de referencia local y obtiene total4/7; predict desde config/checkpoint funciona sin archivos etiquetados. `test_cli_rejects_mismatched_descriptor` prueba igual total distinto target, n_output_components parcial y baseline. `test_geometry_only_never_loads_references` monkeypatch de lectores a excepción, labels=False sigue funcionando. Cambiar todas las etiquetas deja node_attrs, posiciones, edges, cell y layout de features exactamente iguales.
- [x] **2.5 GREEN.** Ejecutar nuevos tests y `$G2M_PY -m pytest src/graph2mat/bindings/torch/data/tests src/graph2mat/tools/lightning/tests/test_cli.py -q -rs --tb=short --disable-warnings`; registrar resultados y omisiones reales.

## Task 3: Pérdidas independientes, máscaras y referencias físicas

**Files:** nuevo `bindings/torch/multitarget.py`, tests nuevos `bindings/torch/tests/test_multitarget_loss.py`; reutilizar metrics/physical existentes sin cambiar defaults.

**Interfaces:** consume layout/target_data_view; produce MultitargetMatrixLoss y stats target-prefixed con sumas y denominadores para Task4.

- [x] **3.1 Tests RED.** `test_weighted_loss_equals_independent_targets`: pesos `{H:2,S:3,DM:5,EDM:7}` frente suma de cálculos independientes para spin1/2, edge_weight1/2 y ambos losses H. `test_component_mask_preserves_other_spin` obtiene la media explícita de solo entradas conocidas. `test_zero_weight_nan_is_not_evaluated` inyecta NaN solo en salida desactivada: loss/gradientes activos finitos, gradiente de esa salida cero/None. Matriz seleccionada ausente sigue fallando en Task1.
- [x] **3.2 Ejecutar** `$G2M_PY -m pytest src/graph2mat/bindings/torch/tests/test_multitarget_loss.py -q --tb=short` y confirmar RED.
- [x] **3.3 Implementar pérdidas mínimas.** Solo targets positivos se evalúan. Elementwise: seleccionar únicamente canales físicos del target (H sin columna S), llamar métrica existente por componente observado y combinar por número real de elementos/edge_weight, no promediar spins con distinto soporte. H gauge: H_pred contra H_ref+S_ref con overlap_weight0, nunca S_pred. Validar finitud sobre observados antes de restar. Stats incluyen `target/{name}/loss`, errores, sumas/counts matriciales y términos físicos con sus propios denominadores. Evitar modificar el filtro legacy por filas para resolver el caso nuevo.
- [x] **3.4 Integrar física opcional.** Generar/reutilizar auxiliares de prepare_physical_data por target certificado sobre el mismo layout; conservar referencias fuera del grafo y partners reversos. Target views entregan atributos physical_* al helper existente. N/Mulliken solo DM; energía H/DM/EDM. Rechazar pesos/kwargs desconocidos, negativos/no finitos, todo0, falta de partner o target activo sin observaciones por muestra. No evaluar physical_loss_terms dos veces al separar canales; coeficientes físicos0 por defecto.
- [x] **3.5 Tests GREEN físicos/gauge.** `test_h_gauge_uses_reference_s`: H_pred=H_ref+.7S_ref anula H gauge aunque S_pred cambie; error S se mantiene separado. `test_physical_targets_match_single_target`: fixtures fraccionarias/periódicas reproducen términos existentes (población periódica .84, no1.4 Gamma), energía referencia0, derivadas de perturbación coinciden; cambiar solo S_pred no altera esos términos. `test_missing_support_and_invalid_weights` cubre cada error con estructura/target/motivo. Ejecutar suite nueva y `core/data/tests/test_physical_losses.py`/`test_physical_references.py`.

## Task 4: Backbone único, Lightning, agregación y checkpoint

**Files:** models/mace.py, tools/lightning/models/mace.py, model.py, bindings/torch/load.py; nuevo `tools/lightning/tests/test_multitarget_training.py`.

**Interfaces:** MatrixMACE consume descriptor y devuelve matrix_predictions; Lightning consume MultitargetMatrixLoss y sus sumas/counts; checkpoint persiste descriptor y config por target.

- [x] **4.1 Tests RED modelo.** `test_named_views_share_one_backbone` contador forward MACE=1 por llamada, slices packed/named coinciden y siguen autograd; loss de cada target activo produce gradientes finitos no nulos hacia su salida y backbone. Parametrizar spin1/2, lean/full, checkpointing y agregación por interacción sum/mean. `test_reference_changes_do_not_change_eval_output` con mismas features y labels distintas produce salida bit a bit igual en eval.
- [x] **4.2 Ejecutar** `$G2M_PY -m pytest src/graph2mat/tools/lightning/tests/test_multitarget_training.py -q --tb=short`; confirmar RED de integración.
- [x] **4.3 Integrar MatrixMACE/Lightning.** Construir un solo backbone/readout con total semántico y vistas tras forward. Usar loss nueva en train/val/test con early preflight; mantener outputs packed y añadir named views en retornos que consumen callbacks/predict. Coefficient-only/long_range/contexto/etapas que reemplazan pérdida nueva fallan explícitamente. Etapas solo de optimizer compatibles no se prohíben. Hparams nuevos solo son obligatorios en modo nuevo.
- [x] **4.4 Agregación por época.** Reutilizar `_EpochWeightedLoss` y acumuladores existentes por target, reduciendo sumas/counts separados de elementos, estructuras y átomos. `val_loss` = suma ponderada de términos globalmente normalizados, no media de losses por batch ni una cuenta packed. Usar nombres `train|val|test/target/{name}/...`; conservar monitor `val_loss` y nombres legacy en ruta anterior. Caso sin aristas no introduce NaN; target sin observaciones por muestra sí falla.
- [x] **4.5 Tests de agregación/Trainer/checkpoint.** `test_epoch_metrics_independent_of_batch_partition` compara lote único frente particiones2+1 con tamaños/máscaras distintos y pesos físicos, resultados1e-10. `test_one_cpu_step_checkpoint_and_geometry_predict`: Trainer CPU1 update real, finito, parámetros activos cambian; guardar/recargar por API Lightning y load_from_lit_ckpt(as_torch=True/False), misma salida eval y descriptor/lossconfig. `test_checkpoint_same_width_wrong_semantics_rejected` y versión desconocida fallan antes de aceptar pesos. Checkpoint legacy sin campos nuevos recarga con mismas claves/valores y predicciones comparadas contra fixture de baseline capturada antes de editar producto.
- [x] **4.6 Compatibilidades y GREEN.** Tests CPU pequeños para grouping=max, low-rank/SK disponibles, pair_envelope, readout_fp32 y output_scaleshift por componentes. Compile: reutilizar tests de preservación de claves y smoke backend=eager CPU; no certificar backend CUDA/cuEq ni combinaciones cartesianas no probadas. Si alguna opción rompe semántica, no desactivarla ni añadir una prohibición arbitraria: conservar reproducción y pedir decisión si exige ampliar diseño. Ejecutar nuevos tests y vecinos `test_physical_training.py`, `test_training_stages.py`, `test_*epoch_metrics.py`, `test_matrix_components.py` y `test_cli.py` (resolver nombres con rg antes del comando).

## Task 5: Reconstrucción, exportación y consumidores

**Files:** processing.py, callbacks.py, server_app.py; nuevos `core/data/tests/test_multitarget_reconstruction.py`, `tools/lightning/tests/test_multitarget_callbacks.py`; ampliar tests servidor existentes.

**Interfaces:** consume layout, packing/named predictions y target_data_view; produce objetos sisl y writers con selección explícita, sin nueva API HTTP.

- [x] **5.1 Tests RED de reconstrucción.** `test_target_reconstruction_uses_predicted_overlap` hace S_pred distinta en diagonal/offsite de S_ref y compara H/DM/EDM completos y sus S con arrays esperados, spin1/2, NumPy/Torch y batch heterogéneo. H niveles = scipy.linalg.eigh(H_pred,S_pred), no referencia; S target es sisl.Overlap escalar. `test_target_required_before_physical_conversion` impide convertir packing4/7 a Hamiltonian/spin implícito. Sin mutación.
- [x] **5.2 Ejecutar** `$G2M_PY -m pytest src/graph2mat/core/data/tests/test_multitarget_reconstruction.py -q --tb=short`; confirmar RED.
- [x] **5.3 Implementar selección/reconstrucción.** Reutilizar separación de batches y labels→CSR; generar clase sisl explícita por target con canales físicos y S_pred separada. No registrar4/7componentes como Hamiltonian ni editar el sparse.py sucio para una nueva heurística. Conversiones directas de data packed sin selección fallan también; no basta proteger matrix_from_data. S singular conserva valores: control espectral falla, no se recorta.
- [x] **5.4 Consumidores.** MatrixWriter valida target/formato y materializa/reconstruye todo el batch antes de crear archivos. Contrato inicial nuevo-modo `.nc` por target (sisl tiene writers H/S/DM/EDM); extensión distinta falla temprano hasta certificación explícita, sin cambiar legacy. S/DM/EDM retienen convención declarada; gauge de archivos se comprueba con el reader, no se supone. SamplewiseMetricsLogger/PlotMatrixError requieren selector y registran unidades/target; sus métricas incompatibles fallan, no se mezclan4/7columnas. create_server_app rechaza modelos cuyo processor tenga layout multitarea antes de registrar/publicar rutas; un objeto single-target ya separado usa writers existentes.
- [x] **5.5 Tests GREEN de escritura/guards.** `test_writer_selected_target_netcdf_roundtrip` escribe/lee H,S,DM,EDM reales con spin1/2 y S_pred; verificar matrices, metadatos y niveles H con rtol/atol1e-10, Ef explícito cuando aplique. `test_export_failure_leaves_no_partial_files` segundo ejemplo inválido, target ausente o extensión incompatible no deja primer archivo ni sobrescribe un centinela existente. `test_callbacks_do_not_mix_targets` compara métrica elegida con cálculo independiente y rechazo sin selector. `test_server_rejects_multitarget_registration` no invoca predictor ni escribe. Ejecutar nuevos tests y bloque4.1 completo de processing/matrix_components/servidor.

## Task 6: Gates, documentación y cierre acotado

**Files:** docs/informe/roadmap/traspaso listados en mapa; este plan y registros propios. Solo medir mediante pequeño test/script dentro del workspace, no añadir benchmark general al producto.

**Interfaces:** entrega matriz requisito→test/log, informe reproducible, estado técnico y límites científicos separados.

- [x] **6.1 Gates CPU finales.** Ejecutar `$G2M_PY -m pytest src -q -rs --tb=short --disable-warnings` una vez después de fixes; guardar log/exit code, explicar cada skip y fallo. Repetir focal afectado si revisión posterior cambia tests/producto. No llamar «suite limpia» a resultados con fallos ni declarar GPU validada por skips. Ejecutar Ruff en Python tocado y `git diff --check`; verificar hashes de los seis cambios previos preservados y diff limitado al alcance.
- [x] **6.2 Documentar uso y evidencia.** Ejemplo YAML ejecutado por test CLI con H/S/DM/EDM, spin explícito, pesos/losses y baselineFalse; ejemplo inferencia sin etiquetas y MatrixWriter(matrix_target=...). Tabla de unidades, máscaras, gauge, pérdidas referenciadas, formatos soportados y rechazos; ruta legacy y carga de checkpoint. Explicar por qué S_pred puede no ser SPD y por qué tests no validan precisión científica.
- [x] **6.3 Overhead CPU acotado.** Con fixtures sintéticas idénticas y entorno/hilos fijos, medir por separado preparación y forward/loss/backward singleH frente multitarget, warmup5/repeticiones20, mediana y dispersión; registrar tamaños, dtype, targets y parámetros. No prueba estadística de precisión ni claim GPU/VRAM. Si no puede medirse reproduciblemente, registrar «no medido» y causa sin bloquear el contrato funcional.
- [x] **6.4 Revisión Native final.** Una revisión fresca independiente sobre diff final/spec/plan y evidencia, no revisiones por tarea. Corregir Critical/Important con test de regresión y repetir gates afectados; no ampliar alcance por sugerencias nuevas. Revisar especialmente soportes por spin, denominadores, persistencia semántica y consumidores legacy.
- [x] **6.5 Cierre documental.** Informe por requisito/test/comando/resultado, roadmap y traspaso: «implementado», «CPU verificado», «etiquetas reales no certificadas» y «campaña pendiente de autorización» separados. 4.1 permanece cerrado; OOD no resuelto, defaults intactos. Siguiente propuesta, sin ejecutar: comparación pareada H-only/H+S/selección multitarget con datos/splits/receta/presupuesto congelados y observables H/niveles/gap/S/carga/energía; pedir antes recurso etiquetado compatible y autorización científica/GPU.

## Autorrevisión y handoff

- Spec§§1–4 → Tasks1/2/4 (opt-in, spin/layout, baseline y legacy); §5 → Tasks1/2/3 (lectura, soporte, fuga); §6 → Tasks3/4 (pérdidas y denominadores); §7 → Tasks4/5 (autograd/checkpoint/export/guards); §8 → Tasks1–6 (CPU/docs/cierre).
- Interfaces de cada consumidor definidas arriba; no framework de heads, dependencia nueva, endpoint nuevo ni nueva política por defecto. El carácter inseparable del recorrido datos→loss→writer justifica un plan único, con seis gates independientes.
- Cinco Review Focus asignados a tests concretos. Puntos adicionales detectados al inspeccionar: filtrado legacy por filas, callbacks genéricos, loader de checkpoint y falta de formato Overlap dedicado; se cubren sin inferir una semántica física del packing.
- Sin commits/worktrees por mandato del usuario. Native conservado; no pedir de nuevo método de ejecución.

**Estado:** seis tareas completadas, ejecución Native cerrada2026-10-18.
Suite final posterior a fixes2892passed/20skip/0fallos (845.34s), Ruff/diff/hashes
correctos. Revisión única3Important corregidos RED→GREEN,0Minor; sin re-review.
Resultados y Rulings en `.superpowers/sdd/2026-10-18-multitarget-7-6/progress.md`
y `auditorias_mejoras/informe_multitarget_7_6_cpu_20261018.md`.
Validación científica pendiente de datos/protocolo/GPU por autorizar.

**Decisiones de ejecución verificadas:** SK mantiene su límite C1/C2; no
packing completo4/7. Scaleshift se certifica con etiquetas completas; con
parciales falla temprano sin mutar buffers (guarda mínima en su callee compartido
`bindings/e3nn/modules/graph2mat.py`). CSV se inicializa al primer uso y selector
nuevo sobre ruta legacy se rechaza antes de truncar. No se promueven defaults.
Rulings y costes completos, revisión única y regresiones RED→GREEN en el ledger;
estas decisiones no significan certificación científica ni ampliación SK/SOC.
