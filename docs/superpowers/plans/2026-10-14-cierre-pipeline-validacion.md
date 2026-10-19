# Cierre del pipeline y validación de Graph2Mat — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Ejecución Native en el checkout actual; no crear worktree ni delegar sin nueva decisión del usuario. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cerrar las incertidumbres de rendimiento y compatibilidad antes de ampliar arquitecturas o lanzar nuevos entrenamientos científicos.

**Architecture:** Reutilizar el runner de medidas y los tests existentes. Primero aislar los costes CPU; solo después seleccionar una optimización pequeña respaldada por perfilado. Validación CUDA y empaquetado forman entregables independientes. Splits científicos y ampliaciones físicas tienen puertas de decisión propias: este documento no autoriza modificaciones en otro repositorio ni campañas nuevas.

**Tech Stack:** Python >=3.9, PyTorch, Lightning, PyG, NumPy/SciPy, pytest y extensiones Cython existentes; sin dependencias nuevas.

**Spec:** Petición del usuario «Haz un plan para ejecutarlo» tras el inventario del roadmap; requisitos y evidencia en `auditorias_mejoras/traspaso_agente_20261013.md`, `auditorias_mejoras/roadmap.md`, `auditorias_mejoras/roadmap_precision_rendimiento.md` y `auditorias_mejoras/mediciones_gpu_pipeline_20261013.md`. Leer los cuatro documentos antes de ejecutar; los resultados fechados del informe GPU prevalecen sobre pendientes históricos ya resueltos.

**Estado de ejecución (2026-10-14):** tareas1–4 cerradas dentro del alcance, con cuatro fallos históricos por `dscribe` ausente declarados, sin regresiones nuevas. Tareas5–6 entregadas como diseño/protocolo y pendientes de aprobación para implementar/encolar. Checkboxes indican pasos realizados, no aprobación de esas ampliaciones. Evidencia y decisiones en `auditorias_mejoras/diagnostico_pipeline_20261014.md`.

## Global Constraints

- Repositorio `/home/christian/repositorios/graph2mat`, rama `g2m-v2`; conservar todos los cambios previos.
- No cambiar defaults, receta científica, pérdida, splits existentes ni configuración del entorno compartido.
- No modificar `g2m_autorun.py`, su cola ni escribir en MD_vs_AtomicDisplacement sin permiso específico.
- Reutilizar datos congelados w90/6x6 para sondas; no iniciar campañas largas como parte de estas sondas.
- GPU compartida: conservar objetivo de al menos 20 % libre; reutilizar límite del allocator y supervisor existente, aborto propio con menos de 25 % libre o temperatura >=80 °C. No tocar procesos ajenos. Una asignación externa instantánea impide garantizar el suelo absolutamente.
- Registrar código, binarios nativos, datos, pesos iniciales, semillas, versiones y telemetría por comparación. No mezclar resultados de revisiones distintas del runner.
- Medir tiempo de pared, startup, RAM y allocated/reserved; no confundir acelerar el writer con acelerar entrenamiento.
- No realizar commits automáticos sobre el árbol sucio. Al cerrar cada entregable, ofrecer un commit selectivo de sus archivos, previa aprobación.

## Review Focus

1. Muestras mutadas o targets reutilizados tras cambiar Q: deben conservar aislamiento y no servir targets obsoletos. Tarea 2.
2. Edges signed, máscaras, componentes y NaN parciales: preservar pérdida, gradientes y fallback certificado. Tarea 2.
3. Workers activos o fallidos: cerrar hijos propios y medir memoria sin confundir RSS agregado con RAM física única. Tarea 1.
4. AMP y checkpoint: comprobar ruta efectiva, scaler cuando corresponda y estado del optimizador; una prueba manual de training_step no certifica Trainer. Tarea 3.
5. Trayectorias relacionadas: impedir fuga de grupos sin inventar identidades cuando falte metadata. Tarea 5.

---

## Orden y alcance

Ejecutar tareas 1–4 tras aprobación del plan. Las tareas 5–6 son decisiones preparatorias: no desbloquean por sí solas nuevos datasets, repositorios ni entrenamientos.

| Orden | Entregable | Recurso | Puerta de salida |
|---|---|---|---|
| 1 | Diagnóstico aislado CPU/DataLoader y RAM | CPU + sonda GPU corta | Causa localizada y comparación interpretable |
| 2 | Una optimización mínima o cierre negativo | CPU + repetición GPU | Equivalencia y beneficio medido, o no recomendarla |
| 3 | Trainer CUDA fp32/AMP y restauración | GPU corta | Matriz de compatibilidad explícita |
| 4 | Suite, wheels e informe consolidado | CPU | Evidencia reproducible sobre el código final |
| 5 | Diseño de splits/provenance 3.12 | CPU | Alcance y esquema aprobados antes de implementar |
| 6 | Protocolo de precisión y siguientes ampliaciones | CPU | Dataset/recursos aprobados antes de encolar |

### Tarea 1: aislar costes y completar la medida de RAM

**Files:**
- Modify: `auditorias_mejoras/bench_pipeline_gpu.py`.
- Test: `auditorias_mejoras/test_bench_pipeline_gpu.py`.
- Create: `auditorias_mejoras/diagnostico_pipeline_20261014.md`.

**Interfaces:** Consumir `memory_fraction(total, free)`, `variant_order(variants, index)` y los datos congelados que usa `child(args)`. Conservar el esquema de resultados existente; añadir campos diagnósticos, sin reinterpretar resultados v2.

- [x] Registrar HEAD, estado sucio y hashes. Verificar presencia de config/datos y extensiones nativas sin instalar paquetes ni regenerar datos.
- [x] Añadir tests de combinaciones workers={0,4} × pin_memory={False,True}; cada contraste debe cambiar solo una variable. Probar orden equilibrado para las cuatro variantes y rechazo de emparejamiento con distinto hash de entrada.
- [x] Ejecutar `python -m pytest auditorias_mejoras/test_bench_pipeline_gpu.py -q`; confirmar primero que los tests del nuevo comportamiento fallan por esa funcionalidad, no por imports.
- [x] Extender el runner solo con esos controles explícitos. Mantener `persistent_workers` constante donde sea aplicable y registrar que no aplica con cero workers. No modificar `MatrixDataModule` para imponer valores nuevos.
- [x] Añadir muestreo del árbol de procesos propio: RSS principal, RSS de hijos, suma RSS simultánea y número de workers, con timestamps. Leer `/proc/<pid>/smaps_rollup` para PSS cuando sea accesible; si no, marcar PSS como no disponible. No sumar máximos de distintos instantes ni presentar RSS compartida duplicada como consumo físico único. Testear desaparición de un PID entre muestras y exclusión de procesos ajenos.
- [x] Perfilar en CPU setup H+S y `CoefficientMatrixMSEProjector.__call__` con `cProfile.Profile`, separando lectura, conversión sparse, snapshot, validación de bloques y proyección. Guardar perfil y contadores; no usar tiempos perfilados como benchmark final.
- [x] Ejecutar smoke y luego cuatro rondas ×10 warmups ×30 unidades por variante, w90/6x6, bajo supervisor. Artefactos en un directorio nuevo, nunca en `experimentos_speed_probe_pipeline_20261014_v2`.
- [x] Informar ratios pareados y dispersión; distinguir caché/lazy y startup/régimen estable. Seleccionar un único cuello de botella de tarea 2. Si no se identifica uno fiable, cerrar con resultado negativo y pasar a tarea 3.

**Aceptación:** Contrastes aislados, datos idénticos, picos RAM del árbol documentados con sus límites y perfil que explica el coste seleccionado. Ningún cambio productivo todavía.

### Tarea 2: corregir solo el cuello de botella demostrado

**Files — seleccionar únicamente los responsables según tarea 1:**
- Targets: `src/graph2mat/bindings/torch/data/coefficient_targets.py` y, si requiere integración, `src/graph2mat/tools/lightning/data.py`.
- Dataset: `src/graph2mat/bindings/torch/data/dataset.py`.
- Sparse: `src/graph2mat/core/data/sparse.py`.
- Tests: `src/graph2mat/tools/lightning/tests/test_coefficient_target_collate.py`, `src/graph2mat/tools/lightning/tests/test_coefficient_matrix_training.py`, `src/graph2mat/bindings/torch/data/tests/test_tensor_cache.py` y `src/graph2mat/core/data/tests/test_sparse_vectorization.py`, según la ruta elegida.
- Update: informe de tarea 1.

**Interfaces:** Preservar `CoefficientMatrixMSEProjector.__call__(batch, nodes_ref, edges_ref)`, `COEFFICIENT_TARGETS_ATTR`, `TypeLayoutCollate` y APIs públicas de dataset/sparse. No diseñar ahora una caché universal.

- [x] Escribir una prueba roja del coste estructural concreto observado: contador de proyecciones redundantes, validaciones repetidas o asignaciones evitables, sin aserciones frágiles de tiempo en CI.
- [x] Antes de editar, buscar todos los callers del símbolo seleccionado con `rg -n`. Añadir casos de orden de muestras distinto, batch incompleto, signed edges, dos componentes y NaN parcial al test responsable. Comparar loss y todos los gradientes con la ruta matricial usando las tolerancias existentes.
- [x] Aplicar la modificación mínima. Priorizar quitar trabajo redundante o vectorizar sobre introducir estado/cache. Si la solución necesita cache persistente o una API pública nueva, detener esta tarea y presentar diseño específico para aprobación.
- [x] Mantener snapshot CPU inmutable del projector. Probar que reemplazarlo mediante `set_coefficient_mse_projector` no reutiliza resultados anteriores; conservar errores de shape y fallback, sin ocultar excepciones nuevas.
- [x] Ejecutar pruebas de foco:

```bash
python -m pytest src/graph2mat/tools/lightning/tests/test_coefficient_target_collate.py src/graph2mat/tools/lightning/tests/test_coefficient_matrix_training.py src/graph2mat/bindings/torch/data/tests/test_tensor_cache.py src/graph2mat/core/data/tests/test_sparse_vectorization.py -q
```

- [x] Repetir comparación pareada baseline/candidato bajo el mismo runner de tarea 1; separar cambios del runner de cambios del producto. Para targets, medir MSE y control gauge fallback. Para sparse, exigir reconstrucción exacta sobre la misma predicción.
- [x] Retener como mejora solo si supera la dispersión de las rondas sin regresión numérica, pérdida de aislamiento ni incremento de memoria no explicado. Si no, no recomendarla como optimización; registrar el resultado y conservar artefactos. No revertir archivos del usuario.

**Aceptación:** Cambio acotado, equivalencia demostrada y efecto end-to-end cuantificado. Un resultado negativo cierra la investigación, no justifica ampliar arquitectura.

### Tarea 3: certificar Trainer CUDA y AMP

**Files:**
- Create: `src/graph2mat/tools/lightning/tests/test_coefficient_cuda_training.py`.
- Reuse: `training_inputs` de `src/graph2mat/tools/lightning/tests/test_coefficient_matrix_training.py` y fixtures de checkpoint existentes.
- Modify solo si falla una ruta: `src/graph2mat/tools/lightning/model.py` o `src/graph2mat/tools/lightning/data.py` en el punto responsable.
- Update: `auditorias_mejoras/roadmap_precision_rendimiento.md`.

**Interfaces:** Ejercitar `LitMACEMatrixModel`, `MatrixDataModule`, `Trainer.fit`, `Trainer.save_checkpoint` y `Trainer.fit(..., ckpt_path=...)`; reutilizar targets CPU y transferencia existente, no enviarlos prematuramente a CUDA.

- [x] Añadir tests CUDA con skip explícito si no hay dispositivo. Parametrizar `32-true`, `16-mixed` y `bf16-mixed`; bf16 solo cuando `torch.cuda.is_bf16_supported()` sea verdadero. No convertir fallo de una precisión soportada en skip.
- [x] Con modelos de pesos idénticos, comparar matriz/coeficientes bajo el mismo contexto de precisión: finitud, pérdida y todos los gradientes no nulos. Usar fp32 rtol/atol=5e-6; AMP rtol=1e-2, atol=1e-3 como gate inicial explícito, e informar además errores absolutos y relativos. Si no se cumple, investigar antes de relajar tolerancias; AMP no demuestra equivalencia científica.
- [x] Ejecutar un fit real pequeño de dos épocas con validación, writer desactivado y logging sin servicios externos. Verificar que el flag activa la ruta certificada para MSE y mantiene fallback gauge.
- [x] Guardar después de una época y reanudar hasta la segunda. Comparar con ejecución continua, incluyendo `global_step`, epoch, parámetros, métricas, estado Adam y scaler cuando aplique. Mantener semillas/orden de datos fijos y registrar tolerancia numérica; no exigir igualdad bit a bit entre ejecuciones CUDA no deterministas.
- [x] Ejecutar el archivo nuevo bajo el mismo supervisor de recursos, sin lanzar pytest CUDA sin protección:

```bash
python -m pytest src/graph2mat/tools/lightning/tests/test_coefficient_cuda_training.py -q
```

- [x] Publicar matriz soportado/falla/no disponible y errores máximos. No activar AMP por defecto ni atribuirle mejoras de precisión.

**Aceptación:** Certificación real de Trainer y resume, no solo `training_step`; cualquier incompatibilidad queda visible y con reproducción mínima.

### Tarea 4: cierre de calidad y empaquetado

**Files:** Informe de tarea 1, `auditorias_mejoras/roadmap.md`, `auditorias_mejoras/roadmap_precision_rendimiento.md` y tests tocados. No cambiar `pyproject.toml` salvo defecto reproducido y aprobado.

- [x] Ejecutar `python -m pytest src -q` sobre source final. Comparar fallos con baseline del mismo entorno; los cuatro de dscribe del traspaso son antecedentes, no permiso para ignorar fallos nuevos.
- [x] Repetir entorno mínimo Python 3.9 y entorno actual usando entornos aislados existentes. Si falta alguno, indicar limitación y pedir permiso antes de instalar o alterar el entorno compartido.
- [x] Construir wheel en entorno aislado con `python -m build --wheel`; instalarla solo en entorno desechable fuera del checkout y ejecutar tests de las rutas modificadas. Registrar ubicación de `graph2mat.__file__` para excluir import accidental del editable y hash del wheel/binarios.
- [x] Revisar diff final, repetir tests de foco y comprobar `git diff --check`. Actualizar únicamente filas con evidencia nueva; distinguir implementado, validado, resultado negativo y bloqueado por recurso.
- [x] Entregar tiempos, memoria, compatibilidad y limitaciones. Ofrecer commit selectivo sin incluir archivos preexistentes ajenos.

**Aceptación:** Sin regresiones nuevas no explicadas, wheel comprueba código instalado y roadmap no mantiene como pendiente lo ya certificado.

### Tarea 5: preparar 3.12 sin modificar experimentos históricos

**Files:** Create `docs/superpowers/specs/2026-10-14-trajectory-splits-design.md`. Inspeccionar `src/graph2mat/tools/lightning/data.py`, `src/graph2mat/tools/lightning/tests/test_datamodule_split.py` y `src/graph2mat/tools/lightning/tests/test_checkpoint_provenance.py`.

- [x] Inventariar qué metadata permite identificar trayectoria, estructura origen y duplicados en los datos realmente disponibles. No deducir trayectoria únicamente del nombre de archivo o proximidad temporal.
- [x] Diseñar split opt-in por grupo con manifiesto congelado: IDs de muestra/grupo, partición, seed y hashes. Sin metadata suficiente, rechazar ese modo con diagnóstico explícito, no aplicar split aleatorio silencioso.
- [x] Especificar tests de grupos disjuntos, muestra duplicada, grupo único, proporciones inviables, IDs ausentes y reproducibilidad ante reordenación de entradas. Incluir tratamiento explícito de grupos que atraviesan archivos ZIP.
- [x] Separar integración Graph2Mat de generación/validación del benchmark externo. Presentar diseño y solicitar aprobación del alcance antes de implementar o tocar ese repositorio. No cambiar splits de campañas previas.

**Aceptación:** Diseño ejecutable y alcance acordable, no afirmar que 3.12 está implementado al terminar este documento.

### Tarea 6: preparar decisiones científicas, no encolarlas

**Files:** Create `auditorias_mejoras/protocolo_siguientes_validaciones_20261014.md`.

- [x] Para 6.4, inventariar dataset heterogéneo real disponible y definir comparación con mismos datos/splits/modelo, al menos tres semillas y registro de muestras, épocas, updates y tiempo. Comparar tanto presupuesto de muestras como presupuesto de cómputo; no confundir distinto número de updates con efecto puro del batching.
- [x] Fijar métricas H alineado y bandas alineadas ID/OOD, ratios pareados e IC95. Para promoción conservar criterio del traspaso: límite superior IC95 de razón <1 y ninguna regresión OOD >5 %. Sin datos suficientes, presentar el bloqueo, no usar fixture sintética como prueba científica.
- [x] Ordenar ampliaciones separadas: 4.1 reconstrucción no ortogonal; 4.4/4.5 pérdidas DM con validación real; 7.6 objetivos DM/EDM. Cada una requiere dataset con etiquetas y diseño aprobado. Spin/SOC 8.1 y especies/ghosts 8.3 solo cuando haya caso concreto.
- [x] Mantener 3.13 pendiente de dos GPU reales; no declarar que DDP CPU sustituye esa validación. Mantener 1.15, residual 3.8 y cambios de defaults como decisiones del usuario.
- [x] Presentar lista exacta de trabajos, dependencias y estimación medida mediante piloto antes de solicitar permiso de encolado. No relanzar automáticamente readouts ya evaluados ni el piloto 7.7 NO_GO.

**Aceptación:** Protocolo y decisiones solicitadas; cero entrenamientos nuevos encolados.

## Estimación de ejecución

- Tareas 1–2: 1–2 jornadas de trabajo, condicionadas al diagnóstico; sondas GPU del orden de decenas de minutos por tanda. Referencia: la tanda anterior de 112 procesos tardó 28 min 26 s, sin preparación.
- Tareas 3–4: 1–2 jornadas adicionales por compatibilidad AMP, suite y wheels. Son rangos de planificación, no tiempos medidos ni promesa de ejecución continua.
- Tareas 5–6: aproximadamente media jornada para diseños/protocolos una vez disponible metadata. Entrenamientos científicos, espera de cola y acceso a segunda GPU no incluidos.
- Reestimar tras tarea 1. No dar fecha de cierre del roadmap científico sin dataset y protocolo aprobados.

## Autorrevisión y entrega

- [x] Separado diagnóstico de cambios productivos y entrenamiento científico.
- [x] Preservados defaults, cola, entorno compartido y archivos previos.
- [x] Cada riesgo de Review Focus asignado a tarea y prueba concreta.
- [x] Declaradas puertas de aprobación cuando falta diseño, dataset o autoridad sobre otro repositorio.
- [x] Sin ganancias prometidas ni reinterpretación de benchmarks como resultados de precisión.
- [x] Usuario revisó el plan y confirmó ejecución Native («Hazlo») antes de implementar.
