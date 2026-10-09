# Graph2Mat — plan de precisión, velocidad y VRAM

> **For agentic workers:** Use `superpowers:executing-plans` para ejecución
> directa, tarea por tarea. Preparar el diseño y los pasos concretos de cada
> subproyecto antes de escribir su código; este documento fija el alcance,
> el orden y los contratos, no una arquitectura única para todos los bloques.

**Goal:** Implementar y evaluar los candidatos acordados de precisión,
velocidad y memoria, conservando una referencia funcional y separando
implementación verificada de mejora experimental demostrada.

**Architecture:** Reutilizar las rutas existentes de coeficientes, datasets,
collate, operaciones e3nn y pérdidas físicas. Las optimizaciones equivalentes
deben preservar salidas y gradientes; las variantes que cambian capacidad u
objetivo científico deben ser opcionales y evaluarse por separado.

**Tech Stack:** Python >=3.9, NumPy/SciPy/sisl, Torch, e3nn, MACE,
PyG, PyTorch Lightning y pytest; reutilizar dependencias existentes.

**Spec:** `auditorias_mejoras/roadmap.md`, puntos 3.14, 5.3, 6.1, 6.2,
6.4, 7.1, 4.4, 4.5 y 7.4; orden aprobado por el usuario el 2026-10-08.
Los contratos y límites siguientes concretan ese acuerdo.

## Global Constraints

- Único repositorio: `/home/christian/repositorios/graph2mat`, rama `g2m-v2`.
- Trabajar en el checkout indicado, sin crear otro worktree.
- Preservar cambios ajenos, APIs públicas, Python >=3.9 y checkpoints previos.
- No trabajar en fonones ni modificar otros repositorios.
- No modificar `g2m_autorun.py`, sus colas, entrenamientos activos o datasets originales.
- No instalar, actualizar ni desinstalar paquetes en el entorno compartido.
- Entornos, artefactos y salidas experimentales propios y aislados.
- No detener procesos ajenos ni ocupar GPU ya utilizada por entrenamientos ajenos.
- Antes de usar GPU, comprobar disponibilidad; evitar concurrencia y registrar carga.
- No push, merge, publicación ni acciones destructivas sobre datos del usuario.
- No nuevas opciones por anticipación: únicamente las necesarias para los candidatos.
- No cambiar defaults científicos basándose solo en tests o un benchmark sintético.
- Commits locales pequeños y explícitos permitidos; no incluir archivos ajenos.

## Review Focus

1. Coeficientes con H+S, readout por interacción y agrupaciones distintas:
   conservar componente, orden, pesos y reconstrucción; nunca truncar en silencio.
2. Bloques heteronucleares y periódicos, s/p/d/f, NaN y grafos vacíos:
   orientación y conteos correctos, simetría y gradientes finitos.
3. Caché con workers/PID/ZIP y mutaciones de una muestra:
   sin contaminación entre accesos, archivos cerrados ni referencias a temporales ajenos.
4. Batches de coste desigual, estructura mayor que el presupuesto y DDP:
   ninguna muestra perdida/duplicada por error, política explícita y métricas ponderadas correctamente.
5. S no ortogonal, spin y ocupaciones fraccionarias:
   convenciones físicas explícitas; no imponer idempotencia a datos que no la cumplen.

## Línea base y criterios de medida

HEAD al preparar el plan: `d338d1a62113af6f0fa1e04df08802ad32abcbf8`.
Última comprobación completa previa: 1541 pasan, 2 omitidos en CPU;
Ruff/mypy verdes. Es evidencia histórica, no una ejecución de este goal.

Cada candidato tendrá dos estados independientes:

- **Implementación:** pendiente / en curso / verificada / descartada con justificación.
- **Beneficio:** no medido / medido favorable / neutro / regresión / recurso ausente.

Que un candidato sea correcto no demuestra que sea más preciso o rápido.
Una evaluación negativa documentada permite rechazar el candidato, pero no
presentarlo como una mejora. Un recurso ausente no completa su evaluación.

Para velocidad y memoria: mismo modelo, batch, dtype, dispositivo y versión;
warmup separado, rondas alternadas referencia/candidato, sincronización CUDA
fuera del tramo de trabajo, mediana y dispersión, pico allocated y reserved.
Separar carga, forward, backward, paso completo e inferencia; incluir coste de
compilación/caché y RAM del host. No comparar ejecuciones bajo cargas distintas.

Para precisión: mismos splits congelados, trayectorias independientes cuando
existan, mismo presupuesto de optimización y selección de checkpoint; semillas
pareadas. Smokes primero, luego al menos tres semillas para una conclusión
comparativa, en materiales pertinentes. Reportar H/DM y métricas físicas según
el objetivo; no extrapolar una ganancia de validación a bandas OOD.

Registrar comandos, versiones, hashes/splits, semillas, tiempos y unidades.
Establecer antes de cada evaluación la métrica primaria y la tolerancia de
no regresión según la dispersión de la referencia. No elegirlas después de ver resultados.

## Task 0 — arranque y referencia enfocada

**Files:** leer `goal_ingenieria.md`, este plan, el roadmap principal y las
familias de tests de coeficientes; actualizar solo el registro de este goal.

**Interfaces:** consume el código y sus entornos ya existentes; produce
HEAD, inventario contrastado, comandos reproducibles y resultado de referencia.

- [x] Confirmar rama, estado Git e instrucciones del repositorio.
- [x] Comprobar qué candidatos ya existen y no repetirlos.
- [x] Verificar importación desde el source con un intérprete aislado disponible.
- [x] Ejecutar la referencia enfocada sin GPU:

```bash
CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=/home/christian/repositorios/graph2mat/src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/tmp/g2m-python312-full.LHyni1/venv/bin/python -m pytest -q \
  src/graph2mat/bindings/e3nn/modules/tests/test_diagnostic_dense_readout.py \
  src/graph2mat/tools/lightning/tests/test_coefficient_epoch_metrics.py \
  src/graph2mat/core/data/tests/test_hamiltonian_training_losses.py \
  --disable-warnings --tb=short
```

Expected: exit 0; registrar el conteo observado, no inventarlo. El entorno
temporal es reutilizable solo si existe y importa el source correcto; si no,
crear uno propio sin tocar el compartido y registrar el comando sustituto.

## Task 1 — irreps: 3.14 necesario → 5.3

**Files:** `core/data/metrics.py`, `core/modules/graph2mat.py`,
`bindings/e3nn/modules/matrixblock.py`, `models/mace.py`,
`tools/lightning/model.py`, `tools/lightning/data.py` bajo `src/graph2mat/`.
Tests: `test_diagnostic_dense_readout.py`, `test_coefficient_epoch_metrics.py`,
`test_hamiltonian_training_losses.py` y regresiones específicas junto a ellos.

**Interfaces:** consume `target_coefficients_from_labels`,
`coefficients_to_labels`, `forward_coefficients`, `forward_with_coefficients`,
`MatrixMACE` y `LitBasisMatrixModel`. Produce una ruta opt-in de entrenamiento
que usa targets proyectados y evita reconstrucciones innecesarias; las rutas
actuales de validación/inferencia y checkpoints siguen funcionando.

- [x] Diseñar el subproyecto leyendo todos los callers y las limitaciones reales.
- [x] Reproducir incompatibilidades: H+S, `ModuleList` de readouts por interacción,
  agrupación/orientación y bases heterogéneas; resolver solo dependencias necesarias.
- [x] Escribir regresiones RED antes del parche; usar readouts e3nn reales.
- [x] Probar ida/vuelta, s/p/d/f, vacíos, NaN por bloque y simetría periódica.
- [x] Proyectar targets fuera del hot path cuando la base/layout no cambian.
- [x] Evitar reconstrucción de predicciones durante el paso de entrenamiento
  cuando la pérdida y callbacks no la necesitan; reconstruir para diagnóstico/inferencia.
- [x] Demostrar equivalencia de pérdida MSE y gradientes con pesos explícitos.
  No confundir media por operación de `coefficient_space_mse` con MSE por elemento.
- [x] No prometer equivalencia MAE/Huber/gauge por ortogonalidad de Q:
  reconstrucción/fallback explícito para objetivos no equivalentes.
  Gauge-MSE puede admitir una ruta con S proyectado, pero requiere demostrar
  por separado su producto interno, pesos, máscaras y shift por estructura.
- [ ] Verificar flag apagado, checkpoints, tests amplios y medir GPU disponible.
  CPU/defaults/restauración/DDP y suite completa final verificados en Task1.g.
  La medida GPU sigue pendiente de hardware libre, no completada.

Expected: pruebas de contrato verdes y ninguna reconstrucción innecesaria en
el camino certificado. Precisión mayor no es un requisito intrínseco de 5.3.

## Task 2 — datos y batching: 6.1 → 6.2 → 6.4

**Files:** `bindings/torch/data/dataset.py`, `tools/lightning/data.py`,
`core/data/processing.py`, `core/data/sparse.py` y tests vecinos.

**Interfaces:** consume `TorchBasisMatrixDataset`, `InMemoryData`,
`TypeLayoutCollate`, conversiones sparse y dataloaders existentes. Produce
caché CPU reutilizable, conversiones equivalentes y batching de coste explícito.

- [x] Perfilar cada tramo antes de optimizarlo y escribir su plan acotado.
- [x] Caché de tensores opt-in: delimitar ownership/invalidez y comprobar que
  mutar una muestra no altera accesos futuros; registrar RAM y coste inicial.
- [x] Vectorizar los tramos medidos de CSR/bloques y reconstrucción, sin
  densificar la matriz global; contrastar elementos y gradientes donde aplique.
- [x] Probar imágenes periódicas, bloques ausentes, multi-componente y orden.
- [x] Introducir lotes por coste de aristas/orbitales con presupuesto declarado;
  estructura sobredimensionada produce singleton o error explícito documentado.
- [x] Probar determinismo, cobertura de muestras, último lote, métricas y DDP;
  documentar que agrupar distinto puede cambiar la trayectoria de optimización.
- [ ] Ejecutar lifecycle/ZIP/stages previos y medir carga, paso, pico y RAM.
  CPU/cobertura/DDP y fullCPU final2209pasan/2omitidos verificados. Medida del
  paso/pico GPU pendiente de hardware libre, no completada como beneficio VRAM.

Expected: mismas conversiones; caché segura; sampler sin pérdida accidental de
muestras. No asumir que caching reduce VRAM ni que batching siempre usa menos memoria.

## Task 3 — readout eficiente: 7.1

**Files:** `bindings/e3nn/modules/edge_operations.py`,
`bindings/e3nn/modules/preprocessing.py`, `models/mace.py`, exports existentes
y tests de operaciones/readout/modelos bajo `src/graph2mat/`.

**Interfaces:** consume irreps, mensajes y radiales existentes; produce una
variante opcional de TP canal a canal o de rango bajo, integrada en la selección
de operaciones actual. Elegir una primera variante, no construir ambas por anticipación.

- [x] Fijar diseño y capacidad alcanzable según el readout real, con plan propio.
- [x] Escribir RED de equivariancia, intercambio de extremos, corte radial,
  matrices simétricas y backpropagation en homo/heteronucleares y s/p/d/f.
- [x] Implementar la variante mínima con pesos/gates necesarios y defaults intactos.
- [ ] Medir parámetros, coste, memoria y throughput de la arquitectura completa.
  Parámetros y tiempos CPU sintéticos medidos; GPU/VRAM/throughput real pendientes.
- [ ] Smoke de entrenamiento, luego evaluación pareada de precisión y física;
  si pierde capacidad, medir el tradeoff y no sustituir la referencia.
  Smoke CPU de dos pasos y checkpoints pasa; precisión/bandas/OOD sin evaluar.

Expected: variante correcta, seleccionable y evaluada; no obligatoriedad de
ganancia, pues se cambia la capacidad del modelo.
El goal acotado de implementación 7.1 permite cerrar código/CPU/protocolo sin
ejecutar GPU; **Task 3 completa sigue abierta por sus experimentos**. Diseño,
resultados y comandos en [informe_readout_rango_bajo.md](informe_readout_rango_bajo.md),
[plan_readout_rango_bajo.md](plan_readout_rango_bajo.md) y
[protocolo_gpu_precision_7_1.md](protocolo_gpu_precision_7_1.md).

## Task 4 — precisión física: 4.4–4.5; después 7.4

**Files:** `core/data/metrics.py`, `core/data/matrices/__init__.py`,
`core/data/processing.py`, `tools/lightning/data.py`,
`bindings/e3nn/modules/hamiltonian_readout.py`, `models/mace.py` y tests vecinos.

**Interfaces:** consume S y matrices/ocupaciones reales disponibles; produce
objetivos físicos opcionales con convenciones explícitas, y después una primera
variante de readout local SO(2) **o** Slater–Koster neuronal con residuo.

- [ ] Inventariar datos propios disponibles: H, S, DM, EDM, spin y ocupaciones.
- [ ] Cerrar únicamente la parte de 4.1 necesaria para esos objetivos y demostrar
  con oráculos independientes el ensamblado no ortogonal.
- [ ] Implementar términos N_e/Mulliken pertinentes y `Tr(DS)` con convención
  de spin/ocupación explícita; probar electrones, gauge, simetría y gradientes.
- [ ] McWeeny solo bajo hipótesis de idempotencia declaradas; EDM ligada a DM/H
  con factores y convención comprobados contra una referencia independiente.
- [ ] Datos H-only no justifican inventar DM/EDM ni convertir sus objetivos en ceros.
- [ ] Evaluar las pérdidas por separado; mantener apagados los términos perjudiciales.
- [ ] Diseñar una primera variante 7.4 con plan separado, bases soportadas
  explícitas y fallback/error para las demás; no dos arquitecturas a la vez.
- [ ] Probar rotaciones, inversión/intercambio, cortes, continuidad del marco
  local, periodicidad y gradientes; entrenar y medir con protocolo pareado.

Expected: código físico bajo sus hipótesis, variantes opcionales y evidencia
experimental real. Si faltan etiquetas imprescindibles, registrar el recurso
concreto y seguir con los bloques independientes, sin cerrar falsamente éste.

## Fuera de esta prioridad

No rehacer Δ-learning, normalización, edge updates, compile ni checkpointing:
ya existen y sus resultados son dependientes del material. No retomar opciones
ya rechazadas sin una hipótesis nueva y concreta. 3.10/3.11 pasan por delante
solo si un input del bloque en curso necesita sus correcciones. 6.5 se decide
después de 5.3 y solo si el cambio de base restante es un cuello medido.
El resto del roadmap principal continúa abierto: este goal no pretende cerrarlo.

## Cierre del goal

- [ ] Cada candidato tiene código/tests o un rechazo técnico/experimental motivado.
- [ ] No hay evaluación necesaria presentada como completa si falta hardware/datos.
- [ ] Referencia preservada, regresiones, suite CPU completa, Ruff/mypy y
  compatibilidad Python/checkpoints proporcionales a los archivos modificados.
- [ ] Medidas GPU y precisión separadas de tests; tradeoffs y límites explícitos.
- [ ] Revisión final independiente de todo el rango, correcciones importantes
  verificadas y estados del roadmap respaldados por evidencias.
- [ ] Informe final por candidato: commits, comandos, P/T/V, aceptado/rechazado
  y recursos pendientes. Ninguna promesa de mejora universal.

## Registro de ejecución — 2026-10-08

- Goal anterior de cinco frentes: completo; no se reabre ni reemplaza su evidencia.
- Autorización actual: crear roadmap/prompt y lanzarlo en esta sesión.
- Ruling: mantener checkout `g2m-v2`, como eligió el usuario; ejecución directa.
- Preparación: HEAD confirmado; tracked limpio; archivos untracked ajenos preservados.
- Estado: **goal activo**, sin presupuesto de tokens solicitado; **Task 0
  completada**, **Task 1 en curso**, Tasks 2–4 pendientes de implementación y evaluación.
- Comandos, logs y cambios de estado se añaden aquí al ejecutar cada sublote.

### Task 0 — referencia reproducida

- No hay AGENTS.md en la raíz ni los ancestros comprobados del checkout.
- Inventario contrastado: las rutas existentes exponen coeficientes pero
  reconstruyen matrices; los targets de coeficientes rechazan etiquetas no 1D
  y las pérdidas/métricas asumen un readout único, no su ModuleList. Son
  dependencias de Task 1; no han sido reparadas por crear este roadmap.
- Δ-learning, output_scaleshift, edge updates, checkpointing, compile y layout
  de collate ya existen. No se incluyen como implementaciones nuevas.
- Primer import del source falló por ausencia de Cython en el venv temporal
  previamente usado para wheels. No fue un fallo de producto ni un test RED.
- Ruling: reutilizar ese venv aislado y añadir solo su dependencia de build
  declarada, sin tocar el entorno compartido. Comando ejecutado:
  `uv pip install --python /tmp/g2m-python312-full.LHyni1/venv/bin/python 'cython>=3,<4'`.
  Resultado: Cython 3.3.0 instalado; `python -m pip check` exit 0,
  `No broken requirements found`. No se cambia la evidencia histórica del goal anterior.
- Import posterior confirmado desde
  `/home/christian/repositorios/graph2mat/src/graph2mat/__init__.py`;
  Torch `2.11.0+cpu`, e3nn `0.4.4`. Este venv sirve para CPU, no para medir CUDA.
- Referencia con el comando de Task 0: **67 passed, 186 warnings, 2.32 s**, exit 0.
  Log: `/tmp/g2m-precision-goal.A2Pbgk/baseline-coefficients.log`.
- Lectura GPU inicial: RTX 5090, 13674/32607 MiB, utilización 100 %.
  No se lanzó trabajo GPU ni se modificaron procesos/colas. Su disponibilidad
  se comprobará de nuevo antes de cualquier medición, no se presume permanente.
- Documentos autocomprobados: alcance/contratos, estados separados y cierre;
  no se ha cambiado código de producto ni repetido la suite completa en este arranque.
- Siguiente acción: plan concreto y primer RED de las incompatibilidades de
  coeficientes de Task 1, comenzando por multi-componente y readout por interacción.

### Task 1.a — componentes y readouts por interacción (implementación verificada)

- Turno previo clasificado como **progreso**: commit f1bb647, documentos,
  goal activo y referencia de 67 tests. No se reabre el goal anterior.
- Plan concreto: `plan_coeficientes_multicomponente.md`; BASE
  `f1bb6472867d16cc8cff04760c5ce600f4cdb09e`. Ejecución directa con TDD.
- RED inicial: **15 fallos, 2 pasan, 24 no seleccionados**;
  `components-red.log`. Fixture sintética ajustada para conservar basis_table
  fuera del ModuleList; segundo RED de producción: **3 fallos**,
  `interactions-red.log`, acceso a types_to_graph2mat del ModuleList.
- Parche: mantener filas de etiquetas/componentes en la proyección, contar
  filas de arista (no componentes), validar dimensiones, conservar vacíos
  `(0, n_components)` y resolver espacios compatibles de readouts. Lightning
  usa el mismo resolver para su schema; firmas/ponderación no cambian.
- Ruling: los tests de precisión construyen RTP directamente en float64,
  en lugar de construir float32 y castear después; no se relaja atol/rtol.
- GREEN de coeficientes y pérdidas existentes: **84 pasan**, 289 warnings,
  2.90 s (`components-green-final.log`).
- Integración adicional: **40 pasan**, 54 no seleccionados, 3874 warnings,
  51.71 s (`components-integration.log`): cuatro casos de NaN/denominadores/
  gradientes multicomponente y 36 MACE reales, sum/mean, una/dos componentes,
  geometrías periódicas y bases s/p/d/f. No es una medición de precisión de entrenamiento.
- Primer gate completo desde source Python: **122 fallos, 1476 pasan,
  2 omitidos**, 335.49 s (`components-full-cpu.log`). No se oculta este gate rojo.
- Diagnóstico: CSR en Python recibe fill_value=0 entero desde el wrapper y
  np.full crea bloques enteros; trunca decimales. Cython convierte el argumento
  anotado a float y no tiene ese fallo. La wheel anterior pasa **90 tests de
  métricas**, mientras source falla en el primer MSE con edge_labels Long.
- Ruling: adelantar la corrección mínima de precisión de 6.2 como prerrequisito
  del gate: `fill_value = float(fill_value)` replica la coerción ya aplicada por
  Cython. No se altera la función de pérdida. Regresión literal .125/-.375:
  **RED 2 fallos/2 pasan**, luego **214 pasan** con métricas/procesamiento,
  176 warnings, 2.75 s (`sparse-fill-red.log`, `sparse-fill-green.log`).
- Ruff y mypy-core py39 pasan tras el parche de CSR; git diff --check limpio.
- Wheels 3.12/3.9 construidas con éxito, con extensión Cython y tests nuevos;
  hashes: 3.12 `81f9d46663ddbb9ba18bf75b2669a74b30a3b0c9924466d785d15a2fdb68eeb8`,
  3.9 `2e611627c3a307362aab54732979b01520ecd3352e768ac2934dec282dcc0d0d`.
- Suite source CPU reparada **1602 pasan, 2 omitidos**, 11911 warnings,
  **363.43 s**, exit 0; log
  `/tmp/g2m-precision-goal.A2Pbgk/components-full-cpu-fixed.log`.
- Wheel 3.9 instalada solo en su venv aislado; pip check pasa. Gate enfocado
  instalado **155 pasan**, 1829 warnings, **79.25 s**, exit 0;
  `components-wheel39-tests.log`. Las siete fuentes Python modificadas/nuevas
  coinciden byte por byte con ambas wheels (unzip -p | cmp), incluidas las regresiones.
- Wheel 3.12 instalada solo en su venv aislado; pip check pasa. Gate instalado
  final **155 pasan**, 4028 warnings, **60.86 s**, exit 0. Log:
  `.superpowers/sdd/plan_coeficientes_multicomponente/task-1-tests.log`.
  Importación -I confirmada desde site-packages y la extensión `_sparse...so`.
- Corrección de precisión CSR comprometida como **bb01435**, con su regresión.
- Revisión propia: callers de proyección/reconstrucción/fit_output_scaleshift/
  pérdidas y schema de épocas leídos; cambios de dimensiones coherentes.
  No sustituye la revisión independiente final del goal amplio.
- Coeficientes/componentes/readouts comprometidos como **bf10967**. El comando
  task-done registró el subproyecto completo en el rango f1bb647..bf10967;
  no se refiere a la Task 1 completa del roadmap amplio.
- Ruling de ejecución: primer task-done se lanzó desde /tmp y falló en su
  preflight Git, antes de ejecutar tests; se relanzó desde el checkout, con
  -I/--pyargs para verificar la wheel instalada. No hay tests ficticiamente verdes.
- **Implementación del sublote verificada; ganancia de entrenamiento/GPU no
  medida.** La corrección CSR garantiza conservación de decimales en la ruta
  Python, pero no implica una mejora nueva de precisión en entrenamientos que
  ya utilizaban la extensión compilada correcta.
- Task 1 general sigue abierta: agrupación/orientación/padding, targets fuera
  del hot path y entrenamiento sin reconstrucción, equivalencia científica y
  mediciones GPU. Tasks 2–4 no se sustituyen por esta compatibilidad.
- Siguiente acción concreta: reproducir orientación de aristas agrupadas
  (`basis_shape`, signed edges) y definir el tratamiento de padding; después
  preparar targets fuera del hot path y el camino de entrenamiento en irreps.
- No hay procesos de prueba de este sublote pendientes; GPU/autorun/colas
  ajenos permanecen intactos. Goal general sigue **active**, sin cierre solicitado.

### Task 1.b — orientación y observación física (implementación verificada)

- Plan `plan_coeficientes_orientacion.md`; BASE
  `e2a906afe35712faa2ead980eb7f0a6b093e3153`; ejecución directa, misma rama.
- Callers de targets/reconstrucción/pérdidas/Lightning/output_scaleshift y
  forward/resorting de Graph2Mat leídos. Sin cambios en pesos ni componentes.
- RED inicial: **33 fallos, 9 pasan**, 189 warnings, 1.45 s;
  `/tmp/g2m-precision-goal.A2Pbgk/orientation-red.log`. Cuatro fallos eran la
  fixture de ghost: PointBasis usa [] para ausencia, no string vacío.
  Corregida la fixture y observado **RED 4 fallos reales** (módulo None).
- Causas: abs agrupado mezclaba módulos +/− en matrices dirigidas; bloques
  físicos negativos no invertían dimensiones; agrupación basis_shape puede
  invertir la orientación original. Padding requería filtros y conteos físicos.
- Parche compartido: conservar signo al seleccionar módulos, transponer ejes
  orbitales en la orientación agrupada simétrica invertida y respetar forma
  dirigida original. Omitir operaciones None, sin exigir coeficientes ghost.
  Reconstrucción max recorta con los filtros existentes por tipo físico.
- Ruling: **targets completos con padding no están determinados** por las
  etiquetas observadas. Rechazar con ValueError que solicita pérdida en
  matrices/enmascarada; no agregar ceros como nuevos objetivos. Max sin padding
  sí admite proyección. No se da por cerrada la equivalencia padded de 5.3.
- Ruling de compatibilidad: los readouts ligeros pueden carecer de filtros;
  usar getattr. Mantener el conteo por coeficientes sin padding, porque la
  selección histórica admite representantes no pareados. Con padding, usar
  filas de edge_labels si existen, o el conteo físico de pares del forward.
- GREEN enfocado final: **130 pasan**, 345 warnings, **3.76 s**, exit 0;
  `orientation-green-final2.log`. Incluye 42 casos nuevos: rectángulos s/p,
  sp/s, d/f, 1/2 componentes, ambos signos, readouts reales, fantasmas, targets
  no observables y max sin padding. Valores/gradientes con tolerancia 1e-12.
- Primer gate completo se lanzó antes del último ajuste de conteo: **1 falla,
  1643 pasan, 2 omitidos**, 351.99 s (`orientation-full-cpu.log`). El proceso
  ya había importado el módulo anterior; se registra rojo, no como evidencia
  del último source. Fallo: test_coefficient_space_mse_aligns_targets_with_symmetric_label_reconstruction.
- Repetición completa desde proceso nuevo, source final: **1644 pasan,
  2 omitidos**, 11931 warnings, **353.01 s**, exit 0;
  `orientation-full-cpu-final.log`, con el comando CPU de Task 1.a sin selección.
- Ruff all-files y selección con el nuevo test, mypy-core py39, compileall
  con Python3.9 y git diff --check pasan. No se han reconstruido wheels para
  este sublote; las wheels/155 tests del bloque previo no lo certifican.
- GPU read-only actualizada: RTX5090 **22559/32607 MiB, 95%**; sin lanzamientos
  propios ni interrupciones/cambios en procesos ajenos.
- **Beneficio: corrección de asociación verificada, no ganancia GPU/precisión
  de entrenamiento medida.** Goal general activo; Task 1 aún en curso.
- Siguiente subproyecto diseñado: `plan_irreps_sin_bloques.md`, kernel
  forward_coefficients simetrizado con mapas locales O(K), sin operador K×K.
  Exploración matemática CPU: 1/16/256/1024 coeficientes, residuo de intercambio
  0 float32 y <=1.56e-15 float64. No es un benchmark. Error de ortogonalidad
  de Q <=1.2e-7 float32 y <=3.11e-15 float64; considerar dtype real al certificar
  MSE, no asumir identidad numérica bit a bit tras cast.
- Nota de diseño para 5.3: filtros max seleccionan copias angulares completas.
  En RTP ij y nodos ij=ji pueden definir coeficientes observables por sus soportes
  locales; probarlo antes de implementar una máscara de pérdida. Simetrías
  ajenas y NaN parciales requieren certificado específico o fallback en matrices.
- Todavía faltan ruta integrada Graph2Mat/MACE/Lightning, targets CPU,
  equivalencia de pesos/MSE, fallback/diagnósticos/callbacks y medidas. Tasks 2–4
  no se reemplazan por estos prerrequisitos.
- Cambios comprometidos como **89c52ca**. Task-done posterior desde source:
  **130 pasan**, 345 warnings, **3.70 s**, exit 0;
  `.superpowers/sdd/plan_coeficientes_orientacion/task-1-tests.log`.
  Registro del subproyecto completo e2a906a..89c52ca; no cierra Task 1 general.

### Task 1.c — kernel sin bloques densos (implementación verificada)

- Plan `plan_irreps_sin_bloques.md`; BASE `2869ef0`; ejecución directa.
- RED real: **67 fallos, 1 pasa**, 2754 warnings, 17.28 s; `kernel-red.log`.
  La ruta simetrizada reconstruía bloques; el roundtrip con Q float32 casteado
  a float64 alteraba coeficientes en 1.17e-8. Fallback no estándar ya pasaba.
- Parche: mapas locales de transposición O(K), sin matriz K×K, certificados
  por Q local. Buffers no persistentes; packing/dos llamadas y componentes
  preservados. Fórmulas sin certificado usan la reconstrucción anterior.
  `forward` y `forward_with_coefficients` de matrices no cambian.
- GREEN inicial: **68 pasan**, 2754 warnings, 16.72 s. Nuevos casos de
  radiales 2s2p2d2f y entradas globales/posicionales elevan el fichero a 86 tests.
  Regresiones ampliadas: **258 pasan**, 9081 warnings, **78.28 s**, exit 0;
  `/tmp/g2m-precision-goal.A2Pbgk/kernel-regressions.log`.
- Suite CPU completa desde source: **1730 pasan, 2 omitidos**, 17156 warnings,
  **406.82 s**, exit 0; `kernel-full-cpu.log` en el mismo directorio.
- Python3.9.25/Torch2.4.1+cpu source: **142 pasan**, 1972 warnings,
  **46.16 s**, exit 0; `kernel-python39-source.log`. Cython3.3.0 instalado
  únicamente en el venv aislado de prueba, sin modificar el entorno compartido.
- Ruff all-files y selección explícita del nuevo fichero, mypy-core py39,
  compileall Python3.9 y git diff --check pasan. Sin nueva wheel para este lote.
- GPU llegó a 521 MiB/0%; preparado wrapper CUDA propio, source y dependencias
  existentes en solo lectura, sin pyc compartido. Antes de lanzar la medida,
  dos entrenamientos Graph2Mat ajenos ocuparon la GPU (9112 MiB/97%).
  **Ningún test ni benchmark CUDA propio lanzado**; procesos intactos.
- Protocolo CUDA contrastado con documentación oficial PyTorch v2.11
  (eventos, sincronización y picos allocated/reserved). No es evidencia medida.
- Probe matemático CPU float64: 30 máscaras físicas, ij/ij=ji, s/p/d/f,
  radiales repetidas y ghost; norma observada 0/1 con error 3.11e-15;
  SSE equivalente con residuo ortogonal: error 5.69e-14, gradiente 1.16e-14.
  **Diseño, no pérdida integrada**. NaN parciales, Q/dtype efectivo y simetrías
  no certificadas todavía requieren tratamiento explícito/fallback.
- **Beneficio no medido.** Graph2Mat/MACE/Lightning siguen usando matrices;
  falta integrar targets CPU y MSE ponderado por elementos físicos. Task 1
  amplia y Tasks 2–4 siguen abiertas. Registro del servicio goal leído como
  **usageLimited**, no complete; el usuario ha autorizado continuar localmente.
- Producto comprometido como **41a03f4**. Task-done real: **258 pasan**, 9081
  warnings, **74.36 s**, exit0, rango 2869ef0..41a03f4. Un primer selector
  de fichero inexistente dio exit4 antes de los tests; corregido, no ocultado.
- Siguiente subplan: `plan_mse_irreps_targets_cpu.md`; QR local del Q efectivo,
  targets CPU, residuo ortogonal y conteos físicos. Integración 5.3 aún pendiente.

### Task 1.d — targets CPU y MSE físico (implementación verificada)

- Subplan `plan_mse_irreps_targets_cpu.md`; BASE `41a03f4`. Nuevo módulo privado
  `bindings/torch/data/coefficient_targets.py`; no flag/default nuevo aún.
- RTP conserva grupos locales de índices por par angular; metadata no persistente.
  Projector CPU guarda QR local del Q efectivo, sin modelo/parámetros/CUDA,
  matrices globales ni Gram K×K. Targets por componentes y tipo físico; SSE
  incluye residuo ortogonal y denominador de elementos físicos, con edge_weight
  tanto en numerador como denominador. No se redefine coefficient_space_mse.
- RED inicial **85 fallos**, 253 warnings, 2.39 s: 84 por ausencia de projector,
  uno por fixture ghost/primer módulo None. Intentos intermedios **84/1** y
  **13/72**: API torch.searchsorted y enumeración errónea de signos homonucleares;
  corregidos a la raíz, sin relajar equivalencia.
- GREEN inicial **85 pasan**, 2.50 s. Revisión propia añadió RED **2 fallos,
  2 controles pasan**, 0.36 s: ModuleList comparado por insertion order y tabla
  CPU que arrastraba lambda de su inicialización al pickle. Comparar por clave y
  snapshot solo de la base; GREEN posterior **89 pasan**, 2.57 s.
- RED adicional AMP **2 fallos**, 89 no seleccionados, 0.73 s; el autocast
  externo reducía matmul del objetivo f32 a bf16 (error relativo hasta 3.68e-4).
  Desactivar autocast únicamente en ese producto interno, sin alterar el modelo.
- GREEN final enfocado **91 pasan**, 253 warnings, **2.46 s**, exit0;
  `/tmp/g2m-precision-goal.A2Pbgk/matrix-mse-amp-green.log`.
  Incluye valores/gradientes fp64 1e-12, fp32 5e-6, cast32→64, 1/2 componentes,
  ambos signos/simetrías, grouping point_type/basis_shape/max, edge_weight0/1/2,
  ruido antisimétrico, ghost/vacíos, NaN por bloque, pickle y límites de entrada.
- NaN parciales, fórmulas sin grupos certificados y máscaras cortando un soporte
  local solicitan fallback explícito mediante CoefficientMSEUnsupported.
  No se añaden ceros como targets del padding. Snapshot debe reconstruirse si Q
  efectivo cambia; no copiar modelos CUDA en cada batch.
- Floor Python3.9.25/Torch2.4.1cpu desde source final **147 pasan**, 119 warnings,
  **3.23 s**, exit0; `matrix-mse-python39-certified.log`. Sin nueva wheel.
- Ruff selección explícita y all-files, mypy-core py39, compileall Python3.9,
  diff --check verdes. Context7 consultado: documentación oficial de autocast;
  ninguna opción global de precisión/CUDA modificada.
- Dos gates completos propios interrumpidos (exit2) al añadir los controles
  de workers y AMP; **no se cuentan como completos**. Tercero definitivo en
  `matrix-mse-full-cpu-certified.log`: **1821 pasan, 2 omitidos**, 17156 warnings,
  **409.36 s**, exit0. Ningún cambio de producto durante este gate final.
- GPU ajena sigue ocupada (14621 MiB/100%); sin lanzamientos propios.
- **Beneficio no medido; integración abierta.** Graph2Mat/MACE/Lightning aún
  reconstruyen durante training; el objetivo nuevo no está conectado. Próximo
  paso: ruta coefficient-only, CPU collate y métricas epoch/DDP sin matrices,
  fallback para callbacks/loss no certificados y checkpoints/defaults intactos.
- Producto comprometido como **48a9f0f**. Task-done **147 pasan**, 254 warnings,
  **2.64 s**, exit0, rango41a03f4..48a9f0f. Siguiente subplan concreto:
  `plan_forward_coeficientes.md`, flag por llamada Graph2Mat/MACE con defaults
  intactos. Collate/Lightning y medidas todavía no implementados/completados.

### Task 1.e — forward coefficient-only Graph2Mat/MACE (verificado)

- Subplan `plan_forward_coeficientes.md`, BASE `a5e6b63`. Flag por llamada,
  defaults y state_dict intactos. Graph2Mat conserva filas por módulo y omite
  bloques/flatten/merge/resorting; MACE agrega coeficientes single/sum/mean.
  GC reentrante y long_range rechazan explícitamente esta ruta directa.
- RED Graph **78 fallos**, 2.72 s; GREEN **78 pasan**, 2.43 s. RED MACE
  **72 fallos**, 65.76 s; GREEN **72 pasan**, 66.65 s (63 no seleccionados).
  Matrices reales, pérdidas y gradientes de todos los parámetros comparados;
  reconstrucción prohibida durante la ruta nueva, geometrías periódicas incluidas.
- RED adicional **2 fallos/1 control**: el helper robaba un keyword de la
  operación. Flags internos posicional-only preservan ese dato legítimo.
- Regresiones ampliadas **228 pasan**, 254 warnings, **4.11 s**, exit0.
  Floor source Python3.9.25/Torch2.4.1cpu **244 pasan**, 63 no seleccionados,
  1959 warnings, **85.96 s**, exit0. Sin nueva wheel.
- Suite source CPU completa `forward-full-cpu.log`: **1974 pasan, 2 omitidos**,
  20984 warnings, **461.56 s**, exit0. Logs en
  `/tmp/g2m-precision-goal.A2Pbgk/`; producto sin cambios durante el gate.
- Ruff explícito/all-files, mypy-core, compileall39 y diff --check verdes.
- **Integración training aún abierta**: collate CPU, Lightning, métricas epoch/
  DDP y fallback para callbacks/objetivos no certificados. No medida GPU,
  entrenamiento de precisión ni cierre de 5.3/goal por estas pruebas CPU.
- Producto **ed975eb**. Task-done real **153 pasan**, 63 no seleccionados,
  4331 warnings, **64.69 s**, exit0, rango a5e6b63..ed975eb.

### Task 1.f — targets integrados en collate CPU (verificado)

- Subplan `plan_collate_irreps_cpu.md`, BASE `6b40f2c`. TypeLayoutCollate usa
  snapshot opcional; setter runtime solo train, sin hparam/caché nueva.
  Targets fuera del store PyG; transferencia profunda de dict/list/tuple.
  Layout simétrico no se aplica a batches dirigidos. Fallback explícito string
  para Unsupported; errors de shape/finite no se ocultan. Complex Q/labels
  rechazados antes de double, sin perder imaginarios silenciosamente.
- RED **29 fallos**, 151 warnings, **1.50 s**. Primer intento GREEN reveló
  fixture con llamada positional inválida al metric: **24 fallos/5 pasan**,
  151 warnings, **90.50 s**, exit1; SIGINT enviado al pytest propio no cambió
  ese resultado. Referencia corregida a kwargs, sin tocar producto/tolerancias.
- GREEN definitivo **29 pasan**, 150 warnings, **36.64 s**, exit0. Workers0/2
  spawn persistentes, dos iteraciones, sym/dir, grouping3, C1/2 y periodicidad;
  comparación de pérdidas/gradientes y get_example sin targets.
- Regresiones source **184 pasan**, 671 warnings, **14.27 s**, exit0;
  stages/lifecycle/layout workers previos incluidos. Floor3.9/Torch2.4.1cpu
  source **176 pasan**, 139 warnings, **30.55 s**, exit0.
- Suite source CPU completa **2003 pasan, 2 omitidos**, 21004 warnings,
  **503.81 s**, exit0. Logs `collate-targets-*` en
  `/tmp/g2m-precision-goal.A2Pbgk/`. Source congelado durante gate completo.
- Ruff selección explícita/all-files, mypy-core, compileall39 y diff verdes.
  Sin wheels/GPU nuevos. **Training_step/epoch/DDP todavía pendientes**;
  siguiente subplan `plan_training_irreps_lightning.md`. 5.3/goal no cerrados.
- Producto **5999120**. Task-done real **29 pasan**, 150 warnings,
  **37.03 s**, exit0, rango6b40f2c..5999120.

### Task 1.g — entrenamiento irreps Lightning (implementación CPU verificada)

- Subplan `plan_training_irreps_lightning.md`, BASE274d7b0. Flag
  `coefficient_training=False`; CPU snapshot tras restore normal/scaleshift,
  training_step decide antes del backbone, solo loss sin matrices en ruta rápida.
  Epoch SSE/count físicos pasan por update envuelto de TorchMetrics, con cache,
  reset y DDP preservados. Validación/test/predict mantienen matrices.
- Fallback con un solo forward y aviso por razón: MAE/Huber/gauge, verbose,
  GC/long_range, callbacks train que leen labels, hooks de transferencia/batch,
  logging custom, estrategia restore-after-setup, precisión/pesos no certificados
  y batches sin targets o NaN parcial. El objetivo por etapas se reevalúa cada batch.
- Projector exige Q fijo y mapas de transpose compatibles con Q efectivo en
  todos los readouts; no pérdida de gradientes de Q trainable ni peers incorrectos.
- RED inicial27fallos/0.99s; GREEN27/33.06s. Fallback fixtures primer intento
  6fallos/8pasan/17.68s; corregidos sin modificar productos/tolerancias salvo
  límite basal AMP explícito. FitRED5/7.15s y GREENintegrado46/56.43s.
- Restore/stages5/10.13s; boundaryRED4/4.09s (Q trainable/peso no certificado),
  vacíos/stages5/5.87s; ModuleListRED3/0.73s (peerQ y mapas).
- Dos ranks Gloo reales: 3casos pareados all_types/missing_types/missing_labels,
  pesos actualizados y métricas físicas iguales, todas gradientes conectadas.
  Control negativo temporal (forzar matriz) dio fallo en ambos ranks
  "DDP silently used matrix fallback"; RETIRADO antes de gates finales.
- Focused final **64 pasan, 33 no seleccionados**, 4864 warnings,
  **108.77s**, exit0. Floor source Python3.9.25/Torch2.4.1cpu **152 pasan**,
  2359 warnings, **92.04s**, exit0. Logs training-irreps-*-final en
  `/tmp/g2m-precision-goal.A2Pbgk/`. Ruff explícito/all-files, mypy-core,
  compileall39 y diff verdes. Suite source CPU completa final **2067 pasan,
  2 omitidos**, 25998 warnings, **612.41s**, exit0;
  `training-irreps-full-cpu-final.log`, sin cambios de source/tests durante el gate.
  Producto **beb0098**. Task-done real **64 pasan, 33 no seleccionados**,
  4864warnings, **107.96s**, exit0, rango274d7b0..beb0098.
- AMP CPU end-to-end NO certificado: backbone basal MACE falla dtype f32/bf16
  antes del readout, tanto con readout_fp32 False como True. Guard de elegibilidad
  y MSE outerAMP se comprueban por separado; integración CUDA AMP sigue pendiente.
- GPU ajena17050MiB/69%, sin lanzamientos propios; sin nueva wheel ni campaña
  de precisión. **5.3 y el goal no cerrados**: equivalencia CPU no es beneficio.
- Siguiente bloque independiente CPU: `plan_cache_tensores_cpu.md` (6.1),
  reutilizar InMemoryData, snapshot/acceso aislados y retirar repr lazy medido.
  Perfil previo propio periódico s/p/d/f,C2; cache legacy mutable y representación
  FDF innecesaria confirmados. No nueva caché/flag/dependencia.

### Task 2.a — caché CPU segura (implementación y medida CPU verificadas)

- Subplan `plan_cache_tensores_cpu.md`, BASEbeb0098. Reutiliza InMemoryData:
  snapshot/acceso deepcopy, límites size antes de leer, guard pathlike sin repr
  de configuraciones. Sin flag nuevo; `store_in_memory=False` permanece default.
  AtomicTable deepcopy preserva referencias runtime y radios atomic/radial
  independientes; no modifica pickle/checkpoints ni reparsea XML.
- RED24fallos/11pasan/44.61s; coreAtomic3 fallos de producto más errores propios
  de fixture corregidos. Control negativo process-local sin deepcopy reproduce
  cache7→0; retirado antes de gates. REDradiosdistintos1 y GREENcertificado57.
- Focus57pasan/51warnings/5.67s; floor3.9/Torch2.4.1cpu171pasan/2322warnings/
  127.17s; full sourceCPU2100pasan/2omitidos/26002warnings/629.52s, exit0 todos.
  Source/tests congelados durante full. Ruff fuente/tests/all-files y benchmark
  directo, mypy-core, compile39/diff verdes; no contar hookSkipped comoPass.
- Benchmark propio s/p/d/f periódico dirigido,C2,fp32,16/72puntos,3warmups y
  12rondas alternadas. Igualdad exacta de muestras/batches. Corrige collate que
  inicialmente usó default simétrico: porcentajes previos sustituidos.
  Lazy5.984→1.127ms y23.251→2.024ms (≈5.3×/11.5×); cacheaccess copia.145/.148ms;
  cachecollate2 .338→.646ms y.353→.674ms (≈+91%,~.31/.32ms adicionales).
  Seguridad **regresa** coste respecto al cache compartido; no caché nueva.
  RSS8samples fresh5044→5228/8744→8800KiB, tensorbytes93424/420240 idénticos.
- Condiciones/script/limitaciones en `informe_cache_tensores_cpu.md` y
  `bench_cache_tensores_cpu.py`; logs cache-tensor-*-certified en registro temporal.
  Carga host no aislada, sin wheel/GPU/VRAM/precisión nuevos: evaluación6.1/goal
  no cerrada como mejora GPU. Siguiente6.2 perfil propio confirma bucles CPU en
  coords/CSR; borde público C2fill0 entero trunca decimales, H nativo usa0.0.
- Producto bc54e91; task-done real57pasan/51warnings/5.02s,exit0, rango
  beb0098..bc54e91. Subplan completo; evaluación GPU/VRAM sigue pendiente.

### Task 2.b — sparse COO/CSR C2 (implementación CPU y mejora local medidas)

- Subplan `plan_sparse_vectorizado.md`, BASEcc334f0. Coordenadas NumPy por
  offsets/repeat, reverse rectangular conserva orden original; CSR C2 por grupos,
  sentinel de capacidad por fila, firstkeyorder y último duplicado explícito.
  Corrige defaultfillint C2 que truncaba decimales y fillfloat que perdía imaginaria.
  Realblocks mantienenfloat64, C2complexcomplex128. Sin APIs/flags/deps/export
  nuevos; scalarCython/vecinos no cambiados, sin extrapolar soportecomplexglobal.
- REDválido33fallos/28pasan/0.20s; GREEN61/0.08s. Primer intento colecciónexit2
  por basename de test nuevo duplicado, corregido renombrando Torch, no producto.
  Doce semillas adicionales de CSR; focused165pasan/63warnings/0.38s,exit0;
  floor3.9/Torch2.4.1cpu222pasan/81warnings/4.78s,exit0. Incluye oráculos,
  gradientes TorchCPU, ghost/vacíos, H/DM/EDM, capacidad/duplicates, ZIP/cache/stages.
- Ruffexplicit/allfiles/benchmarkdirect, mypy-core, compile39/diff verdes.
  Full sourceCPU definitivo2169pasan/2omitidos/25991warnings/650.27s,exit0,
  source/tests congelados durante gate y salida leída. Smoke benchmark source3.9
  points4/rounds1 exit0 con asserts exactos; warnings e3nn basales, no stderrvacío.
- Benchcertified16/72 propios s/p/d/f,periodic,C2,float64,3warmups/12rondas
  alternadas: coords1.228→.080ms/6.641→.260ms (≈15.3×/25.5×), C2
  4.677→2.508ms/21.377→11.214ms (≈1.87×/1.91×). Exactdtype/values/order.
  Constructor fixture independiente no calienta helpers; primera medida con
  CSR armado usando candidato queda sustituida. Hashsourcef5704cd7...;
  baselineAST dos helpers exactos Gitcc334f0, sin modulebody/convertersmutados.
  RSS/outputbytes/primera llamada/dispersión en `informe_sparse_vectorizado.md`;
  RSS deltas incluso0/negativos, no afirmar reducción RAM/VRAM. Host no aislado.
- GPU compute ajeno498MiB/0%, noGPUpropio/wheel/precisión nuevos. Beneficio es
  local CPU, no paso entrenamiento ni kernel escalar compilado.
  Próximo6.4: `plan_batching_coste.md` preparado, **no implementado**; perfil
  readonly de proxy/bytes en fixture desigual, no evidencia de sampler/GPU.
- Producto22029ca; task-done real222pasan/68warnings/5.91s,exit0, rango
  cc334f0..22029ca. Subplan completo CPU; siguiente ejecutar6.4 desde su plan.

### Task2.c — batching por coste6.4 (2026-10-09)

- Subplan plan_batching_coste.md, BASE903a132; mismo checkout, sin agentes por
  tarea ni nuevas dependencias. Único flag max_batch_cost=None conserva defaults.
- Proxy C*(sum n_i² + sum_directed_edges n_i*n_j), greedy + límite batch_size,
  oversize singleton avisado, sin pérdida/último drop. Cache enteros por dataset
  train efectivo/Subset, replacement/finalteardown invalidan. RotatingPool y
  custom samplers incompatibles; pairloss guard antes del backbone y tras etapas.
- DDP nativo reinyectado, máximo de batches sobre shards simulado con fábrica
  Torch/seed/epoch; repite batches locales para igualar pasos, logging explícito
  y métricas de todos los elementos procesados, incluidos repeats/padding nativo.
- Hallazgo RED: Lightning cacheaba primera longitud y truncaba épocas3→4;
  single perdía muestras1,2. Opt-in habilita reload_dataloaders_every_n_epochs=1
  con warning; ctor privado captura época antes de len/reinyección. Sin API
  interna Trainer; datasets/costcache reutilizados. None no toca ese ajuste.
- Finalfocused40pasan/659warnings/32.75s y floor3.9 40pasan/328warnings/29.03s,
  exit0 ambos: workers0/2 spawn/fork, ZIP/Subset/ghost/defaults, coeffcollate,
  MACE float64 C1/C2 symmetric/directed rtol/atol1e-10; cinco fits reales
  ordered/padding,shuffledC2,DDP3→4,single,workers2. Control negativo camino fijo falló
  en ambos ranks al assert de ruta; restaurado antes de gates finales.
- Benchmark CPU sintético,12muestras4/16/72/128,C2dirigido,3warmups/12rondas
  alternadas: pico proxy313152→178944 y tensorbytes1309384→748104 (≈−43%).
  Oversize128 solo, no cota VRAM. Collate1.307→2.557ms (+96%); lazy+collate
  17.834→19.412ms (+8.8%). Prepassloader.153→17.625ms,RSSdelta16→3144KiB,
  cacheenteros472bytes. No ganancia velocidadCPU ni evaluación científica.
- Ruff explícito/all-files, mypy-core y compile39 verdes. Source/tests congelados,
  fullCPU final2209pasan/2omitidos/26518warnings/634.88s,exit0. GPU al comprobar
  5671MiB/100% con cómputo ajeno: sin lanzamiento propio.
  informe_batching_coste.md y bench_batching_coste.py. Implementación verificada
  enfocada; beneficio proxy favorable, GPU/VRAM/precisión recurso ausente/no medidos.
  Goal general abierto; siguiente diseño/perfil7.1.
- Primerfull2207pasan/2skip + SIGABRT al salir de antigua prueba DDP, métricas
  correctas; limpieza explícita Gloo en __main__ del script propio, sin retries.
  Después75pasan en3.12/3.9 y full2208/2skip,653.69s,exit0. Esta full es histórica
  antes del último fix: revisión final REDprefetch workers2 perdía ids1,2,3 al
  invalidar plan antes del len posterior de Lightning. Retener plan durante
  toda época Trainer lo corrige; standalone conserva RNG entre iteraciones.
  Final40enfocados verdes en ambos perfiles; full final2209/2skip exit0.
  Hashdata.py ed975e57... coincide con benchmark final. Producto3a0af15;
  task-done real40pasan/659warnings/30.66s,exit0,rango903a132..3a0af15.
  Subplan cerrado CPU; evaluación GPU pendiente y goal general abierto.

### 2026-10-09 — Task 3 / 7.1: variante de rango bajo, cierre CPU

- Producto/tests `5d607b3`, base `7225ca4`. Una variante opcional:
  `E3nnLowRankEdgeBlock`, proyecciones independientes a rank canales,
  TP `uuu` con todos los caminos angulares admitidos, pesos radiales por
  operación/par y gates SiLU/tanh; skip lineal a ancho original. No equivalente
  al TP completo; capacidad restringida. Defaults/checkpoints anteriores intactos.
- Selección existente en Graph2Mat/MatrixMACE y Lightning, kwargs rank
  serializables al final de la firma. Sin cambios de preprocessing ni dependencias.
  Se rechazan rank/fuentes/formas/radiales 0o o no escalares incorrectos.
- Revisión independiente: un Important reproducido (contexto Hamiltoniano
  concatena features no nulas después del envelope). Guard sobre operaciones
  construidas rechaza esa combinación, incluidas API directa y readouts propios.
  Cinco RED→GREEN nuevos; no Minor aplazados ni cambio del contexto anterior.
- 42 pruebas nuevas: oráculo escalar, O(3)/paridad, intercambio y periódicos
  homo/hetero s/p/d/f, C1/C2, agrupadores y single/sum/mean, empty edges,
  gradcheck, fits CPU de dos pasos y checkpoint candidato/legacy. Con 19
  regresiones de contexto: 61 pasan en 3.12 (33.48s/2715warnings) y 3.9
  (43.62s/1261warnings), exit0 ambos. Ruffall/explicitbench, mypy-core,
  compile39 y diff checks verdes. Full final **2251 pasan/2 omitidos/
  28815warnings/751.75s, exit0**. Primer full falló por PYTHONPATH relativo
  en DDP importando una wheel antigua; corregido el comando, no código ajeno.
- Bench CPU sintético C2,16/72puntos,rank4,fp32,2threads,3warmups/12rondas
  alternadas: paso modelo142.172→107.995 y359.644→179.918ms (−24%/−50%);
  parámetros526016→356400 (−32.2%). Operación aislada **+21.5%/+3.0% de tiempo**.
  Fuente2eed3f3a, hashes/MAD/versiones en JSON; host no aislado. Guard añadido
  después sin cambiar la operación medida ni la fixture sin contexto.
- Informe, plan, script y resultados: `informe_readout_rango_bajo.md`,
  `plan_readout_rango_bajo.md`, `bench_readout_rango_bajo.py`,
  `resultados_readout_rango_bajo_cpu.json`. Protocolo posterior separado en
  `protocolo_gpu_precision_7_1.md`: procesos aislados, CUDA allocated/reserved,
  mismo dtype/budget/splits, warmup/rondas/MAD y tres semillas pareadas.
  **Sin ejecución GPU, AMP CUDA, VRAM ni precisión/bandas/OOD** por decisión
  del usuario. Subplan de implementación CPU cerrado; Task 3 experimental
  y resto del roadmap abiertos. No comenzar 4.4–4.5/7.4 dentro de este goal.

### 2026-10-09 — Nuevo goal 4.4–4.5: inventario inicial, producto pendiente

- El nuevo objetivo es precisión física 4.4–4.5; no incluye 7.4. El usuario
  indicó `MD_vs_AtomicDisplacement` como fuente de materiales, autorizando
  su inspección en lectura sola. No se modificaron archivos de ese repositorio.
- [Inventario y evidencia](inventario_objetivos_fisicos.md): 1560 tripletes
  TSHS/TSDE/HSX localizados en tres conjuntos; matrices y hashes comprobados
  en `MD_steps/0` de cada uno, no validación exhaustiva de todos los snapshots.
- Contrastes independientes de imágenes/Bloch, Mulliken de sisl, referencia
  de energía, rotaciones de base por átomo y ocupaciones fraccionarias.
  La suma periódica da N=8/16/16; Γ puede dar 16.584 en grafeno–hBN y no
  puede sustituirla. Las poblaciones/energías atómicas, no las componentes
  orbitales aisladas, se conservan bajo las transformaciones examinadas.
- McWeeny no aplicable como objetivo exacto a estas muestras fraccionarias.
  Bicapa AA: etiquetas legibles, pero `benchmark_ready=false` por ausencia de
  `electronic_convergence.json`; no se reparó ni se declara benchmark válido.
- Alcance conversacional propuesto pendiente de aprobación. Después deberán
  revisarse diseño y plan; implementación, RED→GREEN, integración, gates,
  costes CPU y protocolo de precisión siguen pendientes. No se cerró Task 4
  ni se atribuyeron mejoras de precisión, velocidad o VRAM a este inventario.
