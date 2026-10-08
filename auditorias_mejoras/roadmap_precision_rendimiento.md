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

- [ ] Diseñar el subproyecto leyendo todos los callers y las limitaciones reales.
- [ ] Reproducir incompatibilidades: H+S, `ModuleList` de readouts por interacción,
  agrupación/orientación y bases heterogéneas; resolver solo dependencias necesarias.
- [ ] Escribir regresiones RED antes del parche; usar readouts e3nn reales.
- [ ] Probar ida/vuelta, s/p/d/f, vacíos, NaN por bloque y simetría periódica.
- [ ] Proyectar targets fuera del hot path cuando la base/layout no cambian.
- [ ] Evitar reconstrucción de predicciones durante el paso de entrenamiento
  cuando la pérdida y callbacks no la necesitan; reconstruir para diagnóstico/inferencia.
- [ ] Demostrar equivalencia de pérdida MSE y gradientes con pesos explícitos.
  No confundir media por operación de `coefficient_space_mse` con MSE por elemento.
- [ ] No prometer equivalencia MAE/Huber/gauge por ortogonalidad de Q:
  reconstrucción/fallback explícito para objetivos no equivalentes.
  Gauge-MSE puede admitir una ruta con S proyectado, pero requiere demostrar
  por separado su producto interno, pesos, máscaras y shift por estructura.
- [ ] Verificar flag apagado, checkpoints, tests amplios y medir GPU disponible.

Expected: pruebas de contrato verdes y ninguna reconstrucción innecesaria en
el camino certificado. Precisión mayor no es un requisito intrínseco de 5.3.

## Task 2 — datos y batching: 6.1 → 6.2 → 6.4

**Files:** `bindings/torch/data/dataset.py`, `tools/lightning/data.py`,
`core/data/processing.py`, `core/data/sparse.py` y tests vecinos.

**Interfaces:** consume `TorchBasisMatrixDataset`, `InMemoryData`,
`TypeLayoutCollate`, conversiones sparse y dataloaders existentes. Produce
caché CPU reutilizable, conversiones equivalentes y batching de coste explícito.

- [ ] Perfilar cada tramo antes de optimizarlo y escribir su plan acotado.
- [ ] Caché de tensores opt-in: delimitar ownership/invalidez y comprobar que
  mutar una muestra no altera accesos futuros; registrar RAM y coste inicial.
- [ ] Vectorizar los tramos medidos de CSR/bloques y reconstrucción, sin
  densificar la matriz global; contrastar elementos y gradientes donde aplique.
- [ ] Probar imágenes periódicas, bloques ausentes, multi-componente y orden.
- [ ] Introducir lotes por coste de aristas/orbitales con presupuesto declarado;
  estructura sobredimensionada produce singleton o error explícito documentado.
- [ ] Probar determinismo, cobertura de muestras, último lote, métricas y DDP;
  documentar que agrupar distinto puede cambiar la trayectoria de optimización.
- [ ] Ejecutar lifecycle/ZIP/stages previos y medir carga, paso, pico y RAM.

Expected: mismas conversiones; caché segura; sampler sin pérdida accidental de
muestras. No asumir que caching reduce VRAM ni que batching siempre usa menos memoria.

## Task 3 — readout eficiente: 7.1

**Files:** `bindings/e3nn/modules/edge_operations.py`,
`bindings/e3nn/modules/preprocessing.py`, `models/mace.py`, exports existentes
y tests de operaciones/readout/modelos bajo `src/graph2mat/`.

**Interfaces:** consume irreps, mensajes y radiales existentes; produce una
variante opcional de TP canal a canal o de rango bajo, integrada en la selección
de operaciones actual. Elegir una primera variante, no construir ambas por anticipación.

- [ ] Fijar diseño y capacidad alcanzable según el readout real, con plan propio.
- [ ] Escribir RED de equivariancia, intercambio de extremos, corte radial,
  matrices simétricas y backpropagation en homo/heteronucleares y s/p/d/f.
- [ ] Implementar la variante mínima con pesos/gates necesarios y defaults intactos.
- [ ] Medir parámetros, coste, memoria y throughput de la arquitectura completa.
- [ ] Smoke de entrenamiento, luego evaluación pareada de precisión y física;
  si pierde capacidad, medir el tradeoff y no sustituir la referencia.

Expected: variante correcta, seleccionable y evaluada; no obligatoriedad de
ganancia, pues se cambia la capacidad del modelo.

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
