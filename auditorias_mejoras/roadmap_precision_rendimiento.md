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
  completada**, Tasks 1–4 pendientes de implementación y evaluación.
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
