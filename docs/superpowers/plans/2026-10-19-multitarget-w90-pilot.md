# Piloto multitarget w90 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Método preservado: Native, no implementación por subagentes; una revisión independiente final antes de GPU.

**Goal:** Medir la viabilidad CUDA, coste y memoria de H-only frente a H/S/DM/EDM mediante una pareja de pilotos acotada, no entrenar la campaña definitiva.

**Architecture:** Un script experimental reutiliza lectores, datasets, collate, modelos, pérdidas y helpers de telemetría existentes. Primero produce evidencia CPU inmutable y un backbone inicial compartido; después supervisa dos hijos secuenciales con un único plazo de 900s y agrega sus resultados, completos o parciales.

**Tech Stack:** Python, NumPy/SciPy/sisl, PyTorch/PyG, Lightning/MACE existentes, pytest; biblioteca estándar para supervisión y nvidia-smi de solo lectura.

**Spec:** `docs/superpowers/specs/2026-10-18-multitarget-w90-pilot-design.md`, aprobado por el usuario con «Apruebo» el 2026-10-19.

**Autorización:** plan aprobado por el usuario con «Apruebo» el2026-10-19.
Implementación Native en curso; no autoriza los seis fits definitivos.

**Estado del intento v1 (histórico):** implementación/verificaciones y cierre documental
ejecutados; pareja científica interrumpida después de H-only por carrera
del supervisor al salir su PID. Joint no lanzado. No reintentos autorizados.
Ver `auditorias_mejoras/informe_piloto_multitarget_w90_20261019.md` y ledger.

**Continuación v2:** usuario«Dale» autorizó parche focal de la carrera del PID
y una nueva pareja acotada. V2 completada:123tests y ambos brazos120updates+
validación150, sin repetir v1. Ver informe con sufijo `_v2.md` y ledger.
Seis fits definitivos continúan sin autorización; no relanzar v1/v2.

## Global Constraints

- Checkout actual `<graph2mat>`, rama `g2m-v2`; preservar todo cambio previo. Sin commits, worktrees, cambios de rama ni dependencias nuevas.
- Único dataset externo de solo lectura: `$MDVS_ROOT/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1500`; no explorar fuentes del otro repositorio.
- Split intacto1198/150/150, exclusiones md_1198/md_1349; duplicados y bloques MD compartidos se notifican, nunca se redistribuyen automáticamente.
- H-only y joint directos, seed0, batch32, fp32,20warmups+100updates medidos por brazo; sin compile/checkpointing/augmentations/cost batching ni test predictivo.
- Receta completa de §3 del diseño, pesos H1/S114.9749324374332/DM211.0388115121933/EDM3.133677534317286. Cosine por update, T_max4000 y eta_min1e-5; no4000épocas ni T_max120.
- GPU0, procesos separados H→joint; 900s totales desde primer hijo, incluidos arranque/validación/esperas, y hasta10s de limpieza. No reintentos, ampliación de plazo ni reducción automática de modelo/batch.
- Temperatura inicial<75°C, allocator=min(0.55,libre/total−0.30), margen asignable≥512MiB. Muestreo0.5s; parar si libre<25%, temperatura≥80°C, cómputo ajeno, error de telemetría o plazo agotado.
- No prometer20% libre como exclusión absoluta de asignaciones externas. Nunca señalizar procesos ajenos; sin cola, autorun, fonones, otros proyectos, defaults ni seis fits definitivos.
- Fallo de referencia SPD/hermiticidad detiene todo antes de GPU; ninguna reparación, corte de S o exclusión silenciosa.

## Review Focus

- Un archivo/FDF cambia tras el certificado o un symlink/include escapa: rechazar antes de abrir y no mutar fuentes (Task1).
- Dos grafos equivalentes empaquetan H/S en distinto orden o padding: comparar loss H y features, no confiar solo en shapes (Task2).
- El último batch14 y los resets de métricas alteran exposición/LR: fijar IDs,38batches/época y scheduler por optimizer.step (Task2).
- Un hijo sale dejando descendientes, aparece proceso ajeno o falla nvidia-smi: limpiar solo grupo propio y conservar diagnóstico (Task3).
- Se agota el plazo entre brazos o faltan updates en un JSON aparentemente correcto: no declarar pareja completa ni extrapolar coste sin marcar insuficiencia (Task3/4).

## Archivos y comandos

Crear `auditorias_mejoras/pilot_multitarget_w90.py` (CLI/preflight/loop/supervisor/resumen) y `auditorias_mejoras/test_pilot_multitarget_w90.py` (tests focales sintéticos y mocks). No modificar `src/graph2mat`, auditor o benchmark previos. Documentación: este plan, diseño aprobado, roadmap, traspaso e informe nuevo `auditorias_mejoras/informe_piloto_multitarget_w90_20261019.md`.

Artefactos bajo `auditorias_mejoras/piloto_multitarget_w90_20261019_v1/`, creado exclusivamente por preflight. Si ya existe, parar y acordar otro nombre: no borrar, sobrescribir ni reutilizar como intento nuevo. Pilot solo puede consumir un preflight completo del mismo directorio y crear una marca exclusiva de lanzamiento una vez.

Intérprete `P=python`, con `PYTHONDONTWRITEBYTECODE=1`, OMP/OPENBLAS/MKL_NUM_THREADS=2. Para los comandos CPU añadir `CUDA_VISIBLE_DEVICES=`; para el piloto identificar GPU0 física y no ocultarla. Reutilizar únicamente overlays de dependencias ya instalados registrados en el traspaso si siguen disponibles, sin seguir overlays hacia código de otros repositorios. Verificar que `graph2mat.__file__` resuelve al checkout actual.

Comandos previstos, siempre desde Graph2Mat:

```bash
$P -m pytest auditorias_mejoras/test_pilot_multitarget_w90.py -q --tb=short
$P auditorias_mejoras/pilot_multitarget_w90.py preflight --out auditorias_mejoras/piloto_multitarget_w90_20261019_v1
$P auditorias_mejoras/pilot_multitarget_w90.py pilot --out auditorias_mejoras/piloto_multitarget_w90_20261019_v1
```

No ejecutar el tercer comando hasta completar los gates CPU y la revisión Native.

### Task 1: Preflight reproducible, suelo de representación y espectro

**Files:** Crear script y test mencionados; consultar sin editar `audit_multitarget_w90.py`, `core/data/multitarget.py`, `core/data/physical.py`, tests de reconstrucción y certificado JSON20261018.

**Interfaces:**
- `freeze_sources() -> dict`: ROOT fijo, hashes certificados antes/después, split IDs, exclusiones, avisos y procedencia del checkout.
- `reference_floor(bundle: dict, processor, data, kpoints: np.ndarray) -> dict`: métricas por target, N/HD−SE, hermiticidad/SPD y espectros original/reconstruido, sin E/S de fuentes.
- `preflight(out: Path) -> dict`: crea salida no-clobber y escribe `manifest.json`, `preflight.json` y arrays k con hashes; un fallo deja `status=blocked`, no un certificado exitoso.

- [ ] **1. RED:** Escribir tests `test_boundary_and_hash_guards` (symlink externo, include FDF y hash cambiado fallan sin leer/mutar fuera), `test_split_is_not_reassigned` (1198/150/150, exclusiones y avisos intactos) y `test_preflight_no_clobber` (segunda llamada no cambia bytes existentes). Ejecutarlos y guardar el fallo de las funciones ausentes.
- [ ] **2. Implementar** `freeze_sources` reutilizando `checked_path`/`verified_file` y el inventario certificado. Validar todas las rutas antes del reader, incluso base y artefactos auxiliares. Capturar hashes de fuentes productivas y del propio script/config, HEAD/rama/dirty y versiones; ningún hash del checkout limpio sustituye al árbol dirty efectivo.
- [ ] **3. RED:** Añadir `test_periodic_floor_and_spectral_controls`: referencia sintética periódica, soporte conocido con una cola omitida de norma no nula, H+cS desplaza niveles en c sin cambiar gaps. Comparar eigh contra Cholesky independiente; S no SPD o no Hermitiana bloquea, sin clipping. Assert shape camino==(151,3), K exacto presente y cuatro bandas ocupadas nominalmente; gap indirecto negativo se conserva. Ejecutar y observar fallo antes de implementar.
- [ ] **4. Implementar** `reference_floor`/`preflight` con `_read_reference_bundle(...require_mesh=True)`, `prepare_multitarget_data` y `processor.matrix_from_data(...matrix_target=target,threshold=None)`. Reindexar las matrices por átomo/orbital/imagen antes de comparar: no asumir que dos CSR tienen el mismo orden de imágenes. Medir error máximo/Frobenius y contribución omitida por target, N y HD−SE mediante `_weighted_reference_rows` cuando su soporte está completo; si falta soporte informar que el observable reconstruido no es certificable, no imputar su valor.
- [ ] **5. Completar** malla leída+camino del diseño, guardar ambos por separado y su unión para checks. Bloch complejo en float64/complex128, todos los orbitales, eigh sin regularización. Definir hermiticidad relativa como `||A−A†||F/max(||A||F,tiny)`; comprobar geometrías/finitud y S original/reconstruida. Registrar niveles/gaps y diferencias de representación por estructura, ambos conjuntos. No inferir predicciones, afinar pesos ni cambiar cutoff por los resultados.
- [ ] **6. GREEN:** Ejecutar nuevos tests CPU con fixtures propias. El preflight real sobre1498 ejemplos se ejecutará una vez en Task4, después de congelar la versión final del script; no producir aquí evidencia real que quedaría desfasada durante Tasks2–3.

### Task 2: Dos rutas comparables y smoke CPU real

**Files:** Mismo script/test. Reutilizar `LitMACEMatrixModel`, `TorchBasisMatrixDataset`/`MatrixDataModule`, `TypeLayoutCollate` y `gauge_invariant_mse`; no editar sus módulos.

**Interfaces:**
- `batch_order(ids: list[str], seed: int, updates: int) -> list[list[str]]`: generador independiente por época `torch.Generator().manual_seed(seed+epoch)`, batch32/drop_lastFalse.
- `make_arm(arm: str, basis_table, backbone_state: dict | None = None)`: modelo H-only o joint, configuración resuelta y estado compartido identificable por nombres, no posiciones.
- `run_arm(out: Path, arm: str, device: str, warmup: int = 20, steps: int = 100) -> dict`: loop real, resultados/updates incrementales; parámetros menores solo para tests CPU, CLI pilot fija20/100.

- [ ] **1. RED:** `test_order_last_batch_and_scheduler` comprueba38batches, último14, cobertura1198 por época, mismos120 updates en ambos brazos aunque consuman distinto RNG de inicialización;100medidos contienen3146muestras y3batches de14. Primer LR0.0018 y siguiente según cosine4000. No reset de optimizer/scheduler en update20 ni salto del último batch. Ejecutar y guardar RED.
- [ ] **2. Implementar** `batch_order` y loaders con mismo orden/seed,0workers/pin_memoryGPU, caché en CPU sin mutaciones. Separar explícitamente RNG de modelos/sampler. Guardar orden completo y hash en preflight. El worker solo carga train/val, nunca test predictivo, y verifica fuentes antes de usarlas.
- [ ] **3. RED:** `test_shared_backbone_and_h_loss` usa matrices reales pequeñas sintéticas con valores H/S distintos, empaquetado igual al de producción: estados del backbone idénticos y features eval iguales; H gauge de legacy y joint coincide con H predicha idéntica. Incluye batch heterogéneo y último incompleto. Assert pesos/descriptor y todas las opciones de §3 resueltas. Ejecutar y guardar RED.
- [ ] **4. Implementar** `make_arm`, copiar por nombres/forma únicamente `model.mace` compartido y verificar igualdad exacta; las inicializaciones de heads no se declaran equivalentes. Guardar `effective-config-{arm}.json`, estado inicial CPU propio con hash, grupos Adam por nombres de parámetros y kwargs efectivos. Las clases se serializan por ruta importable, no repr con direcciones de memoria. No cargar pesos históricos ni usar `strict=False` para ocultar diferencias.
- [ ] **5. RED:** `test_cpu_smoke_both_arms` ejecuta2updates por brazo con modelo pequeño en float32, loss/gradientes/parámetros finitos, cambios de pesos y datos de entrada intactos. `test_validation_h_aggregation` compara agregación global H gauge por denominadores con los stats del modelo y diferentes particiones de batch, sin incluir loss auxiliar; `test_nonfinite_fails` para loss/gradiente/parámetro. Ejecutar RED.
- [ ] **6. Implementar** `run_arm`: usar `model.configure_optimizers()` para conservar grupos/weight decay, CosineAnnealingLR manual después de cada paso. training_step/backward/step reales, logger externo/checkpoint apagados, métricas internas retenidas/reset por épocas como en la ruta entrenada. No reutilizar el Adam genérico del benchmark que tiene otro LR y grouping. Ningún objeto de prueba sintético se usa como artefacto científico.
- [ ] **7. Implementar** timings sincronizados por update y eventos CUDA diagnósticos, incluyendo carga/H2D; conservar muestras/tamaño para throughput ponderado. Registrar picos warmup antes de reset y medidos después, sin empty_cache entre pasos. Validación completa150 al final, sin gradiente ni selección, con métricas H separadas y tiempo propio; no comparar accuracy de modelos distintos como prueba de equivalencia.
- [ ] **8. GREEN:** Ejecutar tests nuevos y smoke CPU, más `src/graph2mat/tools/lightning/tests/test_multitarget_training.py`, `src/graph2mat/bindings/torch/tests/test_multitarget_loss.py` y `src/graph2mat/core/data/tests/test_multitarget_reconstruction.py`. Guardar terminales y conteos reales. No declarar omitidos CUDA como certificados ni repetir suite general por cambios solo experimentales.

### Task 3: Supervisor acotado y agregación verificable

**Files:** Mismo script/test; importar solo helpers `gpu_state`, `memory_fraction`, `process_tree_memory` de `bench_pipeline_gpu.py`, sin ejecutar su main.

**Interfaces:**
- `guard_reason(state: dict, compute_pids: set[int], owned_pids: set[int], now: float, deadline: float) -> str | None`.
- `pilot(out: Path) -> dict`: verifica preflight/config/hashes, marca exclusiva, lanza solo H-only→joint en grupos nuevos y escribe `telemetry.jsonl`, logs y `summary.json`.
- `summarize(results: dict, telemetry: list[dict]) -> dict`: estado de pareja, tiempos/memoria, rango orientativo4000updates; no habilita lanzamiento de fits.

- [ ] **1. RED:** `test_guard_boundaries` cubre libre24.9%/25%,80°C, plazo `now>=deadline`, PID ajeno y telemetría ausente/no numérica; inicio75°C bloqueado. `test_allocator_cap` cubre .55, reserva.30 y umbral512MiB. Ejecutar RED.
- [ ] **2. RED:** `test_process_cleanup_scope_and_deadline` usa Popen/reloj/señales simulados: no arrancar joint si expiró plazo, no reiniciar fallido ni OOM, TERM→KILL solo PGID creado por el runner, limpieza total≤10s aunque leader haya salido; fallo de nvidia-smi aborta. `test_pilot_one_shot` rechaza segunda marca sin mutar logs. Ejecutar RED.
- [ ] **3. Implementar** supervisor mínimo, sin framework de jobs: nvidia-smi identifica GPU0/UUID y procesos de cómputo; distinguir hijos propios por árbol/PGID verificado, no por nombre. No excluir un PID desconocido como si fuera propio. GPU ocupada o estado no verificable deja blocked antes de lanzar; muestrear0.5s en padre CPU. deadline único monotónico desde primer Popen; condición y limpieza se aplican también a validación y entre brazos.
- [ ] **4. Implementar** worker interno del mismo script con allocator fijado antes de crear CUDA tensors, TF32 desactivado en ambos y matmul_precision=highest. Fijar OMP/BLAS2 y comprobar import del checkout. Redactar outputs por brazo de forma incremental, no-clobber; un crash conserva logs, telemetría y contador de updates, sin aparentar JSON completo.
- [ ] **5. RED:** `test_summary_incomplete_and_weighted_cost`: un brazo ausente/99medidos/fuentes cambiadas/fallo de validación no da pareja completa; throughput=`sum(batch_sizes)/sum(wall_s)`, no promedio de rates. Picos allocated/reserved no se confunden con uso total GPU; valores ausentes se marcan desconocidos, no cero. Ejecutar RED.
- [ ] **6. Implementar** `summarize`: completo solo con ambos120updates/100medidos, validación150 completa/finita y fuentes/órdenes/LR comprobados. Mediana/p95/min/max, ratios descriptivos entre brazos; sin IC causal. Estimación central4000updates por muestra de38batches,20validaciones cada200, startup/preflight separados; rango orientativo por p10–p90 de update y margen explícito adicional25% para validación/guardado no caracterizados, no intervalo estadístico ni hora garantizada. No estimar lote si la evidencia es insuficiente; no redondear memoria para ocultar regresiones.
- [ ] **7. GREEN/revisión Native:** Ejecutar tests focales completos, Ruff del script/tests, `git diff --check` y hashes de fuentes productivas. Una revisión independiente de todo el diff nuevo y del cumplimiento del diseño antes de GPU, con el método Native ya elegido. Resolver hallazgos mediante tests RED→GREEN y repetir solo gates pertinentes; no otra implementación por subagentes ni auditoría de campañas cerradas. Si aparece un fallo que exige cambiar producto o instalar dependencias, detenerse e informar, no ampliar esta implementación experimental.

### Task 4: Ejecutar una pareja autorizada y cerrar con evidencia

**Files:** Artefactos propios e informe20261019, roadmap y traspaso; ningún archivo original de datos/resultados se modifica.

**Interfaces:** Consumir únicamente preflight y configuraciones aprobados, `pilot(out)` y `summary.json`; producir informe de coste y siguiente decisión presupuestaria, no una cola.

- [ ] **1. Gate de lanzamiento:** Confirmar revisión sin bloqueos y rama g2m-v2. Congelar script/config y ejecutar una vez el preflight real sobre1498 ejemplos, con salida/log propios: comprobar counts, integridad y ausencia de fallos SPD. Si falla, cerrar bloqueado, sin GPU. Verificar hash de script/config igual al revisado; si hubo cambios relevantes invalidar explícitamente el preflight y conservar su intento, no reutilizarlo como vigente. Consultar estado GPU de nuevo, no basarse en el snapshot del18octubre. No ejecutar suites CUDA adicionales como entrenamientos fuera del tope: esta pareja es la comprobación CUDA autorizada.
- [ ] **2. Ejecutar** una sola vez `pilot --out ...` con supervisión activa; observar progreso/logs sin bloquear comunicación>60s. Al abortar, conservar la causa y parar, sin ajustar knobs ni reintentar. Al completar, confirmar120updates por brazo,100medidos, validación150, orden/LR y hashes iguales. No inferir test ni guardar checkpoint para futuros fits.
- [ ] **3. Cerrar recursos:** Confirmar salida terminal y ausencia de procesos propios/descendientes; releer fuentes para integridad después. Diferenciar tiempo de GPU, preflight CPU y limpieza. No usar lecturas históricas de VRAM como picos de esta pareja.
- [ ] **4. Documentar** comando reproducible, hashes/entorno, resultados por brazo y limitaciones: orden fijo sin réplicas, heads distintos, split correlacionado, sin OOD independiente, sin demostración de accuracy. Si bloqueado o incompleto, identificar qué artefacto/condición faltó; no afirmar éxito por exit0 del padre.
- [ ] **5. Actualizar** roadmap/traspaso con estado real, enlace al informe y coste medido; los seis fits continúan sin autorización. Pedir solo la decisión posterior pertinente: presupuesto total de seis fits si el piloto fue viable, o instrucciones ante el bloqueo concreto. No promover defaults ni declarar solucionado OOD.

## Autorrevisión del plan

Cobertura: §§1–2 del diseño→Task1/límites globales; §3→Task2; §4→Task1;
§5→Tasks2–3; §6→Tasks3–4; §7→se conserva la decisión conservadora de preflight.
Cinco clases de Review Focus tienen tests en sus tareas. Interfaces compartidas
son las nombradas aquí; CLI no expone nuevas rutas de datos, seeds ni presupuestos.
No se ordenan commits pese al formato general de la guía. La aprobación del plan
no certifica sus resultados: cada gate debe quedar registrado en el ledger de ejecución.
