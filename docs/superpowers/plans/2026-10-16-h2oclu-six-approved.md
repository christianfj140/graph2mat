# Ejecución de los seis trabajos h2oclu aprobados

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox syntax for tracking.

**Goal:** Preparar, verificar y encolar exactamente los seis fits científicos autorizados.

**Architecture:** Un lanzador experimental nuevo reutiliza sin cambios el runner de recuperación certificado y sus inicializaciones v4. Evaluación CPU propia usa MatrixDataProcessor y SciPy; el daemon existente ejecuta una cadena de seis entradas, sin cambios de autorun.

**Tech Stack:** Python, Graph2Mat/Lightning/MACE, NumPy/SciPy/sisl y cola JSONL existente.

**Spec:** `auditorias_mejoras/propuesta_batching_h2oclu_20261016.md`.
**Autorización:** usuario «Autorizo esos 6 trabajos», seguido de «Continua». Método Native ya elegido. Este documento concreta ejecución de la propuesta aprobada, no añade experimentos ni solicita otra aprobación del mismo alcance.

## Global Constraints

- Checkout `<graph2mat>`, rama `g2m-v2`; sin commits/worktrees ni instalaciones.
- fixed/cost326283 × seeds0/1/2; 2000 épocas, 64000 muestras/fit, cosine2000/eta_min1e-5 explícito, Adam nativo, best-val_loss cada5.
- Datos/splits/initials y fuentes técnicas v4 intactos. No defaults, fonones, a-SiO2, generación de H0/Δ ni trabajos adicionales.
- GPU secuencial, guardas existentes de VRAM/temperatura/cesión; evaluación solo CPU, H0+Δ, S referencia condicional.
- Tres semillas y OOD observado: evidencia exploratoria; no promoción de defaults.

## Review Focus

- Preparación sin permiso o con piloto/input cambiado: rechazar antes de crear trabajos.
- Repetición de enqueue tras un append parcial: no duplicar ni cambiar dependencias/bytes previos.
- Evaluación con H desplazado por cS: alinear matriz y niveles con el mismo c.
- H no hermítico o S indefinido: no publicar espectros válidos tras simetrizar silenciosamente.
- Checkpoint y salida ajenos/cambiados: rechazar antes de cargar pickle o producir resultados.

## Task 1: Preparación y cola acotadas

**Files:** crear `auditorias_mejoras/run_h2oclu_batching_science.py` y `test_h2oclu_batching_science.py`.
**Consumes:** `make_job/read_job/guarded_fit`, manifest/pilot v4; `append_queue/live_dependencies/environment`.
**Produces:** `prepare(root, pilot_root, approved=False)`, `read_campaign(root)`, `enqueue(root, queue)`; seis configuraciones congeladas, recibo idempotente de cola y CLI prepare/enqueue/run/eval.

- [x] RED: tests sin autorización, fuente alterada, seis presupuestos/initials pareados, enqueue idempotente y continuación tras append parcial.
- [x] GREEN: copiar recetas/initials existentes, no regenerar estados; congelar lanzador/evaluador/config/job.json. Alternar orden de brazos por seed para limitar deriva temporal. Preparar recibo completo antes del primer append, encadenar trabajos.
- [x] Verificar: pytest del archivo nuevo; conservar logs. No commits.

## Task 2: Evaluación física y ejecución

**Files:** crear `auditorias_mejoras/h2oclu_batching_metrics.py`; ampliar test Task1.
**Consumes:** checkpoint best verificado por RecoveryStore y FDF/TSHS/H0 congelados.
**Produces:** `matrix_metrics(Hpred, Href, Sref, orbital_atoms, atomic_numbers)`, `evaluate(root,key)`, `summarize(root)`; per-structure y resumen pareado.

- [x] RED: ejemplo analítico gauge+gap, unión de soporte/falsos ceros, H no hermítico y S indefinido; métricas de un checkpoint nativo pequeño en fixture real.
- [x] GREEN: predicción CPU eager con processor nativo, reconstrucción float64 H0+Δ, guardar matrices propias y métricas. Ventana HOMO−2/LUMO+2, ocupación neutral H/O, umbrales y métricas de la propuesta, bootstrap pareado exacto sobre tres seeds.
- [x] Verificar: focal nuevo y recuperación existente, Ruff/diff; smoke CPU sobre best de piloto existente sin entrenar. Suite general registra por nombre fallos históricos.
- [x] Una revisión independiente final antes de encolar; arreglos Important/Critical RED→GREEN, sin segunda revisión.

## Task 3: Lanzamiento autorizado y traspaso

- [x] Preparar directorio científico nuevo, verificar fuentes y seis jobs, encolar usando append_queue; comprobar prefijo y recibo.
- [x] Comprobar daemon vivo, primer PID/progreso o causa de espera; registrar estado real en propuesta/roadmap/traspaso e informe de campaña.
- [x] Informar seis trabajos y estimación, sin afirmar resultados científicos antes de terminarlos. Conservar evidencias; Goal científico no completo al encolar.
