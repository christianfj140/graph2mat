# Validación comparable de 4.7 en h2oclu — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Comparar tres candidatos con la penalización 4.7 existente contra los tres baselines fixed guardados, conservando el criterio efectivo de selección y la evaluación molecular full-basis.

**Architecture:** Un wrapper experimental nuevo reutiliza recuperación, supervisor, evaluación y append de cola existentes, sin modificar sus fuentes congeladas. Un único callback derivado de ModelCheckpoint mantiene la penalización activa en entrenamiento y la desactiva temporalmente en validación para conservar el best-val_loss original. Los proyectores de S y las referencias de datos se preparan en un overlay local de solo lectura hacia los inputs externos.

**Tech Stack:** Python del entorno compartido existente, NumPy/SciPy/sisl, PyTorch/Lightning y helpers de auditorias. Sin dependencias nuevas.

**Spec:** Alcance propuesto en `auditorias_mejoras/informe_diagnostico_h2oclu_ood_20261017.md`, sección «Siguiente intervención mínima», autorizado mediante «Lo autorizo. Hazlo». Plan aprobado por el usuario mediante «Apruebo» antes de implementarlo.

## Global Constraints

- Checkout `<graph2mat>`, rama `g2m-v2`, sin commits/worktrees; preservar cambios previos.
- Solo tres fits científicos nuevos, semillas0/1/2; reutilizar fixed_s0/1/2 únicamente mientras pase comparabilidad. Ninguna referencia nueva sin preguntar.
- Dataset h2oclu existente:32train/4validation/3OOD; geometrías, etiquetas, H0, bases, pseudos y splits intactos. No escribir en otro repositorio salvo los tres appends autorizados a su cola existente.
- Baseline científico cerrado: `auditorias_mejoras/experimentos_batching_h2oclu_20261016_science_v1/`; manifest SHA256 `da7a0e3a1268cb9504f2ede4dd0b3cfe1df263294cb5ff21324949e26ac0ddca`.
- Receta fixed preservada:batch8,2000épocas,64000exposiciones/8000updates, mismos estados iniciales completos y orden seed/época; MACE/FP32/compile/4workers/cache/Adam originales, cosine2000/eta_min1e-5.
- Tratamiento único: `spectral_window="spectral_null.npz"`, `gauge_invariant_mse(spectral_weight=1e-4)` en train. Candidato histórico fijo, sin barrido ni selección por OOD.
- Columnas C=U_low/sqrt(max(λ,1e-3)), λ<1e-2, como la opción4.7 histórica; padding cero uniforme. Exigir floor inactivo y CᵀSC=I en columnas no nulas para este dataset; si no se cumple, detener preparación. No eliminar direcciones al evaluar espectros.
- Misma selección efectiva:minval_loss de gauge básico, cada5épocas, sin término espectral en validación. No basta que dos callbacks monitoricen el mismo nombre con pérdidas diferentes.
- S de referencia solo como etiqueta; evaluar H0+Δ en eV, mismo soporte/gauge, niveles/gap generalizados full-basis. No predicción de S ni claims sobre derivadas/fonones.
- Reutilizar guardas:cesión ante GPUajena, recuperación de época completa, allocator≤55%, abortar grupo propio si libre<25%/GPU≥80°C/watchdog. Son sondeos, no garantía dura de20% libre.
- Sin defaults, fuentes congeladas, autorun, campañas anteriores, pesos adicionales, datasets, a-SiO2 ni fonones modificados. No autoampliar campaña si el resultado es negativo.

## Review Focus

1. Cambiar la pérdida de validación sin advertirlo cambia el experimento: comprobar su valor global literal y el best seleccionado con Trainer real.
2. Una excepción/resume durante validación no debe dejar spectral_weight=0 al entrenar; callback idempotente/restauración probada.
3. El overlay no debe cambiar orden de muestras, etiquetas, grafo, layout existente o inicialización; comparar tensores nativos con el baseline.
4. C no puede tener orden orbital/tamaño/padding incorrectos ni depender de etiquetas H OOD; comprobación independiente de S y proyector.
5. Enqueue parcial o relanzamiento no debe duplicar IDs, regenerar inicializaciones ni sobrescribir resultados; recibos/hashes/no-clobber.

## Evidencia de auditoría inicial

627hashes originales y tres `validated_result` pasan. Cada fixed conserva2000épocas,
64000muestras y siete evaluaciones. Fuentes productivas/versiones originales intactas.
Inicializaciones por seed:

- 0:`aee243eeb893096c8d5dd696595a0506a920a38eefbb6029c1b3810fa113d1a8`.
- 1:`aa4432036e19a7b52ad3dde7ffba1f039b293292d9c52de3acb1bc4ebcb25288`.
- 2:`7ce43e36387fab2ec5c42befc9b4aacb3ca75e22184e1af0cdb84152c29e47ce`.

La API actual comparte loss_fn entre train/val; configurar solo spectral_weight
cambiaría val_loss. El runner congelado admite un único callback ModelCheckpoint
derivado y RecoveryCallback lo reconoce mediante isinstance: no es necesario
editar el modelo, el supervisor ni la política nativa de recuperación.

### Task 1: Selección comparable y proyectores locales

**Files:** Create `auditorias_mejoras/h2oclu_nullspace_callback.py`,
`auditorias_mejoras/test_h2oclu_nullspace.py`.

**Interfaces:** `NullspaceTrainingCheckpoint(spectral_weight=1e-4, **ModelCheckpoint_kwargs)`;
`projection_columns(S, padded_columns)` devuelve C float64 con ceros de padding.

- [x] Escribir tests RED: control analítico de S diagonal/rotada, orden, padding y CᵀSC; inputs no finitos/indefinidos rechazados.
- [x] Test RED con TrainerCPU real: train usa1e-4, validación usa0, val_loss coincide con gauge básico global; misma predicción de control sin penalización y sin invocar otra loss de producto.
- [x] Test RED de excepción/sanity/resume: restaurar peso1e-4, kwargs de hparams intactos y pesos/estado compatibles con carga por la clase MACE base.
- [x] Implementar callback: on_validation_start desactiva solo metric_kwargs del término existente; on_validation_end ejecuta ModelCheckpoint con val_loss básico y restaura en finally; on_exception restaura también. Rechazar losses/stages no soportados.
- [x] Implementar proyector con scipy/numpy existentes y registrar rango/floor/identidad. No añadir una pérdida nueva.
- [x] Ejecutar tests GREEN; prueba real de un paso con datos existentes en CPU, sin OOD para preparar o elegir el término.

### Task 2: Wrapper de campaña y comparación congelada

**Files:** Create `auditorias_mejoras/run_h2oclu_nullspace.py`; extend new test file only.

**Interfaces:** `prepare(root)`; `read_campaign(root)`; `run(root,key)`;
`evaluate(root,key)`; `summarize(root)`; `enqueue(root)` con recibo propio.

- [x] Tests RED de fuente/resultado baseline cambiado, autorización ausente, destino existente/symlink, orden de IDs y enqueue duplicado/parcial.
- [x] Tests RED de overlay:32train/4val; mismas etiquetas y tensores nativos que baseline salvo spectral_C. No crear C ni etiquetas de entrenamiento a partir de H OOD.
- [x] Preparar referencias locales a FDF/Δ/assets existentes y36C nuevos, sin modificar originales; preservar exactamente el orden de las listas originales de inputs, no asumir que coincide con el orden del manifest.
- [x] Reutilizar `make_job/read_job/guarded_fit`, fingerprint de archivos/versiones y mismas inicializaciones. Cambios permitidos de config:overlay/root/basis-path equivalentes, spectral_window, loss_kwargs, callback derivado y output propio.
- [x] Congelar wrapper/callback/C/config/jobs, tres baselines/recibos/checkpoints/resultados y fuentes previas. Verificar antes/después de preparar y antes/después de cada fit/evaluación.
- [x] Reutilizar evaluación CPU `h2oclu_batching_metrics.evaluate_checkpoint`:21matrices candidatas, cuatro ID+tres OOD por fit, frente a21baselines ya guardadas. No recalcular predicciones de referencia.
- [x] Resumir por estructura/semilla y media de seeds: H-MAE alineado, RMSE niveles y error absoluto gap; ratio pareado y27resamples ordenados/IC95, fallos de simetría/S explícitos. Auxiliares/p95/p99/soporte y costos conservados.
- [x] Guardar tiempo, allocated/reserved/PSS e interrupciones con el ámbito del runner. No afirmar ahorro causal de velocidad/VRAM frente a baselines no contemporáneos.
- [x] Ejecutar GREEN y smokes de carga/checkpoint/selección/recuperación; ante fallo de comparabilidad detener sin lanzar baselines nuevos.

### Task 3: Gates, cola y traspaso

**Files:** Create `auditorias_mejoras/campana_nullspace_h2oclu_20261017.md`;
modify only roadmap/traspaso as documentation. Runtime artifacts in a new
`auditorias_mejoras/experimentos_nullspace_h2oclu_20261017_v1/`.

- [x] Revisión final independiente única Native según skill de ejecución; corregir hallazgos importantes con regresiones, sin rondas ni agentes de implementación por tarea.
- [x] Focales nuevos+campaña/recuperación; Ruff/diff y suiteCPU con fallos históricos dscribe separados. No esconder cuatro HistorySOAPs ni instalar dependencias.
- [x] Registrar tres IDs, hashes, primarias, candidato fijo y criterio antes de ver resultados nuevos. Evidencia exploratoria:OOD ya observado, lineage incompleto, tres seeds.
- [x] Tras aprobación del plan y gates, append tres IDs únicos, secuenciales y detrás de dependencias vivas, con helper/lock/recibo existente. No reiniciar daemon, reordenar cola o repetir enqueue.
- [x] Confirmar recibo completo y arranque/espera real. Cada job hace fit recuperable→evaluaciónCPU→summary antes de liberar el siguiente.
- [x] Al terminar, auditar21matrices nuevas y tres checkpoints, registrar resultados favorables o negativos y propuesta/no promoción. No cambiar defaults automáticamente.

## Criterio científico y duración

Primaria de beneficio físico: reducción pareada del error del gap OOD; publicar
también H y niveles aunque empeoren. Señal robusta requiere IC95 superior de
razón<1, y se señalan todas las regresiones por seed/estructura, especialmente
>5% en H/niveles. No confundir un trade-off útil con promoción general.

Tres baselines existentes consumieron78,70min/fit de media. Estimación inicial
**4–6h exclusivas para los tres candidatos**, más preparación/gates y esperas
ajenas sin cota; el overhead de4.7 con esta receta aún no está medido. Actualizar
con el primer fit real, sin presentar una estimación como resultado.

## Estado de aprobación y ejecución

- [x] Usuario autoriza validar4.7: «Lo autorizo. Hazlo».
- [x] Auditoría de reutilización de tres baselines cerrada, sin relanzarlos.
- [x] Plan redactado y auto-revisado; alcance, firmas y riesgos concretados.
- [x] Revisión/aprobación de este plan antes de implementación/cola: usuario«Apruebo».
- [x] Tasks1–2 implementadas,26tests y smoke CPU real de interrupción/reanudación/carga pasan.
- [x] Task3: auditoría científica/documental completada2026-10-18; tres fits y21matrices nuevas, sin promoción.

Cierre2026-10-18:42matrices incluyendo21fixed,982archivos auditados y Cholesky
independiente≤2,29e-12eV, sin cortar S. GapOOD−61,66%,niveles−69,25%,pero
H OOD+13,47%/ID+19,93%; campaña cerrada como compromiso exploratorio, no OOD
resuelto. Tres IDs`:bootstrap1` DONE; seed0 se reutilizó sin reentrenar tras
el fallo de import del evaluador.33focales pasan39,03s con preload; general
vigente2653/34skip/4dscribe históricos942,67s. Evidencia autoritativa
`experimentos_nullspace_h2oclu_20261017_v1/closure_audit_20261018.json`, comando
reproducible e informe en `campana_nullspace_h2oclu_20261017.md`.

Encolado2026-10-17 a18:42CEST: cola1409→1412, recibo completo/prefijo intacto.
Manifest científico de03dcd283564d1504905dca1a189af1fac8829c0281a5bc83dbe07d2aa8ece3,
925hashes. Primero null_s0, supervisor912675/hijo fitGPU912694, daemon1756952
sin reinicio; null_s1/2 esperan secuencialmente. Gates:26nuevos pasan, focal
150pasan/5CUDAomitidos; general2653pasan/34omitidos/solo4HistorySOAPs por dscribe,
910,30s. No marcar Task3 completo al encolar: faltan resultados/auditoría21matrices.

Se conservó el método Native: implementación en este checkout, sin agentes
por tarea ni commits. Una revisión final independiente, sin implementaciones
paralelas ni segunda ronda de revisión.
