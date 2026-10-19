# Auditoría de cierre — recuperación y batching6.4 — 2026-10-17

## Objetivo sin reducir alcance

Dos entregables: recuperación completa desde frontera de época verificada y
comparación científica fixed/cost en h2oclu32/4/3. No se exige replay intraépoca
exacto, mejora positiva de precisión ni promoción. Un resultado negativo de
generalización sigue siendo una validación, no autoriza nuevas recetas.
La vista igualtiempo quedó explícitamente separada y sin trabajos adicionales
en la propuesta aprobada; no sustituirla por igualar updates ni fingir que se hizo.

Repositorio/branch actuales: `/home/christian/repositorios/graph2mat`, `g2m-v2`,
HEAD270d17d1f204293a059baa09e8f1b3756981e2bf. Checkout sucio previo preservado.
Reanudación del Goal comprobada active en esta continuación. No se crea otro Goal.

## Matriz de requisitos y evidencia autoritativa

| Requisito | Evidencia inspeccionada y alcance | Estado |
|---|---|---|
|Diseño/plan aprobados antes de implementar|spec/plan2026-10-16-training-recovery; ledger con autorizaciones textuales y seis tareas cerradas|Demostrado|
|Checkpoint nativo completo, no solo pesos|training_recovery.py RecoveryStore/Callback; estados de pesos/buffers, Adam, cosine, loops/callbacks/RNG/scaler; pruebas de continuidad|Demostrado en contrato soportado|
|Atomicidad, integridad, corrupción/incompatibilidad|generaciones fsync/rename, dos retenidas, huella y checksums antes de pickle; tests de caída/fallback/allinvalid/symlinks/locks|Demostrado localmente; no garantía de driver/SO hostil|
|Best-val_loss separado/protegido|store copia best por generación, callback restaura score/historial; test de best perdido tras mejora no confirmada; seis best finales hashados|Demostrado|
|Orden/sample_id pareado y restore RNG|loader seed/época, frozen batchsampler, generators separados,0/4workers; CPU toy/MACE fixed/cost entre procesos, dos interrupciones, control negativo sin RNG|Demostrado CPU exacto|
|Ruta real GPU/AMP/compile|post-review-cuda.log3pasan157,02s; compiled-green.log1pasa204,99s, cost/4workers, dos controles/dos cortes,48admitidas/5repetidas|Demostrado con tolerancias previas, no bit-a-bit CUDA|
|Liberar solo recursos propios y guardas|guard PGID propio TERM10s/KILL, sin señales ajenas; pruebas de descendientes/cancelación/foreignGPU/watchdog; liberaciónGPUcompilada≤1,621s observada|Demostrado; latencia no garantizada SO/driver|
|Contabilidad acumulativa/repeticiones|attempts fuera de estado rebobinado; cotas explícitas si heartbeat/corrupción; tests RED→GREEN; campaña6intentos exitosos,0repetidas,≈0,32s espera|Demostrado, no cota superior exacta tras corte|
|No dumps periódicos inseguros|fuentes congeladas no añaden dump_traceback_later; runner7.2 intacto; regresión previa y smokes retenidos|Demostrado para estas rutas|
|Trabajos concretos/presupuesto aprobados antes de cola|usuario«Autorizo esos6trabajos»/«Continua»; propuesta6fits2000épocas,cosine explícito; seis entradas únicas y prefijo anterior intacto|Demostrado|
|Datos/bases/convenciones/splits congelados|39referencias/H0 y36Δ; preflight Δ=H−H0≤8,9e-16eV, S idéntica, SIESTA homogéneo;627hashes actuales y resolución/orden de inputs pasan|Demostrado; lineage incompleto declarado|
|Seis fits pareados completos|native receipts2000épocas/64000muestras cada uno, épocas consumen32;8000vs8139updates; pesos/buffers iniciales0/1/2 por hash; configs iguales salvo budget/root|Demostrado|
|Selección de presupuesto sin mirar OOD|dos rondas de pilotos5épocas, regla mayor budget que reduce allocated;326283 elegido antes de predicciones científicas|Demostrado; pilotos no son evidencia de precisión|
|H físico gauge y espectros/gap ID/OOD|42CSR H0+Δ hashados,7casos/fit; unión Href/Hpred/Sref, c común; FDFH/O; eigensolver generalizado con Sref y128ocupados OOD|Demostrado; Sreferencia condicional|
|Incertidumbre y comparación justa|tres seeds, medias/ratios pareados, exact27resamples/IC95 recalculados; muestras iguales, updates+1,7375%, tiempo no igualado|Demostrado exploratorio, no claim general|
|Tiempo/VRAM/RAM/recuperación|seis fit.json, epochs y telemetry.json completos; allocated/reserved, RSSsum/PSS, setup/publicación/espera/evaluación;27798,14s fits y5590,74s publicación incluida|Demostrado para ámbito de telemetría|
|Pruebas/gates/revisión|revisión Native final única por subplan, Important corregidos RED→GREEN; general2789pasan/39omitidos/4dscribe históricos; revalidación136pasan/5CUDAomitidos248,95s; Ruff/mypy23/diff limpios|Demostrado; suite general no totalmente verde|
|Documentar incluso resultados negativos|campana_batching_h2oclu_20261016.md incluye seis primarias, IC, costes, gaps colapsados y sensibilidad; roadmap/protocolo/traspaso actualizados|Demostrado|
|No defaults/fonones/autorun/externos/dependencias/commits/worktrees|sin edits de src ni fuentes congeladas en cierre;627hashes, HEAD/branch intactos; cola solo seis appends autorizados; campañas7.2 no relanzadas|Demostrado dentro del alcance; no auditoría global de cambios ajenos|

## Contraste independiente de datos finales

Script de verificación, no API nueva:
`.superpowers/sdd/2026-10-16-h2oclu-six-approved/closure-audit.py`.
Verifica fronteras de checkpoint nativo y optimizador/scheduler efectivos,
hashes/coverage de seis resultados y42matrices, identidad de configs/initials,
gates crudos de simetría/S, ocupación y recálculo mediante Cholesky.
Incluye contraste analítico de3niveles y desplazamiento cS, antes de datos reales.
Salida `experimentos_batching_h2oclu_20261016_science_v1/closure_audit_20261017.json`:
statuspassed,627fuentes/inputs,6resultados,42matrices, mismas medias e IC.
SHA256 del script de auditoría final:
`a1ac19911fe2f748edc0b8be56b03fc7bf6b796dd89f30b17f3353e7ed6dd23a`.
No sobrescribe summary/result/selección ni usa predicciones para retocar presupuesto.

## Conclusión que sí soportan los datos

En estos seis fits, cost326283 baja allocated15,66%,reserved25,55%,PSS10,24%
y pared3,77%, con64000exposiciones por brazo/seed. Tiempo y memoria no son
resultados de kernels aislados, sino de este pipeline con recuperación activa.
H OOD−1,14% es pequeño/inconcluso; nivelesOOD−30,81% de media dependen de una
semilla y empeoran en dos; gapOOD mantiene error medio≈2,98eV. No promoción.
Reproducción independiente confirma que el fallo de gap pertenece a estas
matrices/política, no a diferencias entre dos solvers. Direcciones de pequeñoS
amplifican residual; cortar1e-2 reabre gaps pero elimina136–141direcciones y cambia
la referencia. Solo diagnóstico, nunca reemplazo de la métrica predeclarada.

## Límites que no invalidan el cierre acotado

- OOD ya observado, snapshots correlacionados, lineage incompleto y solo3seeds:
  evidencia exploratoria, no test independiente ni mejora universal.
- S no predicha; no conclusión de H+S aprendidos, derivadas o fonones.
- No DDP, rotatingpools, augmentations estocásticas, otros optimizadores/schedulers
  ni replay intraépoca exacto certificados. No migrar checkpoints históricos.
- No cesiones reales nocturnas: los cortes/corrupción se prueban en certificados,
  no se presentan como ahorros reales de trabajo en esta campaña.
- PSS de árbol fit+descendientes, no supervisor/evaluación ni RAM total; RSSsum
  sobrecuenta páginas compartidas. Carga/caché/hardware no aislados.
- No benchmark sin recuperación para restar overhead causal. Publicación de
  recuperación está incluida en pared, no se suma otra vez ni se llama GPU-hours.
- Minor documental previo sigue diferido: caption timing_scope dice genéricamente
  excludes checkpoint save; wall de época puede incluir best científico. Informes
  aclaran scope correcto; fuente congelada se preserva, duraciones no cambian.
- No nuevo cutoff/pérdida/default ni campaña igualtiempo autorizados. Los resultados
  negativos quedan registrados; no hace falta entrenar hasta lograr un ganador.

## Gates y cierre

Revalidación CPU final:136passed/5skipped,248,95s. Comando:
`CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:auditorias_mejoras OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python -m pytest auditorias_mejoras/test_training_recovery.py auditorias_mejoras/test_cost_batching_comparison.py auditorias_mejoras/test_recovery_certification.py auditorias_mejoras/test_h2oclu_batching_science.py auditorias_mejoras/test_bilayer_edge_updates.py auditorias_mejoras/test_bench_pipeline_gpu.py -q --tb=short --disable-warnings`.
Log: `.superpowers/sdd/2026-10-16-h2oclu-six-approved/closure-focused.log`.
Los5omitidos son perfilesGPU con permiso/guard explícito; pruebas CUDA ya
certificadas en snapshot técnico igual, no se relanzaron en este cierre CPU.

Ruff `src tests`+nueve fuentes experimentales:All checks passed. Mypy con
pyproject.toml:Success,no issues found in23sourcefiles. Diff--check sin errores.
Suite general previa reutilizada porque fuentes productivas/experimentales y
versiones siguen iguales:2789passed/39skipped/4fallos históricos,1123,58s.
No se afirma que se haya repetido de nuevo la suite general en este cierre.
Casos exactos: `src/graph2mat/tools/server/tests/test_extrapolation.py::test_real_extrapolation_with_missing_onsite_and_edge_blocks`
`[False-1-HistorySOAPs]`, `[False-2-HistorySOAPs]`, `[True-1-HistorySOAPs]`,
`[True-2-HistorySOAPs]`, todos por dscribe ausente. No instalarlo ni ocultarlos.

Requisitos de este Goal satisfechos con los límites anteriores y resultados
negativos publicados. Última comprobación13:34CEST:627hashes/versiones pasan,
seis IDs DONE, HEAD/branch intactos y diff--check limpio; lectura real de /proc
confirma que no quedan procesos run/fit/eval de esta campaña. Cierre listo para
actualizar servicioGoal a complete. Esto no cierra el roadmap global ni autoriza
otra campaña; siguiente ampliación requiere decisión del usuario.
