# Recuperación experimental de entrenamiento — 2026-10-16

Estado: **punto1 técnico cerrado**, implementación, revisión independiente y
gates finales completados. **2026-10-17: punto2 científico6.4 terminado y auditado**:
seis fits completos, memoria favorable y precisión OOD mixta, sin promoción.
Ver `campana_batching_h2oclu_20261016.md` y `cierre_goal_recuperacion_batching_20261017.md`.
Suitegeneral tiene cuatro fallos históricos, no es
completamente verde. Fuentes nuevas limitadas a auditorias.
Checkout `graph2mat/g2m-v2`, HEAD270d17d1f204293a059baa09e8f1b3756981e2bf;
sin commits, defaults, cola, datasets externos ni runner7.2 modificados.

Actualización2026-10-17: la ausencia de escrituras de cola describe el cierre
técnico anterior. Después se autorizaron y encolaron exactamente seis fits;
ver `campana_batching_h2oclu_20261016.md`. Las referencias inferiores a permiso
pendiente son históricas. La pausa del usuario no detuvo fits; tras reanudación
se auditaron resultados y se cerró la documentación, sin otros trabajos.

## Qué garantiza y qué no

Runner opt-in nuevo: recuperación de última **época completa** mediante checkpoint
nativo de Lightning, no una carga de pesos para reiniciar desde cero. Conserva
parámetros/buffers, cinco grupos Adam nativos, cosine por época, loops/contadores,
callbacks, RNG Python/NumPy/TorchCPU/CUDA y scaler cuando procede.
Best científico por menor val_loss separado, en ruta estable y con copia fijada
en cada una de las dos generaciones retenidas. Una mejora no confirmada no
destruye la selección que corresponde al rollback.

Atomicidad: save temporal, validación CPU, hashes/tamaños/fsync, rename de
generación y después índice. Generación publicada recuperable sin índice;
corrupción permite fallback registrado. Huella distinta bloquea antes de cargar
pickle. Todas inválidas bloquean; no se oculta corrupción reiniciando.
Locks separados de supervisor y escritor; rutas propias sin symlinks.

Solo un dispositivo, datos cacheados deterministas, Adam, cosine por época,
optimización automática y acumulación1. No DDP, rotating pools, augmentations
estocásticas ni recuperación intraépoca exacta. Fixed/cost comparten orden
seed/época e inicialización, **no** se exige igualdad de sus pesos finales:
agrupan distinto y hacen distinto número de updates.

Loader recargado cada época en ambos brazos; generadores de sampler/loader/val
separados. Fixed necesita plan inmutable: prefetch y reapertura de iteradores
provocaron una regresión real con4workers, corregida RED→GREEN. Workers ya no
persistentes en este runner: evita retención atexit con pinning/reload y no
cambia defaults públicos. Su sobrecoste entra en medición pareada.

Supervisor cede al detectar cómputo ajeno en sondeo, sin señalizarlo. Cierre OWNgroup
TERM/10s/KILL, incluido workers; cap allocator55%, reserva inicial30%, aborto
propio libre<25%/80°C. No garantía dura frente a asignaciones ajenas entre sondeos
ni kernels no interrumpibles. Solo ForeignGPU habilita retry; integridad,
configuración, OOM, NaN y watchdog fallan visiblemente.

## Evidencia

Ledger/logs y fixtures retenidos, sin depender de commits:
`.superpowers/sdd/2026-10-16-training-recovery/`.

- Tasks1–4:84tests antes de certificador,22,87s; atomicidad/fork, corrupción,
  symlinks, retención, fallback, best temprano y último commit sin fit.json.
- Primer certificado CPU:10pasan/3CUDAomitidos,228,91s. Ocho combinaciones
  toy/MACE × fixed/cost ×0/4workers, cuatro intérpretes nuevos cada una; dos
  interrupciones. Igualdad exacta de pérdidas/tensores/Adam/cosine/loops/RNG y
  trazas reales admitidas. Control negativo sin restore_rng falla por diferencias
  numéricas, no por import. Se repite tras cambiar política de workers.
- Primer certificado GPU:3pasan,160,47s. MACE real FP32/FP16/BF16, cinco procesos
  propios por perfil, dos controles continuos antes de comparar resume y dos
  interrupciones supervisadas. ScalerFP16 y CUDARNG comprobados. Umbrales previos:
  FP32rtol/atol5e-6; AMP1e-2/1e-3. Scores derivados usan ese mismo techo;
  contadores, optimizer.step, RNG y estado del precision plugin son exactos.
- Medición GPU: regresión real detectada en piloto (picoNone tras teardown de
  Lightning); prueba CLI nativa GPU RED→GREEN. Picos capturados en on_train_end,
  reset en on_fit_start. No inferir GPU de parámetros ya movidos aCPU.
- Repetición final tras workers no persistentes: focal107pasan/4omitidos,260,40s;
  fuente runner/recovery75pasan/1GPUomitido,29,79s. CertificaciónGPU3pasan,158,64s.
  Latencia máxima de liberación propia0,869s en esas cesiones controladas,
  48muestras admitidas/5repetidas por perfil. ScalerFP16 comprobado. Fixtures,
  checkpoints y reportes retenidos en final-focused/ y final-cuda/.
- Suitegeneral antes del fixpass de revisión:2734pasan/37omitidos,1221,65s;
  exactamente cuatro fallos históricos por `dscribe` ausente, sin instalarlo:
  `test_real_extrapolation_with_missing_onsite_and_edge_blocks` de
  `src/graph2mat/tools/server/tests/test_extrapolation.py`, parametrizaciones
  `[False-1-HistorySOAPs]`, `[False-2-HistorySOAPs]`, `[True-1-HistorySOAPs]`,
  `[True-2-HistorySOAPs]`. No presentar suite completamente verde.
- Suitegeneral postreview:2744pasan/39omitidos/los mismos4HistorySOAPs,
  1228,04s. Gatefocal definitivo posterior al guard de enlaces:115pasan/
  5omitidos,254,39s; task-done5 registrado. Focal anterior114pasan/5omitidos
  no incluía la nueva regresión de resolución. Producto src/tests sin cambios
  durante ambas pasadas; esa regresión adicional está cubierta en gatefinal.
- Ruff src/tests+seis fuentes nuevas limpio; mypy23fuentes limpio;
  diff--check limpio. Revisión independiente única:0Critical,3Important,0Minor.
  Contabilidad tras cancel/corrupción/heartbeat y validaciónNaN/Inf corregidas
  con6regresiones RED→GREEN. PerfilGPUFP32 compilado/cost/4workers pasa204,99s,
  48admitidas/5repetidas, liberación máxima1,621s. Dos controles continuos y dos
  interrupciones entre cinco procesos nuevos; mismo techo5e-6 previo. Rerun
  completo postreview y pilotos finalesv4 terminados. GPU eagerFP32/FP16/BF16
  después de fixes:3pasan/157,02s. Guard de resolución/orden de inputs añadido
  durante misma pasada de fixes: retargetreal Δ→H evadía hashes canónicos;
  regresión RED→GREEN,8pruebas relacionadas pasan/1GPUomitido/13,02s.
  Sin Critical/Important pendientes; no second review. Revisor:0Minor; una
  aclaración menor de caption de telemetría detectada por autor se difiere abajo.

Manifest científico preparado y6pilotos técnicos terminados; ver
[propuesta](propuesta_batching_h2oclu_20261016.md). No sustituyen validación
científica. Pilotos finalesv4:341,32s, mismos picos/selección quev2/v3; fuentes,
pesos y costes congelados y verificados. Medición incluye recuperación;
checkpointing1,81–2,43s/5épocas,
no hay comparación independiente de overhead frente a desactivar recuperación.

Contabilidad acumulativa permanece fuera del checkpoint rebobinado. Contadores
confirmados vienen del estado nativo; heartbeat puede preceder la caída, por eso
trabajo repetido es una cota inferior, no cifra exacta inventada. Pared de todos
los intentos y esperas separada; no llamarla GPU-hours de utilización.
Epoch wall sincronizada incluye carga/validación y puede incluir el best científico
guardado en on_validation_end con cadence5. Excluye publicación de recuperación
posterior; `checkpoint_seconds` mide esta última, no todos los guardados nativos.
Setup y primera época pueden incluir compilación lazy;
no se ha aislado un tiempo de compilación puro. Telemetría del supervisor mide
VRAM total, temperatura y RSS/PSS del árbol propio.

## Hallazgo científico que requiere autorización

YAML histórico h2oclu declaraba CosineAnnealingLR, pero LightningCLI lo ignoraba
sin optimizer explícito; checkpoint histórico tiene lr_schedulers=[] y LR0,0018
constante en cinco grupos. Runner nuevo aplica cosine explícito al Adam nativo,
conforme al diseño técnico aprobado. **No es la receta efectiva histórica.**
Comparación futura requiere baselines fixed nuevos y aprobación de esta receta,
además de trabajos/presupuesto. No se atribuye mejora de precisión/velocidad/VRAM
a la recuperación ni a los smokes sintéticos.6.4 científico sigue sin ejecutar.

Decisiones y costes de desviaciones documentados como `Ruling:` en ledger
y en el anexo inferior. Directorio preservado porque no hay commits
autorizados que puedan reemplazar esta evidencia.

## Decisiones de ejecución y coste si fueran incorrectas

En orden del ledger; todas permanecen visibles, incluidas las sustituidas:

1. Checkout existente/sin instalar ni worktree: preferencia expresa; coste,
   menos aislamiento y obligación de preservar dirtyfiles.
2. Conservar workspace/logs: commits prohibidos; coste, espacio local adicional.
3. Locks supervisor/escritor separados: evita deadlock padre/hijo; coste, dos locks.
4. Rangos BASE..HEAD no representan cambios: review de archivos nuevos/worktree;
   coste, paquete manual y hashes en vez de confiar en diff de commits.
5. Capturar seed en __new__: primer intento, sustituido por decisión7; coste
   demostrado, jsonargparse perdía parámetros del datamodule.
6. Restaurar RNG al comienzo de época, tras on_train_start: respeta startup
   repetido y hooks legítimos por época; coste, dependencia de orden certificado.
7. __init__variádico sin wraps y firma explícita sustituye __new__: CLI real
   prueba introspección; coste, adaptador experimental dependiente de parser.
8. Cosine explícito sobre Adam nativo: cumple diseño técnico; coste, receta
   efectiva distinta de historia, necesita aprobación científica.
9. PilotCLI en Task6: necesita manifest/dataset/presupuesto; coste, Task4 por sí
   solo no produce piloto ejecutable.
10. Revisión/gates después de prepararTask6: revisa todo runner; coste,
    Task5 permanece abierto hasta cierre global.
11. FrozenFixedBatchSampler: evita reapertura/prefetch que cambiaba permutación;
    coste, plan inmutable adicional en memoria.
12. Picos medidos/reset en hooks antes de teardown: CPUposfit no identificaGPU;
    coste, nueva ventana experimental y repetición de pilotos.
13. Workers no persistentes con reload por época: evita recursos atexit retenidos;
    coste, arranque de procesos cada época medido en ambos brazos.
14. Cero jobs igualtiempo adicionales por ahora: no igualar updates con compute;
    coste, vista temporal y su scheduler permanecen pendientes.
15. Paquete review manual tras helperrechazar rango vacío: sin commits artificiales;
    coste, revisar fuentes nuevas directamente.
16. Precisión/promoción y vista temporal no juzgadas por reviewer: sin claim ni
    permiso científico; coste, Goal abierto.
17. Sin reproducción abinitio independiente por reviewer: audit39referencias y
    hashes por preparador; coste, evidencia exploratoria/condicional.
18. Compilación elevada a Important y probada: no se deja fuera; coste,
    ensayoGPU acotado adicional.
19. DDP/estocasticidad/otros optimizers/schedulers/intraépoca/migración fuera del
    contrato: no prometer soporte; coste, alcance reducido.
20. Powerloss/raceshostiles/driverininterrumpible sin garantía dura: fsync/rename
    y ownership locales probados; coste, riesgo residual de plataforma.
21. Dirtyfiles/campañasanteriores/dependencias/cola fuera del review: preservados;
    coste, no auditoría global del repositorio.
22. No rerunsGPU/suite por reviewer: inspección de logs y reproducciónCPU,
    executor repite gates; coste, GPU no reproducida por observador independiente.

Minors del revisor: ninguno. Minor detectado por autor y diferido: caption JSON
`timing_scope` dice genéricamente «excludes checkpoint save», pero con cadence5
epoch wall puede incluir best científico. La separación correcta está descrita
arriba; no altera duraciones, presupuesto ni picoGPU. Corregir caption en próxima
versión del runner, preservando hashes de estos artefactos finales.

## Snapshot final

SHA256:

| Archivo en auditorias_mejoras | SHA256 |
|---|---|
| training_recovery.py |43bc0868793fec46cbdce800ae0e19b2311d2e5e40f3eeb8b979d97321b33a64|
| run_cost_batching_comparison.py |60aca040bdb56ccc480af4c1f74c08650e7610a1f0156073e398cac8e8fc838f|
| recovery_certification.py |9c8db753f2fccda2eb3021de2aaead5e094a8d12bb13f6e4f374f33ee6fd7896|
| test_training_recovery.py |bc8e10db2569afaaf7dfb4477f8f19e1f7f38022d71f5d6df7444e7a988a86b5|
| test_cost_batching_comparison.py |ed0084323db8213c76751b19e7a7492ac72b906bd9d814048aafb71dc4a42d7d|
| test_recovery_certification.py |1183a1565baed50b291adeba57bb9ef5e0b93ae834b72b0d27769e6f4269bb34|

Wrappers congelados7.2/bench_pipeline y coredata conservan sus tres hashes
registrados en preflight. HEAD intacto. Seis generaciones finales y sus bests
verificados en v4;160muestras confirmadas por job, sin extrapolar a ciencia.
