# Traspaso — Graph2Mat/g2m-v2 — 2026-10-15

## 3.12 implementado como guardrail de aviso — 2026-10-19

Usuario«Haz 3.12» autoriza solo la alarma antes descrita en chat. Hook ligero
en setup, antes de caché/pool, sobre train/val/test ya disponibles; predict
no dispara la comprobación. Fuentes repetidas, IDs/hashes/grupos explícitos
compartidos generan logs, no excepciones. Metadata ausente/inválida no permite
evaluar independencia; no confundir con contaminación ni certificar disjunción.
No leer matrices adicionales, cambiar splits/orden/semillas/defaults, bloquear
entrenamiento, inferir grupos por carpeta ni introducir umbrales geométricos.
No manifiesto, generador ni diseño estricto histórico implementados.

Código `tools/lightning/split_guardrail.py`, anotaciones opcionales en
`runs_json.split_metadata` o `sample.metadata.split_guardrail`; documentación
en `docs/data_loading.rst`. Soporte Subset/caché/pool con identidades originales,
rutas canónicas/alias léxicos, ZIP con miembro exacto y raíces temporales.
Hashes declarados, no leídos ni verificados. No GPU, fits, datasets nuevos,
autorun, commits o worktrees. Cambios preexistentes conservados.
Revisión única:3Important/1Minor corregidos con regresiones RED→GREEN.
Gate focal89passed; suiteCPU final3226passed/26skip/0fallos,48155warnings,
1226.05s, salida0; Ruff/diff correctos. No queda prueba/cálculo propio pendiente.
[Informe, tests y límites](informe_guardrail_splits_20261019.md).
Pendientes restantes:1.15/3.8/6.5/8.2/8.3 condicionados;3.13excluido ySOCaplazado.
7.6 permanece cerrado sin promoción; no reabrir campañas por este cambio.

## 7.6 cerrado sin promoción; decisiones de alcance — 2026-10-19

Usuario«Vale cierra7.6». Seis fits terminados4000updates/126110exposiciones
cada uno en39.223min. Evaluación CPU de checkpoints seleccionados por H-val
completa152.211s,150val+150test,1800predicciones sin labels en features,
malla400+camino151, espacio completo sin cortes de S. Test(media3seeds):
H raíz gauge-MSE1.980→14.182meV; niveles8bandas con S_ref en camino
34.173→400.576meV; empeora3/3seeds. Joint×1.731tiempo y×4.221allocated;
S_pred sin fallos SPD en lo comprobado, DM/EDM/carga diagnosticados aparte.
No tuning ni nuevos fits/defaults; conservar opción, no promover receta.
No demuestra inutilidad general de multitarea, convergencia ni OOD resuelto.

[Informe final](campana_multitarget_w90_20261019.md).
Artefactos nuevos `auditorias_mejoras/cierre_multitarget_w90_20261019_v1/`:
provenance+summary, seisrowsJSONL y1800NPZ. Entrenamientos/pilotos anteriores
intactos, no resetear ni relanzar started. No queda job GPU propio pendiente.
Evaluador mínimo nuevo +10tests CPU; revisión única encontró contracción CSR
entre nsc distintas, corregida RED→GREEN alineando imágenes sc_off explícitas.
Validación H reproducida con tolerancia: máximo relativo0.00228%, no bitwise.
HD−SE bruto guardado no es gauge-invariante ni energía total: no inferir
inconsistencia física de sus21–25eV/orbital, dominados por cero libre de H.
Suite CPU final ampliada **3205passed/26skip/0fallos**,48153warnings,1249.17s,
salida0; foco59passed/0fallos. Auditoría adicional de1800NPZ contra referencias
originales reproduce c/MSE/IDs y hashes; Cholesky/eigh≤1.409e-11eV.
Sin producto/defaults modificados. Cierre completo, no tareas/cálculos propios
pendientes de esta campaña; no reabrir ni relanzar sin nueva autorización.

**Decisiones expresas del usuario:**
- 3.13multi-GPU **fuera del alcance**, no hacerlo ni dejarlo esperando hardware.
- 8.1SOC **prioridad futura aplazada**. No diseñar/implementar/buscar datasets
  ahora; retomar solo cuando el usuario lo pida.
Otros pendientes entonces:1.15política default de checkpoint,3.8residual de ruta no usada,
3.12guardrail solo aviso (implementado después, ver cabecera),6.5disperso condicionado a perfil y8.2/8.3condicionados
a necesidad física/datos. Ninguno se autoriza automáticamente por este cierre.
Las cabeceras de lanzamiento siguientes son históricas, no estado de cola vivo.

## Seis fits7.6 autorizados — 2026-10-19

Usuario«Lanzalos» autoriza los seis entrenamientos completos descritos tras
el piloto v2: H-only y H/S/DM/EDM directos, seeds0/1/2,4000updates por fit,
misma receta/splits/pesos/backbone inicial/orden por seed; validación150 cada
200updates y checkpoint por H gauge-MSE, no pérdida total joint. No reabrir
el Goal técnico ni repetir pilotos. Lanzador experimental nuevo, sin editar
producto/defaults/datos/autorun; Native con gate CPU y revisión independiente.
**Lanzado y comprobado:** supervisorPID1869115, PGID/SID1869115, PPID1;
workerH seed0 PID1869163 activo, cinco fits pendientes. Más de600updates
reales, validaciones200/400/600 y checkpointbest-u0600 verificados. Estado
vivo `auditorias_mejoras/campana_multitarget_w90_20261019_v1/state.json`,
progreso por job. [Informe de campaña](campana_multitarget_w90_20261019.md).
Revisión única Important1 corregido RED→GREEN; gate80passed/0fallos,
Ruff/diff y28hashes productivos intactos. Suite CPU general adicional sigue
ejecutándose; no afirmar su resultado ni precisión final de los seis fits.
Telemetría inicial119lecturas: libre mínimo93.833%,Tmax41C, sin abortos.
[Protocolo de ejecución](../docs/superpowers/plans/2026-10-19-multitarget-w90-six.md).
Ledger recuperable `.superpowers/sdd/2026-10-19-multitarget-w90-six/progress.md`.
Tope conservador20min por fit/90min lote, seis secuenciales, sin reintentos;
abortos dejan los pendientes parados. Las prohibiciones históricas de lanzar
seis sin nueva autorización quedan satisfechas por este mensaje del usuario,
no por autorizaciones previas de pilotos. Evaluación científica CPU posterior;
todavía sin beneficio de precisión demostrado ni OOD independiente.

## Piloto7.6 v2 COMPLETO — 2026-10-19

Estado actual: usuario«Dale» autorizó parche PID+UNA nueva pareja15min.
Corrección focal,5regresiones RED→GREEN, gate123passed/0skip/0fallos,
32.52s; sin source productivo modificado. Preflight v2 completo1498/89.36s,
configs/orden/estados iniciales byte-idénticos a v1. Ambos brazos CUDA
terminaron120updates/100medidos/3146muestras y validación150. Contextos
liberados, sources/IDs/LR verificados y ningún proceso propio pendiente.

[Informe final v2](informe_piloto_multitarget_w90_20261019_v2.md).
H-only:mediana58.915ms,allocated698.903/reserved858MiB; joint:104.713ms,
2941.485/3384MiB. Joint×1.777tiempo,×4.209allocated,×3.944reserved; no ahorro
GPU frente a H-only. Libre mínimo conjunto86.092%,Tmax42°C. Es una pareja
exploratoria w90/spin1/fp32/directa, no precisión final ni rendimiento general.
Seis fits de4000updates:estimación36–48min,central39min; presupuestar≈1h de
train/val, evaluación científicaCPU aparte y no caracterizada aún.

**No lanzar/encolar seis fits sin autorización nueva.** No repetir v1/v2:
ambos tienen started y resultados preservados. V1 fue interrumpida; su
source/tests exactos están archivados en el workspace del plan. Runner
actual SHAe4c13d2c...; cierreCPU `closure-v2.json` y logs completos allí.
La corrección tolera solo el líder no recogido y, tras recogerlo, su registro
CUDA obsoleto sin /proc por≤2s dentro de900s; PIDs ajenos/reutilizados siguen
bloqueando. Mantener alarmas de splits, suelo DM/EDM y límites de S_pred/OOD.
Las cabeceras siguientes son historia: el bloqueo de v1 ya fue corregido.

## Corrección del PID y nueva pareja autorizadas — 2026-10-19

Usuario«Dale» autoriza corregir el falso positivo y repetir SOLO el piloto
acotado15min, no seis fits. V1 se conserva íntegra, incluido su source/tests
ejecutados (hashes originales) en el workspace del plan. Corrección mínima
del supervisor: líder Popen no recogido sigue siendo propio aunque no haya
RSS; tras recogerlo, exigir listaCUDA vacía y permitir solo su entrada
obsoleta sin `/proc` por≤2s dentro del mismo plazo900s. PID reutilizado o
ajeno se rechaza. Cinco regresiones nuevas RED→GREEN, gate focal en curso.
No cambios de modelo/entrenamiento/datos/defaults/autorun. Próximo intento
en directorio nuevo `piloto_multitarget_w90_20261019_v2`, nunca resetear v1.
Esta cabecera sustituye la espera de autorización de abajo; no repetir nada
si el ledger ya registra v2 terminada/abortada. GPU v2 aún no lanzada.

## Piloto7.6 interrumpido; NO relanzar v1 — 2026-10-19

Estado autoritativo: un intento GPU realizado tras preflightCPU1498estructuras
y gate118passed/0fallos. H-only terminó120updates,100medidos, validación150;
mediana60.493ms,allocated698.903MiB,reserved858MiB,libre mínimo93.839%.
**Joint no arrancó.** Al salir PID1780761, `/proc` dejó de enumerar su RSS,
pero nvidia-smi todavía lo listó; supervisor derivó owned vacío y abortó con
`RuntimeError: foreign`. Falso positivo de su propio PID reproducido en CPU.
No fue OOM ni un PID ajeno observado. No reintentar automáticamente: protocolo
aprobado exige parar tras abortar. Corregir guardia+repetir pareja acotada es
la autorización siguiente; seis fits siguen sin autorizar. Runner congelado
intacto, bug pendiente; no resetear marca started ni sobrescribir v1.

[Informe y evidencia](informe_piloto_multitarget_w90_20261019.md). Artefactos
`piloto_multitarget_w90_20261019_v1/`, incluidos result-h y summary originales:
summary tiene campos parciales nulos pese a result-h completo porque abortó
antes de incorporarlo. No hay result-joint. Scripts/datos/producto intactos
tras cierre; ningún proceso propio ni contexto CUDA pendiente. Preflight
cuantifica colas fuera de grafo DMmáx0.08738/EDMmáx0.05345eV, no error aprendido.
No comparación completa, presupuesto de seis fits, precisión/OOD resueltos ni
defaults promovidos. Secciones de preparación/implementación de abajo son historia.

## Piloto7.6 autorizado, implementación Native — 2026-10-19

Usuario«Apruebo» al plan2026-10-19-multitarget-w90-pilot. Script experimental
`pilot_multitarget_w90.py` y tests propios implementados;21tests focales pasan
(703warnings), gate previo de9nuevos+regresiones7.6=95passed/0fallos.
Ruff/diff y28hashes de producto anteriores intactos. Única revisión Native
independiente en curso ANTES de preflight real y GPU. No declarar terminada
la medición ni lanzar seis fits. Registro recuperable:
`.superpowers/sdd/2026-10-19-multitarget-w90-pilot/progress.md`.

## Diseño aprobado y plan Native redactado — 2026-10-19

Usuario«Apruebo» al diseño del piloto7.6. Receta/pesos y dos brazos seed0,
20warmup+100updates cada uno,15min totales aprobados para este piloto; no
para seis fits definitivos. [Plan Native](../docs/superpowers/plans/2026-10-19-multitarget-w90-pilot.md)
redactado y autorrevisado, pendiente de revisión/aprobación del usuario antes
de implementar según writing-plans. Cuatro tareas: preflightCPU, rutas de
entrenamiento comparables, supervisor/revisión final y ejecución/cierre.
El preflight real se hará tras congelar el script, no antes de modificarlo.
Sin runner/test nuevo, GPU, inferencia, entrenamiento o cola lanzados.
Rama g2m-v2/HEAD270d17d; conservar dirty y no reabrir Goal técnico. La lectura
GPU del18octubre que figura abajo es histórica, no estado actual.

## Diseño del piloto comparativo7.6 — 2026-10-18

Usuario respondió «Hazlo» al siguiente paso de protocolo/presupuesto del
piloto. [Diseño concreto](../docs/superpowers/specs/2026-10-18-multitarget-w90-pilot-design.md)
redactado para revisión y aprobación antes de implementar. No confundirlo
con autorización de seis fits ni reabrir el Goal técnico ya cerrado.
Propone dos brazos secuenciales, seed0, cada uno20warmup+100updates medidos,
**15min totales**, mismo backbone inicial/orden de muestras y scheduler por
update; protecciones de VRAM/temperatura/procesos ajenos sin tocar autorun.
Preflight CPU de representación y espectros con K, usando solo el dataset
ya autorizado. Splits preservados con avisos, sin nuevo recurso OOD.
Consulta nvidia-smi de solo lectura: GPU0 RTX5090,32607MiB totales,
32096MiB libres,36°C,0%utilización en ese instante; no reserva ni garantía
de disponibilidad futura. No se inició GPU, runner, cola ni entrenamiento.
Tras aprobación del diseño: plan Native, implementación/comprobaciones y
piloto; después informe de coste y nueva decisión expresa sobre seis fits.

## Etiquetas w90 para7.6 — actualización2026-10-18

Usuario «Haz esto pero en la rama g2m-v2» y «Hazlo» a la lectura sola del
recurso explícito `Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1500`
en MD_vs_AtomicDisplacement. Vuelto sin force a g2m-v2; diff rastreado intacto,
HEAD270d17d. Commits physics-prior permanecen allí, no mezclar ni reabrir fonones.

Auditoría CPU de1500muestras reales con helper7.6: las cuatro etiquetas cargan,
máscaras completas, bases/órdenes/gauge/soporte correctos;13512archivos únicos
verificados antes/después, sin cambios. SCF original300+900+300pasos convergidos.
N=8 con error≤1.8e−14; HD−SE por orbital≤9.04e−10eV;600000S(k) en malla20×20×1,
λmín0.16444,condiciónmáx15.20764, sin cortes. No beneficio ML demostrado.

Split congelado1198train/150val/150test, excluye1198/1349 por gap1frame.
**Aviso, no bloqueo automático:** md_0/md_300(train) ymd_1200(val) tienen las
cuatro matrices idénticas; no duplicado exacto con test. Train/val comparten
bloque300K yval/test450K; no independencia ni OOD general certificados.
Ningún split/muestra movido, borrado ni regenerado.

Script propio `auditorias_mejoras/audit_multitarget_w90.py`, salida JSON y log
con sufijo20261018;4tests RED→GREEN del auditor. Primera ejecución rechazó por
Z placeholderH del reader TSHS: error del auditor corregido con regresión,
no del dataset; especies se atan a FDF/ORB_INDX/XML. Ningún código productivo
modificado. Suite general+4tests terminal:2896passed/20skip/0fallos,
43146warnings,850.60s,exit0. Log
`auditorias_mejoras/audit_multitarget_w90_20261018_tests.log`; Ruff/diff correctos.
28hashes de cierre técnico siguen intactos. No procesos propios pendientes;
no repetir la suite por cambios documentales ni reabrir el Goal técnico.

[Informe y receta propuesta, no aprobada](certificacion_multitarget_w90_20261018.md):
H-only directa vs H/S/DM/EDM,3seeds cada brazo, mismo backbone y4000updates
propuestos, no4000épocas; pesos explícitos por varianza de etiquetas train,
selección por H gauge equivalente, sin cambiar monitor default. Hace falta
diseño/plan y aprobación de receta/pesos, recurso OOD independiente si se exige
esa conclusión y presupuesto de piloto GPU antes de correr. No se preparó
runner/cola/entrenamiento ni se accedió a otro dataset/código externo.
El bloqueo por falta de autorización de lectura de este recurso está resuelto;
las frases sobre ausencia de datos más abajo son históricas del cierre técnico.

## Cierre4.1 y7.6 — 2026-10-18

Usuario aprobó diseño4.1 y después «Apruebo el plan». Native en checkout
g2m-v2, HEAD270d17d sin commits/worktrees.4.1 cierra como certificación de
contrato existente, no nueva implementación productiva. Cuatro módulos de
tests y docs: espacio orbital completo/periodicidad, [1,2]/Cholesky/cS, S
predicha frente etiqueta en batches, NetCDF/TSHS/HSX y referencias sin duplicarS.
Revisión independiente única:2Important en tests corregidos con mutaciones;
0Critical/0Minor. Fuente productiva y seis hashes de cambios previos intactos.

Evidencia: `auditorias_mejoras/informe_reconstruccion_no_ortogonal_20261018.md`.
Focal final261passed/0skip; regresiones1323/1skip. SuiteCPU final2732passed,
20skipped,40974warnings,822.22s. Skips: ejemplo externo ausente, inductorGPU,
CUDA coeficientes y supervisorGPU de fusión no autorizado. Dependencias ya
instaladas reutilizadas víaPYTHONPATH; no instalaciones. Logs y ledger propios:
`.superpowers/sdd/2026-10-18-nonorthogonal-contract-4-1/`. No borrar: sin commits
son el registro. Primera suite desfasada interrumpida; solo full-suite-final.log
es gate completo, complementado por focused-final.log para último refuerzo de
metadatos. No repetir campañas previas ni promover defaults por este cierre.

**7.6 implementado y verificado CPU:** diseño y plan aprobados con «Apruebo»;
Native terminado en el checkout actual, sin commits/worktrees.
[Diseño](../docs/superpowers/specs/2026-10-18-multitarget-7-6-design.md) y
[plan](../docs/superpowers/plans/2026-10-18-multitarget-7-6.md).
Un backbone, descriptor H/S/DM/EDM, named views, pérdidas/pesos
y máscaras por componente, física4.4–4.5 referenciada, epochs/checkpoints y
reconstrucción/NetCDF explícitos con S_pred; single-target/defaults conservados.

Una revisión fresca, reanudada tras cuota con el mismo agente:0Critical,
3Important corregidos,0Minor. Soporte archivado debe ser Hermitiano incluso
con ceros explícitos; scaleshift no admite etiquetas parciales para fitting;
selector de métricas sobre legacy falla antes de truncar CSV.11regresiones
RED→GREEN,176focales y suite final **2892passed/20skip/0fallos**,43146warnings,
845.34s, exit0. Ruff/diff y28hashes de snapshot correctos; los seis cambios
previos conservan exactamente sus hashes. Solo docs cambiaron después del gate.

Registro: `.superpowers/sdd/2026-10-18-multitarget-7-6/`; `task-6-tests.log`
es el gate terminal autoritativo, `full-suite.log` anterior está desfasado.
`final-review.md`, `progress.md`, `final-source-sha256.txt` y benchmark CPU
conservados: no borrar workspace sin commits. [Informe final](informe_multitarget_7_6_cpu_20261018.md).
SK mantiene C1/C2, no packing completo4/7; estándar/low-rank CPU probados.
ScaleshiftFalse (default) sí permite pérdidas parciales; no imputar ni cambiar
buffers con NaN. Export protege preflight, no fallos arbitrarios de E/S.

Goal4.1+7.6 cerrado técnicamente; no recurso real H/S/DM/EDM compatible
certificado ni beneficio científico/VRAM demostrado. S_pred puede no ser SPD,
OOD no resuelto y ningún default promovido. Siguiente decisión del usuario:
autorizar datos coincidentes, receta/pesos/splits, piloto y presupuesto GPU
para H-only directa vs multitarget,3semillas cada uno (6fits); H+S intermedio
es otro brazo a aprobar. No preparar/encolar ni reutilizar ΔH como baseline
sin nueva autorización y comparabilidad. Las secciones siguientes son historia:
no reabrir campañas ni repetir4.1. Sin GPU/autorun/fonones/otros proyectos.

## Cierre4.7 verificado — 2026-10-18

Usuario pide auditoría/cierre y señalar siguiente implementación, no lanzar más.
Tres IDs efectivos`:bootstrap1` DONE, tres fits2000épocas/64000muestras/8000updates,
21matrices candidatas y21fixed de referencia. No procesos propios pendientes,
no repetir prepare/enqueue/fit. IDs originales fallidos/bloqueados son historia;
no reactivarlos ni editar state. Fuentes originales y relaunch congelados intactos.

Auditoría reproducible `auditorias_mejoras/audit_h2oclu_nullspace.py`; resultado
autoritativo `experimentos_nullspace_h2oclu_20261017_v1/closure_audit_20261018.json`:
982archivos hashados,42matrices/métricas reproducidas, siete referencias físicas,
checkpoint/selección/hparams/budget y prefijos de cola verificados. Cholesky
independiente full-basis reproduce niveles≤2,29e-12eV. No inferencia ni GPU nueva.
Per-structure/arm/seed, desgloses y todas las regresiones>5% en el JSON.

Conclusión OOD: gap error3,109454→1,192056eV (−61,66%,IC95razón0,1231–0,5567),
niveles928,4878→285,4976meV (−69,25%), H5,703058→6,471390meV (+13,47%);
H ID+19,93%. Gap/niveles medios mejoran tres seeds, H empeora tres. Caso0/001
con gap5,185933→0,163931eV sigue muy mal; regresiones>5% en5/9H,1/9niveles,
1/9gap OOD. Evidencia exploratoria con S referencia y test observado, no solución
general ni promoción/default nuevo. Costes no contemporáneos: no aceleración/RAM
causal; allocated+≈1%,reserved+2MiB,3,70191h pared total de fits supervisados.

33focales pasan39,03s con preload, Ruff/diff limpios. Suite general recuperada
del relanzamiento:2653pasan/34omitidos/4HistorySOAPs históricos por dscribe,
942,67s. No repetir toda la suite por cambios de documentación/auditoría.
El foco directo probe_site sin bootstrap dio1fallo/32pasan por el fallo original;
se conserva log y no se oculta: usar wrapper certificado. No instalar cython.
Informe/roadmap/plan actualizados; Task3 cierra la campaña, no roadmap global.
Sin Goal nuevo, commits/worktrees ni fuentes productivas modificadas.

Siguiente según protocolo: delimitar/cerrar4.1 (H+S ya leído/escrito, no rehacer
esa funcionalidad); siguiente ampliación nueva inequívoca7.6 (targets DM/EDM
conjuntos), con diseño/datos/aceptación antes de implementar o entrenar.
1.15 es decisión sobre política default,3.8 residual ruta no usada; no reabrir
esos puntos, splits pausados, fonones o a-SiO2 por inferencia. El usuario solo
pidió informar del siguiente paso, no ejecutarlo aún. Historia debajo.

## Recuperación4.7 autorizada y relanzada — 2026-10-17,21:55CEST

Usuario:«Corrige y relanza», después del diagnóstico del fallo de evaluación.
null_s0 terminó2000épocas/64000muestras/8000updates en67,22min; sus pesos,
fit.json y925hashes originales están intactos. Falló solo el subprocessCPU:
importó callback/graph2mat antes de cargar extensiones instaladas, y probe_site
hizo caer en _sparse.py fuente sin cython. Reproducción exacta exit1; preload
existente recovery_api antes del import da exit0. No instalar cython.

Nuevo `relaunch_h2oclu_nullspace.py`, sin editar fuentes congeladas ni manifest:
preload nativo, reutilizar fit terminado, evaluaciónCPU en el mismo proceso;
guarded_fit original solo para fits ausentes. No-clobber/selección/full-basis
originales preservados.33tests nuevos+campaña pasan42,19s, Ruff/diff limpios.
Suite general fresca en curso (`relaunch-full-suite.log`); no afirmar finalizada.
La suite previa sigue2653/34skip/4HistorySOAPs históricos, sin cambios de src.

Tres appends autorizados,1412→1415entradas/prefijo intacto; nuevos IDs efectivos
`experimentos_nullspace_h2oclu_20261017_v1:{null_s0,null_s1,null_s2}:bootstrap1`,
secuenciales. `relaunch_receipt.json` completo fija launcher+manifest+recibo
original+fit0+best0 por hash. State/entradas antiguas no reescritos: failed0 y
sus dos dependientes viejos son historial, NO los IDs efectivos que consultar.
No repetir enqueue ni editar el launcher nuevo, también congelado por el recibo.

A21:55:11 bootstrap0 DONE, siete matrices evaluadas y validadas, sin reentrenar0;
bootstrap1 supervisor1053391 arrancado, bootstrap2 espera. Daemon1756952 intacto.
Auditoría21:56:40:fitGPU1053418 vivo,31épocas/992muestras confirmadas,
125updates/1000observadas; GPU29158/32607MiB libre40°C.925+5hashes pasan,
`relaunch_audit.json` registra estado y pruebas. Al continuar, recoger también
resultado de la suite general nueva; no repetir esa suite por falta de memoria.
Resultados parciales seed0: H OOD4,8426→5,2682meV (+8,8%), niveles1760,91→535,89meV,
gap error4840,24→2898,16meV. Sigue grande; H/validación empeoran, una sola semilla:
no declarar mejora robusta ni OOD resuelto, no variar peso ni receta.
Quedan dos fits/evaluaciones, auditoría21matrices y cierre Task3; sin defaults.
Las notas de arranque/fallo inferiores son historia y conservan trazabilidad.

## Validación4.7 aprobada; tres candidatos en marcha — 2026-10-17

Usuario:«Lo autorizo. Hazlo», referido a validar4.7 con receta actual y baselines
comparables. Auditoría nueva de627hashes/versions y fixed_s0/1/2 pasa; todos
2000épocas/64000muestras/siete evaluaciones, inicializaciones preservadas.
No hacen falta referencias nuevas por integridad/procedencia.

Hallazgo de selección: loss_fn nativo comparte train/val; activar1e-4 sin
adaptación cambiaría el criterio efectivo de best-val_loss. Plan propone
callback experimental derivado del único ModelCheckpoint para mantener
validación de gauge básica y penalizar solo train; ni modelo ni runner
congelado deben editarse. Proyectores C y overlay solo en graph2mat.
Plan Native: `docs/superpowers/plans/2026-10-17-h2oclu-nullspace-validation.md`,
aprobado después mediante«Apruebo». Tasks1–2 implementadas,26tests pasan.
Smoke CPU real preflight_v4 pasa interrupción tras época1, resume hasta2/4muestras,
cuatro workers y carga por la clase base; val_loss0,0478701778 coincide con cálculo
básico independiente0,0478701741. Overlay36/36 exacto salvo C, rangos4–32.
Metadata redundante basis_table=None normalizada solo al guardar checkpoints
nuevos: el CLI tras resume generaba texto no recargable. Tabla real intacta.
Historial de errores/soluciones e incidente de extensión restaurada por hash
en `.superpowers/sdd/2026-10-17-h2oclu-nullspace-validation/progress.md`.
Revisión independiente única terminada,1Important y1Minor reclasificado Important
corregidos RED→GREEN en una sola pasada: no-clobber/reutilizar resultados completos
y exclusiones espectrales con seed/brazo/estructura/motivo. Sin menores diferidos.
Focal definitivo150pasan/5CUDAguarded omitidos297,75s; general2653pasan/
34omitidos/solo4HistorySOAPs por dscribe910,30s. Ruff src/tests+nuevos y diff limpios.
**Tres encolados18:42CEST**, cola1409→1412, prefijo anterior intacto, recibo completo:
`experimentos_nullspace_h2oclu_20261017_v1/queue_receipt.json`.
IDs `experimentos_nullspace_h2oclu_20261017_v1:{null_s0,null_s1,null_s2}`,
secuenciales, primero sin dependencia viva. Daemon1756952 no reiniciado.
ManifestSHA de03dcd283564d1504905dca1a189af1fac8829c0281a5bc83dbe07d2aa8ece3,
925hashes verificados después de arrancar; no editar fuentes/configs/C congelados.
A18:43:19 supervisor912675/hijo fitGPU912694 vivos,24épocas/768muestras
confirmadas en recuperación,96updates observadas y best protegido presente.
GPU29472/32607MiB libres,38°C. Otros dos esperan, ninguno DONE/FAILED aún.
Evidencia `submission_audit.json`; no confundir primer progreso con mejora física.
No repetir prepare/enqueue ni crear/reabrir Goal. Cada entrada hace fit recuperable→
evaluaciónCPU→summary, luego libera siguiente. Si evaluación queda parcial, nuevo
wrapper la preserva y detiene; requiere decisión para intento nuevo, no sobrescribir.
Siguiente paso: consultar solo estos3IDs y procesos reales; al terminar, auditar
21matrices/checkpoints, resultados/ratios/IC y costes, actualizar informe/roadmap.
Task3 queda pendiente de esa auditoría, no declarar cerrado al encolar.
Preflights_v1–v4 son solo ensayos CPU, no encolarlos. No mezclar sus snapshots.
Informe previo a resultados: `campana_nullspace_h2oclu_20261017.md`.
Tres candidatosseeds0/1/2,peso fijo1e-4,2000épocas; sin barrido, defaults,
fonones/autorun/dependencias/commits/worktrees. Estimación4–6h exclusivas
desde arranque, no medida definitiva del overhead ni compromiso de hora de fin.
Las restricciones anteriores a esta autorización se conservan como historial.

## Diagnóstico OOD h2oclu autorizado — 2026-10-17

Usuario autorizó diagnóstico CPU con las matrices guardadas y documentación,
no corregir modelos ni entrenar/encolar. Trabajo analítico terminado:
`diagnose_h2oclu_ood.py`,11controles focales, informe
`informe_diagnostico_h2oclu_ood_20261017.md`. Resultado autoritativo
`diagnostico_h2oclu_ood_20261017/results_v3.json` (v1/v2 son iteraciones previas).
42registros reproducen métricas originales;627hashes/versiones y resultados/
checkpoints/NPZ verifican integridad antes/después, sin inferencia nueva.

Mecanismo: error amplificado en subespacioλ(S)<0,01 y reordenación ocupada;
86,57%medio de energía cuadrática transformadaOOD está en LL. Oráculo LL
reduce errorgap medio3,044974→0,270305eV sin eliminar ninguna de736direcciones
ni cambiar S/referencia. No es una corrección disponible: usa etiquetas.
H0 cancela hasta4,44e-16eV. Onsite/offsite/especies/familias tienen cancelaciones;
ni los tres bloques atómicos dominantes son una solución universal.
HOMO/LUMO por sí solos pueden ocultar intrusos que alteran el llenado.
No causa única del entrenamiento aislada; no declarar generalización resuelta.

Siguiente propuesta mínima: auditar comparabilidad para validar **4.7 ya
existente** (spectral_null/spectral_weight), no escribir otra pérdida ni
repetir investigación. Sus cifras históricas no validan esta receta cosine.
Necesita nueva autorización de protocolo/trabajos; no hay nada preparado o
encolado. Defaults, fuentes congeladas, datos y resultados originales intactos;
no fonones, commits, dependencias o worktrees. No reabrir el Goal anterior.
Gate definitivo nuevo:32focales pasan17,71s; `pytest src tests`2653pasan/
34omitidos/4HistorySOAPs por dscribe,878,07s. El comando general no incluye
auditorias: no confundir su recuento con2789histórico. Ruff src/tests+nuevos
limpio; diff--check limpio. Log general persistido junto al JSON del diagnóstico.
El análisis CPU final tarda26,96s; contraste independiente de42oráculos LL
reconstruidos en base original/Cholesky coincide≤2,84e-13eV. No repetir análisis
ni reabrir este alcance; esperar decisión sobre protocolo/validación4.7.

## Cierre verificado recuperación+6.4 — 2026-10-17

Goal reanudado por el usuario; sus dos objetivos ya tienen evidencia de cierre.
Seis IDs `experimentos_batching_h2oclu_20261016_science_v1:{fixed_s0,cost_s0,cost_s1,fixed_s1,fixed_s2,cost_s2}`
DONE/no FAILED; seis fits2000épocas/64000muestras,42matrices,627hashes y versiones
verificados. No PID de fit pendiente ni necesidad de relanzar/enqueue.
Cost326283 vs fixed: allocated−15,66%,reserved−25,55%,hostPSS−10,24%,pared−3,77%
de media. Precisión OOD mixta: H−1,14%,niveles−30,81% medio dominado porseed0,
niveles empeoran2/3seeds y todos los IC OOD incluyen1; gap error≈2,98eV. No promoción.
Contraste Cholesky independiente reproduce espectros/gaps; diagnóstico de
pequeñoS no autoriza cutoff1e-2 (elimina136–141direcciones y cambia referencia).
Último fit acabó07:25:04CEST,7,72h supervisadas; sin cesiones/reintentos reales.
Focal final136pasan/5CUDAomitidos248,95s; general vigente2789pasan/39omitidos/4
dscribe históricos,Ruff/mypy23/diff limpios. No instalación/commits/worktrees,
defaults/fonones/autorun/datasets ni fuentes de campañas cerradas modificados.
[Auditoría de requisitos/limitaciones](cierre_goal_recuperacion_batching_20261017.md)
y [resultados finales](campana_batching_h2oclu_20261016.md). JSON de evidencia
`experimentos_batching_h2oclu_20261016_science_v1/closure_audit_20261017.json`.
No crear Goal duplicado, no reabrir Tasks1–6 ni encolar más. No declarar cerrado
el roadmap global/vista igualtiempo por este cierre acotado. Siguiente ampliación
requiere decisión del usuario; diagnóstico del error OOD puede usar matrices
existentes sin entrenar. Las notas de pausa/entrenamiento inferiores son historia.

## Pausa del agente solicitada — 2026-10-17

Usuario: «Mejor te dejo en pausa hasta mañana. Revisaré los trains al despertarme.
Tú termina lo que estás haciendo justo ahora y detente. Mañana seguimos».
Pausar el Goal/agente, **no los seis entrenamientos autorizados**. No encolar
nada más ni cambiar fuentes congeladas. Daemon y evaluaciónCPU siguen solos.
Última comprobación00:03:50CEST: daemon1756952, supervisor56088 y fitGPU56113
vivos con comandos correctos. A00:02:16 fixed_s0 tenía657épocas en heartbeat,
2632updates/21056muestras observadas; ningún ID científico DONE/FAILED aún.
627hashes y versiones verificados; ninguna evaluación final todavía.
La reanudación previa activó el servicioGoal; las menciones históricas a blocked
no describen esa reanudación. Ahora se solicita estado paused explícitamente.
Mañana consultar primero recibo/IDs, procesos reales, fit/result/summary y logs;
no repetir enqueue ni asumir terminación por tiempo transcurrido.

## Autorización científica recibida — 2026-10-16

Usuario: «Autorizo esos 6 trabajos», después «Continua». Autoriza fixed/cost326283
×seeds0/1/2,2000épocas/64000muestras por fit, cosine2000 explícito/eta_min1e-5,
Adam nativo, best-val_loss cada5, evaluación H0+Δ/niveles/gap con S referencia,
splits32/4/3 intactos. No repetir puerta de aprobación del mismo alcance.
Seis encolados2026-10-16 23:40CEST: cola1403→1409, prefijo previo intacto.
Manifest/recibo en `experimentos_batching_h2oclu_20261016_science_v1/`.
No ejecutar prepare de nuevo ni repetir fits; consultar IDs del recibo y state.json.
Primero fixed_s0 sin dependencias vivas previas; los otros cinco encadenados.
Daemon1756952 permanece vivo, sin cambios/reinicio.
Arranque verificado23:41:58CEST: fixed_s0supervisorPID56088, hijoGPU56113,
28épocas/112updates/896muestras confirmadas y best protegido.627hashes intactos,
GPU29308/32607MiB libres,40°C. Los otros cinco dependen secuencialmente.
Suite final2789pasan/39omitidos/4HistorySOAPs por dscribe; ningún fallo adicional.
Lanzador/evaluador21tests pasan; tres Important de revisión única corregidos
RED→GREEN. Smoke CPU con pilotoID+OOD96átomos pasa3,235s, sin nuevos fitsGPU.
Informe: `campana_batching_h2oclu_20261016.md`; summary/result propios tras fits.
Lanzador/evaluador nuevos separados de fuentes técnicas v4 congeladas.
Plan operativo: `docs/superpowers/plans/2026-10-16-h2oclu-six-approved.md`.
Orden autorizado concretado fixed0,cost0,cost1,fixed1,fixed2,cost2, secuencial.
Sin ampliaciones, defaults, autorun, fonones, commits ni worktrees.
Las menciones inferiores de autorización pendiente son históricas y quedan
sustituidas por esta autorización expresa. Goal científico no cerrado al encolar.
Siguiente paso: supervisar estado/errores de los seis, no añadir trabajos. Cada
entrada hace fit recuperable y evaluaciónCPU; al terminar todos, auditar resumen,
tiempo/VRAM/RAM, ratios/IC y registrar conclusión en roadmap/informe/traspaso.
No cambiar fuentes congeladas mientras corren, pues abortaría hashes de campaña.

## Recuperación técnica cerrada; estado previo a autorización — 2026-10-16

Usuario: «Los apruebo. Inclúyelos en el goal y lanzalo». Goal creado en la
herramienta; no tiene presupuesto de tokens explícito. Dos objetivos: recuperación
segura del entrenamiento al ceder la GPU y comparación científica fixed/cost.

El bloqueo anterior de aprobación del enfoque quedó resuelto por «Apruebo» del
usuario el 2026-10-16. Diseño escrito y aprobado después por «Listo, apruebo»:
`docs/superpowers/specs/2026-10-16-training-recovery-design.md`.
Recuperación desde época completa, generaciones atómicas con estado/RNG y copia
protegida del best, orden por seed/época y cierre propio acotado. Runner7.2 y
defaults intactos. Sin commits ni entrenamientos científicos nuevos.
Plan escrito y auto-revisado: `docs/superpowers/plans/2026-10-16-training-recovery.md`.
Seis tareas: store atómico, loaders pareados, callback/estado/best, supervisor,
certificación entre procesos y preparación de propuesta h2oclu. Método Native
preservado, una revisión final independiente conforme a la skill de ejecución.
Usuario aprobó también el plan mediante «Apruebo». Implementación Native terminada,
sin pedir permisos entre tareas; no interpretar aprobación técnica como permiso
de campaña científica. Estado/evidencia reanudable:
`.superpowers/sdd/2026-10-16-training-recovery/progress.md`.
Tasks1–4 implementadas: store, loaders, callback/RNG/best y supervisor/CLI real.
Task5: revisión independiente única0Critical/3Important/0Minor, todos los Important
corregidos RED→GREEN: accountingcancel/corrupción, validaciónNaN/Inf y certificación
compilada. Guard propio adicional de resolución/orden detecta retarget de links.
Focaldefinitivo115pasan/5omitidos,254,39s. MACEGPUFP32compilado,cost,
4workers pasa204,99s, release1,621s,48admitidas/5repetidas. AMP/eager3pasan/
157,02s; general2744pasan/39omitidos/4HistorySOAPs históricos por dscribe,
1228,04s. Ruff/mypy23/diff--check limpios. Tasks1–6complete en ledger.
Task6: v4final repite6pilotos de5épocas,341,32s: candidato326283, allocated
−10,85%ambasrondas, reserved+2MiB (noahorro). v1diagnóstico interrumpido por
picoNone pos-teardown; v2/v3snapshots anteriores, conservados. No usar sus
hashes como snapshotfinal. Preflight39casos pasa;6generaciones/bests/cobertura
v4 y hashes/orden de inputs verificados después de finalizar.
Fuentes nuevas `training_recovery.py`, `run_cost_batching_comparison.py`,
`recovery_certification.py` y sus tests, solo auditorias.
Sin entrenamientos científicos ni cola nuevos; no volver a pedir aprobar el plan.

Dataset y observables aprobados: `h2oclu`, manifest existente
`MD_vs_AtomicDisplacement/Comparison/results/dataset_design_curves_v1/g2m_improvements/bench/h2oclu/frozen_split_manifest.json`.
Splits sin modificar: 32 train (8×6, 12×12, 12×24 átomos), 4 validation
(2×12, 2×24) y 3 test OOD (96 átomos). Sin añadir monómeros ni crear resplits.
Primarias: H alineado al gauge y niveles electrónicos/gap moleculares; no bandas
dispersivas cristalinas. Test OOD ya observado: comparación exploratoria, no
validación final independiente. Inventario de solo lectura: las 39 referencias
TSHS, marcadores `0_NORMAL_EXIT` y mensajes `SCF Convergence` existen. Esto no
sustituye la auditoría física completa ni el congelado de hashes de la campaña.

Preflight ampliado de solo lectura: 39 TSHS/H0 compatibles, espín no polarizado,
bases/pseudos idénticos; 36 etiquetas Δ verificadas contra H−H0 (error máximo
8,9e-16 eV) y S idéntico. Riesgos de recuperación identificados: directorio
variable de ModelCheckpoint pierde historial de best, RNG de iteradores/workers
persistentes y orden de hooks/scheduler. Evidencia y límites en
[preflight actual](preflight_h2oclu_recuperacion_20261016.md). No prueba continuidad
ni mejora de precisión. Enfoque, diseño y plan aprobados después de ese preflight;
implementación técnica ya cerrada. La continuación automática no autoriza campaña científica.

Estado actual: punto1 técnico cerrado, sin trabajos científicos lanzados
o encolados. Runner7.2 cerrado intacto. Siguiente paso: pedir autorización concreta
de la campaña, no reimplementar ni reabrir Tasks1–6. Propuesta escrita en
`propuesta_batching_h2oclu_20261016.md`:6fits fixed/cost0/1/2,2000épocas,
64000muestras/fit,8000/8139updates,4–7h exclusivas estimadas y cosine explícito.
Pedir autorización antes del entrenamiento científico
o de escribir en la cola. La aprobación de dataset/observables no autoriza una
lista de trabajos ni presupuesto. Ambos ya están presentados en la propuesta,
pero **aún no autorizados**. No lanzar/encolar por inferencia.

Artefactos actuales: `experimentos_batching_h2oclu_20261016_v4/manifest.json`
y `pilot.json`; informe `informe_recuperacion_entrenamiento_20261016.md`.
Vista igualtiempo/compute diferida, cero jobs adicionales propuestos.6.4
científico y Goal permanecen incompletos. HerramientaGoal conserva statusblocked;
no existe APIresume disponible, trabajo técnico se ejecutó con aprobación del
usuario sin crear objetivo duplicado ni marcar completo el objetivo científico.

Hallazgo de receta: histórico h2oclu tenía lr_schedulers=[] y AdamLR0,0018
constante, aunque YAML declaraba cosine. Runner técnico aprobado soporta
cosine explícito sobre cinco grupos Adam nativos. Propuesta científica debe
pedir aprobación de esta diferencia efectiva; nunca reutilizar histórico para
comparación pareada ni afirmar que se ha conservado su scheduler efectivo.

Restricciones vigentes: checkout actual `graph2mat/g2m-v2`, preservar cambios
previos; sin defaults nuevos, commits/worktrees, fonones, a-SiO2, modificaciones
de autorun, dependencias compartidas o datos/código externos. No reabrir 7.2,
ya cerrado. Las instrucciones inferiores de esperar una decisión sobre otro
punto quedan sustituidas por este Goal; el historial de 7.2 se conserva.

**Cierre actualizado2026-10-16:** los seis trabajos7.2 están DONE y evaluados;
usar `:retry1` como IDefectivo de tbg_s0.8.306hashes y seis checkpoints completos
verificados. Resumen final de tres semillas: tbg OODH+16,8%/bandas−7,5%;
grhbn_bl OODH−5,3%/bandas−9,5%, pero bandasID+35,4% en las tres semillas.
BandasOOD mejoran en una de tres semillas por material: no promoción general,
defaults intactos. Informe de campaña/roadmap actualizados. Las menciones
inferiores de entrenamiento en curso y «siguiente acción» son históricas.
No reencolar ni reanudar estos trabajos, ya completos. Esperar decisión del
usuario sobre otro punto del roadmap; no hay nueva campaña autorizada.

**Actualización de recuperación:** intento originaltbg_s0falló02:18a los1290pasos.
Causa reproducida: SIGSEGV del hilo `faulthandler.dump_traceback_later`, no VRAM.
Timer retirado del runner, control2500pasos/340,38s pasa. Usuario autorizó
«Investiga que es y relanzalo»: retryúnico `...v6:tbg_s0:retry1` encolado;
`grhbn_bl_s0` espera ese retry, demás dependencias iguales. Manifest original
y runner anterior archivados; solo hash del runner actualizado,8.305otrosintactos.
No repetir enqueue ni tocar estado/código de autorun. Ver
[diagnóstico actual](diagnostico_relanzamiento_72_20261015.md) y
`experimentos_edge_updates_bilayer_20261015_v6/relaunch_receipt.json`.
Las menciones inferiores al primer PID/enqueue son históricas.

RetryPID2974768, PPID1756952,4750pasos observados después del fix. Los otros
cinco siguen encolados. GateCPU del fix:2662pasan/34omitidos/4fallos históricos
`HistorySOAPs` por `dscribe`,935,55s;39tests afectados pasan. Al consultar estado,
usar `relaunch_receipt.json.retry_job.id` para tbg_s0, no el IDoriginalfallido.
No confundir el control diagnóstico2500pasos con un entrenamiento de campaña.

Repositorio exclusivo: `<graph2mat>`, rama `g2m-v2`.
HEAD `270d17d1f204293a059baa09e8f1b3756981e2bf`, cambios previos sin commit
preservados. No commits/worktrees nuevos, cambios de defaults, fonones,
reanudación a-SiO2 ni cambios de autorun autorizados.

## 6.7 cerrado

Implementación opt-in únicamente en `edge_operations.py` y `matrixblock.py`.
Una revisión independiente, Important de broadcasting corregido con seis tests
RED→GREEN. Task-done1–5 registrados en
`.superpowers/sdd/2026-10-14-homonuclear-weight-fusion-6-7/progress.md`.

Evidencia final: general2653pasan/34omitidos/4fallos históricos `dscribe`,
source39 204pasan/9omitidos, wheels3.9/3.12 cada una265pasan/9omitidos,
CUDA/AMP/compile9pasan; Ruff/mypy23files/diff--check limpios. Logs completos
y ocho rulings en workspace. No hallazgos pendientes/deferred minors.

Medidas32hijos previas al guard final de shapes Python: d/f allocated−15,18%
y−16,29%; df72pared×1,004853, mixto/inconcluso; s/p VRAM≈igual. No aceleración
general ni precisión demostradas. Snapshot de tiempos distinto del árbol final,
explícitamente documentado; sin cambios de kernels en inputs válidos.
Ver `informe_fusion_homonuclear_20261014.md` y roadmap. No repetir benchmark
ni nuevos entrenamientos por iniciativa propia.

## 7.2 nocturno autorizado y encolado

Usuario: «Apruebo7.2 en ambas bicapas,seis trabajos(recomendado)».
Seis trabajos, `edge_update_layers=1`, tbg/grhbn_bl, semillas0/1/2, recetas,
datos/splits congelados. Referencias históricas verificadas, reevalúadas en
CPU con el mismo código actual. No entrenamientos de referencia nuevos.

Runner/tests nuevos, solo Graph2Mat:
`auditorias_mejoras/run_bilayer_edge_updates.py`, `test_bilayer_edge_updates.py`.
Contrato: `campana_edge_updates_bicapas_20261015.md`.
Campaña definitiva, ignorada por git:
`auditorias_mejoras/experimentos_edge_updates_bilayer_20261015_v6/`.

Manifest con hashes de fuentes, extensiones, datos, Δ/H0, bases y6checkpoints;
config científica expandida desde hiperparámetros guardados. **No editar archivos
congelados mientras corren los trabajos**: fallarán de forma cerrada si cambian.
Docs/roadmap no forman parte de esos hashes y pueden actualizarse.

Preflight:40pasos por material, gradientes activos, pérdida finita; lectura y
predicción H/bandas ID/OOD de candidato/referencia pasan. Pilotosreserved
1886/3518MiB. Ocho tests pasan. Carpetasv1–v5 conservan intentos de preparación
y diagnóstico; no mezclarlas con resultados de campaña ni tratarlas como semillas.
v3 tuvo bloqueo sin causa determinada, se mató únicamente su grupo propio;
v4/v6 no lo reprodujeron, watchdog añadido (no decir que la causa está arreglada).

Cola externa autorizada solo para estos seis appends:
`$MDVS_ROOT/Comparison/results/dataset_design_curves_v1/g2m_improvements/autorun/queue.jsonl`.
1396→1402entradas, prefix SHA256 intacto; `queue_receipt.json` guarda prueba eIDs.
Daemon existentePID1756952 vivo, no reiniciado. IDs:
`experimentos_edge_updates_bilayer_20261015_v6:{tbg_s0,grhbn_bl_s0,tbg_s1,grhbn_bl_s1,tbg_s2,grhbn_bl_s2}`,
en ese orden, encadenados. No repetir `enqueue` si ya existe el recibo: comprobar
los IDs; el helper detecta colisiones, no añadir duplicados/referencias.
68pendientes antiguos/82fallidos intactos; no reactivarlos.

Primer lanzamiento confirmado: `tbg_s0` PID2858309, PPID1756952, heartbeat780pasos;
otros5encolados.8.306hashes congelados y bytes anteriores de cola verificados
después del arranque. No son todavía resultados científicos finales.

Supervisor: allocator≤55%, reserva inicial30%, aborta OWNgroup si libre<25%,
GPU≥80°C o sin progreso900s; piloto300s. Cede ante cómputo ajeno y reintenta
en directorio propio nuevo, sin señales ajenas. Daemon conserva controlCPU/térmico.
No es garantía dura de20% frente a asignaciones ajenas entre muestras.

Cada trabajo fit→best-val_loss→evaluación CPU de candidato/referencia→result.json
→summary.json, antes de liberar la dependencia del siguiente. Fallos bloquean
los siguientes; no lanzar otro tipo de campaña automáticamente. Necesita equipo
encendido; los procesos persistentes no dependen del chat. Estimación9–12h inicial,
revisar con entrenamiento completo, no tiempos de pilotos ni wall histórico como
prueba de aceleración.

## Siguiente acción

Consultar estado/logs SOLO de esos6IDs; no imprimir el mapa `state.pids` completo
(contiene muchos PIDs viejos). Comprobar `*_s*/train_done.json`, `result.json`,
`summary.json` y la cola. Al terminar los6, registrar H/bandas ID/OOD por3semillas,
medias/razones y regresiones en informe/roadmap. Tres semillas y OOD ya observado:
exploratorio, no test independiente ni autorización para promoción automática.

Si falla compatibilidad/preflight o falta un dato, detener ese lanzamiento y
registrarlo: usuario no autorizó otras recetas/materiales ni referencias nuevas.
No hay goal activo en la herramienta al cerrar6.7; no crear uno por inferencia.
