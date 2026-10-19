# Campaña 6.4 — batching por coste en h2oclu — 2026-10-16

**Cierre científico2026-10-17:** seis trabajos DONE,2000épocas y64000muestras
por trabajo, sin fallos ni cesiones/reintentos. Coste reduce los picos medios
allocated15,66%,reserved25,55%,PSS10,24% y pared3,77%. Precisión OOD mixta,
sin mejora robusta ni promoción de defaults. Resultados finales y auditoría abajo;
los apartados de preparación y pausa conservan el historial, no el estado actual.

## Estado

**Actualización2026-10-17:** usuario pausa agente/Goal hasta mañana, no la cola.
Comprobación00:03:50CEST: daemon1756952, supervisor56088 y fitGPU56113 vivos.
A00:02:16 heartbeat fixed_s0:657épocas,2632updates/21056muestras observadas;
cinco trabajos siguientes esperan en cadena, ninguno terminado/fallido.
627hashes y versiones actuales verificados. Sin resultados científicos finales.
No se reinició ni se señalizó ningún entrenamiento. No se encolaron extras.

Seis fits autorizados mediante «Autorizo esos 6 trabajos», seguido de «Continua».
**Seis encolados** el2026-10-16 a23:40CEST, después del gate CPU final.
Manifest: `experimentos_batching_h2oclu_20261016_science_v1/manifest.json`.
627 hashes de inputs/fuentes/configs/job.json/inicializaciones; dataset externo
solo lectura y fuentes de recuperación v4 intactas. Recibo real:
`experimentos_batching_h2oclu_20261016_science_v1/queue_receipt.json`, complete=True,
seis IDs únicos. Cola1403→1409entradas, SHA256 del prefijo previo intacto.
Cada ID empieza por `experimentos_batching_h2oclu_20261016_science_v1:`;
el sufijo es el trabajo de la tabla. Daemon1756952 vivo, sin reiniciarlo.
Arranque verificado23:41:58CEST: supervisor fixed_s0PID56088/PPID1756952,
hijoGPU56113. Recuperación confirma28épocas/112updates/896muestras y best
protegido.627hashes intactos tras arrancar. GPU29308/32607MiB libres,40°C
en esa lectura; no es garantía sobre futuras asignaciones ajenas.

## Comparación predeclarada

Hipótesis: max_batch_cost puede reducir allocated con trade-offs de updates,
tiempo y precisión. Referencias fixed **nuevas**, no checkpoints históricos.
Datos existentes32train/4validation/3OOD, sin resplits. MACE histórico, FP32,
compile=True,4workers,cache=True; recuperación opt-in igual en ambos brazos.
Adam nativo LR0,0018 y cosine2000/eta_min1e-5 explícito aprobado. El histórico
tenía LR constante pese al YAML; no afirmar que se conserva ese scheduler efectivo.
2000épocas,64000muestras/fit; fixed8000updates, cost8139updates previstos.
Menor val_loss cada5épocas, sin early stopping.

| Orden | Trabajo | Budget | Seed | Estado final |
|---|---|---:|---:|---|
|1|fixed_s0|batch_size8|0|DONE|
|2|cost_s0|326283|0|DONE|
|3|cost_s1|326283|1|DONE|
|4|fixed_s1|batch_size8|1|DONE|
|5|fixed_s2|batch_size8|2|DONE|
|6|cost_s2|326283|2|DONE|

Orden alterna brazos por seed para limitar deriva temporal; no aísla hardware/
CPU compartidos. Inicializaciones completas y orden concatenado por época pareados.
Solo seis fits, cero variantes de igual tiempo/compute y cero seeds adicionales.

## Evaluación

Predicción CPU eager con checkpoint best verificado por RecoveryStore y processor
nativo. Geometría/especies FDF, no números atómicos placeholder de TSHS.
H físico enfloat64=H0existente+ΔHpred; guardar CSR propio y SHA256 por estructura.
Soporte numérico exacto no nulo unión Href/Hpred/Sref, umbral0; gauge c por mínimos
cuadrados, mismo c para matriz y niveles. Publicar MAE/RMSE/Frobenius/p95/p99,
on/offsite/paresH-O, soporteF1/falsosceros y defectos hermíticos crudos.
Γ, eighfloat64 generalizado con S referencia; proyección simétrica solo después
de defectos≤1e-8 y λminS>1e-10. Sin cutoffcanónico1e-2; número de direcciones
debajo de ese valor se registra como diagnóstico, no se descartan.
Ocupación neutral no polarizada (6NO+NH)/2; ventana HOMO−2eV/LUMO+2eV.
Medias por estructura/seed, razones pareadas y bootstrap exacto27resamples
ordenados de tres seeds. Tres seeds tienen poca potencia. Validation selecciona
checkpoint, no testID independiente; OOD ya observado, evidencia exploratoria.
Si falla el gate espectral, se conserva diagnóstico matricial, sin razón/IC
espectral válido. Resumen verifica cobertura7casos y procedencia/hash del resultado.

## Guardas y recuperación

Daemon existente, cadena secuencial; no editar/reiniciar autorun. Reserva inicial
30%, límite allocator55%, abort con free<25% o temperatura≥80°C. Cesión a todo
cómputoGPUajeno; cierre solo del grupo propio y recuperación de época completa.
Checkpoint científico separado del estado de recuperación. No reiniciar de cero
ante corrupción de todas las generaciones ni ampliar campaña tras fallos.
Registrar intentos, muestras confirmadas/repetidas, espera, allocated/reserved,
RSS/PSS, guardado de recuperación y tiempo total. Wall supervisado no es GPU-hours
de utilización. Caption timing_scope del runner v4 es impreciso: wall de época
puede incluir guardado de best; checkpoint_seconds mide publicación de recuperación.

## Evidencia previa, no resultado científico

- Pilotos v4: candidato326283 allocated−10,85%; reserved+2MiB, sin ahorro de
  reserved/free demostrado. No resultado final de precisión ni velocidad.
- Smoke CPU con best de piloto5épocas: un casoID y unoOOD96átomos,3,12s,
  ambos gates espectrales válidos. No entrenamiento científico ni elección por OOD.
- Lanzador/evaluador:21tests pasan17,84s; revisión única0Critical/3Important/0Minor,
  los tres corregidos con11regresiones RED→GREEN. Ruff/diff limpios.
- Suite focal anterior:124pasan/5omitidos247,50s. Suite general definitiva:
  **2789pasan/39omitidos/4fallos históricos**,1123,58s; ningún fallo adicional.
  Los cuatro son `src/graph2mat/tools/server/tests/test_extrapolation.py::test_real_extrapolation_with_missing_onsite_and_edge_blocks`
  `[False-1-HistorySOAPs]`, `[False-2-HistorySOAPs]`, `[True-1-HistorySOAPs]`,
  `[True-2-HistorySOAPs]`: ModuleNotFoundError de dscribe, no instalado.
  Log completo `.superpowers/sdd/2026-10-16-h2oclu-six-approved/full-suite-final.log`.
- Estimación6fits:4–7h exclusivas, más evaluaciónCPU y esperasGPUajenas sin cota.
  Gate general previo históricamente≈20min; no mezclar esa cargaCPU con medición.

## Resultados científicos

### Evidencia y comparabilidad finales

Verificados627hashes actuales, versiones, tres inicializaciones por hash de
tensores y ambos brazos por seed. Configs idénticos tras retirar únicamente
max_batch_cost/default_root_dir; mismos sample_ids y resolución/orden de inputs.
Seis generaciones nativas completas, best protegido y score/recibos/hash de
resultados; los cinco grupos Adam y cosine efectivamente terminaron en época2000,
LR1e-5. Cada época consumió32estructuras:8000updates fixed,8139cost (+1,7375%),
64000exposiciones por fit y384000 en la campaña. No se igualó tiempo/compute.
Seis entradas de cola únicas, todas DONE/no FAILED, prefijo anterior intacto.
No hubo muestras repetidas registradas ni necesidad real de recuperación nocturna;
la continuidad ante cesión se demuestra con los certificados previos, no con esta
campaña sin interrupciones.

Los42artefactos H físicos y seis resultados pasan hashes/cobertura7estructuras
(4IDselección+3OOD) y gates espectrales. Auditoría CPU independiente de las
matrices guardadas reproduce H/gauge y todos los niveles mediante Cholesky
`L^-1 H L^-T`+`numpy.linalg.eigvalsh`, no el helper de evaluación original.
Tolerancias de comprobación1e-8eV en niveles y1e-5meV en métricas; identidad
analítica de tres niveles y gauge comprobada antes. Estadísticas y bootstrap
exacto27resamples pareados recalculados, coinciden con summary.json.
Evidencia generada: `experimentos_batching_h2oclu_20261016_science_v1/closure_audit_20261017.json`.
Comando reproducible: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:auditorias_mejoras OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python .superpowers/sdd/2026-10-16-h2oclu-six-approved/closure-audit.py`.
El script solo lee campaña/datos; no entrena, encola ni sustituye resultados.
Revalidación enfocada CPU final136pasan/5CUDAomitidos248,95s; Ruff/mypy23fuentes
limpios. Suite general vigente2789pasan/39omitidos/4fallos históricos de dscribe,
no repetida al cerrar porque fuentes/versiones siguen iguales. Auditoría completa
de requisitos/comandos: [cierre del Goal](cierre_goal_recuperacion_batching_20261017.md).

### Primarias: medias entre tres semillas

H y niveles/gap en meV. Razón=media cost/media fixed; menor es mejor.
IC95 descriptivo de bootstrap pareado sobre tres seeds; baja potencia, seis
métricas sin corrección múltiple, no prueba confirmatoria de promoción.

| Población | Métrica | Fixed | Cost | Cambio | Razón(IC95) |
|---|---|---:|---:|---:|---|
|IDselección|H-MAE gauge|5,596|5,480|−2,09%|0,979(0,953–0,998)|
|IDselección|RMSE niveles alineados|63,458|59,082|−6,90%|0,931(0,860–1,008)|
|IDselección|Error absoluto gap|75,495|66,621|−11,76%|0,882(0,802–0,947)|
|OODexploratorio|H-MAE gauge|5,703|5,638|−1,14%|0,989(0,967–1,007)|
|OODexploratorio|RMSE niveles alineados|928,488|642,380|−30,81%|0,692(0,480–1,143)|
|OODexploratorio|Error absoluto gap|3109,454|2980,494|−4,15%|0,959(0,804–1,220)|

Variación pareada OOD, cambios% por seed0/1/2: H−4,01/−1,14/+1,20;
niveles−55,34/+16,27/+1,87; gap−21,98/+26,99/−0,39. La mejora media grande
de niveles procede de seed0; dos de tres semillas empeoran y todos los IC OOD
incluyen1. No declarar mejora OOD robusta ni equivalencia/no inferioridad formal.
La regresión espectral de seed1 supera5%, por lo que tampoco pasa el criterio
previo de promoción. Validación usada en selección, no testID independiente.

Auxiliares, medias de estructuras y seeds:

| Población/brazo | H-RMSE meV | Frob.relativo | p95 meV | p99 meV | SupportF1 |
|---|---:|---:|---:|---:|---:|
|ID/fixed|12,615|0,005530|22,829|53,654|0,9565|
|ID/cost|12,500|0,005494|22,182|53,823|0,9565|
|OOD/fixed|23,312|0,017651|24,512|89,753|0,9148|
|OOD/cost|23,117|0,017504|24,224|89,151|0,9148|

Falsosceros exactos0 en ambos brazos. SupportF1 usa soporte numérico umbral0,
no identifica efectos físicos de números arbitrariamente pequeños. Descomposición
OOD onsite43,670→43,396meV/offsite4,604→4,545; H-H7,696→7,635/H-O5,679→5,606/
O-O4,010→3,956. Valores completos por estructura en result.json. Mayor defecto
hermítico crudo Hpred2,634e-10; los42casos cumplen≤1e-8, S positiva.

### Recursos: mejora local medida, no promesa universal

Picos: media entre los tres picos por brazo; no mezclar con máximo global de
VRAM total del dispositivo. Los seis fits tienen una tentativa exitosa cada uno.
Máquina: NVIDIA GeForceRTX5090,32607MiB anunciados; driver595.84 al comprobar
el cierre. No es validación en hardware independiente.

| Métrica | Fixed | Cost | Cambio medio | Cambios pareados seeds0/1/2 |
|---|---:|---:|---:|---|
|Pared/fit,min|78,702|75,732|−3,77%|−5,62/−4,71/−0,95%|
|Peak allocated,MiB|1889,95|1593,99|−15,66%|−15,53/−15,55/−15,90%|
|Peak reserved,MiB|2546,67|1896,00|−25,55%|−22,77/−25,21/−28,27%|
|Peak hostPSS,GiB|13,842|12,425|−10,24%|−11,62/−11,12/−7,99%|

RSSsum medio117,090→109,717GiB es suma de páginas compartidas, **no RAM física
ocupada**; PSS es la lectura interpretable. Árbol de memoria medido=fit-hijo y
descendientes, excluye supervisor padre/evaluaciónCPU y puede perder picos entre
sondeos. No afirmar ahorro de RAM total del sistema ni estabilidad ilimitada.
Memoria larga mayor que piloto5épocas; conservar ambos como alcances distintos.

Fits supervisados27798,14s=7,7217h. Primer inicio2026-10-16 23:40:30CEST,
último fit termina2026-10-17 07:25:04CEST; span7,7427h, más cierre/evaluación
final. EvaluaciónCPU31,34s sumados; esperas inicialesGPU≈0,32s. Publicación de
recuperación5590,74s=1,5530h,20,11% de pared, **ya incluida** en los fits;
sin referencia con recuperación desactivada, no es overhead causal aislado.
Setup126,49s; suma de épocas sincronizadas14958,87s. Resto del wall incluye
transiciones de loaders/workers, hooks y finalización no aislados. No sumar estos
componentes a la pared ni llamar todo ello GPU-hours de utilización.
Primera época incluye posible compile lazy, no compilación pura medida.
La estimación4–7h y posterior extrapolación corta5½–6h infravaloraron la duración
total; se conserva la desviación, no se reajustan retrospectivamente resultados.
Reducción de pared consistente en estas tres parejas, pero no benchmark aislado
de kernels: agrupación y updates distintos, hardware/cachés/CPU no aislados.
VRAM libre mínima registrada88,0915%; temperaturaGPUmáxima59°C. Guardas respetadas
en las muestras; no garantía dura frente a asignaciones ajenas entre sondeos.

### Auditoría del gap OOD: no ocultar el resultado negativo

Los gaps de referencia de los tres clústeres son4,9742/5,1859/4,5326eV.
FDF confirma32O+64H,256electrones de valencia y128niveles ocupados sin polarizar.
Recalcular con Cholesky reproduce niveles/gaps originales: no diferencia de
solver, ocupación, unidades o desplazamiento global que explique por sí sola
el error. El gauge común desplaza niveles, no el gap. S no predicha: todo este
diagnóstico es condicional al S de referencia, no certifica predicción H+S.

En `h2oclu_test_n32_001`, los seis gaps predichos son0,039–0,205eV frente a
5,1859eV. H-MAE3,94–6,75meV no evita ese fallo. Smin0,00125637 y condición4180,48;
la norma espectral del residual gauge-alineado es1,31–3,63eV, pero la de
`S^-1/2 deltaH S^-1/2` es52,44–120,68eV. Medida consistente con gran sensibilidad
a direcciones de pequeño solapamiento; no significa que S sea singular ni
identifica todavía causa de entrenamiento/arquitectura.

Sensibilidad separada, no criterio nuevo: proyectar artificialmente S a valores
>1e-2 elimina141de736direcciones en ese caso. Cambia gapref a5,3853eV y gaps
pred a5,1046–5,5286eV. No adoptar el corte: modifica el espacio de base y también
la referencia. Otras dos estructuras descartan137/136direcciones. No sustituir
con esos resultados las primarias de la tabla ni llamarlo mejora del modelo.
Concluye sensibilidad de representación; no se corrigió generalización OOD.

### Decisión y frontera de cierre

- Implementación y comparación exploratoria6.4 completas para esta receta/dataset.
- Evidencia favorable y consistente de memoria local; pared modestamente menor.
- Evidencia insuficiente de mejora física OOD robusta; espectros/gap pueden ser
  malos en ambos brazos. Conservar opción opt-in, **defaults intactos**.
- MODERATE: tres seeds, OOD previamente observado y lineage incompleto; no test
  independiente. MAJOR para claim de mejora general: regresiones espectrales y
  gap OOD colapsado. No atribuir beneficios de fonones/derivadas ni universalidad.
- Evidencia: comparación interna controlada exploratoria, no claim-ready general.
- Siguiente diagnóstico opcional de mayor valor: atribuir los errores en las
  direcciones de pequeño solapamiento usando matrices ya guardadas. No está
  autorizado otro entrenamiento, dataset, cutoff ni cambio de pérdida/default.
