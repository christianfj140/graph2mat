# Campaña7.6 w90 — cierre sin promoción

2026-10-19, Graph2Mat/g2m-v2, HEAD270d17d, checkout dirty preservado.
Usuario«Lanzalos» autoriza este lote después del piloto v2. No es una
reanudación del Goal técnico cerrado ni repetición de pilotos.

## Dictamen final — 2026-10-19

**7.6 cerrado como opción implementada y evaluada, no promovida.** Los seis
fits terminaron4000updates/126110exposiciones cada uno, en39.223min de lote.
El usuario pidió cerrar7.6 tras conocer el resultado negativo en validación;
se completó además la evaluación CPU de los checkpoints seleccionados, sin
reentrenar, ajustar pesos con test, tocar defaults ni usar GPU adicional.
La receta conjunta empeora H y bandas en las tres semillas al presupuesto
fijo. No demuestra que toda multitarea sea inútil ni identifica una causa
única (convergencia, escalas de gradientes o interferencia entre objetivos).
Los tres checkpoints joint mejores son el update4000: no se afirma convergencia.

Evaluación reproducible:
`cierre_multitarget_w90_20261019_v1/summary.json`, `provenance.json` y seis
`<job>/rows.jsonl`, con300estructuras/1800predicciones en total, más1800NPZ
de predicciones empaquetadas.150validation+150test históricos, sin resplit.
Sources/certificado/configs/órdenes/estados iniciales y checkpoints elegidos
verificados; resultados originales de entrenamiento intactos. CPU152.211s.

## H y bandas: comparación principal común con S_ref

Misma receta directa, semillas0/1/2 pareadas,4000updates; validación cada200
y selección por la misma gauge-MSE de H. Inferencia desde geometría sin labels.
H reportada como raíz de MSE global empaquetada (nodos y aristas únicas con
peso1), primero por seed y luego media aritmética de las tres. No es H-MAE.

| Métrica, media de3seeds, meV | H-only | H/S/DM/EDM | Razón joint/H |
|---|---:|---:|---:|
| H validation |1.98585|14.20303|7.152|
| H test |1.97951|14.18238|7.165|
| Niveles test, malla400k |35.86586|328.12710|9.149|
| Niveles test, camino151k |34.17346|400.57646|11.722|
| Gap directo test, camino |13.74777|130.23646|9.473|
| Error gap indirecto test, camino |9.70815|90.30815|9.302|

| Seed | H-only val/test | Joint val/test | Bandas test H-only/joint, camino |
|---|---:|---:|---:|
|0|1.59455/1.62970|14.77399/14.75204|27.53247/387.57614|
|1|1.88288/1.84656|17.72866/17.71161|28.29362/617.23306|
|2|2.48012/2.46229|10.10646/10.08350|46.69431/196.92018|

Niveles: RMSE sobre las8bandas y los puntos k, después media por estructura
y seed. No es la MAE histórica en ventana±2eV: no comparar directamente sus
cifras con otras campañas. Un único c por estructura calculado en el packing
de entrenamiento, `Hpred-cSref`, nunca realineado por k. Reference H/S completas
del recurso, sin cortar S ni añadir colas exactas a Hpred. Incluye el suelo
de representación ya certificado, no se presenta como error aprendido puro.

Gap: llenado nominalN8, cuatro bandas doblemente ocupadas; directo e5−e4 y
indirecto firmado min(e5)−max(e4), sin truncar a cero. No HOMO–LUMO molecular
ni ocupación exacta DFT; los datos tienen ocupaciones fraccionarias.
150/150casos válidos por seed/split/conjunto k en la ruta común.

## S_pred, DM y EDM: diagnósticos separados

S_pred fue positiva en todos los400+151puntos evaluados de cada caso, sin
reparaciones: mínimo autovalor0.1542012. Ruta Hpred−cSpred/Spred:0fallos sobre
900predicciones de estructura joint; denominadores e IDs inválidos conservados
en el esquema, nunca desaparecen de la media por silencio. No es garantía
SPD para geometrías o k arbitrarios. En test, niveles con S_pred:
561.442meV en malla y674.963meV en camino (medias3seeds), peores que H(S_ref).
La c es la misma de la comparación principal; no se optimiza con S_pred.
Como H se entrena módulo cS_ref, esta ruta también mezcla el cero libre de H
con el error de S_pred: no permite atribuir por sí sola todo el empeoramiento
a S ni demostrar una causa única. La regresión principal ya existe con S_ref.

| Seed | RMSE S test, adimensional | RMSE DM test, ocupación sisl | RMSE EDM test, eV |
|---|---:|---:|---:|
|0|0.000585241|0.00387035|0.0246511|
|1|0.000615133|0.00392828|0.0260319|
|2|0.000648958|0.00386514|0.0195381|

Son errores sobre soporte empaquetado, no una comparación contra tres modelos
independientes ni prueba de mejora de DM/EDM. Las colas fuera del grafo del
writer siguen presentes en los límites del certificado (DMmáx0.08738,
EDMmáx0.05345eV); no confundirlas con error del modelo.
N referencia8. Error absoluto medio test Tr(Dpred Sref):0.000475/0.001108/
0.003073electrones; con Spred:0.002457/0.004214/0.002309, seeds0/1/2.
Contracciones por imagen física sc_off sobre unión de soportes, no índices
CSR prestados entre nsc distintas. La cola ausente de una predicción finita
es cero del writer, no una referencia desconocida ni una etiqueta imputada.

**HD−SE guardado es un diagnóstico bruto no alineado de gauge**, no energía
total ni error físico gauge-invariante: H-loss deja libre el cero de H, mientras
EDM se ajusta en gauge archivado. Los c medios joint son23.015/24.564/20.371eV;
el residual bruto máximo por orbital promedia23.742/25.351/21.010eV en test.
No utilizar esas cifras como evidencia independiente de inconsistencia
aprendida, ni atribuirlas a error de energía total. No se impone McWeeny.

## Coste y seguridad del lote terminado

H-only: fit/validación medio4.414min; joint7.643min (×1.731). Allocated pico
698.917→2950.129MiB (×4.221); reserved860→3386MiB. No ahorro frente a H-only;
no comparación contra cuatro modelos independientes. Pared descriptiva, orden
fijo H→joint y carga CPU concurrente parcial: no estimación causal universal.
Libre mínimo86.086%,Tmax54C,4083lecturas, último contexto propio liberado;
sin OOM/abortos/reintentos, ningún fit adicional pendiente.

## Reproducción y verificaciones de cierre

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python \
auditorias_mejoras/close_multitarget_w90.py --out \
/home/christian/repositorios/graph2mat/auditorias_mejoras/<directorio_nuevo>
```

No reutilizar una salida existente. Todas las dependencias ya estaban instaladas.
H-validation reproducida con tolerancia relativa0.2%, diferencia observada
máxima0.00228%; no afirmar identidad bit a bit CPU/CUDA. Por caso se guardan
c, métricas por target, ambos conjuntos k/rutas y controles Cholesky/eigh
de espacio completo. Auditoría adicional `audit.json`:1800NPZ hashados y
recomputados contra las etiquetas originales, c y MSE por target coinciden;
IDs150/150 y hashes de filas/entradas íntegros. Discrepancia máxima entre
eigh generalizado y Cholesky independiente1.409e-11eV. Pruebas exacta/cS, pequeño S positivo sin cortes,
S no positiva/denominadores, inferencia real sin labels, writer H/joint,
contracciones y nsc1/3/5.10tests nuevos pasan, foco con regresiones en ledger.
Revisión independiente única:1Important por imágenes periódicas, corregido
RED→GREEN; sin otros Critical/Important. Producto y scripts de entrenamiento
congelados no modificados. Gate focal final59passed/0fallos. Suite CPU completa
posterior con el evaluador nuevo: **3205passed/26skip/0fallos**,48153warnings,
1249.17s, salida0 (`closure-full-cpu.log` en el workspace del ledger). No
confundir los skips de recursos con cobertura real de multi-GPU/SOC.

## Límites y decisión

Duplicado train/val y bloques MD compartidos permanecen como alarmas;
test no certifica independencia/OOD general. Tres seeds no permiten una
ley universal. No afinar receta, extender updates ni probar nuevos brazos
tras mirar test sin un experimento nuevo explícitamente autorizado.
**Recomendación: conservar7.6 opt-in, no promover esta receta y cerrar esta
campaña.** Sin reabrir el Goal técnico, defaults, fonones o autorun.
Por decisión posterior del usuario,3.13multi-GPU fuera del alcance y8.1SOC
prioridad futura aplazada; no iniciar ninguno.

## Estado de lanzamiento histórico — sustituido por el cierre anterior

Supervisor independiente PID1869115, PGID/SID1869115, PPID1 comprobado.
Arranque con stdin DEVNULL, stdout/stderr en log propio y sesión independiente;
no depende del turno del agente ni modifica autorun. Directorio exclusivo:
`campana_multitarget_w90_20261019_v1/`. El estado vivo autoritativo está en
`state.json`, y el progreso de cada fit en `<job>/progress.json` y
`<job>/updates.jsonl`. Este informe documenta el lanzamiento, no seis resultados.

Orden cerrado: h_s0,joint_s0,h_s1,joint_s1,h_s2,joint_s2. Un worker activo
como máximo, resto pendientes. Fallo/guardia/plazo detienen los siguientes;
no hay reintentos automáticos ni permiso para relanzar al día siguiente.
No borrar `started.json`, adoptar este directorio ni sobrescribir sus artefactos.

Control inicial: H seed0 PID1869163,634/4000updates y20000exposiciones;
validaciones en200/400/600, checkpoint `best-u0600.ckpt`25.75MB.119lecturas
de telemetría: libre mínimo93.833%,Tmax41C. Es una instantánea, no el estado
final: consultar los JSON vivos. Las métricas tempranas no son el resultado
comparativo ni permiso para ajustar pesos.

## Protocolo y comprobaciones

[Protocolo Native](../docs/superpowers/plans/2026-10-19-multitarget-w90-six.md).
4000updates/fit, no4000épocas; 126110exposiciones a muestras por brazo/seed,
validación150 cada200updates (20validaciones). Selección mínima de H gauge-MSE
en ambos brazos, no pérdida joint. Adam y coseno4000 continúan sin reinicios;
misma receta directa/pesos/splits/orden por seed. Inicializaciones frescas y
backbone/features compartidos exactos en los tres pares, diferencia entre las
dos implementaciones de H loss<=5.96e-8. No pesos entrenados del piloto.

Certificado v2 completo1498 y kpoints reutilizados bajo hashes, fuentes13512
y código productivo verificados; no se vuelven a resolver espectros idénticos.
Inicializaciones/configs/órdenes/runner/tests quedan hashados en campaign.json.
No editar código productivo ni runner mientras corre: los checks abortarán si
cambia. Roadmap/traspaso y este informe pueden documentar estado sin tocar datos.

Gate focal final80passed/0fallos,3858warnings,40.94s. Incluye rutas reales CPU,
recarga exacta de ambos checkpoints, criterio H independiente, continuidad
LR/IDs y terminación del hijo propio ante fallos de GPU/plazo/telemetría/I/O.
Revisión independiente única encontró un hueco de cleanup antes de escribir
PID: corregido con regresión RED→GREEN; Critical0/Important pendiente0/Minor0.
Ruff/diff limpios y piloto/producto preservados. Suite CPU completa adicional
iniciada; no confundir el gate focal con su resultado aún pendiente.

## Seguridad y presupuesto

Topes20min/fit y90min/lote. Fracción allocator<=55% y margen inicial30%;
arranque<75C, abortar>=80C, libre<25%, cómputo ajeno, error de telemetría o plazo.
Muestreo0.5s; busca>=20%libre, no garantiza protección contra asignaciones
externas entre lecturas. Solo se terminan grupos de workers propios.
Estimación previa36–48min, presupuesto prudente≈1h train/validación; costes
completos de Lightning/checkpointing y carga concurrente pueden variar.

## Límites / siguiente paso

No hay resultado final de precisión todavía. La comparación científica CPU
usará checkpoints seleccionados por validación H, H(S_ref) común y diagnósticos
S_pred/DM/EDM separados; no modificar matrices ni reparar S_pred para esconder
fallos. No evaluar test para ajustar pesos ni añadir un brazo sin autorización.
Se mantienen avisos de duplicados/bloquesMD entre splits y colas fuera del
grafo DM/EDM; este recurso no demuestra OOD independiente ni exactitud de toda
la matriz exportada. Sin defaults promovidos, nuevo dataset, fonones o autorun.
