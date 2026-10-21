# Piloto7.6 w90: intento interrumpido después de H-only

Fecha:2026-10-19. Checkout Graph2Mat/g2m-v2, HEAD270d17d. Diseño y plan
aprobados por el usuario; un solo intento, sin seis fits, cola o defaults nuevos.

**Nota de continuidad:** este informe conserva el intento v1. Después de su
cierre, el usuario«Dale» autorizó corregir el supervisor y una nueva pareja
v2. El source/tests ejecutados en v1 están preservados con sus hashes exactos
en `.superpowers/sdd/2026-10-19-multitarget-w90-pilot/v1-executed-source.py`
y `v1-executed-tests.py`; la ruta principal del runner puede contener ya la
corrección. No interpretar las decisiones pendientes de v1 como estado actual.

## Resultado y siguiente decisión

**La pareja comparativa no se completó.** H-only terminó120updates
(20warmup+100medidos) y validación150. El supervisor abortó durante la salida
del worker por un falso positivo de proceso ajeno. Multitarget no arrancó.
No se puede comparar coste/VRAM entre brazos ni presupuestar los seis fits.
No se relanzó, cambió batch/modelo, amplió el plazo ni corrigió el runner
congelado después del intento: el protocolo exige detenerse tras abortar.

Siguiente autorización necesaria: corregir únicamente la identificación del
PID propio durante la salida, con regresión CPU del caso, y repetir la pareja
acotada con el mismo presupuesto máximo15min. No autorizar por inferencia
seis fits ni modificaciones del modelo, pérdidas, cutoff o datos.

## Datos medidos de H-only, provisionales y sin comparación

| Medida | Resultado |
|---|---:|
|Updates totales/medidos|120/100|
|Muestras en ventana medida|3146; incluye3batches de14|
|Mediana/p95 por update, con carga/H2D/backward/Adam y controles|60.493/65.525ms|
|Rango por update medido|58.805–505.474ms|
|Throughput ponderado|472.526muestras/s|
|Pico allocated/reserved PyTorch, warmup y medidos|698.903/858MiB|
|Máximo uso total GPU observado|2009MiB, incluye escritorio/contexto, no solo tensores|
|Mínimo libre observado|30598/32607MiB =93.839%|
|Máxima temperatura observada|40°C|
|Preparación del worker/total registrado|19.883/29.765s, sin import/arranque exterior|
|Validación|150muestras,76800elementos,0.478s, H gauge-MSE0.0001727056573eV²|

No hubo OOM ni falta de VRAM observada. El mínimo libre es muestreado cada
0.5s, no una garantía de exclusión de asignaciones externas entre lecturas.
No comparar estos números con baselines históricos como aceleración causal;
ni interpretar la pérdida tras120updates como precisión final. La pareja no
tiene réplicas independientes ni orden contrabalanceado. No hay inferencia
de test, evaluación predictiva OOD ni certificación de SPD de S_pred.

## Causa reproducida en CPU

Worker propio:PID1780761. En la penúltima telemetría figura simultáneamente
como miembro del árbol propio y como proceso CUDA. En la última, nvidia-smi
todavía enumera ese mismo PID, pero `/proc` ya no devuelve sus páginas RSS y
`process_tree_memory` produce `pids=[]`. El supervisor deriva `owned=set()`
y clasifica `{1780761}−owned` como `foreign`.

La reproducción CPU, sin consultar GPU, comprueba:
`guard_reason(último_estado,{1780761},set(),0,900)=="foreign"`;
con el PID propio conservado daNone. No apareció otro PID de cómputo en las
observaciones. Se trata de una carrera entre la salida del proceso y la
liberación/visibilidad del contexto CUDA, no de evidencia de un job ajeno.
No basta con desactivar el control de procesos: la corrección debe conservar
la identidad/vida del grupo que creó el supervisor y seguir rechazando PIDs
realmente ajenos. Esta corrección y su relanzamiento siguen pendientes.

`result-h.json` existe y valida120updates/150muestras. El padre abortó antes
de incorporarlo y conservó las120líneas parciales en `summary.json`, por eso
ese resumen tiene picos/validación null aunque el resultado de H sí los tenga.
Se conservan ambos originales; este informe distingue su estado, sin
reescribir el resumen ni fabricar un resultado conjunto.

## Preflight real y suelo de representación

1498estructuras,88.853s antes de serializar JSON. Malla SCF400puntos más
caminoΓ–K–M–Γ151puntos, sin regularizar/cortar S. MínimoλS0.1644395031;
máxima diferencia eigh–Cholesky3.456e−9eV. Guardados espectros completos
originales y reconstruidos por estructura; no son predicciones del modelo.

Se verificaron ambos lectores reales, sus grafos/labels H/S, features y
backbone inicial idénticos. Diferencia H-loss5.96e−8 en float32. IDs de las
120actualizaciones coinciden con el orden congelado. Splits1198/150/150 sin
cambios, avisos por duplicado train/val y bloques MD compartidos conservados.

| Target | Error máximo original−representado | Máxima normaF omitida |
|---|---:|---:|
|H|0.0001000482eV =0.100048meV|0.0002875464eV|
|S|1.9923e−13|0|
|DM|0.0873755250, unidad sisl_occupation|0.2592038398|
|EDM|0.0534494887eV|0.1941535544eV|

Los errores DM/EDM son colas fuera del grafo, no errores de entrenamiento.
No se cambió el cutoff para ocultarlos. Error máximo N original/reconstruido
3.553e−15; residual máximo HD−SE reconstruido por orbital8.6375e−7eV;
0casos de soporte físico no certificable en ambos. N conservado no demuestra
que toda la DM/EDM exportada sea exacta. Una comparación científica posterior
debe separar este suelo de representación del error aprendido; no presentar
la evaluación sobre el grafo como reconstrucción exacta de la matriz completa.

## Pruebas, procedencia y reproducción

Script ejecutado:`auditorias_mejoras/pilot_multitarget_w90.py`.
SHA256:`c05a402a7247eeb37e3140fc5ffd27f44e8f08be21f813b4d1f34ec48d65a345`.
Tests:`test_pilot_multitarget_w90.py`, SHA256
`d7eecb557857f6ba37bfadc4835dda0b8a50522d2ef47ac597890c32f354cd4f`.
Gate focal terminal:118passed/0skip/0fallos,3207warnings,32.71s. Ruff y
diff-check correctos. Revisión Native única:5Important+2Minor; estos dos se
recalificaron Important por efecto en evidencia científica/medición. Los7
corregidos con regresiones RED→GREEN; ningún menor diferido. La carrera de
salida descubierta en GPU es un bloqueo nuevo, no estaba cubierta por esas
pruebas. No declarar el runner libre de defectos por el gateCPU.

Artefactos originales propios, no sobrescribir:
`auditorias_mejoras/piloto_multitarget_w90_20261019_v1/`:
manifest, kpoints, preflight, order, prepared, configs efectivas, estados
iniciales sin entrenar, result-h, updates-h, telemetry y summary.
No hay result-joint ni checkpoint entrenado. El directorio ocupa≈525MiB.
Logs/gates y reproducción CPU del fallo:
`.superpowers/sdd/2026-10-19-multitarget-w90-pilot/`, incluidos
`preflight-real.log`, `pilot-real.log`, `closure-evidence.json` y
`preflight-summary.json`. Estos dos últimos son resúmenes derivados,
no sustituyen a los artefactos originales.

Comandos usados desde Graph2Mat, con el intérprete ya instalado y variables
OMP/OPENBLAS/MKL_NUM_THREADS=2, PYTHONDONTWRITEBYTECODE=1 y overlays
registrados en el ledger:

```bash
# CPU; usar una salida NUEVA si se autoriza otro intento, no repetir sobre v1.
CUDA_VISIBLE_DEVICES= "$P" auditorias_mejoras/pilot_multitarget_w90.py preflight --out auditorias_mejoras/piloto_multitarget_w90_20261019_v1
# Una sola ejecución ya realizada; v1 rechaza un segundo lanzamiento.
"$P" auditorias_mejoras/pilot_multitarget_w90.py pilot --out auditorias_mejoras/piloto_multitarget_w90_20261019_v1
```

`P=python`.
Los hashes de datos y código se volvieron a contrastar tras abortar: intactos.
Los28hashes de producto/tests del cierre7.6 también coinciden. GPU0 vuelve a
32096MiB libres,36°C,0% en la consulta de cierre; ningún proceso propio de
pilot/worker ni contexto CUDA listado. No se tocó autorun, dependencias,
defaults, datos, otros datasets, ramas, commits, worktrees ni fonones.

## Decisiones de ejecución y sus límites

- Normalizar espacios de encabezados del plan para el extractor: sin efecto funcional.
- Mantener checkout dirty, sin commits ni borrar workspace: preserva cambios; riesgo de perder evidencia si se limpia manualmente.
- Gate focal aprobado, no suite global nueva: no recertifica todo el repositorio; no se modificó producto.
- Consultas de telemetría acotadas, reutilizando helpers existentes: evitan esperas indefinidas; añaden sobrecoste de supervisión.
- Revisión de archivos explícitos en vez de rango de commits vacío: cambios sin procedencia commiteada, compensada por hashes/ledger.
- Preguntas reales de datos/CUDA asignadas al preflight/piloto; precisión científica fuera: puede cerrar interrumpido y no implica mejora predictiva.

Sin promoción, explicación OOD nueva ni autorización implícita de trabajos.
