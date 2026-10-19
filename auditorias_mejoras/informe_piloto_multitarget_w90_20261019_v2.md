# Piloto7.6 w90 v2 — pareja completada

Fecha:2026-10-19. Graph2Mat/g2m-v2, HEAD270d17d, checkout dirty preservado.
Usuario«Dale»: corrección mínima del supervisor y UNA nueva pareja con tope
total15min. No autoriza seis fits ni cambiar modelo, receta, defaults o datos.

## Dictamen

**H-only y H/S/DM/EDM terminaron120updates cada uno**,20warmup+100medidos,
con validación150 por brazo. Resultado `status=complete`, sin OOM, deadline,
proceso ajeno ni repetición automática. Ambos contextos CUDA liberados antes
de cerrar/pasar al siguiente brazo. No queda ningún job propio ejecutándose.

Multitarget funciona en CUDA, pero aquí **no mejora velocidad ni consumo de
VRAM frente a H-only**: predice cuatro objetivos y cuesta1.777× por update,
4.209× allocated y3.944× reserved. Esto no compara con cuatro modelos
independientes ni demuestra beneficio/perjuicio final de precisión.

## Medidas contemporáneas

RTX5090,32607MiB totales, driver595.84. fp32/TF32off, batch32,0workers,
cache CPU, mismo backbone inicial y orden de muestras, seed0. Sin compile,
checkpointing o escalado nuevo. Wall incluye carga/H2D, forward/loss,
backward/Adam/scheduler y controles de finitud; logger externo desactivado.

| Medida | H-only | H/S/DM/EDM |
|---|---:|---:|
|Updates totales/medidos|120/100|120/100|
|Muestras medidas|3146|3146|
|Mediana por update|58.915ms|104.713ms|
|p95|62.638ms|110.191ms|
|Rango medido|56.005–332.146ms|100.634–801.609ms|
|Throughput ponderado|502.337muestras/s|278.344muestras/s|
|Pico allocated, medidos|698.903MiB|2941.485MiB|
|Pico allocated, warmup|698.903MiB|2943.702MiB|
|Pico reserved, ambos periodos|858MiB|3384MiB|
|Máximo uso total GPU muestreado|2009MiB|4535MiB|
|Mínimo libre observado|93.839%|86.092%|
|Máxima temperatura|40°C|42°C|
|Preparación interna del worker|19.922s|28.647s|
|Total interno del worker|29.330s|45.796s|
|Validación150, tiempo|0.483s|0.994s|
|PSS pico observado del hijo|2769.979MiB|2644.155MiB|

Ventana de telemetría de la pareja:83.370s. Los totales internos no incluyen
import/arranque exterior; preparación CPU previa aparte. Uso total GPU incluye
escritorio/contexto y no equivale a allocated/reserved. PSS es del hijo, no
del padre supervisor ni de toda la sesión; no afirmar ahorro general de RAM.
La reserva libre superó20% en todas las lecturas, pero el muestreo0.5s no
garantiza exclusión frente a asignaciones externas entre lecturas.

### Presupuesto orientativo de seis fits futuros

Para4000optimizer updates por fit, con validación150 cada200updates y al
final (20validaciones), mismo pipeline de este piloto:

| Estimación por fit | H-only | Joint |
|---|---:|---:|
|Central|280.09s ≈4.67min|500.62s ≈8.34min|
|Banda orientativa|263.90–348.10s|464.41–609.19s|

Tres semillas por brazo:central39.04min; banda36.42–47.86min. El extremo
superior incluye25% de margen sobre p90+setup/validación; no es un IC ni una
garantía. Presupuestar aproximadamente1h para entrenamiento/validación sería
prudente; inferencia/evaluación científica CPU, checkpointing/logging completos,
importaciones y posibles interrupciones no están completamente medidos.
**No se lanzaron ni encolaron esos seis fits.** Requieren nueva autorización.

## Precisión: todavía no hay conclusión

La validación es solo un control de finitud después de120updates:
H gauge-MSE=0.0001727057865eV² en H-only y0.03746595055eV² en joint.
Joint va menos avanzado en H en este punto temprano; esto merece atención
en el experimento final, pero no mide convergencia ni beneficio al presupuesto
de4000updates. No se eligieron pesos ni checkpoints con estos valores y no
se ajustó la receta tras observarlos. Heads de distinta anchura no tienen
pesos iniciales idénticos aunque sí compartan backbone/features exactos.

No hay inferencia de test ni nuevo recurso OOD. Se conserva el split histórico
1198/150/150 y sus avisos (duplicado train/val y bloques MD compartidos), sin
redistribución. Solo seed0 y orden fijo H→joint, sin réplicas independientes
o contrabalanceo: ratios descriptivos del piloto, no demostración causal de
rendimiento estable en otros materiales, bases o tamaños de batch.

Preflight v2 real:1498estructuras,89.360s antes de serializar, espacio completo
en400puntos SCF+151deΓ–K–M–Γ. Referencias y lectores correctos, S sin cortes
ni reparación; H-loss entre implementaciones difiere5.96e−8 en float32.
Kpoints, receta, orden y estados iniciales son byte-idénticos a v1. Se repitió
el preflight para certificar el source corregido, no un entrenamiento previo.

Las colas DM/EDM fuera del grafo siguen siendo un suelo de representación,
no un error aprendido: el análisis y límites están en el informe v1 y en
los espectros/métricas completos del preflight v2. No cambiar cutoff ni
presentar precisión sobre el grafo como exactitud de toda la matriz exportada.

## Corrección del falso proceso ajeno

La carrera de v1 se reprodujo con `pilot()` en CPU: el líder Popen aún no
había sido recogido, pero `/proc` ya no ofrecía RSS y CUDA seguía enumerándolo.
Se conserva como propio hasta que `poll()` recoge su salida: mientras no se
recoge, su PID no puede reutilizarse. Para hijos desaparecidos se captura
únicamente `ProcessLookupError` al verificar su grupo.

Después de la limpieza, se exige lista CUDA vacía. Solo puede tolerarse por
≤2s la entrada obsoleta del líder recién salido cuando `/proc/PID` no existe,
siempre dentro del plazo global900s. Si reaparece ese PID, aparece otro,
falla telemetría, sube temperatura o falta VRAM, se aborta. No se desactivó
ningún guardia ni se añadieron reintentos automáticos.

Cinco regresiones nuevas verificaron RED→GREEN: carrera real de `pilot()`,
PID ajeno, PID reutilizado, límite de2s y deadline global más corto. Gate
focal terminal123passed/0skip/0fallos,3207warnings,32.52s; Ruff/diff correctos.
Solo supervisor/tests experimentales cambiaron. Producto7.6 y entrenamiento
intactos;28hashes anteriores de producto/tests siguen coincidiendo.
No nueva revisión independiente completa: parche focal verificado por
regresión, gate cercano y ejecución de ambos brazos en hardware.

## Reproducción y procedencia

Artefactos: `auditorias_mejoras/piloto_multitarget_w90_20261019_v2/`:
manifest/configs/estados iniciales/order/preflight, result-h/result-joint,
updates por brazo, telemetry y summary. V1 queda intacta, incluidos sus hashes
de result-h/summary. Source/tests exactos de v1 archivados en el workspace.

Source v2 SHA256:
`e4c13d2c5de2c0caa7dba8d40d8da8a189a330b6de04cee9de6074c8c3fca21a`.
Tests v2 SHA256:
`931b7f33916863e6f3d25fde2c90cc2bc2fd3be7865ed2ab256bafbac1aea3b0`.
Stack:NumPy2.4.4, SciPy1.17.1, sisl0.16.4, torch2.11.0,
mace-torch0.3.15, Lightning2.6.1. Usar el intérprete instalado ya autorizado.

```bash
# Desde Graph2Mat; P es el intérprete del entorno registrado.
# Comandos YA realizados una vez: v2 rechaza sobrescritura/relanzamiento.
CUDA_VISIBLE_DEVICES= "$P" auditorias_mejoras/pilot_multitarget_w90.py preflight --out auditorias_mejoras/piloto_multitarget_w90_20261019_v2
"$P" auditorias_mejoras/pilot_multitarget_w90.py pilot --out auditorias_mejoras/piloto_multitarget_w90_20261019_v2
```

`P=/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python`;
PYTHONDONTWRITEBYTECODE=1, OMP/OPENBLAS/MKL_NUM_THREADS=2 y overlays instalados
registrados en el ledger. Evidencia y logs propios:
`.superpowers/sdd/2026-10-19-multitarget-w90-pilot/{pid-fix-red.log,pid-fix-green.log,pid-fix-cpu-gate.log,preflight-v2.log,pilot-v2.log,closure-v2.json}`.

Cierre CPU independiente del resumen: IDs reales por update, LR cosine4000,
120/100updates,3146muestras, validación150 y liberación de ambos contextos
contrastados; fuentes/código nuevamente hashados intactos. Nada de autorun,
dependencias nuevas, defaults, otros datasets, commits, ramas o fonones.

## Siguiente paso

El coste ya permite solicitar presupuesto para el comparativo científico de
seis fits (semillas0/1/2 por brazo,4000updates, misma receta/splits/selección H).
Su propósito sería medir precisión/convergencia, no prometer ahorro GPU que
este piloto no observa. Mantener evaluación con S_ref separada de S_pred,
contar fallos SPD sin reparar y distinguir el suelo de representación.
No promover defaults ni dar por resuelto OOD; esperar autorización del lote.
