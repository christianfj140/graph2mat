# 6.4 — batching de entrenamiento por coste declarado

Fecha2026-10-09; subplan `plan_batching_coste.md`, BASE903a132.
Implementación CPU verificada: fullCPU final2209pasan/2omitidos,exit0.
Beneficio: menor pico del **proxy** en la fixture; no ahorro VRAM ni velocidad
GPU/precisión científica demostrados. No se han iniciado entrenamientos ajenos,
autorun ni trabajos de fonones.

## Contrato y alcance

Única opción pública nueva: `MatrixDataModule(max_batch_cost=None)`, añadida al
final del constructor y guardada en los hparams. None conserva el camino fijo;
val/test/predict siguen usando batch_size, también al activar la opción.
Entero positivo, bool/float/no finitos rechazados antes de I/O.

Proxy por muestra: `C*(sum_i n_i² + sum_directed_edges n_i*n_j)`. Los tamaños
orbitales y componentes vienen del procesador; no se usan valores de etiquetas.
Prepass sobre train efectivo/Subset tras split; solo enteros cacheados por
identidad del dataset, reconstrucción al reemplazarlo y limpieza al teardown
final. Datos/geometría/basis/C deben permanecer inmutables mientras se reutiliza.
No materializa otra caché de muestras y no cambia formatos/checkpoints.

BatchSampler privado greedy conserva el orden del sampler y todas las muestras;
batch_size es límite adicional de estructuras. Oversize singleton con warning,
último batch conservado, coste cero permitido para ghost. Presupuesto no cubre
activaciones/backbone/gradientes/allocator y **no garantiza evitar OOM**.
RotatingPool dinámico y samplers no estándar rechazados. La pérdida conocida
pair_difference_mse falla antes del backbone, también tras cambiar por etapas;
no se añade un sampler de parejas ni se certifican pérdidas custom con partners.

DDP usa el DistributedSampler nativo reinyectado por Lightning. Para igualar los
pasos calcula el máximo de batches sobre los shards de todos los ranks mediante
la misma fábrica/seed/epoch Torch, sin collectives en len. Repite batches locales
cíclicamente hasta ese máximo y registra el padding. Se suma al padding de
muestras nativo; las métricas cuentan todo lo efectivamente procesado, incluidos
repetidos. No representa evaluación única de cada estructura original.

## Hallazgo de integración: longitud variable por época

El plan inicial omitía que Lightning guarda los límites del primer loader.
Regresión real RED: DDP debía pasar3→4 batches pero procesaba3→3; un proceso
perdía las muestras1 y2 en la siguiente época. No bastaba set_epoch.

La opción activa con warning el ajuste público
`reload_dataloaders_every_n_epochs=1`: recrea loaders train/val, no sus datasets
ni la caché de enteros. El constructor privado captura la época actual antes de
la reinyección DDP y las consultas iniciales de len. Single usa generator
privado con initial_seed+epoch; cada len reserva el mismo plan que iter durante
la época, sin consumir RNG global. Sin tocar internals de Trainer ni dependencias.
Este cambio observable queda documentado; None no modifica ese ajuste.

Revisión final detectó otra interacción: dos workers pueden agotar el sampler
durante prefetch, antes de que Lightning vuelva a consultar len. Invalidarlo
ahí reservaba otra permutación más corta y truncaba el batch activo. RED real
con costes9,1,1,1,1 seed0: faltaban ids1,2,3. El plan se retiene si el constructor
recibe una época explícita de Trainer, incluso después de agotarse; siguiente
loader/epoch lo reemplaza. Standalone sin Trainer conserva progresión del RNG
entre iteraciones. GREEN40 en ambos perfiles incluye fit real workers2,3→2
batches por época y cobertura única de las cinco muestras.

## Evaluación CPU pareada

Primaria prefijada: reducir pico del proxy con cobertura exacta. No regresión
numérica de particiones: float64 rtol/atol1e-10 en MACE real, C1/C2,
simétrico/dirigido y bases heteronucleares s/p/d/f. No es prueba de precisión
científica ni equivalencia de trayectoria de entrenamiento: cambian agrupación,
pasos por época y peso de cada update; futuras comparaciones deben parearlos.

Python3.12.3/Torch2.11cpu/float32; fixture sintética dirigida C2 propia del
benchmark de caché. Secuencia congelada de puntos:
`4,4,72,4,16,128,4,16,72,4,16,4`. Warmup3,12rondas alternadas, mismo trabajo
total, muestras y collate; sin modelo/optimizador en los timings CPU.
Presupuesto100656, límite5estructuras. Baseline BatchSampler estándar versus
sampler de producto, orden secuencial idéntico para aislar el empaquetado.

| Medida | Fijo | Coste |
|---|---:|---:|
| Batches / 12 muestras | 3 | 7 |
| Pico del proxy | 313152 | 178944 |
| Pico bytes de tensores del batch | 1309384 | 748104 |
| Collate+sampler por época, mediana ms | 1.307 | 2.557 |
| Lazy acceso+collate+sampler por época, mediana ms | 17.834 | 19.412 |
| Primer loader, proceso nuevo, ms | 0.153 | 17.625 |
| Delta RSS al crear loader, KiB | 16 | 3144 |
| Bytes de caché de costes enteros | 0 | 472 |

Pico proxy≈−42.9%; bytes de tensores≈−42.9%. La estructura128puntos supera
budget y sale sola: coste178944, no presupuesto estricto del batch ni VRAM.
Collate≈+96%, ~1.25ms/época adicional; acceso+collate≈+8.8%, ~1.58ms.
**No ganancia de velocidad CPU**: los siete batches requieren más collates.
El prepass inicial añade ~17.47ms en configuraciones en memoria, no parsing de
ficheros. RSS incluye runtime/allocator y transitorios; peak_delta0 en ambos
solo significa que no superan un pico previo, no ausencia de consumo adicional.

Dispersión min,max ms: collate fijo(1.135,1.344),coste(2.517,2.711);
lazy fijo(17.569,18.292),coste(19.186,20.206). Host no aislado; fullCPU propio
concurrente. Pares alternados y dispersión conservada, no comparación GPU.

SHA256 tensores de la secuencia:
`62812a5cc964045b36179c1f2eec1b6d3177ae876a986a9e9764510eb744c06c`.
Source data.py:
`ed975e5722f238f0a055779adedcc368dcdb5e61817d087d8523b942b4ffe2ef`.
Source model.py:
`ac1b34ed4657b16e5f28ba87356c809080931f8622fdba0dc57949a531d8103a`.

## Reproducir

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/christian/repositorios/graph2mat/src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/tmp/g2m-python312-full.LHyni1/venv/bin/python \
auditorias_mejoras/bench_batching_coste.py
```

Reutiliza `bench_cache_tensores_cpu.dataset_for`; no datasets del usuario.
Asserts de costes literales, cobertura exacta, trabajo total y singleton antes
de medir. Subprocesos nuevos para RSS/inicialización. CUDA habilitada o rounds
inválidos se rechazan. Logs/JSON en `/tmp/g2m-precision-goal.A2Pbgk/cost-batching-*`.

## Verificación

- Baseline79pasan/245warnings/2.63s,exit0. Primer RED22fallos por flag/sampler
  inexistentes; GREEN22. BoundaryRED2fallos por pairguard; GREEN29.
- DDP fixture primero sin init_model (modelo base exige inicialización);
  luego mutación postconstructor revertía sampler por replay de Lightning.
  Ambos corregidos solo en fixture. Ordenado usa overfit_batches=1.0, opción
  pública que desactiva shuffle train; shuffled prueba el hook de producto.
- RED longitud variable2fallos/2pasan/17.71s; GREEN38/27.44s. Control negativo
  desactiva temporalmente el budget: ambos ranks fallan con
  "DDP silently used fixed batching"; retirado antes de gates finales.
- Focused final39pasan/36deseleccionados/659warnings/28.37s,exit0. Incluye cuatro
  fits reales: DDP ordenado/padding, shuffled C2,3→4 pasos, y single sin pérdida
  de muestras. SSE/count global contra literal de los elementos procesados.
- Workers0/2 spawn/fork persistentes, mutación sin contaminar caché,
  Subset/cacheidentity/finalteardown, ZIP nativo/temporales propios, ghostcost0,
  targets de coeficientes C1/C2 y default fijo/pares intactos.
- Floor Python3.9.25/Torch2.4.1cpu39pasan/328warnings/25.97s,exit0; mismas
  regresiones nuevas, también DDP real y equivalencia MACE. No wheel nueva.
- Floor regresiones previas126pasan/609warnings/18.23s,exit0: split/stages/ZIP/
  temporales/trainingstages/checkpointprovenance/CLI/epochmetrics. Benchmark
  floor --rounds1 exit0, asserts pasan; no evaluación comparativa adicional.
- Ruff explícito source/tests/benchmark y all-files, mypy-core y compile39
  verdes. Source/tests congelados durante fullCPU final; strictRSTparser y
  git diff --check también exit0.
- Primera fullCPU:2207pasan/2omitidos/26518warnings/594.68s,exit1 por SIGABRT
  nativo al salir del subprocess de una prueba DDP existente (dominios).
  Ambos rankJSON ya tenían valores literales correctos; aislada pasa1/5.38s.
  El script no cerraba Gloo explícitamente; PL2.6.6 teardown no lo destruía.
  El aborto al salir es compatible con liberación tardía; sin backtrace C++
  no se atribuye concluyentemente el SIGABRT a un destructor concreto.
  Se añadió destroy_process_group al final del main de cada proceso de pruebas,
  conforme al [cierre documentado por PyTorch](https://docs.pytorch.org/docs/2.11/distributed.html#shutdown).
  Sin retries ni ignorar abortos, sin cambio de producto por este fallo.
  Source intacto, tests refrozen para repetir DDP y fullCPU; se exigió exit0
  con los archivos finales antes de certificar.
- Después del cierre explícito, floor3.9 **75pasan/328warnings/196.18s**,exit0:
  archivo DDP completo y todas las pruebas nuevas, no solo los cuatro fits nuevos.
- Misma selección finalPython3.12 **75pasan/659warnings/247.69s**,exit0.
  Las métricas literales y los errores de ruta/presupuesto siguen exigidos;
  no se relajan tolerancias ni se añade retry al abortado.
- FullCPU posterior al cierre Gloo:2208pasan/2omitidos/26518warnings/653.69s,
  exit0, pero **anterior** a la corrección final de prefetch. No certifica esa
  versión final: se repitió fullCPU sobre source/tests nuevamente congelados.
- Último focused40pasan/36deseleccionados/659warnings/32.75s,exit0;
  floor3.9 40pasan/328warnings/29.03s,exit0. Benchmark repetido con el nuevo hash
  de source, mismas muestras/tensores y primarias; tabla anterior sustituye la
  medida pre-prefetch. No escoger la variante más favorable.
- **Full sourceCPU final2209pasan,2omitidos,26518warnings,634.88s,exit0.**
  cost-batching-full-cpu-final.log; salida leída, source/tests congelados y sus
  hashes idénticos al finalizar. Incluye CLI/checkpoints/ZIP/stages previos y
  todos los DDP, además del nuevo caso de prefetch. Sin wheel nueva.
- Producto local **3a0af15**; task-done real40pasan/36deseleccionados/
  659warnings/30.66s,exit0, rango903a132..3a0af15. Ledger y salida leídos.
  Subplan cerrado CPU, sin cierre del goal completo ni afirmación de mejora
  GPU/precisión científica. Archivos ajenos sin modificar, sin push/merge.

GPU al verificar: RTX5090,5671MiB,100%, cómputo ajeno. Ningún lanzamiento propio;
allocated/reserved, paso/throughput y entrenamientos científicos pendientes.
Siguiente implementación:7.1 (readout eficiente), con su diseño/perfil previo.
El goal completo permanece abierto.
