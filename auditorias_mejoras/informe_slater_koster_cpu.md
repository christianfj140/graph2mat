# 7.4 — Slater–Koster neuronal + residual: evidencia CPU

Estado 2026-10-10: **goal acotado de implementación y verificación CPU cerrado**.
Implementación opt-in, benchmark, gates y revisión independiente terminados.
GPU/VRAM y precisión científica **no evaluadas**: 7.4 global sigue abierto.
Defaults intactos.

## Alcance y selección

Base de producto `8c8dc3e`, diseño `8d49845`, plan `3981be9`. Implementación
`92e1ad3`, `80f3cf5`, `9c50ee5`; benchmark `f72a96a`, corrección RSS `f0acdad`.
HEAD de código/tests verificado: `8a5cc7c`; los commits posteriores solo
consolidan documentación y evidencia, no cambian el producto medido.
Mismo checkout `g2m-v2`; sin dependencias nuevas, GPU, fonones, autorun, colas,
otros repositorios, push/merge ni cambios ajenos.

```yaml
model:
  edge_block_readout: graph2mat.bindings.e3nn.E3nnSlaterKosterEdgeBlock
  edge_block_readout_kwargs:
    rank: 4
```

SK predice integrales sigma/pi/delta/phi por par de capas radiales y componente,
las transporta con armónicos globales mediante mapas pequeños calibrados en Q,
y suma `E3nnLowRankEdgeBlock` sin modificar 7.1. No construye marcos Euler en
forward. Mantiene RTP y reconstrucción densos: **no es SO(2) general ni una
garantía O(L³)**, y no equivale al TP completo a rango finito.

Admite s/p/d/f naturales completos, capas repetidas, homo/hetero, C1/C2 reales,
grupos point_type/basis_shape/max y readout single/sum/mean. C2 no representa
SOC. Rechaza l ocupado >3, paridad no natural, vectores cero/no finitos/
direcciones inconsistentes, shifts de arista no nulos y contexto Hamiltoniano
no envuelto, también con custom readout. `max` puede contener `0x4e`: es una
capa ausente, no un orbital g admitido.

Compile, gradient checkpointing y AMP no obtienen certificación nueva por
herencia; restricciones existentes se conservan. La certificación aquí es
CPU float32/float64, no CUDA, complejo, SOC ni precisión de materiales.

## Corrección e integración comprobadas

| Requisito | Evidencia nueva |
| --- | --- |
| 16 pares s/p/d/f, cada integral y capas radiales repetidas | Oráculos axiales literales y transporte orbital en tres rotaciones, float32/64; Q mal calibrado rechazado |
| Normalización s-s/s-p/p-p y C1/C2 | Fórmulas analíticas independientes, dirección genérica |
| O(3), libertad de marco y continuidad | Rotación/inversión, dos marcos de test, ejes y direcciones casi polares, f-f L=6 |
| Ambas ramas efectivas | Ablación por pesos; residual con componente fuera del subespacio axial SK |
| Conditioning y fuentes | Solo 0e, incluso desplazado tras 1o/0o; dos fuentes de distinta dimensión y kwargs invertidos |
| Derivadas y cutoff | Gradcheck inputs/parámetros; diferencias finitas de posición con shifts y topología fija; valor/derivada cero en corte suave |
| MACE y coefficient-only | 36 combinaciones C/grupo/modo/reutilización, pérdidas/gradientes contra matriz, layouts, tipo ausente, fallback real por certificado inválido y mp_cutoff |
| Periodicidad física | sisl s/p/d/f, imágenes opuestas y self-image, cartesian/spherical/siesta_spherical, rotación/inversión, permutación y supercelda |
| Entrenamiento y serialización | Dos pasos CPU C1/C2; cambio de SK/residual y reload estricto exacto; class path CLI existente |
| Legacy | 165 claves sin cambios; carga estricta de captura anterior y predicciones exactamente iguales, float64; defaults sin buffers SK |

Tolerancias angulares: 2e-12 float64, 2e-5 float32; Q generado en su dtype
real, no float32 convertido pretendiendo precisión doble. Calibración certifica
ortogonalidad, soporte, rango e intertwiner con rotación genérica (128 eps).
El primer GREEN de mapas detectó permutación de componentes no axiales que
el chequeo axial solo no distinguía; se corrigió la certificación, no el oráculo.

RED observado: 37 fallos por helper ausente; 20 por operación/export ausente;
integración por `edge_vectors` no publicado. `max` detectó rechazo erróneo de
multiplicidad cero. Fixture PointBasis ya rechaza paridad antes del wrapper;
selección de warnings distingue aviso científico de avisos de TorchScript.

Gates previos reales: 209 passed / 1 skip existente (54.22s), regresión modelo/
coeficientes/contexto/CLI 291 passed (313.64s), operación Python3.9 70 passed
(6.82s). No sustituyen los gates finales posteriores que se registrarán abajo.

## Benchmark emparejado CPU

Comando real, exit 0, SHA medido `f0acdadccacab38d4f596d39f14953db564b9f33`:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/tmp/g2m-python312-full.LHyni1/venv/bin/python \
auditorias_mejoras/bench_slater_koster_cpu.py \
--points 72 --rank 4 --rounds 12 --warmup 3
```

Intel Core Ultra 9 285K, 24 cores disponibles, Torch usa 2 hilos. Python3.12,
Torch2.11.0+cpu, e3nn0.4.4, MACE0.3.15, Lightning2.6.1; pip check exit0.
Fixture sintética propia C2, 72 puntos/432 aristas dirigidas, s/p/d/f;
backbone16x0e+16x1o+16x2e+16x3o, dos interacciones. Backbone y onsite idénticos
por carga estricta y hashes; mismo input, dtype y trabajo. Capacidad distinta.

Mediana ± MAD, ms, modelo completo:

| Variante | Forward | Backward | Paso Adam | Parámetros totales / edge |
| --- | ---: | ---: | ---: | ---: |
| baseline | 145.652 ± 0.916 | 213.017 ± 3.602 | 392.043 ± 4.329 | 526016 / 197120 |
| 7.1 low_rank | 52.398 ± 0.558 | 119.227 ± 1.422 | 184.953 ± 3.550 | 356400 / 27504 |
| 7.4 SK+residual | 56.002 ± 0.801 | 119.842 ± 2.442 | 190.720 ± 4.014 | 364898 / 36002 |

Operación aislada, mismo wrapper/pair16x16/C2/Q y features, mediana ± MAD ms:

| Variante | Forward | Backward | Paso Adam |
| --- | ---: | ---: | ---: |
| baseline | 147.800 ± 7.460 | 60.057 ± 2.140 | 197.566 ± 3.308 |
| low_rank | 5.999 ± 0.361 | 11.630 ± 0.218 | 18.321 ± 0.657 |
| SK+residual | 8.957 ± 0.754 | 13.175 ± 0.199 | 22.589 ± 0.792 |

SK frente al baseline: **−51.35 %** tiempo por paso completo, **−30.63 %**
parámetros totales. Frente a 7.1: **+3.12 %** tiempo y **+2.38 %** parámetros.
La operación aislada también cuesta más que 7.1. SK es un prior adicional,
no una optimización gratuita. Son observaciones de esta fixture CPU/carga,
no garantías de materiales, GPU o entrenamiento de igual precisión.

RSS pico de proceso fresco Linux (MiB): baseline **1943.51**, low_rank
**1214.85**, SK+residual **1216.09**. Incluye imports, construcción, batch y
cuatro pasos Adam; **no readout-only, no VRAM**. Payload de parámetros/buffers
en bytes, respectivamente: baseline2104064/424212, low_rank1425600/439460,
SK1459592/439904. No incluye allocator ni overhead Python en estos payloads.

Primer intento RSS quedó contaminado por el padre con tres modelos residentes
(2896.27MiB en todos los hijos). Probe local confirmó Linux conserva el pico
pre-exec; workers ahora arrancan **antes** de construir modelos de timing.
Regresión RED→GREEN del orden. JSON inicial retenido como superseded en la
evidencia, no usado en esta tabla. El suelo de imports heredado sigue siendo
parte del alcance y puede ocultar diferencias pequeñas.

Timing incluye clones/loss en forward; backward excluye forward; paso incluye
forward/backward/Adam, no zero_grad. Checks finitos/gradientes fuera del reloj.
Solo warmup y rondas de la fase step actualizan pesos; warmup inicializa Adam.
Orden rotatorio/reverso, misma cantidad de actualizaciones por variante.
Batch/features conservan hashes. Host no aislado de carga ajena. Los tiempos
de fases separadas no tienen que sumar el tiempo del paso completo.

Crudos, median/MAD/min/max, hashes y scopes:
[resultados_slater_koster_cpu.json](resultados_slater_koster_cpu.json).
No son matrices/etiquetas científicas ni validación de MAE de materiales.

## Gates finales y revisión

Posteriores al último cambio de producto, CUDA desactivada y PYTHONPATH absoluto:

- Python3.12/Torch2.11cpu: **120 passed**, 6462 warnings, 142.22s, exit0.
- Python3.9/Torch2.4.1: **120 passed**, 3023 warnings, 171.40s, exit0.
- Ruff src/tests y script de benchmark explícito: exit0. Mypy-core: 23 archivos,
  exit0 (no equivale a typecheck de todo MACE/e3nn). CompileallPython3.9 y
  git diff --check: exit0. Pip check: sin dependencias rotas.
- Revisión independiente única, fresh-context gpt-6-astra, rango
  `8c8dc3e..8a5cc7c`: **0 Critical / 0 Important / 0 Minor**; regraduada por
  efecto real, sin fixes ni menores diferidos. Todos los hashes y18resúmenes
  del benchmark concuerdan con sus muestras. Comprobaciones independientes
  adicionales: capas repetidas+C2+rotación impropia (error máximo3.77e-15),
  reload estricto float64→float32, pérdidas dirigidas en3grupos y ajuste real
  de escalado C2: exit0.
- Suite completa CPU: **2597 passed / 2 skipped**, 36532 warnings,
  **850.43s (14m10s)**, exit0. Omisiones existentes: ejemplo HSX local ausente
  y prueba torch.compile/inductor que exige GPU. Ningún test nuevo omitido.
- Repetición final enfocada al registrar Task5 tras documentación: **120 passed**,
  6462 warnings, 134.99s, exit0. Razones de las dos omisiones confirmadas con
  `-rs` en `skip-reasons.log`. No cambios de producto entre gates.

Fallos abiertos: **ninguno** en el alcance CPU. Los fallos RED y el primer
RSS contaminado se corrigieron; no cuentan como resultados finales válidos.

Los 120 tests nuevos/enfocados se corresponden con
`bindings/e3nn/modules/tests/test_slater_koster_readout.py` (71),
`tools/lightning/tests/test_slater_koster_model.py` (47) y
`tests/test_slater_koster_benchmark.py` (2). Ninguno omitido. Comando común:

```bash
python -m pytest \
src/graph2mat/bindings/e3nn/modules/tests/test_slater_koster_readout.py \
src/graph2mat/tools/lightning/tests/test_slater_koster_model.py \
tests/test_slater_koster_benchmark.py -q --tb=short
```

Se usaron los dos intérpretes existentes arriba/Python3.9; env CUDA/PYTHONPATH/
hilos idéntico al benchmark. Para el gate agregado: `python -m pytest
src/graph2mat tests -q --tb=short`. El histórico2477/2 no sustituye este gate.

## Decisiones y límites, incluidos los apartados por el revisor

| Decisión | Motivo / coste si no cubre otro uso |
| --- | --- |
| Mismo checkout, sin worktree ni instalaciones | Preferencia/restricción explícita; no aislamiento de rollback, commits propios conservan recuperación |
| Parser CLI real de test_cli.py, no ubicación errónea del plan | Reutilizar API existente; posible diferencia de comando completo, cubierta por regresión CLI vecina |
| Digest portable de claves legacy; captura de pesos/salidas en gate local | No añadir skip dependiente de scratch; CI no repite la captura histórica exacta |
| RSS antes de modelos de timing | Evita suelo del padre con3modelos; imports/construcción aún pueden ocultar diferencias pequeñas |
| No juzgar precisión/bandas/gap/OOD/promoción | Fixtures sintéticas no bastan; prior opcional podría empeorar materiales, defaults intactos |
| No certificar GPU/VRAM/AMP/compile/checkpointing nuevos | Alcance CPU y restricciones conservadas; modos no probados podrían fallar o regresar |
| No extrapolar rendimiento | Solo fixture/host medidos; otros datos/hardware pueden cambiar resultados |
| No certificar corrupción manual Q/mapas o overrides inconsistentes de metadatos generados | Construcción normal, conversión dtype y checkpoints propios estrictos sí probados; estados manipulados pueden invalidar mapas científicos |
| Gates agregados a cargo del ejecutor, no del veredicto del revisor | No cierre por opinión; una conclusión prematura ocultaría regresiones |
| Ruff --config include en vez de flag --include inexistente | Misma regla/archivo explícito; descubrimiento de otros archivos podría diferir |
| Conservar scratch propio y archivar evidencia en vez de borrarlo | El plan prohíbe borrados; scratch ignorado podría perderse con git clean, por eso existe copia versionada |

Revisión completa, comandos/diagnósticos, ledger, captura legacy y logs:
[evidencia_slater_koster_cpu.tar.gz](evidencia_slater_koster_cpu.tar.gz).
Scratch propio conservado en `.superpowers/sdd/2026-10-09-slater-koster-readout-7-4/`
según la restricción de no borrar. No se ha solicitado una segunda revisión.

## Pendiente y siguiente paso

Preparado, **no ejecutado**:
[protocolo_gpu_precision_7_4.md](protocolo_gpu_precision_7_4.md).
Faltan dataset/material y manifests de splits concretos antes de autorizar
comparación de precisión. Tres semillas emparejadas, H gauge-aligned primaria,
H raw/bandas/gap/OOD secundarias; sin degradación OOD >5 %. GPU/VRAM por batch
fijo y máximo por separado. 7.4 global y Task4 completo permanecen abiertos.
