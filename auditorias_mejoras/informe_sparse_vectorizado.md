# 6.2 — COO y CSR multicomponente vectorizados

Fecha2026-10-09; subplan plan_sparse_vectorizado.md, BASEcc334f0.
Sin API/flag/dependencia nueva. Cambia solo sparse.py, no kernel escalar Cython
ni búsqueda de vecinos. COO usa offsets/repeat con memoria O(nnz de bloques),
sin matriz global densa. Reverse edges conservan orden original de valores,
incluidos rectángulos/imágenes/ghost. CSR C2 agrupa entradas por bloque, respeta
primer encuentro del dict, último duplicado y primer col negativo de cada fila.

## Corrección de precisión

Default público fillint0 asignaba decimales C2 a arrays enteros; fillfloat también
descartaba imaginaria C2. RED literales demuestran ambos. Nueva asignación
float64 para datos reales como el camino nativo anterior; complex128 conserva
imaginaria C2. No cambiar kernel escalar ni afirmar soporte complejo end-to-end.
El loader H ya pasaba0.0 y DM/EDMNaN: no evidencia de corrupción de jobs activos.
Los índices COO vacíos quedan enteros; orbitals/edges TorchCPU se convierten
localmente a NumPy (mgrid anterior fallaba con dtype Torch). No soporte CUDA nuevo.

## Evaluación CPU pareada

Primarias fijadas: tiempo coords y CSRC2; no regresión>10%, elementos/dtype/orden
exactos. Python3.12.3/Torch2.11cpu/NumPy2.4.4; float64,C2; warmup3,12rondas
alternadas,4repeticiones. Fixture propia s/p/d/f periódica16/72 puntos,
48/216 aristas únicas,11184/50328nnz,shape160x480/720x2160, sin datos del usuario.
Baseline son AST de exactamente los dos helpers de Gitcc334f0: sin ejecutar
modulebody/converter registration, checkout ni mutación de source.
Constructor literal independiente prepara CSR, no calienta helper candidato.
La primera medida preparaba CSR con candidato: sustituida, especialmente memoria.

Source sparse.py SHA256:
`f5704cd791698dce6804ef64df17ee725ff023f50280e7d504ab3335d98c5798`.

| Mediana ms, base → candidato | 16 puntos | 72 puntos |
|---|---:|---:|
| Coordenadas COO | 1.228 → 0.080 | 6.641 → 0.260 |
| CSR C2 → bloques | 4.677 → 2.508 | 21.377 → 11.214 |
| Primera llamada coords, proceso nuevo | 1.263 → 0.109 | 6.124 → 0.338 |
| Primera llamada C2, proceso nuevo | 4.754 → 2.773 | 21.348 → 12.551 |
| RSS delta coords KiB | 16 → 0 | 1084 → 0 |
| RSS delta C2 KiB | 0 → 680 | 0 → −104 |
| Bytes de salida, cada helper, idénticos | 178944 | 805248 |

Dispersión min,max: coords16 base(1.169,1.283),candidato(.075,.083);
72 base(6.459,6.904),candidato(.251,.274). C2 16 base(4.644,4.743),
candidato(2.474,2.603);72 base(21.282,21.623),candidato(11.011,11.338).
Baseline ASTextraction4.55/4.26ms es coste del harness, no inicialización producto;
fixture5.33/20.31ms excluida de timings, idéntica dentro de cada par.

Decisión CPU local favorable: coords≈15.3×/25.5×,C2≈1.87×/1.91×.
No atribuir ganancia al kernel escalar compilado: _sparse.py fue el fallback
Python real y no se cambió. RSS incluye allocator/runtime, incluso liberaciones
durante llamada (−104KiB); peak_delta0 significa pico previo no superado.
**No evidencia de ahorro RAM/VRAM** con estos deltas; salida en bytes idéntica.
Carga host no aislada, fullCPU propio concurrente durante medidas certificadas;
pares alternados en el mismo proceso y dispersión registrada. No paso GPU,
entrenamiento completo ni precisión científica medidos.

## Reproducir

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/christian/repositorios/graph2mat/src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/tmp/g2m-python312-full.LHyni1/venv/bin/python \
auditorias_mejoras/bench_sparse_vectorizado.py --points 16
# Repetir --points 72; necesita Git con BASEcc334f0 y dependencias de tests.
```

Assert de igualdad exacta/dtype/dictorder antes de medir; comandos CLI rechazan
points/rounds inválidos y CUDA habilitada. Sin modificar converters del producto.

## Verificación y límites

- Primer REDexit2: colisión de basenames de tests nuevos, corregida renombrando
  solo Torch. REDválido33fallos/28pasan/55warnings/0.20s,exit1; GREEN61/0.08s.
- Doce semillas CSR/oráculo por entrada adicionales, sin tolerancias relajadas.
  Focused165pasan/63warnings/0.38s,exit0 incluye public/sparse/fill/ghost/
  capacity/duplicates/components/Torchgrad/NaN/threshold/Hermiticidad periódica.
- Floor sourcePython3.9.25/Torch2.4.1cpu222pasan/81warnings/4.78s,exit0,
  incluye recursos/ZIP/cacheAtomic/stages previos. No wheel nueva.
- Ruff fuente/tests explícito/all-files, mypy-core y compile39 verdes; benchmark
  Ruff directo AllchecksPassed. mypy primer intento rebindgeneric corregido a
  variable local, no cambio de interfaz. Benchcertified16/72 exit0/stderrvacío.
- Full sourceCPU definitivo **2169pasan,2omitidos,25991warnings,650.27s,exit0**;
  source/tests congelados durante gate. sparse-vector-full-cpu.log, salida leída.
- Smoke benchmark sourcePython3.9 --points4 --rounds1 exit0 con asserts exactos,
  no evaluación comparativa adicional; stderr contiene warnings basales e3nn/
  torch.load, no afirmar vacío como en certified16/72 Python3.12.
- Logs /tmp/g2m-precision-goal.A2Pbgk/sparse-vector-*. GPU compute ajeno
  (pipeline_ui.py,498MiB) presente aunque utilización0; ningún lanzamiento propio.
- A cerrar con commit/task-done. Próximo bloque6.4; goal sigue abierto.
