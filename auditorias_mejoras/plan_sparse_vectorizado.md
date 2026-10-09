# Sparse medido: coordenadas COO y CSR multicomponente — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL `superpowers:executing-plans`.
> Ejecución directa/TDD, mismo checkout; revisión independiente solo al final
> del goal completo, no agentes por tarea.

**Goal:** Retirar bucles Python por bloque en coordenadas COO y por elemento en
CSR multicomponente, conservando orden/valores y corrigiendo relleno entero C2.

**Architecture:** En sparse.py, coordenadas por offsets/repeat/arange con memoria
O(nnz de bloques), no matriz global. CSR C2 agrupa entradas por bloque y asigna
arrays; conserva primer encuentro del bloque, último valor de coordenadas
duplicadas y el corte al primer col negativo por fila (capacidad sisl sin usar).
Mantener dtype float64 legacy para datos reales y conservar datos complejos C2
sin descartar imaginaria. Ninguna API, flag, export o dependencia nuevos.

**Tech Stack:** Python>=3.9, NumPy/SciPy/sisl/Torch existentes.

**Spec:** roadmap_precision_rendimiento.md Task2/6.2. No vectorizar kernel escalar
Cython ni cambiar búsqueda de vecinos: perfil aquí solo demuestra cuello Python.

## Global Constraints

- Solo Graph2Mat/g2m-v2/mismo checkout; no fonones, autorun, colas ni datasets.
- Preservar APIs/checkpoints/defaults, NaN/fill y componentes; no densificación global.
- CPU propio, entornos aislados. No GPU mientras haya procesos compute ajenos.
- Ganancia local CPU no prueba velocidad de entrenamiento ni VRAM/precisión.
- Full source CPU final tras congelar source/tests y floor3.9 reales.

## Review Focus

- Rectángulos heterogéneos: reverse COO conserva orden de valores original, no
  row-major del bloque transpuesto. Imagen0→0 y otras isc→n_supercells-isc.
- Ghost cero orbitales, vacíos, edge_index no contiguo, NaN/threshold y Torchgrad.
- CSR no finalizado: col negativo corta toda esa fila incluso si luego hay col válido.
- Duplicados último gana, orden dict primer encuentro, fill0/1/float/NaN y dtype.
- Public csr_to_block_dict con selección/reordenación de componentes y input intacto.

## Diseño / perfil previo

Mapa: _blockmatrix_coo_coords alimenta _nodes_and_edges_to_coo (NumPy/Torch) y
nodes_and_edges_to_sparse_orbital C2. csr_to_block_dict usa helperC2 para >1
componente y _sparse._csr_to_block_dict para escalar. configuration.py pasa0.0
para H y NaN para DM/EDM: el bug fillint público NO implica corrupción activa.
No cambios necesarios en callers/exports/config; tests de roundtrips/lifecycle
existentes son gates, no nuevos componentes.

Perfil propio s/p/d/f periódico, 16/72 puntos, 48/216 aristas únicas,
11184/50328nnz y shape160x480/720x2160: coords1.196/6.146ms,
CSRscalar2.651/12.329ms y CSRC2 4.752/21.562ms. Python fallback real, no Cython
compilado. cProfile atribuye C2 casi entero al bucle por entrada; coords a mgrid
por bloque/listas Python. Primer intento de perfil tuvo fixture edge_labels
de ambas direcciones con unique_edges; corregido sin parche de producto.
Logs sparse-profile-valid.log en /tmp/g2m-precision-goal.A2Pbgk/.

Alternativas: solo dtype arregla precisión pero deja cuello medido; densificar
global viola contrato; reescribir Cython duplica optimización sin evidencia de
wheel. Elegimos los dos helpers NumPy medidos. Subproyecto acotado autorizado por
goal explícito; no pausa rutinaria de brainstorming/aprobación.

### Task 1: equivalencia y vectorización de los dos helpers

**Files:** Modify src/graph2mat/core/data/sparse.py;
Create src/graph2mat/core/data/tests/test_sparse_vectorization.py;
Create src/graph2mat/bindings/torch/data/tests/test_torch_sparse_vectorization.py;
Modify tests/test_sparse_fill_precision.py; Create auditorias_mejoras/
bench_sparse_vectorizado.py e informe_sparse_vectorizado.md; docs/roadmaps.

**Interfaces:** Consume arrays sparse/ptr/cols/orbitals/edges existentes; devuelve
mismo dict y coordenadas, sin alterar conversiones registradas ni kernel escalar.

- [x] Step 1: RED literal C2 decimales con fill entero y complejos. Oráculos
  independientes COO por nested loops y CSR por entrada, para direcciones,
  imágenes, ghost/vacíos, capacidad/duplicados/orden. Guard mgrid prohibido
  certifica retirada del constructor por bloque, no umbral temporal flaky.
- [x] Step 2: vectorizar COO con offsets/repeat; reverse swap de coordenadas
  originales y offsets de imagen. GREEN igualdad exacta y dtype índices enteros.
- [x] Step 3: vectorizar CSR C2 con grouping, sentinel por fila y selección
  explícita del último duplicado; dtype compatible con fillfloat64, imaginaria
  conservada. GREEN public/default/components sin mutar inputs.
- [x] Step 4: TorchCPU float32/64, NaN/threshold, ghost y empty: COOvalues y
  gradientes contra literal; public sisl raw/polarized y roundtrips existentes.
- [x] Step 5: benchmark base local de los helpers anteriores vs candidato,
  misma fixture, warmup3,12rondas alternadas y memoria/RSS fresh procesos.
  Primaria tiempo coords y CSRC2, tolerancia no regresión10%; valores exactos.
  Registrar dispersión/dtype/source/kernelPython, sin extrapolar a wheel/GPU.
- [x] Step 6: gates focused, floor3.9, full sourceCPU con source/tests congelados,
  Ruff/mypy-core/compile39/diff; docs/report y commit explícito + task-done.
  Expected exit0 leído y límites abiertos honestos; después batching6.4.
