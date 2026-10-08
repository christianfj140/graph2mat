# Orientación y observación de coeficientes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:executing-plans`;
> ejecución directa con TDD, sin agentes por tarea.

**Goal:** Corregir la asociación de coeficientes con bloques físicos agrupados
y permitir su reconstrucción recortada, sin inventar targets en elementos ausentes.

**Architecture:** La selección de módulos conserva el signo agrupado cuando
la matriz es dirigida. La proyección/reconstrucción aplica el mismo cambio de
orientación que el forward, pero los tests usan bloques físicos literales y
el forward real como oráculos independientes. La reconstrucción `max` recorta
por los filtros existentes; su proyección directa se rechaza si hay padding,
porque no determina de forma única todos los coeficientes agrupados.

**Tech Stack:** Python >=3.9, Torch/e3nn, pytest y gates existentes.

**Spec:** `auditorias_mejoras/roadmap_precision_rendimiento.md`, Task 1.

## Global Constraints

- Solo checkout `/home/christian/repositorios/graph2mat`, `g2m-v2`, sin worktree.
- Firmas, checkpoints, defaults y media por operación preservados.
- CPU aislada: GPU ocupada por trabajos ajenos, sin modificar procesos ni colas.
- No datasets originales, entornos compartidos, autorun ni otros repositorios.
- Este sublote no cierra 5.3 ni su pérdida MSE enmascarada para `max`.

## Review Focus

- Rectángulos s/p, sp/s y d/f: no confundir reshape con transposición.
- Dos direcciones no simétricas: módulos y filas independientes, no `abs`.
- Componentes: transponer solo orbitales, conservar H/S y el orden de bloques.
- Fantasmas sin base: no exigir módulos/coeficientes de bloques de tamaño cero.
- Padding: reconstrucción recortada y gradientes físicos; error explícito al
  intentar una pérdida en coeficientes completos no observados.

### Task 1: orientación, vacíos y reconstrucción observada

**Files:**
- Create: `src/graph2mat/core/data/tests/test_coefficient_orientation.py`.
- Modify: `src/graph2mat/core/data/metrics.py`.
- Modify: `docs/data_loading.rst`.
- Modify: `auditorias_mejoras/roadmap_precision_rendimiento.md` y `roadmap.md`.

**Interfaces:** consume `target_coefficients_from_labels(readout, batch,
basis_table, nodes_ref, edges_ref) -> dict`, `coefficients_to_labels(readout,
batch, basis_table, coefficients) -> dict` y `readout.forward(...,
return_coefficients=True)`; conserva firmas y claves por operación. Produce
coeficientes correctamente orientados para `point_type`/`basis_shape` y labels
recortados para `max`. La pérdida física enmascarada de 5.3 sigue abierta.

- [x] Step 1: escribir RED con referencias literales: nodos diagonales y
  bloques A→B `arange(1, m*n*c+1)/17`, B→A con valores distintos y forma `(n,m,c)`.
  Para `basis_shape`, los pares s/p, sp/s y d/f invierten el orden agrupado;
  proyectar el bloque transpuesto solo si el readout simétrico necesita esa
  orientación. Comparar también salida/coefs/gradientes del forward real.
- [x] Step 2: ejecutar selección nueva con el comando CPU de Task 0 del spec,
  sustituyendo los ficheros por `test_coefficient_orientation.py`.
  Expected: fallos de asociación o dimensiones, no import errors.
- [x] Step 3: parche mínimo compartido: elegir `op_edge_type == mapped_type`
  (o `abs(mapped_type)` si symmetric), invertir dimensiones de aristas físicas
  negativas, transponer ejes orbitales para orientación agrupada negativa,
  omitir módulos None. Reconstrucción `max`: usar filtros físicos ya existentes
  y conteo de filas físicas, nunca el tamaño padded. Si sus filtros no son todos
  True, proyección directa: ValueError que solicite pérdida en matrices/máscara.
- [x] Step 4: nuevos tests más coeficientes/componentes/épocas/modelos/pérdidas.
  Expected: exit 0; valores y gradientes 1e-12 en readouts construidos float64.
- [x] Step 5: suite source CPU completa, Ruff/mypy y diff --check. Documentar
  el contrato y el límite de targets padded. Expected: sin regresiones nuevas.
- [ ] Step 6: commit local explícito y task-done con la selección CPU final;
  registrar comandos/resultados. No atribuir ganancia GPU a una reparación.

## Pre-flight

No hay interfaces entre tareas (una tarea). Callers leídos: pérdidas/composite,
Lightning y ajuste output_scaleshift. Este último también recibe el error
explícito si pide targets padded completos; un ajuste enmascarado necesita
definición separada. El siguiente subproyecto es 5.3, no otro retoque cosmético.

Ruling: se mantiene la ejecución directa autorizada por el goal sin nuevas
preguntas rutinarias; el límite de padding se hace explícito, no se da por
implementada la equivalencia enmascarada. Coste si equivocado: el caller debe
seleccionar la pérdida física mientras se completa 5.3.
