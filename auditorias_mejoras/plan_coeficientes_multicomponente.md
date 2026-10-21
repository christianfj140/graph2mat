# Coeficientes multicomponente y readout por interacción — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL:
> `superpowers:executing-plans`; ejecución directa con TDD y registro persistente.

**Goal:** Reparar dos incompatibilidades de 3.14 necesarias para 5.3, sin
redefinir pesos de pérdidas ni declarar terminado el entrenamiento en irreps.

**Architecture:** La proyección conserva las filas de elementos y añade el
eje de componentes al reshape de cada bloque. Los coeficientes siguen siendo
filas por bloque, con componentes concatenadas como ya define MatrixBlock.
Un resolver compartido selecciona el espacio de un ModuleList compatible;
las métricas Lightning usan el mismo espacio para su schema de operaciones.

**Tech Stack:** Python >=3.9, Torch/e3nn/PyG/Lightning, pytest y gates existentes.

**Spec:** `auditorias_mejoras/roadmap_precision_rendimiento.md`, Task 1.

## Global Constraints

- Checkout `<graph2mat>`, `g2m-v2`; no worktree nuevo.
- APIs, Python >=3.9, checkpoints y defaults científicos preservados.
- No cambios de media por operación a media por elemento.
- Sin GPU, autorun/colas, datasets originales, entornos compartidos ni otros repositorios.
- Este sublote no cierra agrupación max/orientación ni evita reconstrucción del forward.
- Los límites del roadmap específico siguen siendo obligatorios.

## Review Focus

- H/S y spin: ninguna componente truncada o permutada al proyectar/reconstruir.
- Referencias 1D para readout multicomponente: error explícito, no broadcasting.
- Vacíos: eje de componentes y dtype conservados, gradientes finitos.
- ModuleList incompatible o vacío: error visible, no primera capa elegida en silencio.
- NaN: una operación sin bloques observados no aumenta su denominador.

### Task 1: proyección/reconstrucción y schema compatibles

**Files:**
- Create: `src/graph2mat/core/data/tests/test_coefficient_compatibility.py`.
- Modify: `src/graph2mat/core/data/metrics.py`.
- Modify: `src/graph2mat/tools/lightning/model.py`.
- Modify: `src/graph2mat/tools/lightning/tests/test_coefficient_epoch_metrics.py`.
- Modify: `src/graph2mat/tools/lightning/tests/test_models.py` (integración MACE real).
- Modify: `docs/data_loading.rst` (contrato público de componentes y coeficientes).
- Modify: `src/graph2mat/core/data/_sparse.py` (coerción del relleno equivalente a Cython).
- Create: `src/graph2mat/core/data/tests/test_sparse_fill_precision.py`.

**Interfaces:** consume `target_coefficients_from_labels(readout, batch,
basis_table, nodes_ref, edges_ref) -> dict` y `coefficients_to_labels(readout,
batch, basis_table, coefficients) -> dict`; conserva sus firmas y los tensores
por operación de forma `(n_blocks, n_components * dim_irrep)`.
Produce los mismos diccionarios, ahora también para componentes y ModuleList
compatibles; `_coefficient_epoch_terms` conserva nombres y ponderación.

- [x] Step 1: escribir tests con readouts e3nn reales. Caso literal s:
  nodos A/B/A `[[1,2],[7,8],[3,4]]`, arista A-B `[[5,6]]`;
  coeficientes A `[[1,2],[3,4]]`, B `[[7,8]]`, A-B `[[5,6]]`.
  Añadir s/p/d/f, vacíos, NaN, errores de componentes, ModuleList y épocas.
- [x] Step 2: ejecutar los tests nuevos y registrar fallos de producto RED.
  Run: comando CPU de Task 0, sustituyendo la selección por ambos ficheros
  de tests listados aquí. Expected: incompatibilidades reproducidas, no import errors.
- [x] Step 3: parche mínimo en los consumidores compartidos. Usar
  `block.reshape(*shape, n_components).unsqueeze(0)` para varias componentes;
  no `reshape(-1)` de etiquetas que mezcla componentes con elementos.
  Resolver readouts solo si su metadata/contrato de base es compatible.
- [x] Step 4: ejecutar los tests del sublote y familias existentes de coeficientes.
  Expected: exit 0 y valores/gradientes de referencia preservados.
- [x] Step 5: suite CPU completa, Ruff, mypy y diff --check; registrar todos
  los fallos/omisiones. Expected: sin regresiones ni errores propios de tipos/lint.
- [x] Step 6: commit explícito y registro de comandos/resultados; marcar solo
  este sublote como verificado. Ganancias GPU/precisión siguen sin medir.

## Pre-flight

El productor de coeficientes ya concatena componentes en MatrixBlock;
la proyección pública los rechaza y la selección de aristas cuenta escalares
en lugar de filas. Lightning y pérdidas deben compartir el resolver de
readout para no producir schemas diferentes. Se prueba cada interfaz en
los ficheros anteriores. El objetivo más amplio continúa pendiente.

Ruling durante el gate completo: la ruta Python de CSR trunca decimales con
el relleno entero por defecto, a diferencia de la extensión Cython. Se incluye
una corrección mínima de precisión de 6.2 necesaria para comparar ambas rutas:
coerción a float ya impuesta por Cython. Reproducción independiente con etiquetas
.125/-.375 y fill 0/1 antes del parche; no se cambian métricas ni tolerancias.
