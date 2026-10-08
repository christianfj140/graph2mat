# Kernel de coeficientes sin bloques — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:executing-plans`;
> ejecución directa con TDD, revisión independiente al cierre del goal amplio.

**Goal:** Quitar la reconstrucción densa de `forward_coefficients` para bloques
simetrizados estándar; no presentar todavía el entrenamiento 5.3 como completo.

**Architecture:** El RTP conocido `ij` de una base consigo misma se construye
por pares de copias de irreps. Su transposición intercambia esos pares mediante
una permutación con signo de los coeficientes. Se deriva de los pequeños RTP
locales ya disponibles, sin crear una matriz K×K. `ij=ji` usa identidad. Los
índices/signos son buffers no persistentes; fórmulas ajenas conservan la ruta
actual. El forward de matrices y `forward_with_coefficients` no cambian.

**Tech Stack:** Python >=3.9, Torch/e3nn, pytest y gates existentes.

**Spec:** `roadmap_precision_rendimiento.md`, Task 1. Precondición verificada:
subproyecto `plan_coeficientes_orientacion.md`.

## Global Constraints

- Solo `/home/christian/repositorios/graph2mat`, `g2m-v2`, mismo checkout.
- APIs/checkpoints/defaults preservados; ningún buffer persistente nuevo.
- CPU aislada; no GPU ocupada, procesos, colas, datos ni entornos ajenos.
- No matriz K×K adicional ni cambios de capacidad/operación del readout.
- No redefinir la media de coeficientes como MSE de matrices.
- Targets CPU, route Graph2Mat/MACE/Lightning y evaluación GPU siguen abiertas.

## Review Focus

- Radiales repetidas y s/p/d/f: signo y copia correctos en el intercambio.
- Varias componentes: permutar dentro de cada componente, no entre H/S.
- Inputs todos pares frente a global/no-pares: una o dos llamadas, como antes.
- Escala/desplazamiento y vacíos: aplicar transformación al output completo.
- Checkpoints/dtype/fórmulas no estándar: strict loading y fallback correcto.

### Task 1: transposición y coeficientes simetrizados

**Files:**
- Modify: `src/graph2mat/bindings/e3nn/irreps_tools.py`.
- Modify: `src/graph2mat/bindings/e3nn/modules/matrixblock.py`.
- Create: `src/graph2mat/bindings/e3nn/modules/tests/test_coefficient_only_block.py`.
- Modify: `auditorias_mejoras/roadmap_precision_rendimiento.md`.

**Interfaces:** consume RTP local (posiciones orbitales, Q local y orden de
irreps) y `_compute_coefficients(*args, **kwargs)`. Produce metadatos opcionales
`transpose_index`/`transpose_sign` y `forward_coefficients` con firma/forma
inalteradas. La prueba de equivalencia se hace reconstruyendo fuera del tramo
certificado; los targets/loss futuros consumen el mismo orden de coeficientes.

- [ ] Step 1: RED con bloque real: calcular primero `legacy = block(...)`,
  reemplazar `coefficients_to_block` en el test por una función que falla,
  pedir `forward_coefficients` y contrastar `einsum(c,Q)` con legacy y sus
  gradientes. Expectativas independientes: transposición del bloque legacy,
  tolerancias 1e-12 float64 y 5e-6 float32. Casos s/p/sp/s-p-d-f, 1/2
  componentes, `ij`/`ij=ji`, inputs pares/no-pares, escala y vacío.
- [ ] Step 2: ejecutar ese fichero con el comando CPU de Task 0.
  Expected: falla por la reconstrucción del camino actual, no imports.
- [ ] Step 3: mapa O(K): registrar slices ordenados por `(row,end_row,col,
  end_col,dim_output)`; buscar el par invertido, obtener signo de su producto
  local con Q transpuesto y almacenar índices consecutivos. Para `ij=ji`,
  identidad. Registrar buffers persistent=False solo si symm_transpose.
  Apilar ambas direcciones si todos los inputs son pares y devolver
  `(c_forward + c_backward[...,index]*sign)/2`; conservar fallback actual.
- [ ] Step 4: GREEN y pruebas existentes de RTP/readouts/orientación/modelos;
  dtype cast y strict state_dict conservan matrices. Expected: exit 0.
- [ ] Step 5: suite CPU completa, Ruff/mypy, sintaxis Python3.9 y diff --check.
  Expected: sin regresiones; reportar avisos/omisiones, no ocultarlos.
- [ ] Step 6: commit local y task-done real; registrar implementación y límite
  pendiente (el entrenamiento integrado aún reconstruye). No ganancia GPU.

## Pre-flight y evidencia de diseño

Una tarea; interfaces no compartidas. Exploración CPU independiente con Q:
permutación con signo para 1,16,256,1024 coeficientes de las bases s, sp, spdf
y 2s2p2d2f. Residuo transposición: 0 float32, <=1.56e-15 float64.
Q no es ortogonal bit a bit: error Gram <=1.2e-7 float32 y <=3.11e-15 float64.
No transformar esa observación en una promesa de pérdidas numéricamente
idénticas tras un cast float32→float64. Verificar matrices y gradientes;
la certificación de pérdida 5.3 deberá considerar el Q/dtype efectivo.

Ruling: mapas locales O(K), no operador denso K×K — evita trasladar coste a
VRAM/init. Coste si falla la identidad local: conservar fallback denso, sin
declarar la ruta certificada para esa fórmula/base.
