# Auditoría de pérdidas, métricas y física — 2026-10-21

## Alcance y resultado

Se revisaron las fórmulas y sus consumidores en `core/data/metrics.py`, `physical.py`, `spin_orbit.py`, `multitarget.py` y `bindings/torch/multitarget.py`, con HEAD `a66b46cb0955ca344693d50737c5e9952fc257fc`. Se contrastaron el diff frente a main (`a9b011c`) y la introducción de los cambios posteriores a `origin/g2m-v2` (`a5e81345`). No se modificó código productivo ni tests. Se confirmaron tres defectos: uno en el tratamiento físico del spin collinear, otro en la combinación de la proyección casi nula de S con la pérdida espectral, y otro en la detección de parejas con celdas no reducidas.

Las reproducciones usaron exclusivamente CPU y matrices sintéticas propias o fixtures analíticos del mismo HEAD. Se ejecutaron mediante `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/audit-python`, que importa la copia aislada con extensiones compiladas desde este HEAD. No se ejecutaron reproducciones con las extensiones editables obsoletas del entorno original. No se usaron GPU, datos externos ni campañas.

## M1 — P1: el gauge independiente por spin elimina una separación magnética física

**Localización:** `src/graph2mat/core/data/metrics.py:1940` y `:1944`; consumidor multitarget en `src/graph2mat/bindings/torch/multitarget.py:114`.

**Escenario real:** entrenamiento collinear con dos componentes H, usando `gauge_invariant_mse`, tanto en la ruta legacy con H↑/H↓/S como en multitarget con `matrix_target_spin=2`. La función estima un desplazamiento distinto para cada componente H. Una predicción `H↑_pred = H↑_ref + b S`, `H↓_pred = H↓_ref - b S` recibe pérdida cero y gradiente cero. Ese cambio introduce una separación de spin de `2b`; no es un desplazamiento del cero de energía común del sistema.

**Oráculo:** el problema generalizado por spin es `Hσ c = εσ S c`. Añadir `+b S` a un spin y `-b S` al otro desplaza sus niveles en direcciones opuestas. La transformación de gauge físicamente redundante es `Hσ → Hσ + c S` con el mismo `c` para ambos spins. Minimizar el error bajo ese único `c` conserva la diferencia magnética. En el ejemplo mínimo de un orbital, referencia `[0,0]`, predicción `[1,-1]`, `S=1`, la MSE mínima bajo gauge común es `1 eV²`; la implementación devuelve `0`.

**Reproducción:**

```bash
/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/audit-python /tmp/graph2mat-codex-audit-20261021-1n6tqn2d/metrics/repro_gauge.py
/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/audit-python /tmp/graph2mat-codex-audit-20261021-1n6tqn2d/metrics/repro_spin_multitarget.py
```

La segunda reproducción atraviesa la preparación multitarget real, `TypeLayoutCollate` y `MultitargetMatrixLoss`, con el fixture no ortogonal `analytic_matrices(spin=2)`. Salida:

```text
reference_eigenvalues_both_spins_eV = [0.16666666666666666, 1.0]
pred_up_eV = [1.1666666666666667, 2.0]
pred_down_eV = [-0.8333333333333334, 5.551115123125783e-17]
pred_spin_splitting_eV = [2.0, 2.0]
multitarget_H_gauge_mse = 0.0 max_abs_gradient = 0.0
```

**Introducción:** `7b97b7a595dfcf74d8c48693cb0476d1b17a7bd8` ya calcula `shift` con forma `(estructuras, n_h)` y sustrae cada canal por separado. Está dentro de los 267 commits. `65d2253` hace alcanzable la misma fórmula desde el entrenamiento multitarget spin2; no causa el defecto original. No es heredado de main ni del HEAD de origin indicado.

**Consecuencia:** pérdida, selección por gauge-H y entrenamiento pueden certificar como exacto un Hamiltoniano con separación de spin errónea. Los gradientes no corrigen esa dirección. También afecta al término energético de H cuando recibe estos deltas alineados. Severidad P1 por anular una señal física del objetivo admitido; no se afirma que los entrenamientos spin1 registrados estén afectados.

**Propuesta mínima:** para dos canales collineares, ajustar un único escalar por estructura usando ambos spins en numerador y denominador, y aplicarlo a ambos. Si se conserva alguna opción de alineación por canal como diagnóstico, debe distinguirse explícitamente del gauge físico. Añadir el oráculo `(+bS,-bS)` y conservar el oráculo `(+cS,+cS)`.

## M2 — P2: la proyección casi nula con floor penaliza un cambio puro de gauge

**Localización:** `src/graph2mat/core/data/metrics.py:1003` y `:1004`; llamada con deltas sin alinear en `:1966`; productor en `src/graph2mat/tools/spectral_targets.py:52`.

**Escenario real:** uso documentado de `spectral_window="spectral_null.npz"` y `gauge_invariant_mse(spectral_weight>0)`, con al menos dos direcciones seleccionadas de S que tienen normas S distintas por el floor. `null_space_columns` divide cada columna por `sqrt(max(λ,floor))`. Para `λ<floor`, se cumple `CᵀSC = diag(λ/floor)`, no `I`. `_spectral_window_loss` sigue quitando `mean(diag(CᵀΔHC)) I`, suponiendo `CᵀSC=I`. Por ello `ΔH=cS` deja residuo y gradiente distintos de cero.

**Reproducción:**

```bash
/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/audit-python /tmp/graph2mat-codex-audit-20261021-1n6tqn2d/metrics/repro_spectral_pipeline.py
```

El script escribe únicamente en su directorio temporal un NetCDF sintético, llama al productor real `null_space_columns` y después a la API pública de la pérdida. El S es simétrico y positivo, tiene diagonal `[1,1,1,1]` y autovalores `[0.0002,0.005,1.5,2.4948]`. Se usan `threshold=0.01`, `floor=0.001`, `H_ref=0.5S`, `H_pred=H_ref+S`. Resultado:

```text
C.T S C = [[0.2, 0.0], [0.0, 1.0]]
pure_gauge_base_mse = 3.4666738998970245e-33
pure_gauge_total_mse = 0.08000000000001876
spectral_rmse = 0.2828427124746522
max_abs_gradient = 60.000000000007816
```

**Esperado:** pérdida y gradiente cero, salvo redondeo, pues todos los autovalores físicos cambian en el mismo `1 eV`. **Real:** término espectral `0.08 eV²` y gradiente grande. Analíticamente, la matriz proyectada es `diag(0.2,1)`, su media diagonal `0.6`, y el residuo `diag(-0.4,0.4)` produce `(0.16+0.16)/4=0.08`.

**Introducción:** `496d792e5ed2622e80b64374d83d0afbc26c9a9e` introduce la pérdida asumiendo columnas S-ortonormales, lo cual es válido para sus eigenvectores de ventana originales. La combinación incompatible queda disponible íntegramente en este repositorio con `7d417974daeb30490175919f9669defc1174f772`, que incorpora el productor casi nulo y documenta su consumo por la misma pérdida. Ambos commits están dentro de los 267. La receta equivalente anterior fuera del repositorio no se auditó ni se usa para atribuir resultados de campañas previas.

**Consecuencia:** entrenamiento con la proyección casi nula puede perseguir un cero absoluto de energía que el objetivo gauge pretende eliminar. Este resultado no demuestra por sí solo que una campaña registrada haya sido afectada: harían falta sus autovalores de S, proyecciones y predicciones concretas.

**Propuesta mínima:** entregar a la proyección espectral el delta que ya se alineó con S, o eliminar el gauge proyectado usando `CᵀSC` real en lugar de `I`. Mantener el floor existente: quitarlo cambiaría la regularización física y no es necesario para corregir esta incompatibilidad. Añadir el caso anterior con autovalores a ambos lados del floor.

## M3 — P2: frac-round pierde parejas válidas en una celda no reducida

**Localización:** `src/graph2mat/core/data/metrics.py:1798` y `:1799`; efecto en `pair_difference_mse` a partir de `:1861`.

**Escenario real:** la misma red periódica se representa con una base de celda muy sesgada, sin reducir. Redondear por separado las coordenadas fraccionarias no calcula en general la imagen mínima cartesiana. Una pareja de desplazamientos `±0.02 Å` puede dejar de reconocerse por un mero cambio entero de base de la misma red.

**Reproducción:**

```bash
/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/audit-python /tmp/graph2mat-codex-audit-20261021-1n6tqn2d/metrics/repro_pair_cell.py
```

Se comparan `cell=diag(1,1,10)` y `cell'=M cell`, con `M=[[100,1,0],[99,1,0],[0,0,1]]` y `det(M)=1`. Ambas generan exactamente la misma red; las posiciones cartesianas y las etiquetas son idénticas. La distancia mínima física es `0.04 Å`. Con umbral `0.05 Å`:

```text
orthogonal pairs = [(0, 1)] loss = 5.0
same_lattice_skew pairs = [] loss = 1.0
expected_pairs=1 error = found 0 +/- pairs in the batch, expected 1
```

**Esperado:** ambas representaciones reconocen `[(0,1)]` y dan pérdida `5`. **Real:** desaparece el término de diferencias, o el guardrail aborta el batch. En la celda sesgada, la separación tiene fracciones `(-3.96,4,0)`; redondearlas resta una traslación cartesiana incorrecta y produce aproximadamente `4 Å` en lugar de `0.04 Å`.

**Introducción:** `6c44eac5def9c70f21d10faa722f3dbd7dab6f87`, dentro de los 267 commits. No es heredado de main/origin indicados.

**Consecuencia y límite:** parejas válidas no entrenan el término antisimétrico en celdas no reducidas; con `expected_pairs` se interrumpe el entrenamiento. La reproducción usa deliberadamente una celda muy sesgada. No demuestra un fallo para las celdas ortogonales habituales ni para desplazamientos pequeños en las celdas hexagonales convencionales.

**Propuesta mínima:** usar una operación de imagen mínima válida para celdas generales, o reducir la base de la celda antes de buscar la traslación más corta. Añadir invariancia ante el cambio unimodular de base anterior.

## Archivos, opciones y contratos revisados sin otro hallazgo confirmado

- `core/data/metrics.py`: lectura completa. Se comprobaron selección por fila NaN, ramas vacías, ponderación de edges, denominadores por elemento/componente/bloque/estructura/dominio, MAPE y ramas absolutas, Huber, umbrales duros y sigmoidales, proyección/reconstrucción de coeficientes, gauge, Γ y error de densidad. Las métricas elementales no convierten eV a meV; las reproducciones conservan eV/eV². La pérdida de diferencias conserva unidades de H, no representa directamente una derivada dividida por el stencil. No se elevó su normalización histórica como defecto nuevo.
- `core/data/physical.py`: lectura completa de preparación, validación de referencias y contracciones. Se comprobaron geometría/base/imágenes/spin, finitud y máscara Hermitiana, positividad de S sin clipping, soporte observado frente a cero explícito, partner opuesto en imágenes, ocupación sin factor dos añadido, poblaciones completas frente a soporte truncado, baseline DM, residuo `HD-ES` en gauge común y normalización por estructura/átomo. Las pérdidas usan partners de referencia, no predicciones conjuntas de todas las matrices; su significado es el contrato referenciado documentado. No se encontró otro error confirmado en estas fórmulas.
- `core/data/multitarget.py`: lectura completa. Descriptor y orden H/S/DM/EDM, spin1/2, observaciones por componente, NaN en DM/EDM ausentes, soporte Hermitiano, máscaras restrictivas, vistas sin copiar gradientes, preparación física por target y reconstrucción con S predicha explícita. Se comprobó el uso deliberado de `threshold=None` para S.
- `bindings/torch/multitarget.py`: lectura completa. Pesos exactos por target, términos de peso cero sin evaluación, denominador por observación y edge_weight, preservación del otro spin cuando un componente es desconocido, requisitos por estructura, física con denominador independiente y propagación de gradientes. El hallazgo M1 alcanza esta ruta por reutilización de la pérdida gauge.
- `core/data/spin_orbit.py`: lectura completa. Mapeos SIESTA↔canales TRS y ocho canales completos, signos de transposición, paridad temporal, comprobación de finitud/residual y adjunto explícito de S. El consumidor `bindings/e3nn/modules/spin_orbit.py` rechaza salidas de coeficientes no certificadas; no se reporta como fallo una combinación expresamente no soportada.
- Consumidores leídos parcialmente: `tools/lightning/data.py` (`TypeLayoutCollate`, campos físicos, layout y máscaras), `tools/lightning/model.py` (conteos RMSE, reducción epoch de gauge/parejas/física/multitarget, selección y preflight), `tools/spectral_targets.py` completo. Se revisaron llamadas de conversión en `core/data/processing.py` y el contrato de métricas en `docs/data_loading.rst`.

## Tests y evidencia

Se inspeccionaron los tests relevantes de `test_metrics.py`, `test_metrics_valid_labels.py`, `test_per_structure_losses.py`, `test_physical_references.py`, `test_physical_losses.py`, `test_multitarget.py`, `test_multitarget_reconstruction.py`, `bindings/torch/tests/test_multitarget_loss.py`, y los de Lightning `test_epoch_special_metrics.py`, `test_epoch_metrics.py`, `test_gauge_epoch_metrics.py`, `test_pair_epoch_metrics.py`. Se contrastaron los oráculos y opciones de los tests de conversión spin-orbit y los puntos de integración multitarget; no se afirma lectura exhaustiva de todos los tests de Lightning.

Comprobación focal ejecutada en la copia aislada: `test_physical_losses.py`, `test_multitarget_loss.py`, `test_spin_orbit.py`, `test_spin_orbit_full.py`: **159 passed, 2 skipped, 115 warnings, 2.68 s, salida 0**. Log: `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/metrics/focal.log`. Las dos omisiones corresponden a fixtures SIESTA externos ausentes. La primera ejecución focal terminó sin que se conservara su salida; se repitió para registrar este resultado verificable. No se ejecutó suite general desde esta subárea.

Los tests existentes de gauge spin2 comprueban el promedio de shifts independientes, pero no contrastan su efecto sobre la separación física de spins. Los tests espectrales usan ventanas S-ortonormales y no el floor del productor casi nulo. El test de imagen mínima usa celda ortogonal. Que esos tests pasen no cubre los tres contraejemplos reproducidos.

Se descartó la pista previa que proponía excluir estructuras totalmente NaN del promedio por estructura/dominio: el comportamiento cero más denominador total está fijado explícitamente por `test_size_domain_epoch_means_and_structure_counts` y los tests de grafos vacíos. Se registra como semántica actual, no como hallazgo confirmado. El denominador `q0` de `normalized_density_error` también está explícitamente documentado; no se reinterpretó como conteo de electrones del estado cargado.

## Límites científicos

Los tres hallazgos tienen oráculos algebraicos y reproducción CPU; no dependen de precisión ML ni de observar una campaña. Las referencias sintéticas comparten geometría, base, orden orbital, soporte y unidades dentro de cada comparación. En M1 se usa S de referencia deliberadamente para aislar el fallo del gauge; la comparación espectral es condicional a esa S y demuestra la separación exacta de spins. En M2 la S sintética es positiva y Hermitiana; no se recorta su espectro para evaluar H. En M3 el control es la identidad exacta de la red bajo una matriz unimodular. No se midieron OOD, semillas, velocidad, VRAM ni repercusión cuantitativa en resultados previos. No se certifica ausencia global de defectos.
