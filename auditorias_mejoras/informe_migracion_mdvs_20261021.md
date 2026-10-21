# Migración a graph2mat de las funciones del roadmap que vivían en MD_vs_AtomicDisplacement (2026-10-21)

Objetivo (petición del usuario): que todas las funciones del roadmap se puedan usar clonando solo graph2mat.
Plan: [`docs/superpowers/plans/2026-10-21-migracion-mdvs.md`](../docs/superpowers/plans/2026-10-21-migracion-mdvs.md).
En este informe, `$MDVS` es la raíz del repositorio MD_vs_AtomicDisplacement y `$DATA` la de sus resultados.

## Qué se migró

| Roadmap | Origen (`$MDVS/Comparison/scripts/`) | Destino en graph2mat |
|---|---|---|
| 5.1 Δ-learning | `g2m_delta.py` (H0, H−H0, `add_h0`) | `graph2mat.tools.siesta.delta`, CLI `graph2mat siesta delta h0\|label\|add-h0` |
| 8.4 S en inferencia | `predict_model_on_dataset.copy_h0_overlaps` | `graph2mat.tools.siesta.delta.stage_h0_overlap` |
| 4.4 / 4.7 | `g2m_delta.write_spectral_windows` / `write_null_space` | `graph2mat.tools.spectral_targets` |
| Evaluación | `dataset_design_curves_v1.structure_metrics`, `band_dos_metrics`; `g2m_ood_eval.metrics`, `bands`; `g2m_bench_benchmark._bands`, `band_window_metrics`, `species_pair_errors` | `graph2mat.tools.analysis.metrics` |
| 4.3 | `g2m_error_breakdown.py` | `graph2mat.tools.analysis.breakdown` |
| Atribución OOD | `g2m_ood_band_attribution.py` (sin versionar en MD_vs) | `graph2mat.tools.analysis.attribution` |
| 8.2 control | `g2m_bench_kb_control.without_kb_tail` (sin versionar en MD_vs) | `graph2mat.tools.analysis.kb_tail` |
| 5.4 | `g2m_label_noise.py` | `graph2mat.tools.siesta.label_noise` |
| 7.7 (NO_GO) | `g2m_dm_pilot.py` (sin versionar en MD_vs) | `graph2mat.tools.siesta.dm_pilot` |
| SIESTA | ruta fija de la máquina en cada script | `graph2mat.tools.siesta.runner` (`--siesta`, `$SIESTA` o `siesta` del `PATH`) |

Los lanzadores de benchmarks no se migraron: dependen del arnés `dataset_design_curves_v1.py` (2345 líneas), del
registro de muestras, de la cola GPU y de datos no distribuidos. En su lugar,
[`docs/examples/benchmarks/`](../docs/examples/benchmarks/README.md) documenta qué se midió, dónde vive cada
herramienta, la mejor receta de grafeno como config de graph2mat (`delta_recipe.yaml`) y un flujo Δ-learning
ejecutable (`delta_workflow.py`, probado de punta a punta con un SIESTA falso).

## Cambios respecto a los originales (deliberados)

- `combine_hamiltonians` suma todos los componentes de spin; el original solo el primero y escribía siempre sin
  spin (con spin polarizado perdía el canal down). Falla si los spins de H y H0 difieren.
- Pseudopotenciales `*.psf` y `*.psml` en todos los ejecutores (ruido de etiquetas y piloto DM solo copiaban
  `*.psf` del propio directorio).
- `noise_fdf` añade `DM.Tolerance` si el fdf no la define (el original no cambiaba nada en ese caso).
- `aligned_error` devuelve 0 con error exactamente nulo (el original fallaba; nunca ocurre con SIESTA real).
- `band_window_metrics` lanza un error claro si no hay frontera ocupada/vacía (el original indexaba `-1`).
- `stage_h0_overlap` no sobrescribe un TSHS existente salvo `overwrite=True`; nombra el fichero con el
  `SystemLabel` del fdf de la estructura.
- Camino k, malla de la DOS y ficheros de base son parámetros (por defecto, los valores de los benchmarks).

## Verificación de equivalencia

Script fuera del repositorio, solo lectura de `$MDVS` y `$DATA`. Criterio: igualdad exacta o `atol=1e-12`.

| Prueba | Datos | Resultado |
|---|---|---|
| G1 fdf de H0 | 12 RUN.fdf reales (w90, H2O, 6x6) | idénticos |
| G2 labels H−H0 | mismos 12, contra la caché de labels Δ de MD_vs | diferencia máxima 0.0 |
| G3 `_combine` original vs port, ida y vuelta | mismos 12 | 0.0; ida y vuelta ≤ 3.6e-15 eV |
| G5 H0 real con SIESTA | celda w90 de 2 átomos, contra la caché | H y S bit a bit iguales, fdf idéntico |
| G4 proyecciones 4.4/4.7 | 5 muestras de 6x6, H2O y w90 | anchuras iguales, diferencia 0.0 |
| G6 métricas | predicciones sintéticas (referencia + ruido 10 meV con semilla) en 6x6 (2), bicapas registry (2), BaTiO3 y TMD | todas las claves con diferencia 0.0 |
| G6 desglose 4.3, atribución y contrafactuales | 6x6 y bicapas (H0 de la caché) | 0.0 |
| G7 cola KB | BaTiO3 (165504 elementos quitados) y TMD (36450), directo y con H0 | 0.0 |
| G8 fdf de ruido de etiquetas | 3 muestras × 3 variantes | idénticos |
| G9 piloto DM | 1 muestra 6x6, 22 filas, mismo SIESTA falso en ambos lados | diferencia 0.0 |

`git status` y los hashes de los scripts de MD_vs son iguales antes y después; solo se regeneró su caché de
bytecode (`__pycache__`, ignorada por git). No se escribió nada en `$DATA`.

## Tests

33 tests nuevos en `src/graph2mat/tools/{siesta,analysis,tests}/`, sin datos externos ni SIESTA (fixtures
sintéticas y un SIESTA falso en `src/graph2mat/tools/conftest.py`); ruff limpio. Suite CPU completa (`pytest src tests`, sin GPU): 3342 pasan, 36 omitidos y 4 fallos
conocidos (`test_extrapolation` HistorySOAPs: falta el paquete `dscribe`; ya fallaban antes), 1015.60 s.

## Límites

- G6 usa predicciones sintéticas, no las de los modelos entrenados (MD_vs las borraba tras evaluar); prueba la
  equivalencia de las fórmulas, no reproduce cifras históricas.
- G5 comprobó un H0 real; los demás H0 de SIESTA no se recalcularon.
- El piloto 7.7 y el ruido 5.4 no se relanzaron con SIESTA real (decenas de ejecuciones); su lógica se comparó
  con SIESTA sustituido.
- No se migró la predicción en bloques de `predict_model_on_dataset.py` (troceo de MACE/Graph2Mat para memoria):
  no es un punto del roadmap; graph2mat predice con `graph2mat models mace main predict` y `MatrixWriter`.
