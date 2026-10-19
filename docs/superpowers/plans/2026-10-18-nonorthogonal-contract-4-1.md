# Nonorthogonal Contract 4.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Ejecución Native elegida por el usuario; sin worktrees ni commits.

**Goal:** Certificar la representación, reconstrucción y serialización no ortogonal H+S existentes mediante observables independientes en CPU.

**Architecture:** Ampliar tests vecinos y documentación, sin un lector, writer ni API nuevos. El recorrido existente sigue siendo configuración → etiquetas → procesador → matriz sisl → formato físico. Solo corregir producto ante una carencia reproducible dentro del contrato aprobado; una nueva API requiere ampliar y aprobar el diseño.

**Tech Stack:** Python >=3.9, NumPy/SciPy, sisl 0.16.x, Torch/PyG/Lightning, pytest y extras HTTP ya instalados. Sin dependencias nuevas ni GPU.

**Spec:** `docs/superpowers/specs/2026-10-18-nonorthogonal-contract-4-1-design.md`, aprobado por el usuario el 2026-10-18.

## Global Constraints

- Solo `/home/christian/repositorios/graph2mat`, rama `g2m-v2`, checkout actual y cambios previos preservados.
- CPU; sin entrenamientos científicos, GPU, cola/autorun, commits, ramas, worktrees ni instalaciones.
- Sin cambios de defaults, losses, checkpoints, datos o artefactos de campañas congeladas; fonones y otros proyectos excluidos.
- Matrices reales no polarizadas/colineales; raw/SOC no adquiere una semántica física nueva.
- Espacio orbital completo: sin cutoff de S, regularización, reparación ni simetrización silenciosa de entradas.
- Contrato sobre el soporte representado por el grafo; no prometer recuperar coeficientes fuera del cutoff.
- S de etiquetas y S de predicciones se distinguen; H-only no obtiene S por inferencia implícita.
- H y niveles en eV; S adimensional. H0 no se construye en el procesador genérico.
- `.nc` leído por sisl devuelve H-Ef*S; `.TSHS` del servidor exige Ef=0; MatrixWriter/HSX tiene su contrato propio.
- Comparar autovalores, no autovectores arbitrarios en degeneraciones. Sin promesas de precisión ML, velocidad ni VRAM.

## Review Focus

- Predicción S distinta de la etiqueta: reconstruir la S predicha y preservar la referencia (Task 1).
- Batch con tamaños diferentes: no mezclar estructuras ni perder componentes al separar predicciones (Task 1).
- S singular con diagonal positiva: fallo del control espectral sin recorte; no atribuir al writer una validación SPD inexistente (Task 1).
- Ef no nulo y canales colineales: desplazar ambos espectros exactamente una vez en `.nc`, rechazar `.TSHS` incompatible (Task 2).
- Referencia residual con S explícita: restaurar H y S una vez, sin mutar caller ni confundir H0 con baseline aislada (Task 2).

## Archivos y funciones existentes

- `src/graph2mat/core/data/tests/test_periodic_hermiticity_oracles.py`: campos independientes y Bloch sums; ampliar sin extraer un framework.
- `src/graph2mat/core/data/tests/test_processing.py`: contratos NumPy/Torch y no mutación; añadir control analítico pequeño.
- `src/graph2mat/tools/lightning/tests/test_matrix_components.py`: componentes, batches y MatrixWriter/HSX.
- `src/graph2mat/tools/server/tests/test_physical_serialization.py`: `_model`, `_request`, HTTP, formatos y referencia residual.
- `docs/data_loading.rst`, `docs/server.rst`: contrato y límites junto a las APIs actuales.
- `auditorias_mejoras/informe_reconstruccion_no_ortogonal_20261018.md`: evidencia de cierre, comandos y requisitos certificados.
- `auditorias_mejoras/roadmap.md`, `auditorias_mejoras/traspaso_agente_20261015.md`: estado final sin promover defaults.

No nueva export pública, config ni fixture científica. Cambios productivos, si hicieran falta, se deciden por causa reproducida, no por adelantado.

## Entorno reproducible y baseline

Usar el intérprete preexistente `/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python` como herramienta, sin entrar en ese proyecto ni modificar su entorno.

Los extras se reutilizan desde sus directorios existentes, para este proceso:

```bash
env CUDA_VISIBLE_DEVICES= \
 PYTHONPATH=/tmp/g2m-server-tests.IcPjmO:/tmp/g2m-http-runtime.LOcBGk:/tmp/g2m-soap-deps.o0Kjat \
 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python -m pytest \
 src/graph2mat/core/data/tests/test_spin_collinear.py \
 src/graph2mat/core/data/tests/test_periodic_hermiticity_oracles.py \
 src/graph2mat/tools/lightning/tests/test_matrix_components.py \
 src/graph2mat/tools/server/tests/test_physical_serialization.py \
 -q -rs --tb=short --disable-warnings
```

Baseline vigente antes de nuevas pruebas: **125 passed, 53 warnings, 3.86 s**.
FastAPI 0.142.2 y uvicorn 0.54.0; el import verificado de graph2mat apunta a
este checkout. El primer baseline 52/1 omitido usaba solo el entorno base;
el intermedio 117/8 fallos carecía de dscribe en el path, causa contrastada.
No instalar dependencias si desaparecen estos temporales: informar del recurso
faltante y no declarar certificación HTTP a partir de omisiones.

## Task 1: Contrato espectral en memoria y batches

**Files:** tres archivos de tests core/Lightning enumerados arriba.

**Interfaces:** consumir `MatrixDataProcessor.matrix_from_data(data, predictions=None, threshold=None, is_batch=None, out_format=None)`, `TorchBasisMatrixData.new` y `Batch.from_data_list`; no producir APIs nuevas.

- [x] **1.1 Ampliar el oráculo periódico existente.** En cada k ya fijado (0, 0.17 y -0.31 en x), conservar los campos analíticos independientes y comparar `reconstructed.eigenvalue(k=k, spin=channel).eig` con `scipy.linalg.eigh(H_analytic(k), S_analytic(k), eigvals_only=True)` para spin 1/2, simetría dirigida/media arista y bases s/p y s/d/f. Verificar número completo de autovalores. Mantener los guards raw existentes sin extender su semántica.
- [x] **1.2 Añadir `test_nonorthogonal_analytic_levels_and_gauge_shift` en `test_processing.py`.** Dos orbitales: S=[[1,.25],[.25,1]], H=[[1.375,-.125],[-.125,1.375]], niveles exactos [1,2] eV. Probar reconstrucción NumPy y Torch sin mutación y H+0.7S → [1.7,2.7], mismo gap. Contrastar con `L=cholesky(S)`, `eigvalsh(L^-1 H L^-T)`, sin inversas explícitas ni cortes. Usar `solve_triangular` para los dos lados.

  Aserciones mínimas de ese control:

  ```python
  np.testing.assert_allclose(levels, [1., 2.], rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(shifted_levels - levels, .7, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(np.diff(shifted_levels), np.diff(levels), rtol=1e-10, atol=1e-10)
  ```

- [x] **1.3 Añadir `test_nonorthogonal_controls_detect_wrong_or_singular_overlap`.** Con la misma H, S=I no reproduce [1,2]; S=[[1,1],[1,1]] provoca `numpy.linalg.LinAlgError`/su equivalente SciPy en el problema generalizado. Las entradas permanecen intactas. No probar un rechazo SPD del writer que no existe.
- [x] **1.4 Añadir `test_h_and_overlap_batch_predictions_keep_distinct_structures` en `test_matrix_components.py`.** Construir dos grafos de dos/tres átomos con misma tabla, H/S simétricas explícitas y S bien condicionada. Collate PyG real; predicciones cambian H y S de cada estructura con valores diferentes. Reconstrucción tuple y `yield_from_batch` conservan geometría, canales, matrices y niveles independientes; point_labels/edge_labels/batch originales no cambian. Matrices esperadas se construyen antes del aplanado, no desde los resultados.
- [x] **1.5 Ejecutar tests nuevos y vecinos.** Comando del baseline añadiendo `src/graph2mat/core/data/tests/test_processing.py`; exigir cero fallos y no omitir controles aprobados. Al certificar comportamiento existente los tests pueden pasar directamente; no fabricar un fallo productivo para aparentar RED→GREEN. Los controles negativos deben demostrar que S equivocada o singular no pasa el oráculo. Si aparece un bug, conservar su reproducción y seguir diagnóstico/TDD antes de cualquier fix.

Tolerancias fijadas: matrices de campos independientes, las actuales (igualdad cuando aplique); H(k)/S(k), atol=2e-14; niveles float64 bien condicionados rtol=1e-10 y atol=1e-10 eV. No relajarlas por un resultado inesperado.

## Task 2: Round-trip, gauge y restauración de referencia

**Files:** tests Lightning/servidor enumerados arriba.

**Interfaces:** reutilizar `_model(tmp_path, kind="hamiltonian", spin=False, **options)`, `_request`, `MatrixWriter._on_batch_end`, `write_prediction` y rutas de extrapolación; ninguna firma productiva nueva.

- [x] **2.1 Ampliar `test_http_prediction_roundtrips_real_matrix_spin_overlap_and_fermi`.** En sus ramas H no polarizada/colineal y local/upload, comparar niveles leídos con el problema independiente H_original-Ef*S usando Ef=.5. Confirmar niveles = niveles_originales-.5. Las ramas DM/EDM conservan sus assertions y no reciben una S ni corrección inventadas.
- [x] **2.2 Añadir `test_http_nonorthogonal_tshs_zero_fermi_preserves_generalized_levels`.** Parametrizar local/upload y spin; usar `_model(..., fermi_level=0.)`, formato `.TSHS`, comparar H/S completos y niveles. Reutilizar el rechazo ya existente de Ef no nulo, verificando que no hay escritura/overwrite parcial.
- [x] **2.3 Añadir `test_matrix_writer_h_and_overlap_hsx_preserves_generalized_levels`.** Callback MatrixWriter real sobre predicciones H+S no ortogonales, spin 1/2, S no diagonal. Leer con geometría/base explícitas, comparar H/S, canales y niveles contra arrays originales; verificar no mutación. Reutilizar trainer SimpleNamespace y helpers vecinos; no introducir Trainer.fit ni nuevo export.
- [x] **2.4 Ampliar controles de referencia existentes.** En `test_extrapolated_h_only_delta_restores_explicit_reference_overlap` y `test_h_overlap_delta_and_reference_recover_s_without_doubling_it`, comparar niveles con arrays esperados y retener assertions de no mutación. El caso H+S debe detectar duplicar S. Si hace falta referencia no nula de delta-S para evitar un control trivial, fijar de antemano S_ref distinta de S_total, ambas SPD; el delta no necesita ser SPD.
- [x] **2.5 Ejecutar bloque focal entero y comprobar resultados sin omisiones.** Mismo comando de 1.5. Un HTTP 500 no se clasifica por su status solamente: exponer excepción y causa antes de editar código. Registrar por separado entorno, defecto de fixture y defecto productivo.

Precisión fijada antes de implementar: `.nc` y `.TSHS` usan float64; matrices H/S y niveles rtol=1e-10, atol=1e-10 (eV para H/niveles). El writer HSX instalado usa version=2 y dtype=float64 por defecto, confirmado leyendo su implementación; usar la misma tolerancia, no asumir HSX float32. Mantener guards existentes de finitud, formatos, Ef y publicación atómica sin expandirlos.

## Task 3: Documentación, regresiones y cierre acotado 4.1

**Files:** docs e informe/roadmap/traspaso enumerados arriba, este plan y registro de continuidad.

**Interfaces:** producir informe reproducible y estado 4.1 certificado, o estado incompleto con bloqueo exacto. 7.6 continúa pendiente de diseño propio aprobado; no se cierra con los tests H+S.

- [x] **3.1 Documentar el contrato comprobado.** Actualizar docs existentes con soporte del grafo, componentes y orden, H-only frente H+S, S etiqueta/predicción, spin, formatos/unidades/gauge, diferencia entre baseline aislada y H0, y límites SPD/cutoff/raw. Añadir ejemplo mínimo que use la API actual, no una interfaz prometida.
- [x] **3.2 Ejecutar regresiones relevantes.** Con el mismo entorno CPU, ejecutar `pytest src/graph2mat/core/data/tests src/graph2mat/tools/lightning/tests/test_matrix_components.py src/graph2mat/tools/lightning/tests/test_physical_training.py src/graph2mat/tools/server/tests -q -rs --tb=short --disable-warnings`. Registrar fallos históricos o de entorno con causa; ningún fallo nuevo sin resolver. No afirmar que esto es la suite de todo el proyecto.
- [x] **3.3 Ejecutar estilo y revisar diff.** `/tmp/g2m-quality-deps.UsKznt/bin/ruff check` sobre archivos Python tocados y `git diff --check`. Si no hay cambios productivos, mypy del producto no demuestra nada nuevo y no se ejecuta como gate ficticio. Comprobar que los cambios previos ajenos siguen presentes.
- [x] **3.4 Revisión Native final.** Una revisión independiente final del alcance 4.1 y su matriz requisito→evidencia; corregir hallazgos importantes con prueba específica y repetir gates afectados. No revisiones por tarea ni commits.
- [x] **3.5 Informe y actualización de estado.** Registrar nombres de pruebas, comandos, resultados efectivos y límites. Marcar 4.1 completo solo si cada requisito aprobado tiene evidencia. No confundir certificación representacional con precisión del modelo, positividad garantizada de S aprendida ni solución OOD. Actualizar roadmap/traspaso y presentar después diseño 7.6 para aprobación.

## Revisión propia del plan

- Cobertura: spec §§4.1–4.3 → Task 1 + referencia Task 2; §4.4 → Task 2; aceptación §5 → Tasks 1–2; docs/entorno/cierre §§6–7 → Task 3.
- Interfaces reutilizadas y nombres corresponden a archivos actuales; no dependencias entre nuevas APIs.
- Los cinco casos Review Focus tienen pasos explícitos. Los fallos de entorno ya identificados tienen baseline completo posterior, no excepciones ocultas.
- Sin pasos de commits/worktrees por prohibición expresa del usuario. Los tests de certificación no exigen cambiar producto cuando ya cumple.

**Estado:** 4.1 cerrado2026-10-18: controles, regresiones y revisión final verificados; sin cambios productivos, commits ni worktrees. El Goal conjunto sigue pendiente de 7.6.
