# Migración a graph2mat de las funciones del roadmap que viven en MD_vs_AtomicDisplacement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** que todas las funciones del roadmap se puedan usar clonando solo graph2mat: Δ-learning (5.1), generadores de
proyecciones espectrales (4.4/4.7), métricas de evaluación (H alineado, bandas, DOS, gap, capas, pares de especies),
desglose del error (4.3), atribución del error de bandas OOD, techo de ruido de etiquetas (5.4), piloto DM→H (7.7) y
control de la cola KB fuera del grafo (8.2), más un ejemplo documentado de los benchmarks.

**Spec:** petición del usuario (2026-10-21) y el inventario de esta sesión: `MD_vs_AtomicDisplacement/Comparison/scripts/`
`g2m_delta.py`, `g2m_ood_eval.py`, `g2m_error_breakdown.py`, `g2m_ood_band_attribution.py`, `g2m_label_noise.py`,
`g2m_dm_pilot.py`, `g2m_bench_kb_control.py`, `g2m_bench_benchmark.py` (`_bands`, `band_window_metrics`,
`species_pair_errors`), `dataset_design_curves_v1.py` (`structure_metrics`, `band_dos_metrics`),
`predict_model_on_dataset.py` (`copy_h0_overlaps`, 8.4).

**Architecture:** port fiel de la lógica a módulos de librería que reciben rutas y parámetros explícitos (sin registro de
muestras, sin `ROOT`, sin rutas de la máquina). SIESTA se lanza con un ejecutor común (`SIESTA` del entorno o `siesta`
del `PATH`). La evaluación trabaja sobre pares (predicción, referencia) en ficheros, independiente de cómo se predijo.
Los lanzadores de benchmarks no se portan (dependen del arnés de 2345 líneas, la cola GPU y datos de `/mnt`); en su lugar,
un ejemplo documentado con la receta real y un flujo Δ-learning ejecutable.

**Tech Stack:** Python ≥3.9, NumPy, SciPy, sisl 0.16, typer (CLI). Sin dependencias nuevas. Tests:
`python -m pytest` con `CUDA_VISIBLE_DEVICES=''`.

## Global Constraints

- No modificar nada en `MD_vs_AtomicDisplacement` ni en `$G2M_DATA_ROOT` (solo lectura). Estado `git status` de
  MD_vs idéntico antes y después.
- Fórmulas, umbrales y nombres de métricas idénticos a los originales: `INTERLAYER_DZ=1.5` Å, ventana ±2 eV (Γ y
  camino), ventana espectral 4.4 `3.0` eV, espacio nulo 4.7 `threshold=1e-2`, `floor=1e-3`, `CANON_THR=1e-2`,
  camino hexagonal `Γ–K–M–Γ = [[0,0,0],[1/3,2/3,0],[0.5,0.5,0],[0,0,0]]`, DOS σ=0.1 eV en `E_F±3` eV con 601 puntos,
  conchas `SHELLS_ANG=(0.0,0.1,1.6,2.6,3.0,3.9,4.5,5.2,6.0,99.0)`, claves `H0_KEYS` verbatim.
- Generalizaciones permitidas (documentadas): camino k, malla DOS, ficheros de base y ejecutable SIESTA como parámetros;
  `combine_hamiltonians` suma todos los componentes de spin (el original solo el 0 y escribía siempre sin spin) y falla
  si los spins difieren; pseudos `*.psf` y `*.psml` en todos los ejecutores.
- Python 3.9 (sin `match`, sin `X | Y` evaluado). Código, docstrings y commits en inglés.
- Ningún fichero nuevo contiene rutas absolutas de la máquina.

## Review Focus

1. Predicción `h_only` con S identidad: toda métrica espectral resuelve con la S de la referencia (test con S≠1).
2. `nsc` distinto entre H0/predicción y referencia: se iguala a la de la referencia como el original (test).
3. Hamiltoniano con spin polarizado en Δ-learning: se conservan ambos componentes (test).
4. SIESTA ausente o que no escribe el TSHS: error claro, sin dejar un H0 a medias en caché (test con ejecutable falso).
5. Ventana de energía vacía (aislante con E_F a mitad de gap): `nan` como el original, sin excepción (test).

---

### Task 1: ejecutor SIESTA común

**Files:** Create `src/graph2mat/tools/siesta/runner.py`; Test `src/graph2mat/tools/siesta/tests/test_runner.py`.

**Interfaces — Produces:**
- `replace_fdf_keys(text: str, keys: Mapping[str, str]) -> str` (borra líneas cuya clave coincide ignorando caso, `.`, `-`, `_`; añade `key  value` al final; termina en `\n`).
- `system_label(fdf: Path) -> str` (`SystemLabel` o `"siesta"`).
- `find_pseudos(directory: Path) -> list[Path]` (`*.psf`+`*.psml` del primer directorio con alguno entre `directory` y sus padres; `RuntimeError` si ninguno).
- `siesta_command(siesta: Optional[str] = None) -> list[str]` (argumento > env `SIESTA` (shlex) > `shutil.which("siesta")`; `FileNotFoundError` claro).
- `run_siesta(work_dir: Path, fdf_text: str, pseudos: Sequence[Path], *, siesta=None, extra_files: Mapping[str, Path] = {}, nice: int = 10) -> Path` (limpia y crea `work_dir`, escribe `RUN.fdf`, copia pseudos/extras, ejecuta con stdin=RUN.fdf, stdout=`RUN.out`, `OMP_NUM_THREADS=1`; devuelve `work_dir`).

- [ ] Tests: `replace_fdf_keys` reproduce el original sobre un fdf con `DM.Tolerance`/`dm-tolerance`; `system_label`; `find_pseudos` en el padre; `siesta_command` respeta `SIESTA` y falla sin ejecutable; `run_siesta` con un ejecutable falso (script que copia stdin a `RUN.out` y escribe `x.TSHS`).
- [ ] Implementar, pasar tests, commit `feat(siesta): shared SIESTA runner for H0 and label tools`.

### Task 2: Δ-learning (5.1) y S de H0 para long_range (8.4)

**Files:** Create `src/graph2mat/tools/siesta/delta.py`; Test `src/graph2mat/tools/siesta/tests/test_delta.py`.

**Interfaces — Consumes:** Task 1. **Produces:**
- `H0_KEYS` (verbatim), `h0_fdf(text: str) -> str = replace_fdf_keys(text, H0_KEYS)`.
- `compute_h0(reference_dir: Path, out_dir: Path, *, reference_matrix: Optional[Path] = None, siesta=None, geometry_tol: float = 1e-4) -> Path` (caché si existe `out_dir/<label>.TSHS`; tras SIESTA comprueba geometría frente a `reference_matrix` y borra el TSHS si difiere > tol).
- `combine_hamiltonians(a, b, sign: float) -> sisl.Hamiltonian` (a + sign·b, S de `a`, `b.set_nsc(a.nsc)` sobre copia).
- `write_delta_sample(reference_dir: Path, reference_matrix: Path, h0: Path, out_dir: Path) -> Path` (copia `RUN.fdf`, escribe H−H0 con el nombre de la referencia; devuelve la ruta del label).
- `add_h0(predicted: Path, h0: Path, out: Optional[Path] = None) -> Path` (H_pred + H0, en sitio por defecto).
- `stage_h0_overlap(h0: Path, structure_fdf: Path) -> Path` (copia el TSHS de H0 junto al fdf con el nombre que daría SIESTA).

- [ ] Tests (sisl sintético, sin SIESTA): ida y vuelta `add_h0(write_delta_sample(...)) == H_ref` a 1e-12; S del label = S de la referencia; spin polarizado conserva 2 componentes; spins distintos → `ValueError`; nsc distinto se iguala; `compute_h0` con SIESTA falso: escribe fdf con `H0_KEYS`, copia pseudos, falla y borra si la geometría difiere, error claro si no hay TSHS.
- [ ] Implementar, pasar, commit `feat(siesta): delta learning on the non-self-consistent H0 (roadmap 5.1)`.

### Task 3: proyecciones espectrales 4.4/4.7

**Files:** Create `src/graph2mat/tools/spectral_targets.py`; Test `src/graph2mat/tools/tests/test_spectral_targets.py`.

**Produces:** `SPECTRAL_WINDOW_EV = 3.0`, `WINDOW_FILE = "spectral_window.npz"`, `NULL_FILE = "spectral_null.npz"`,
`spectral_window_columns(matrix: Path, window_ev=3.0) -> np.ndarray`, `null_space_columns(matrix: Path, threshold=1e-2, floor=1e-3) -> np.ndarray`,
`write_projection_files(columns: Sequence[np.ndarray], out_dirs: Sequence[Path], name: str) -> int` (relleno a la anchura máxima, float32, escritura atómica, devuelve anchura).

- [ ] Tests: columnas S-ortonormales (`C^T S C = I` en la ventana); espacio nulo escala `1/sqrt(max(λ,floor))`; relleno y anchura; fichero leído por `graph2mat` (`np.load(...)["C"]`).
- [ ] Implementar, pasar, commit `feat(tools): writers of the spectral window and S null-space projections (4.4, 4.7)`.

### Task 4: métricas de evaluación

**Files:** Create `src/graph2mat/tools/analysis/{__init__,metrics}.py`; Test `src/graph2mat/tools/analysis/tests/test_metrics.py`.

**Produces:** `HEXAGONAL_PATH`, `gauge_shift(delta, overlap) -> float`,
`gamma_metrics(predicted, reference, window_ev=2.0) -> dict` (claves de `structure_metrics` sin `sample_id`),
`layer_masks(reference, interlayer_dz=1.5)`, `layer_metrics(predicted, reference, *, kpath=HEXAGONAL_PATH, n_per_segment=40, window_ev=2.0) -> dict` (claves de `g2m_ood_eval.metrics`),
`path_bands(predicted, reference, kpath, n_per_segment)`, `band_dos_metrics(predicted, reference, *, kpath, n_per_segment=60, grid=(24,24,1)) -> dict`,
`canonical_bands(h, s, path_k, threshold) -> np.ndarray`, `band_window_metrics(predicted, reference, *, kpath, n_per_segment=20, canon_threshold=1e-2) -> dict`,
`species_pair_errors(predicted, reference, structure_fdf) -> dict`. Todas reciben rutas.

- [ ] Tests sintéticos: predicción = referencia → errores 0; predicción = referencia + c·S → error alineado 0 y `gauge_c_meV = 1e3 c`; S identidad en la predicción no cambia bandas; ventana vacía → `nan`; máscaras intra/inter en bicapa.
- [ ] Implementar, pasar, commit `feat(analysis): gauge-aligned H, band, DOS and gap metrics`.

### Task 5: desglose (4.3), atribución OOD y cola KB (8.2)

**Files:** Create `src/graph2mat/tools/analysis/{breakdown,attribution,kb_tail}.py`; Tests en `src/graph2mat/tools/analysis/tests/`.

**Produces:** `SHELLS_ANG`, `orbital_info(geometry, species_z, basis_files)`, `min_image_distances(geometry)`,
`structure_breakdown(predicted, reference, structure_fdf, basis_files, shells=SHELLS_ANG) -> dict`, `summarize_breakdown(rows) -> dict`;
`CLASSES`, `element_classes(h_ref, pred_nsc)`, `attribute(predicted, reference, n_per_segment=20, kpath=HEXAGONAL_PATH) -> dict`, `counterfactuals(predicted, reference, h0, ...) -> dict`;
`without_kb_tail(hamiltonian, radii, baseline=None) -> (sisl.Hamiltonian, dict)`.

- [ ] Tests: categorías suman el total; `share_*` suman 1; predicción exacta → 0; `without_kb_tail` borra solo pares a distancia ≥ R_i+R_j y con `baseline` corta H−H0 y devuelve H0 intacto.
- [ ] Implementar, pasar, commit `feat(analysis): error breakdown, OOD band attribution and KB-tail control`.

### Task 6: ruido de etiquetas (5.4) y piloto DM→H (7.7)

**Files:** Create `src/graph2mat/tools/siesta/{label_noise,dm_pilot}.py`; Tests en `src/graph2mat/tools/siesta/tests/`.

**Produces:** `VARIANTS`, `SHIFT_ANG`, `noise_fdf(text, shift: bool, tolerance: str, shift_ang=SHIFT_ANG) -> str`,
`run_label_noise(reference_dir, work_dir, variants=VARIANTS, siesta=None) -> dict[str, Path]`, `gamma_mae_mev(a, b) -> float`;
`DM_KEYS`, `LEVELS`, `aligned_error(H, ref)`, `symmetric(geometry, csr)`, `keep_charges(csr, overlap, geometry)`,
`pilot_structure(reference_dir, h0, work_dir, *, seed=0, levels=LEVELS, siesta=None) -> dict`.

- [ ] Tests: `noise_fdf` desplaza solo el bloque de coordenadas y cambia `DM.Tolerance`; `keep_charges` conserva las cargas de Mulliken; `symmetric` da DM simétrica; `pilot_structure` con el ejecutor SIESTA sustituido por un falso devuelve filas con las mismas claves.
- [ ] Implementar, pasar, commit `feat(siesta): label noise floor (5.4) and DM-to-H pilot (7.7)`.

### Task 7: CLI, ejemplo de benchmarks y equivalencia con el original

**Files:** Create `src/graph2mat/tools/cli/siesta/delta.py` (+ registro en `main_cli.py`), `docs/examples/benchmarks/{README.md,delta_recipe.yaml,delta_workflow.py}`,
`auditorias_mejoras/informe_migracion_mdvs_20261021.md`; Modify `auditorias_mejoras/roadmap.md` (filas 4.3, 4.4, 4.7, 5.1, 5.4, 7.7, 8.2, 8.4 apuntan a graph2mat).

- [ ] CLI `graph2mat siesta delta h0|label|add-h0` (test con `typer.testing.CliRunner` y SIESTA falso).
- [ ] Equivalencia (script en el scratchpad, no versionado, solo lectura de MD_vs y `/mnt`): G1 `h0_fdf` sobre fdf reales; G2 label H−H0 contra los labels de la caché de MD_vs; G3 ida y vuelta y `_combine` original; G4 proyecciones 4.4/4.7 contra las funciones originales; G5 un H0 real con SIESTA (celda w90 de 2 átomos) contra la caché; G6 todas las métricas contra las originales sobre predicciones sintéticas (referencia + ruido con semilla, S identidad); G7 `without_kb_tail`; G8 fdf de ruido; G9 piloto DM con SIESTA falso en ambos. Criterio: igualdad exacta o `allclose(rtol=0, atol=1e-12)`; G5 `atol=1e-6` eV.
- [ ] Informe con resultados (sin rutas de la máquina), roadmap actualizado, suite CPU completa, commit `docs: ...`.
