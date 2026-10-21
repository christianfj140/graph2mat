# Auditoría roadmap ↔ código — graph2mat `g2m-v2` @ a66b46c (2026-10-21)

Solo lectura. Fuente: `auditorias_mejoras/roadmap.md` (filas de las tablas Ola 0–8 y «Descartado»).
Rutas relativas a la raíz del repo. `S:` = código, `T:` = test. Test ejecutado: los 33 tests de los módulos
migrados (`src/graph2mat/tools/{siesta,analysis}/tests`, `tools/tests/test_spectral_targets.py`) pasan en CPU (0,85 s).
El resto se verificó con grep/lectura, sin ejecutar.

Categorías: OK · discrepancia · falta · decisión del usuario · histórico-sin-código.

## Recuento

| Estado real | Filas |
|---|---|
| OK | 64 |
| discrepancia | 8 (0.8, 1.9, 1.12, 2.8, 4.7, 6.8, 6.9, 8.1) |
| falta | 0 |
| decisión del usuario | 5 (1.13, 1.15, 3.8, 3.13, 6.5) |
| histórico-sin-código | 4 (0.10, 2.9, 4.6, X.1) |
| **Total** | **81** |

La sección «Banco de materiales» no tiene ids propios: solo cita filas existentes (7.3, 5.2, 7.2, 4.7, 8.4…); sus
ejecutores (`g2m_bench_materials.py`, `g2m_bench_benchmark.py`) siguen en MD_vs a propósito.

## Tabla

| id | estado declarado | estado real | evidencia (file:line) | nota breve |
|---|---|---|---|---|
| 0.1 | ✅ | OK | S: `core/modules/graph2mat.py:1119` (`_pair_swap_index`); T: `bindings/e3nn/modules/tests/test_edge_pairing_and_safety.py:81` | homonuclear no simétrico ve el receptor |
| 0.2 | ✅ | OK | S: `tools/lightning/data.py:181,192-198,415`; `tools/lightning/cli.py:105,190` | `SYMMETRIC_MATRIX_TYPES` H/DM/EDM; `processing.py:167` sigue en False pero la CLI/DataModule lo deducen |
| 0.3 | ✅ | OK | S: `core/data/metrics.py:2278`; T: `core/data/tests/test_density_numpy_oracle.py:8` | |
| 0.4 | ✅ | OK | S: `core/data/metrics.py:1610,1647,1684`; T: `core/data/tests/test_metrics_valid_labels.py:20-22` | |
| 0.5 | ✅ | OK | `.github/workflows/tests.yaml:21,50,90-91` | CI configurada; el roadmap ya dice que no se ejecutó remotamente |
| 0.6 | ✅ | OK | T: `core/modules/tests/test_graph2mat.py:48,88`; `tools/lightning/tests/test_models.py:91,118` | líneas citadas coinciden; 72b468c en HEAD |
| 0.7 | ✅ | OK | S: `tools/lightning/cli.py:323,329,340`; `bindings/torch/load.py:59`; T: `tools/lightning/tests/test_cli.py:83` | cae9955 en HEAD |
| 0.8 | ✅ | discrepancia | S: `tools/lightning/models/mace.py:213-215,222-227` | guardas de `version!="new"` e `initial_node_feats` existen, pero **ningún test** las ejercita (grep sin resultados) |
| 0.9 | ✅ | OK | T: `core/data/tests/test_coefficient_matrix_mse.py:159`; `core/data/tests/test_coefficient_only_forward.py:63` | grafos sin aristas/ghost |
| 0.10 | 🔶 | histórico-sin-código | runner `g2m_improvements_benchmark.py` solo en MD_vs; `docs/examples/benchmarks/README.md:19-22` | ejecutor fuera a propósito; «falta el perfilado (2.9)» desactualizado: 2.9 ya tiene perfil |
| 1.1 | ✅ | OK | S: `tools/lightning/models/mace.py:104` (`readout_fp32=False`); T: `bindings/e3nn/modules/tests/test_hamiltonian_context.py:424` | fp32 es precisión del Trainer |
| 1.2 | ✅ | OK | S: `tools/lightning/models/mace.py:64` | ✅ = receta; default sigue `num_polynomial_cutoff=3` (coherente con que poly6 empeora OOD) |
| 1.3 | ⚪ | OK | S: `tools/lightning/models/mace.py:85` (`avg_num_neighbors=1.0`) | solo configuración; medida en MD_vs (`g2m_bench_neutral.py`) |
| 1.4 | 🔶 | OK | S: `tools/lightning/models/mace.py:71,284,297`; `models/mace.py:26`; T: `test_edge_pairing_and_safety.py:190` | se testea `drop_last_mace_interaction`, no el flag a nivel LitMACE |
| 1.5 | 🔶 | OK | S: `bindings/e3nn/modules/graph2mat.py:226`; `tools/lightning/models/mace.py:115`; 9 ficheros de test | |
| 1.6 | 🔶 | OK | S: `tools/lightning/models/mace.py:75` (`4x0e+4x1o+4x2e`) | default «4 canales por l» coincide |
| 1.7 | ✅ | OK | S: `tools/lightning/models/mace.py:94,97`; `edge_operations.py:68`; `node_operations.py:60`; T: `bindings/e3nn/modules/tests/test_e3nngraph2mat.py:188` | default arista = `E3nnLinearEdgeBlock`, nodo = simple (como dice) |
| 1.8 | 🔶 | OK | S: `core/data/metrics.py:1722` (`edge_weight=1.0`); T: `tools/lightning/tests/test_selection_metric.py:58` y otros 16 | |
| 1.9 | ⚪ | discrepancia | S: `bindings/e3nn/modules/edge_operations.py:275` | `E3nnEdgeBlockNodeMix` **sin ningún test** (ni equivarianza) pese a la regla 1 del roadmap; default no cambiado (decisión del usuario respetada) |
| 1.10 | ❌ | OK | S: `bindings/e3nn/modules/hamiltonian_readout.py:167,249`; T: `bindings/e3nn/modules/tests/test_hamiltonian_readout.py` | descartado para la receta; código conservado y testado |
| 1.11 | ⚪ | OK | S: `tools/lightning/models/mace.py:63` (`num_bessel=10`) | solo configuración |
| 1.12 | 🔶 | discrepancia | `core/modules/graph2mat.py:380-381`; T: `test_edge_pairing_and_safety.py:118` | «SWA no es posible (deepcopy del modelo)» está desactualizado: deepcopy/pickle ya funcionan y se testean. EMA/clipping no tienen código propio (callbacks/Trainer de Lightning) |
| 1.13 | ✅ | decisión del usuario | S: `tools/lightning/models/mace.py:70,74,87` (`num_interactions=2`, `hidden_irreps=20x…`, `correlation=1`) | el ✅ dice «los defaults se cambian al cerrar la Ola 1»: no se cambiaron; cambiar defaults es del usuario |
| 1.14 | 🔶 | OK | S: `bindings/e3nn/modules/matrixblock.py:97-104`; T: `test_edge_pairing_and_safety.py:141` | |
| 1.15 | ✅ | decisión del usuario (opt-in OK) | S: `tools/lightning/model.py:232-241,621,648`; `tools/lightning/cli.py:25,306`; T: `tools/lightning/tests/test_selection_metric.py:53-104` (20 casos) | `selection_metric=None` conserva `val_loss`; el monitor por defecto sigue pendiente de decisión |
| 2.1 | 🔶 | OK | S: `bindings/e3nn/modules/preprocessing.py:116`; T: `test_edge_pairing_and_safety.py:164` | |
| 2.2 | 🔶 | OK | S: `tools/lightning/models/mace.py:107` (`lean_mace=True`); T: `test_edge_pairing_and_safety.py:213` | default coincide |
| 2.3 | ✅ | OK | S: `models/mace.py:44`; T: `test_edge_pairing_and_safety.py:212` | default False |
| 2.4 | ✅ | OK | S: `core/modules/graph2mat.py:539`; T: `test_edge_pairing_and_safety.py:255` | activo para aristas homonucleares simétricas |
| 2.5 | ✅ | OK | S: `tools/lightning/data.py:810-811` | `pin_memory` sin test (menor); `persistent_workers` sí (`test_tensor_cache.py:168`) |
| 2.6 | ✅ | OK | S: `core/data/metrics.py:95`; T: `core/data/tests/test_complete_labels_skip_mask.py:34` | |
| 2.7 | 🔶 | OK | S: `tools/lightning/models/mace.py:72`; T: `test_hamiltonian_context.py:456` | |
| 2.8 | 🔶 | discrepancia (texto) | S: `models/mace.py:121-127`; `tools/lightning/models/mace.py:98-99`; T: `test_hamiltonian_context.py:323,347` | la fila dice «`readout_per_interaction` pendiente», pero está implementado y testado (lo corrige solo «Pendiente real», roadmap.md:122) |
| 2.9 | 🔶🧪 | histórico-sin-código | S: `node_operations.py:78` (`E3nnProjectedLinearTSQNodeBlock`); T: `test_e3nngraph2mat.py:188` | perfil = medida; el 🧪 de precisión de `linp` ya está resuelto (❌ en punto 7 de Extrapolación, roadmap.md:722-728) |
| 2.10 | ✅ | OK | S: `bindings/e3nn/modules/matrixblock.py:339`; T: `test_homonuclear_fusion.py:127` | |
| 3.1 | ✅ | OK | T: `tools/lightning/tests/test_epoch_metrics.py`, `test_ddp_cpu.py`, `test_gauge_epoch_metrics.py`… | |
| 3.2 | ✅ | OK | T: `core/data/tests/test_metrics_valid_labels.py` | commit citado 4239aa1 no está en la historia de g2m-v2 (solo `backup/g2m-v2-before-squash`) |
| 3.3 | ✅ | OK | S: `tools/lightning/callbacks.py:28-37` (splits=["predict"]), `:251`; T: `test_cli.py:60` | 48897b2 fuera de la historia (squash) |
| 3.4 | ✅ | OK | T: `tools/server/tests/test_server_app.py` (9 tests), `test_physical_serialization.py:64`, `test_extrapolation.py:12`, `tools/lightning/tests/test_plot_matrix_error.py` | |
| 3.5 | ✅ | OK | T: `tests/test_minimal_install.py`; `.github/workflows/tests.yaml:48` | |
| 3.6 | ✅ | OK | S: `tools/lightning/data.py:306`; `core/data/processing.py:269,2100`; T: `test_datamodule_temporary_roots.py`, `core/data/tests/test_data_filters_and_subgraphs.py:60` | |
| 3.7 | ✅ | OK | S: `core/data/processing.py:88,820`; T: `core/data/tests/test_irreps_from_data_rtps.py` | |
| 3.8 | 🔶 | decisión del usuario | S: `bindings/e3nn/modules/preprocessing.py:15-19` (sin residual, documentado); `core/data/node_feats.py:46`; T: `core/data/tests/test_node_feats_symmetry.py:10` | residual = decisión del usuario; `E3nnInteraction` no tiene ningún test. 3a38754 reescrito como e585e0f |
| 3.9 | ✅ | OK | S: `tools/lightning/data.py:312-313`; `tools/lightning/model.py:36`; `pyproject.toml:177,185`; T: `test_checkpoint_provenance.py` | 3cc3aa3 fuera de la historia (squash) |
| 3.10 | ✅ | OK | S: `core/data/basis.py:215`; T: `core/data/tests/test_orbital_conventions.py` | |
| 3.11 | ✅ | OK | S: `core/data/configuration.py:471,535`; `core/data/sparse.py:115`; `core/data/neighborhood.py:30`; `core/data/processing.py:2021`; T: `test_nonstandard_sc_off.py`, `test_sparse_ncol.py`, `test_neighborhood_degenerate_cell.py` | default `pbc=(T,T,T)` sin cambiar, como dice; d9075c8 fuera de la historia (squash) |
| 3.12 | ✅ | OK | S: `tools/lightning/split_guardrail.py:119`; `tools/lightning/data.py:433`; T: `tools/lightning/tests/test_split_guardrail.py` (17 tests) | |
| 3.13 | ⏸ | decisión del usuario | — | multi-GPU excluido |
| 3.14 | ✅ | OK | T: `tools/lightning/tests/test_coefficient_cuda_training.py` (CUDA; se omite en CPU), `test_coefficient_*` | |
| 4.1 | ✅ | OK | T: `core/data/tests/test_processing.py:163,218`; `test_periodic_hermiticity_oracles.py`; `test_physical_serialization.py` | tests del plan presentes |
| 4.2 | ✅ | OK | S: `core/data/metrics.py:1890`; `tools/analysis/metrics.py:94` (`H_MAE_gauge_meV`); T: 15 ficheros | la fila aún cita la métrica c* en scripts de MD_vs; ya está en `graph2mat.tools.analysis.metrics` (fila sin actualizar) |
| 4.3 | ✅ | OK | S: `tools/analysis/breakdown.py:78`; T: `tools/analysis/tests/test_breakdown_attribution.py:29` | migrado |
| 4.4 | 🔶⚪ | OK | S: `core/data/metrics.py:981,1892`; `tools/lightning/data.py:310`; `tools/spectral_targets.py:36,55`; T: `core/data/tests/test_metrics.py:179`; `tools/tests/test_spectral_targets.py:22` | |
| 4.5 | 🔶 | OK | S: `bindings/torch/multitarget.py:52`; T: `bindings/torch/tests/test_multitarget_loss.py:153,180`; `core/data/tests/test_physical_losses.py` | McWeeny sin código, coherente con «no aplicable» |
| 4.7 | 🔶 | discrepancia (menor) | S: `bindings/torch/data/dataset.py:182`; `tools/spectral_targets.py:46`; T: `tools/tests/test_spectral_targets.py:32` | la rama `spectral_window="<fichero>.npz"` (cómo se usa `spectral_null.npz`) **no tiene test**; todos los tests usan `True` |
| 4.6 | ⚪ | histórico-sin-código | — | no hace falta código (medida en grafeno) |
| 5.1 | ✅ | OK | S: `tools/siesta/delta.py:68,134,147`; `tools/cli/siesta/main_cli.py:11`; T: `tools/siesta/tests/test_delta.py:40-123` (incl. CLI) | migrado y pasa; límite SOC en 8.1 |
| 5.2 | 🔶 | OK | igual que 1.5 | |
| 5.3 | 🔶 | OK | S: `tools/lightning/model.py:231`; T: `test_coefficient_matrix_training.py`, `test_coefficient_target_collate.py` | default False; «forwarded975eb» = errata de «forward ed975eb» (en HEAD) |
| 5.4 | ✅ | OK | S: `tools/siesta/label_noise.py:58`; T: `tools/siesta/tests/test_label_noise_dm_pilot.py:7` | migrado |
| 6.1 | 🔶 | OK | S: `tools/lightning/data.py:307`; T: `bindings/torch/data/tests/test_tensor_cache.py` | |
| 6.2 | 🔶 | OK | T: `core/data/tests/test_sparse_vectorization.py`, `bindings/torch/data/tests/test_torch_sparse_vectorization.py` | |
| 6.3 | ✅ | OK | S: `tools/lightning/data.py:119`; `tools/lightning/model.py:291`; T: `test_e3nngraph2mat.py:287-292` (+11 ficheros) | |
| 6.4 | 🔶 | OK | S: `tools/lightning/data.py:44,314,400` (`max_batch_cost=None`); T: `tools/lightning/tests/test_cost_batching.py` | default intacto |
| 6.5 | ⚪ | decisión del usuario | — | ruta dispersa no justificada |
| 6.6 | ⚪ | OK | S: `bindings/e3nn/modules/graph2mat.py:54`; T: `test_e3nngraph2mat.py:18` | |
| 6.7 | 🔶 | OK | S: `bindings/e3nn/modules/matrixblock.py:50` (`fuse_homonuclear=False`); T: `test_homonuclear_fusion.py`, `test_homonuclear_fusion_model.py` | plan con 45 casillas sin marcar |
| 6.8 | 🔶❌ | discrepancia | S: `models/mace.py:535-541`; `tools/lightning/models/mace.py:112-114,444`; T: `test_edge_pairing_and_safety.py:355` (solo GPU) | la docstring ofrece `"reduce-overhead"` (CUDA graphs) como modo válido, pero el roadmap lo marca ❌ (violación de segmento con torch 2.11). Ni aviso ni rechazo en el código |
| 6.9 | 🔶 | discrepancia (menor) | S: `tools/lightning/models/mace.py:36,110,295`; T: `tools/lightning/tests/test_magnetic_model.py:202` | ningún test construye un modelo con `cueq=True`; el único test es su rechazo en modelos magnéticos |
| 7.1 | 🔶 | OK | S: `edge_operations.py:185`; T: `test_low_rank_edge_readout.py`, `test_low_rank_edge_model.py` | |
| 7.2 | 🔶⚪ | OK | S: `preprocessing.py:121-125`; `tools/lightning/models/mace.py:77` (default 0); T: `test_edge_pairing_and_safety.py:399` | |
| 7.3 | 🔶❌ | OK | S: `models/mace.py:101`; `tools/lightning/models/mace.py:79` (False); T: `test_edge_pairing_and_safety.py:461` | |
| 7.4 | 🔶 | OK | S: `bindings/e3nn/modules/slater_koster.py:104`; T: `test_slater_koster_readout.py`, `test_slater_koster_model.py` | |
| 7.5 | 🔶 | OK | S: `tools/lightning/models/mace.py:90`; T: `tools/lightning/tests/test_shared_readout_max.py` | default `point_type` |
| 7.6 | ✅ | OK | S: `core/data/multitarget.py:22`; `tools/lightning/models/mace.py:129-132,194-212`; T: `test_multitarget*.py` (6 ficheros) | plan del piloto con 26 casillas sin marcar |
| 7.7 | ❌ | OK | S: `tools/siesta/dm_pilot.py:45,105`; T: `tools/siesta/tests/test_label_noise_dm_pilot.py:46,59` | NO_GO; código migrado |
| 8.1 | 🔶 | discrepancia (depende de MD_vs) | S: `core/data/spin_orbit.py:38,99`; `bindings/e3nn/modules/spin_orbit.py:29,96`; `bindings/e3nn/modules/magnetic.py:11`; `tools/lightning/models/mace.py:136-193`; `tools/lightning/cli.py:83`; `tools/lightning/callbacks.py:105`; `tools/server/server_app.py:103`; `tools/siesta/delta.py:39-54,114-131`; `auditorias_mejoras/run_soc_campaign.py:39-42,79-106` | F1–F3 y código magnético presentes y testados (`test_spin_orbit*.py`, `test_magnetic*.py`). Hueco: la receta SOC recomendada (Δ) no se puede ejecutar solo con graph2mat: proyección TRS de H−H0, predicción y métricas viven en `run_soc_campaign.py`, que importa `dataset_design_curves_v1`, `g2m_bench_*` y `g2m_delta` de MD_vs. `graph2mat siesta delta label` resta los 8 canales sin proyectar y el cargador rechaza residuos TRS > 1e-5; `compute_h0` no fuerza el arranque sin espín (requisito del informe F0). MatrixWriter y servidor rechazan SOC. F4/F5 magnéticas = decisión del usuario |
| 8.2 | 🔶 | OK | S: `tools/lightning/models/mace.py:144` (0.0); `tools/analysis/kb_tail.py:17`; T: `core/data/tests/test_edge_cutoff_padding.py`, `tools/lightning/tests/test_edge_cutoff_padding_model.py`, `test_breakdown_attribution.py:61` | «Pendiente real» (roadmap.md:128-129) aún la da por condicionada; plan con 6 casillas sin marcar |
| 8.3 | 🔶 | OK | S: `core/data/processing.py:1976`; T: `core/data/tests/test_same_z_different_basis.py` | |
| 8.4 | 🔶 | OK | S: `models/long_range.py:91`; `tools/lightning/models/mace.py:82-83` (smearing 2.0 Å); `tools/siesta/delta.py:155`; T: `tools/lightning/tests/test_long_range.py:62,107`; `test_delta.py:113` | |
| X.1 | ❌ (descartado) | histórico-sin-código | — | |

## Comprobaciones transversales

**Dependencia de MD_vs_AtomicDisplacement.** `src/`, `tests/` y `docs/examples/` no tienen referencias a
`MD_vs`/`MDVS_ROOT`/`dataset_design_curves`. Las referencias en `docs/superpowers/` son de procedencia. Siguen
dependiendo de MD_vs estos scripts de campaña en `auditorias_mejoras/`: `run_readout_comparison.py:24-26,152`
(7.1/7.4/1.9), `run_soc_campaign.py:37-42` (8.1 F5), `run_magnetic_campaign.py:40-42` (8.1 magnético, pausado),
`magnetismo_f0_fe.py:32`, `soc_f0_bi2se3.py:29`, `audit_multitarget_w90.py:14` (ruta absoluta). Son ejecutores, que
se dejaron fuera a propósito; solo 8.1 pierde por ello una capacidad que no tiene sustituto en graph2mat.
`docs/examples/benchmarks/delta_workflow.py` y `delta_recipe.yaml` no tienen ningún test en el repo.

**Apéndice.** 217 IDs, todos únicos, y todos apuntan a filas existentes. Las filas sin ID de Claude son 2.10, 3.14,
4.7 y 8.4, todas (GPT), como se espera. `roadmap_map.py` («se comprueba con…») no existe en el repo. Todas las filas
tienen estado. Las filas 5.1, 7.7, 8.1 y 8.2 contienen `|` sin escapar (`h0|label|add-h0`, `|H−H0|`, `|Hk−Hk^H|`, `|H|²`),
que parten la tabla en Markdown.

**Higiene del roadmap.** La línea 1 de roadmap.md es un fragmento suelto de una fila de tabla (sobra). La tabla «Orden de
ejecución» (roadmap.md:622-633) está desactualizada (A 🧪, B–G ⏳ aunque la Ola 3 está cerrada y casi todo lo de 4–7 hecho).
El título «Banco de materiales (pendiente de definir)» también. 4.7 aparece antes que 4.6. Los commits citados d9075c8 (3.11),
18f4ca2 (2.8), 4239aa1 (3.2), 48897b2 (3.3) y 3cc3aa3 (3.9) solo están en `backup/g2m-v2-before-squash`; 3a38754 (3.8)
corresponde a e585e0f en HEAD. El código de todos ellos sí está.

**Planes con casillas sin marcar y punto dado por hecho.**

| Plan | sin marcar / marcadas | Punto |
|---|---|---|
| 2026-10-14-homonuclear-weight-fusion-6-7.md | 45 / 7 | 6.7 cerrado opt-in |
| 2026-10-19-multitarget-w90-pilot.md | 26 / 0 | 7.6 ✅ (piloto completo) |
| 2026-10-19-soc-8-1.md | 32 / 0 | 8.1 F1–F3 implementadas |
| 2026-10-20-readout-padding-8-2.md | 6 / 0 | 8.2 implementado |
| 2026-10-21-magnetismo-codigo.md | 12 / 0 | 8.1 «código cerrado y revisado» |
| 2026-10-21-migracion-mdvs.md | 15 / 0 | migración hecha (sección 2026-10-21) |

`2026-10-19-multitarget-w90-six.md` y `2026-10-19-soc-f4-f5.md` no tienen casillas. Los otros 9 planes tienen
todas las casillas marcadas.
