# 8.2 Aristas de readout largas (edge_cutoff_padding) — Implementation Plan

> **Estado (2026-10-22):** ejecutado; `edge_cutoff_padding` implementado el 2026-10-20 (fila 8.2). Las casillas no se marcaron durante la ejecución; el estado vigente está en la sección «Estado vigente» de [`roadmap.md`](../../../auditorias_mejoras/roadmap.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Graph2Mat predice los bloques de H más allá de R_i + R_j (colas de proyectores KB) alargando solo el grafo del readout.

**Architecture:** Un padding escalar en Å se suma al radio de cada punto al construir el grafo de matrices (`MatrixDataProcessor.get_cutoff`), así que las aristas llegan a R_i + R_j + 2·padding y SIESTA ya trae esas etiquetas. El modelo alarga en consecuencia la base radial y la envolvente por pares del readout. MACE puede seguir con su radio anterior vía `mp_cutoff` (roadmap 2.7).

**Tech Stack:** Python ≥3.9, NumPy, sisl, Torch/Lightning, MACE; pytest con `python -m pytest`.

**Spec:** diseño aprobado en chat (usuario, 2026-10-20: «me vale»). Motivación: auditoría F4 de Bi2Se3 (`auditorias_mejoras/roadmap.md`, fila 8.1): elementos fuera del grafo hasta 10,1 Å (Bi–Bi), 9,7 Å (Bi–Se) y 8,7 Å (Se–Se), frente a cortes de 8,10, 7,41 y 6,73 Å. Recortarlos desplaza autovalores hasta 12 eV porque S está mal condicionada.

## Global Constraints

- Default `edge_cutoff_padding = 0.0`: grafos, etiquetas, modelos, checkpoints y resultados idénticos a los actuales. Checkpoints antiguos sin la clave cargan con 0.
- `edge_cutoff_padding` es un float finito ≥ 0 en Å; cualquier otro valor lanza `ValueError`.
- El padding se suma una vez a cada punto: un par (i, j) es arista si d < (R_i + p) + (R_j + p). Con un radio escalar de la tabla, el corte global es 2(R + p).
- El mismo valor en datos y modelo. La CLI lo enlaza de `data` a `model` (patrón `link_arguments` de `matrix_targets`). `load_from_lit_ckpt` lo pasa al procesador desde los hiperparámetros del datamodule.
- Modelo: `matrix_r_max = 2·(maxR + p)`. Con `pair_envelope`, los radios por especie son R + p. Con `mp_cutoff` debe cumplirse 0 < mp_cutoff < matrix_r_max (validación existente).
- No commitear en `MD_vs_AtomicDisplacement`. Compatibilidad con Python 3.9.

## Review Focus

- Default 0 bit-idéntico, incluido el conteo de aristas en estructuras periódicas (Task 1: `test_zero_padding_is_identical`).
- Un elemento de H entre R_i+R_j y R_i+R_j+2p aparece en las etiquetas y vuelve en la reconstrucción (Task 1).
- Supercelda (`nsc`) insuficiente para las aristas nuevas: error claro existente, no truncado silencioso (Task 1).
- Desacuerdo de padding entre datos y modelo: lo resuelve la CLI; en la API Python hay un chequeo en `on_fit_start`/`setup` (Task 2).
- Equivariancia con aristas largas y `mp_cutoff` (Task 2).

---

### Task 1: Datos — `edge_cutoff_padding` en procesador, DataModule y carga de checkpoint

**Files:** Modify `src/graph2mat/core/data/processing.py` (dataclass `MatrixDataProcessor`: campo, validación en `__post_init__`, `get_cutoff`), `src/graph2mat/tools/lightning/data.py` (`MatrixDataModule.__init__` y la creación del procesador), `src/graph2mat/bindings/torch/load.py`. Test: `src/graph2mat/core/data/tests/test_edge_cutoff_padding.py` (nuevo).

**Interfaces — Produces:** `MatrixDataProcessor(edge_cutoff_padding: float = 0.0)`, `get_cutoff` devuelve `R + p` (array por tipo) o `2·(R + p)` (escalar). `MatrixDataModule(edge_cutoff_padding: float = 0.0)`, guardado en hparams.

- [ ] Tests que fallan:
  - `test_zero_padding_is_identical`: para una geometría periódica s/p, `edge_index`, `neigh_isc` y etiquetas idénticas sin argumento y con 0.0.
  - `test_padding_adds_edges_and_their_labels`: dos átomos con orbitales de R = 1 Å a 2,5 Å y un elemento de H no nulo entre ellos en un `sisl.Hamiltonian`. Con p = 0 no hay arista. Con p = 0,3 la arista existe, su etiqueta es el valor exacto y `matrix_from_data` lo reconstruye (atol 1e-12).
  - `test_padding_validation`: −0,1, NaN e inf lanzan `ValueError`.
  - `test_checkpoint_loader_keeps_padding`: `load_from_lit_ckpt` en un checkpoint mínimo con `edge_cutoff_padding` en los hparams del datamodule reproduce el valor; sin la clave, 0.
- [ ] RED, implementación, GREEN; `pytest src/graph2mat/core/data src/graph2mat/tools/lightning/tests/test_cli.py -q` sin fallos nuevos.
- [ ] Commit `feat(data): edge_cutoff_padding to extend the matrix graph beyond R_i + R_j (roadmap 8.2)`.

### Task 2: Modelo y CLI

**Files:** Modify `src/graph2mat/tools/lightning/models/mace.py` (`LitMACEMatrixModel.__init__`: `matrix_r_max`, radios de `pair_envelope`; chequeo de acuerdo con el datamodule junto al de `spin_orbit`), `src/graph2mat/tools/lightning/cli.py` (`link_arguments("data.edge_cutoff_padding", "model.edge_cutoff_padding")`). Test: `src/graph2mat/tools/lightning/tests/test_edge_cutoff_padding_model.py` (nuevo).

**Interfaces — Consumes:** Task 1. **Produces:** `LitMACEMatrixModel(edge_cutoff_padding: float = 0.0)`.

- [ ] Tests que fallan:
  - `test_model_radial_range_follows_padding`: `matrix_r_max` de la base radial del readout y radios de la envolvente con p = 0 y p = 0,5.
  - `test_long_edges_equivariant_with_mp_cutoff`: MACE pequeño con `mp_cutoff` = 2·maxR, p = 0,5 y geometría con aristas en (2R, 2R + 2p). Equivariancia ante 8 rotaciones (patrón de `test_models.py`) y predicción no nula en las aristas largas.
  - `test_padding_disagreement_fails`: datamodule con p = 0,5 y modelo con 0 (y al revés) lanzan `ValueError` en `trainer.fit` y en `validate`.
  - `test_cli_links_padding`: la CLI propaga el valor de `data` a `model`.
- [ ] RED, implementación, GREEN; `pytest src/graph2mat/tools/lightning -q` sin fallos nuevos.
- [ ] Commit `feat(lightning): readout radial range follows edge_cutoff_padding (roadmap 8.2)`.

### Task 3: F5 brazo directo (controlador)

`auditorias_mejoras/run_soc_campaign.py`: brazo `direct`. Etiquetas H completas proyectadas a TRS (sin restar H0), `edge_cutoff_padding = 1.2` (cubre las distancias de la auditoría) y `mp_cutoff = 2·maxR` de la base. Mismas métricas que Δ, sin sumar H0. Prueba corta de 20 pasos y luego 3 semillas en `g2m_autorun`. Registrar en roadmap 8.1/8.2.
