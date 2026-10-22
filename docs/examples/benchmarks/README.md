# Benchmarks of the g2m-v2 improvements

The improvements of the `g2m-v2` branch (roadmap in `auditorias_mejoras/roadmap.md`) were
measured with SIESTA datasets of several systems:

- graphene: 2-atom cell (w90) and 6x6 supercell, displaced and thermal (MD) structures;
- out of distribution: twisted/registry bilayer graphene and graphene/hBN trained on
  monolayers or on other stackings;
- molecules: H2O and (H2O)n clusters;
- a materials bench with ionic and polar solids (BaTiO3, ZnO, LiF, MoS2-type TMDs).

Each benchmark trained graph2mat with a recipe (a graph2mat CLI config), predicted frozen
test and out-of-distribution sets and evaluated the predictions against SIESTA, with
several seeds per variant. The results, with their uncertainties and what was promoted or
discarded, are in the roadmap. The largest single gain was Delta-learning on the
non-self-consistent H0 (roadmap 5.1): on H2O the aligned H error went from 5.6 to 1.01 meV
and the HOMO-LUMO gap error from 80 to 9.6 meV.

The original benchmark runners are not part of graph2mat: they depend on a private dataset
registry, a GPU job queue and data that are not distributed. Everything they used from the
model side is in graph2mat:

| What | Where |
|---|---|
| Delta-learning (H0, H - H0 labels, adding H0 back) | `graph2mat.tools.siesta.delta`, `graph2mat siesta delta ...` |
| Spectral window / S null-space projections (4.4, 4.7) | `graph2mat.tools.spectral_targets` |
| Aligned H, Gamma spectrum, bands, DOS, gap, layers, species pairs | `graph2mat.tools.analysis.metrics` |
| Error breakdown by shell, l pair and range (4.3) | `graph2mat.tools.analysis.breakdown` |
| Attribution of out-of-distribution band errors | `graph2mat.tools.analysis.attribution` |
| Out-of-graph (KB) tail control (8.2) | `graph2mat.tools.analysis.kb_tail` |
| Label noise floor (5.4), DM-to-H pilot (7.7) | `graph2mat.tools.siesta.label_noise`, `graph2mat.tools.siesta.dm_pilot` |

## Running the best recipe on your data

- `delta_recipe.yaml`: the best graphene recipe (fp32, linear edge readout, gauge-invariant
  loss with S as a label, last-layer irreps, `torch.compile`, 16000 steps), learning H - H0.
- `delta_workflow.py`: prepares H0 and the delta samples from your SIESTA runs, and after
  training and prediction adds H0 back and evaluates. Its docstring lists the four steps.

SIESTA is taken from `--siesta`, the `SIESTA` environment variable or `siesta` on the
`PATH`. H0 must be computed with the same SIESTA version, basis and pseudopotentials as the
labels; `compute_h0` checks that the geometry matches the label's.

H0, density-matrix pilot and label-noise results are reused only when their
`.graph2mat-siesta.json` receipt matches the rewritten FDF, pseudopotentials, extra
input files (including the initial DM), and output bytes. A changed or legacy cache
without a receipt raises an error and is left intact; choose a new output directory.
Failed runs and H0/DM outputs that fail matrix or geometry validation stay in a hidden
partial directory and are never published as completed results. The receipt does not
certify the SIESTA executable version or independently recover species/basis identity
from TSHS, which does not store those labels. Keep the same SIESTA build and input basis.

`system_label` resolves FDF includes through the FDF reader. Text-only H0, DM and
label-noise rewrites require self-contained FDF input: `%include` and `<` redirections
are rejected explicitly, since an included setting could override the rewritten keys
and its file would not be staged. Resolve those external settings before using these
tools. Coordinate shifts in `noise_fdf` require coordinates in Angstrom, as before.

The DM pilot's `aligned_error` supports unpolarized matrices. Its gauge uses the full
reference overlap; RMS uses the union of stored positions in H, H_ref and S, including
positions whose error is exactly zero. Delta arithmetic matches periodic images by
their lattice translations and retains the larger supercell. Matching pseudopotentials,
basis identities and orbital ordering remain caller requirements; geometry and orbital
counts reject obvious layout mismatches but cannot identify an undocumented basis.
