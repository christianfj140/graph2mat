Model options and the recommended recipe
=========================================

Every model and data option added on this branch is opt-in: with the defaults the model
behaves as before, except for the changed defaults listed in ``CHANGELOG.md``. This page
says what each option is for and what it did when it was measured, so that you can choose
which ones to try. The full parameter descriptions are in the docstrings of
`graph2mat.tools.lightning.models.mace.LitMACEMatrixModel` and
`graph2mat.tools.lightning.MatrixDataModule` (also shown by ``graph2mat models mace main fit --help``).

The measurements below come from the project's benchmark campaigns (graphene ``w90`` and
``6x6``, graphene bilayers, H2O, TMD, BaTiO3, ZnO, Bi2Se3). They are the effect on those
systems, not a guarantee for others.

Recommended recipe
------------------

The best tested configuration for SIESTA Hamiltonians is in
``docs/examples/benchmarks/delta_recipe.yaml``, with the data preparation in
``docs/examples/benchmarks/delta_workflow.py``:

- **Δ-learning** (``graph2mat siesta delta``): the model learns H − H0, with H0 the
  non-self-consistent Hamiltonian of the superposed atomic densities, and H0 is added back to
  the prediction. Largest gain measured: H2O aligned H error 5.6 → 1.0 meV, gap error
  80 → 9.6 meV.
- **Gauge-invariant loss**: ``matrix_component_policy="h_and_overlap"`` data,
  ``loss: graph2mat.metrics.gauge_invariant_mse`` and ``n_output_components: 1`` (S is a
  label, not predicted). The energy zero of each structure is not learnable by a local model;
  the loss removes the best ``c·S`` shift (one per structure, common to both spins).
  ``6x6`` aligned H 1.05 → 0.70 meV; H2O 29.6 → 5.6 meV.
- **Model size**: ``correlation: 3``, ``max_ell: 3``, ``num_interactions: 3``,
  ``num_polynomial_cutoff: 6``, fp32 (``trainer.precision: 32-true``).

These are not the API defaults (``correlation=1``, ``num_interactions=2``,
``num_polynomial_cutoff=3``, ``loss=elementwise_mse``), so that existing configurations
keep their behaviour. ``num_polynomial_cutoff=3`` extrapolates better to structures unlike
the training set (monolayer → bilayer, bilayer → moiré).

Readout
-------

``edge_block_readout``
    Default ``E3nnLinearEdgeBlock`` (tensor product plus a linear term): ``w90`` H error
    12.6 → 5.1 meV. Alternatives, measured without a general gain:
    ``E3nnLowRankEdgeBlock`` (``edge_block_readout_kwargs: {rank: ...}``),
    ``E3nnSlaterKosterEdgeBlock`` (better H in ``w90``, worse out of distribution),
    ``E3nnEdgeBlockNodeMix`` (better in TMD, worse out of distribution in graphene).
``edge_hidden_irreps``, ``edge_message_mid_irreps``
    Width of the edge messages, and of the node features projected before they are copied to
    every edge. ``edge_message_mid_irreps`` is the main VRAM knob of the edge readout.
``edge_update_layers``, ``edge_update_node_irreps``
    Messages that depend jointly on both atoms of an edge. No robust gain measured.
``pair_envelope``
    Readout radial features vanish at each pair's R_i + R_j. No gain measured on
    graphene/hBN.
``readout_per_interaction``, ``keep_last_layer_irreps``, ``output_scaleshift``, ``basis_grouping``
    Measured neutral or within seed noise.
``hamiltonian_local_context``
    Scalar local-Hamiltonian context channels in the readout. Experimental.

Physics
-------

``long_range`` (``"charges"`` or ``"qeq"``)
    Learned charges and their Ewald potential added to H as ``(V_i + V_j)/2 · S_ij``. Needs
    ``h_and_overlap`` data. Gains depend on the material: BaTiO3 gap and bands −25 to −45 %
    with fixed charges, which break polar ZnO slabs; ``qeq`` avoids that.
``spectral_window`` (data) with ``loss_kwargs: {spectral_weight: ...}``
    First-order eigenvalue error in a window around E_F at Γ. A string selects other
    S-orthonormal columns, e.g. the near-null directions of S: better spectrum, worse H.
``physical_labels`` (data)
    Electron count, Mulliken charges and energy consistency as extra loss terms.
``edge_cutoff_padding`` (data and model) with ``mp_cutoff``
    Readout edges longer than R_i + R_j to reach Kleinman–Bylander projector blocks. With
    Δ-learning H0 already carries those tails, so it is only needed without Δ.

Spin
----

``matrix_component_policy="spin_h_only"``
    Collinear spin-polarized Hamiltonians (two components). Stable.
``matrix_component_policy="spin_orbit"`` with ``spin_orbit: true``
    Spin-orbit Hamiltonians without magnetism (time-reversal-symmetric channels). Trained on
    Bi2Se3; the practical recipe is Δ-learning (all of the SOC comes through H0). See
    :doc:`data_loading`.
``matrix_component_policy="spin_orbit_full"`` with ``magnetic_moments``
    Magnetic spin-orbit with per-atom moments as input. Code and tests only, never trained on
    real data.

Training control
----------------

``training_stages``
    Stages with their own loss, learning rate and frozen or trained parts (``mace``,
    ``readout``, ``context``). Under DDP, freezing needs ``find_unused_parameters``.
``selection_metric``
    Epoch metric used for ``hp_metric`` and the best checkpoint, e.g. ``val_matrix_rmse``
    when the loss changes between stages. Default: ``val_loss``.
``lr_scheduler`` (CLI)
    Attached to the model's Adam optimizer. Before this branch it was silently ignored with
    ``optimizer: null``.
``coefficient_training``
    Loss on irreducible coefficients instead of matrix elements. Works; no speed-up measured.
``matrix_targets``
    Joint H + S + DM + EDM prediction. Works; no gain over H alone in the measured recipe.

Performance and memory
----------------------

``lean_mace``
    Default ``True``: MACE node features only. Same matrices, less memory.
``gradient_checkpointing``
    ``6x6`` peak memory −24 % (fp32) at about +34 % time.
``simplify_node_irreps``
    ``w90`` −38 % time but twice the VRAM; slower on ``6x6``. Only for small, launch-bound
    systems.
``torch_compile`` (``torch_compile_mode``), ``cueq``
    Compiled model, cuEquivariance kernels. ``reduce-overhead`` is rejected.
``max_batch_cost`` (data)
    Batches limited by a matrix-element budget instead of a structure count.
``edge_block_readout_kwargs: {fuse_homonuclear: true}``
    Shared weights for both directions of homonuclear edges: less VRAM with d/f bases.
