# Investigación técnica: bug de shape con `out_matrix=hamiltonian` en `ejemplo_spin_o2`

## Hallazgo principal

La causa raíz más probable es **desalineación de componentes de matriz**:

- El dataset de `hamiltonian` está llegando con labels multi-componente (`point_labels`, `edge_labels` 2D: `[N, C]`, en el caso reportado `C=3`).
- La cabeza de salida del modelo (`MatrixMACE` + `Graph2Mat`) está inicializada por defecto con `n_matrix_components=1`, por lo que produce salidas 1D (`[N]`).
- La loss (`block_type_mae`) falla al restar `nodes_pred - nodes_ref` por shape incompatible.

## Evidencia del flujo

1. `MatrixDataModule` recibe `n_matrix_components` pero por defecto es `1`. Se pasa al `MatrixDataProcessor`. (`src/graph2mat/tools/lightning/data.py`)
2. El modelo `LitMACEMatrixModel` también tiene `n_matrix_components=1` por defecto, y se lo pasa a `MatrixMACE`/`E3nnGraph2Mat`. (`src/graph2mat/tools/lightning/models/mace.py`)
3. El CLI solo enlaza algunos argumentos `data.* -> model.*`, pero **no enlaza** `n_matrix_components`; por lo tanto, puede quedar desincronizado o en default si no se setea explícitamente en ambos lados. (`src/graph2mat/tools/lightning/cli.py`)
4. Las labels se construyen desde `config.matrix.to_flat_nodes_and_edges`; si los bloques son 3D, se aplanan como 2D `[n_elements, n_components]`. (`src/graph2mat/core/data/matrices/basis_matrix.py`)
5. El `Graph2Mat` en forward colapsa a 1D cuando el bloque de salida es 3D (`output.ndim != 4 -> ravel`) y solo conserva 2D cuando el bloque es 4D (incluye componentes). (`src/graph2mat/core/modules/graph2mat.py`)
6. La métrica `get_predictions_error` hace resta directa sin adaptar dimensiones (`node_error = nodes_pred - nodes_ref`). (`src/graph2mat/core/data/metrics.py`)

## Sobre soporte Hamiltonian spin

- `PhysicsMatrixType` incluye `hamiltonian` (soporte nominal de tipo). (`src/graph2mat/core/data/configuration.py`)
- Existen tests para colineal polarizado de **2 componentes** en data/sparse/procesado. (`src/graph2mat/core/data/tests/test_spin_collinear.py`)
- No hay cobertura equivalente para `n_matrix_components=3` en entrenamiento MACE/lightning.
- En conversiones a sisl para multi-componente, el código asume explícitamente `spin="polarized"` cuando hay más de una componente CSR, lo cual es una pista de soporte parcial orientado a 2 componentes. (`src/graph2mat/core/data/sparse.py`)

## Recomendación mínima correcta

1. Detectar automáticamente el número de componentes del dataset (p. ej. en `MatrixDataModule.setup`, desde el primer sample), y:
   - setear `data.n_matrix_components`
   - y sincronizar con `model.n_matrix_components`.
2. O como mínimo, exigir en validación de config que `data.n_matrix_components == model.n_matrix_components` y fallar temprano con error claro.
3. Añadir test de integración lightning para caso multi-componente (incluyendo mismatch intencional para verificar error temprano).

## Qué NO hacer como arreglo final

- No hacer `nodes_ref = nodes_ref[:, 0]` sin una justificación física explícita del canal a seleccionar.
- No parchear solo la métrica: la métrica solo expone una incompatibilidad previa del pipeline.
