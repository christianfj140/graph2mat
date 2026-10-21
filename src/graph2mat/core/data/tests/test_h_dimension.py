import os
import sisl
import numpy as np

# Local check of a reconstructed H2O prediction; set G2M_H2O_EXAMPLE to the example
# workspace (with results/... and test/4/siesta.TSHS) to run it.
EXAMPLE = os.environ.get("G2M_H2O_EXAMPLE", "")
path = os.path.join(EXAMPLE, "results/hamiltonian_reconstructed/predicted_matrices/4.HSX")

if not EXAMPLE or not os.path.exists(path):
    import pytest

    pytest.skip(f"local example file not available: {path}", allow_module_level=True)

print("Tamaño:", os.path.getsize(path))

sile = sisl.get_sile(path)

print("Tipo de sile:", type(sile))

with sile:
    H = sile.read_hamiltonian()

print("Leído correctamente")
print("shape:", H.shape)
print("nnz:", H.nnz)



pred_path = path
real_path = os.path.join(EXAMPLE, "test/4/siesta.TSHS")

with sisl.get_sile(pred_path) as f:
    H_pred = f.read_hamiltonian()

with sisl.get_sile(real_path) as f:
    H_real = f.read_hamiltonian()

pred = H_pred.tocsr()
real = H_real.tocsr()

print("pred shape:", pred.shape, "nnz:", pred.nnz)
print("real shape:", real.shape, "nnz:", real.nnz)

diff = pred - real
diff_data = diff.data if diff.nnz > 0 else np.array([0.0])

print("diff nnz:", diff.nnz)
print("max abs diff:", np.max(np.abs(diff_data)))
print("mean abs diff:", np.mean(np.abs(diff_data)))
print("rmse:", np.sqrt(np.mean(diff_data**2)))