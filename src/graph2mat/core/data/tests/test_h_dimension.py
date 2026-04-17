import os
import sisl
import numpy as np

path = "/home/christian/Escritorio/CINN/repositorios/ejemplo_graph2mat_h2o/results/hamiltonian_reconstructed/predicted_matrices/4.HSX"

print("Existe:", os.path.exists(path))
if os.path.exists(path):
    print("Tamaño:", os.path.getsize(path))
else:
    raise FileNotFoundError(path)

sile = sisl.get_sile(path)

print("Tipo de sile:", type(sile))

with sile:
    H = sile.read_hamiltonian()

print("Leído correctamente")
print("shape:", H.shape)
print("nnz:", H.nnz)



pred_path = "/home/christian/Escritorio/CINN/repositorios/ejemplo_graph2mat_h2o/results/hamiltonian_reconstructed/predicted_matrices/4.HSX"
real_path = "/home/christian/Escritorio/CINN/repositorios/ejemplo_graph2mat_h2o/test/4/siesta.TSHS"

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