# Magnetismo (8.1): representación de 8 canales, momentos en los datos y modelo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Graph2Mat carga, aprende y reconstruye Hamiltonianos magnéticos con momentos magnéticos nominales por átomo como entrada: spinor completo con SOC (8 canales reales, sin imponer inversión temporal) y spin colineal (`spin_h_only`) con momentos.

**Spec:** `auditorias_mejoras/roadmap_magnetismo.md` (D1–D3, Review Focus, Tasks 2–4). Leer entero antes de empezar.

**Architecture:** Los datos ganan una política `spin_orbit_full` (8 canales) y un array de nodo `magnetic_moments` (μB; escalar con signo en colineal, vector cartesiano 1e con SOC) leído de `magnetic_moments.json` junto al fdf. El modelo evalúa MACE una vez (solo geometría), añade a los rasgos de nodo la salida de una capa magnética local (lineal en m, sobre el grafo del readout) y evalúa el readout con m y con −m. La mitad par va a los canales pares en T y la mitad impar a los impares: inversión temporal exacta por construcción.

**Tech Stack:** Python ≥3.9, NumPy, sisl 0.16.4, Torch, e3nn 0.4.4, MACE, Lightning. Tests: `python -m pytest` con `CUDA_VISIBLE_DEVICES=''`.

## Global Constraints

- Defaults bit a bit iguales: sin la política nueva ni momentos, datos, modelos, checkpoints y salidas no cambian (`h_only`, `spin_h_only`, `spin_orbit`).
- Python 3.9 (sin `match`, sin `X | Y` en anotaciones evaluadas). Sin dependencias nuevas.
- No tocar `MD_vs_AtomicDisplacement` ni `g2m_autorun.py`.
- Canales `spin_orbit_full`, en este orden: `[a0, bx, by, bz, b0, ax, ay, az]`; signos de transpuesta `[1, -1, -1, -1, -1, 1, 1, 1]`; paridad T `[+1, +1, +1, +1, -1, -1, -1, -1]`.
- Desde los 8 valores sisl `c = [Re uu, Re dd, Re ud, Im ud, Im uu, Im dd, Re du, Im du]`:
  `a0=(c0+c1)/2, az=(c0-c1)/2, ax=(c2+c6)/2, by=(c2-c6)/2, bx=(c3+c7)/2, ay=(c7-c3)/2, b0=(c4+c5)/2, bz=(c4-c5)/2`.
  Inversa: `c0=a0+az, c1=a0-az, c2=ax+by, c6=ax-by, c3=bx-ay, c7=bx+ay, c4=b0+bz, c5=b0-bz`.
  Los canales 0–3 son exactamente los de `spin_orbit` (`[h0, gx, gy, gz]`); los 4–7 son sus residuos TRS.
- Momentos: `magnetic_moments.json` junto al fdf con `{"nominal": [...], "converged": [...] (opcional)}`; forma `(n_atoms,)` colineal o `(n_atoms, 3)` cartesiano. El modelo solo usa `nominal`.
- `magnetic_moments` en los datos: vector convertido con `data_processor.cartesian_to_basis` como `positions` (m es axial, pero el cambio de base es una rotación propia: misma conversión). Escalar sin conversión.
- Toda opción no certificada con `spin_orbit_full` o con momentos se rechaza con `ValueError` explícito (lista de `LitMACEMatrixModel` para `spin_orbit`, ampliada: `readout_per_interaction`, `long_range`, `hamiltonian_local_context`, coeficientes, `matrix_targets`, multitarget, `physical_labels`, `spectral_window`).
- Mantener `jit_script_fx=False` con SOC (cuelgue TorchScript, 8.1).

## Review Focus

- Oráculo independiente de Pauli: spinor `Σ_μ σ_μ ⊗ (a^μ + i b^μ)` construido a mano frente a `Hk` de sisl (Task 1).
- Ningún camino silencioso: momentos ausentes, forma errónea o no finitos fallan con mensaje claro; con `spin_orbit_full`, momentos obligatorios (ceros permitidos).
- Equivariancia conjunta posiciones+momentos con rotaciones propias e impropias (m axial: bajo inversión no cambia) en float64; T exacto (≤1e-10); colineal s→−s intercambia ↑/↓ exactamente.
- Dos estructuras con la misma geometría y momentos distintos dan datos y predicciones distintas (sin cachés que las confundan).
- Checkpoints anteriores (SOC y no SOC, sin las claves nuevas) cargan y predicen igual.

---

### Task 1: Política `spin_orbit_full` (datos)

**Files:** Modify `src/graph2mat/core/data/spin_orbit.py` (constantes y conversiones nuevas), `configuration.py` (tipo `MatrixComponentPolicy`, `component_transform`, validaciones junto a las de `spin_orbit`), `sparse.py` (`transpose_sign` por política, reconstrucción a sisl con 8 canales), `processing.py` (validaciones de la política; `labels_to` solo sisl). Test: `src/graph2mat/core/data/tests/test_spin_orbit_full.py` (nuevo; reutilizar fixtures de `test_spin_orbit.py`).

**Interfaces — Produces:** `SOC_FULL_TRANSPOSE_SIGN`, `SOC_FULL_TIME_REVERSAL_PARITY` (arrays de 8), `siesta_to_soc_full_channels(values) -> (..., 8)` (rechaza no finitos, no comprueba TRS), `soc_full_channels_to_siesta(channels) -> (..., 8)`, política `matrix_component_policy="spin_orbit_full"` (`n_matrix_components=8`).

- [ ] Tests que fallan:
  - Oráculo de Pauli float64: para bloques aleatorios complejos `h^μ`, spinor a mano → 8 valores sisl (vía un `sisl.Hamiltonian` spin-orbit y su `Hk`) → canales → inversa: ida y vuelta exacta y canales iguales a `a^μ, b^μ`.
  - Con `b0 = a^k = 0` los canales 0–3 coinciden con `siesta_to_soc_channels`.
  - Sistema periódico no TRS (fixture `make_soc_system` con residuos TRS añadidos): carga con `spin_orbit_full`, `matrix_from_data` reconstruye el sisl original (atol 1e-12), H(k) hermítico; con `spin_orbit` sigue fallando por TRS.
  - Signos de transpuesta: bloque `(j, i, -R)` = transpuesta con signo de `(i, j, R)` en los labels.
  - Rechazos: matriz no spin-orbit, no finitos, `out_format` no sisl, `out_matrix` distinto de hamiltonian.
- [ ] RED, implementación mínima reutilizando `component_transform` y `transpose_sign`, GREEN.
- [ ] `pytest src/graph2mat/core/data -q` sin fallos nuevos.
- [ ] Commit `feat(spin-orbit): spin_orbit_full policy with the eight real spinor channels (roadmap 8.1, magnetism)`.

### Task 2: Momentos magnéticos en los datos

**Files:** Modify `src/graph2mat/core/data/configuration.py` (campo `magnetic_moments: Optional[np.ndarray] = None` en `BasisConfiguration`/`OrbitalConfiguration`; `from_run` lee `magnetic_moments.json` junto al fdf, también dentro de ZIP), `processing.py` (campo `MatrixDataProcessor.magnetic_moments: Optional[str] = None` con valores `None | "collinear" | "vector"`, validación de forma y finitud, clave de nodo `magnetic_moments` en `_init_keys`/`_node_attr_keys`, conversión a la base para "vector"), `src/graph2mat/tools/lightning/data.py` (`MatrixDataModule(magnetic_moments=...)` pasado al procesador y guardado en hparams; `spin_orbit_full` exige `"vector"`, `"collinear"` exige `spin_h_only`), `src/graph2mat/bindings/torch/load.py` (desde hparams; ausente → `None`). Test: `src/graph2mat/core/data/tests/test_magnetic_moments.py` (nuevo).

**Interfaces — Consumes:** Task 1. **Produces:** `data.magnetic_moments` (tensor `(n_nodes,)` o `(n_nodes, 3)` en la base del procesador) en cada `TorchBasisMatrixData` y en los batches.

- [ ] Tests que fallan:
  - `from_run` con JSON junto al fdf (directorio y ZIP): `config.magnetic_moments` igual al nominal; sin JSON y con el procesador pidiendo momentos → `ValueError` que nombra el archivo.
  - Formas: colineal `(n,)`, vector `(n, 3)`; forma errónea, número de átomos distinto o NaN → `ValueError`.
  - Vector: `data.magnetic_moments` = `cartesian_to_basis(m)`; batch de dos estructuras concatena en orden de nodos.
  - Misma geometría, momentos distintos → `magnetic_moments` distintos en los datos (ningún reuso entre muestras).
  - Procesador sin momentos: datos idénticos a hoy (mismas claves, mismos arrays).
  - `load_from_lit_ckpt` restaura `magnetic_moments`; checkpoint sin la clave → `None`.
- [ ] RED, implementación, GREEN; `pytest src/graph2mat/core/data src/graph2mat/bindings/torch src/graph2mat/tools/lightning -q` sin fallos nuevos.
- [ ] Commit `feat(data): nominal per-atom magnetic moments as node data (roadmap 8.1, magnetism)`.

### Task 3: Modelo — bloque de 8 canales, capa magnética e inversión temporal

**Files:** Modify `src/graph2mat/bindings/e3nn/modules/spin_orbit.py` (bloque completo: `b0` con la base de `h0` y transpuesta −1; `a^k` con la RTP `(l_i⊗l_j)⊗1e` y transpuesta +1; generalizar con signos por canal), new `src/graph2mat/bindings/e3nn/modules/magnetic.py` (capa magnética), `src/graph2mat/models/mace.py` (`MatrixMACE`: rasgos magnéticos concatenados a `node_feats` antes del readout; doble pasada y combinación por paridad), `src/graph2mat/tools/lightning/models/mace.py` (`LitMACEMatrixModel(magnetic_moments: Optional[str] = None)`; con `spin_orbit=True` y `"vector"` → 8 canales; con `"collinear"` → 2 canales de `spin_h_only`; acuerdo con el datamodule en `setup`/`on_fit_start`), `src/graph2mat/tools/lightning/cli.py` (deriva `model.spin_orbit` y `model.magnetic_moments` de los datos; rechaza conflictos explícitos), `callbacks.py` y `server/server_app.py` (rechazos para `spin_orbit_full`), `docs/data_loading.rst` (sección «Magnetic moments»). Tests: `src/graph2mat/bindings/e3nn/modules/tests/test_spin_orbit_full_block.py`, `src/graph2mat/tools/lightning/tests/test_magnetic_model.py` (nuevos).

**Interfaces — Consumes:** Tasks 1–2. **Produces:** entrenamiento y predicción con `matrix_component_policy="spin_orbit_full"` + momentos vector, y `spin_h_only` + momentos colineales.

Capa magnética (contrato; arquitectura interna a criterio del implementador):
- Entrada: `node_feats` de MACE (sin m), `magnetic_moments` (`1x0e` colineal o `1x1e` vector), aristas del readout (`edge_index`, armónicos esféricos y rasgos radiales ya calculados en `MatrixMACE._readout`).
- Salida: irreps fijas (p. ej. `Kx0e + Kx1o + Kx1e + Kx2e + Kx2o`, ambas paridades) **lineales en m**: término propio `m_i ⊗ node_feats_i` y término vecino `Σ_j R(r_ij) (m_j ⊗ Y(r_ij))` normalizado por número medio de vecinos. Así el bloque on-site de i ve m_i y los m_j vecinos, y el de arista ve m_i y m_j.
- Doble pasada: `F(m)` y `F(-m)` con el mismo `node_feats` de MACE (MACE se evalúa una vez). SOC: canal par `(F(m)+F(-m))/2`, impar `(F(m)-F(-m))/2` según `SOC_FULL_TIME_REVERSAL_PARITY`. Colineal: `up=(u(m)+d(-m))/2`, `down=(d(m)+u(-m))/2`. Antes de la pérdida.
- Sin momentos (`magnetic_moments=None`) el modelo y su forward son exactamente los actuales.

- [ ] Tests que fallan:
  - Bloque de 8 canales: equivariancia (≥8 rotaciones incluidas impropias, float64, bases s/p/d, `symmetric_matrix=True`), signos de transpuesta y Hermiticidad de H(k) reconstruido.
  - Modelo completo pequeño (MACE mínimo): equivariancia conjunta posiciones+momentos (vector axial), T exacto (`m→-m` da canales pares iguales e impares negados, ≤1e-10 en float64), `m=0` da canales impares exactamente 0, colineal `s→-s` intercambia ↑/↓, permutación de átomos, predicción sensible a m (dos configuraciones distintas → salidas distintas).
  - Alcanzabilidad: con las irreps ocultas de la campaña (`16x0e+16x0o+16x1o+16x1e+16x2e+16x2o`, aristas hasta l=3) ningún coeficiente de los 8 canales es estructuralmente cero (patrón del hallazgo 0o de 8.1).
  - Acuerdo datos/modelo: desacuerdo de política o de momentos → `ValueError` en fit/validate/test/predict; CLI deriva y rechaza conflicto explícito.
  - Rechazos: `MatrixWriter`, servidor HTTP y opciones no certificadas.
  - Compatibilidad: checkpoints SOC y no SOC anteriores cargan y predicen igual (`test_spin_orbit_model.py`, `test_cli.py` verdes).
- [ ] RED, implementación, GREEN.
- [ ] Smoke de entrenamiento CPU de 3 pasos con datos sintéticos de cada variante (SOC vector y colineal): pérdida finita y decreciente.
- [ ] `pytest src -q` sin fallos nuevos frente a la referencia de la Task 0 del goal.
- [ ] Commit `feat(lightning): magnetic moments as model input with exact time reversal (roadmap 8.1, magnetism)`.
