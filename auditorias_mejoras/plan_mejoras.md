# Plan de mejoras de graph2mat (rama g2m-v2)

Fecha: 2026-09-27. Objetivo: mejorar graph2mat como herramienta general, priorizando la
precisión, más los arreglos de VRAM y velocidad que sean fáciles. El grafeno de
`MD_vs_AtomicDisplacement` se usa como benchmark de validación.

## Estado (2026-09-27)

- **Fase 0 hecha.** Informe en
  `MD_vs_AtomicDisplacement/Comparison/results/dataset_design_curves_v1/g2m_phase0/README.md`.
  - bf16 casi duplica el error en los saltos a primeros vecinos. Prioridad alta.
  - Los pares a 4–5,2 Å se predicen como cero (envolvente p=3). Hallazgo nuevo.
  - El gauge tiene un efecto moderado y ambiguo.
  - Bloques KB: descartados para grafeno.
  - 34 vecinos por átomo con `avg_num_neighbors=1`.
- **Fase 1 hecha** (cambios sin commit en `g2m-v2`). Tests en
  `bindings/e3nn/modules/tests/test_edge_pairing_and_safety.py`.
- **Siguiente:** la Ronda 1 pasa a ser tres cambios **sin código**: fp32, `num_polynomial_cutoff=6`
  y `avg_num_neighbors=34,2`. Runner: `Comparison/scripts/g2m_improvements_benchmark.py`, `w90`
  con 3 semillas.
- **Ronda 1 hecha** (`w90`, 3 semillas; `g2m_improvements/summary.json`). H-MAE medio:

  | Variante | H-MAE (meV) | Por semilla |
  |---|---:|---|
  | Línea base reentrenada | 24,5 | 20,2 / 29,6 / 23,8 |
  | fp32 | 15,0 | 13,6 / 15,8 / 15,6 |
  | poly6 | 18,0 | 14,8 / 23,6 / 15,4 |
  | avgnn | 25,2 | sin mejora |
  | combo (fp32 + poly6 + avgnn) | 14,6 | 13,2 / 15,3 / 15,4 |

  - Bandas: 27,9 → 19,5 meV (combo). DOS L1: 5,3% → 3,1% (combo).
  - fp32 sube la VRAM de `w90` de 1074 a 1346 MiB.
  - Pendiente: fp32 + poly6 sin avgnn, y confirmar con 5 semillas. Para `6x6` hace falta 2.1
    (cabeza en fp32) o 3.1 (menos VRAM), porque ahí fp32 no cabe.
- **Implementado** (opciones desactivadas por defecto salvo la pérdida; cada una con test):

  | Opción | Plan | Qué hace |
  |---|---|---|
  | `readout_fp32` | 2.1 | Readout de Graph2Mat en fp32 aunque MACE vaya en bf16 |
  | `edge_message_mid_irreps` | 3.1 | Proyecta las features de nodo antes de copiarlas a cada arista |
  | `output_scaleshift` | 2.3 | Escala por irrep y desplazamiento 0e on-site, ajustados con las etiquetas de entrenamiento al empezar `fit` |
  | `keep_last_layer_irreps` | 2.5 | MACE con una interacción extra que se descarta; todas las capas quedan con l>0 |
  | `mp_cutoff` | 3.5 | MACE hace message passing solo por aristas cortas; el readout ve todas |
  | EMA | 2.6 | Callback `EMAWeightAveraging` de Lightning, que ya funciona porque el modelo se puede copiar |
  | Pérdida por defecto `elementwise_mse` | 2.7 | Sustituye a `block_type_mse` |
  | Menos sincronizaciones, `pin_memory`, workers persistentes, steps sin la salida de MACE | 3.2–3.4 | — |

  **Pendiente de medir:** `r32poly6`, `fp32poly6_mid16`, `fp32poly6_ema`, `fp32poly6_ss`,
  `fp32poly6_kl`, `fp32poly6_mp3/mp4` en `w90`, y la sonda de VRAM de `6x6`.
- **Corrección:** la referencia de `w90` es `graphene.TSHS` (sisl no desplaza el TSHS). Aun así,
  las etiquetas están referidas a E_F por estructura, como comprobó el commit 442b8e1b del
  benchmark.

## 0. Punto de partida: lo que ya dice el benchmark

- La pérdida era el techo (§11.7): `block_type_mae` → `elementwise_mse`, de 5,29 a 2,00 meV en
  `6x6`. Las pruebas de ancho, profundidad y `max_ell` (§11.4–11.5) se hicieron con la
  pérdida vieja: sus conclusiones negativas hay que revalidarlas con la receta nueva.
- Las campañas entrenan en `bf16-mixed` y fp32 no cabe en `6x6` con 32 GB. En bf16, un valor
  de ~10 eV se redondea con saltos de ~60 meV, y uno de ~3 eV con saltos de ~16 meV. Es un
  posible suelo en los elementos grandes. Aquí reducir VRAM es mejorar la precisión.
- `avg_num_neighbors=1.0`: `CROSS_TESTING_MODEL` no lo fija, y se usa `correlation=3` con
  decenas de vecinos.
- `edge_hidden_irreps` nunca se ha probado: sigue en `4x0e+4x1o+4x2e`. El "2× ancho" solo
  tocó `hidden_irreps`.
- Las etiquetas son H − E_F·S con un E_F distinto por estructura: se guarda HSX, y el lector
  HSX de sisl 0.16.4 aplica `H.shift(-Ef)`. Es la sospecha (c) de §11.6.
- Los bloques KB ya están en el HSX; es graph2mat quien los descarta al construir el grafo
  (sospecha b de §11.6). No hace falta regenerar etiquetas.
- `configs/config_md.yaml` y `config_fc.yaml` usan `num_interactions: 1`. Con MACE 0.3.15 eso
  da solo escalares, y los bloques on-site salen isótropos. Esa versión de MACE no tiene
  `keep_last_layer_irreps`.
- El benchmark usa `symmetric_matrix: true`: los bugs del modo no simétrico no le afectan, pero
  sí a la herramienta.

Entorno del benchmark: mace-torch 0.3.15, e3nn 0.4.4, sisl 0.16.4, torch 2.11, RTX 5090
(32 GB). graph2mat está instalado en modo editable desde este repositorio. Si se modifica un
módulo Cython (`_labels_resort.py`, `_sparse.py`), hay que reinstalar con `pip install -e .`.

## 1. Protocolo de benchmark

**Referencia única:** `w90` 3D R=0,12, N=64, con la mejor configuración ya usada
(`elementwise_mse`, cosine, 4000 épocas, mismo test congelado), **2 semillas fijas**.

- **Línea base:** se vuelve a entrenar una vez, con esas 2 semillas, sobre el HEAD actual. Así
  cada variante se compara emparejada: mismas semillas, mismos datos, mismo test.
- **Qué se da por mejora:** las dos semillas mejoran respecto a su pareja de la línea base, y la
  mejora media supera ~2× la diferencia entre las dos semillas de la línea base. Por debajo de
  eso, el resultado se anota como "no concluyente", no como "no funciona".
- **Confirmación al final:** solo la combinación ganadora se repite con 5 semillas (y en
  `6x6` N=8 si el resultado es dudoso).
- **Coste:** ~35 min por modelo, así que ~1,2 h por variante.
- **VRAM de `6x6`:** se mide con un único forward/backward sobre un batch real, sin entrenar.
  Basta para saber si fp32 cabe tras 3.1.
- **Multiespecie:** los arreglos que solo se ven con varias especies (reordenación
  heteronuclear) se validan con tests en graph2mat, no con entrenamientos.
- **Métricas:**
  - las existentes: H-MAE, bandas y DOS;
  - H-MAE alineado en gauge;
  - error desglosado en on-site, 1ª/2ª capa de vecinos y pares lejanos;
  - VRAM pico y segundos por paso.
- **En graph2mat:** tests `pytest` de corrección en CPU, más un script de VRAM y tiempo por
  paso sobre un batch fijo.

## 2. Fase 0 — Diagnóstico sin reentrenar

Scripts en `MD_vs_AtomicDisplacement` sobre los checkpoints existentes (§11.7–11.8, §11.11).

| # | Qué medir | Cómo | Qué decide |
|---|---|---|---|
| 0.1 | ¿Es bf16 un suelo? | Evaluar los checkpoints en fp32 y en bf16; dtype de las salidas; error en elementos grandes | Prioridad de 2.1 y 3.1 |
| 0.2 | Parte del error debida al gauge | std(E_F) entre estructuras; c* = ⟨ΔH,S⟩/⟨S,S⟩ por estructura; MAE alineado frente a bruto | Prioridad de 2.4 |
| 0.3 | Peso de los bloques KB | ‖H fuera del grafo‖/‖H‖; cambio de autovalores con esos bloques a cero | Si merece la pena 4.1 |
| 0.4 | Dónde está el error | Desglose por on-site, distancia y par de l | Orienta 2.3 y 2.5 |
| 0.5 | Vecinos por átomo | E/N del grafo | Valor de 2.2 |

## 3. Fase 1 — Correcciones y red de seguridad (graph2mat)

Esfuerzo: S = medio día o menos, M = 1–3 días, L = más de 3 días.

| Arreglo | Dónde | Esfuerzo |
|---|---|---|
| Modelo copiable: `self.numpy = torch` impide deepcopy y pickle (causa del fallo de SWA). Pasarlo a atributo de clase | `core/modules/graph2mat.py:330` | S |
| Aristas homonucleares en modo no simétrico: reciben `(m_ij, m_ij)` | `core/modules/graph2mat.py:944-945` | S |
| Reordenación ±tipo en aristas heteronucleares (modo no simétrico) | `core/modules/_labels_resort.py:74-78`, `_forward_interactions` | M |
| `symmetric_matrix` deducido de `out_matrix` (True para H, DM y EDM) | `tools/lightning/data.py`, `tools/lightning/models/mace.py` | S |
| Ruta de coeficientes evaluada 2–4 veces → una sola vez | `core/modules/graph2mat.py:862-872, 967-981`; `bindings/e3nn/modules/matrixblock.py:109-113` | S |
| `_map_types` con el objeto identidad de `point_type` | `core/data/metrics.py:117-125` | S |
| Métricas: signo del error, `normalized_density_error`, `mapestdmix`, duplicadas | `core/data/metrics.py` | S |
| `initial_node_feats` se ignora | `tools/lightning/models/mace.py` | S |
| Avisos: irreps inalcanzables; features de nodo solo escalares | `E3nnIrrepsMatrixBlock.__init__` | S |

Tests mínimos:
- asociación predicción ↔ etiqueta al permutar átomos;
- equivariancia con rotación aleatoria, orbitales d, `siesta_spherical` y el preprocesado de
  aristas por defecto;
- que el modelo se pueda copiar con `deepcopy`;
- alcanzabilidad de irreps.

Más un workflow de CI que ejecute `pytest` en CPU.

## 4. Fase 2 — Precisión

Cada cambio va detrás de un flag. El valor por defecto se cambia solo tras confirmarlo en el
benchmark, salvo los bugs.

- **2.1 Política de precisión numérica (S).**
  - Con autocast activo, ejecutar en fp32 las operaciones de bloque, el cambio de base, la
    reordenación y la pérdida; el backbone sigue en bf16.
  - Evaluar siempre en fp32.
  - Opción `readout_precision: fp32|autocast`.
- **2.2 `avg_num_neighbors="auto"` (S).**
  - Calcularlo a partir de la geometría del conjunto de entrenamiento (E/N) y guardarlo en los
    hiperparámetros.
  - Avisar si queda en 1,0 con muchos vecinos.
- **2.3 ScaleShift equivariante (M).**
  - Salida = μ + σ·â, con σ por (tipo de bloque, instancia de irrep) y μ solo en las
    componentes 0e on-site. Es la referencia on-site de H.
  - Se ajusta con `block_to_coefficients` sobre el conjunto de entrenamiento.
  - Es una reparametrización, no una reponderación: la pérdida sigue en eV elemento a
    elemento.
- **2.4 Gauge (M, según 0.2).**
  - E_F y fichero de origen en la metadata.
  - Opción `h_reference: raw|fermi`.
  - Métricas alineadas en gauge.
  - Pérdida invariante con c* en forma cerrada. Requiere S como etiqueta auxiliar no predicha.
  - E_F de bandas y DOS recalculado a partir de los H y S predichos.
- **2.5 Capacidad no probada (M).**
  - `edge_hidden_irreps` de 16 a 32 canales (tras 3.1).
  - Irreps completas en la última capa de MACE: reconstruir `interactions[-1]` y
    `products[-1]`, o subir la versión de MACE.
  - Repetir profundidad, ancho y `correlation` con la receta nueva.
- **2.6 EMA (S, tras el arreglo de copia).** `AveragedModel` con decaimiento ~0,99; validar y
  guardar con los pesos EMA.
- **2.7 Valores por defecto (S).**
  - `elementwise_mse` con peso ×2 en aristas en almacenamiento simétrico (Frobenius).
  - Scheduler cosine configurable.
  - Las pérdidas por estructura y por dominio quedan como opción, con el dominio tomado de la
    metadata.

## 5. Fase 3 — VRAM y velocidad fáciles (en paralelo con la Fase 2)

| Cambio | Nota | Esfuerzo |
|---|---|---|
| Proyectar antes del gather en `E3nnEdgeMessageBlock` (`edge_mid_irreps`) | Hoy se copian 1584 dimensiones por arista (`bindings/e3nn/modules/preprocessing.py:127-130`). Palanca más probable para que fp32 quepa en `6x6` | S–M |
| `training_step` sin devolver la salida completa de MACE | `tools/lightning/model.py:145` | S |
| Sincronizaciones fáciles: conteos por tipo precalculados, claves parseadas una vez, `abs()` una vez | `_forward_interactions` | S |
| `pin_memory` y workers persistentes en validación | `tools/lightning/data.py` | S |
| Grafo de message passing separado (`mp_cutoff`) | Menos VRAM y un knob de campo receptivo independiente de la base | M |
| `symm_transpose` con una sola evaluación (pesos simetrizados) | ~2× en el readout de grafeno | M |
| MACE como extractor de features: sin cabezas de energía, reutilizando SH y radial | También evita el fallo de DDP | M |
| Plan B si fp32 no cabe: checkpointing del camino de aristas | — | S |

## 6. Fase 4 — Condicional o posterior

- **4.1 Aristas KB en el grafo de H**, si 0.3 lo justifica. Envolvente geométrica R_i+R_j+2R_KB,
  igual en entrenamiento e inferencia.
- **4.2 Pérdida espectral de primer orden** con autovectores precalculados.
- **4.3 Envolvente y cutoff por par de especies** (solo importa con varias especies).
- **4.4 Caché offline y lectura CSR vectorizada**, si el perfilado lo pide.
- **4.5 Más adelante:** batching dinámico, readout SO(2), cuEquivariance, spin y SOC.

## 7. Orden y puntos de decisión

1. Fase 0 y Fase 1 en paralelo.
2. Ronda 1: 2.1 + 2.2 + 2.6 + 3.1 + 3.2 + 3.3 → benchmark "receta v2". Si `6x6` cabe en fp32,
   repetir con fp32 completo.
3. Ronda 2: 2.3 y 2.4 → benchmark.
4. Ronda 3: 2.5 y la repetición de las ablaciones → benchmark.
5. Después, el resto de la Fase 3 (verificación numérica) y la Fase 4 según las mediciones.

**No hacer todavía:** agrandar MACE, `torch.compile` o kernels especializados, readout SO(2),
pérdida espectral antes de resolver bf16 y el gauge.

**Riesgos:**
- mace-torch 0.3.15 fija e3nn 0.4.4: cualquier API nueva debe existir en esas versiones.
- Los cambios de valores por defecto rompen la comparabilidad con resultados anteriores:
  documentarlos.

## 8. Estimación de tiempo

La implementación la hace Claude. El cuello de botella es la GPU y tu revisión, no escribir
código.

| Bloque | Implementación (Claude) | GPU (`w90`, 2 semillas) |
|---|---|---|
| Línea base `w90` en el HEAD actual | — | ~1,2 h |
| Fase 0 (diagnóstico sobre checkpoints existentes) | ~1–2 h | Minutos (inferencia) |
| Fase 1 (bugs, tests, CI) | ~3–5 h | — (tests en CPU) |
| Ronda 1 (2.1, 2.2, 2.6, 3.1–3.3) | ~2–3 h | ~1,2 h por variante |
| Ronda 2 (2.3, 2.4) | ~3–4 h | ~1,2 h por variante |
| Ronda 3 (2.5) | ~2–3 h | ~1,2 h por variante |
| Resto de la Fase 3 (grafo de MP, `symm_transpose`, extractor MACE) | ~4–6 h | Minutos (equivalencia numérica y VRAM) |
| Confirmación final (5 semillas, solo la combinación ganadora) | — | ~3 h |

**Total: ~15–25 h de trabajo de Claude y ~10–15 h de GPU** (8–10 variantes de ~1,2 h más la
confirmación). En la práctica, unos 3–5 días de sesiones. Los entrenamientos pueden correr por
la noche o mientras se implementa la ronda siguiente.
