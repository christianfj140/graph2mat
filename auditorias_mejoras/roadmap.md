# Roadmap completo de mejoras de graph2mat (rama g2m-v2)

Fecha: 2026-09-27. Recoge **todas** las propuestas de las 7 auditorías de `auditorias_mejoras/`:

- la de Claude, con sus 217 hallazgos (cada ID está asignado a un punto en el apéndice);
- las 6 de GPT (Astra, Sol, Computational Material Physics, Maths and Physics, ML, Scholar), que
  coinciden casi siempre con la de Claude. Lo que solo aparece en ellas va marcado **(GPT)**.

Sustituye a la "Fase 5" de `plan_mejoras.md`. Ese fichero queda como historial de las fases 0–4.

## Resumen de resultados (noche del 27 al 28 de septiembre)

Todas las cifras son H-MAE en meV, con 3 semillas salvo indicación.

**Precisión en `6x6` (N=8, receta de referencia 1,94):**
- bloques con término lineal (`E3nnLinearTSQNodeBlock` + `E3nnLinearEdgeBlock`), con poly3:
  - bf16: 1,80;
  - fp32: **1,24** (5 semillas, −36 %);
- con 16.000 pasos sin EarlyStopping: bf16 **1,12** (−42 %), fp32 **1,08** (−44 %);
- quitar el EarlyStopping de la receta (paciencia 80) da por sí solo −13 % (1,69), sin coste;
- el suelo de ruido de las etiquetas de SIESTA es ≈0,002 meV: queda margen.

**Extrapolación:**
- bicapa → registros de moiré no vistos: 20,95 → **6,92** con fp32 + lin (−67 %), bandas alineadas
  69,6 → 14,5;
- monocapa (`6x6`) → bicapa: 78 → **61** con lin bf16 + 16.000 pasos; poly6 la empeora (≈125–130);
- monocapa (`w90`, celda de 2 átomos) → bicapa: no funciona con ninguna variante (150–450).

**Velocidad y VRAM (`6x6`):**
- `torch_compile`: paso 2× más rápido, −25 % del tiempo total, misma precisión, −28 % de pico en bf16;
- nodo proyectado: 2,3× menos tiempo de GPU, pero pierde precisión (bf16: 2,32 meV); descartado para precisión;
- `lean_mace` (ya por defecto): −14 % de pico;
- checkpointing: −24 % (fp32) / −34 % (bf16) de pico del proceso, +34 % de tiempo.

**Recetas recomendadas:**
- `6x6` y extrapolación: bloques lin + poly3 + 16.000 pasos sin EarlyStopping, en bf16 (más `torch_compile`
  para compensar el tiempo);
- solo precisión dentro de distribución: lo mismo en fp32;
- bicapa → moiré: fp32 + lin (+ `eh16`), 6,2–6,9 frente a 20,95;
- `w90`: fp32 + poly6 + lin (5,09 frente a 12,6).

**Descartado:** `nodemix`, `hro` (no cabe), `bessel16`, `eh16` dentro de distribución, EMA, `mp_cutoff`,
`avg_num_neighbors`, cuEquivariance (memoria).

## Cómo leerlo

- **Gana:** P = precisión, V = VRAM, T = tiempo, C = corrección (sin ella el resultado está mal o falla).
- **Esfuerzo:** S < 1 h, M = 1–4 h, L = más de un día, validación incluida.
- **Riesgo:** probabilidad de romper algo o de invertir sin retorno.
- **Estado:** ✅ hecho · 🔶 parcial u opción implementada · 🧪 midiéndose · ⏳ pendiente · ❌ descartado,
  con la prueba que lo descarta.

## Reglas del proceso

1. **Opción desactivada por defecto, con test.** Todo cambio de modelo va detrás de un flag. El test
   comprueba la equivarianza y que, con el flag apagado, la salida no cambia. Un cambio pasa a ser
   el valor por defecto solo cuando gana.
2. **Medición.** `w90` con 3 semillas, emparejadas con fp32 + poly6 (12,6 meV de H-MAE en el test
   común; bandas 18,9 meV). Una tanda de 3 jobs tarda unos 40 min y caben 2 variantes a la vez
   (máximo 6 jobs, por temperatura).
3. **Criterio.** Una mejora cuenta si supera el ruido entre semillas (2–5 meV en `w90`). Las que
   ganen se confirman con 5 semillas y en `6x6` (N=8, 8000 pasos, referencia 1,94 meV).
4. **Orden.** Primero lo que más gana por unidad de esfuerzo y riesgo. Los cambios que dependen de
   otro van en bloque detrás de él (lista de dependencias más abajo).
5. **Sin GPU en paralelo.** Mientras la GPU mide una ola, se implementa lo que no necesita GPU (Ola 3).

## Orden de ejecución y puntos de decisión

| Paso | Qué | GPU | Estado |
|---|---|---|---|
| A | Medir la Ola 1 en `w90` (readouts, pérdida, radiales) | sí | 🧪 en marcha |
| A' | Medir la Ola 2 (MACE ligero, checkpointing): sonda de VRAM de `6x6` y perfil de un paso (2.9) | poca | ⏳ en cuanto haya hueco |
| B | Ola 3 completa (fallos sin GPU), en lotes con tests, mientras corre A | no | ⏳ |
| **D1** | **Decisión tras A.** Si algún readout gana (1.6/1.7/1.9/1.10): combinar los ganadores, confirmar con 5 semillas y `6x6`, y pasar la 7.1 por delante de la Ola 4. Si ninguno gana, el cuello de botella no está en el readout: seguir con las Olas 4 y 5 | — | — |
| C | Ola 4: paquete S (4.1 primero; el resto depende de ella) | sí | ⏳ |
| D | Ola 5: el objetivo (Δ-learning, normalización, entrenar en irreps) | sí | ⏳ |
| **D2** | **Decisión tras 2.9.** Si `w90` está limitado por lanzamientos y sincronizaciones: 6.3 → 6.8. Si lo limitan los datos: 6.1/6.2. Si lo limitan los FLOPs de aristas: 6.5/6.7/7.1 | — | — |
| E | Ola 6 según D2 | sí | ⏳ |
| F | Ola 7 según D1 | sí | ⏳ |
| G | Ola 8 cuando haya datasets que lo necesiten (spin, elementos pesados, sistemas polares) | sí | ⏳ |

### Dependencias (van juntos o en este orden)

- **4.1 (leer S) antes que 4.2, 4.3, 4.4 y 4.5.** Todos necesitan S. 8.1 (spin) comparte el mismo
  código de lectura de columnas.
- **6.3 (sin sincronizaciones, formas estables) antes que 6.8 (`torch.compile`/CUDA graphs).**
  Compilar antes da recompilaciones o grafos partidos. 6.6 también se apoya en 6.3.
- **5.3 (entrenar en irreps) hace innecesaria gran parte de 6.5** durante el entrenamiento: probar
  5.3 primero.
- **7.1 (TP de rango bajo o canal a canal) resuelve también el coste mul² de PERF_GPU-10.**
- **5.2 generaliza 1.5.** Como 1.5 fue neutro, la expectativa de 5.2 es baja.
- **2.7 (grafo corto de MACE) y 8.2 (aristas KB largas)** son dos caras del mismo desacoplamiento
  de grafos: si se hace 8.2, reutilizar el mecanismo de 2.7.
- **1.1 (fp32) antes que cualquier AMP/TF32** (6.8/6.9). La precisión reducida solo tiene sentido
  con islas fp32 en readout, cambio de base y pérdida (GPT: "después de normalizar el objetivo").

## Extrapolación a otras estructuras (monocapa → bicapa)

Script: `MD_vs_AtomicDisplacement/Comparison/scripts/g2m_ood_eval.py`. Checkpoints entrenados en
monocapa (`w90`) predicen:

- los 144 registros laterales de una bicapa de 4 átomos (`tbg_registry_grid_12x12`: AA, AB, SP y
  todo lo intermedio, es decir, los entornos locales de un moiré);
- los 20 frames de test de la MD de bicapa AB.

El error en Γ se separa en intracapa e intercapa, con bandas Γ-K-M-Γ (±2 eV) brutas y
alineadas (restando el desplazamiento medio).

| Variante (`w90`, 3 semillas) | H-MAE | intracapa | intercapa | bandas alineadas | Por semilla (H / bandas alineadas) | Dentro de distribución |
|---|---:|---:|---:|---:|---|---|
| `eh16` (readout de aristas 16x) | **153** | 232 | 74 | **152** | 143/124 · 100/166 · 216/167 | neutro |
| `mp4` | 164 | 250 | 78 | 217 | 182/303 · 158/209 · 153/138 | neutro en H, peor en bandas |
| `mp3` | 174 | 280 | 68 | 474 | — | peor |
| poly6 (bf16) | 183 | 297 | 69 | 331 | 209/332 · 138/499 · 203/162 | mejor |
| `lin` (fp32 + poly6) | 221 | 326 | 116 | 480 | 179/380 · 253/348 · 232/710 | **−60 %** |
| línea base (bf16, poly3) | 230 | 396 | 64 | 282 | 169/331 · 213/207 · 307/308 | — |
| fp32 + poly6 | 289 | 515 | 63 | 497 | 415/599 · 119/484 · 332/408 | mejor |
| fp32 | 309 | 553 | 66 | 435 | — | mejor |
| `kl` / `ss` / `mid16` | 358 / 373 / 452 | 629–814 | 86–104 | 404–534 | — | variable |

(Registros; los 20 frames de MD AB dan la misma ordenación.)

Conclusiones, con cautela porque la dispersión entre semillas es enorme (fp32 + poly6: de 119 a 415 meV):

1. **Monocapa → bicapa no funciona con ninguna variante**: 150–450 meV, frente a 5–25 meV dentro de
   distribución. Ver átomos a 3,35 Å fuera del plano, que nunca vio, corrompe también los bloques
   intracapa. El c* de gauge no lo explica: no es un desplazamiento del cero de energía.
2. **Lo que mejora dentro de distribución no mejora fuera.** fp32 empeora la extrapolación (ajusta
   detalles propios de la monocapa) y `lin` tampoco la mejora.
3. **`eh16` es lo único consistentemente mejor fuera de distribución** (bandas alineadas 124–167 en las
   tres semillas, frente a 207–331 de la línea base), aunque dentro de distribución sea neutro.
   Limitar el alcance de MACE (`mp4`) ayuda algo.
4. **Entrenar con la supercelda 6x6 extrapola mucho mejor** que con la celda de 2 átomos de `w90`:
   la referencia de `6x6` (N=8, bf16) da 78 meV (82 intracapa, 74 intercapa, bandas alineadas 162)
   en los registros, frente a 150–450 meV de `w90`. Otra vez fp32 empeora la extrapolación
   (fp32 + poly6 + kl en `6x6`: 161 meV). Hipótesis: el ruido de bf16 actúa como regularización.
   **`lin` en `6x6` (bf16) mejora también fuera de distribución**: 63 meV frente a 78 (intracapa 63
   frente a 82; bandas alineadas 100 frente a 162). Es la primera variante que mejora dentro y fuera.
   `ew2` en `w90` no ayuda fuera (271 meV). 16.000 pasos sin EarlyStopping en `6x6` no perjudica la extrapolación (73 frente a 78 meV; intracapa 70 frente a 82).
6. **En `6x6` lo que empeora la extrapolación es poly6, no fp32.** bf16 + poly6 + lin: 125 meV
   (121/147/106 por semilla); fp32 + poly6 + lin: 130; lin con poly3: 63 (60/77/51). Hipótesis: la
   envolvente p=6 mantiene peso en los pares de 3,35–5 Å, que en la bicapa son los átomos de la otra
   capa, nunca vistos (encaja con que limitar el alcance de MACE ayude). Dentro de distribución, en
   cambio, poly6 ayuda: bf16 + poly6 + lin 1,70 (1,44/2,01/1,65); fp32 + poly6 + lin 1,51. Hay dos
   recetas según el objetivo; 🧪 fp32 + lin con poly3, lin con 16.000 pasos y lin proyectado con poly3.
7. **Receta ganadora en `6x6`: fp32 + lin con poly3.** Dentro: **1,056/1,469/1,163 → 1,23 meV (−37 %
   frente a 1,94; Δ −0,91/−0,54/−0,68)**. En bicapa: **72 meV (69/93/54), bandas alineadas 107**
   (referencia 78 y 162). Mejora dentro y fuera: lo que perjudicaba la extrapolación era poly6, no fp32.
   Con 16.000 pasos y sin EarlyStopping: **1,024/1,270/0,954 → 1,08 meV (−44 %)** y 69 meV en bicapa
   (bandas alineadas 116). **lin en bf16 con 16.000 pasos: 0,949/1,335/1,083 → 1,12 meV (−42 %) y 61 meV en
   bicapa (bandas alineadas 94): la mejor extrapolación de todas** y casi la misma precisión que fp32.
   **Receta recomendada para `6x6`: bloques lin + poly3 + 16.000 pasos sin EarlyStopping, en bf16** (la
   más barata y la que mejor extrapola); fp32 si solo importa la precisión dentro de distribución
   (1,08 frente a 1,12).
   Nodo proyectado (fp32, 8000 pasos): 1,09/1,022/2,134 → 1,42 meV (−27 %; frente a lin Δ −0,03/−0,45/+0,97):
   no concluyente y con más varianza. En bf16 es claramente peor: 2,509/1,924/2,530 → 2,32 meV (+20 % frente a
   la referencia). La proyección a 16 canales pierde precisión: ❌ para precisión; como opción de velocidad solo
   en fp32 y con cautela (siguiente paso posible: proyección a 32 canales).
   Nodo proyectado fuera de distribución: 71 meV (lin 72).
   `torch_compile` en entrenamiento real (fp32 + lin, semilla 0): 1,057 meV frente a 1,056 (misma precisión)
   en 30,3 frente a 40,3 min (−25 %, compilación incluida; carga de la máquina no controlada). Requiere que
   el PATH no tenga entradas inaccesibles (torch 2.11 ejecuta `nvcc --version` en cada compilación);
   `g2m_autorun` ya las filtra. En bf16 (lin, 3 semillas): 1,239/1,524/1,432 → 1,40 meV (sin compilar 1,80;
   diferencia dentro del ruido) y **pico del proceso 5520 → ~4000 MiB (−28 %)**.
   **5 semillas de fp32 + lin (8000 pasos): 1,056/1,469/1,163/1,150/1,380 → 1,24 meV (−36 %)**; las cinco
   por debajo de la mejor semilla de la referencia (1,839).
8. En `w90`, fp32 + lin con poly3: 7,32 meV dentro (4,12/11,36/6,48; bandas 16,6) y 229 meV en bicapa: en
   `w90` poly6 sí ayuda dentro de distribución (fp32 + poly6 + lin 5,09) y la extrapolación desde la celda
   de 2 átomos no mejora con ninguna receta.
9. En `w90`, bf16 + poly6 + lin: 7,55 meV dentro (8,40/8,07/6,19; fp32 + poly6 + lin 5,09) y 190 meV
   en bicapa (fp32 + poly6 + lin 221, poly6 solo 183). En `w90` fp32 sí aporta dentro de distribución.
5. **Escenario realista, bicapa → moiré** (`g2m_bilayer_benchmark.py`: entrenar con la MD de AA/AB/BA,
   n474, 8000 pasos; predecir los registros de `tbg_registry_grid_12x12` no vistos en el entrenamiento):

   | Receta (3 semillas) | Validación MD (meV) | Registros no vistos | intra / inter | Bandas brutas / alineadas |
   |---|---:|---:|---:|---:|
   | Referencia (bf16, poly3, bloques base) | 21,85 | 20,95 | 22,9 / 19,0 | 92,2 / 69,6 |
   | **fp32 + lin (poly3)** | **6,45 (−70 %)** | **6,92 (−67 %)** | 8,8 / 5,1 | 27,9 / **14,5 (−79 %)** |

   Por semilla, registros no vistos: referencia 21,98/17,65/23,23; fp32 + lin 6,64/7,71/6,39. Entrenando
   con bicapa, el error en entornos de moiré no vistos es igual al de dentro de distribución, y la receta
   ganadora lo divide por tres. Con `eh16` además (fp32 + lin + eh16): validación 6,57, registros no vistos
   6,20 (4,36/6,13/8,11), bandas 20,9/13,2: ligera mejora fuera de distribución, dentro del ruido.
   lin en bf16: validación 9,47, registros no vistos 8,42, bandas 31,1/14,8. Entrenando con bicapa, fp32 sí
   compensa (y no perjudica la extrapolación): **para bicapa → moiré, fp32 + lin (+ `eh16` opcional)**.

## Ola 0 — Red de seguridad y fallos críticos

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 0.1 | Emparejamiento de aristas en modo no simétrico: heteronuclear (reordenación) y homonuclear (m_ij, m_ij) | C | S | bajo | ✅ |
| 0.2 | `symmetric_matrix` deducido del tipo de matriz (H/DM/EDM son simétricas) | C V T | S | bajo | ✅ |
| 0.3 | `normalized_density_error`: re-sumaba la DM atómica; signo on-site/off-site | C | S | bajo | ✅ |
| 0.4 | Métricas mal definidas o duplicadas (`mapestdmix`, `mapemaemix`, `mapemsemix`) | C | S | bajo | ✅ |
| 0.5 | CI que ejecuta pytest | C | S | bajo | ✅ `.github/workflows/tests.yaml` |
| 0.6 | Tests: equivariancia con rotación genérica, localidad de bloques, reordenación, asserts que siempre pasan | C | M | bajo | 🔶 faltan ENG_QUALITY-04 y los tests del modelo completo con supercelda |
| 0.7 | Carga con torch ≥ 2.6 (e3nn, checkpoints) | C | S | bajo | 🔶 falta el KeyError con variables `E3MAT_*` |
| 0.8 | `initial_node_feats` ignorado; rama `version != "new"` rota | C | S | bajo | 🔶 queda borrar la rama muerta |
| 0.9 | Batch sin aristas o sin nodos con base | C | S | bajo | ✅ |
| 0.10 | Benchmark reproducible, perfilado y regresión numérica | — | M | bajo | 🔶 existe `g2m_improvements_benchmark.py` (`w90`/`6x6`); falta el perfilado (2.9) |

## Ola 1 — Precisión barata (configuración u opciones pequeñas)

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 1.1 | fp32 en vez de bf16 (y `readout_fp32` para AMP solo en MACE) | P | S | bajo | ✅ `w90` 24,5 → 15,0 meV; `readout_fp32` no ayuda |
| 1.2 | Envolvente `num_polynomial_cutoff=6` (p=3 anulaba los pares a 4–5 Å) | P | S | bajo | ✅ junto con fp32: 13,0 meV (5 semillas), bandas 19,7 |
| 1.3 | `avg_num_neighbors` | P | S | bajo | ❌ 34,2 empeora en `w90` (25,2 frente a 24,5) |
| 1.4 | Última capa de MACE con l>0 (`keep_last_layer_irreps`) | P | S | bajo | 🔶 opción. `w90`: −1,2 meV en H (ruido), −3,4 en bandas. `6x6`: 2,01 frente a 1,94 (sin mejora), ×2 en tiempo y VRAM |
| 1.5 | Escala/desplazamiento equivariante de la salida (`output_scaleshift`) | P | S | bajo | 🔶 opción; ≈0 en `w90` |
| 1.6 | Readout de aristas más ancho (`edge_hidden_irreps` 16x; por defecto son 4 canales por l) | P | config | bajo | 🔶 dentro de distribución neutro (12,63 frente a 12,61; bandas 20,2 frente a 18,9), pero **el mejor fuera de distribución** (monocapa → bicapa, bandas alineadas 152 frente a 282 de la línea base) |
| 1.7 | Término lineal junto al producto tensorial, más sesgo 0e on-site (`E3nnLinearTSQNodeBlock`, `E3nnLinearEdgeBlock`) | P | S | bajo | ✅ **`w90`: 12,61 → 5,09 meV (−60 %; 4,92/5,57/4,78), bandas 18,9 → 13,5.** Misma VRAM. **`6x6`** (receta de referencia bf16 + lin): 1,23/2,58/1,58 → media 1,79 frente a 1,94 (Δ −0,73/+0,57/−0,26; −7 %, ruidoso); **fp32 + poly6 + lin en `6x6`: 1,19/1,58/1,74 → media 1,51 frente a 1,94 (−22 %; Δ −0,77/−0,43/−0,10)**. Pero fuera de distribución (bicapa) empeora: 130 frente a 78 meV, mientras que lin en bf16 da 63. Compromiso: fp32 mejora dentro y empeora fuera (en `w90` igual: fp32 solo 309, poly6 solo 183, línea base 230). 🧪 bf16 + poly6 + lin |
| 1.8 | Peso de las aristas en la pérdida (`loss_kwargs: {edge_weight: 2}` = MSE sobre la matriz completa) | P | S | bajo | 🔶 `w90`: 11,29 frente a 12,61 meV (Δ por semilla −4,2/−0,3/+0,6), bandas 18,2 frente a 18,9: dentro del ruido. Opción disponible (`loss_kwargs`) |
| 1.9 | `E3nnEdgeBlockNodeMix`: pesos del TP que dependen de la distancia y de ambos nodos (necesita `preprocessing_edges_reuse_nodes`) | P | config | bajo | ❌ `w90`: 11,77 frente a 12,61 meV (Δ +1,0/−1,8/−1,7), bandas 19,5 frente a 18,9: pequeño, dentro del ruido; **y fuera de distribución es la peor variante (557 meV en bicapa)** |
| 1.10 | Readouts condicionados del fork (`E3nnHamiltonian{Node,Edge}Block`). Vigilar su VRAM **(GPT Astra)** | P | config | medio | ❌ con esta receta falla al construirse: intenta reservar 139 GB de RAM (el MLP que genera los pesos del TP recibe las features de MACE, 48 canales × 9 segmentos). Solo sería viable proyectando antes las features de nodo a pocos canales |
| 1.11 | Resolución radial `num_bessel=16` | P | config | bajo | ❌ `w90`: 17,0 frente a 12,61 meV (15,7/18,8/16,4; bandas 25,9 frente a 18,9): peor |
| 1.12 | EMA, SWA/segunda etapa, clipping, warmup | P | S | bajo | 🔶 EMA empeora (+2,8 meV); SWA y clipping sin probar |
| 1.13 | `correlation`, `lmax`, número de interacciones; defaults de `LitMACEMatrixModel` para H | P | config | bajo | ✅ la receta ya usa corr 3, lmax 3 y 3 interacciones. Los defaults se cambian al cerrar la Ola 1 |
| 1.14 | Irreps de salida inalcanzables (se predecían como cero sin avisar) | C | S | bajo | 🔶 ahora avisa; la solución es subir lmax |
| 1.15 | Grupos de weight decay; selección del checkpoint dependiente de la pérdida | P | S | bajo | ⏳ |

## Ola 2 — VRAM y velocidad baratas

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 2.1 | Proyectar las features de nodo antes de copiarlas a cada arista (`edge_message_mid_irreps`) | V | S | bajo | 🔶 `6x6` fp32: 11,4 → 8,4 GB; `w90` +0,7 meV |
| 2.2 | MACE ligero (`lean_mace`): sin readouts de energía ni gradientes respecto a posiciones | T V | S | bajo | 🔶 hecho y testado (misma salida y gradientes); falta medir |
| 2.3 | Gradient checkpointing de interacciones y readout (`gradient_checkpointing`) | V | S | bajo | ✅ opción. Sonda `6x6` (N=16, batch 16, pico del proceso): fp32 10008 → 7654 MiB (−24 %), bf16 6862 → 4516 MiB (−34 %); microbenchmark: +34 % de tiempo |
| 2.4 | `symm_transpose` con una llamada (las dos direcciones apiladas) | T | S | bajo | ✅ activo por defecto, mismo resultado (test). Reducir los FLOPs a la mitad es la 6.7 |
| 2.5 | DataLoader: `pin_memory` y workers persistentes | T | S | bajo | ✅ |
| 2.6 | Máscara de NaN solo cuando las etiquetas los tienen (H no) | T | S | bajo | ⏳ aplazado a 6.3: saber si hay NaN ya es una sincronización |
| 2.7 | Grafo corto para el message passing de MACE (`mp_cutoff`) | V T | M | medio | 🔶 opción; en grafeno empeora (`mp3` +4,6 meV; `mp4` +20 meV en bandas) |
| 2.8 | Irreps concatenadas sin simplificar; coste y media de `readout_per_interaction` | T | S | bajo | 🔶 opción `simplify_node_irreps` (18f4ca2; test: menos instrucciones); falta medir. `readout_per_interaction` pendiente |
| 2.9 | Perfilado de un paso de `w90` y de `6x6` | — | S | bajo | 🔶 `w90` (batch 32, fp32): **6009 kernels/paso** y 20 sincronizaciones; GPU ocupada ~14,7 ms de ~60 ms/paso (medido con carga): **limitado por lanzamientos de kernels**. `lean_mace`: −10 % kernels, −10 % pico; `simplify_node_irreps`: −23 % kernels, pero +43 % pico; ambos: −33 % kernels; checkpointing: pico −53 % (591 → 278 MiB) con +80 % de tiempo. `6x6` bf16 (8 estructuras): **limitado por la GPU** (137 de 171 ms ocupada): `aten::copy_` 24 % (casts de autocast), matmul 29 %, mul/add/sum ~30 %. `lean_mace` −3 % tiempo y **−14 % pico** (2968 → 2542 MiB); `simplify_node_irreps` +8 % tiempo de GPU y +41 % pico (solo compensa en sistemas pequeños); checkpointing **−44 % pico** (1429 MiB) con +34 % tiempo. Reparto del tiempo en `6x6` (fwd+bwd): MACE 66 ms, mensajes de arista 38, **bloques de nodo 36** (tensor square sobre las 1584 dimensiones de MACE), bloques de arista 16. → `E3nnProjectedLinearTSQNodeBlock` (tensor square sobre una proyección a 16 canales; el término lineal ve todo): **GPU 126 → 54 ms/paso (2,3×), pico 2546 → 2121 MiB**; con `edge_message_mid_irreps=16`, 40 ms y 1556 MiB (−39 %). Precisión 🧪 (`fp32poly6_linp`, `poly6_linp`). 6.7 (simetrizar pesos) descartado: los bloques de arista son solo 16 ms. Reparto `w90` (kernels/paso): forward de MACE 853, readout 764 (tensor square del nodo C 304 con 63 instrucciones sobre 9 segmentos, mensajes de arista 236, bloques de arista 20–83), pérdida 26, backward ~3700, optimizador 11. Falta `6x6` y los tiempos sin carga |
| 2.10 | **(GPT Astra)** La ruta de coeficientes evaluaba el readout dos veces | T | S | bajo | ✅ `forward_with_coefficients` |

## Ola 3 — Fallos de ingeniería y métricas (sin GPU)

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 3.1 | Agregación de métricas por época (media de RMSE por batch, media de máximos) y `batch_size` en `self.log` | C | S | bajo | 🔶 ponderado por estructuras y máximo de máximos (commit 28d9d9e); falta el RMSE global acumulado |
| 3.2 | Pérdidas MAPE/mix con H: división por cero → NaN | C | S | bajo | ✅ se enmascara antes de dividir; test de gradientes finitos (4239aa1) |
| 3.3 | `SamplewiseMetricsLogger` (fuga de ficheros); `MatrixWriter` (escribe en cada paso, H sin S) | C T | S | bajo | ✅ un fichero por split, sin sanity check, solo rango 0; `MatrixWriter` solo en `predict` por defecto (48897b2). Adjuntar S a H: ver 4.1 |
| 3.4 | Servidor (`/test` roto, `/predict`), `local_write_predict`, extrapolación, `PlotMatrixError` con `neigh_isc` | C | S | bajo | ⏳ |
| 3.5 | `import graph2mat` sin torch; dependencias de la CLI; versiones fijadas | C | S | bajo | ⏳ |
| 3.6 | DataModule: `copy_root_to_tmp`, `setup` carga todos los splits, carrera de zipfile, filtros de labels descartados, `torch_predict`, `node_types_subgraph` | C | S | bajo | 🔶 `torch_predict` y `copy_root_to_tmp` (2778903); el resto pendiente |
| 3.7 | RTPs: caché vacía, dtype congelado, RTP lento en cada llamada | T | S | bajo | ⏳ |
| 3.8 | `reuse_nodes` con `E3nnSimpleEdgeBlock`, `E3nnInteraction` sin residual, features experimentales no equivariantes, robustez de `PointBasis`, EDM en el registro de matrices | C | S | bajo | 🔶 entradas por nombre con `reuse_nodes` (5cdc3e0); paridad de `PointBasis` (2b32072); faltan residual, EDM y features experimentales |
| 3.9 | Linter y tipos, tests con semilla, reproducibilidad del split, tests de pérdidas y de simetría H_ij = H_jiᵀ | C | M | bajo | 🔶 asserts que siempre pasaban y test M = Mᵀ (3cc3aa3); faltan linter y semillas |
| 3.10 | Validar la convención de orbitales (l creciente, m = −l..l); `qe_spherical` incorrecta para l ≥ 2 | C | M | medio | 🔶 error si las capas no están completas y en l creciente (dade02d); el orden de m depende de `basis_convention`; `qe_spherical` pendiente |
| 3.11 | Supercelda: `isc` mal asignado sin aviso, coherencia grafo–`isc`, `pbc` por defecto, `ncol` de sisl, fichero fuente implícito, geometría XV frente a la de la etiqueta | C | M | medio | 🔶 error si un desplazamiento supera nsc//2 (d9075c8; ningún dataset del benchmark lo dispara). Pendiente: `sc_off` no estándar (CRIT-04), XV frente a TSHS |
| 3.12 | Splits por trayectoria de MD y provenance **(GPT Maths and Physics, Astra)** | C | S | bajo | ⏳ (sobre todo en el benchmark) |
| 3.13 | Multi-GPU: parámetros sin gradiente, métricas no sincronizadas | C | M | medio | ⏳ no verificable con una sola GPU |
| 3.14 | **(GPT Astra)** Incompatibilidades de las pérdidas de coeficientes | C | S | bajo | ⏳ |

## Ola 4 — Paquete S (4.1 primero; el resto depende de ella)

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 4.1 | Leer S junto a las etiquetas; reconstruir matrices no ortogonales; el servidor escribe el TSHS con S real | C P | M | medio | 🔶 `matrix_component_policy="h_and_overlap"` del fork ya la lee; falta la reconstrucción y la salida |
| 4.2 | Pérdida invariante de gauge (c por estructura en forma cerrada) y métrica c* | P | M | medio | 🔶 métrica: `g2m_ood_eval.py` calcula c* por mínimos cuadrados y el error alineado. Pérdida: ✅ `gauge_invariant_mse` (7b97b7a): con etiquetas (H, S) (`h_and_overlap`, S la última componente), c óptimo por estructura en forma cerrada y MSE de ΔH − c·S más MSE de S; test con un desplazamiento distinto por estructura. Falta probarla en el benchmark: la predicción de `MD_vs_AtomicDisplacement` asume `h_only` y la escritura de H con S del fork tiene un test que ya fallaba |
| 4.3 | Métricas espectrales en validación y desglose del error por especie, par, l y distancia | — | M | bajo | ⏳ el benchmark ya calcula bandas por fuera |
| 4.4 | Pérdidas físicas con S: N_e, Mulliken, autovalores de primer orden | P | M/L | medio | ⏳ |
| 4.5 | DM/EDM: Tr(DS) = N, McWeeny, EDM ligada a DM y H | P C | M | medio | ⏳ |
| 4.6 | Ceros estructurales y máscara orbital en vez de bloque atómico denso **(GPT CMP, Maths and Physics, ML)** | P | S/M | bajo | ⏳ |

## Ola 5 — El objetivo: residuo y normalización

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 5.1 | Referencia on-site por especie → Δ-learning sobre H0 (atómico o Slater–Koster; H0 de SIESTA como versión completa) | P | M → L | medio | ⏳ las 7 auditorías lo ponen entre lo de más retorno |
| 5.2 | Normalización equivariante por especie, par e irrep | P | M | bajo | ⏳ expectativa baja (1.5 fue neutro) |
| 5.3 | Entrenar en el espacio de irreps (Q ortonormal: pérdida sin cambio de base) | P T V | M | medio | ⏳ |
| 5.4 | Techo de ruido de las etiquetas y tiempo hasta una precisión objetivo **(GPT Astra)** | — | S | bajo | ✅ **suelo de ruido de SIESTA en `6x6` ≈ 0,002 meV** (SCF 1e-4 frente a 1e-6: 0,0015; traslación rígida bajo la malla: 0,0011; ambas: 0,0022; 4 estructuras, `g2m_label_noise.py`). El modelo (1,94 meV) está ~1000× por encima: **6x6 no está limitado por las etiquetas**, hay margen real. **Y en parte está limitado por la optimización**: la misma receta con 16.000 pasos y sin EarlyStopping da 1,716/1,933/1,490 → **1,71 meV (−12 %, las tres semillas mejoran: −0,25/−0,08/−0,35)**. El EarlyStopping de la receta (paciencia 80 validaciones) paraba algunas semillas a 1/3 del coseno. **`noes` (8000 pasos sin EarlyStopping): 1,684/1,575/1,805 → 1,69 meV (−13 %)**: con la receta de referencia, toda la ganancia de «16.000 pasos» viene de quitar el EarlyStopping (cambio gratis). Con lin no está separado (lin bf16 8000 pasos con EarlyStopping 1,80; 16.000 sin él 1,12) |

## Ola 6 — Rendimiento en profundidad (según el perfil 2.9)

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 6.1 | Caché del preprocesado en tensores (`store_in_memory` ya evita reparsear) | T | M | bajo | ⏳ |
| 6.2 | Vectorizar CSR → bloques y la reconstrucción; lista de vecinos | T | M | bajo | ⏳ |
| 6.3 | Aristas ordenadas por tipo en el collate y targets reordenados: sin `nonzero` ni sincronizaciones en el forward | T | M/L | medio | ⏳ Medido: el proceso principal de cada entrenamiento de `6x6` gira al 100 % de un núcleo esperando a la GPU; con sincronización bloqueante (`CU_CTX_SCHED_BLOCKING_SYNC`) baja al 65 % con +5 % de tiempo: el resto es trabajo de Python y lanzamientos. Importa porque esta máquina está limitada por la temperatura de la CPU (2–4 entrenamientos a la vez por debajo de 80 °C) |
| 6.4 | Batching por coste (aristas y orbitales), no por número de estructuras | V | M | bajo | ⏳ |
| 6.5 | Cambio de base disperso por pares de capas | T | M | bajo | ⏳ ver 5.3 |
| 6.6 | Menos módulos pequeños por tipo; `basis_grouping` (`basis_shape`, **GPT ML**) | T | L | medio | ⏳ |
| 6.7 | `symm_transpose` simetrizando los pesos: mitad de FLOPs en aristas homonucleares | T V | M | medio | ❌ en `6x6` los bloques de arista son 16 de ~150 ms/paso: ahorraría ≤5 % |
| 6.8 | `torch.compile` y CUDA graphs (después de 6.3) | T | L | alto | 🔶 opción `torch_compile` (35d0172): modelo sin TorchScript de e3nn y, al empezar el fit, `compile_submodules()` compila en el sitio las capas de MACE y las operaciones del readout (formas dinámicas; mismas claves de checkpoint). **Paso de `6x6` bf16: 138 → 70 ms (2,0×), pico 2425 → 1685 MiB**, tras ~90 s de compilación; test GPU: mismas matrices y gradientes. Requirió dar permiso de ejecución a los binarios de Triton del venv (`triton/backends/nvidia/bin/*`). En `w90` (fp32, limitado por lanzamientos) solo 91 → 68 ms frente a e3nn sin TorchScript, con 160 s de compilación: no compensa en sistemas diminutos. Entrenamiento real `6x6` fp32 + lin: misma precisión (1,057 frente a 1,056) y −25 % de tiempo total. CUDA graphs: pendiente (necesita 6.3) |
| 6.9 | Kernels rápidos de TP (cuEquivariance, OpenEquivariance) | T V | L | alto | 🔶 opción `cueq` (backbone MACE, disposición `mul_ir`, convenciones e3nn): en `6x6` bf16 usa más de 6 GB frente a 2,5 GB de e3nn (repetido con Triton ya funcional) y en `w90` no terminó en 10 min. Aparcado: `torch_compile` da más con menos memoria |

## Ola 7 — Arquitectura del readout (según D1)

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 7.1 | TP de aristas canal a canal o de rango bajo, con pesos por par y gates no lineales | P V | L | medio | ⏳ |
| 7.2 | Capas de actualización de aristas | P | L | medio | ⏳ |
| 7.3 | Envolvente y radiales por par (R_i+R_j) | P | M | bajo | ⏳ solo cambia algo con varias especies |
| 7.4 | Readout en el marco local de la arista (SO(2)) o Slater–Koster neuronal más un residuo **(GPT CMP, Maths and Physics, Astra)** | P T | L | alto | ⏳ |
| 7.5 | Readout compartido entre especies | P | L | medio | ⏳ |
| 7.6 | Multi-tarea H + S + DM + EDM | P | L | medio | ⏳ |
| 7.7 | H = H_KS[DM predicha] con un paso de SIESTA | P | L | alto | ⏳ necesita SIESTA |

## Ola 8 — Cobertura física (necesita datasets que lo requieran)

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 8.1 | Spin colineal y SOC; referencia atómica coherente con spin | C P | L | medio | 🔶 `spin_h_only` del fork |
| 8.2 | Bloques de H mediados por proyectores KB (aristas de readout más largas) | P | L | medio | ❌ para grafeno (despreciable según la Fase 0); sí para elementos pesados |
| 8.3 | Tipos de punto por Z: dos especies con la misma Z y distinta base (ghosts) | C | M | bajo | ⏳ |
| 8.4 | **(GPT Astra, Maths and Physics)** Electrostática de largo alcance para sistemas polares | P | L | alto | ⏳ |

## Descartado

| # | Cambio | Motivo |
|---|---|---|
| X.1 | Esparsidad tipo SPHNet; Gaunt tensor product | Poco retorno con bases s/p/d; Gaunt pierde caminos antisimétricos |

## Apéndice — ID de la auditoría de Claude → punto del roadmap

Los 217 IDs, cada uno en un solo punto. Se comprueba con el script `roadmap_map.py` de la sesión.

| Punto | IDs |
|---|---|
| 0.1 | ARCH-01, CRIT-07, ARCH-02, MATH_EQUIV-01, PERF_GPU-01 |
| 0.2 | PHYS_H-13, ARCH-03, LOSS_TRAIN-18, PERF_GPU-07, SOTA-05 |
| 0.3 | PHYS_H-12, LOSS_TRAIN-02, ENG_QUALITY-05, LOSS_TRAIN-01 |
| 0.4 | LOSS_TRAIN-03, ENG_QUALITY-06, LOSS_TRAIN-04, ENG_QUALITY-07 |
| 0.5 | ENG_QUALITY-01 |
| 0.6 | ENG_QUALITY-02, ENG_QUALITY-03, MATH_EQUIV-09, ENG_QUALITY-27, ENG_QUALITY-04 |
| 0.7 | ENG_QUALITY-11, LOSS_TRAIN-10 |
| 0.8 | ENG_QUALITY-08, ARCH-27, ENG_QUALITY-09 |
| 0.9 | ARCH-25 |
| 0.10 | ENG_QUALITY-19 |
| 1.1 | LOSS_TRAIN-24, MATH_EQUIV-18, PERF_GPU-18 |
| 1.2 | ARCH-09, LOSS_TRAIN-19, DATA_PIPE-15, PHYS_H-07 |
| 1.3 | ARCH-08, PERF_GPU-23, DATA_PIPE-16, SOTA-08, ENG_QUALITY-10, LOSS_TRAIN-20 |
| 1.4 | ARCH-04, MATH_EQUIV-02, SOTA-09 |
| 1.5 | ARCH-13, SOTA-12, MATH_EQUIV-06 |
| 1.6 | ARCH-10, MATH_EQUIV-04 |
| 1.7 | ARCH-12 |
| 1.8 | LOSS_TRAIN-11 |
| 1.9 | ARCH-23, MATH_EQUIV-11, PERF_GPU-21 |
| 1.10 | SOTA-11 |
| 1.11 | CRIT-06 |
| 1.12 | SOTA-20, LOSS_TRAIN-21 |
| 1.13 | ARCH-07, SOTA-07, ARCH-05, ARCH-28 |
| 1.14 | ARCH-06, SOTA-10, MATH_EQUIV-05 |
| 1.15 | LOSS_TRAIN-22, LOSS_TRAIN-23 |
| 2.1 | ARCH-16, PERF_GPU-09 |
| 2.2 | ARCH-19, LOSS_TRAIN-27, PERF_GPU-12, ENG_QUALITY-22, SOTA-22 |
| 2.3 | PERF_GPU-20 |
| 2.4 | PERF_GPU-06 |
| 2.5 | DATA_PIPE-11, PERF_GPU-22 |
| 2.6 | LOSS_TRAIN-26, PERF_GPU-15 |
| 2.7 | ARCH-17, PERF_GPU-08, DATA_PIPE-17, SOTA-14 |
| 2.8 | PERF_GPU-14, ARCH-24, PERF_GPU-13 |
| 2.9 | PERF_GPU-24 |
| 3.1 | LOSS_TRAIN-06, ENG_QUALITY-23 |
| 3.2 | LOSS_TRAIN-05 |
| 3.3 | LOSS_TRAIN-08, LOSS_TRAIN-09 |
| 3.4 | PHYS_H-21, ENG_QUALITY-15, ENG_QUALITY-16, PHYS_H-22, PHYS_H-20 |
| 3.5 | ENG_QUALITY-12, ENG_QUALITY-13, ENG_QUALITY-14 |
| 3.6 | DATA_PIPE-21, ENG_QUALITY-24, DATA_PIPE-12, DATA_PIPE-09, DATA_PIPE-08, DATA_PIPE-24, DATA_PIPE-25 |
| 3.7 | MATH_EQUIV-20, MATH_EQUIV-21, ENG_QUALITY-17 |
| 3.8 | ARCH-22, ARCH-26, MATH_EQUIV-22, MATH_EQUIV-23, PHYS_H-19, ENG_QUALITY-25 |
| 3.9 | ENG_QUALITY-18, ENG_QUALITY-26, LOSS_TRAIN-25, LOSS_TRAIN-29, ENG_QUALITY-20 |
| 3.10 | MATH_EQUIV-03, PHYS_H-18, MATH_EQUIV-10 |
| 3.11 | PHYS_H-16, DATA_PIPE-19, CRIT-04, PHYS_H-17, DATA_PIPE-28, PHYS_H-15, CRIT-05 |
| 3.12 | ENG_QUALITY-21 |
| 3.13 | LOSS_TRAIN-07 |
| 4.1 | PHYS_H-02, MATH_EQUIV-16, SOTA-02, DATA_PIPE-18, CRIT-02 |
| 4.2 | PHYS_H-03, LOSS_TRAIN-13, SOTA-03, ARCH-14, CRIT-01 |
| 4.3 | LOSS_TRAIN-14, LOSS_TRAIN-16 |
| 4.4 | PHYS_H-09, LOSS_TRAIN-15 |
| 4.5 | PHYS_H-10, PHYS_H-11 |
| 4.6 | PHYS_H-08 |
| 5.1 | PHYS_H-05, SOTA-01, LOSS_TRAIN-12 |
| 5.2 | PHYS_H-06 |
| 5.3 | MATH_EQUIV-13, ARCH-21 |
| 5.4 | PHYS_H-14 |
| 6.1 | DATA_PIPE-05, DATA_PIPE-02, DATA_PIPE-22, DATA_PIPE-06, DATA_PIPE-13, PHYS_H-23 |
| 6.2 | DATA_PIPE-03, DATA_PIPE-04, DATA_PIPE-26, PHYS_H-24, DATA_PIPE-23, DATA_PIPE-14 |
| 6.3 | PERF_GPU-02, PERF_GPU-03, PERF_GPU-04, PERF_GPU-05, DATA_PIPE-07, DATA_PIPE-27 |
| 6.4 | DATA_PIPE-10, LOSS_TRAIN-28 |
| 6.5 | MATH_EQUIV-12, PERF_GPU-11, SOTA-16 |
| 6.6 | PERF_GPU-16, ARCH-20, MATH_EQUIV-19 |
| 6.7 | MATH_EQUIV-14, ARCH-18 |
| 6.8 | PERF_GPU-17, SOTA-23 |
| 6.9 | PERF_GPU-19, SOTA-17 |
| 7.1 | SOTA-21, PERF_GPU-10 |
| 7.2 | ARCH-11 |
| 7.3 | SOTA-13, MATH_EQUIV-08 |
| 7.4 | MATH_EQUIV-15, SOTA-18 |
| 7.5 | SOTA-15 |
| 7.6 | SOTA-25 |
| 7.7 | SOTA-04 |
| 8.1 | PHYS_H-01, MATH_EQUIV-17, SOTA-24, ARCH-29, CRIT-03 |
| 8.2 | PHYS_H-04, MATH_EQUIV-07, ARCH-15, LOSS_TRAIN-17, DATA_PIPE-01, SOTA-06 |
| 8.3 | DATA_PIPE-20 |
| X.1 | SOTA-19, SOTA-26 |
