🧪 `g2m_delta.py`: H0 = H de la densidad atómica superpuesta (SIESTA, 1 iteración; no depende del peso de mezcla); el modelo aprende H − H0 y al predecir se suma H0 (ida y vuelta exacta). H − H0 en Γ: grafeno `6x6` 2,5 meV frente a 200 de H; H2O 248 frente a 3044. **H2O (`gaugeh`, 3 semillas): H alineado 5,6 → 1,01 meV (−82 %), gap 80 → 9,6 meV (−88 %), ocupados 8,2 → 1,5.** 🧪 `6x6` y bicapa |

Fecha: 2026-09-27. Recoge **todas** las propuestas de las 7 auditorías de `auditorias_mejoras/`:

- la de Claude, con sus 217 hallazgos (cada ID está asignado a un punto en el apéndice);
- las 6 de GPT (Astra, Sol, Computational Material Physics, Maths and Physics, ML, Scholar), que
  coinciden casi siempre con la de Claude. Lo que solo aparece en ellas va marcado **(GPT)**.

Sustituye a la "Fase 5" de `plan_mejoras.md`. Ese fichero queda como historial de las fases 0–4.

## Resumen de resultados (27–28 de septiembre)

Todas las cifras son H-MAE en meV, con 3 semillas salvo indicación.

**Precisión en `6x6` (N=8, receta de referencia 1,94):**
- bloques con término lineal (`E3nnLinearTSQNodeBlock` + `E3nnLinearEdgeBlock`), con poly3:
  - bf16: 1,80;
  - fp32: **1,24** (5 semillas, −36 %);
- con 16.000 pasos sin EarlyStopping (5 semillas): bf16 **1,11** (−43 %), fp32 **1,02** (−48 %;
  1,024/1,270/0,954/0,930/0,902);
- término lineal solo en las aristas (el nuevo valor por defecto), 16.000 pasos, `torch_compile`:
  bf16 1,23 (5 semillas), fp32 1,14 (2 semillas);
- quitar el EarlyStopping de la receta (paciencia 80) da por sí solo −13 % (1,69), sin coste;
- el suelo de ruido de las etiquetas de SIESTA es ≈0,002 meV: queda margen.

**Extrapolación:**
- bicapa → registros de moiré no vistos: 20,95 → **6,92** con fp32 + lin (−67 %), bandas alineadas
  69,6 → 14,5;
- monocapa (`6x6`) → bicapa (5 semillas): 78 → **66** con lin bf16 + 16.000 pasos (−15 %; con 3
  semillas parecía 61), fp32 75; la dispersión entre semillas (47–109) es mayor que las diferencias
  entre recetas; poly6 la empeora (≈125–130);
- monocapa (`w90`, celda de 2 átomos) → bicapa: no funciona con ninguna variante (150–450).

**Otro material, H2O** (moléculas perturbadas, 120 de entrenamiento; ver su sección):
- la receta nueva (fp32, término lineal en aristas, sin EarlyStopping) baja H de 55,3 a **48,7** (−12 %,
  gana en las 3 semillas) y la mediana del error del gap de 565 a 379 meV (−33 %); el lin de nodos
  empeora el gap (1054);
- **con la pérdida invariante de gauge (4.2) y S solo como etiqueta: H alineado 29,6 → 5,6 meV (−81 %),
  gap 986 → 80 meV (−92 %), autovalores ocupados −97 %.** En una molécula E_F cae en el gap y ~40 % del
  error con MSE era ese desplazamiento del cero de energía por estructura.

**Pérdida invariante de gauge (4.2) con S solo como etiqueta** (`gauge_invariant_mse`, etiquetas (H, S),
`n_output_components=1`: el modelo predice solo H y S solo define el desplazamiento; misma VRAM que sin
gauge). Se compara el H sin el desplazamiento c·S y las bandas alineadas, porque deja libre el cero de
energía de cada estructura (al predecir hay que fijarlo, p. ej. con la E_F de la H predicha):
- **`6x6` dentro de distribución (lin bf16 16.000 pasos, 5 semillas): H alineado 0,926/1,306/1,012/0,939/1,048
  → 0,622/0,808/0,691/0,686/0,689 (1,05 → 0,70 meV, −33 %, gana en las 5 y con mucha menos dispersión)**;
- **bicapa → moiré (fp32 + lin, 3 semillas): H 6,39 → 5,64, intracapa 7,7 → 4,4 (−43 %), bandas alineadas
  14,5 → 7,3 (−50 %; 6,8/6,9/8,3 frente a 13,6/14,5/15,5)**;
- monocapa (`6x6`) → bicapa (5 semillas): H 62,8 → 53,2 (−15 %), pero bandas alineadas 110 → 162 (peores en
  3 de 5 semillas);
- prediciendo también S era peor y más caro: `6x6` 0,86 meV con +48 % de VRAM; bicapa → moiré bandas 11,7
  con una semilla mala.
- **+ `kl` (`keep_last_layer_irreps`: la última capa de MACE conserva l>0), 5 semillas: la mejor receta de
  `6x6`.** Dentro 0,732/0,740/0,781/0,529/0,791 → 0,72 meV (empata con `gaugeh`, 0,70; −32 % frente a lin).
  **Monocapa → bicapa: registros H 62,8 → 40,4 meV (−36 % frente a lin, −49 % frente a la referencia 78,5;
  mejor en las 5 semillas), intracapa 67 → 38, bandas alineadas 110 → 100; MD AB H 64 → 43, bandas 91 → 83.**
  Misma VRAM (6,0 GB), +30 % de tiempo.
- Lote nocturno (3 semillas cada uno; H alineado en meV):
  - `gaugeh` con 32.000 pasos: 0,592/0,791/0,670 → 0,68 (16.000: 0,71): −3 %, dentro del ruido, por el
    doble de tiempo. ❌
  - `gaugeh` + `kl` en fp32 (`6x6`): 0,536/0,783/0,681 → **0,67 (bf16 0,75 en las mismas semillas; la mejor
    dentro de `6x6`)**; bicapa H 25/46/41 → 37 (bf16 38), pero bandas alineadas 125 (bf16 94).
  - bicapa → moiré con `kl` (fp32): H 5,64 (igual que sin `kl`), bandas alineadas 6,7 (sin `kl` 7,3).
  - H2O con `kl`: H alineado 5,9 (sin `kl` 5,6), gap 101 (80): `kl` no ayuda en moléculas.
  - **`w90` (fp32 + poly6 + lin + `kl`) + gauge con S como etiqueta: validación alineada 2,17/2,17/1,46 →
    1,04/1,75/1,21 (1,93 → 1,33, −31 %)**; test común alineado 1,84/2,71/1,70.

**Δ-learning sobre H0 de SIESTA (5.1): la mayor mejora del roadmap.** El modelo aprende H − H0, con H0 el
H no autoconsistente de la densidad atómica superpuesta (1 iteración de SIESTA, ~18 s en `6x6`), y al
predecir se suma H0. Con la pérdida de gauge (S como etiqueta), 3 semillas:
- `6x6` H alineado 0,71 → **0,14 meV** (con `kl` 0,75 → **0,12**; referencia 1,94 bruto). **Con `kl`, 5 semillas:
  0,145/0,122/0,103/0,138/0,141 → 0,130 meV; bicapa registros 5,75, bandas alineadas 17,8; MD AB 6,2 / 12,9;**
- monocapa → bicapa: registros 52 → **6,3 meV** (con `kl` 38 → **5,3**; referencia 78), bandas alineadas
  156 → **19** (con `kl` 94 → **17**; referencia 162); MD AB 53 → **6,8** (con `kl` 41 → **5,9**);
- H2O H alineado 5,6 → **1,0 meV**, gap 80 → **9,6 meV**.
- **`w90` (celda de 2 átomos) → bicapa, que nunca había funcionado (150–450 meV):** fp32 + lin + gauge + Δ, registros
  4,3/4,5/4,7 → **4,5 meV** (sin Δ, `fp32_lin`: 171/277/204), bandas alineadas **22,6**, MD AB **5,0**; con poly6 +
  `kl` 5,2 meV y bandas **16,7**, MD AB 5,4 / 12,8. Dentro de `w90` (test común alineado): 2,08 → **0,62 meV**.
  **Entrenando solo con la celda de 2 átomos, la bicapa sale igual que entrenando con la supercelda `6x6`.**
- **bicapa → moiré** (fp32 + lin + gauge, registros no vistos): H alineado 5,64 → **0,96 meV** (1,15/0,93/0,79;
  referencia 20,95), intracapa 4,4 → 0,67, intercapa 6,8 → 1,25, bandas alineadas 7,3 → **2,0 meV**
  (referencia 69,6).

**Dónde está el error (4.3, `g2m_error_breakdown.py`, `6x6`, sin el desplazamiento c·S):** fuera del
alcance de los orbitales ≤1 % (la máscara orbital, 4.6, no aporta en grafeno). lin quitó el 82 % del error
de los pares a ~4,3 Å (cerca del corte); la pérdida de gauge mejora todas las capas de vecinos (primeros
6,1 → 3,6 meV por elemento), pero el **on-site** apenas (12,3 → 10,3): es ya el 23 % del error que queda y
el mayor por elemento.

**Valores por defecto:** `LitMACEMatrixModel` usa `E3nnLinearEdgeBlock` desde a39439a (gana en `w90`,
`6x6` y H2O). El bloque de nodos sigue siendo `E3nnSimpleNodeBlock`: el lineal ayuda en `6x6` con bf16,
es neutro con fp32 y empeora el espectro de H2O.

**Velocidad y VRAM (`6x6`):**
- `torch_compile`: paso 2× más rápido, −25 % del tiempo total, misma precisión, −28 % de pico en bf16;
- nodo proyectado: 2,3× menos tiempo de GPU, pero pierde precisión (bf16: 2,32 meV); descartado para precisión;
- `lean_mace` (ya por defecto): −14 % de pico;
- checkpointing: −24 % (fp32) / −34 % (bf16) de pico del proceso, +34 % de tiempo.

**Recetas recomendadas** (con etiquetas (H, S), añadir la pérdida invariante de gauge con S solo como
etiqueta: −33 % en `6x6`, bandas −50 % en bicapa → moiré, −81 % en H2O; para extrapolar de monocapa a
bicapa, con `kl`: H −36 % y bandas −9 %):
- **`6x6` y extrapolación: lin + poly3 + 16.000 pasos sin EarlyStopping, bf16, pérdida de gauge con S como
  etiqueta, `kl` y Δ-learning sobre H0 de SIESTA (`lin_steps16k_gaugeh_kl_delta`): 0,13 meV dentro, 5,8 meV en
  bicapa y bandas 18 (referencia 1,94 bruto / 78 / 162);**
- **celda de 2 átomos (`w90`) y extrapolación a bicapa: fp32 + poly6 + lin + `kl` + gauge con S como etiqueta + Δ:
  0,62 meV en el test de `w90`, 5,2 meV y bandas 16,7 en bicapa;**
- **bicapa → moiré: fp32 + lin + gauge con S como etiqueta + Δ: 0,96 meV en registros no vistos, bandas 2,0
  (referencia 20,95 / 69,6);**
- `6x6` y extrapolación: bloques lin + poly3 + 16.000 pasos sin EarlyStopping, en bf16 (más `torch_compile`
  para compensar el tiempo): 1,11 dentro y 66 en bicapa;
- solo precisión dentro de distribución: lo mismo en fp32 (1,02, pero 75 en bicapa);
- otros materiales (H2O): fp32 + lin solo en aristas, sin EarlyStopping;
- bicapa → moiré: fp32 + lin (+ `eh16`), 6,2–6,9 frente a 20,95;
- `w90`: fp32 + poly6 + lin, **5 semillas: 4,92/5,57/4,78/4,67/6,58 → 5,30 meV frente a 12,98 (−59 %)**;
  añadiendo `kl` (`keep_last_layer_irreps`: la última capa de MACE conserva l>0), **4,32 meV y bandas 10,5 (−67 %): la mejor receta de `w90`**.

**poly6 empeora la extrapolación en los tres escenarios** (monocapa → bicapa, bicapa → moiré); úsese poly3 si hay que extrapolar.

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
   **Con 5 semillas** (dentro / bicapa por semilla, H / intracapa / bandas alineadas):
   - lin bf16 16.000 pasos: 0,949/1,335/1,083/1,086/1,075 → **1,11 (−43 %)**; bicapa 55/81/47/91/58 →
     **66** (intracapa 75, bandas alineadas 110). Las semillas 3 y 4 extrapolan peor: el 61 de 3
     semillas era optimista.
   - fp32 + lin 16.000 pasos: 1,024/1,270/0,954/0,930/0,902 → **1,02 (−48 %, la mejor dentro)**;
     bicapa 68/91/47/109/58 → 75 (intracapa 89, bandas 119). Se mantiene el compromiso: fp32 gana
     dentro y bf16 fuera, aunque la dispersión fuera (47–109 con la misma receta) es mayor que la
     diferencia entre recetas.
   - lin solo en aristas (el valor por defecto desde a39439a) + `torch_compile`, bf16 (`rec_bf16`):
     1,503/1,289/1,116/1,188/1,067 → 1,23; bicapa 73 (intracapa **85**, como la referencia: sin lin
     en nodos la mejora fuera es solo intercapa; bandas alineadas 104). Frente a lin completo gana
     este en 4 de 5 semillas fuera (−7 meV de media) y dentro Δ +0,55/−0,05/+0,03/+0,10/−0,01.
     En fp32 (`rec_fp32`, 2 semillas): 1,229/1,055 → 1,14; bicapa 70 (intracapa 74), bandas
     alineadas **71, las mejores**: con fp32 el lin de nodos no aporta.
   Nodo proyectado (fp32, 8000 pasos): 1,09/1,022/2,134 → 1,42 meV (−27 %; frente a lin Δ −0,03/−0,45/+0,97):
   no concluyente y con más varianza. En bf16 es claramente peor: 2,509/1,924/2,530 → 2,32 meV (+20 % frente a
   la referencia). La proyección a 16 canales pierde precisión: ❌ para precisión; como opción de velocidad solo
   en fp32 y con cautela (siguiente paso posible: proyección a 32 canales). Con poly6 en bf16, en cambio, empata con
   lin (1,911/1,471/1,598 → 1,66 frente a 1,70), y en `w90` (fp32 + poly6) queda cerca (5,24/—/5,02 frente a
   4,92/5,57/4,78). En `6x6` fp32 + poly6: 1,194/1,362/1,42 → 1,33 frente a 1,51 de lin (mejor). Comportamiento
   inconsistente: con poly6 empata o gana; con poly3 pierde (bf16 2,32, fp32 1,42 con una semilla mala).
   fp32 + poly6 + lin sin EarlyStopping: 1,114/1,585/1,161 → 1,29 (frente a 1,51 con él, −15 %); la mejor receta
   dentro de distribución sigue siendo fp32 + lin con poly3 y 16.000 pasos (1,08), que además extrapola mejor.
   Nodo proyectado fuera de distribución: 71 meV (lin 72).
   `torch_compile` en entrenamiento real (fp32 + lin, semilla 0): 1,057 meV frente a 1,056 (misma precisión)
   en 30,3 frente a 40,3 min (−25 %, compilación incluida; carga de la máquina no controlada). Requiere que
   el PATH no tenga entradas inaccesibles (torch 2.11 ejecuta `nvcc --version` en cada compilación);
   `g2m_autorun` ya las filtra. En bf16 (lin, 3 semillas): 1,239/1,524/1,432 → 1,40 meV (sin compilar 1,80;
   diferencia dentro del ruido) y **pico del proceso 5520 → ~4000 MiB (−28 %)**.
   **5 semillas de fp32 + lin (8000 pasos): 1,056/1,469/1,163/1,150/1,380 → 1,24 meV (−36 %)**; las cinco
   por debajo de la mejor semilla de la referencia (1,839).
8. En `w90`, fp32 + poly6 + lin + `eh16`: 4,88/5,36/4,38 → 4,87 meV dentro (lin sin eh16 5,09; bandas 15,0 frente a
   13,5): neutro; en bicapa 181 meV (lin 221, eh16 sin lin 153).
9. En `w90`, fp32 + lin con poly3: 7,32 meV dentro (4,12/11,36/6,48; bandas 16,6) y 229 meV en bicapa: en
   `w90` poly6 sí ayuda dentro de distribución (fp32 + poly6 + lin 5,09) y la extrapolación desde la celda
   de 2 átomos no mejora con ninguna receta.
10. En `w90`, bf16 + poly6 + lin: 7,55 meV dentro (8,40/8,07/6,19; fp32 + poly6 + lin 5,09) y 190 meV
   en bicapa (fp32 + poly6 + lin 221, poly6 solo 183). En `w90` fp32 sí aporta dentro de distribución.
11. En `w90`, variantes de los bloques lin sobre fp32 + poly6 (3 semillas, dentro de distribución / bandas):
   **`lin_kl` (lin + `keep_last_layer_irreps`) 4,48/4,95/3,52 → 4,32 meV, bandas 10,5: la mejor receta de `w90` hasta ahora**
   (−67 % frente a la referencia 12,98, y las bandas bajan de 13,2 a 10,5); `linedge` (lin solo en aristas)
   4,81/4,36/4,40 → 4,52, bandas 15,0 (la precisión de H viene de las aristas, las bandas empeoran);
   `linnode` (lin solo en nodos) 12,65/13,38/14,97 → 13,67, bandas 18,3: sin lin en aristas no hay ganancia.
   Nota: en `6x6`, fp32 + poly6 + `kl` sin lin no mejoraba (2,01 frente a 1,94 de la referencia); combinado con lin en `w90` es la mejor.
12. En `w90`, capacidad sobre fp32 + poly6 + lin (dentro / bandas / min): `mp4` (`mp_cutoff` = 4 Å)
   5,42 / 16,2 / 24,3 (lin solo 5,30 / 13,2 / ~25): neutro en H y peor en bandas; `mp3` 7,92 / 20,8 (peor);
   `mid16` 6,27 / 21,9 (peor). Sin lin la tendencia es la misma (`fp32poly6_mp4` 12,86 pero bandas 39,0,
   `mp3` 17,19, `mid16` 13,28 frente a 12,98 de fp32 + poly6): **más capacidad no compra nada en `w90`;
   lo que importa son los bloques lin (y `kl` encima)**.
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
   bf16 + poly6 + lin: validación 8,91, registros no vistos **11,72 (+39 % frente a lin con poly3)**, intracapa
   17,3, bandas 56,5/23,3: poly6 también empeora la extrapolación entrenando con bicapa (y apenas cambia dentro).
   fp32 + poly6 + lin: validación 6,58, registros no vistos **9,83 (+42 % frente a fp32 + lin con poly3)**, bandas
   41,6/20,3: lo mismo en fp32.
   Más variantes (validación / no vistos / bandas alineadas): fp32 + poly6 + lin + eh16 5,74 / 8,13 / 20,9 (la mejor
   dentro de distribución); fp32 + poly6 + eh16 **sin lin** 6,30 / 10,87 / 19,4. Y **fp32 + poly6 sin lin ni eh16:
   8,61 / 10,56 / 18,8**: con poly6, lin apenas cambia la extrapolación (9,83 frente a 10,56) y lo que la
   arruina es poly6 en sí. fp32 también cuenta: sin lin, fp32 + poly6 baja los no vistos de 20,95 a 10,56, y con lin y
   poly3, fp32 frente a bf16 los baja de 8,42 a 6,92 (−18 %).
   Con `mp4` (`mp_cutoff` = 4 Å) sobre fp32 + poly6: sin lin 8,54 / 9,41 / 23,2 y con lin 7,69 / 12,82 / 26,0.
   Curioso: `mp4` reparte el error al revés (intracapa 8,1–8,7 frente a 13,8–14,9, pero intercapa 10,7–16,9
   frente a 5,9–6,2), así que las bandas empeoran; **como en `w90`, limitar el alcance del paso de mensajes no ayuda**.

## Otro material: H2O

Script: `MD_vs_AtomicDisplacement/Comparison/scripts/g2m_h2o_benchmark.py`.

- **Datos:** 200 moléculas de agua con ruido gaussiano de 0,05 Å por coordenada sobre la geometría
  relajada de `AtomDisplacement`, calculadas con SIESTA en punto único (DZP, PBE, 200 Ry, caja de 15 Å).
  Reparto: 120 de entrenamiento, 30 de validación y 50 de test.
- **Entrenamiento:** 8000 pasos, MSE por elemento.
- **Métricas:** H en Γ; los 4 autovalores ocupados y el gap HOMO-LUMO, con la S de referencia. La
  ventana de ±2 eV del grafeno queda vacía en una molécula.
- **Medianas:** se dan por estructura, porque unas pocas moléculas muy distorsionadas dominan las
  medias (H-MAE máximo 315 meV frente a una mediana de 29).

Las etiquetas son H − E_F·S y en una molécula E_F cae en el gap, donde SIESTA la coloca de forma casi
arbitraria: el desglose del error (4.3) muestra que ~40 % del error de H de estos modelos es un
desplazamiento c·S por estructura (|c*| medio ≈ 210 meV). Por eso se comparan también métricas
invariantes: H tras quitar el c* de mínimos cuadrados y los autovalores ocupados sin su desplazamiento
medio (H + cS desplaza todos los autovalores exactamente c); el gap ya lo es.

| Receta (3 semillas) | H-MAE | H alineado (mediana) | ocupados alineados (mediana) | gap (mediana) | Por semilla (H alineado) |
|---|---:|---:|---:|---:|---|
| `base` (bf16, bloques de siempre, EarlyStopping) | 55,3 | 33,8 (28,9) | 324 (55) | 1193 (565) | — |
| `rec_fp32` (fp32, lin en aristas, sin EarlyStopping) | 48,7 | 29,6 (22,1) | 248 (44) | 986 (379) | — |
| `lin_fp32` (lo mismo con lin también en nodos) | 51,3 | 32,2 (26,0) | 286 (50) | 1828 (1054) | — |
| `gauge_fp32` (`rec_fp32` + pérdida invariante de gauge, 4.2; predice también S) | — | 7,2 (5,9) | 9,9 (8,5) | 102 (46) | 7,6/7,0/6,9 |
| `gaugeh_fp32` (lo mismo, S solo como etiqueta: predice H) | — | 5,6 (4,2) | 8,2 (6,5) | 80 (37) | 5,8/5,1/5,9 |
| **`delta_gaugeh_fp32` (lo mismo aprendiendo H − H0, 5.1)** | — | **1,01** | **1,5** | **9,6** | 0,97/0,97/1,08 |

- **La receta nueva gana también en otro material**, incluso en las métricas invariantes: `rec_fp32`
  frente a `base`, −12 % en H alineado y −33 % en la mediana del gap. El lin de nodos no ayuda y
  empeora el gap, así que el bloque de nodos por defecto no cambia.
- **La pérdida invariante de gauge es la mayor mejora del roadmap en H2O:** frente a `rec_fp32`, H
  alineado −76 %, autovalores ocupados −96 % (mediana −81 %) y gap −90 %; la peor molécula baja de
  73–147 a 23–25 meV. Las 3 semillas coinciden. El H-MAE bruto no cuenta para ella (1,5–2,4 eV):
  deja libre el cero de energía de cada estructura, que es justo lo que no se puede aprender.
- **Sin predecir S (`n_output_components=1`) es aún mejor y más barato:** H alineado 7,2 → 5,6 (−22 %),
  gap 102 → 80, ocupados 9,9 → 8,2, pico de VRAM 2160 → 1372 MiB (−36 %). Frente a `rec_fp32`: H
  alineado −81 %, gap −92 %.
- Con MSE, el mejor checkpoint llegaba hacia la época 800–1100 de 2000 (sobreajustaba el ruido de
  E_F); con la pérdida de gauge sigue mejorando pasada la 1400. Los errores grandes de autovalores
  no venían de una S mal condicionada, sino de ese ruido de las etiquetas.

## Ola 0 — Red de seguridad y fallos críticos

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 0.1 | Emparejamiento de aristas en modo no simétrico: heteronuclear (reordenación) y homonuclear (m_ij, m_ij) | C | S | bajo | ✅ |
| 0.2 | `symmetric_matrix` deducido del tipo de matriz (H/DM/EDM son simétricas) | C V T | S | bajo | ✅ |
| 0.3 | `normalized_density_error`: re-sumaba la DM atómica; signo on-site/off-site | C | S | bajo | ✅ |
| 0.4 | Métricas mal definidas o duplicadas (`mapestdmix`, `mapemaemix`, `mapemsemix`) | C | S | bajo | ✅ |
| 0.5 | CI que ejecuta pytest | C | S | bajo | ✅ `.github/workflows/tests.yaml` |
| 0.6 | Tests: equivariancia con rotación genérica, localidad de bloques, reordenación, asserts que siempre pasan | C | M | bajo | 🔶 los 14 tests del fork que ya fallaban en a5e8134 están arreglados (72b468c; suite: 470 pasan). Faltan ENG_QUALITY-04 y los tests del modelo completo con supercelda |
| 0.7 | Carga con torch ≥ 2.6 (e3nn, checkpoints) | C | S | bajo | 🔶 falta el KeyError con variables `E3MAT_*` |
| 0.8 | `initial_node_feats` ignorado; rama `version != "new"` rota | C | S | bajo | ✅ `initial_node_feats` validado; la rama `version != "new"` (llamaba a `init_model` con argumentos que `MatrixMACE` ya no acepta) eliminada: el argumento se conserva para los checkpoints y cualquier otro valor da error |
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
| 1.7 | Término lineal junto al producto tensorial, más sesgo 0e on-site (`E3nnLinearTSQNodeBlock`, `E3nnLinearEdgeBlock`) | P | S | bajo | ✅ **`w90`: 12,61 → 5,09 meV (−60 %; 4,92/5,57/4,78), bandas 18,9 → 13,5.** Misma VRAM. **`6x6`** (receta de referencia bf16 + lin): 1,23/2,58/1,58 → media 1,79 frente a 1,94 (Δ −0,73/+0,57/−0,26; −7 %, ruidoso); **fp32 + poly6 + lin en `6x6`: 1,19/1,58/1,74 → media 1,51 frente a 1,94 (−22 %; Δ −0,77/−0,43/−0,10)**. Pero fuera de distribución (bicapa) empeora: 130 frente a 78 meV, mientras que lin en bf16 da 63. Compromiso: fp32 mejora dentro y empeora fuera (en `w90` igual: fp32 solo 309, poly6 solo 183, línea base 230); lo que perjudicaba fuera era poly6 (punto 6 de la extrapolación). **`E3nnLinearEdgeBlock` es el bloque de aristas por defecto desde a39439a** (gana en `w90`, `6x6` y H2O); el de nodos sigue siendo el simple (ver el resumen y la sección de H2O) |
| 1.8 | Peso de las aristas en la pérdida (`loss_kwargs: {edge_weight: 2}` = MSE sobre la matriz completa) | P | S | bajo | 🔶 `w90`: 11,29 frente a 12,61 meV (Δ por semilla −4,2/−0,3/+0,6), bandas 18,2 frente a 18,9: dentro del ruido. Opción disponible (`loss_kwargs`) |
| 1.9 | `E3nnEdgeBlockNodeMix`: pesos del TP que dependen de la distancia y de ambos nodos (necesita `preprocessing_edges_reuse_nodes`) | P | config | bajo | ❌ `w90`: 11,77 frente a 12,61 meV (Δ +1,0/−1,8/−1,7), bandas 19,5 frente a 18,9: pequeño, dentro del ruido; **y fuera de distribución es la peor variante (557 meV en bicapa)** |
| 1.10 | Readouts condicionados del fork (`E3nnHamiltonian{Node,Edge}Block`). Vigilar su VRAM **(GPT Astra)** | P | config | medio | ❌ con esta receta falla al construirse: intenta reservar 139 GB de RAM (el MLP que genera los pesos del TP recibe las features de MACE, 48 canales × 9 segmentos). Solo sería viable proyectando antes las features de nodo a pocos canales |
| 1.11 | Resolución radial `num_bessel=16` | P | config | bajo | ❌ `w90`: 17,0 frente a 12,61 meV (15,7/18,8/16,4; bandas 25,9 frente a 18,9): peor |
| 1.12 | EMA, SWA/segunda etapa, clipping, warmup | P | S | bajo | 🔶 EMA empeora (+2,8 meV). **Gradient clipping (norma 1) sobre la mejor receta de `6x6` (gauge + `kl` + Δ): neutro** (0,144/0,125/0,103 frente a 0,145/0,122/0,103; bicapa 5,4 frente a 5,3). SWA no es posible (deepcopy del modelo) |
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
| 3.7 | RTPs: caché vacía, dtype congelado, RTP lento en cada llamada | T | S | bajo | 🔶 `irreps_from_data` sigue el dtype de las etiquetas (72b468c); caché vacía y RTP lento pendientes |
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
| 4.1 | Leer S junto a las etiquetas; reconstruir matrices no ortogonales; el servidor escribe el TSHS con S real | C P | M | medio | 🔶 `matrix_component_policy="h_and_overlap"` lee (H, S) y ya escribe H con S (la política no llegaba a la conversión a `sisl_H`, 72b468c); la predicción del benchmark la admite. Falta la reconstrucción no ortogonal completa |
| 4.2 | Pérdida invariante de gauge (c por estructura en forma cerrada) y métrica c* | P | M | medio | ✅ **con S solo como etiqueta (`n_output_components=1`, misma VRAM): `6x6` H alineado 1,05 → 0,70 meV (−33 %, 5 de 5 semillas); bicapa → moiré bandas alineadas 14,5 → 7,3 (−50 %, 3 de 3) e intracapa −43 %; H2O H alineado 29,6 → 5,6 (−81 %), gap 986 → 80 meV (−92 %).** Monocapa → bicapa: H −15 % pero bandas peores en 3 de 5. Prediciendo también S: peor y +48 % de VRAM. `gauge_invariant_mse` (7b97b7a, 71c1afa) con `h_and_overlap`; métrica c* en `g2m_ood_eval.py`, `g2m_h2o_benchmark.py`, `g2m_error_breakdown.py` y `val_metrics` (`H_MAE_gauge_meV`) |
| 4.3 | Métricas espectrales en validación y desglose del error por especie, par, l y distancia | — | M | bajo | ✅ `g2m_error_breakdown.py`: error de Γ sin el desplazamiento c·S, por on-site/capa de vecinos/par l, y fuera de alcance orbital. `6x6` (semilla 0, meV por elemento): lin 1,79 → 0,93 quitó el 82 % del error a ~4,3 Å; la pérdida de gauge 0,93 → 0,62 mejora todas las capas (primeros vecinos 6,1 → 3,6, terceros 2,8 → 1,4) pero el **on-site** solo 12,3 → 10,3: es ya el 23 % del error y el mayor por elemento, el siguiente objetivo |
| 4.4 | Pérdidas físicas con S: N_e, Mulliken, autovalores de primer orden | P | M/L | medio | ⏳ |
| 4.5 | DM/EDM: Tr(DS) = N, McWeeny, EDM ligada a DM y H | P C | M | medio | ⏳ |
| 4.6 | Ceros estructurales y máscara orbital en vez de bloque atómico denso **(GPT CMP, Maths and Physics, ML)** | P | S/M | bajo | ❌ en grafeno (base SZ) el error fuera del alcance de los orbitales es ≤1 % del total y los ceros exactos ya salen a cero: no hay nada que enmascarar |

## Ola 5 — El objetivo: residuo y normalización

| # | Cambio | Gana | Esf. | Riesgo | Estado |
|---|---|---|---|---|---|
| 5.1 | Referencia on-site por especie → Δ-learning sobre H0 (atómico o Slater–Koster; H0 de SIESTA como versión completa) | P | M → L | medio | ✅ **la mayor mejora del roadmap.** `g2m_delta.py`: H0 = H de la densidad atómica superpuesta (SIESTA, 1 iteración de punto único; no depende del peso de mezcla; se comprueba que la geometría coincide con la de la etiqueta); el modelo aprende H − H0 y al predecir se suma H0 (ida y vuelta exacta). H − H0 en Γ: grafeno `6x6` 2,5 meV frente a 200 de H; H2O 248 frente a 3044. **H2O (`gaugeh`): H alineado 5,6 → 1,01 meV (−82 %), gap 80 → 9,6 (−88 %). `6x6` (3 semillas, H alineado / registros de bicapa H / bandas alineadas / MD AB H): `gaugeh` 0,707 / 52 / 156 / 53 → con Δ 0,140 / 6,3 / 19 / 6,8; `gaugeh_kl` 0,751 / 38 / 94 / 41 → con Δ 0,123 / 5,3 / 17 / 5,9 (−84 %, −86 %, −82 %, −86 %).** **Bicapa → moiré: registros no vistos 5,64 → 0,96 meV, bandas alineadas 7,3 → 2,0 (referencia 20,95 / 69,6).** Coste al predecir: un cálculo SIESTA de 1 iteración (~18 s en `6x6`) |
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
