# 8.1 SOC — fase F0: viabilidad SIESTA en Bi2Se3 (2026-10-19)

Autorizada por el usuario («autorizo esa fase», 2026-10-19). Diseño:
`docs/superpowers/specs/2026-10-19-soc-8-1-design.md`. Script:
`auditorias_mejoras/soc_f0_bi2se3.py`; salidas (ignoradas por git) en
`auditorias_mejoras/experimentos_soc_f0_bi2se3_20261019/`, con `results.json`.

## Montaje

- Celda primitiva romboédrica de Bi2Se3 (5 átomos), red experimental
  a = 4,143 Å, c = 28,636 Å, u_Bi = 0,4008, u_Se2 = 0,2117, sin relajar.
  Distancias Bi–Se 2,851/3,075 Å y Se–Se entre QL 3,518 Å.
- PseudoDojo nc-fr-04 PBE standard psml (`relativity="dirac"`, semicore Bi 5d
  y Se 3d en simple zeta), copiados a
  `MD_vs_AtomicDisplacement/materials/bench_common/pseudos_fr/` con
  `PROVENANCE.json` (sin commitear en ese repositorio).
- Ajustes del banco (DZP, EnergyShift 0,005 Ry, polarización no perturbativa,
  0,025 eV) con `DM.Tolerance 1e-6`; 90 orbitales (18 por átomo), nsc 7×7×7.
- SIESTA 5.4.2, 4 procesos MPI en E-cores; CPU máxima 51 °C.

## Resultados

| Caso | Gap en Γ (eV) | Peso Bi VBM / CBM | Tiempo (s) |
|---|---:|---|---:|
| Sin SOC (FR, `Spin none`), k6 | 0,226 | 0,16 / 0,75 | 82 |
| SOC offsite k4 | 0,261 | — | 203 |
| SOC offsite k6 | 0,279 | 0,67 / 0,16 | 273 |
| SOC offsite k8 | 0,286 | 0,67 / 0,16 | 387 |
| SOC offsite k6, malla 400 / 700 Ry | 0,278 / 0,279 | — | 192 / 336 |
| SOC onsite k6 | 0,169 | 0,22 / −0,27 | 372 |
| SOC + `SOC.Split.SR.SO` k6 | 0,279 (idéntico a k6) | — | 271 |
| H0 SOC (1 iteración, inicio sin espín) | 0,170 | 0,19 / 0,75 | 67 |

Pesos de Mulliken del Bi en el último estado de valencia y el primero de
conducción en Γ.

1. **Lectura:** sisl lee el `.TSHS` como `Spin{spin-orbit}`, dim 9 (8 + S),
   no ortogonal. El TSHS no trae especies; hay que leer la geometría del fdf.
2. **Inversión de bandas:** sin SOC, la valencia en Γ es de Se y la
   conducción de Bi (0,16/0,75). Con SOC se invierte (0,67/0,16): la firma del
   aislante topológico. Un modelo sin SOC no puede reproducirla.
3. **Hermiticidad** de H(k): defecto relativo ≤ 7,4·10⁻¹⁰ en Γ y 3 k
   aleatorios.
4. **Inversión temporal en crudo:** residuo relativo de los canales impares
   (Im h0, Re hx, Re hy, Re hz) ≤ 4·10⁻⁷ con k6 (≤ 2·10⁻⁶ con k4): restos de
   magnetización de la SCF. Kramers: desdoblamiento ≤ 2,6·10⁻⁵ eV (k6).
   La proyección a 4 canales del diseño (§4.1) lo elimina; la tolerancia de F1
   puede fijarse en ~10⁻⁵ relativo.
5. **Magnitudes:** |h0| ≤ 47 eV; |gx|, |gy| ≤ 1,04 eV; |gz| ≤ 1,20 eV.
6. **Hipótesis H-Δ confirmada exactamente:** g(H) − g(H0) ≤ 2,2·10⁻¹⁶ eV en
   los tres canales; h0 sí cambia (máx 1,9 eV, RMS 5,4 meV). Con la variante
   *offsite* todo el SOC está en H0: con la receta Δ el modelo solo aprende h0
   y el SOC llega exacto por H0. La ruta directa es la que obliga a aprender g.
7. **Defecto encontrado y corregido:** SIESTA arranca las SCF no colineales
   ferromagnéticas (0,357 μB en z). En la SCF converge a cero, pero el H0 de
   una iteración lo conserva: rompía la TRS (Re hz 2 %, Kramers 1,9 eV). Con
   `DM.InitSpin` a cero, residuo 3·10⁻¹⁶ y Kramers 10⁻⁷ eV. **El H0 del
   dataset debe arrancar sin espín** (el `g2m_delta` actual no lo hace).
8. **Convergencia:** malla 550 Ry suficiente (400 → 700 Ry: 0,3 meV en el
   gap). k no está convergido en 6 (k6 → k8: +7 meV); usar densidad ≥ k8 de la
   primitiva (n·|a| ≥ 79 Å) en las etiquetas.
9. **Variantes:** *onsite* da un gap muy distinto (0,169 eV) y no es la
   referencia; se usa *offsite* (defecto de SIESTA). `SOC.Split.SR.SO` no
   cambia el TSHS.
10. **Coste:** SOC ×3,3 frente a sin SOC en la primitiva (k6); 5 átomos,
    4 procesos, ~4,5 min. Escala con superceldas por medir en F4.

## Decisión

Puerta F0 superada: datos SOC legibles, hermíticos y con TRS a nivel de
precisión SCF; inversión de bandas reproducida; coste asumible. Pendiente de
decidir con el usuario qué ruta es la principal, dado que con Δ el SOC llega
íntegro por H0.

Límites: celda sin relajar y sin vdW; una sola geometría; pesos de Mulliken
como indicador de inversión, no Z2; residuo de TRS medido en una estructura.
