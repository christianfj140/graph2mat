# F0 de la fase magnética (roadmap 8.1): viabilidad SIESTA con Fe bcc

Fecha: 2026-10-21. Script: `auditorias_mejoras/magnetismo_f0_fe.py` (`relax`, `run`, `analyze`).
Salidas: `auditorias_mejoras/experimentos_magnetismo_f0_fe_20261021/` (ignorado por git; `results.json`).
Plan: `auditorias_mejoras/roadmap_magnetismo.md`, Task 1.

## Configuración

- SIESTA 5.4.2, PBE, PseudoDojo nc-fr-04 standard `Fe.psml` (relativista completo, 16 e de valencia,
  sha256 959e0d8b…, añadido a `materials/bench_common/pseudos_fr/PROVENANCE.json`). DZP con
  EnergyShift 0,005 Ry y polarización no perturbativa: 19 orbitales por Fe (3s, 3p SZ; 4s, 3d DZ; 4p).
- Mesh 550 Ry (SIESTA usa 789 Ry por la factorización de la malla), Fermi-Dirac 0,05 eV.
- Momentos nominales en `DM.InitSpin`: |m| = 2,2 μB con signo (colineal) o con ángulos (SOC).
- `Diag.ParallelOverK T`: el SCF de 16 átomos colineal con k6 pasa de 25 a 10 s por iteración.

## Resultados

| Comprobación | Resultado |
|---|---|
| Red PBE (2 átomos, celda variable) | a = 2,8214 Å, 2,23 μB/átomo |
| Malla k (16 átomos, H frente a k8) | k6: 0,045 meV MAE, 5,1 meV máx.; k4: 0,22 / 36 meV; smearing 0,1 eV frente a 0,05: 15 meV máx. |
| TSHS SOC magnético | 8 componentes + S, legible con sisl |
| Inversión temporal (2 átomos, m → −m) | canales pares iguales e impares opuestos: 2·10⁻⁹ eV (SCF), 10⁻¹¹ eV (H0) |
| C4 sobre z con m fuera del eje | 2·10⁻⁹ eV; con el espín sin girar (control) 0,5–1,9 eV |
| Inversión con m fijo | 3·10⁻⁹ eV; tratando m como vector polar (control) 1,0–2,3 eV: **m es axial** |
| b0 | ≤ 1,5·10⁻¹⁵ eV (cero, como predice la teoría) |
| Marco de espín | el momento convergido sigue al nominal (0,013°): ejes de espín = ejes cartesianos; `DM.InitSpin` toma θ, φ en grados |
| Mulliken | suma sisl por átomo = momento total de SIESTA; factor n↑ − n↓ |
| Momentos desde H (diagonalización + Fermi-Dirac, `run_magnetic_campaign.moments_from_h`) | frente a Mulliken del DM: 10⁻⁵ μB (colineal), 7·10⁻⁵ μB (SOC) |
| H − H0, colineal (16 átomos) | ↑/↓ hasta 1,5–2,0 eV; (↑ − ↓)/2: H hasta 1,6–1,7 eV, H − H0 hasta 0,41–0,53 eV |
| H − H0, SOC | b (SOC) 10⁻¹⁶ eV: el SOC llega entero por H0; a (intercambio) H hasta 1,3–1,7 eV, H − H0 hasta 0,38–0,51 eV |
| Fuera del grafo (R_i + R_j = 7,57 Å) | 2·10⁻¹² de \|H\|² fuera de sitio, 0,03 meV máx. (hasta 9,2 Å): no hace falta 8.2 |
| S (16 átomos, Γ) | λmin = 7,6·10⁻⁵: mal condicionada; bandas también con S canónica (λ > 10⁻²) |
| Rotación congelada (B') | DM convergido en z girado rígidamente a n1, una iteración, frente al SCF en n1: a0 0,65 meV, a 5,7 meV máx., momentos a 0,5° |

### Convergencia del SCF y metaestabilidad

- Colineales de 16 átomos con DM.Tolerance 10⁻⁶: FM, flip2, flip6, flip8 y AFM (001) convergen
  (78–135 iteraciones) y conservan el signo de cada momento (|m| 1,7–2,6 μB). flip1 y flip4 no
  convergen en 300 iteraciones: H oscila (dHmax 0,2 meV en flip4; 13 meV en flip1, también con mezcla
  0,05 e historia 10 y con smearing 0,1 eV hasta 46 iteraciones).
- SOC de 16 átomos con k6: unos 74 s por iteración (diagonalización 75 %). Los ferromagnetos se
  quedaban en dDmax ≈ 10⁻⁵ con tolerancia 10⁻⁶; con 10⁻⁵ convergen en 1–2 iteraciones más. El
  patrón con 3 espines invertidos oscila a ~5 meV tras 112 iteraciones (k6), y a 40–67 meV tras 22 con
  k4 y 0,1 eV. No se calculó el de 6 inversiones (coste).

## Decisiones para F4

- DM.Tolerance 10⁻⁵ (valor del bench) y como máximo 120 iteraciones. Las etiquetas que no convergen
  o que pierden el patrón nominal se marcan y se excluyen (`fe_moment_audit`, `moment_audit.json`).
- A (colineal): 16 átomos, k6, 0,05 eV; 28/6/4 + 2 AFM (001) + 2 de 32 átomos.
- B (SOC): 16 átomos, k4 (coste), 0,05 eV; mitad ferromagnetos con eje aleatorio, mitad con
  inversiones; 22/5/4 + 1 AFM (001) + 1 de 32 átomos.
- Sin `edge_cutoff_padding` en el brazo directo.
- B' (no colineal congelado) viable como prueba: la rotación rígida reproduce el SCF a pocos meV.
  Queda para después de F5 si hay tiempo.
- **Temperatura:** con 3 SIESTA de 4 procesos más tests en paralelo se leyeron 82–86 °C y
  `g2m_autorun` pausó un cálculo a 85 °C. F4 usa como mucho 3 cálculos y arranca cada uno solo por
  debajo de 76 °C.
