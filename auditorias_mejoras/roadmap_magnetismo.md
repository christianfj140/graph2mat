# Graph2Mat — magnetismo: momentos magnéticos como entrada y SOC sin inversión temporal

> **For agentic workers:** usa `superpowers:subagent-driven-development` si está
> disponible; si no, `superpowers:executing-plans`. Este documento fija alcance,
> orden, contratos y criterios de cierre. Antes de tocar el código de cada
> tarea, escribe su plan detallado en
> `docs/superpowers/plans/2026-10-2x-magnetismo-<tarea>.md` leyendo antes todos
> los callers afectados.

**Goal:** Graph2Mat predice el Hamiltoniano de materiales magnéticos, colineales
y con SOC, a partir de la geometría **y de los momentos magnéticos por átomo**.
Las predicciones son equivariantes ante E(3) y ante la inversión temporal.

**Por qué:** con solo posiciones como entrada, un material con más de un estado
magnético no se puede aprender. Un antiferromagnético y un ferromagnético con la
misma geometría tienen H distintos. Con SOC, H depende además de la dirección de
la magnetización respecto a la red. El usuario quiere la implementación completa
(2026-10-19: «me gustaría después implementarlo todo»; 2026-10-20: «merece la
pena una implementación de eso», sobre los momentos como entrada).

**Estado de partida (roadmap 8.1):** SOC no magnético cerrado y medido en Bi2Se3.
Política `spin_orbit` con 4 canales reales que imponen TRS, bloque
`E3nnSpinOrbitMatrixBlock`, Δ-learning con H0 y 8.2 (`edge_cutoff_padding`).
Spin colineal (`spin_h_only`, canales ↑/↓) cargable, pero sin momentos como
entrada y nunca entrenado con datos magnéticos reales. Diseño anterior:
`docs/superpowers/specs/2026-10-19-soc-8-1-design.md` (§4.3 dejó la
representación completa para esta fase).

**Tech Stack:** Python ≥3.9, NumPy/SciPy, sisl 0.16.4, Torch, e3nn 0.4.4, MACE,
Lightning, pytest. SIESTA 5.4.2 con PseudoDojo nc-fr-04 PBE standard (psml).
Intérprete: `/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python`.

## Global Constraints

- Repositorio `/home/christian/repositorios/graph2mat`, rama `g2m-v2`, este checkout.
- **Sin merge, push ni PR.** La rama debe contener todo el roadmap antes de fusionar.
- Commits locales pocos y con sentido; resultados y estado del roadmap en lotes.
  Trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (o el del agente que ejecute).
- No commitear en `MD_vs_AtomicDisplacement`. Se pueden editar sus scripts de
  bench (`Comparison/scripts/g2m_bench_materials.py`) y escribir en sus `runs/`.
- Defaults intactos: sin momentos, todo lo existente (datos, modelos,
  checkpoints, `spin_h_only`, `spin_orbit`) da resultados idénticos.
- Todo lo nuevo es opt-in. Sin defaults nuevos sin medición fuera de distribución.
- Python 3.9 compatible. Sin dependencias nuevas.
- GPU solo por `g2m_autorun` (`queue.jsonl`, `append_queue` de
  `auditorias_mejoras/run_readout_comparison.py`), ≥20 % de VRAM libre, sin OOM.
- CPU por debajo de 80 °C. SIESTA en E-cores, con la espera térmica que ya usa `label`.
- Nunca tocar procesos de otros usuarios (amaro) ni trabajos ajenos de la cola.
  Para buscar PIDs: `ps -C python -o pid=,args=` filtrado con awk. Nunca `pkill -f`
  sobre una línea de comandos que contenga la ruta del script.
- Registrar todo resultado, bueno o malo, en `auditorias_mejoras/roadmap.md` (fila 8.1)
  y en el registro de este documento.

## Diseño

### D1. Representación completa del spinor (8 canales reales)

Bloque spinor de cada par: H_ij = Σ_μ σ_μ ⊗ h^μ_ij, con μ ∈ {0, x, y, z} y
h^μ = a^μ + i·b^μ (matrices reales de n_i × n_j orbitales).

| Canal | Hoy | Rotación | Transpuesta (H_ji(−R) = H_ij(R)^†) | Inversión temporal |
|---|---|---|---|---|
| a^0 | `h0` de `spin_orbit` | l_i⊗l_j | +1 | par |
| b^x, b^y, b^z | `g` de `spin_orbit` | (l_i⊗l_j)⊗1e | −1 | par |
| b^0 | rechazado (residuo TRS) | l_i⊗l_j | −1 | impar |
| a^x, a^y, a^z | rechazado (residuo TRS) | (l_i⊗l_j)⊗1e | +1 | impar |

Desde las 8 componentes de sisl en memoria, c = [Re↑↑, Re↓↓, Re↑↓, Im↑↓, Im↑↑, Im↓↓, Re↓↑, Im↓↑]:
a^0 = (c0+c1)/2, a^z = (c0−c1)/2, a^x = (c2+c6)/2, b^y = (c2−c6)/2,
b^x = (c3+c7)/2, a^y = (c7−c3)/2, b^0 = (c4+c5)/2, b^z = (c4−c5)/2.
Inversa: c0 = a^0+a^z, c1 = a^0−a^z, c2 = a^x+b^y, c6 = a^x−b^y,
c3 = b^x−a^y, c7 = b^x+a^y, c4 = b^0+b^z, c5 = b^0−b^z.
Las fórmulas TRS ya certificadas de `spin_orbit.py` son el caso b^0 = a^k = 0.
**Verificar con un oráculo independiente** (spinor construido con matrices de
Pauli frente a `Hk` de sisl) antes de usarlas.

- Política nueva `matrix_component_policy="spin_orbit_full"` (nombre a criterio
  del ejecutor si choca con algo). Canales [a^0, b^x, b^y, b^z, b^0, a^x, a^y, a^z],
  signos de transpuesta [1, −1, −1, −1, −1, 1, 1, 1]. Sin comprobación TRS.
  Exige matrices `is_spinorbit`, excluye S como `spin_orbit` y reconstruye solo
  a formatos sisl.
- Con momentos nulos, la política completa coincide con `spin_orbit` en los
  canales pares y da canales impares nulos.
- **b^0 ≡ 0 esperado en datos SIESTA:** con orbitales reales, sin DFT+U ni
  corrientes orbitales, cinética, Hartree, KB y XC B(r)·σ son reales y solo V_SO
  (independiente de la densidad) es imaginario. Por eso b^0 = 0 y b^k(H−H0) = 0
  (en Bi2Se3, g(H−H0) = 0 a 2·10⁻¹⁶). La representación conserva b^0 para la ida
  y vuelta sin pérdida y para otros códigos; las métricas lo reportan aparte.
  F0 confirma b^0 = 0 a la tolerancia del SCF; si no es cero, se investiga antes
  de seguir.
- El modelo extiende `E3nnSpinOrbitMatrixBlock` o lo generaliza con un signo de
  transpuesta por canal (decisión del ejecutor). Mismas restricciones:
  `symmetric_matrix=True`, pérdidas de la lista blanca, opciones no
  certificadas rechazadas, irreps ocultas de ambas paridades.

### D2. Momentos magnéticos como entrada

- Dato nuevo por átomo en la configuración: `magnetic_moments`, en μB.
  Forma (n_atoms,) con signo en el caso colineal; (n_atoms, 3) cartesiano en el
  caso con SOC. Viaja a `TorchBasisMatrixData` como atributo de nodo, entra en
  batches, cachés, ZIP y checkpoints.
- **Entrada = momentos nominales** (decisión del usuario, 2026-10-21): lo que
  recibe `DM.InitSpin` (módulo con signo en colineal; módulo y ángulos con SOC).
  Es lo que el usuario tendrá al predecir, y H0 usa los mismos valores. Las
  muestras cuyo SCF no conserva el patrón nominal (signo de cada átomo y, con
  SOC, ángulo con el eje nominal por debajo de un umbral fijado en F0) se
  descartan del dataset y se cuentan en la auditoría.
- **Momentos convergidos** solo para filtrar y para métricas: Mulliken con sisl
  (`DensityMatrix.mulliken(projection="atom")` sobre el DM de SIESTA; sisl 0.16
  da [Q, Sx, Sy, Sz] con SOC y (Q, Sz) en polarizado). Comprobar factor
  (n↑−n↓ o la mitad) y signo contra la salida Mulliken de SIESTA.
- Cada etiqueta guarda un JSON con momentos nominales y convergidos. El cargador
  lee los nominales de ahí (o de un argumento explícito en la API Python).
- **Predicción:** el usuario da los momentos nominales por API Python o con el
  mismo JSON junto a la estructura en la CLI.
- Simetrías de la entrada: el momento es par ante inversión espacial e impar
  ante inversión temporal. Con SOC es un vector axial (irrep 1e). En colineal
  sin SOC el espín no se acopla a la red y el momento es un escalar con signo (0e).
- **Inyección local tras MACE** (propuesta basada en xDeepH, Li et al.,
  Nat. Comput. Sci. 3, 321, 2023: introducir {M} de forma estrictamente local
  entrena mejor que tratar {R} y {M} por igual). MACE solo ve la geometría y se
  evalúa una vez. Una capa magnética local combina los rasgos de nodo con m_i y
  con los m_j de los vecinos del grafo del readout. La capa alimenta los
  bloques de nodo y de arista. La arquitectura concreta la decide el ejecutor,
  con un contrato fijo: el bloque on-site de i ve m_i y los m_j vecinos, y el
  bloque de arista ve m_i y m_j.
- Momentos con política colineal (`spin_h_only`): opt-in. Sin ellos, el
  comportamiento actual se conserva bit a bit.

### D3. Inversión temporal exacta por doble pasada

T: m → −m y H → Θ H Θ⁻¹ (Θ = iσ_y K). En canales: los pares no cambian y los
impares cambian de signo. En colineal: ↑ ↔ ↓.

El modelo evalúa el readout con m y con −m (MACE una sola vez).
- SOC: canal par = [F(m) + F(−m)]/2 y canal impar = [F(m) − F(−m)]/2.
- Colineal: ↑(m) = [u(m) + d(−m)]/2 y ↓(m) = [d(m) + u(−m)]/2.

Es exacto por construcción, sin etiquetas de paridad temporal en e3nn, y con
m = 0 da canales impares nulos. La simetrización va antes de la pérdida.

### D4. Material y datos: Fe bcc

Propuesta de Claude (2026-10-20), confirmada por el usuario (2026-10-21): el sistema más barato y robusto. Celda pequeña,
pseudo FR en el mismo tarball PseudoDojo (URL y sha256 en
`MD_vs_AtomicDisplacement/materials/bench_common/pseudos_fr/PROVENANCE.json`;
hoy solo están extraídos Bi y Se). Exchange grande (~1–2 eV, canales impares
bien medibles) y SOC débil: la dirección de m se queda donde se inicializa.

**Límite de SIESTA 5.4.2:** no hay DFT con momentos restringidos por átomo
(en el binario solo aparecen `Spin.Fix`/`Spin.Total` para el momento total y
`Spin.Spiral` sin SOC). xDeepH generó configuraciones no colineales
aleatorias con DFT restringido de OpenMX. Aquí no se puede igual. Datos:

1. **A, colineal sin SOC:** supercelda de Fe bcc (p. ej. cúbica de 16 átomos)
   con rattle y deformación. Patrones aleatorios de espines invertidos vía
   `DM.InitSpin` (+/−), SCF completo. Configuraciones metaestables conocidas en
   Fe; F0 mide qué fracción conserva el patrón. Entrada: momentos convergidos.
2. **B, SOC colineal en eje arbitrario:** las mismas familias con
   `Spin spin-orbit` y un eje global aleatorio por muestra (`DM.InitSpin` con
   ángulos). Ejercita la dependencia de la dirección de m y la equivariancia con
   m rotado. El brazo directo aprende a^0, b^k y a^k; el brazo Δ solo a^0 y a^k
   (b^k llega entero por H0). b^0 es cero en ambos.
3. **B' exploratorio (solo como prueba, tipo slabs en 8.1):** configuraciones
   no colineales «congeladas». Se rota por átomo la parte de espín del DM
   convergido y se calcula H[DM rotado] en una iteración, sin SCF. Solo si F0
   muestra que es coherente: una rotación global rígida debe reproducir el SCF
   en ese eje, salvo la anisotropía. Nunca se mezcla con A/B en las métricas
   principales; sirve de OOD no colineal.

Pseudo FR también para A (base y H0 compartidos entre A y B). Base congelada en
snapshot, como Bi2Se3 (`freeze_basis` solo snapshot para materiales SOC).
Recuentos, k-mesh, smearing y tamaños los fija F0 por coste medido. Referencia
de Bi2Se3: 63 estructuras en una noche.

**Δ-learning:** H0 de una iteración con `DM.InitSpin` igual a los momentos de
entrada (módulo y ángulos), para que en predicción baste con los momentos del
usuario. El SOC llega exacto por H0 (g(H−H0) = 0 en Bi2Se3); los canales
impares, no: dependen de la densidad de espín. Brazo principal Δ; brazo
directo como comparación (en Bi2Se3 el directo fue 30–40× peor en bandas).
Predecir con Δ exige una iteración SIESTA para H0, como la receta Δ actual; el
brazo directo no necesita SIESTA. graph2mat predice lo que el modelo aprendió
(Δ o H); sumar H0 sigue en los scripts, como en `run_soc_campaign.py`.

**Si F0 rechaza parte del plan con Fe** (patrones invertidos que no sobreviven,
SOC que inclina los momentos, AFM por capas que colapsa): se permite cambiar
clases de configuración dentro de Fe y documentarlo. Cambiar de material no:
se detiene esa rama, se informa al usuario y se sigue con lo independiente.

### D5. Métricas (fijadas antes de ver resultados)

- H por grupo de canales en meV (MAE y máximo): a^0, b^k (SOC), a^k
  (magnético) y b^0 (esperado 0), cada uno aparte. En colineal: ↑, ↓ y la
  diferencia ↑−↓.
- Bandas: RMSE alineado de autovalores en una ventana fija alrededor de E_F
  (ventana decidida en F0) en Γ y k aleatorios con semilla fija. S canónica si
  F0 encuentra S mal condicionada (criterio de `run_soc_campaign.py`).
- **Momentos recalculados:** DM desde el H predicho con el mismo smearing y la
  misma malla k, y Mulliken por átomo; error en μB y en ángulo (grados) frente
  al mismo cálculo con el H de referencia. Métrica física principal del magnetismo.
- Ablación: poner a cero la parte impar de lo que predice la red (Δ en el brazo
  Δ, H en el directo), sin tocar H0. Referencia adicional en Δ: solo H0 (Δ = 0).
- **Control «sin momentos»:** brazo Δ con la red sin m (solo geometría, mismo
  H0 magnético), en A y en B. Es lo que demuestra que los momentos como entrada
  aportan algo más allá de H0. En el brazo directo el control es trivial: sin m,
  la doble pasada anula los canales impares.
- OOD: supercelda mayor (p. ej. 54 átomos) con patrones nuevos y una clase de
  configuración ausente del entrenamiento (p. ej. AFM por capas (001)). Si B'
  sale adelante, una espiral de espín.
- Tres semillas por brazo. Tiempos y VRAM de pico registrados.

## Review Focus

1. Convenios: fórmulas de canales, ejes del espín de SIESTA = ejes cartesianos
   de los datos, signo y factor de los momentos de Mulliken. Todo con oráculos
   y datos reales, nunca solo con tests sintéticos del propio código.
2. Equivariancia conjunta: girar posiciones y momentos (rotaciones propias e
   impropias, m axial) transforma H como D1. T exacto (float64, ≤1e-10).
   Permutaciones de átomos.
3. Cachés y datasets: dos muestras con la misma geometría y distintos momentos
   nunca comparten entrada en caché (cachés de 6.1, ZIP, workers).
4. Compatibilidad: checkpoints antiguos cargan; `spin_h_only` y `spin_orbit`
   sin momentos son bit a bit iguales; acuerdo datos/modelo comprobado en
   `setup` y `on_fit_start` como con `spin_orbit`; CLI que deriva y rechaza.
5. Rechazos explícitos donde no esté certificado: MatrixWriter y servidor HTTP
   con la política completa, opciones de readout no probadas, pérdidas fuera
   de la lista blanca, momentos ausentes cuando la política los exige.

## Task 0 — arranque

- [ ] Leer este documento, la fila 8.1 de `roadmap.md`, el diseño SOC, el traspaso
  de agentes que indique `AGENTS.md` y `git log` desde `d1c2283`.
- [ ] Comprobar HEAD, rama, estado de la cola `g2m_autorun` y GPU.
- [ ] Referencia de tests enfocada (SOC, spin, padding, datos) y registrar el
  conteo observado:
  `pytest src/graph2mat/core/data src/graph2mat/bindings/e3nn src/graph2mat/tools/lightning -q`.

## Task 1 — F0: viabilidad SIESTA con Fe (CPU, informe)

**Files:** `auditorias_mejoras/magnetismo_f0_fe.py` (nuevo),
`auditorias_mejoras/informe_magnetismo_f0_fe_<fecha>.md`. Material `fe` en
`g2m_bench_materials.py`, sin commit en ese repo.

- [ ] Extraer Fe FR del tarball PseudoDojo verificando sha256 y añadirlo a la
  procedencia. Relajar la red y medir el momento (PBE: a ≈ 2,83–2,85 Å, ≈2,2 μB).
- [ ] Convergencia de malla k y smearing (metal) con coste por SCF, con y sin SOC.
- [ ] Sintaxis y marco de `DM.InitSpin` con ángulos: Fe FM inicializado en x da Sx > 0.
- [ ] TSHS SOC magnético legible: 8 componentes + S, Hermiticidad, canales impares
  no nulos y de la escala esperada.
- [ ] Comprobaciones de simetría en datos reales, a la tolerancia del SCF:
  (i) T: invertir todos los momentos da canales pares iguales e impares con
  signo opuesto; (ii) C4 sobre z con m fuera del eje: girar estructura y m da H
  transformado según D1 (patrón de `soc_c3_check.py`); (iii) operación
  impropia (inversión de una celda con rattle, m sin cambiar): valida que m es
  axial, cosa que una rotación propia no distingue; (iv) b^0 = 0.
- [ ] Metaestabilidad: fracción de patrones de espín invertido que conservan
  signo y |m| > 1,5 μB tras el SCF (A y B).
- [ ] H0 de una iteración con momentos: magnitud de Δ por grupo de canales.
- [ ] λmin(S) y auditoría fuera del grafo (colas KB) para decidir 8.2.
- [ ] B': rotación global rígida congelada frente a SCF en ese eje. Decidir si B'
  entra como prueba.
- [ ] Fijar recuentos, tamaños y presupuesto de F4. Commit del informe.

## Task 2 — representación de 8 canales (datos, CPU)

**Files:** `core/data/spin_orbit.py`, `core/data/configuration.py`,
`core/data/sparse.py`, `core/data/processing.py` y sus tests.

- [ ] RED: oráculo de Pauli en float64 (ida y vuelta exacta de los 8 canales,
  signos de transpuesta, Hermiticidad de H(k)); ida y vuelta en el TSHS real de
  F0; con b^0 = a^k = 0 coincide con `spin_orbit`.
- [ ] Implementación mínima reutilizando el hook `component_transform`.
- [ ] GREEN y suite de datos sin regresiones. Commit.

## Task 3 — momentos magnéticos en los datos (CPU)

**Files:** `core/data/configuration.py` (campo y lectura desde SIESTA/DM),
`bindings/torch/data/*`, `tools/lightning/data.py`, cachés y ZIP, `bindings/torch/load.py`.

- [ ] RED: lectura de momentos (Mulliken sisl frente a stdout de SIESTA en F0);
  formas y validaciones (no finitos, forma errónea, ausentes cuando la política
  los exige); batch y collate; claves de caché distintas para la misma geometría;
  rotación de momentos en los helpers de equivariancia de tests; checkpoint
  sin la clave carga igual que antes.
- [ ] GREEN, suite de datos y Lightning. Commit.

## Task 4 — modelo: capa magnética, bloque de 8 canales y T (CPU)

**Files:** `bindings/e3nn/modules/spin_orbit.py`, módulo nuevo para la capa
magnética, `models/mace.py`, `tools/lightning/models/mace.py`,
`tools/lightning/cli.py`, `tools/lightning/callbacks.py`, `server/server_app.py`.

- [ ] RED: equivariancia conjunta (≥8 rotaciones con impropias, float64, orbitales
  s/p/d); T exacto; m = 0 da canales impares nulos; colineal s → −s intercambia
  ↑/↓; permutaciones; alcanzabilidad de todos los canales con las irreps de la
  campaña (como el hallazgo 0o de 8.1); acuerdo datos/modelo y derivación en
  CLI; rechazos de Review Focus 5; checkpoints SOC y no SOC anteriores intactos.
- [ ] Mantener `jit_script_fx=False` para SOC (cuelgue TorchScript de 8.1).
- [ ] Smoke de entrenamiento CPU de pocos pasos con datos sintéticos y con 2–3
  etiquetas reales de F0. Commit.
- [ ] Documentación: sección «Magnetic moments» en `docs/data_loading.rst`.

## Task 5 — F4: dataset Fe (CPU, SIESTA)

- [ ] `fe_samples()` (A y B, y B' si F0 lo aprueba) con splits congelados:
  entrenamiento/validación/test/OOD. Etiquetas, momentos, H0 con momentos y
  auditoría. Térmica controlada; registrar duración y máximos de temperatura.
- [ ] Auditoría: distribución de |m|, ángulos y canales; fuera del grafo; λmin(S).
- [ ] Resumen en roadmap 8.1. Commit del lote de resultados.

## Task 6 — F5: campañas GPU

**Files:** `auditorias_mejoras/run_magnetic_campaign.py` (nuevo; reutilizar
`run_soc_campaign.py` todo lo posible: muestras Δ, evaluación y resumen).

- [ ] Prueba corta de 20 pasos por brazo y estimación de tiempo y VRAM.
- [ ] Colineal (A): Δ y Δ sin momentos. SOC magnético (B): Δ, Δ sin momentos y
  directo. 3 semillas por brazo (15 entrenamientos).
  Receta base de Bi2Se3 (nodos l≤2 de ambas paridades, aristas l≤3, correlación 3,
  2 interacciones, 8000 pasos, batch 2), ajustada solo si la prueba corta lo exige.
- [ ] Métricas de D5 sobre test y OOD, ablación sin canales impares y, si existe, B'.
- [ ] Registrar en roadmap 8.1 y en este documento. Commit del lote.

## Cierre del goal

- Revisión independiente final (revisor sin contexto previo). Hallazgos
  Critical/Important corregidos y verificados.
- Suite completa y Ruff/mypy sin fallos nuevos frente a la referencia de Task 0.
- Fila 8.1 de `roadmap.md` actualizada: implementación verificada separada del
  beneficio medido, límites conocidos (SIESTA sin DFT restringido, B' sin SCF).
- Entrega: por tarea, implementación, pruebas, medidas, commits y límites.
- Completo solo con Tasks 0–6 hechas o con un rechazo motivado con evidencia.
  «Aplazado» o «falta recurso» no cierran una tarea.

## Fuera de alcance

- Magnetismo no colineal sin SOC como política propia (la completa ya lo
  representa). Espirales con `Spin.Spiral`. DFT restringido con otro código.
- Fuerzas magnéticas, energías y dinámica de espín.
- Promover defaults o recetas a otros materiales.

## Registro de ejecución

**2026-10-21 01:00–04:30 (Claude, goal lanzado por el usuario).**
- Task 0: HEAD e328e29. Test roto de 8.2 arreglado (fixture sin padding en el modelo, 37a5589). Suite completa: 2932 pasan, 36 omitidos, 4 fallos conocidos (`test_extrapolation` HistorySOAPs, falta `dscribe`).
- Task 1 (F0, `magnetismo_f0_fe.py`, salidas en `experimentos_magnetismo_f0_fe_20261021/`): Fe FR (sha256 959e0d8b…, en `pseudos_fr/PROVENANCE.json`, 16 e de valencia, 19 orbitales por átomo). Red PBE a = 2,8214 Å, 2,23 μB. Malla: en 16 átomos k6 frente a k8 da 0,045 meV MAE (5 meV máx.); k4 0,22 meV (36 meV máx.); smearing 0,1 frente a 0,05 eV, 15 meV máx. Simetrías en datos reales SOC (2 átomos, k12): T exacta (2·10⁻⁹ eV SCF, 10⁻¹¹ H0); C4 con m fuera del eje 2·10⁻⁹ eV frente a 0,5–1,9 eV con el espín sin girar; inversión con m fijo 3·10⁻⁹ eV frente a 1,0–2,3 eV tratando m como polar (m axial confirmado); b0 ≤ 1,5·10⁻¹⁵ eV. Marco: el momento convergido sigue al nominal (0,013°); suma Mulliken sisl = momento total de SIESTA. `Diag.ParallelOverK T` acelera ×2,5. Colineales de 16 átomos: fm, flip2/6/8 y AFM (001) convergen (78–135 iteraciones); flip1 y flip4 no convergen en 300 (oscilan, dDmax ~5·10⁻³): prueba de mezcla más suave en curso. SOC de 16 átomos k6: ~74 s por iteración, 1,5–2 h por etiqueta (diagonalización 75 %).
- **Temperatura:** con 3 SIESTA de 4 procesos más tests en paralelo se leyeron picos de 82–86 °C al final de algunos cálculos (regla < 80 °C). Para F4: menos concurrencia y arranque solo por debajo de 76 °C.
- Plan de código `docs/superpowers/plans/2026-10-21-magnetismo-codigo.md`, desarrollo con subagentes: Task 1 de código `spin_orbit_full` (4d8b56e, revisión aprobada); Task 2 momentos en datos (ec87d85, aprobada); Task 3a capa magnética, bloque de 8 canales y doble pasada (83fefa6, en revisión). La cuota de la sesión se agotó a las 03:50 y se reanudó a las 04:18 con el latido.

**2026-10-21 04:30–06:40.** F0 cerrado: [informe](informe_magnetismo_f0_fe_20261021.md). Momentos desde H validados (10⁻⁵ μB colineal, 7·10⁻⁵ μB SOC). Fuera del grafo 0,03 meV (sin 8.2); S con λmin 7,6·10⁻⁵ (bandas también con S canónica); rotación congelada coherente a 6 meV (B' viable como prueba). SCF: los patrones con espines invertidos a veces no convergen (2 de 5 colineales, el SOC con 3 inversiones): F4 usa 10⁻⁵, como mucho 120 iteraciones y excluye y cuenta los que fallan. F4 lanzado 06:15: A colineal (42 etiquetas, 2 trabajadores) y B SOC (33, k4, mitad ferromagnetos, 1 trabajador), arranque bajo 76 °C. `g2m_autorun` pausó un SIESTA a 85 °C a las 06:07 (validación en P-cores + tests del subagente a la vez): desde entonces nada pesado en P-cores mientras corre SIESTA. Script de campaña `run_magnetic_campaign.py` (brazos, métricas D5, momentos desde H) escrito y probado en datos F0. Código: Task 3b (Lightning/CLI) en ronda de corrección por la normalización de vecinos al cargar checkpoints.
