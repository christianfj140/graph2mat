# Informe de evolución de Graph2Mat — rama `g2m-v2`

**Fecha de corte:** 2026-10-06  
**Estado auditado:** `g2m-v2` local en `e0a88d6ae1ab1971aecebc2cd9d6b366f48fc81b`  
**Baseline:** `4cc8fd19f3d17e569ec56a4f24d1a65d306b394a`  
**Repositorio:** `<fork>/graph2mat`

## Alcance, método y niveles de evidencia

Esta auditoría reconstruye la evolución técnica, matemática y científica de la rama, no su narrativa de Git. Se contrastaron, por este orden:

1. código y `diff` frente al ancestro común;
2. artefactos experimentales (`result.json`, resúmenes y predicciones) del repositorio compañero `MD_vs_AtomicDisplacement`;
3. tests actuales y registros de tests anteriores;
4. commits y PR de GitHub;
5. `auditorias_mejoras/roadmap.md` y otros informes versionados;
6. el hilo de Claude que creó `g2m-v2` y los chats posteriores.

Las cifras científicas se expresan en meV salvo indicación. «H alineado» significa que se elimina por estructura el desplazamiento de gauge `cS`; no es intercambiable con el MAE bruto. Cuando una conclusión procede sólo del chat se marca explícitamente. No se ejecutó ni se modificó trabajo de fonones durante esta auditoría.

### Advertencia crítica de reproducibilidad

El remoto de GitHub no contiene el estado auditado completo. Al corte:

- `origin/main = 4cc8fd1`;
- `origin/g2m-v2 = a5e8134`;
- `g2m-v2` local `= e0a88d6`, **78 commits por delante** del remoto;
- GitHub sólo registra los PR `#1–#6`, correspondientes al soporte inicial de H y multicomponente.

Por tanto, casi todas las mejoras de septiembre–octubre, el CI, los tests nuevos y el roadmap sólo son reproducibles desde la copia local hasta que se publique la rama. Los artefactos cuantitativos principales tampoco viven en este repositorio, sino bajo `MD_vs_AtomicDisplacement/Comparison/results/dataset_design_curves_v1/`.

## 1. Resumen ejecutivo

`g2m-v2` convirtió un soporte mínimo de Hamiltonianos, esencialmente escalar, en un pipeline capaz de leer y reconstruir H multicomponente, H colineal y `(H,S)`, entrenar H con S sólo como referencia de gauge, usar readouts e3nn más expresivos, acelerar el backbone MACE y ensayar física de largo alcance y carga. El avance científico dominante no fue una red más compleja: fue cambiar el objetivo. La combinación de gauge físico y, sobre todo, Δ-learning respecto a un H0 de SIESTA redujo errores de H del orden de 80–90 % en varios bancos. El avance de ingeniería dominante fue `torch.compile`, con paridad numérica y una reducción de tiempo dependiente del tamaño y de la carga.

No todas las cifras de portada son propiedades del paquete por defecto. La mejor receta requiere configuración explícita; Δ-learning y gran parte de la evaluación OOD están implementados en scripts del repositorio compañero, no en la API central de Graph2Mat. La inferencia Δ tampoco es puramente ML: requiere construir H0 con una iteración de SIESTA en la geometría objetivo.

| Modificación | Problema que resuelve | Resultado principal | Evidencia | Estado |
|---|---|---|---|---|
| Pipeline H/S/spin multicomponente | Se leía sólo la columna 0 y se perdían S/spin | Lectura, entrenamiento y serialización explícitas para H, `(H,S)` y spin colineal | código, tests de componentes y sparse | Infraestructura validada técnicamente; spin sin validación científica |
| Correcciones de simetría, batching y seguridad | Bloques/edges incorrectos o fallos silenciosos | Reconstrucción hermítica, validaciones fail-closed y carga de checkpoints robusta | código + tests | Validada técnicamente |
| Readout lineal e3nn | El readout de aristas era una forma homogénea cuadrática | `w90` 12,61→5,09; mejora también `6x6` y H2O; nuevo default | 3–5 semillas y tests de equivariancia | Validada |
| Conservación de irreps de la última capa (`kl`) | Última interacción MACE sólo escalar | Mejora fuerte mono→bicapa; neutra en H2O; +30 % tiempo en una receta | campañas 5 semillas | Prometedora/material-dependiente |
| Pérdida invariante de gauge | El cero de energía por estructura no es aprendible localmente | `6x6` 1,05→0,70; H2O 29,6→5,6; bandas bicapa→moiré 14,5→7,3 | resultados multisemilla + test analítico | Validada |
| Δ-learning sobre H0 de SIESTA | H absoluto contiene una parte fácil/física que domina la escala | `6x6` 0,71→0,14; H2O 5,6→1,0; LiF/BTO/H2O −79 a −91 % en validación | resultados crudos y bancos multisemilla | Claramente útil; dependencia externa de SIESTA |
| Pérdida espectral Γ | H-MAE no garantiza bandas | Correcta matemáticamente, pero no mejora bandas OOD y cuesta ~40 % más | tests + 3 semillas | Descartada para mono→bicapa en esta forma |
| Penalización del espacio casi nulo de S | Bases difusas amplifican errores pequeños de H | H2O32: gap bruto 1847→260 y 3/9→0/9 gaps cerrados; val H +17 % | 3 semillas | Prometedora con compromiso |
| `output_scaleshift` equivariante | Escalas muy distintas por bloque/irrep | BTO −17 %; H2O neutra; LiF +25 % peor | 3 materiales, 3 semillas | Material-dependiente; no default |
| Actualización de aristas | Readout separable, sin dependencia conjunta de ambos extremos | H2O −17 % val; neutral globalmente en LiF/BTO/grafeno | tests + bancos | Experimental/material-dependiente |
| Envolvente radial por par | Discontinuidad en `R_i+R_j` | Neutra en cinco sistemas | tests + bancos | Descartada como mejora general |
| MACE ligero, checkpointing y proyección | Coste/VRAM del backbone y readout | `lean_mace` −14 % pico; checkpoint −24/−34 % pico con +34 % tiempo; proyección más rápida pero pierde precisión | perfilado + regresiones | Infraestructura/opciones |
| Layout por tipo + `torch.compile` | Sincronizaciones host y miles de kernels | 23→0 sync; paso 138→70 ms; −25 % total limpio, hasta ×2,8 con carga variable; misma precisión | tests eager/compiled + benchmarks | Validada, con límites |
| Largo alcance y Qeq | Campos, superficies y estados de carga fuera del radio local | Cargas fijas rompen ZnO polar; Qeq evita ese fallo y mejora casos concretos, no universalmente | tests físicos + 4 materiales | Experimental/prometedora |
| `pair_difference_mse` | MSE de H no supervisa bien la pendiente `dH/dR` | λ=10 mejora breathing pero no alcanza el gate <10 % | código/tests + campaña externa | Implementada; insuficiente para EPC breathing |

### Estado cuantitativo del roadmap

El roadmap contiene 80 puntos únicos (las repeticiones del apéndice no cuentan):

| Estado editorial | Puntos | Fracción |
|---|---:|---:|
| ✅ completados/validados según el ledger | 23 | 28,75 % |
| 🔶 parciales, opcionales o experimentales | 30 | 37,50 % |
| ⚪ medidos como neutros/inconclusos | 6 | 7,50 % |
| ❌ descartado explícito | 1 | 1,25 % |
| ⏳ pendientes | 20 | 25,00 % |

Así, el **75 % está abordado** de alguna forma. Si se da crédito completo a decisiones cerradas (✅/⚪/❌) y medio crédito a 🔶, el avance efectivo es **56,25 %**. Esto mide ejecución del roadmap, no validación científica: sólo 23/80 figuran como ✅.

## 2. Estado base de Graph2Mat

### Identificación del baseline

El ancestro común de `origin/main` y `g2m-v2` es `4cc8fd1` (2026-04-16), merge del PR #1, «Minimal Hamiltonian support: IO, mapping, sub_point guard, and metric guard». `main` no tiene commits posteriores; por tanto no hay que incorporar ningún cambio adicional al baseline.

### Qué podía hacer

El baseline ya disponía de la idea central de Graph2Mat:

- representación de una matriz orbital por bloques on-site `M_ii` y off-site `M_ij` asociados a nodos y aristas;
- cambio de base e3nn desde coeficientes irreducibles a bloques orbitales;
- wrapper de MACE como extractor equivariante;
- entrenamiento Lightning para DM/EDM y soporte mínimo de H;
- reconocimiento de `sisl.Hamiltonian`, salida `.TSHS` y desactivación de la resta de matriz atómica para H.

### Limitaciones relevantes

1. **Una sola componente.** `csr_to_block_dict` usaba `spmat.data[:,0]`; S, spin down y componentes NC/SOC se descartaban. La reconstrucción llamaba a `fromsp(geometry, csr)` sin S ni spin.
2. **Ambigüedad física.** No existía una política que distinguiera `H`, `(H,S)`, canales de spin o componentes crudas.
3. **Objetivo y métrica débiles.** El default era `block_type_mse`, sin tratamiento del gauge `H→H+cS`, espectro, S mal condicionada o pendientes.
4. **Readout limitado.** El bloque de arista era sólo tensor product cuadrático; no había término lineal. La última interacción MACE perdía irreps `l>0` por su diseño para energías.
5. **Rendimiento.** MACE calculaba salidas de energía y derivadas de posiciones innecesarias para H; no había checkpointing, layout precomputado ni compile.
6. **Robustez.** Existían fallos en emparejamiento de aristas, batching simétrico, métricas, serialización `(H,S)` y carga con Torch moderno.
7. **Sin evidencia OOD sistemática.** No había un banco congelado de monocapa→bicapa/moiré ni de H2O/LiF/BaTiO3/ZnO.

Evidencia: `git show 4cc8fd1:src/graph2mat/core/data/sparse.py:103`, `.../processing.py:100-133`, `.../tools/lightning/models/mace.py:30-114`; PR #1 de GitHub.

## 3. Objetivos reconstruidos de `g2m-v2`

El primer mensaje sustantivo del hilo de Claude pidió «mejorar la capacidad de predicción de matrices, sobre todo Hamiltoniano, pero las demás también» y «reducir VRAM y velocidad de entrenamiento». Después se añadieron requisitos de extrapolación de grafeno a bicapa/moiré, prueba en H2O, autonomía con límites térmicos/OOM y trazabilidad de cada decisión.

El código, commits y benchmarks concretaron esos objetivos en cinco preguntas:

1. ¿Puede el pipeline representar y escribir correctamente H, S y spin?
2. ¿Qué parte del error viene del readout, del objetivo, de la optimización o de física ausente?
3. ¿Las mejoras sobreviven fuera del conjunto de entrenamiento y en otros materiales?
4. ¿Se puede reducir coste sin cambiar la función aprendida?
5. ¿La precisión de H basta para observables espectrales y derivadas?

El chat sólo aporta la motivación y la secuencia de hipótesis. Las conclusiones que siguen se basan en código/resultados; las afirmaciones tempranas del chat que fueron corregidas no se conservan como evidencia.

## 4. Modificaciones realizadas

## 4.1. Pipeline de Hamiltonianos multicomponente, overlap y spin colineal

### Estado anterior

El baseline leía exclusivamente la primera columna CSR y reconstruía una matriz escalar. Un `.TSHS` no ortogonal o polarizado se reducía silenciosamente.

### Problema y motivación

Para resolver `HC=εSC`, calcular bandas o definir el gauge de H hace falta S. Para spin colineal hacen falta `H↑` y `H↓`. Mezclar estas columnas sin una política explícita podía producir un objeto físicamente falso aunque el tensor tuviera dimensiones válidas.

### Cambio realizado

Se introdujeron:

- `n_matrix_components` de extremo a extremo;
- `matrix_component_policy ∈ {h_only, spin_h_only, h_and_overlap, raw_components}`;
- detección de `S_idx`, validación del número de componentes y errores fail-closed;
- bloques con eje final de componentes y transposición que no transpone dicho eje;
- serialización a `sisl.Hamiltonian` de `(H,S)`, `(H↑,H↓)` y `(H↑,H↓,S)`;
- inferencia de componentes desde el primer dato y sincronización modelo–DataModule;
- salida de coeficientes irreducibles y metadatos por operación.

El tensor de un bloque pasa conceptualmente de `B_ij ∈ R^(n_i×n_j)` a `B_ijc ∈ R^(n_i×n_j×C)`. Para simetría orbital se aplica `B_jic = B_ijc^T` manteniendo `c`.

### Verificación de implementación

Hay tests de extracción, políticas, transposición, inferencia CLI, reconstrucción sparse, H/S y coeficientes. La auditoría focal actual incluye estos caminos y pasó dentro de **170 tests**. El diagnóstico H2O confirmó además que proyectar H a coeficientes y reconstruirla produce `0,000127 meV` de MAE numérico.

### Evaluación científica y resultados

La capacidad `(H,S)` habilitó la pérdida de gauge y todos los observables generalizados posteriores. Predecir S conjuntamente no fue la receta ganadora: en H2O, H alineado fue 7,2 meV prediciendo H+S frente a 5,6 prediciendo sólo H, y el pico de VRAM fue 2160 frente a 1372 MiB; en `6x6`, 0,86 frente a 0,70 meV y +48 % de VRAM.

No se encontró un benchmark consolidado de **S-MAE**, ni validación científica de spin colineal, NC o SOC. `raw_components` preserva datos, pero deliberadamente no se serializa como Hamiltoniano físico.

### Interpretación

La mejora esencial es semántica: impide confundir columnas y permite usar S sin obligar a aprenderla. El mejor patrón actual es H-only con S de referencia/H0 para gauge y espectro. Esto no equivale a predecir S desde geometría.

### Estado y evidencia

**Infraestructura validada técnicamente; H/S útil como soporte; predicción de S y spin no validadas científicamente.**  
Commits: `1d52abb`, `535c01b`, `db74d63`, `ec4f742`, `467ffb4`, `9fb7f82`, `4efeb1a`, `1a131f1`, `72b468c`, `71c1afa`, `15f98d0`.  
Código: `configuration.py:652-732`, `processing.py:107-160`, `sparse.py:457-526`, `matrixblock.py:65-106`.  
Tests: `test_matrix_components.py`, `test_sparse.py`, `test_multicomponent_blocks.py`, `test_cli.py`.

## 4.2. Correcciones de simetría, batching, métricas y carga

### Estado anterior y problema

Había slicing incorrecto de edges simétricos al agrupar grafos, representación frágil de la arista única, `symmetric_matrix=False` aunque H/DM/EDM son simétricas, re-suma incorrecta de la DM atómica en una métrica, divisiones por cero en pérdidas relativas, límites de supercelda no validados y carga de objetos Lightning incompatible con el cambio `weights_only` de Torch ≥2.6.

### Cambio realizado

- Emparejamiento explícito de direcciones homo/heteronucleares y selección robusta de representantes.
- Deducción de simetría por tipo de matriz y una sola llamada para ambas direcciones (`symm_transpose`).
- Corrección de métricas, máscaras antes de dividir y agregación por estructura.
- Validaciones de shells, paridad de bases, alcance de supercelda, `initial_node_feats` y parámetros sin grupo del optimizador.
- `torch_predict` crea `batch` por nodo, no un escalar.
- Carga de checkpoints confiables con `weights_only=False` y preservación de `basis_table` cuando se desactivan defaults.
- Workflow CI de pytest.

### Verificación y resultado

Son correcciones de validez, no hipótesis de precisión. Se verifican mediante regresiones de simetría `M_ij=M_ji^T`, reordenación, gradientes finitos, round-trip sparse, batches vacíos, pickle/copy y tres regresiones del loader. En la ejecución actual, los seis ficheros focales sumaron **170 passed**. Antes de los dos últimos commits se registró una suite completa de **491 passed, 2 skipped**; la suite completa actual se volvió a lanzar durante esta auditoría (resultado en §11).

### Estado y evidencia

**Validada técnicamente.** No debe contabilizarse como reducción de H-MAE, sino como eliminación de resultados silenciosamente incorrectos.  
Commits principales: `9d8cc0c`, `cacb938`, `7648951`, `b7c612d`, `72b468c`, `19057a4`, `cae9955`.  
Código/tests: `core/modules/matrixblock.py:85-128`, `tools/lightning/data.py:68-154`, `test_edge_pairing_and_safety.py`, `test_metrics.py`, `test_cli.py`.

## 4.3. Diagnóstico de representabilidad y coeficientes irreducibles

### Estado anterior y problema

Un H-MAE alto podía deberse a tres causas distintas: que el cambio de base no pudiera representar el target, que el readout compartido no tuviera suficientes direcciones paramétricas o que la optimización no encontrase la solución. Sin separar esas causas era fácil aumentar datos o ancho sin atacar el cuello real.

### Cambio realizado

Se añadió retorno de coeficientes por bloque/operación, pérdidas en espacio de coeficientes y una batería de diagnósticos de rango exacto, rango jacobiano y readout denso sobre un checkpoint H2O.

### Verificación y resultados

| Diagnóstico H2O | Resultado |
|---|---:|
| Dimensión de coeficientes | 276 |
| Rango paramétrico del readout compartido | 174/276 |
| Rango jacobiano de todos los parámetros | 196/276 |
| Rango del readout denso diagnóstico | 276/276 |
| Reconstrucción H con readout denso | 0,000783 meV |

El target sí cabe en la base irreducible; el cuello es la controlabilidad del readout compartido. Sin embargo, `coefficient_space_mse` falló con un mapping real `BasisTableWithEdges.group`, de modo que la pérdida de producción no quedó validada para ese checkpoint.

### Interpretación y estado

Esto justifica buscar un término lineal/denso o mejor parametrización, pero no convertir el readout denso diagnóstico en default sin medir coste/generalización. **Diagnóstico validado; pérdida de coeficientes inconclusa por incompatibilidad.**  
Commits: `4efeb1a`, `1a131f1`, `a5e8134`.  
Evidencia: `Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/report.md:5-13,47-107,119-127`.

## 4.4. Readout lineal e3nn y capacidad de MACE

### Estado anterior

`E3nnSimpleEdgeBlock` calculaba sólo productos tensoriales bilineales de los mensajes. Era una forma homogénea cuadrática: no podía expresar una contribución lineal aunque las features ya contuvieran la señal. Además, la última interacción MACE estándar está diseñada para energía y conserva sólo escalares.

### Cambio realizado

1. `E3nnLinearEdgeBlock` suma `Linear(a+b)` al tensor product, sin bias para que el bloque se anule al corte.
2. Existe una variante lineal+tensor-square para nodos y una variante proyectada de menor coste.
3. `keep_last_layer_irreps` construye una interacción adicional y elimina la última, conservando `l>0` en la capa profunda.
4. Se añadieron `edge_hidden_irreps`, `mp_cutoff`, readout por interacción, simplificación de irreps y contexto local experimental.

### Verificación de implementación

Tests de equivariancia, localidad, simetría, irreps inalcanzables, paridad del modelo completo y gradientes. `E3nnLinearEdgeBlock` es el default desde `a39439a`; los checkpoints antiguos conservan su clase guardada.

### Evaluación científica

| Sistema/receta | Antes | Después | Lectura |
|---|---:|---:|---|
| `w90`, H | 12,61 | 5,09 | −60 %, 3 semillas |
| `w90`, bandas | 18,9 | 13,5 | mejora |
| `6x6`, bf16 | 1,94 | 1,79 | −7 %, ruidoso |
| `6x6`, fp32+poly6 | 1,94 | 1,51 | −22 % |
| H2O receta completa | 55,3 | 48,7 | −12 %, 3/3 semillas |

El término lineal de aristas contiene prácticamente toda la ganancia: en `w90`, aristas-only 4,52 meV frente a nodos-only 13,67. El nodo proyectado redujo GPU 126→54 ms/paso, pero en bf16 empeoró `6x6` a 2,32 meV; se descartó para precisión.

`keep_last_layer_irreps` fue decisivo para extrapolación mono→bicapa (H 62,8→40,4; 5/5 semillas), pero no en H2O (5,6→5,9 y gap 80→101) y añadió ~30 % de tiempo en esa comparación.

### Interpretación y estado

El readout lineal corrige una restricción funcional real y mostró ganancias en tres dominios: **validado y default**. Conservar irreps profundas es **prometedor/material-dependiente**, no un default universal. La proyección es una opción de rendimiento con pérdida de precisión.

Commits: `a0f6d18`, `bd1d64d`, `a39439a` y commits de resultados `7f93d4d`, `e77b836`, `fe49168`.  
Código: `edge_operations.py:35-87`, `models/mace.py:22-94`, `tools/lightning/models/mace.py:47-102`.  
Resultados: `roadmap.md:15-24,34-67,193-220`; resúmenes `g2m_improvements/{summary_6x6,h2o_summary}.json`.

## 4.5. Pérdida elemento a elemento y estrategia de optimización

### Estado anterior y problema

El baseline ponderaba promedios por tipo de bloque. Eso podía dar el mismo peso a grupos con números de elementos muy distintos. EarlyStopping detenía algunas semillas aproximadamente a un tercio del coseno, y comparar por épocas hacía depender el presupuesto del tamaño de N.

### Cambio realizado

La pérdida principal pasó a ser

\[
L_\mathrm{elem}=\frac{N_n\,\mathrm{MSE}_n+w_eN_e\,\mathrm{MSE}_e}{N_n+w_eN_e},
\]

donde `w_e=2` cuenta que cada bloque de arista simétrico almacenado representa también su transpuesta. Se añadieron `loss_kwargs`, scheduler/etapas de entrenamiento, logging ponderado y selección explícita del checkpoint. La receta de alta precisión usa 16.000 actualizaciones y no EarlyStopping.

### Verificación y resultados

El test unitario comprueba exactamente el peso de aristas. Quitar EarlyStopping, sin cambiar el modelo, redujo `6x6` 1,94→1,69 meV (−13 %). Con readout lineal y 16k:

- bf16: 1,11 meV (5 semillas);
- fp32: 1,02 meV (5 semillas).

Duplicar a 32k dio sólo 0,71→0,68 meV con gauge, dentro del ruido y al doble de tiempo. Gradient clipping fue neutro; EMA empeoró; SWA no fue viable por `deepcopy` del modelo.

### Interpretación y estado

La ponderación elemento a elemento y retirar un EarlyStopping prematuro son mejoras de entrenamiento bien respaldadas. Más pasos sin señal de validación no lo son. **Validada la pérdida/default y el cambio de stopping; descartadas 32k/EMA como receta general.**

Commits: `b7c612d`, `e77b836`, `4ef0d25`; código `metrics.py:1561-1600`; tests `test_metrics.py:71-88`.

## 4.6. Pérdida invariante de gauge usando S como etiqueta

### Estado anterior

Se minimizaba el error absoluto de H aunque el cero de energía de cada cálculo SIESTA varía con `E_F`/potencial medio. Un modelo local no puede deducir un escalar global de la estructura; terminaba gastando capacidad en algo no identificable.

### Problema y motivación física

En base no ortogonal, desplazar todas las energías por `c` equivale a

\[
H' = H+cS,
\]

sin cambiar los autovectores y desplazando uniformemente los autovalores. Penalizar `cS` como error físico mezcla forma del Hamiltoniano y convención del cero.

### Cambio realizado

Para cada estructura y componente de H se calcula en forma cerrada

\[
c^*=\frac{\sum_k w_k\,\Delta H_k S_k}{\sum_k w_k S_k^2},\qquad
L_H=\frac{1}{N}\sum_k w_k(\Delta H_k-c^*S_k)^2.
\]

Los `k` recorren elementos on-site/off-site y `w_k` incorpora `edge_weight`. El modelo puede:

- predecir `(H,S)` y añadir MSE de S; o
- predecir sólo H con etiquetas `(H,S)`, usando S únicamente para eliminar el gauge.

La segunda opción reduce la cabeza a la mitad y fue la mejor.

### Verificación de implementación

Un test construye desplazamientos `cS` independientes por estructura y comprueba pérdida cero, gradientes y el modo H-only. El layout precomputado conserva el cálculo por estructura sin sincronizaciones host.

### Evaluación científica y resultados

| Dominio | MSE/receta previa | Gauge H-only | Cambio |
|---|---:|---:|---:|
| `6x6`, H alineado, 5 seeds | 1,05 | 0,70 | −33 %, gana 5/5 |
| H2O, H alineado | 29,6 | 5,6 | −81 % |
| H2O, error de gap | 986 | 80 | −92 % |
| bicapa→moiré, bandas alineadas | 14,5 | 7,3 | −50 %, 3/3 |
| mono `6x6`→bicapa, H | 62,8 | 53,2 | −15 % |

En el último caso las bandas empeoraron en 3/5 semillas: mejorar H alineado no garantiza el espectro OOD. Predecir además S fue peor y más caro.

### Interpretación

Es una corrección de identificabilidad, no sólo una regularización. También explica por qué los JSON que registran `H_MAE_meV` bruto muestran cientos o miles de meV para modelos gauge: ese valor incluye el cero libre. Para esos modelos son válidas `H_MAE_gauge_meV` y bandas alineadas; comparar el bruto con una MSE convencional sería incorrecto.

### Estado y evidencia

**Validada.** Requiere S en las etiquetas o en la ruta de predicción, aunque no se prediga.  
Commits: `7b97b7a`, `71c1afa`, `31d7bcc`.  
Código: `metrics.py:1709-1782`. Tests: `test_metrics.py:107-150`.  
Resultados: `roadmap.md:38-58,532`; `h2o_summary.json` y `result.json` individuales con `H_MAE_gauge_meV`.

## 4.7. Δ-learning sobre H0 de SIESTA

### Estado anterior y problema

La red aprendía H absoluto. Sus on-sites son de eV, mientras las variaciones químicas relevantes pueden ser meV. Una gran parte de H ya la describe la densidad atómica superpuesta, pero el modelo debía redescubrirla.

### Cambio realizado

El script externo `Comparison/scripts/g2m_delta.py` genera, para la misma geometría y base, un cálculo SIESTA no autoconsistente de una iteración:

\[
\Delta H(R)=H_\mathrm{SCF}(R)-H_0(R),\qquad
H_\mathrm{pred}(R)=H_0(R)+\Delta H_\theta(R).
\]

Se comprueba la coincidencia geométrica y el round-trip. H0 no depende del peso de mixing. En `6x6` tarda ~18 s. No se añadió una clase Δ al core: el dataset y la reconstrucción final se gestionan en el repositorio compañero.

### Verificación de implementación

Round-trip exacto H→ΔH→H, hashes/manifiestos de geometría, predicciones que vuelven a sumar H0 y evaluación sobre H final. Los bancos congelan base, pseudopotenciales y splits.

### Evaluación científica

| Sistema/tarea | Sin Δ | Con Δ | Cambio |
|---|---:|---:|---:|
| `6x6`, H alineado | 0,71 | 0,14 | −80 % |
| `6x6` + `kl`, 5 seeds | 0,75 | 0,130 | −83 % |
| `6x6`→bicapa, H | 52 | 6,3 | −88 % |
| `6x6`→bicapa, bandas | 156 | 19 | −88 % |
| H2O, H alineado | 5,6 | 1,01 | −82 % |
| H2O, gap | 80 | 9,6 | −88 % |
| bicapa→moiré, H/bandas | 5,64 / 7,3 | 0,96 / 2,0 | −83 / −73 % |
| H2O clusters, val/(H2O)32 | 19,75 / 9,12 | 4,16 / 2,40 | −79 / −74 % |
| LiF, validación | 20,3 | 1,73 | −91 % |
| BaTiO3, validación | 14,4 | 2,44 | −83 % |
| ZnO, validación/volumen72 | 10,6 / 6,2 | 1,66 / 0,90 | −84 / −86 % |

En `w90`, Δ permitió pasar de 150–450 meV a 4,5 meV al extrapolar una celda de 2 átomos a bicapa. Con celdas pequeñas de grafeno/hBN válidamente emparejadas, el moiré de 28 átomos pasó de 13,3 meV H y 876 meV de bandas a 1,26 y 19,6.

### Interpretación y límites

Δ es la mejora más consistente entre materiales porque reduce rango dinámico e incorpora parte de la física electrostática/atómica. Pero hay tres límites:

1. requiere SIESTA, base y pseudopotenciales disponibles en inferencia;
2. H0 ya ve la geometría objetivo, por lo que no es una extrapolación puramente aprendida;
3. no resuelve física ausente: nanocubos, defectos cargados, niveles en el gap y superficies polares siguieron fallando.

La conclusión del chat de fonones de que Δ no daba una ventaja estable en **número de etiquetas** pertenece a otro diseño de dataset/métrica y no invalida estas reducciones de H; sí impide afirmar que Δ reduzca automáticamente el coste de etiquetado SCF.

### Estado y evidencia

**Validada científicamente para precisión H/bandas en los sistemas medidos; dependencia externa que impide llamarla solución ML autónoma.**  
Commits de documentación/resultados: `6449ddc`, `0888107`, `58b8df3`, `6df452a`, `846dc00`, `7ce970e`, `08619f0`, `09a7cef`, `f0a84d8`.  
Scripts/artefactos: `g2m_delta.py`, `g2m_bench_benchmark.py`, `g2m_ood_eval.py`, `g2m_improvements/*_summary.json`.

## 4.8. Objetivos espectrales y métrica de S

### Estado anterior y problema

Un MAE pequeño y uniforme en H no garantiza autovalores correctos cuando S está mal condicionada o cuando el observable se concentra en un subespacio pequeño. H2O32 mostró estados espurios en el gap aunque H fuera 2,4 meV.

### Cambio realizado

#### Pérdida espectral de primer orden

Con autovectores de referencia S-ortonormales `C` en una ventana de Γ se penaliza

\[
L_\mathrm{spec}=\left\|C^T\Delta H(\Gamma)C-overline{\mathrm{diag}}\,I\right\|_F^2.
\]

La diagonal aproxima errores de autovalor de primer orden; los off-diagonal, mezcla dentro de la ventana. Restar la media la hace invariante a `cS` porque `C^TSC=I`.

#### Penalización del subespacio casi nulo

Para autovectores `u` de S con `λ<10^-2`, se usan direcciones `u/sqrt(max(λ,10^-3))`. Esto aproxima la métrica de Löwdin `S^-1/2 ΔH S^-1/2`, que amplifica justo los errores peligrosos para el problema generalizado.

### Verificación

Tests ensamblan H(Γ), comprueban invariancia a `H+cS`, reproducen cambios de autovalor y validan estructuras homo/heteronucleares, imágenes y batches. Los vectores espectrales se precomputan en `spectral_window.npz`/`spectral_null.npz`.

### Resultados

La pérdida espectral Γ en `6x6` no mejoró bandas OOD:

| Peso | Val H | Bicapa H/bandas | Moiré H/bandas | Tiempo |
|---:|---:|---:|---:|---:|
| 0 | 0,136 | 4,43 / 15,1 | 0,86 / 9,7 | ~45 min |
| 1 | 0,134 | 5,38 / 15,3 | 1,11 / 10,8 | 62–64 min |
| 10 | 0,166 | 4,27 / 20,3 | 0,84 / 13,2 | 62–64 min |

La ventana de monocapa en Γ no contiene el acoplamiento entre capas que domina el test.

En H2O32, la penalización casi nula sí cambió el espectro bruto:

| Peso | Val H | H2O32 H | Error gap bruto | Gaps cerrados | Bandas brutas |
|---:|---:|---:|---:|---:|---:|
| 0 | 4,16 | 2,40 | 1847 | 3/9 | 386 |
| `10^-4` | 4,86 | 2,60 | 260 | 0/9 | 121 |
| `3·10^-5` | 4,52 | 2,46 | 603 | 1/9 | 169 |

### Interpretación y estado

- Espectral Γ: **implementada y matemáticamente validada, descartada para extrapolación mono→bicapa/moiré en esta forma**. Podría reevaluarse con sistema objetivo o más puntos k.
- Métrica de S: **prometedora** si importa el espectro sin ortogonalización canónica, con compromiso explícito de H. No es una mejora universal.

Commits: `496d792`, `8aa6a43`, `46bc937`.  
Código: `metrics.py` (`_spectral_window_loss`, `gauge_invariant_mse`), `tools/lightning/data.py` (`spectral_window`).  
Artefactos: `bench_h2oclu_summary.json`.

## 4.9. Normalización equivariante, actualización de aristas y envolvente por par

### 4.9.1. `output_scaleshift`

Se ajusta por tipo de bloque e instancia de irrep una escala RMS. Sólo los `0e` on-site reciben shift; desplazar otra irrep rompería equivariancia y desplazar una arista impediría decaimiento a cero:

\[
y_{\alpha lm}=\mu_{\alpha}\,\delta_{l0,+}+\sigma_{\alpha l}\,f_{\theta,\alpha lm}.
\]

Tests verifican equivariancia y que el modo H-only con etiquetas `(H,S)` ajuste sólo H.

| Material | Efecto |
|---|---|
| `6x6` | val 0,136→0,117 (−14 %), pero bandas OOD +17–27 % |
| H2O clusters | neutro |
| LiF | val 1,73→2,17 (+25 %), gap nanocubo +3,1→+5,2 eV |
| BaTiO3 | val 2,44→2,02 (−17 %), tests −19 % |
| ZnO | val −28 %, espectro OOD peor |

**Estado:** implementada; útil dentro de distribución en algunos materiales, no default.

### 4.9.2. Actualización residual de aristas

El mensaje se actualiza como `m←m+act(Linear(TP(m,P(h_receptor));w(r)))`, introduciendo dependencia conjunta de emisor y receptor. Tests cubren receptor, equivariancia, corte y simetría.

- H2O clusters: val 4,16→3,47 (−17 %), H2O32 2,40→2,24 (−7 %).
- LiF/BaTiO3: neutral globalmente; cambia favorablemente F neutro y empeora F+1.
- Grafeno: baja error interlayer 9–16 %, pero sube intralayer y no mejora H total.

**Estado:** prometedora sólo en dominios con términos de tres centros relevantes; experimental.

### 4.9.3. Envolvente por par

Multiplica features radiales por una envolvente que se anula en `R_i+R_j`, además del corte global. Es técnicamente correcta y testeada, pero fue neutra o peor en grafeno, grafeno/hBN, H2O, LiF y BaTiO3. **Descartada como mejora general; se conserva como opción.**

Commits: `a5e8134`, `15f98d0`, `63e7608`, `cc261fb`, `26e48b9`, `762217c`, `0c9b235`, `37e6093`, `2c09c0b`.  
Código: `bindings/e3nn/modules/matrixblock.py:65-150`, `bindings/e3nn/modules/graph2mat.py:276-303`, `models/mace.py:97-102,517-521`.

## 4.10. Supervisión de diferencias finitas (`pair_difference_mse`)

### Problema e hipótesis

Dos errores iguales en `H(+δ)` y `H(-δ)` pueden dar buen H absoluto pero una pendiente mala. Para dH/dR se añadió una señal sobre la parte antisimétrica:

\[
L=L_\mathrm{elem}+\lambda\left\{[H_+^\theta-H_-^\theta]-[H_+^\mathrm{ref}-H_-^\mathrm{ref}]\right\}^2.
\]

### Implementación y seguridad

Los pares se identifican por desplazamiento minimum-image en una ventana. Matches ambiguos fallan; `expected_pairs` impide entrenar silenciosamente sin pares. El término sólo se aplica en training, porque estructuras próximas de validación se emparejaban espuriamente.

### Resultado científico

En breathing/shear de bicapa, añadir pares como datos o `λ=1` dejó el error breathing alrededor de 22,5 %. `λ=10` lo redujo, y una campaña posterior llegó a `12,4±5,4 %`, pero siguió por encima del gate preregistrado de 10 %. La señal breathing (~3 meV RMS por elemento a ±0,02 Å) está cerca del ruido de ajuste H (4–7 meV); los modos in-plane tienen mucha más señal.

### Estado y evidencia

**Implementada y testeada; prometedora como señal de pendiente, insuficiente para validar EPC interlayer.** No es la pérdida default.  
Commits: `6c44eac`, `fe219a0`, `6b2b619`.  
Código/tests: `metrics.py:1603-1706`, `test_metrics.py:285-330`.  
Artefactos externos: `bilayer_epc_6x6/verdict_fdt*.json`; informe retrospectivo compañero `docs/auditoria_retrospectiva_2026-10-06.md:228-234`.

## 4.11. Rendimiento: MACE ligero, memoria, layout y compilación

### Estado anterior y problema

MACE ejecutaba la ruta de energía/fuerzas aun cuando Graph2Mat sólo necesitaba node features. El forward creaba máscaras por tipo y transferencias device→host, y e3nn lanzaba miles de kernels pequeños.

### Cambios

1. **`lean_mace`:** calcula embeddings/interacciones/productos sin energy readouts ni `requires_grad` de posiciones.
2. **Gradient checkpointing:** recomputa interacciones y readout durante backward.
3. **Proyección previa a aristas:** reduce features antes de replicarlas por edge.
4. **`TypeLayoutCollate`:** calcula índices/reordenación en workers CPU; forward y loss sólo hacen gathers.
5. **`torch.compile`:** compila in-place interacciones, products y operaciones del readout con claves de checkpoint estables.
6. `pin_memory`, workers persistentes y una sola llamada para simetrizar.

### Verificación de implementación

Tests comparan salida, matrices, gradientes y `state_dict` entre full/lean/checkpoint y eager/compiled. El layout produce igualdad bit a bit en homo/heteronuclear, imágenes, batches y una/dos componentes.

### Resultados de VRAM y tiempo

| Cambio/medida | Antes | Después | Coste/nota |
|---|---:|---:|---|
| `lean_mace`, pico `6x6` | 2968 MiB | 2542 MiB | −14 %, ~−3 % tiempo GPU |
| checkpoint fp32, proceso | 10008 | 7654 MiB | −24 %, +34 % tiempo |
| checkpoint bf16, proceso | 6862 | 4516 MiB | −34 %, +34 % tiempo |
| proyección nodo, GPU/step | 126 | 54 ms | 2,3×; pierde precisión bf16 |
| layout, sync host/step | 23 | 0 | `6x6` 121→115 ms (−5 %) |
| compile bf16, step | 138 | 70 ms | 2,0×; ~90 s compilación |
| compile bf16, pico | 2425 | 1685 MiB | −28 % |
| compile fp32, training limpio | 40,3 | 30,3 min | −25 %, H 1,057≈1,056 |
| compile mejor receta, carga variable | 127 | 46 min | ×2,8; no benchmark controlado |
| max-autotune, step | 133,0 | 125,7 ms | −5,5 %, ~12 min primer autotune |

En `w90`, compile sólo dio 91→68 ms con 160 s de compilación: el overhead no se amortiza en sistemas diminutos. `reduce-overhead`/CUDA graphs produjo segmentation fault con Torch 2.11. cuEquivariance consumió >6 GB frente a ~2,5 GB y se aparcó.

### Interpretación y estado

- `lean_mace`: **default técnico**, paridad probada y menor memoria; falta un benchmark aislado más limpio para tiempo total.
- checkpoint: **opción validada de memoria**, no de velocidad.
- layout: **validado**, ganancia modesta por sí solo y habilitador de compile.
- compile: **validado para `6x6`**, con beneficio dependiente del tamaño; no universal.
- RAM del host e inferencia end-to-end no se midieron sistemáticamente. El intento de readout condicionado que requería 139 GB de RAM fue descartado.

Commits: `bd1d64d`, `35d0172`, `19057a4`, `fe65bfe`, `f1d8ca5`, `a7b48b0`.  
Código: `models/mace.py:40-94,405-478`, `tools/lightning/data.py:34-65`, `core/modules/graph2mat.py:36-74`.  
Tests: `test_edge_pairing_and_safety.py:213-242,355-395`.

## 4.12. Electrostática de largo alcance y Qeq

### Estado anterior y evidencia del problema

El modelo era local. LiF mostró que un nanocubo con entornos localmente similares al volumen tenía 3,7× más error que un defecto bulk y un gap +3 eV; ZnO polar mantuvo errores de bandas/campo después de añadir superficies no polares al train. El estado F neutro/+1, a idéntica geometría, cambiaba H ~10 meV y hasta 1,1 eV localmente, pero el modelo sin carga producía la misma salida.

### Cambio realizado

Se añadió

\[
\Delta H_{ij}^{LR}=\frac{V_i+V_j}{2}S_{ij}.
\]

`V` se calcula en recíproco con cargas gaussianas. Hay dos modos:

- `charges`: cargas locales aprendidas y neutralizadas por estructura;
- `qeq`: electronegatividad `χ_i` y dureza positiva `J_i` aprendidas, resolviendo

\[
\min_q\ \sum_i\chi_iq_i+\frac12\sum_iJ_iq_i^2+\frac12q^TAq,
\quad \sum_iq_i=Q.
\]

Q procede de `NetCharge`. Todo es diferenciable respecto a `χ,J`; la geometría del Ewald se trata fija.

### Verificación técnica

Tests comprueban Madelung rocksalt, el término `V·S`, conservación de carga, requisito explícito de S, estacionariedad Qeq y respuesta a carga total. Pasaron en la ejecución focal actual.

### Resultados científicos

| Caso | Sin LR | Cargas fijas | Qeq | Conclusión |
|---|---:|---:|---:|---|
| ZnO no polar, gap | 144 | 71 | 20 | Qeq gana |
| ZnO polar ×6, H | 4,3 | 16,3 | 4,3 | cargas fijas rompen; Qeq neutro |
| LiF val | 1,98 | 1,88 | 1,78 | mejora moderada |
| LiF F neutro H | 5,27 | 4,78 | 3,99 | Qeq −24 % |
| LiF F+1 H | 6,62 | 6,94 | 7,55 | empeora sin datos cargados |
| BaTiO3 val | 2,44 | 2,27 | 2,42 | cargas fijas mejor |
| H2O val/H2O32 | 4,16/2,40 | — | 3,80/2,21 | mejora pequeña |

Con seis estructuras cargadas en train, F+1 mejoró 6,62→4,80 sin Qeq y 4,71 con Qeq, mientras F neutro empeoró a ~7,2. Los datos cargados aportaron más que Qeq; persiste el conflicto de estados.

### Interpretación y límites

El ansatz de primer orden captura shifts electrostáticos suaves, no reconstruye metalización superficial, screening no lineal ni niveles de defecto. Las cargas locales de volumen pueden generar un campo polar sin apantallar y empeorar catastróficamente. Qeq evita ese fallo, pero no es una mejora universal.

Además, la ruta requiere S como última etiqueta incluso en predicción, está excluida de `torch.compile`, el Ewald crece con el número de vectores recíprocos y Qeq resuelve un sistema denso por estructura; no existe benchmark de escalado grande.

### Estado y evidencia

**Experimental/prometedora; no default.**  
Commits: `6834407`, `0dc3a2d`, `fe0ffeb`, `4a428e0`, `88b3bee`, `56ac339`.  
Código: `models/long_range.py:1-142`, `models/mace.py:624-655`; tests `test_long_range.py:12-144`.  
Resultados: `bench_{lif,batio3,zno,h2oclu}_summary.json`.

## 4.13. Banco de materiales, OOD y trazabilidad experimental

### Cambio realizado

Se construyó un ejecutor externo que congela splits, hashes, base y pseudopotenciales; valida etiquetas; entrena 3–5 semillas; y evalúa H alineado, pares de especies, gap y bandas. La fase 1 incluyó 126 estructuras etiquetadas: H2O 39, LiF 45 y BaTiO3 42. La fase 2 añadió ZnO bulk/superficies y variantes de datos.

### Controles relevantes

- Hermiticidad de etiquetas `≤4·10^-8 eV`.
- Geometrías/átomos/orbitales y carga leídos de vuelta.
- H fuera del grafo: despreciable en H2O, pequeño en LiF, hasta 50 meV O–O en BaTiO3.
- S muy mal condicionada: `λ_min≈10^-3` en H2O y `3–5·10^-5` en LiF/BTO.
- Separación por tests: validación, defectos, nanocubos, paredes/modos, slabs y moirés.

### Qué se aprendió

1. Δ es la única mejora general en todos los materiales medidos.
2. Cambios arquitectónicos pequeños son material-dependientes.
3. Añadir el tipo de entorno ausente suele ganar más que una arquitectura: dos bicapas redujeron H OOD 38–42 % y bandas 57–78 %; slabs no polares resolvieron la slab no polar de ZnO.
4. Después de cubrir superficies quedan física global, carga y espectro condicionado por S.

### Estado y evidencia

**Infraestructura experimental sólida pero externa al paquete.** Los resúmenes son derivados; la fuente primaria son los `runs/**/result.json`, manifests y predicciones.  
Commits de ledger: `e3b3209`–`f0a84d8`, `46bc937`, `0dc3a2d`–`56ac339`.  
Scripts: `g2m_improvements_benchmark.py`, `g2m_bench_materials.py`, `g2m_bench_benchmark.py`, `g2m_ood_eval.py`, `g2m_bilayer_benchmark.py`, `g2m_h2o_benchmark.py`, `g2m_autorun.py`.

## 5. Resultados experimentales consolidados

### 5.1. Grafeno y H2O molecular

| Experimento | Métrica | Referencia | Mejor/variante | Comentario |
|---|---|---:|---:|---|
| `6x6`, 8 estructuras | H bruto | 1,94 | 1,02 | fp32+lin, 16k, 5 seeds |
| `6x6`, gauge | H alineado | 1,05 | 0,70 | 5/5 seeds |
| `6x6`, gauge+`kl`+Δ | H alineado | 0,75 | 0,130 | 5 seeds con Δ |
| `w90` | H | 12,98 | 4,32 | fp32+poly6+lin+`kl` |
| H2O monómero | H alineado | 29,6 | 5,6 | gauge H-only |
| H2O monómero | H alineado/gap | 5,6 / 80 | 1,01 / 9,6 | +Δ |
| H2O clusters→H2O32 | H | 9,12 | 2,40 | +Δ |
| H2O32 espacio casi nulo | gap bruto | 1847 | 260 | peso `10^-4`, H val peor |

### 5.2. Extrapolación estructural

| Entrenamiento → prueba | Métrica | Baseline | Mejor resultado válido | Cautela |
|---|---|---:|---:|---|
| bicapa → registros moiré no vistos | H / bandas | 20,95 / 69,6 | 0,96 / 2,0 | fp32+lin+gauge+Δ |
| `6x6` mono → bicapa | H / bandas | 78 / 162 | 5,3 / 17 | gauge+`kl`+Δ |
| `w90` 2 átomos → bicapa | H / bandas | 150–450 / centenares | 4,5 / 22,6 | Δ; no pura ML |
| `w90` → moiré real | H / bandas | 65,4 / 372 | 0,79 / 19,2 | Δ |
| MD bicapa → moiré real | H / bandas | 4,7 / 31,2 | 0,185 / 1,08 | Δ |
| gr/hBN 4 átomos → moiré 28 | H / bandas | 13,3 / 876 | 1,26 / 19,6 | test válido bicapa→bicapa |
| `6x6` + 2 frames bicapa → OOD | H / bandas | 4,43 / 15,1 | 2,57 / 3,3 | dato específico supera arquitectura |

### 5.3. Banco multimatérico

| Sistema | Entrenamiento/prueba | Referencia H | Con Δ H | Lectura |
|---|---|---:|---:|---|
| H2O clusters | val / H2O32 | 19,75 / 9,12 | 4,16 / 2,40 | generaliza tamaño, S condiciona espectro |
| LiF | val | 20,3 | 1,73 | −91 % |
| LiF | Schottky / F0 / F+1 | 16,0 / 28,2 / 27,6 | 4,6 / 4,7 / 6,9 | carga/defecto no resueltos |
| LiF | nanocubo / MD | 137 / 163 | 16,8 / 18,2 | gran mejora, residuo largo alcance |
| BaTiO3 | val | 14,4 | 2,44 | −83 % |
| BaTiO3 | pared / modo polar | 4,00 / 4,33 | 0,46 / 0,48 | gap a ~±50 meV |
| ZnO | val / bulk72 | 10,6 / 6,2 | 1,66 / 0,90 | volumen resuelto |
| ZnO | slab no polar | 155 | 25,8; 3,9 con datos slab | falta de datos separada |
| ZnO polar ×2/×4/×6 | 133/40/19 | 31,4/11,1/6,1 | gaps aún ~1,1 eV sin datos slab |

### 5.4. Qué comparaciones no deben mezclarse

- H bruto frente a H alineado por gauge.
- Bandas brutas frente a bandas alineadas.
- Espectro completo frente a ortogonalización canónica al eliminar `λ(S)<10^-2`.
- Inferencia ML pura frente a Δ, que ejecuta H0 en el target.
- Validación IID frente a tests de nueva geometría, superficie, carga o composición.
- Tiempos con carga/pausas térmicas frente a microbenchmarks alternados.

## 6. Impacto sobre precisión

### 6.1. Hamiltoniano H

La evidencia es fuerte. La progresión causal fue:

1. pérdida elemento a elemento y entrenamiento suficiente: ~5–6→~2 meV en campañas históricas y 1,94→1,69 al retirar EarlyStopping;
2. readout lineal: hasta −60 % en `w90`, −12 % H2O y mejoras en `6x6`;
3. gauge: −33 % en `6x6`, −81 % en H2O;
4. Δ: −79 a −91 % en validación de cuatro familias materiales.

La mayor parte del error restante cambia de naturaleza: on-site/gauge, S mal condicionada, superficies, campos y estados de carga; ya no es simplemente capacidad local del readout.

### 6.2. Overlap S

`g2m-v2` puede cargar, transportar, predecir y serializar S junto a H. Sin embargo:

- no hay tabla consolidada de S-MAE;
- predecir S junto a H empeoró H y elevó VRAM;
- las mejores recetas usan S como etiqueta o la obtienen del cálculo H0;
- la ruta H-only no puede generar un TSHS no ortogonal desde sólo geometría sin una fuente externa de S.

Conclusión: **la infraestructura S está implementada; no se ha demostrado una mejora de predicción de S**.

### 6.3. Otras matrices

- DM y EDM heredan correcciones de simetría, batching, métricas y loader.
- La restricción `Tr(DS)=N`, McWeeny y la relación EDM–DM–H siguen pendientes (roadmap 4.5).
- No se encontró banco científico posterior que cuantifique mejora de DM/EDM.
- Spin colineal está implementado; NC/SOC no.

No debe extrapolarse la mejora de H a «todas las matrices» sin experimentos.

## 7. Impacto sobre rendimiento

### 7.1. VRAM

Las reducciones robustas son checkpointing (−24 % fp32, −34 % bf16), `lean_mace` (~−14 %) y compile en una sonda bf16 (−28 %). La proyección previa puede llegar a −39 % combinada, pero sacrificó precisión en la receta relevante. Predecir H+S incrementó ~48 % la memoria respecto a H-only.

### 7.2. Tiempo de entrenamiento

`torch.compile` es la mejora principal: ~2× por paso y −25 % end-to-end en una comparación limpia. El ×2,8 de la mejor receta es indicativo, no controlado, porque la carga de máquina cambió. Checkpointing ahorra memoria a costa de ~34 % de tiempo. `keep_last_layer_irreps` añadió ~30 % en la comparación citada.

### 7.3. Tiempo de inferencia

No hay campaña sistemática aislada de latencia versus tamaño. Las predicciones guardan tiempos, pero mezclan IO, preparación y tamaños. Compile tiene warm-up y no compensa `w90`. Δ añade ~18 s de SIESTA para `6x6`; ése domina una inferencia corta. No se puede afirmar una aceleración end-to-end universal.

### 7.4. RAM y escalabilidad

La RAM host no fue perfilada sistemáticamente. El readout Hamiltoniano condicionado intentó reservar 139 GB y se descartó. El modelo local escala con número de edges; `mp_cutoff` reduce grafo MACE pero empeoró precisión. La corrección Ewald/Qeq añade costes globales y no tiene benchmark grande; Qeq incluye un solve denso, potencialmente cúbico en átomos por estructura.

## 8. Impacto sobre generalización y extrapolación

### Evidencia positiva

- Bicapa→registros/moiré con lin+gauge+Δ: H 20,95→0,96 y bandas 69,6→2,0.
- Monocapa→bicapa: Δ reduce un orden de magnitud; `kl` mejora especialmente la parte OOD.
- `w90`→bicapa/moiré, imposible sin Δ, cae a pocos meV.
- Grafeno/hBN bicapa→moiré válido: 1,26 meV H, 19,6 meV bandas.
- H2O clusters→H2O32 y BaTiO3 celda→pared/modo polar generalizan bien con Δ.
- Volumen ZnO→volumen mayor generaliza; no a superficies no vistas.

### Evidencia negativa o condicional

- Poly6 mejora IID y empeora los tres escenarios OOD; poly3 es preferible para extrapolar.
- `w90` sin Δ no extrapola a bicapa con ninguna arquitectura probada.
- Edge updates, scaleshift y precisión fp32 pueden mejorar IID y empeorar bandas OOD.
- LiF nanocubo, defectos y F+1; ZnO polar; niveles de defecto: no resueltos.
- Dos frames de la familia objetivo superan muchas modificaciones de arquitectura.

### Correcciones de conclusiones previas

1. Un primer test grafeno/hBN mezclaba train tricapa y test bicapa; no era una prueba válida. `f228724` lo atribuyó inicialmente a una distancia de 5 Å, `ec9ca14` corrigió esa medición y `846dc00` construyó el test bicapa→bicapa válido.
2. Una curva de aprendizaje agrupaba OOD de N=1/2/4 con N=8 por nombre de variante; se corrigió en `g2m_ood_eval.py` y `roadmap.md:107-110`.
3. «fp32 empeora OOD» se refinó: el factor dañino principal era poly6; con poly3, fp32+lin podía mejorar.

Conclusión: sí hay generalización demostrada, pero es específica de dominio y a menudo asistida por H0. No hay evidencia de extrapolación arbitraria a composiciones o física global no observada.

## 9. Impacto sobre `dH/dR` y EPC

### Qué cambió en Graph2Mat

- La ruta `lean_mace` evita gradientes de posiciones sólo cuando no se piden fuerzas; el modelo completo sigue siendo diferenciable.
- El repositorio compañero implementa jacobianos de posiciones por VJP vectorizado y verificó autograd contra finite differences del propio modelo.
- `pair_difference_mse` añade supervisión directa de pendientes en pares ±.
- Gauge y Δ reducen ruido de H, condición necesaria para diferenciar diferencias pequeñas.

### Evidencia existente

En una campaña externa anterior con checkpoints `6x6` del mismo sistema/funcional se midió ~2,5–3 % de error en `dH/dR` Γ y 3,18 % en A1'(K), con ~1,3 % en el acoplamiento del subespacio de Dirac. Esa evidencia pertenece al repositorio compañero, no es un test del HEAD actual ni una consecuencia aislada de un commit de `g2m-v2`.

Para breathing interlayer, la pérdida de pares no pasó el gate <10 %: mejor valor consolidado `12,4±5,4 %`. Por tanto:

- **dH/dR intralayer de grafeno:** evidencia prometedora/validada en un protocolo externo;
- **EPC físico completo:** no puede declararse validado sólo con H;
- **breathing interlayer:** no validado;
- hacen falta también `dS/dR`, convención de energía, bases consistentes, fonones/modos y normalización de acoplamientos.

Una contradicción histórica importante: campañas antiguas de finite differences usaban FDF que heredaban MD y generaban referencias en geometrías incorrectas; esas conclusiones fueron invalidadas y el generador se hizo single-point/fail-closed. No se usan aquí como prueba.

## 10. Qué modificaciones han funcionado realmente

### Claramente útiles

- Política explícita H/S/spin y reconstrucción multicomponente.
- Correcciones de simetría, batching, métricas y checkpoints.
- `elementwise_mse` y eliminación de EarlyStopping prematuro.
- Readout lineal de aristas, ahora default.
- Pérdida de gauge H-only con S como etiqueta.
- Δ-learning sobre H0 en todos los materiales medidos.
- `TypeLayoutCollate` y `torch.compile` en sistemas no diminutos.
- `lean_mace` como optimización con paridad.
- Añadir unos pocos datos del dominio OOD cuando éste se conoce.

### Prometedoras

- `keep_last_layer_irreps` para extrapolación de capas.
- Penalización en la métrica de S cuando importa espectro bruto con bases difusas.
- Qeq para sistemas polares/cargados, con datos apropiados.
- Edge updates en H2O/términos de tres centros.
- `pair_difference_mse` como supervisión de pendiente.

### Inconclusas o material-dependientes

- `output_scaleshift`.
- `edge_hidden_irreps=16`, `mp_cutoff`, simplificación de irreps.
- fp32 frente a bf16 fuera de distribución.
- pérdida con edge weight 2.
- cargas fijas de largo alcance.
- defectos/superficies añadidos como datos: útiles localmente, no resuelven toda la física.

### Sin evaluar suficientemente

- S-MAE y predicción S autónoma.
- Spin colineal científicamente; NC/SOC.
- DM/EDM después de los cambios.
- inferencia end-to-end y escalado grande de Ewald/Qeq.
- multi-GPU.
- contexto Hamiltoniano local, readout por interacción y entrenamiento directo en irreps.

### Descartadas o sustituidas

- Envolvente radial por par como mejora general.
- Readout condicionado que requería 139 GB.
- nodo proyectado para la receta de precisión bf16.
- poly6 cuando el objetivo es extrapolación.
- pérdida espectral sólo Γ para mono→bicapa/moiré.
- 32k pasos, EMA, bessel16, nodemix, cuEquivariance en este hardware.
- cargas fijas para slabs polares de ZnO.
- CUDA graphs `reduce-overhead` con Torch 2.11 por segfault.

## 11. Estado actual de `g2m-v2`

### ¿Qué puede hacer ahora que el baseline no podía o hacía peor?

1. Leer y reconstruir explícitamente H, H spin-colineal y `(H,S)` sin descartar columnas silenciosamente.
2. Entrenar H-only usando S para resolver el gauge y evaluar métricas espectrales coherentes.
3. Producir matrices con un readout lineal de aristas mucho más expresivo y conservar irreps profundas de MACE opcionalmente.
4. Aprender residuos respecto a H0 mediante el pipeline externo y recuperar H final.
5. Penalizar bandas de primer orden, espacio casi nulo de S y diferencias de pares ±.
6. Añadir una corrección electrostática fija o Qeq dependiente de carga total.
7. Entrenar con menos VRAM mediante MACE ligero/checkpointing y más rápido mediante layout+compile.
8. Medir de forma trazable IID/OOD en varias estructuras/materiales, no sólo loss de validación.

### Qué está activo por defecto

En `LitMACEMatrixModel` son default `E3nnLinearEdgeBlock`, `lean_mace=True` y `elementwise_mse`. No son default: gauge, Δ, `keep_last_layer_irreps`, 16k pasos, compile, scaleshift, edge updates, pair envelope, spectral/null loss ni long-range/Qeq. Los números de 0,1–1 meV no aparecen automáticamente al instalar la rama; pertenecen a recetas configuradas.

### Salud técnica al corte

- Diff baseline→HEAD: **109 archivos, 34.280 inserciones, 363 eliminaciones**. Gran parte de las inserciones son informes/diagnósticos, no código runtime.
- Verificación focal actual: **170 passed**.
- Suite completa actual: **494 passed, 2 skipped** en 196,49 s; sin fallos, con 5985 warnings, principalmente deprecaciones TorchScript/sisl/PyParsing.
- El workflow CI existe localmente, pero no en `origin/g2m-v2` porque no se han publicado los 78 commits.
- El working tree ya contenía `.agents/`, `.claude/` y `agent-skills/` sin trackear; no forman parte de esta auditoría.

## 12. Limitaciones actuales

### Científicas

- H0 introduce dependencia de SIESTA y del mismo convenio de base/pseudopotencial.
- Gauge H-only no fija el cero absoluto; para usos que lo necesiten hay que aportar una convención externa.
- S no está validada como target autónomo; NC/SOC no están soportados.
- Un modelo local no resuelve por sí solo campos, screening, carga, superficies y niveles de defecto.
- El grafo omite acoplos KB más allá de `R_i+R_j`; llegan a 50 meV en BaTiO3 O–O.
- Bases difusas pueden hacer que pocos meV en H destruyan el espectro por `λ(S)≪1`.
- La mayoría de bancos usa 3 semillas. Es suficiente para señales grandes, no para diferencias del 2–5 %.
- H-MAE, incluso alineado, no sustituye bandas, gap, derivadas o EPC.

### De ingeniería y producto

- Los cambios principales no están en GitHub y los resultados viven en otro repositorio.
- Δ no es una abstracción del core ni un comando autocontenido de predicción.
- La mejor receta no está encapsulada como preset versionado.
- La ruta H-only+S necesita que el predictor aporte S; el core no la genera de geometría.
- `long_range` usa etiquetas S durante prediction, desactiva compile en esa sección y no tiene benchmark de escalado.
- Servidor `/test`/`/predict`, import sin Torch, DataModule completo, multi-GPU y coefficient loss siguen abiertos.
- Hay miles de warnings de deprecación; no rompen la suite actual, pero anuncian deuda con TorchScript y dependencias.
- No hay comparación sistemática de RAM host ni inferencia end-to-end.

### Registro de contradicciones y documentos obsoletos

| Fuente/afirmación | Evidencia posterior | Veredicto |
|---|---|---|
| Auditoría inicial: H perdía S/spin | Código actual los soporta | Era cierta para baseline, ya corregida |
| `roadmap.md` 4.1: «falta reconstrucción no ortogonal completa» | `sparse.py:488-503` reconstruye si H y S están disponibles | Parcialmente obsoleto; H-only aún necesita S externa |
| Primer gr/hBN: mala distancia ~5 Å | Re-medición: train tricapa vs test bicapa | Corregido en `ec9ca14`; test reemplazado |
| Curva N=1/2/4 OOD | Agregador mezclaba N con N=8 | Valores corregidos; no usar tabla antigua |
| «fp32 empeora OOD» | Aislando variables, poly6 era el principal culpable | Conclusión refinada |
| JSON `H_MAE_meV` de modelos gauge parece enorme | `H_MAE_gauge_meV` y bandas alineadas mejoran | Métricas de gauges distintos no comparables |
| `torch_compile` ×2,8 | Comparación limpia da −25 % total | ×2,8 es carga variable, no cifra universal |
| Cargas de largo alcance arreglarían slabs polares | Cargas fijas las empeoran ×2,4–3,8 | Hipótesis refutada; Qeq sólo evita el daño |
| FD histórica dH/dR tenía SNR<1 | Referencias heredaban MD y no eran ±δ reales | Evidencia invalidada; no se usa |
| Ledger marca puntos «pendientes/en marcha» arriba | Resultados posteriores los cierran abajo | Estado editorial no normalizado; usar conteo de tablas |

## 13. Trabajo pendiente

### Prioridad 0 — reproducibilidad

1. Publicar o respaldar los 78 commits locales y crear un tag del estado auditado.
2. Versionar manifests/resúmenes mínimos de resultados o enlaces con hashes al repositorio compañero.
3. Convertir la receta recomendada en configuración/preset reproducible, con versión de SIESTA/MACE/Torch.
4. Resolver warnings/deprecaciones que afectarán futuras versiones, empezando por TorchScript/e3nn.

### Prioridad 1 — cerrar el producto H/S

1. Integrar Δ/H0 como pipeline explícito de entrenamiento/predicción, sin lógica dispersa.
2. Definir de dónde sale S en inferencia H-only y verificar escritura TSHS end-to-end.
3. Medir S-MAE/espectro de una cabeza S separada antes de decidir si se predice o calcula.
4. Arreglar compatibilidad de `coefficient_space_mse` con mappings reales.
5. Cerrar servidor, import opcional sin Torch y provenance/splits.

### Prioridad 2 — física

1. Estado de carga: entrenamientos balanceados neutro/cargado y representación que no intercambie un estado por otro.
2. Largo alcance: benchmark de escalado y modelos de screening más allá de `V·S` antes de hacer Qeq default.
3. Acoplos KB fuera del grafo en BaTiO3/elementos pesados.
4. Pérdida de métrica S en LiF/ZnO y con criterios espectrales preregistrados.
5. Spin colineal multisemilla; después diseñar NC/SOC hermítico complejo.
6. DM/EDM: `Tr(DS)=N`, idempotencia/McWeeny y consistencia EDM–DM–H.

### Prioridad 3 — rendimiento

1. Latencia/VRAM/RAM versus tamaño, batch y número de especies, incluyendo inferencia y Qeq.
2. Separar mediciones con carga fija y calentamiento de compile.
3. Investigar low-rank/channelwise TP sólo si el profiler confirma de nuevo ese cuello.
4. Multi-GPU y métricas sincronizadas sólo cuando haya hardware para verificarlas.

### Puntos explícitamente pendientes del ledger

`1.15`, `2.6`, `3.4`, `3.5`, `3.12–3.14`, `4.5`, `5.3`, `6.1`, `6.2`, `6.4–6.6`, `7.1`, `7.4–7.7`, `8.3`: 20/80. Varios puntos 🔶 contienen además subtareas abiertas; por eso «75 % abordado» no significa «75 % terminado».

## 14. Cronología técnica

| Fecha/orden | Modificación | Motivación | Resultado |
|---|---|---|---|
| 2026-04-16 | baseline H mínimo | poder leer/escribir Hamiltoniano | soporte escalar inicial |
| 17–21 abr | batching, reconstrucción, spin multicomponente | no perder componentes ni romper simetría | pipeline colineal y validaciones |
| 19–20 may | políticas H/S/raw y coeficientes | eliminar ambigüedad; diagnosticar capacidad | infraestructura H/S + rank diagnostics |
| 27 sep | escala/shift, irreps inalcanzables | normalizar y detectar canales siempre cero | opción y warnings |
| 28 sep | readouts, MACE lean/checkpoint, pérdidas, seguridad | precisión y coste | primera ola implementada |
| 28 sep | compile y gauge | kernels/sincronización; cero de energía | grandes mejoras técnicas/científicas |
| 28–29 sep | campañas w90/6x6/H2O | seleccionar receta | readout lineal default; gauge confirmado |
| 30 sep–1 oct | `kl`, Δ, moirés reales | extrapolación | Δ se vuelve mejora dominante |
| 1–2 oct | layout, compile modes, curva N | rendimiento y cantidad de datos | 0 sync; curva H plana desde N=1 |
| 2–3 oct | espectral, edge update, pair loss, pair envelope | bandas, tres centros, derivadas, cortes | implementados; resultados mixtos |
| 2–3 oct | gr/hBN y correcciones | comprobar otro material | test inválido detectado y reemplazado |
| 3–4 oct | banco H2O/LiF/BTO | separar arquitectura de física | Δ general; límites de S/carga/largo alcance |
| 5 oct | ZnO, superficies, LR y Qeq | probar campos polares | fixed charges fallan; Qeq mixto |
| 6 oct | charged defects, loader de checkpoints | cerrar evidencia/compatibilidad | resultados documentados; suite estable |

## 15. Mapa de commits

Los 92 commits posteriores al baseline se agrupan aquí por modificación conceptual. Los merges `56cad47`, `5fa9a63` y `b24a79a` son administrativos y no añaden cambios distintos de sus padres.

| Modificación conceptual | Commits asociados | Archivos principales |
|---|---|---|
| Batching/reconstrucción H | `9d8cc0c`, `cacb938` | `processing.py`, `sparse.py`, tests |
| Spin/multicomponente | `1d52abb`, `56cad47`, `535c01b`, `5fa9a63`, `db74d63`, `b24a79a`, `ec4f742` | data/config, MatrixBlock, CLI |
| Política H/S/raw | `467ffb4`, `9fb7f82`, `72b468c` | `configuration.py`, `processing.py`, `sparse.py` |
| Coeficientes/rango | `4efeb1a`, `1a131f1` | graph2mat/e3nn metrics + diagnostics |
| Scale/shift e irreps | `a5e8134`, `15f98d0`, `63e7608` | e3nn matrixblock/graph2mat |
| Readouts y MACE eficiente | `a0f6d18`, `bd1d64d`, `a39439a` | `edge_operations.py`, `node_operations.py`, `models/mace.py` |
| Losses/utilidades/validación | `b7c612d`, `7648951` | `metrics.py`, Lightning, basis/supercell |
| Roadmap inicial y selección de recetas | `44becc1`, `239fbc8`, `2afaa32`, `aa4cde9`, `44be1cf`, `6cb4c23`, `1a748d7`, `71bd54c`, `7f93d4d`, `174efca`, `8db90e8`, `650259c`, `e77b836` | `roadmap.md` |
| Compile | `35d0172`, `a7b48b0`, `fe65bfe` | MACE Lightning/core, roadmap |
| Gauge | `7b97b7a`, `1116988`, `adf8092`, `71c1afa`, `31d7bcc`, `fe49168`, `4ef0d25` | `metrics.py`, data module, roadmap |
| Δ y moirés | `6449ddc`, `0888107`, `58b8df3`, `6df452a`, `e05475e`, `9136495` | roadmap; scripts/resultados externos |
| Layout por tipo | `19057a4`, `f1d8ca5` | collate, Graph2Mat forward, losses |
| gr/hBN y corrección del protocolo | `f228724`, `ec9ca14`, `846dc00`, `ae4fda1`, `364fe9c` | roadmap; benchmark externo |
| Pérdida espectral | `496d792`, `8aa6a43` | `metrics.py`, data module/tests |
| Edge updates | `cc261fb`, `26e48b9` | edge preprocessing/tests |
| Diferencias ± | `6c44eac`, `fe219a0`, `6b2b619` | `metrics.py`, tests |
| Envolvente por par | `762217c`, `0c9b235`, `37e6093` | `models/mace.py`, tests, roadmap |
| Diseño/banco de materiales | `e3b3209`, `8b41121`, `c22d73d`, `4f0f01f`, `8340e63`, `7ce970e`, `08619f0`, `09a7cef`, `68afc72`, `f89c44a`, `47fdbe0`, `b1a4969`, `2c09c0b`, `f0a84d8` | `roadmap.md`; artefactos externos |
| Null-S, defectos y ZnO | `46bc937` | spectral-window filename + roadmap |
| Largo alcance/Qeq | `6834407`, `0dc3a2d`, `fe0ffeb`, `4a428e0`, `88b3bee`, `c26254b`, `56ac339` | `long_range.py`, `mace.py`, tests, roadmap |
| Loader checkpoint final | `cae9955`, `e0a88d6` | CLI/tests/roadmap |

## 16. Guion para explicárselo a mi superior

### 16.1. Versión de 2 minutos

«Partimos de una versión que apenas podía tratar un Hamiltoniano escalar: leía sólo una columna del TSHS y no tenía una semántica explícita para overlap o spin. La rama `g2m-v2` corrige eso y añade un pipeline H/S/spin colineal, tests de simetría y reconstrucción, y un readout e3nn más expresivo.

La mejora científica más importante no fue hacer la red mucho mayor. Primero, una pérdida invariante al gauge `H→H+cS` eliminó un cero de energía que un modelo local no puede aprender: bajó 33 % en grafeno 6×6 y 81 % en H2O. Después, Δ-learning hizo que la red aprendiera `H−H0`, con H0 de una iteración de SIESTA. Eso redujo típicamente 80–90 % el error en grafeno, H2O, LiF, BaTiO3 y ZnO, y permitió extrapolar a bicapas y moirés con pocos meV.

En ingeniería, MACE ligero, layout precomputado y `torch.compile` mantienen la precisión y reducen memoria/tiempo; compile dio 2× por paso y −25 % en una comparación end-to-end limpia. Lo que no está resuelto es S autónoma, spin/SOC, defectos cargados, superficies polares y largo alcance. Qeq es prometedor, pero mixto. Además, 78 commits aún no están publicados en GitHub y los resultados viven en el repositorio compañero; ése es el riesgo inmediato de reproducibilidad.»

### 16.2. Versión de 10 minutos

1. **Baseline (1 min).** Arquitectura por bloques equivariantes y MACE ya existían, pero H era esencialmente una matriz escalar: columna 0, sin S/spin y con pérdidas genéricas.
2. **Corrección física del target (1 min).** Políticas `h_only`, `spin_h_only`, `h_and_overlap`; reconstrucción física; simetría y tests. Esto hace válido el pipeline, pero no demuestra precisión de S/spin.
3. **Capacidad del readout (1 min).** El bloque cuadrático tenía un cuello de rango. Añadir un término lineal fue una mejora simple y repetible: −60 % en `w90`, ganancias en `6x6` y H2O; quedó como default.
4. **Gauge (1 min).** Explicar `H+cS`; minimizar sobre `c` por estructura. Cifras: 1,05→0,70 en `6x6`, 29,6→5,6 en H2O, bandas moiré 14,5→7,3.
5. **Δ-learning (2 min).** Explicar `H=H0+ΔH`; 80–90 % en varios materiales. Aclarar coste (~18 s `6x6`) y que no es inferencia ML pura.
6. **Generalización (1 min).** Mostrar bicapa→moiré 20,95/69,6→0,96/2,0 y gr/hBN válido 13,3/876→1,26/19,6. Dos frames del dominio objetivo todavía ganan mucho: extrapolación está limitada por datos/física.
7. **Rendimiento (1 min).** Lean/checkpoint/compile: memoria versus tiempo; compile −25 % limpio, checkpoint −24/−34 % VRAM con +34 % tiempo.
8. **Resultados negativos (1 min).** Spectral Γ no ayuda OOD; pair envelope neutra; fixed charges rompen ZnO polar; scaleshift/edge update dependen del material.
9. **Estado y siguiente acción (1 min).** Suite 494+2; roadmap 75 % abordado/56,25 % efectivo/23 ✅. Publicar la rama, encapsular receta/Δ/S y atacar carga/largo alcance con benchmarks preregistrados.

### 16.3. Preguntas probables y respuestas

1. **¿Cuál fue la mejora más importante?**  
   Δ-learning sobre H0, con reducciones de 79–91 % en validación multimatérica. Gauge es la segunda y corrige un problema físico de identificabilidad.

2. **¿Es una mejora del modelo o del workflow?**  
   Ambas: gauge/readout/compile están en Graph2Mat; Δ y los bancos están principalmente en el workflow externo. No conviene atribuir toda la ganancia a la arquitectura.

3. **¿Graph2Mat predice ahora S?**  
   Puede hacerlo, pero no es la receta recomendada ni hay S-MAE consolidado. La mejor receta predice H y usa S como etiqueta/fuente externa.

4. **¿Por qué el H bruto de algunos modelos gauge es enorme?**  
   Porque el cero `cS` queda libre deliberadamente. Hay que comparar H alineado y bandas alineadas, no H bruto.

5. **¿Δ-learning evita SIESTA?**  
   No. Reduce la dificultad ML, pero H0 requiere una iteración SIESTA en el target; no reduce por sí solo las etiquetas SCF ni crea inferencia independiente.

6. **¿Generaliza de una celda pequeña a un moiré?**  
   Sí en los tests realizados con Δ: `w90`→moiré real <1 meV H y ~16–19 meV bandas. Es generalización asistida por H0 y dentro de familias químicas compatibles.

7. **¿Generaliza a otros materiales?**  
   Δ sí mostró mejoras grandes en H2O, LiF, BaTiO3 y ZnO. Las demás modificaciones son más dependientes del material.

8. **¿Por qué no basta un H-MAE pequeño?**  
   S mal condicionada amplifica errores y observables como gap/defectos dependen de subespacios pequeños. H2O32 cerraba gaps con sólo 2,4 meV de error.

9. **¿Se redujo VRAM?**  
   Sí: checkpoint −24 % fp32/−34 % bf16, lean ~−14 %, compile −28 % en una sonda. Cada cifra tiene una configuración concreta.

10. **¿Se aceleró el entrenamiento?**  
    En `6x6`, compile dio 2× por paso y −25 % total limpio. No compensa igual en sistemas pequeños y tiene coste de compilación.

11. **¿La electrostática larga ya está resuelta?**  
    No. Fixed charges pueden empeorar slabs polares; Qeq evita ese fallo y ayuda casos concretos, pero no arregla screening/niveles de defecto universalmente.

12. **¿Está validado dH/dR/EPC?**  
    Hay evidencia externa buena para derivadas intralayer de grafeno, pero breathing interlayer no pasó el gate y EPC completo necesita más que H. No debe presentarse como cerrado.

13. **¿Qué falló claramente?**  
    Pair envelope fue neutra, spectral Γ no mejoró OOD, readout condicionado pidió 139 GB, CUDA graphs segfault, fixed charges rompieron ZnO polar y poly6 perjudicó extrapolación.

14. **¿Está listo para producción/reproducible?**  
    Técnicamente la suite pasa, pero no todavía como entrega reproducible: 78 commits no están en GitHub, Δ no está integrado y los resultados están en otro repo.

15. **¿Cuál es el siguiente paso con mejor retorno?**  
    Publicar/taggear, empaquetar preset+Δ+fuente de S y después estudiar carga/largo alcance con datos balanceados y métricas espectrales; no añadir otra arquitectura local sin evidencia de cuello.

### 16.4. Índice de evidencia primaria

- Código Graph2Mat: `src/graph2mat/{core/data,core/modules,bindings/e3nn,models,tools/lightning}`.
- Ledger: `auditorias_mejoras/roadmap.md`.
- Diagnóstico de rango: `Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/report.md`.
- Resultados externos: `../MD_vs_AtomicDisplacement/Comparison/results/dataset_design_curves_v1/g2m_improvements/` y `runs/`.
- Scripts externos: `../MD_vs_AtomicDisplacement/Comparison/scripts/g2m_*.py`.
- Chat original de creación/ejecución de la rama: `~/.claude/projects/<project>/0a698975-5150-4a17-943d-a9cd374ce083.jsonl`.
- GitHub: PR #1–#6 y refs recuperadas el 2026-10-06; no contienen los 78 commits locales finales.
