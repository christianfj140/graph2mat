# Auditoría completa del roadmap de Graph2Mat — 21 de octubre de 2026

## Dictamen y cobertura

El roadmap tiene **81 puntos numerados** actuales: las 80 filas de `44becc1` más 4.7, añadido en `7ce970e`. Se han contrastado todos, además del banco de materiales, las decisiones posteriores y los **217 hallazgos originales de Claude**. El apéndice contiene los 217 IDs exactamente una vez, sin omisiones, duplicados ni IDs extra. Hay **cero enlaces Markdown locales rotos** en `roadmap.md`; la referencia textual a `roadmap_map.py` no corresponde a un script distribuido en el repositorio. La comprobación se ha rehecho con una extracción independiente.

La mayor parte de las opciones y correcciones existen. El principal problema del documento es que acumula estados contradictorios: sus cabeceras, tablas y planes auxiliares aún presentan como pendientes opciones que ya tienen código, pruebas o campañas cerradas. Hay también compromisos originales parcialmente cubiertos que una marca ✅ demasiado agregada oculta. **No se ha encontrado en esta área motivo para reabrir automáticamente campañas, promover defaults ni implementar el backlog condicionado.**

La tabla completa está en `roadmap_tabla.md`: una fila por punto, con implementación, certificación disponible, medidas, beneficio, defaults y residuales; después aparece el contraste individual de los 217 IDs. Los recuentos se aplican a los 81 puntos, no a los 217 subhallazgos ni a los ejes del banco:

| Clasificación | Puntos | Interpretación |
|---|---:|---|
| OK | 55 | La implementación/decisión coincide con el contrato que realmente declara la fila. No significa que toda opción mejore la física. |
| Discrepancia | 9 | Contradicciones documentales de estado: 1.9, 1.15, 2.8, 2.9, 7.1, 7.4, 7.5, 8.1, 8.2. |
| Parcial real | 5 | 0.10, 1.12, 3.11, 4.3, 7.3 conservan partes originales no cubiertas. La existencia de una parte pendiente no autoriza ejecutarla. |
| Pendiente autorizado | 6 | 1.13, 3.8, 4.6, 6.5, 6.6, 8.3 están aplazados o condicionados legítimamente. |
| Histórico/experimental | 4 | 1.10, 6.8, 6.9, 7.7 tienen rechazo/limitación dependiente de receta o entorno, no imposibilidad universal. |
| Excluido | 2 | 3.13 y X.1; no son trabajo activo pendiente. |

## Base, método y límites

Base: rama `g2m-v2`, HEAD `a66b46cb0955ca344693d50737c5e9952fc257fc`. Se usó el original `44becc1:auditorias_mejoras/roadmap.md`, el historial posterior de ese archivo, `plan_mejoras.md`, la auditoría original de Claude y las propuestas GPT reflejadas en el roadmap. Se leyeron AGENTS y ambos traspasos; sus estados de cola y decisiones históricas no se trataron como autorización de trabajo nuevo.

Se aplicaron los contratos de `.agents/skills/scientific-experiment-audit/SKILL.md` y `.agents/skills/graph2mat-validation/SKILL.md`: distinguir implementación, consistencia numérica, experimento comparativo, evidencia física y promoción. Se inspeccionaron fuentes, firmas de APIs, enlaces CLI, tests pertinentes, configuraciones y resúmenes JSON pequeños ya existentes. **Este agente no ejecutó pytest, GPU, SIESTA, entrenamiento ni evaluación de datasets.** Los conteos de suites citados por el roadmap son históricos; la verificación presente de producto corresponde al informe principal y al entorno del coordinador.

No se cargaron datasets externos grandes, checkpoints ni los 1800 NPZ de 7.6. Sí se recalcularon desde `summary.json` las medias finales de 7.6 y se contrastaron las razones finales registradas de 4.7, 6.4 y 7.1/7.4. Esto detecta desacuerdos entre resumen y documento, pero no vuelve a certificar referencias originales ni independencia de splits. El inventario y las extracciones quedan en `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/roadmap/`.

## Discrepancias y carencias priorizadas

### R-01 — MODERATE: el estado operativo actual aparece contradicho por estados históricos sin separar

**Evidencia.** La cabecera `roadmap.md:117` todavía presenta 1.15 como pendiente, aunque la fila `roadmap.md:859` y el commit `1a0160f` implementan `selection_metric` opt-in, sin cambiar el default. En `roadmap.md:128` se condicionan 8.1/8.2 como trabajo futuro; las filas `roadmap.md:983` y `roadmap.md:984` incluyen campañas SOC completas y padding implementado. La propia fila 8.1 conserva «F4/F5 sin autorizar» y «ningún entrenamiento SOC» antes de añadir F4/F5 terminadas. La fila 8.2 conserva «aristas largas todavía no implementadas» antes de describir su implementación y ablación.

También quedan «readout_per_interaction pendiente» en `roadmap.md:872`, aunque la API ofrece `readout_per_interaction` y agregación `sum/mean` (`models/mace.py:121`, `tools/lightning/models/mace.py:98`); «falta 6x6» y 6.7 descartado en `roadmap.md:873`; experimentos 7.1/7.4 pendientes antes de campañas terminadas (`roadmap.md:971`, `roadmap.md:974`); y medición 7.5 pendiente antes de sus resultados (`roadmap.md:975`). La tabla de ejecución A–G sigue enteramente en marcha/pendiente (`roadmap.md:624`).

**Consecuencia.** Un agente nuevo puede repetir trabajo cerrado, confundir certificación CPU con falta de código o interpretar un permiso viejo como vigente. El problema es documental; no demuestra un bug del modelo.

**Cambio mínimo propuesto.** Una única columna de estado vigente y secciones históricas fechadas/enlazadas, sin mezclar ambos tiempos en la misma celda. Actualizar cabecera y pasos A–G a partir de las filas y cierres actuales. No lanzar ningún trabajo por esta limpieza.

**Estado temporal correcto.** SOC no está simplemente aplazado: F4/F5 Bi2Se3 se hicieron. La fase magnética posterior está pausada expresamente; `d957ffd` y `roadmap_magnetismo.md:362` prohíben iniciar auditoría de momentos, H0 o F5 hasta decisión nueva del usuario. Los lanzamientos A/B ya autorizados podían terminar. Una revisión de documentación no sustituye esa decisión.

### R-02 — MODERATE: 4.3 no cumple toda la integración de validación espectral del original

**Requisito original.** `44becc1`, fila 4.3, y `claude_opus5.5ultracode.txt:643` (LOSS_TRAIN-14) piden métricas espectrales en validación: resolver H(k)c=εS(k)c con periodicidad, ventana física y métricas de niveles/gap.

**Evidencia actual.** La migración `041e0ad` ofrece APIs reales de evaluación sobre archivos: `tools/analysis/metrics.py:80` y `:237`, más `tools/analysis/breakdown.py:78`. Esto sí existe y está probado. En cambio, los callbacks Lightning disponibles son MatrixWriter, SamplewiseMetricsLogger y PlotMatrixError (`tools/lightning/callbacks.py:28`, `:251`, `:476`); no se encontró integración de autovalores/bandas exactos dentro de la validación. El `spectral_rmse` del loss procede de la proyección Γ de 4.4 (`core/data/metrics.py:1963`, `tools/lightning/model.py:461`), que no sustituye ese observable.

**Consecuencia.** 4.3 ✅ acredita análisis offline y desglose, pero no toda la petición original. No significa que las campañas carezcan de evaluación espectral: muchas la ejecutan externamente. Sí significa que un usuario del flujo Lightning estándar no obtiene esa evaluación por activar una métrica de validación ya existente.

**Cambio mínimo propuesto.** Marcar por separado «métricas offline/desglose: implementados» y «integración espectral periódica en validación: no implementada». Una futura integración puede reutilizar la API migrada, pero no está autorizada por esta auditoría.

### R-03 — MODERATE: 3.11 agrega como cerrado un contrato geometría/etiqueta todavía parcial

**Evidencia.** Se corrigieron sc_off no canónico (`configuration.py:535`, `test_nonstandard_sc_off.py:38`, `928dee0`), ncol (`b5e9d51`) y celda periódica degenerada (`498d0c2`). Esas correcciones son reales. Sin embargo, `configuration.py:627` permite leer geometría de salida; `configuration.py:669` entrega geometría al lector de matriz cuando hay base, y `configuration.py:678` acepta una geometría externa sin comparación general contra la etiqueta. El lector H está protegido por la geometría que aporta TSHS; no se encontró protección general equivalente para DM/EDM de archivos sin coordenadas.

La fila `roadmap.md:890` admite «DM/EDM desde .DM sin comprobar, sin uso actual», pero se presenta globalmente ✅. El original CRIT-05 (`claude_opus5.5ultracode.txt:1474`) es más amplio. La validación de referencias físicas opcionales en `core/data/physical.py:51` no cubre automáticamente toda carga legacy.

**Consecuencia.** Hay una limitación real de ese flujo, no una campaña demostrada corrupta. La auditoría no ha reproducido una etiqueta DM desalineada ni demuestra que afecte a los resultados registrados.

**Cambio mínimo propuesto.** Separar las correcciones cerradas del caso DM/EDM sin coordenadas y de la selección implícita del archivo fuente. Mantener el caso como limitación explícita hasta una decisión y prueba focal; no cambiar PBC históricos ni datos existentes de oficio.

### R-04 — MODERATE: 7.3 cubre la envolvente por par, no toda la radial normalizada por par

**Evidencia.** PHYS_H-07 (`claude_opus5.5ultracode.txt:72`) propone distancia normalizada por par para envolvente y Bessel; MATH_EQUIV-08/SOTA-13 incluyen alcance por orbital. `models/mace.py:601` calcula la radial con el embedding global de MACE; `models/mace.py:612` solo añade la envolvente multiplicativa por par. No se cambia a una base Bessel de d/(R_i+R_j), ni se implementa máscara por par de orbitales.

**Consecuencia.** La opción concreta `pair_envelope` existe y fue medida; su resultado neutro/adverso no prueba que todas las alternativas radiales originales estén implementadas o descartadas. Tampoco obliga a implementarlas: son hipótesis adicionales sin beneficio demostrado.

**Cambio mínimo propuesto.** Delimitar la fila como «envolvente por par implementada; normalización de radial/máscara orbital no implementada». Conservar el rechazo de promoción de la opción medida. No repetir sus campañas ni convertir una alternativa original en compromiso inmediato.

### R-05 — MINOR: las recetas y los defaults de 1.13 siguen confundidos

**Evidencia.** `roadmap.md:857` marca ✅ por una receta corr3/lmax3/3interacciones y promete cambiar defaults al cerrar la ola. El constructor actual tiene `num_interactions=2`, `correlation=1`, `max_ell=3` (`tools/lightning/models/mace.py:65`, `:70`, `:87`). El ejemplo explícito sí usa 3/3/3 (`docs/examples/benchmarks/delta_recipe.yaml:30`). El modelo directo también conserva `symmetric_matrix=False`; la CLI deduce True mediante el DataModule (`tools/lightning/cli.py:190`). No es una contradicción de implementación si se habla del flujo CLI, pero no debe describirse como default universal de toda API.

**Consecuencia y cambio mínimo.** Registrar receta recomendada y defaults reales por API. La promoción sigue siendo decisión del usuario; no resolver una inconsistencia documental cambiando silenciosamente valores por defecto.

### R-06 — MINOR: pendientes de 1.12, rechazo de NodeMix y checklists históricos necesitan alcance explícito

1.12 conserva warmup/SWA/segunda etapa en el título, pero solo registra pruebas de EMA/clipping. La frase «SWA no es posible (deepcopy del modelo)» (`roadmap.md:856`) no cuenta con una prueba actual identificable; existe una prueba de deepcopy/pickle de Graph2Mat (`test_edge_pairing_and_safety.py:118`), y `plan_mejoras.md:44` registra EMA funcionando tras arreglar copia. Esto **no certifica SWA con Lightning/MACE**. La corrección mínima es «SWA no validada en este flujo; motivo histórico por reproducir», sin obligación de reabrirla.

NodeMix está descartado sin ámbito en el resumen (`roadmap.md:585`), mientras la fila 1.9 termina «candidato a default del banco» tras mejoras reales de H y regresiones de bandas ZnO (`roadmap.md:853`). El usuario pidió no hacerlo default (`traspaso_agente_20261013.md:47`). Debe quedar «opción medida, no promovida», con resultados por dominio; ni eliminación del código ni promoción se deducen de los datos.

Los planes `2026-10-14-homonuclear-weight-fusion-6-7.md:95` y `2026-10-21-migracion-mdvs.md:60` mantienen casillas sin marcar aunque existen código/tests y cierres. El plan auxiliar de precisión todavía presenta Task3/Task4 experimentales abiertos (`roadmap_precision_rendimiento.md:207`, `:239`, `:245`, `:270`) pese a campañas posteriores. Son inconsistencias documentales, no ausencia de implementación. El plan general sí conserva pendientes de alcance mayor, como la vista igualtiempo; no se recomienda marcarlo entero cerrado.

### R-07 — MODERATE: la reproducibilidad de campañas y equivalencias no viaja completa con el código

**Evidencia.** La migración excluye deliberadamente runners del arnés privado y distribuye un ejemplo (`informe_migracion_mdvs_20261021.md:23`, `docs/examples/benchmarks/README.md:20`). Las funciones de Δ, proyecciones, métricas, desglose, atribución, control KB, ruido y piloto DM sí están en `src/graph2mat`; no se encontró aquí dependencia funcional de esas APIs respecto al código MD_vs. Los datos y muchas salidas científicas permanecen fuera de Git. La equivalencia G1–G9 se atribuye a «script fuera del repositorio» (`informe_migracion_mdvs_20261021.md:43`); sus resultados documentados no equivalen a un registro independiente distribuido de ese gate.

Además, Git conserva cinco symlinks absolutos históricos bajo `Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view/`, a datos/base/checkpoint externos. El coordinador los excluyó de su copia de pruebas; no se resolvieron ni abrieron para esta área. Son artefactos diagnósticos, no dependencia funcional de `src` ni prueba de que la migración esté incompleta.

**Consecuencia.** Clonar graph2mat permite usar las funciones con datos/SIESTA propios; no reproduce por sí solo todas las campañas ni todas las equivalencias reales reclamadas. Ausencia de evidencia portátil no significa ausencia del código ni invalidez de las cifras locales disponibles.

**Cambio mínimo propuesto.** Documentar con precisión ese límite. Si se quiere un paquete reproducible público, incorporar manifiesto/script de equivalencia y rutas relativas o descriptores de artefactos, conservando la procedencia original y sin copiar datasets privados automáticamente.

## Evidencia científica disponible y conclusiones admitidas

| Punto | Contraste realizado aquí | Conclusión admitida | Conclusión no respaldada |
|---|---|---|---|
| 7.1/7.4 | `precision_summary.json` local contiene ambos materiales, razones pareadas/IC95 y decisión de promoción; coincide con campaña `campana_readout_w90_6x6_20261010.md:142`. | Opciones implementadas y medidas; SK mejora H-test w90, pero regresiones OOD impiden promoción. | Mejor precisión física general o ahorro GPU deducido del benchmark CPU. |
| 6.4 | Summary local 6 aprobados/6 completos; H OOD razón0,988562 y niveles0,691856, IC de OOD incluyen1. | Control de picos con ahorro VRAM registrado; efecto científico exploratorio. | OOD resuelto o trayectoria de optimización idéntica al lote fijo. |
| 4.7 | Summary local 3/3; H OOD1,134723, gap0,383365, niveles0,307487. | Ganancia espectral concreta y regresión H; sin promoción. | Usar260 meV de campaña histórica como resultado del comparativo cosine nuevo; recortar S para ocultar el fallo. |
| 7.6 | Recalculadas medias de seis jobs: H1,979515→14,182384meV; niveles camino34,173464→400,576462meV. | Cierre negativo en receta/presupuesto probados; opción se conserva. | Que toda multitarea falle; OOD independiente; mejora DM/EDM frente a modelos separados no entrenados. |
| 7.2 | Se leyeron cierre/variabilidad/coste, sin cargar checkpoints. | El cierre seis fits no muestra mejora conjunta robusta; no promoción. | Media OOD favorable implica mejora en las tres semillas o coste supervisado equivale a GPU-hours. |
| 8.1/8.2 | Código/tests y revisiones F4/F5/ablación; sin volver a diagonalizar datos Bi2Se3. | SOC no magnético operativo y medido; Δ contiene SOC por H0; padding opt-in medido sin mejora en brazo directo. | La red aprendió todo el SOC de Δ cuando el blanco g(H−H0) es cero; bandas canónicas garantizan el espectro completo; magnetismo ya validado por entrenamiento. |
| Migración | Presencia de funciones/CLI/tests y contrato de equivalencia G1–G9. | Código portado y pruebas sintéticas; informe registra un H0 real idéntico. | Fórmulas equivalentes reproducen automáticamente todas cifras históricas o todos formatos/spins. |

Las campañas con OOD observado repetidamente y tres semillas son evidencia exploratoria/controlada en sus dominios. Los duplicados/correlaciones w90 y limitaciones de soporte están reconocidos por los cierres. El guardrail 3.12 ayuda a detectar identidades declaradas; no certifica independencia ni corrige esos splits. No se ha detectado que las campañas negativas examinadas oculten sus regresiones principales en sus cierres. La mayor fuente de sobrelectura es la mezcla de estados y recetas dentro del documento acumulativo.

## Banco de materiales: contraste de todos los ejes y pendientes

| Eje/lista del banco | Estado contrastado | Qué falta o no debe confundirse |
|---|---|---|
| Varias especies/radios distintos | H2O, LiF, BaTiO3, ZnO y TMD medidos; 1.3/1.9/1.11/5.2/7.3/7.5 tienen resultados por material. | La tabla introductoria `roadmap.md:998` que aún dice faltan ligandos/metales de transición es histórica; no son datasets ausentes hoy. |
| Orbitales d/f | d reales en Ti/Ba/Zn/Mo/W/Bi; f en oráculos/perfiles sintéticos7.1/7.4/6.5/6.7. | No se encontró validación científica con material f. Tampoco medida de máscara 4.6 en d/f; no presentar la regla «cada ⚪ se re-mide» como cumplida en todos los subcasos. |
| Carga transferida/largo alcance | LiF/ZnO/BaTiO3 y controles Qeq/defectos cargados registrados. | Datos cargados importan y existe compromiso entre estados;8.4 no resuelve toda física de superficie. |
| Aislantes frente metales | Aislantes cristalinos del banco hechos; grafeno semimetálico; Fe magnético F0/código, etiquetas F4. | Campaña magnética F5 pausada; no hay cierre de entrenamiento de metal Fe equivalente al de aislantes. No reanudarla. |
| Defectos/desorden | LiF con Schottky/vacantes/carga y grafeno con vacante; superficies ZnO medidas. | a-SiO2 no completó puerta de vidrio216 átomos y quedó pausado; historial v4/v5/v6 no autoriza relanzamiento. |
| Elementos pesados/KB/SOC | Bi2Se3 SOC y controles BaTiO3/TMD, padding 8.2 con ablación. | No es cobertura universal de magnetismo, paridades o elementos pesados; la cabecera del banco aún dice ninguno disponible. |
| Misma Z/base distinta/ghosts | Error explícito y prueba en 8.3. | Soporte completo pendiente hasta dataset/necesidad pertinente; no incumplimiento autorizado. |
| Dimensionalidad0D/2D/3D | Clústeres H2O, grafeno/bicapas, LiF/BaTiO3/ZnO;6.4 medido con heterogeneidad real h2oclu. | No concluir que la misma receta controla todos tamaños/materiales; no hace falta esperar a-SiO2 para acreditar3D. |
| Fase1 H2O/LiF/BaTiO3 | Etiquetado y campañas múltiples registrados, con límites físicos explícitos. | Archivos/datasets originales en MD_vs/no distribuidos; se auditan claims disponibles, no se recertifica cada SCF aquí. |
| Fase2 ZnO/TMD/a-SiO2 | ZnO/TMD completados y controlesKB; a-SiO2 parcial/pausado. | SOC de TMD inicialmente planeado no se convirtió en obligación tras elegir Bi2Se3; no hay evidencia de campaña SOC de MoSe2/WSe2. |
| Fase3 Fe y SOC | SOC no magnético Bi2Se3 hizo de material efectivo; Fe amplió el alcance con autorización posterior. | Entrenamientos Fe detenidos por decisión, no por falta de implementación. La lista histórica definitiva no sustituye decisiones posteriores. |
| MgO/GaAs para8.4 | Usuario eligió materiales; pilotos de red/base/gap registrados en `roadmap.md:1009`, commit775cb90. | Etiquetas/campañas completas no reclamadas ni completadas; siguiente trabajo depende del estado y permiso vigentes, no de este informe. |
| Vista igualtiempo/compute | Plan auxiliar la mantiene diferida (`roadmap_precision_rendimiento.md:771`). | No está cerrada por haber terminado seis fits6.4; tampoco es condición para negar existencia del sampler. |

El título «Banco de materiales (pendiente de definir)» (`roadmap.md:988`) ya no describe su contenido: hay lista, pilotos, etiquetas y comparativos. El cambio mínimo es fechar la introducción/tablas originales y enlazar el estado vigente por material.

## Qué sobra y trazabilidad del alcance

No se considera sobrante defectuoso una opción que quedó sin promover: conservar LowRank, SK, NodeMix, max, pair_envelope, long_range o multitarget mantiene experimentos/checkpoints reproducibles. Su coste de mantenimiento puede discutirse después; los resultados negativos por sí solos no autorizan borrarlas.

| Elemento que excede una lectura literal del roadmap inicial | Trazabilidad | Dictamen |
|---|---|---|
| 4.7 loss de espacio casi nuloS | Añadido7ce970e por diagnóstico del banco; campaña aprobada y cerrada. | Ampliación justificada, no sobra sin permiso. |
| Recuperación de entrenamientos, runners/supervisores y diagnósticos | Diseño/plan aprobado el 16 de octubre y cierres 6.4/4.7; fuera de producto principal en auditorías. | Infraestructura de campañas autorizada; no convertirla en default de Lightning. |
| Fullspin, momentos magnéticos y doble pasada temporal | `roadmap_magnetismo.md:17`, decisiones nominales `:102`, commits 4d8b56e..da478db. | Ampliación expresa posterior de 8.1; F5 pausado. No marcar feature como no solicitada. |
| Migración runner SIESTA/métricas/atribución/controlKB | Petición usuario/plan21oct; commits d28c65b..7ee73f2. | Dependencias necesarias para usar opciones sin MD_vs; diseño local deliberado. |
| DenseReadout diagnóstico no equivariante y HamiltonianLocalContext | Ya existen en44becc1; origen1a131f1/4efeb1a, no añadidos por campañas auditadas. `dense_readout.py:1` avisa de falta de equivariancia y uso opt-in. | Histórico/experimental; sin trazabilidad nueva a un punto numerado, no bug automáticamente. |
| E3nnProjectedLinearTSQNodeBlock | Derivado del perfil 2.9; filas/descripción y rechazo de receta bf16 documentados. | Opción de perfil/diagnóstico justificada, no mejora universal. |
| Archivo RTP precomputado vacío | Ruta alternativa cacheada y perfil 3.7 justifican no generarlo. | No es funcionalidad faltante ni obligación de producir archivo masivo. |
| Cinco symlinks absolutos de diagnóstico | Artefactos históricos de Comparison, no imports de producto. | Sobra portable/incompleta en un clone; límite de evidencia/archivo, no dependencia funcional MD_vs. |
| Estados de cola y planes sin casillas actualizadas | Se preservan como historia, pero no siempre están rotulados localmente. | Ruido documental y riesgo de reejecución. Reducir duplicación manteniendo la evidencia. |

## Decisiones válidas y propuesta mínima

Se puede afirmar que las principales rutas del roadmap existen y que numerosos candidatos fueron probados, incluidos resultados negativos. No se puede afirmar «todo el roadmap original implementado», «todo científicamente mejorado», «todos los defaults promovidos» ni «campañas enteramente reproducibles clonando solo este repositorio».

La intervención mínima con mayor valor es **normalizar el estado documental**: separar código/certificación/medida/promoción y cerrar las contradicciones descritas, preservando el historial. Después, y solo si el usuario lo prioriza, los huecos de alcance más concretos son la integración espectral de 4.3 y la validación general de geometría/etiqueta de 3.11. Los demás subalcances radiales/SWA/máscara/ghosts/CoB disperso no pasan a ser trabajo obligatorio por aparecer en un hallazgo original. No se propone ni ejecuta una nueva campaña científica en esta auditoría.
