# Auditoría de bugs y del roadmap de `g2m-v2` (2026-10-21)

Estado auditado: `g2m-v2` @ `a66b46c` (tras el merge con upstream y la migración desde MD_vs_AtomicDisplacement).
Seis revisiones en paralelo, solo lectura: roadmap frente a código, y bugs en cinco zonas (pérdidas y métricas;
Lightning; datos y `Graph2Mat`; readouts e3nn y modelos; servidor, CLI y herramientas migradas). Cada hallazgo
«confirmado» tiene una reproducción mínima en CPU; «plausible» es razonamiento sobre el código sin reproducir. Los
hallazgos graves se reprodujeron de nuevo de forma independiente antes de este informe. No se ha cambiado código.

Tabla completa del roadmap: [auditoria_roadmap_vs_codigo_20261021.md](auditoria_roadmap_vs_codigo_20261021.md).

## Resumen

- **Roadmap (81 filas):** 64 OK, 8 con discrepancia, 0 ausentes, 5 decisiones del usuario, 4 históricas sin código.
- **Bugs:** 39 (31 confirmados, 8 plausibles): 6 graves, 18 medios, 15 menores. No se encontró ninguna rotura de
  equivarianza ni de hermiticidad (rotaciones propias e impropias, error relativo ≤ 3,5e-15).
- **Origen:** la mayoría está en código de `g2m-v2`. Algunos se heredaron al portar MD_vs tal cual (espín solo en el
  canal 0, `set_nsc` que recorta H0, cachés sin clave suficiente); uno viene de upstream (`group("max")` con
  `AtomicTableWithEdges`); dos son del código nuevo de hoy (el CLI `graph2mat siesta delta h0` y el
  `delta_recipe.yaml`).

## Impacto en resultados ya publicados en el roadmap

1. **Ningún entrenamiento histórico usó el learning rate coseno.** Las configs tenían `optimizer: null` y
   `lr_scheduler: CosineAnnealingLR`; Lightning solo engancha el scheduler si hay `optimizer`, y
   `LitMACEMatrixModel.configure_optimizers` devuelve un Adam fijo. Comprobado en checkpoints reales de 6x6
   (incluida la receta de 16k pasos), w90 y H2O: 0 schedulers guardados y lr = 0,0018 en la última época. Las
   comparaciones entre variantes siguen siendo justas (todas a lr constante), pero las recetas descritas como
   «cosine» no lo eran, y queda sin medir si el scheduler mejora la precisión.
2. **La pérdida del espacio nulo de S (4.7) no es invariante de gauge** para columnas con λ < 1e-3: penaliza el
   desplazamiento c·S que la pérdida de gauge deja libre. Puede explicar parte de la degradación de H medida en 4.7.
3. **La partición automática train/val depende del orden de listado del sistema de ficheros** (glob sin ordenar):
   el mismo `split_seed` da otra validación desde un zip o en otra máquina. Las particiones históricas siguen siendo
   las que fueron; el riesgo es reproducirlas o reanudar desde otro sitio.

## Bugs graves

| # | Zona | Ubicación | Defecto | Evidencia |
|---|---|---|---|---|
| G1 | datos | `core/data/matrices/basis_matrix.py:105`, inversa `processing.py:1397-1400` | Con `sub_point_matrix=True` (por defecto para DM) la referencia atómica se resta de un bloque `(n,n,C)` por broadcasting sobre (columna, componente): DM con espín polarizado sale mal o falla | átomos de 2 orbitales: ida y vuelta difiere 0,70; de 4 orbitales: `ValueError` de broadcasting |
| G2 | Lightning | `tools/lightning/cli.py:171-363`, `models/mace.py:455-523` | `lr_scheduler` sin `optimizer` se acepta y se ignora | ver «Impacto» 1 |
| G3 | Lightning | `tools/lightning/data.py:632`, `_helpers.py:192` | Split automático depende del orden del glob | mismo seed: validación {03,13,15,16,20,21} en directorio vs {00,06,12,13,15,21} en zip |
| G4 | Lightning | `tools/lightning/callbacks.py:128-151,185-216` | Receta de gauge que predice solo H con etiquetas (H, S): `MatrixWriter` escribe S = identidad sin avisarlo | S escrita [1,0,0] frente a la de referencia [1,0.2,0.15] |
| G5 | herramientas (nuevo) | `tools/cli/siesta/delta.py:23` | `delta h0` nombra la carpeta de H0 solo por el nombre del directorio: `A/run` y `B/run` comparten H0; con `.` borra todo el directorio de salida | reproducido con SIESTA falso |
| G6 | herramientas | `tools/siesta/delta.py:88-110` | `compute_h0` reutiliza un TSHS escrito por un SIESTA que luego falló, sin comprobar la geometría | H0 desplazado 0,5 Å devuelto como válido |

## Bugs medios

| # | Ubicación | Defecto |
|---|---|---|
| M1 | `core/data/metrics.py:1003-1004` | Pérdida 4.7 no invariante de gauge (ver «Impacto» 2); arreglo: ajustar el desplazamiento con `G = CᵀSC` o pasar los deltas alineados |
| M2 | `tools/siesta/delta.py:123-125` | `add_h0` recorta H0 a la supercelda de la predicción (`set_nsc`): con nsc de SIESTA mayor que la del modelo, H0 no se suma entero (25 meV en la prueba) |
| M3 | `tools/analysis/{metrics,breakdown,attribution}.py`, `spectral_targets.py:42` | Solo se evalúa el canal de espín 0: errores en espín down invisibles |
| M4 | `tools/analysis/kb_tail.py:29-35,50-51` | Con espín polarizado: con `baseline` devuelve H sin espín; sin él solo corta el canal 0 |
| M5 | `tools/siesta/dm_pilot.py:50-52` | Caché por `<tipo>_<nivel>` sin semilla ni hash del DM: repetir con otra semilla mezcla ejecuciones |
| M6 | `tools/server/extrapolation.py:345-381` | `HistoryCopy` desalinea descriptor y etiqueta cuando falta un paso; un test fija el valor erróneo (no se usa por HTTP/Lua) |
| M7 | `tools/lightning/callbacks.py:115-151` | `MatrixWriter` (un objetivo) nombra la salida por el directorio: varias estructuras en una carpeta se sobrescriben |
| M8 | `tools/lightning/callbacks.py:358-392` | `SamplewiseMetricsLogger` compara etiquetas (H, S) con predicciones solo H: todo NaN o excepción |
| M9 | `tools/lightning/callbacks.py:343-345` | Bajo DDP solo rank 0 escribe su parte: el CSV por muestra cubre 1/world_size |
| M10 | `models/mace.py:762-788` | `long_range` toma la última columna como S sin comprobar la política: con `spin_h_only` usa H_down como S; con espín polarizado solo desplaza H_up; no se rechaza con SOC |
| M11 | `models/mace.py:612-629`, `hamiltonian_context.py:174-189` | Los canales del contexto Hamiltoniano no se anulan en el corte: con contexto, la envolvente por pares (7.3) no actúa (bloques de 0,49 a 4,199 Å con corte en 4,2 Å) |
| M12 | `models/mace.py:563-573` | `gradient_checkpointing` + etapa que congela MACE: la pérdida no tiene grad y `backward()` falla |
| M13 | `core/data/table.py:336-347,368-381` | `basis_shape`/`max` agrupan bases con distinto orden de capas (solo `PointBasis` hechos a mano) |
| M14 | `core/data/processing.py:2008-2047` | `from_config(nsc=X)` con X distinto del de la matriz lee bloques de otra imagen |
| M15 | `bindings/torch/data/coefficient_targets.py:121,178` | Entrenamiento por coeficientes, matriz no simétrica y aristas de la misma especie: `KeyError` |
| M16 | `bindings/torch/data/dataset.py:178` | `NetCharge` leído con regex: `Net.Charge` o `%include` dan carga 0 (afecta a `long_range="qeq"`) |
| M17 | `tools/lightning/model.py:249`, `data.py:575` | (plausible) Orden de especies tomado de un glob sin ordenar en modelo y datos por separado, sin comprobar que coincidan |
| M18 | `models/mace.py:772-776`, `long_range.py:115` | (plausible) Signo de la respuesta a `NetCharge` en QEq probablemente invertido (+V·S con V potencial electrostático) |

## Bugs menores

- `core/data/metrics.py:1080,1106,1141,1144`: pérdidas por estructura/dominio promedian también estructuras sin etiquetas.
- `tools/server/api_client.py:130`: `predict(geometry=<ruta>, output=DensityMatrix)` lee el DM sin geometría.
- `tools/siesta/runner.py:42`: `system_label` no normaliza la etiqueta fdf (`System.Label`).
- `tools/siesta/label_noise.py:46-49`: comentarios dentro del bloque de coordenadas rompen el desplazamiento; `%block … < fichero` no se desplaza.
- `tools/siesta/templates/lua/graph2mat.lua:111-134`: parámetros sin codificar en la URL (plausible).
- `tools/analysis/attribution.py:89,157-158`: `set_nsc` recorta la predicción/H0 a la supercelda de la referencia (plausible).
- `tools/siesta/delta.py:150-151`: `add_h0` en sitio no es atómico (plausible).
- `tools/lightning/model.py:436-442,1346-1356`: etapa con `freeze` falla bajo DDP con `find_unused_parameters=False`.
- `tools/lightning/model.py:1144-1180`: la etapa activa no se restaura fuera de `fit` (plausible).
- `tools/lightning/model.py:1069`: `output_scaleshift` sin datamodule falla con `AttributeError` (plausible).
- `models/mace.py:601-606`: `mp_cutoff` sin `readout_radial_embedding` da bloques nulos más allá del corte (solo uso directo).
- `models/long_range.py:29`: cota de vectores k incorrecta para celdas oblicuas (error relativo 7e-7).
- `models/mace.py:289-291,297-331`: opciones ignoradas en silencio en algunas combinaciones.
- `core/data/table.py:381`: `group("max")` con `AtomicTableWithEdges` falla (heredado de upstream).
- `core/data/processing.py:1286-1315`: el respaldo por suma de subconjuntos puede elegir aristas arbitrarias (plausible).

## Roadmap frente a código: discrepancias

1. **8.1:** la receta SOC con Δ (proyección de la parte que viola la inversión temporal, H0 sin espín, predicción a
   fichero) solo existe en `auditorias_mejoras/run_soc_campaign.py`, que depende de MD_vs; `MatrixWriter` y el servidor
   rechazan SOC. Es el único hueco de la migración que deja una capacidad sin sustituto en graph2mat.
2. **6.8:** las docstrings ofrecen `mode="reduce-overhead"` de `torch.compile`, descartado por segfault con torch 2.11.
3. **1.13:** los valores por defecto anunciados al cerrar la Ola 1 no se cambiaron (decisión del usuario).
4. **Sin tests:** 1.9 (`E3nnEdgeBlockNodeMix`), 6.9 (`cueq=True`), 4.7 (`spectral_window="<fichero>.npz"`), 0.8
   (guardas de rechazo) y el ejemplo `docs/examples/benchmarks/`.
5. **Texto desactualizado:** 2.8 y 1.12 (ya hechos), 8.2 en «Pendiente real», 4.2 cita scripts de MD_vs; tablas
   rotas por `|` sin escapar en 5.1, 7.7, 8.1 y 8.2; la línea 1 del roadmap es un fragmento suelto.
6. **Planes** con casillas sin marcar aunque el trabajo está hecho: 6.7, 7.6, 8.1 (SOC y magnético), 8.2 y migración.

## Orden de arreglo propuesto

1. Lo que da resultados físicos erróneos sin avisar: G1, M10, M3/M4, M2, M11, M1.
2. Reproducibilidad y datos: G3 (y M17) con `sorted`, G2 (rechazar o enganchar el scheduler), G4, M7, M14.
3. Robustez de las herramientas nuevas: G5, G6, M5.
4. Huecos del roadmap: 8.1 SOC sin MD_vs, tests que faltan, texto y tablas.
5. Menores.

Arreglar G2 y G3 cambia el comportamiento respecto a los entrenamientos históricos (scheduler activo, otra
partición): conviene anotarlo en el roadmap al hacerlo.

## Estado de los arreglos (2026-10-22)

Cada arreglo lleva un test que falla sin él (salvo donde se indica) y se comprobó con los tests focalizados de su
zona; la suite completa se pasa al final. Los arreglos de las herramientas migradas se volvieron a comparar con el
código original de MD_vs_AtomicDisplacement sobre datos reales: sin cambios para matrices sin espín (0,0).

| Hallazgo | Estado | Commit |
|---|---|---|
| G1 referencia atómica en bloques multicomponente | arreglado | `ba9b440` |
| G2 `lr_scheduler` ignorado con `optimizer: null` | arreglado; efecto histórico anotado en el roadmap; sonda coseno vs constante | `81b18ca`, `69af9aa`, `d2082af` |
| G3 + M17 orden del glob y tipos de base | arreglado | `2caeb69` |
| G4 + M7 + M8 `MatrixWriter` y `SamplewiseMetricsLogger` con predicciones solo H | arreglado | `6509fad` |
| G5 + G6 + M5 claves y caché de SIESTA | arreglado | `eca05a8` |
| M1 pérdida 4.7 no invariante de gauge | arreglado; efecto en 4.7 anotado | `75e75d2` |
| M2 + M3 + M4 supercelda de H0 y espín en herramientas | arreglado | `2c3b840` |
| M6 `HistoryCopy` | arreglado (dos tests fijaban el valor erróneo) | `1e12f10` |
| M9 métricas por muestra bajo DDP | arreglado | `b52c73b` |
| M10 `long_range` y la columna S | arreglado | `5a4d900` |
| M11 contexto Hamiltoniano y envolvente | arreglado | `4b5194b` |
| M12 checkpointing con partes congeladas | arreglado | `624101e` |
| M13 + `group("max")` de tablas atómicas | arreglado | `4dd8f9c` |
| M14 `from_config(nsc=X)` | arreglado (error explícito) | `f4fe6f7` |
| M15 coeficientes con aristas de la misma especie | arreglado | `67aa8e0` |
| M16 `NetCharge` | arreglado | `fb236ca` |
| M18 signo de QEq con carga neta | arreglado (convención documentada) | `a492fbf` |
| 8.1 SOC con Δ sin MD_vs | implementado | `c6c4850` |
| 6.8 `reduce-overhead` | rechazado explícitamente | `55dbbde` |
| Tests que faltaban (1.9, 4.7, 0.8, 6.9, ejemplo) | añadidos | `f3b4d29` |
| Menores de herramientas (etiqueta fdf, `noise_fdf`, cliente, atribución) | arreglado | `17e246f` |
| `add_h0` en sitio no atómico | arreglado (escritura temporal + `os.replace`) | `c6c4850` |
| Etapas con `freeze` bajo DDP, etapa activa fuera de `fit`, `output_scaleshift` sin datamodule | arreglado | `2248f52` |
| Respaldo por suma de subconjuntos en aristas simétricas | eliminado: ahora lanza error | `d3d7fe5` |
| Parámetros de la URL en `graph2mat.lua` | codificados (`url_encode`); comprobado con lupa, sin test en la suite | `2df88f8` |
| `mp_cutoff` sin embedding radial, opciones ignoradas en `mace.py` y `E3nnGraph2Mat`, vectores k en celdas oblicuas | arreglado | `e2d204a` |
| Test de autocast dependiente del orden (visto al pasar los tests) | arreglado | `90361cc` |
| Texto del roadmap (2.8, 1.12, 4.2, 8.2, tablas, primera línea suelta) | arreglado | commit de este informe |
| Casillas sin marcar en los planes (6.7, 7.6, 8.1, 8.2, migración) | **no cambiado**: los planes son historia; el estado vigente está en el roadmap | — |
| Pérdidas por estructura promedian estructuras sin etiquetas | **no cambiado**: es un diseño coherente y probado en el paso, la época y DDP (tests explícitos); cambiarlo exige rehacer la agregación por época. Limitación conocida: un lote con estructuras sin ninguna etiqueta diluye la pérdida. | — |

## Revisión independiente de Codex (2026-10-22)

Codex revisó estos arreglos (`auditorias_mejoras/auditoria_codex_20261021/revision_claude_20261022/`) y corrigió
lo que quedaba mal; revisado por Claude y commiteado en `1c5229d`, `0a83cd2` y `86eb980`:

- El gauge de la pérdida usaba un desplazamiento por canal de espín y ocultaba errores rígidos del desdoblamiento;
  ahora es común (no afecta a cifras registradas).
- Los hooks de `2248f52` imponían la pérdida de la última etapa; ahora se restaura la etapa guardada en el checkpoint.
- «Todos los entrenamientos con LR constante» era exagerado: los runners de 6.4/4.7 y 7.6 instalaban su scheduler.
- Huecos adicionales: capas de la base, valores complejos, `load_labels=False`, métrica de selección obsoleta,
  recibos de caché SIESTA, includes fdf, imágenes periódicas, imagen mínima general y presupuesto de pasos.
- `aligned_error` del piloto DM cambia de normalización (nota en la fila 7.7 del roadmap).

Comprobado: los originales de MD_vs siguen dando lo mismo en datos reales (G1–G8), salvo G9 por ese cambio de
normalización.
