# Preflight de h2oclu y recuperación — 2026-10-16

Goal activo en `graph2mat/g2m-v2`, HEAD
`270d17d1f204293a059baa09e8f1b3756981e2bf`. Inspección de solo lectura;
no implementación, entrenamiento, modificación de datasets ni escritura en cola.
La aprobación de dataset/observables está registrada en el traspaso. Sigue
pendiente, al realizar este preflight, la aprobación del enfoque de recuperación.
**Actualización posterior:** usuario aprobó ese enfoque mediante «Apruebo» y
el [diseño](../docs/superpowers/specs/2026-10-16-training-recovery-design.md) mediante
«Listo, apruebo». Plan aprobado mediante «Apruebo» e implementación Native en
curso. Preflight ampliado por prepare:39 versiones homogéneas SIESTA5.4.2-11-g4e9a46060,
Smin0,0012563677/condición máxima4180,4797, H/S hermíticas bajo1e-8 relativo.
Los resultados de inspección
siguientes no cambian.

## Afirmación y evidencia

- Afirmación limitada: las referencias existentes de h2oclu son compatibles con
  preparar la comparación fixed/cost aprobada. No se afirma mejora de precisión,
  velocidad, VRAM ni continuidad de entrenamiento todavía.
- Evidencia necesaria para el resultado final: recuperación certificada, piloto
  de memoria, presupuesto autorizado, campañas pareadas, evaluación física común,
  variabilidad y contabilidad de coste. No está obtenida por este preflight.
- Guías aplicadas: `graph2mat-validation` y `scientific-experiment-audit`.

## Datos y comprobaciones realizadas

Raíz externa, solo lectura:
`$MDVS_ROOT/Comparison/results/dataset_design_curves_v1`.
Manifest: `g2m_improvements/bench/h2oclu/frozen_split_manifest.json`, SHA256
`90f446b3a6577f81cc2b1a86a0f0e4f20acb06ba5d8b0503effc64267c7e79d3`.
Su esquema contiene material/rows, pero no hashes de artefactos ni identificadores
explícitos de trayectoria. No confundirlo con el futuro manifest de campaña.

- Train: 8 estructuras de 6 átomos/46 orbitales, 12 de 12/92 y 12 de 24/184.
- Validation: 2 de 12/92 y 2 de 24/184.
- Test OOD: 3 de 96/736, clusters de 32 moléculas; no se añadieron monómeros
  ni se modificaron splits.
- Las 39 referencias TSHS, marcadores `0_NORMAL_EXIT` y mensajes de convergencia
  SCF están presentes. Leídas con sisl 0.16.4/NumPy 2.4.4 en CPU: datos finitos,
  espín no polarizado y `nsc=(1,1,1)`. Geometrías TSHS/FDF coinciden con tolerancia
  de comprobación de 1e-6 Å; ningún caso rechazado.
- H/O `.ion.xml` y H/O `.psf`: un SHA256 por archivo en las 39 estructuras.
  Ambas bases XML también son idénticas byte a byte a las usadas por la receta
  histórica en `MD_vs_AtomicDisplacement/AtomDisplacement/relaxed`.
- Configuración FDF común comprobada: GGA/PBE, DZP, PAO.EnergyShift 0.03 eV,
  MeshCutoff 200 Ry, SpinPolarized F, DM.Tolerance 1.d-5. `sisl.get` convierte
  MeshCutoff a eV: 2721.138624598741; no es una diferencia de receta.
  ElectronicTemperature, SCF.MustConverge y NetCharge no están explícitos en
  estos FDF; no se inventaron sus defaults. Un RUN.out inspeccionado identifica
  SIESTA `5.4.2-11-g4e9a46060`; no se ha certificado la versión de las 39 salidas.
- Leídos los 39 H0 existentes: dimensiones compatibles y diferencia máxima de
  geometría frente a referencia de 0 Å. Leídas las 36 etiquetas Δ de train/val:
  máximo absoluto de `ΔH-(H-H0)` sobre matrices CSR, 8.881784197001252e-16 eV;
  diferencia de S respecto a referencia, 0. Datos Δ finitos. No se generaron
  etiquetas ni se llamó a funciones que las regeneran.

## Procedencia y límites científicos

El constructor histórico `h2o_cluster_samples` usa seis grupos de generación en
train, dos en validation y tres en test, con semillas distintas por grupo. Dentro
de cada grupo hay snapshots de una misma trayectoria. Esta es evidencia del
procedimiento del código, no un certificado de lineage de cada archivo: el
manifest carece de semillas/trajectory_id explícitos y el extxyz inspeccionado
tampoco los incluye. Mantener el split manual aprobado, declarar correlación
interna y no afirmar independencia entre todos los snapshots.

Fuente inspeccionada: `Comparison/scripts/g2m_bench_materials.py`, SHA256
`94b8217c610f6f40dbd8fa821073b68f4e0490c6e0408520fba9da6ee236f0cb`.
Los tres OOD ya se observaron en campañas anteriores: evidencia exploratoria,
no test final independiente. No se calcularon errores de nuevos modelos ni se
usó OOD para elegir presupuesto de coste. No se ha auditado aún cercanía de
estructuras bajo simetrías ni independencia efectiva de las trayectorias.

Observables aprobados: H alineado al gauge y niveles/gap moleculares. La
evaluación espectral deberá resolver el problema generalizado con S de referencia
y declararse condicional a ese S. Este preflight no certifica todavía positividad
de S, política de canonización, ventana espectral, soporte de métricas ni gaps.

## Evidencia operativa que condiciona el diseño

- `run_bilayer_edge_updates.guarded_fit` crea un intento nuevo tras ForeignGPU
  sin pasar `ckpt_path`; el fit actual comienza de nuevo. Su selección best-val_loss
  depende de la carpeta de cada intento.
- En Lightning instalado, `ModelCheckpoint.load_state_dict` no restaura
  best_model_score/best_k_models/kth_value/last_model_path si cambia `dirpath`.
  No basta pasar pesos o cambiar únicamente `ckpt_path`: hay que conservar el
  estado de selección científica y comprobarlo con regresiones.
- `fit_loop.on_advance_end` llama callbacks no monitorizados, después el hook
  del módulo y después callbacks monitorizados. El scheduler no-plateau por época
  se actualiza al último batch; los schedulers plateau se actualizan después de
  los hooks de época. No se certifica una frontera arbitraria solo por llamarse
  `on_train_epoch_end`; las pruebas deben cubrir el orden efectivo soportado.
- El loader fijo usa RandomSampler sin generador explícito y workers persistentes.
  En Torch instalado, crear un iterador consume una base_seed, mientras resetear
  workers persistentes no vuelve a ejecutar ese constructor. Restaurar solo el RNG
  global no demuestra el mismo orden tras reiniciar procesos. El sampler de coste
  ya tiene semilla por época; ambos brazos necesitan una política pareada y
  certificada, sin cambiar defaults de Graph2Mat ni fingir equivalencia con la
  trayectoria de las referencias históricas.

Estas son inspecciones de código, no pruebas dinámicas de recuperación. Fuentes
propias inspeccionadas, SHA256:

- `run_bilayer_edge_updates.py`: `5a97dc9b32c16c4967233fa9e145bcdf40b252172b89c78fe445632e907c3216`.
- `bench_pipeline_gpu.py`: `878249f4f07019fb9d65df82248c078e456b8ddef50747f466d141f754adce1f`.
- `src/graph2mat/tools/lightning/data.py`: `b79347d3da5d3dcecc6ac2c119b228bcb8c5b89062b4e2af7541b1f6174a0bc5`.

## Auditoría y siguiente paso

Nivel actual: **Diagnostic-only**, preflight parcial favorable; no comparación
científica ejecutada. MODERATE: lineage incompleto, snapshots correlacionados,
solo tres OOD y test previamente observado. BLOCKER para declarar una mejora:
faltan recuperación probada, campaña autorizada y resultados pareados.

Decisión propuesta tras el preflight, ya aprobada posteriormente: recuperar desde
frontera de época, con checkpoint atómico separado del mejor val_loss y contabilidad
del trabajo repetido. La alternativa intraépoca añade estado de
sampler/workers/prefetch y no se considera exacta sin prueba específica.
Después siguen revisión del diseño/plan y ejecución; la lista y presupuesto
concretos de entrenamientos científicos tendrán su propia aprobación.
