# Diseño propuesto: splits por trayectoria y provenance (3.12)

Estado: automatización estricta histórica no implementada; guardrail de aviso
implementado el 2026-10-19, con el alcance limitado siguiente. Alcance exclusivo Graph2Mat.

## Alcance ejecutado posteriormente (2026-10-19)

Usuario«Haz 3.12» tras la propuesta en chat de alarma, sin rehacer particiones.
Se implementa únicamente el guardrail ligero descrito en `docs/data_loading.rst`:
identidades de fuentes y sample_id/sha256/group_id explícitos; ausencia o
metadata inválida es aviso de independencia no evaluable, nunca bloqueo.
Sin I/O de matrices adicional ni umbral de similitud. Metadatos opcionales en
`runs_json.split_metadata` o `sample.metadata.split_guardrail`, sin nueva opción
del constructor ni defaults científicos. El manifiesto/generador/fail-closed
que sigue permanece histórico y **no está autorizado ni implementado**.

## Decisión posterior (2026-10-14)

El usuario indica que ya hizo las particiones manualmente y permite aplazar esta
mejora o limitarla a una alarma no bloqueante. Se elige aplazarla para conservar
la prioridad de precisión/rendimiento. No regenerar ni modificar sus splits.
Las secciones de manifiesto/generador y validación fail-closed de este documento
son el diseño anterior, **no una autorización ni requisitos activos de ejecución**.

Si se retoma: proponer un guardrail pequeño de avisos por muestras duplicadas,
grupos explícitos compartidos o falta de metadata para valorar independencia.
No impedir entrenamientos ni alterar particiones por esas alarmas. Separar
«solapamiento detectado» de «independencia no evaluable»: información ausente no
demuestra contaminación. La diferenciación geométrica/distribucional requiere
definir métrica y umbral; no incluirla como heurística especulativa. No se ha
auditado el reparto manual ni se afirma que contenga fugas.

## Evidencia y límites

`MatrixDataModule` ya ofrece `split_seed` y `val_fraction`; sus tests fijan el
default histórico (seed 32, 10 %). El dataset admite configuraciones, geometrías,
FDF locales y ZIP. Sus rutas identifican fuentes, no garantizan identidad de
trayectoria. La provenance del checkpoint registra versiones/checkout, pero no
demuestra independencia de las particiones. Los datos congelados w90/6x6 y sus
configs conservan train/validation, sin contrato explícito de trajectory_id.
No se puede certificar independencia MD inventando grupos a partir de carpetas.

## Contrato propuesto

Entrada opt-in: manifiesto JSON externo de muestras, no inferencia de grupos.
Cada registro declara `sample_id`, `group_id`, `source`, `sha256` y `split`.
`source` es ruta relativa al root declarado o par `{archive, member}` para ZIP;
el archivo ZIP no es por sí mismo una trayectoria. `group_id` identifica la
unidad independiente más amplia relevante: trayectoria MD, estructura padre o
familia de generación correlacionada, según el generador del dataset.

Cabecera: `schema_version=1`, política de grupo documentada, seed, proporciones
solicitadas y hashes de los registros ordenados por sample_id. El manifiesto
aprobado fija las particiones; cargarlo no regenera el split. Su digest se añade
a la provenance del checkpoint, junto a conteos por split/grupo, sin incrustar
rutas absolutas sensibles. Checkpoints antiguos continúan cargando sin él.

Un script preparatorio Graph2Mat puede generar una propuesta reproducible a
partir de identidades explícitas. Ordenar grupos antes de usar la seed elimina
dependencia del orden de entrada. Asignar grupos completos, informar proporción
real obtenida y no prometer tamaños exactos cuando las trayectorias sean desiguales.
La primera implementación debe separar claramente preparación y consumo del
manifiesto, sin sustituir el split aleatorio histórico cuando la opción no está activa.

## Validación fail-closed

- sample_id duplicado: error; mismo hash de contenido en splits distintos: error.
- group_id en varias particiones: error, aunque sus miembros estén en ZIP distintos.
- Falta de identidad/hash/source o hash discrepante: error antes de entrenamiento.
- Un solo grupo con validación solicitada o grupos insuficientes: error explicativo,
  nunca dividir una trayectoria para alcanzar una proporción.
- Muestras adicionales o ausentes respecto al manifiesto: error; no descartarlas
  silenciosamente. `val_runs` explícito no debe competir con otro split activo:
  rechazar ambas políticas simultáneas antes de crear datasets.
- ZIP: validar member normalizado, existencia y contenido; rechazar paths con `..`
  o absolutos. Hash de contenido del member, no solo mtime del contenedor.
- Declarar correlaciones compartidas entre trayectorias cuando el generador las
  conozca; un hash distinto no demuestra independencia física.

## Pruebas de aceptación de implementación futura

Extender `test_datamodule_split.py`: grupos A/B/C distribuidos sin intersección;
reordenar entradas produce igual asignación; mismo seed reproduce; grupo único,
ID ausente, hash modificado y fuentes faltantes se rechazan. Duplicados de
contenido con distinto nombre no atraviesan splits. Dos ZIP con miembros del
mismo grupo permanecen juntos. Probar explícitamente proporciones inviables y
proporciones reales distintas por grupos desiguales. Mantener verdes todos los
tests de defaults existentes y round-trip provenance/checkpoint antiguo.

## Puerta de aprobación

Confirmar quién aporta group_id y qué unidad física representa antes de escribir
código. Implementar solo loader/validador y provenance en Graph2Mat si se aprueba
ese alcance; modificar generadores o benchmarks de MD_vs_AtomicDisplacement
requiere autorización separada. No regenerar ni corregir retrospectivamente
campañas publicadas: una partición distinta crea una campaña nueva.
