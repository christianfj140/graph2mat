# Auditoría independiente de bugs — snapshot de 267 commits

Auditoría cerrada el 22 de octubre de 2026 sobre `a66b46cb0955ca344693d50737c5e9952fc257fc`. Este informe describe el estado original del 21 de octubre, antes de los arreglos posteriores de Claude. La revisión de esos arreglos y las correcciones de Codex se registran por separado en `revision_claude_20261022/`.

**Estado posterior:** los 25 contraejemplos quedan cerrados tras verificar los arreglos de Claude y aplicar las correcciones adicionales. La fuente combinada pasa 3599 pruebas CPU, con 25 omisiones. Véase [revisión final y límites](revision_claude_20261022/README.md); este informe conserva intacta la descripción del snapshot original.

## Resultado

Se confirman **25 defectos funcionales: 19 introducidos en el rango de 267 commits y 6 heredados de su base**. Ocho tienen prioridad P1 y diecisiete P2. Además existe una regresión de tests: cuatro casos del contrato JSON del servidor no actualizan sus expectativas a los nuevos campos del procesador. No se cuenta esa regresión como cuatro bugs de producto.

P1 indica resultados físicos/diagnósticos incorrectos silenciosos en una combinación admitida; P2 indica comportamiento incorrecto acotado, interrupción o pérdida de información. La prioridad no afirma que todos los entrenamientos registrados activasen la condición. Cada hallazgo incluye su escenario en el informe de área. Las hipótesis no reproducidas, las limitaciones dtype heredadas de PyTorch Lightning y las combinaciones expresamente no certificadas quedan fuera del recuento.

| ID | Prioridad | Defecto confirmado | Origen respecto a los 267 | Detalle |
|---|---|---|---|---|
| DATA-01 | P1 | Baseline DM polarizada mezcla ejes y no recupera la matriz original | Heredado | `areas/data.md` |
| DATA-02 | P1 | Override de nsc consulta bloques de otra imagen | Heredado; fill cero reciente agrava silencio | `areas/data.md` |
| DATA-03 | P1 | Guarda same-Z acepta igual dimensión con irreps diferentes | Heredado; guarda reciente incompleta | `areas/data.md` |
| DATA-04 | P2 | NetCharge incluido se sustituye por cero en QEq | Introducido | `areas/data.md` |
| DATA-05 | P2 | Conversión escalar compleja crea bloques reales y falla | Heredado | `areas/data.md` |
| DATA-06 | P2 | load_labels=False se ignora para configuraciones/muestras preparadas | Heredado | `areas/data.md` |
| M1 | P1 | Gauge por canal elimina separación física entre spins | Introducido | `areas/metrics.md` |
| M2 | P2 | Proyección con floor de S penaliza un cambio puro de gauge | Introducido | `areas/metrics.md` |
| M3 | P2 | Frac-round pierde parejas en celdas no reducidas equivalentes | Introducido | `areas/metrics.md` |
| E1 | P1 | Kernel Ewald depende de la base elegida para la misma red | Introducido | `areas/e3nn.md` |
| E2 | P1 | Corrección electrostática escalar se suma solo a H↑ | Introducido | `areas/e3nn.md` |
| E3 | P2 | Checkpointing desconecta readout cuando MACE está congelado | Introducido | `areas/e3nn.md` |
| L1 | P2 | Cargar un checkpoint fuera de fit pierde la loss de su etapa | Heredado | `areas/lightning.md` |
| L2 | P2 | CSV por muestra bajo DDP omite ranks distintos de cero | Introducido | `areas/lightning.md` |
| L4 | P2 | Selección de checkpoint utiliza métrica obsoleta tras cambiar loss | Introducido | `areas/lightning.md` |
| T1 | P1 | Aritmética Δ suma imágenes distintas si sc_off difiere | Introducido | `areas/tools.md` |
| T2 | P1 | Métricas declaran perfecto un Hamiltoniano con spin-down incorrecto | Introducido | `areas/tools.md` |
| T3 | P2 | Cache H0 reutiliza salida de proceso fallido sin validar geometría | Introducido | `areas/tools.md` |
| T4 | P2 | Gauge del piloto DM devuelve NaN ante error off-diagonal | Introducido | `areas/tools.md` |
| T5 | P2 | Parser SystemLabel no respeta normalización FDF | Introducido | `areas/tools.md` |
| T6 | P2 | Proyección canónica vacía conserva todas las direcciones | Introducido | `areas/tools.md` |
| T7 | P2 | Desglose agregado falla con predicción exacta | Introducido | `areas/tools.md` |
| T8 | P2 | Imagen mínima del desglose falla con posiciones no envueltas | Introducido | `areas/tools_review.md` |
| X1 | P2 | Runners registran updates solicitados distintos de los ejecutados | Introducido | `areas/tools.md` |
| X2 | P2 | Primer job de energía puede arrancar sin comprobar guardia térmica/VRAM | Introducido | `areas/tools_review.md` |

**D1 — regresión de tests:** `tools/server/tests/test_server_app.py::test_model_info_serializes_real_processor_as_json`, cuatro parametrizaciones. La respuesta incluye correctamente `edge_cutoff_padding` y `magnetic_moments`, pero la expectativa no incluye esos campos. El entorno habitual omite el módulo completo por falta de FastAPI y por eso no descubre la regresión. Se reproduce con el perfil CPU completo disponible.

## Evidencia y cobertura

Se revisaron los cambios y consumidores de cinco áreas de producto: métricas/física; datos/matrices/core; e3nn/MACE/long-range; Lightning/checkpoints/callbacks/CLI; servidor/herramientas migradas. La revisión adicional de experimentos trazó presupuestos, integridad/selección de checkpoints, recuperación, publicación y guardias de lanzamiento. La profundidad de scripts históricos está declarada en los informes; no se certifica lectura exhaustiva de cada línea de todos los benchmarks ni todas las combinaciones de parámetros.

La base remota es `a5e81345c8494d6cc1c7f9c917b46467507d3b97`; `origin/g2m-v2..a66b46c` contiene exactamente **267 commits**. Frente a main `a9b011c` son 271: se distinguen ambos rangos. El inventario original está en `evidencia/commits_267.tsv` y el diff por archivo en `evidencia/diff_main_numstat.tsv`.

Se detectaron extensiones Cython obsoletas en el entorno editable. Todas las reproducciones auditadas usaron una copia inmutable con ambas extensiones compiladas desde a66b46c. Se verificó que ningún archivo Python de esa copia cambió. Hashes/imports/build en `evidencia/environment.json`, `runtime_imports.json`, `build_extensions.log` y `frozen_source_verified.json`.

Los contraejemplos de datos/métricas se repitieron desde coordinación; herramientas tuvieron una revisión independiente que intentó refutarlos; las otras áreas conservan reejecuciones, aserciones y hashes. Hay controles positivos: oráculo de **60 combinaciones** de especies/agrupación/simetría/componentes sin fallos, pruebas focales de física/multitarget y controles Ewald/etapas. Las reproducciones se empaquetan en `reproducciones/`; necesitan el código del snapshot y sus dependencias, y algunas usan fixtures incluidos en el repositorio.

## Verificación del snapshot

La ejecución general inicial `pytest src tests` produjo **3341 passed, 38 skipped, 5 failed**, en 1058.88 s. Cuatro fallos eran `dscribe` ausente; el quinto era el benchmark que consulta `git rev-parse` desde un gitarchive sin metadatos Git. Estos cinco son problemas de preparación del entorno, no cinco defectos del producto.

Se corrigió únicamente la preparación temporal: perfil CPU completo preexistente, con FastAPI/dscribe, y metadatos Git temporales apuntando al mismo a66b46c. El gate focal de servidor más benchmark produjo **172 passed y 4 failed**, en 54.80 s. Los cinco casos anteriores pasan; los cuatro fallos son D1. No se afirma una suite general verde ni se suman como si fueran ejecuciones disjuntas. Logs y JUnit completos: `evidencia/pytest_general.*` y `evidencia/pytest_followup.*`.

No se ejecutaron GPU, cálculos SCF, nuevos fits científicos ni cambios de cola. No se alteró código productivo durante esta auditoría original; el árbol cambió posteriormente por el trabajo de Claude, registrado en otra fase. No se revalidaron datasets completos, OOD, precisión ML o coste científico; los contraejemplos no autorizan atribuir por sí solos un cambio de cifras a campañas históricas.

## Decisión

El snapshot no permite declarar «sin bugs» ni «todo el roadmap completo». Antes de publicar conviene cerrar los P1 y las rutas P2 que se vayan a admitir. Las soluciones mínimas y escenarios están descritos por área. La fase posterior contrasta cuáles resolvió Claude, verifica sus regresiones y corrige lo que siga mal bajo la autorización del 22 de octubre; ese estado posterior no reescribe la evidencia original.
