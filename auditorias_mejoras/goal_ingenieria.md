# Goal de ingeniería Graph2Mat — 2026-10-07

Objetivo activo: cerrar **completamente** 3.1, 3.4, 3.5, 3.6 y 3.9 del roadmap,
con código, regresiones y evidencia. Un sublote verde no completa el goal.

## Estado y límites

- Checkout solicitado: `/home/christian/repositorios/graph2mat`, `g2m-v2`.
- HEAD inicial verificado: `d7c87f4`; sin modificaciones tracked previas.
- Preservar `.agents/`, `.claude/`, `agent-skills/`, el informe de evolución y los planes previos untracked.
- Implementación directa autorizada por el usuario, sin consultas de continuación entre bloques.
- Sin fonones, otros repositorios, cambios en el entorno compartido, entrenamientos,
  `g2m_autorun.py`, colas, push, merge ni publicación.
- Preservar APIs, Python >=3.9, checkpoints y los objetivos científicos existentes.
- Python compartido se usa solo para ejecutar pruebas CPU; instalaciones nuevas van a entornos temporales aislados.
- Evidencia inicial: `/tmp/g2m-engineering-goal.mJBRD9/baseline.log`.
- Línea base reproducida: **515 passed, 1 skipped, 6023 warnings**, 62.05 s.
- Turno de goal anterior interrumpido: solo lectura de skills/estado del goal, sin cambios de producto.

## Cobertura obligatoria y auditorías leídas

| Punto | Auditorías asignadas | Evidencia necesaria para cerrar |
|---|---|---|
| 3.1 | LOSS_TRAIN-06, ENG_QUALITY-23 | Agregación global de pérdidas y diagnósticos, medias/std/máximos, coeficientes/pares/ventana espectral, denominadores válidos, reset por época/split, batch_size explícito y errores fail_ok visibles; tests invariantes al batching |
| 3.4 | PHYS_H-21, ENG_QUALITY-15/16, PHYS_H-22, PHYS_H-20 | Servidor /info,/test,/predict y escritura segura; H/EDM físicamente válidos según su formato; extrapolación sin mutar matrices ni usar índices de relleno, periodicidad SOAP/dependencia explícita, API HistoryCopy compatible; mapas de errores con imágenes periódicas y conteo NaN correcto |
| 3.5 | ENG_QUALITY-12/13/14 | Base sin Torch preservada, CLI mínima funcional, dependencias declaradas/rangos comprobados, constraints e instalación limpia de perfiles base/completo, construcción MatrixMACE compatible, CI realmente probada localmente y estado remoto distinguido |
| 3.6 | DATA_PIPE-21, ENG_QUALITY-24, DATA_PIPE-12/09/08/24/25 | Temporales aislados con ZIP y supervivencia entre stages; setup por stage sin reconstrucción; cierre de pools; reapertura por PID y archivo ZIP después de lectura parental; filtros preservados; subgrafos reindexados; torch_predict conserva batch por nodo |
| 3.9 | ENG_QUALITY-18/26, LOSS_TRAIN-25/29, ENG_QUALITY-20 | Gates efectivos de lint/tipos, tests deterministas/temporales limpios, split configurable y pool sembrado, provenance/versiones, pruebas numéricas de pérdidas/gradientes, equivariancia/supercelda, checkpoints, DDP CPU y simetría periódica/spin/S/d-f |

Todos estos puntos continúan **abiertos** hasta verificar cada requisito; no se cierran
mediante etiquetas de aplazado ni por tener configurada una CI.

## Secuencia de ejecución

Correcciones acotadas sobre las rutas existentes, sin nuevos subsistemas. Mantener
el alcance de cinco frentes; trabajar en sublotes verificables y conservar aquí los
pendientes entre turnos. TDD: reproducir → parche mínimo → regresión → suite completa.
Una revisión independiente final cubrirá el rango completo del goal.

Primer sublote: API de filtros/subgrafos de 3.6, seguido de setup, ZIP y lifecycle
de datos; las otras cuatro áreas permanecen en la tabla y no se excluyen.

### Sublote 3.6.a — filtros y subgrafos

- Responsabilidad: `core/data/processing.py` es la capa compartida de NumPy y Torch;
  todos los callers de `_sanitize_data`, `copy` y `node_types_subgraph` fueron buscados.
- Fallos observados en código: filtros aceptados pero ausentes del diccionario de
  datos/copias; subgrafo rechaza claves de etiquetas aunque su valor sea None;
  conserva índices de nodos viejos y transforma de nuevo posiciones ya en la base.
- Regresión: ambas clases deben conservar máscaras al copiar/convertir; un grafo
  sin etiquetas debe seleccionar nodos 0/2 y reindexar su conexión a 0/1, conservar
  coordenadas en convención SIESTA y permitir selección vacía; un grafo etiquetado
  debe seguir rechazándose explícitamente como documenta esta API.
- Diseño mínimo: guardar filtros opcionales en el contenedor y en `_init_keys`,
  copiar None/ausencias sin convertirlos en arrays object; calcular máscara/índices
  mediante NumPy y construir el subgrafo con `already_basis=True`.
- No redefinir pérdidas ni implementar filtrado de matrices etiquetadas en esta API.
- Prueba enfocada: `pytest -q src/graph2mat/core/data/tests/test_data_filters_and_subgraphs.py`.
- Gate completo: `PYTHONPATH=/tmp/g2m-server-tests.IcPjmO OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/pytest -q --disable-warnings --tb=short`.
- RED comprobado: 6 fallos y 2 pruebas de caracterización pasan. Máscaras ausentes
  en ambos backends, rechazo incorrecto de etiquetas None en NumPy y reconversión
  de arrays Torch como si fueran NumPy. Log: `data-api-red.log` en el directorio de evidencia.
- Parche: reutilizar `copy(cls=BasisMatrixData)` para normalizar arrays del subgrafo;
  conservar máscaras y None en copias, reindexar nodos y no cambiar la base dos veces.
- GREEN: 8 regresiones pasan; suite completa **523 passed, 1 skipped**, 6023 warnings,
  62.26 s (`data-api-full.log`). No se ha cerrado todo 3.6.

### Sublote 3.6.b — setup por stage

- Diseño acotado: inicializar datasets una vez, construir solo splits requeridos
  (fit=train/val, validate=val, test=test, predict=predict, None=todos) y conservar
  objetos ya creados. Si validate necesita el split automático, usar train como
  fuente perezosa y cargar solo su subset de validación en memoria.
- Validar componentes de los splits solicitados y omitir datasets vacíos; no abrir
  pools de entrenamiento durante validate/test/predict.
- Regresiones reales: splits inactivos apuntan a archivos inexistentes y no se leen;
  identidad del dataset se conserva entre stages; validate sin val explícito funciona;
  stage desconocido se rechaza.
- Prueba enfocada: `pytest -q src/graph2mat/tools/lightning/tests/test_datamodule_stages.py`.
- RED: seis fallos (cuatro stages leyendo archivos inactivos, pérdida de identidad
  del dataset, stage desconocido aceptado) y una caracterización pasan. Otra prueba
  reproduce IndexError al inferir componentes de un dataset vacío.
- Parche: selección por stage, caché de datasets/procesador, fuente train perezosa
  cuando validate necesita split automático, inferencia solo de splits solicitados
  no vacíos, apertura del pool solo al pedir train.
- Regresión adicional RED: tras validate→fit, store_in_memory dejaba train como
  Subset perezoso. Corregido moviendo la materialización después del split y
  aplicándola también a datasets cacheados, solo en las etapas solicitadas.
- GREEN: 9 pruebas de etapas; 34 pruebas enfocadas con componentes y filtros.
  Suite completa **532 passed, 1 skipped**, 6023 warnings, 61.61 s
  (`data-stage-full.log`); `stage-transition-red.log` registra el fallo previo.
- Lifecycle/ZIP/temporales siguen abiertos; los otros cuatro frentes permanecen abiertos.
- Turno previo: entrega del prompt y lectura de estado, sin progreso de producto.
  No había bloqueo; esta continuación implementa y verifica la transición de etapas.

## Cierre del goal

Releer el objetivo, el roadmap y los IDs arriba. Para cada requisito registrar
archivo/prueba/comando y salida real; comprobar cobertura, no solo color verde.
Solo después de completar todos, pasar revisión final, corregir hallazgos y ejecutar
todos los gates se podrá marcar el goal completo. Antes, seguir activo.
