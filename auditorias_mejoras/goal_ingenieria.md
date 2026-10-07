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
- La última consulta de estado fue **sin progreso de producto**; get_goal sigue
  active, HEAD=6227edf y no hay bloqueo. Esta continuación retoma los tests RED
  existentes de checkpoints, sin repetir las correcciones comprometidas.
- Corrección de evidencia CPU: el entorno compartido ve CUDA y los Trainer
  originales usaban accelerator=auto. Los gates anteriores prueban resultados
  verdes, **no uso exclusivo de CPU**. Desde este sublote se ejecutan con
  CUDA_VISIBLE_DEVICES='' y el Trainer de integración declara accelerator=cpu,
  devices=1. No se cambió el entorno ni los entrenamientos.

## Cobertura obligatoria y auditorías leídas

| Punto | Auditorías asignadas | Evidencia necesaria para cerrar |
|---|---|---|
| 3.1 | LOSS_TRAIN-06, ENG_QUALITY-23 | Agregación global de pérdidas y diagnósticos, medias/std/máximos, coeficientes/pares/ventana espectral, denominadores válidos, reset por época/split, batch_size explícito y errores fail_ok visibles; tests invariantes al batching |
| 3.4 | PHYS_H-21, ENG_QUALITY-15/16, PHYS_H-22, PHYS_H-20 | Servidor /info,/test,/predict y escritura segura; H/EDM físicamente válidos según su formato; extrapolación sin mutar matrices ni usar índices de relleno, periodicidad SOAP/dependencia explícita, API HistoryCopy compatible; mapas de errores con imágenes periódicas y conteo NaN correcto |
| 3.5 | ENG_QUALITY-12/13/14 | Base sin Torch preservada, CLI mínima funcional, dependencias declaradas/rangos comprobados, constraints e instalación limpia de perfiles base/completo, construcción MatrixMACE compatible, CI realmente probada localmente y estado remoto distinguido |
| 3.6 | DATA_PIPE-21, ENG_QUALITY-24, DATA_PIPE-12/09/08/24/25 | Temporales aislados con ZIP y supervivencia entre stages; setup por stage sin reconstrucción; cierre de pools; reapertura por PID y archivo ZIP después de lectura parental; filtros preservados; subgrafos reindexados; torch_predict conserva batch por nodo |
| 3.9 | ENG_QUALITY-18/26, LOSS_TRAIN-25/29, ENG_QUALITY-20 | Gates efectivos de lint/tipos, tests deterministas/temporales limpios, split configurable y pool sembrado, provenance/versiones, pruebas numéricas de pérdidas/gradientes, equivariancia/supercelda, checkpoints, DDP CPU y simetría periódica/spin/S/d-f |

Cada punto requiere verificar todos sus requisitos; no se cierra mediante etiquetas
de aplazado ni por tener configurada una CI. La revisión final cubrirá los cinco.

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

### Sublote 3.6.c — ZIP, temporales y lifecycle

- Diseño: lectores ZIP propios por PID y archivo, sin mutar archivos prestados.
  Guardar la ruta reabierta devuelta por el dataset; carga y fichero espectral usan
  la API real de zipfile.Path, no un Path ficticio dentro del archivo.
- RED: tres fallos de ZIP (`zip-red.log`), con/sin lectura previa del padre y dos
  archivos; fallo real de NetCharge dentro de ZIP. GREEN: tres regresiones, con
  cierre de descriptores propios y lectura posterior de los archivos prestados.
- Temporales: mkdtemp por instancia/proceso, archivo ZIP copiado sin extraer,
  subdirectorio interno conservado; stage teardown retiene los datos. Cierre final
  explícito y atexit limpian solo su carpeta. Cada rango DDP tiene copia propia:
  evita carreras de borrado, a costa de I/O/espacio duplicados por rango.
- RED: cuatro fallos de aislamiento, borrado entre stages y copytree de ZIP
  (`temp-roots-red.log`). Dos regresiones adicionales reprodujeron un descriptor
  de origen que seguía abierto y datasets heredados que seguían apuntando al
  temporal parental (`temp-ownership-red.log`). Parche en la capa de lifecycle:
  cerrar fuentes/cache, permitir reutilizar el módulo tras cierre final e invalidar
  datasets heredados cuando cambia el PID. Las ocho regresiones de temporales
  pasan, incluidas salidas de procesos reales que ejecutan atexit.
- Pool: stop Event + lectura de cola con timeout, parar primero el hilo manteniendo
  vivo el emisor de tensores, terminar/join del lector, cerrar cola/Manager.
  Muestreo sembrado desde NumPy global o seed explícita; validar tamaño positivo
  y dataset no vacío antes de crear recursos. close es idempotente.
- RED: ausencia de close, muestreo no reproducible y pool vacío de tamaño cero
  (`pool-red.log`, `pool-invalid-red.log`). Regresión del DataModule confirma fuga
  tras teardown (`pool-teardown-red.log`). Ahora fit/None cierran el pool, conservan
  parent_data y permiten un nuevo fit; validate no abre otro pool. Cinco pruebas
  pasan con tensores y datos reales. La sustitución asíncrona depende del scheduling:
  no se promete entrenamiento idéntico solo por sembrar el pool.
- torch_predict: regresión de pooling real sobre tres nodos exige batch por nodo
  y ptr=[0,3], y reconstruye la matriz diagonal esperada de valor 6.
- Documentación pública: docs/data_loading.rst, enlazada desde docs/index.rst.
- GREEN enfocado: 48 pruebas, 60 warnings, 1.25 s (`lifecycle-focused.log`);
  temporales con exit: ocho pasan (`temp-exit-green.log`).
- Gate completo: **549 passed, 1 skipped**, 6031 warnings, 73.30 s
  (`lifecycle-full.log`). Incremento frente a la línea base: 34 regresiones nuevas.
- 3.6 implementado/verificado localmente; pendiente la revisión transversal final.
  Ningún otro frente queda cerrado por este sublote.
- Reanudación tras usageLimited: get_goal confirma active. El último turno de
  estado no implementó código; este turno reproduce y corrige teardown y ownership.

### Sublote 3.9.a — split reproducible configurable

- Añadir split_seed=32 y val_fraction=0.1 al final de los parámetros existentes,
  preservando llamadas posicionales/checkpoints y el split por defecto.
- Semilla válida de RandomState y fracción en [0,1); conservar al menos una muestra
  de entrenamiento cuando el dataset no esté vacío. Configuraciones explícitas de
  val_runs conservan prioridad; guardar los parámetros con save_hyperparameters.
- Regresiones de partición disjunta/completa, repetición, cambio de semilla/fracción,
  defaults históricos y casos cero, un solo dato y fracciones inválidas.
- Después: provenance, gates de calidad y pruebas científicas/DDP; siguen abiertos.
- RED: 13 fallos y una caracterización pasan por kwargs ausentes (`split-red.log`).
  Implementación de split_seed/val_fraction, guardas antes de leer archivos y
  fracción limitada para retener train. GREEN inicial: 31 pruebas con stages y
  temporales (`split-green.log`). Se añadió caracterización de val_runs=[]:
  conserva todo train aunque val_fraction=0.99, sin crear un split automático.

### Sublote 3.9.b — provenance aditiva de checkpoints

- Usar on_save_checkpoint existente, sin tocar state_dict ni necesitar nuevos campos
  en on_load_checkpoint. Versiones mediante importlib.metadata (sin imports nuevos
  del stack ML); Python y Git del directorio del módulo, no del cwd del usuario.
- Registrar commit y dirty de archivos tracked; None explícito si falta Git o el
  paquete instalado no tiene checkout. Dependencias opcionales ausentes → None.
- Reproducir falta de provenance; verificar que un cwd externo no cambia el commit,
  fallback sin checkout y carga de checkpoints antiguos sin ese nuevo campo.
- RED reproducido en esta continuación, CPU forzada: **5 failed, 2 passed**,
  223 warnings, 2.20 s (`provenance-resume-red.log`). Todos los fallos se deben
  al campo ausente; el roundtrip real de MACE de formato antiguo ya pasa.
- Parche en el hook existente: importlib.metadata para seis distribuciones, Python,
  git ls-files para comprobar que el módulo está tracked antes de registrar HEAD
  y dirty. Una wheel dentro de otro repo no hereda un hash ajeno; Git ausente,
  sin checkout o fallido → None, sin impedir guardar. Sin cambios a state_dict,
  strict_loading ni al hook de carga.
- Fixture corregida antes del parche: hidden_irreps con multiplicidades distintas
  (2 escalares/1 vector) viola el número uniforme de canales de symmetric_contraction
  en MACE instalado y causaba un error TorchScript de reshape. La fixture soportada
  usa 2 escalares/2 vectores; no se parcheó el modelo para esconder el fallo.
- Primer comando enfocado tenía un nombre de archivo incorrecto
  (test_datamodule_temporary_data.py), exit 4; corregido al archivo real
  test_datamodule_temporary_roots.py, sin cambio de producto por este error.
- GREEN enfocado en CPU: **44 passed**, 274 warnings, 8.29 s
  (`reproducibility-focused.log`). Incluye entrenamiento de un batch, guardar,
  cargar checkpoints reales antiguos/nuevos y predicciones idénticas (rtol=atol=0).
  Las dos caracterizaciones posteriores cubren Git no instalado y val_runs=[].
- Documentado seed_everything, trainer.deterministic y su coste/límites, prioridad
  de val_runs y metadatos de checkpoint en docs/data_loading.rst.
- Gate completo **CPU forzada**: **571 passed, 2 skipped**, 6216 warnings,
  63.27 s (`reproducibility-full-cpu.log`), exit 0. Comando: el gate completo de
  arriba con CUDA_VISIBLE_DEVICES='' antes de PYTHONPATH. Son 23 regresiones
  añadidas sobre las 549 previas, menos una prueba GPU que ahora se omite porque
  CUDA está oculta. La otra omisión era la fixture local de H ya ausente en la
  línea base. No hay fallo nuevo silenciado. Los warnings existentes se conservan
  como evidencia, no se afirma una suite libre de warnings.
- Los cuatro frentes todavía abiertos y la revisión final no quedan cerrados
  por este sublote. Próximo: fixtures válidas/deterministas y pruebas científicas.
- Commit local: **7fe54b5**, split y provenance. Omisiones confirmadas con
  pytest -rs sobre los dos nodos (`cpu-skips-confirmed.log`), exit 0. Un intento
  anterior con -k torch_compile seleccionaba cero tests (exit 5); no se usó como
  evidencia de la prueba GPU.

### Sublote 3.9.c — fixtures deterministas, válidas y con comprobaciones de valor

- Reutilizar los tests existentes: default_rng(0) para matrices/coordenadas,
  semilla NumPy/Torch por test para legado, seed_everything en integración.
  No alterar algoritmos de producción ni semillas explícitas ya repetibles.
- test_processing genera la dimensión orbital correcta (1 o 6), matrices
  simétricas solo cuando se declara esa propiedad y reconstruye valores originales
  con matrix_from_data para ambas políticas, convenciones y clases de configuración.
  Añade 48 casos; no se llama éxito solo por construir un objeto.
- test_lightning usa matriz simétrica con soporte en el grafo, loader_threads=0,
  CPU explícita y tmp_path en lugar de dos NamedTemporaryFile(delete=False).
  Comprueba ausencia de escritura antes de test, CSV real y matriz escrita.
- test_basis: tamaño, capas, radios, base vacía, tipo de átomo real y conversión
  desde p de sisl; se conserva la prueba de paridad y capas ya corregida.
- GREEN enfocado: **127 passed**, 355 warnings, 3.90 s (`fixtures-focused.log`).
  Gate completo CPU: **619 passed, 2 skipped**, 6206 warnings, 62.20 s
  (`fixtures-full-cpu.log`), exit 0. Sin cambios científicos de producción.
- Persisten las omisiones justificadas y los warnings. Próximo: reactivar las
  pruebas comentadas de equivariancia/supercelda con MatrixMACE real, float64 y
  rotaciones aleatorias sembradas, y completar las demás pruebas/gates del goal.

## Cierre del goal

Releer el objetivo, el roadmap y los IDs arriba. Para cada requisito registrar
archivo/prueba/comando y salida real; comprobar cobertura, no solo color verde.
Solo después de completar todos, pasar revisión final, corregir hallazgos y ejecutar
todos los gates se podrá marcar el goal completo. Antes, seguir activo.
