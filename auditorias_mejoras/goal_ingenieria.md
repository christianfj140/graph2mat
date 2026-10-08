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
- Commit local de fixtures: **d35f1a6**.

### Sublote 3.9.d — MACE completo y DDP CPU

- Sustituido test_models.py (antes enteramente comentado y con imports/API viejos)
  por las pruebas reales equivalentes: dos bases (s/p y DZP desigual con d),
  bimolécula/cuadrado/cadena periódica, float64 restaurando el dtype al salir.
- 64 rotaciones propias aleatorias sembradas por caso: **384 rotaciones** del
  pipeline geometría→datos→MatrixMACE→matriz, comparadas con D·H·Dᵀ por imagen.
  Se exige salida finita/no nula y tolerancia 1e-8 relativa/1e-10 absoluta.
- La predicción debe conmutar con tile(2) en los tres ejes, incluyendo las imágenes
  de la cadena. No se cambió producción. Las 12 pruebas pasan, 337 warnings,
  5.95 s (`mace-properties-first.log`). No prueba aún un oráculo independiente
  de armónicos d/f de SIESTA; ese requisito de ENG_QUALITY-20 sigue abierto.
- Nuevo smoke DDP real aislado: dos rangos CPU, Gloo, dos pasos, pesos actualizados
  y loss finita, con todos los tipos presentes y con especies ausentes del batch.
- Primer harness con fork: **2 failed** por timeout de 90 s, 180.89 s total
  (`ddp-red.log`); procesos hijos en futex. La primera pareja quedó huérfana al
  expirar el padre y se cerró solo esa pareja de PIDs verificados; la segunda
  ya no existía al finalizar. Ningún entrenamiento del usuario fue tocado.
- Ruling: usar lanzamiento normal de scripts (`popen`) documentado por Lightning
  en lugar de fork — evita heredar el estado de hilos al bifurcar el modelo
  ya construido — no acredita el modo notebook/fork, fuera del smoke pedido.
  El harness aislado ahora limpia su propio process group si expira.
- RED con scripts: **2 failed**, 13.47 s (`ddp-script-red.log`), fallo DDP en el
  segundo paso por parámetros desconectados. Callback de cobertura real después
  del backward reproduce los dos fallos (`ddp-gradient-red.log`, 13.43 s) y nombra
  exactamente las tres capas de energía de MACE y los bloques de especies/pares
  ausentes. No es un fallo de Gloo ni del launcher.
- Ruling: conectar una dependencia de valor cero por tensor trainable **solo
  con proceso distribuido inicializado**, en training_step compartido — no cambia
  la función de pérdida, los checkpoints, la ruta single-process ni los grupos
  existentes del optimizador; produce gradientes cero de módulos ausentes para
  que DDP reduzca todos los parámetros. Coste: materializar/comunicar esos ceros.
  Evita cambiar APIs de bloques o confiar en que todo operador acepte batch vacío.
- Parche mínimo en model.py, con comprobación de gradientes de todos los parámetros
  y entrenamiento real de dos pasos. GREEN enfocado: **28 passed**, 548 warnings,
  20.97 s (`mace-ddp-focused.log`), con modelos/roundtrip/checkpoints/stages.
- Comprobación adicional independiente: dos rangos con samples idénticos deben
  producir los mismos pesos/buffers que dos pasos single-process sin DDP, con
  rtol=1e-6/atol=1e-7 sobre todo el state_dict. Pasa, igual que el caso con especies
  ausentes: **2 passed**, 52 warnings, 17.00 s (`ddp-serial-proof.log`). Los workers
  guardan sus logs completos en el tmp_path del test, aunque no fallen.
- Gate completo CPU: **6 failed, 627 passed, 2 skipped**, 6392 warnings,
  82.44 s (`mace-ddp-full-cpu.log`), exit 1. Los seis fallos son las nuevas
  pruebas de equivariancia: pasan en proceso fresco, fallan después de otros tests
  que llenan la caché RTP con float32. DDP y superceldas pasan. No se han relajado
  tolerancias ni ocultado fallos. DDP métrico global sigue siendo requisito
  separado de 3.1, no cerrado por el smoke.
- La consulta de estado anterior no cambió producto (no progress). Esta
  continuación vuelve a verificar HEAD=d35f1a6 y reproduce la pérdida de precisión
  al cambiar el dtype tras calentar la caché; no repite los cuatro commits verdes.
- Hipótesis trazada: LazyDict conserva el change_of_basis creado con el dtype de
  la primera llamada; ReducedTensorProducts lo copia a float64 sin recuperar los
  bits perdidos. Dos regresiones del consumidor real comparan float32→float64→
  float32 con el RTP de e3nn, para ij y ij=ji.
- RED confirmado: **2 failed**, 99 deselected, 96 warnings, 0.35 s
  (`rtp-dtype-red.log`); diferencias de hasta 2.97e-8 respecto a e3nn float64.
  El primer sondeo de ij=ji con l=1 ya pasaba tras normalizar; el caso l=2
  reproduce la pérdida de precisión también en esa rama, sin cambiar tolerancias.
- Ruling: regenerar una entrada cuando su dtype difiere del default actual —
  conserva las claves triples y la API de Mapping, no cambia el dtype global ni
  obliga a todas las llamadas a usar float64. Coste: recomputación al alternar
  dtypes, aceptada en vez de añadir otra caché por dtype. Es la corrección mínima
  de la dependencia 3.7 necesaria para verificar la equivariancia de 3.9; no se
  amplía este goal a optimizaciones de RTP.
- GREEN ordenado RTP→MACE→DDP: **115 passed**, 3989 warnings, 47.82 s
  (`rtp-mace-ddp-green.log`); reproduce la caché calentada por pruebas anteriores.
- Gate completo CPU: **635 passed, 2 skipped**, 6579 warnings, 84.16 s
  (`mace-ddp-rtp-full-cpu.log`), exit 0. Las seis pruebas de equivariancia antes
  fallidas ahora pasan con las mismas tolerancias. Son 16 pruebas más que las 619
  del gate anterior: 12 MACE, dos DDP y dos de dtype de RTP. Las omisiones siguen
  siendo la fixture local ausente y la prueba GPU; no se ha omitido ningún fallo.
- El resto de 3.9, las métricas pendientes de 3.1, servidor/instalación y la revisión
  independiente final permanecen abiertos. Próximo: diagnósticos y denominadores.
- Commit local del bloque MACE/DDP y precisión: **aaa5eb8**.

### Sublote 3.1.a — denominadores de objetivos básicos y momentos de errores

- Releídos LOSS_TRAIN-06 y ENG_QUALITY-23 y las rutas de train/val/test. El
  batch_size explícito por estructuras no convierte una media de elementos en la
  media global correcta cuando cambian los tamaños. Las desviaciones de cada
  batch tampoco se pueden promediar para obtener la desviación de toda la época.
- Regresiones de valores literales para los MSE/MAE por bloque, elemento y rama,
  edge_weight=2, hp_metric y train_loss; repetir particiones y épocas para probar
  reset. Mantener el valor y gradiente de la pérdida de cada paso sin cambios.
- Ruling: usar estadísticos suficientes de errores válidos en TorchMetrics —
  sumas y conteos por rama; medias/std/máximos independientes de la partición.
  Mantener la desviación muestral de Torch cuando hay al menos dos elementos,
  definir cero con menos elementos para logging. Coste: acumuladores float64 y
  reducciones distribuidas; no cambia la pérdida científica de entrenamiento.
- Siguen abiertos los objetivos/diagnósticos especiales de coeficientes, pares,
  espectral, dominios/Huber, tratamiento de NaN/vacíos de todas las pérdidas,
  fail_ok y la prueba distribuida de estos valores. Este sublote no cierra 3.1.
- RED: **19 failed, 4 passed**, 140 warnings, 0.91 s
  (`epoch-objectives-red.log`). La primera fixture hacía coincidir algunas medias
  por cancelación; se cambió a cinco etiquetas de nodo y cuatro de arista para que
  todos los nuevos casos de objetivos detecten el denominador incorrecto.
- Parche compartido de logging, sin cambiar funciones de pérdida ni el loss
  devuelto por training_step: acumular sumas/conteos por rama antes de aplicar
  el objetivo básico (seis clases MSE/MAE), y momentos para mean/std/max. Mantener
  nombres train_loss_step/epoch, val_loss, test_loss y hp_metric. Objetivos especiales
  conservan por ahora su agregación por estructuras, con sync_dist explícito.
- GREEN con epochs/stages y DDP existente: **31 passed**, 140 warnings, 17.65 s
  (`epoch-objectives-green.log`). TorchMetrics reinicia entre épocas/splits como
  antes; los estados nuevos no son persistentes ni alteran state_dict/checkpoints.
- Prueba distribuida adicional con scripts reales: dos rangos CPU/Gloo reciben
  muestras de tamaños distintos, sin padding del sampler. Ambos deben producir
  los mismos valores globales literales de loss/hp_metric, RMSE, mean/std/max y
  spin. **2 passed**, 2 deselected, 52 warnings, 10.64 s
  (`epoch-metrics-ddp.log`). Esto no prueba los diagnósticos especiales pendientes.

### Sublote 3.1.b — errores fail_ok visibles

- El callback SamplewiseMetricsLogger captura fallos por métrica y escribe NaN en
  el CSV por defecto, pero no informa del fallo. Añadir aviso la primera vez por
  clase de métrica/callback, manteniendo CSV y re-raise cuando fail_ok=False.
- Dos casos ejercitan el callback real y sus filas CSV con una métrica que falla
  deliberadamente: tres batches solo avisan una vez; la política estricta sigue
  levantando la excepción original sin escribir filas falsas.
- RED: **1 failed, 1 passed**, 5 deselected, 52 warnings, 0.60 s
  (`fail-ok-red.log`); cero avisos con fail_ok=True, política estricta ya correcta.
- Parche en el único except que oculta fallos: RuntimeWarning con clase, tipo y
  mensaje de excepción; set de nombres por callback para no avisar cada batch.
  No se cambian las excepciones de la política estricta ni las filas CSV.
- GREEN conjunto: **36 passed**, 140 warnings, 1.12 s
  (`epoch-and-callback-green.log`), con epochs, stages, escritura y warnings.
- Documentados objetivos cubiertos, desviación muestral, política fail_ok y el
  posible padding del sampler DDP (los tests distribuidos usan divisibilidad exacta).
- Gate CPU completo: **653 passed, 2 skipped**, 6638 warnings, 94.93 s
  (`epoch-objectives-full-cpu.log`), exit 0. Incremento: 14 casos nuevos de
  objetivos, dos comprobaciones DDP de estadísticas y dos de fail_ok. Las pruebas
  de mean/std/loss/reset también fortalecen las existentes, sin omitir fallos.
- Los apartados especiales de 3.1 y los demás frentes/revisión final permanecen
  abiertos. En particular, las pérdidas subyacentes aún necesitan tratamiento
  completo de nodos NaN/vacíos y de umbrales sin elementos; no se declara cerrado
  ese requisito porque el logging de std con una sola muestra ya sea finito.

### Sublote 3.1.c / 3.9 — pérdidas con NaN y ramas vacías

- Turno anterior: progreso verificado, commits aaa5eb8 y **0f3529f**, suite CPU
  653 passed/2 skipped. HEAD y árbol tracked limpios confirmados; no se repite ese trabajo.
- Traza: get_predictions_error filtra NaN solo en aristas; las pérdidas relativas
  además filtran la referencia de aristas por escalar, incompatible con canales.
  mean/std/max vacíos y std Torch de un solo elemento contaminan pérdidas y stats;
  elementwise multiplica un NaN por un conteo cero. Los umbrales dividen por filas,
  no por elementos cuando hay canales, y promedian conjuntos vacíos.
- Regresiones con resultados literales en NumPy/Torch y una/dos componentes:
  15 pérdidas básicas, filas parcialmente NaN y cuatro políticas de rama vacía.
  Gradientes finitos, cero en etiquetas ausentes; node_mse/edge_mse conservan su
  exclusión intencional de la otra rama. Este lote no redefine esas pérdidas.
- Ruling: conservar la máscara por fila (cualquier componente NaN excluye la fila)
  ya usada en aristas, aplicarla también a nodos; las referencias relativas usan
  la misma máscara. Reducciones vacías contribuyen cero conectado al grafo; std
  de menos de dos elementos es cero, conservando std existente con más datos.
  Coste: se descartan también los canales válidos de una fila parcialmente ausente,
  como ya declaraba la política de etiquetas multicomponente.
- RED básico: **180 failed**, 507 warnings, 3.02 s (`valid-labels-red.log`).
  Parche: una máscara compartida por fila, referencias relativas alineadas y
  mean/std/max seguros para vacío/singleton; conteo escalar en elementwise/umbrales.
  La actualización mecánica se hizo con apply_patch. Un chequeo del script detectó
  nombres solapados antes de escribir: se ordenaron los patrones por longitud;
  no se aplicó el reemplazo inválido. GREEN básico: **180 passed**, 48 warnings,
  0.15 s (`valid-labels-green.log`), con valores y gradientes reales.
- RED por época: **10 failed, 3 passed**, 20 deselected, 91 warnings, 0.77 s
  (`node-nan-epochs-red.log`). El conteo RMSE de nodos seguía incluyendo etiquetas
  filtradas; per_structure calculaba su propia resta sin máscara de nodos.
- Parche en los dos consumidores: _rmse_count usa la máscara común en ambas ramas
  y per_structure alinea los IDs de estructura con ella, conservando peso igual
  por estructura. Se unifica pred−ref allí sin alterar el MSE.
- GREEN enfocado: **325 passed**, 153 warnings, 3.43 s
  (`valid-labels-focused.log`), con pérdidas normales/composite/Huber, gradientes,
  epochs y nodos NaN. Gate CPU completo: **843 passed, 2 skipped**, 6648 warnings,
  96.47 s (`valid-labels-full-cpu.log`), exit 0. Este gate precede las pruebas
  nuevas de bloques físicos y pesos fraccionarios; no prueba el árbol posterior.
- Los bloques físicos con etiquetas parcialmente ausentes y los objetivos
  coeficiente/gauge/parejas aún necesitan comprobaciones propias; no se consideran
  cubiertos por las 15 pérdidas básicas ni se cierra 3.1/3.9 con este lote.

### Sublote 3.1.d / 3.9 — máscaras de bloques físicos y pesos fraccionarios

- Commit local del lote NaN/bloques: **35585f7**, suite completa 875/2.
- Consulta de estado anterior: sin cambios de producto; goal active, HEAD 0f3529f,
  lote NaN sin commit. No hay bloqueo ni pruebas previas vivas.
- Las 30 pruebas iniciales de Huber fallaban en la preparación del fixture:
  una matriz densa cero perdía sus bloques onsite al pasar a sparse; cambiarla
  por identidad conservaba onsite pero perdía las aristas. Esos KeyError no
  demuestran fallos de Huber. Se aisló su prueba usando una matriz de unos con
  todos los bloques y targets NaN explícitos. Se retiró el parche prematuro antes
  de repetir RED; no se contabiliza el intento anterior como prueba del arreglo.
- RED real: **28 failed, 4 passed**, 213 deselected, 74 warnings, 1.15 s
  (`masked-blocks-real-red.log`). Falla la selección de representantes usando
  longitud ya filtrada, los punteros físicos no coinciden con ella y las medias
  vacías producen NaN. Dos casos composite sin ramas vacías y dos pruebas de
  pesos fraccionarios ya pasan; se preservan como caracterización.
- Parche: seleccionar las aristas con la longitud de referencia completa,
  comprimir límites con el prefijo de la máscara de filas, omitir bloques sin
  target y conservar las etiquetas de tipo de los bloques restantes. Rama vacía
  aporta cero conectado al grafo. Se conserva la validación de longitudes, los
  pesos y las fórmulas para datos válidos; solo helpers privados reciben refs.
- Regresión adicional ya RED en masked-blocks-red.log: dos pérdidas globales
  elementwise con todos los nodos NaN y edge_weight=0.125 daban 2.5/12.5 en vez
  de 10/25. La reducción por época limitaba erróneamente el denominador a 1.
  Ahora protege solo denominadores exactamente cero; valores y gradientes del
  paso no cambian. Los dos casos pasan con distintos batches y resets.
- Un comando enfocado nombró test_hamiltonian_losses.py, que no existe: exit 4,
  sin ejecución de pruebas. Se localizó test_hamiltonian_training_losses.py y
  se ejecutó realmente: **23 passed**, 47 warnings, 0.09 s.
- GREEN de pérdidas básicas/bloques/epochs: **334 passed**, 159 warnings,
  3.41 s (`masked-blocks-real-focused.log`). Gate CPU completo: **875 passed,
  2 skipped**, 6654 warnings, 96.10 s (`masked-blocks-full-cpu.log`), exit 0.
  Comando: gate completo anterior con CUDA_VISIBLE_DEVICES='' y el mismo overlay
  temporal del servidor, OMP/OPENBLAS/MKL=2. Subapartado básico/bloques verificado;
  objetivos y diagnósticos especiales, servidor, CLI/gates y revisión final siguen
  abiertos. No se cierran 3.1 ni 3.9 por este resultado.
- Pendiente adicional 3.9: investigar y probar el pipeline de matrices con
  bloques completamente cero/ausentes. No rellenar DM con cero por conveniencia:
  csr_to_block_dict declara fill_value NaN para DM/EDM y cero para H. Deben
  preservarse ambos contratos, incluidos componentes y periodicidad.

### Sublote 3.9.e — bloques sparse completamente ausentes

- Commit local **6cf523f**, gate completo 931/2 y wheel base verificada.
- Traza: csr_to_block_dict conserva fill_value dentro de los bloques existentes,
  pero no crea bloques sin ninguna entrada sparse. BasisMatrix.to_flat_nodes_and_edges
  los indexaba obligatoriamente. Una matriz completamente cero o solo diagonal
  lanzaba KeyError; DM/EDM perdían además la política NaN al faltar el bloque entero.
- RED: **56 failed**, 63 warnings, 0.72 s (`zero-blocks-red.log`), KeyError real
  en onsite/arista. Ocho roundtrips NumPy/CSR con bloques desiguales, y 48 casos
  H/DM/EDM: una/dos componentes, periódico/no periódico, una entrada/todo ausente,
  almacenamiento simétrico/no simétrico. Valores esperados literales; H además
  reconstruye los canales completos por separado, incluidas imágenes periódicas.
- Parche: retener fill_value y número de componentes seleccionados como campos
  privados init=False del contenedor; no cambian sus argumentos posicionales.
  Al aplanar, devolver un bloque de relleno solo si no existe, sin insertarlo ni
  densificar la entrada. Bloques existentes no cambian; dtype/componentes de
  un contenedor antiguo se infieren del primer bloque presente, y el núcleo no
  adquiere dependencia ML. H conserva 0; DM/EDM conserva NaN.
- Ruling: la primera salida post-parche mostró **8 failed, 48 passed** porque
  el fixture pedía serializar H spin con raw_components. Se mantuvo el rechazo
  deliberado del escritor y se corrigió la política de la prueba a spin_h_only;
  no se permite escribir una H física ambigua para obtener verde. Coste si esta
  política fuera errónea: se perdería distinción spin/overlap; las regresiones
  existentes siguen comprobando ese rechazo.
- GREEN cercano: **205 passed**, 205 warnings, 1.42 s (`zero-blocks-focused.log`):
  pruebas nuevas y processing/spin/sparse/matrices. Gate completo CPU: **931 passed,
  2 skipped**, 6670 warnings, 96.23 s (`zero-blocks-full-cpu.log`), exit 0.
- El chequeo directo vía PYTHONPATH del source sin compilar falló por ausencia de
  Cython en el venv base (zero-blocks-core-only.log), no por Torch ni por bloques.
  No se usa como prueba de instalación. Se construyó la wheel actual con build
  isolation: exit 0, zero-blocks-wheel-build.log. Wheel cp312, sha256
  1e455dd8511f41c817261d12bb8059c3f586bdde3fe4071b8688c49dc7d6f6be.
  Instalada con sus dependencias base en un **nuevo** venv temporal
  /tmp/g2m-zero-blocks-wheel.35HRaK/venv: exit 0, zero-blocks-base-install.log.
  Núcleo instalado verificado sin Torch ni Cython runtime: extensión .so real,
  roundtrip NumPy de matriz cero y DM spin vacía con NaN y ejes (2,2)/(1,2);
  exit 0, zero-blocks-installed-core.log. El venv de la línea base permanece intacto.
- Diagnóstico 3.5 en esa instalación base limpia: graph2mat --help sigue fallando
  (exit 1, base-cli-help-red.log). No se presenta el núcleo verde como CLI verde;
  el siguiente lote de instalación debe resolver su cadena real de imports.
- Este sublote no completa la cobertura de pérdidas ni los oráculos independientes
  spin no collinear/S/d-f; 3.9 y los demás frentes/revisión final siguen abiertos.

### Sublote 3.1.e — Huber básico global por época

- La interrupción de cuota ocurrió después de escribir las pruebas RED, antes
  del parche. Reanudación solicitada por el usuario; HEAD 6cf523f y únicamente
  las dos pruebas de epochs/DDP modificadas, sin procesos pytest vivos.
  get_goal aún informa usageLimited; no se altera su estado mediante update_goal.
- RED real de train/val/test: **17 failed, 1 passed**, 34 deselected, 108 warnings,
  0.89 s (huber-epochs-red.log). Huber de validación daba 4 frente al global 4.3;
  train_loss_epoch daba 1.5 frente a 1.0. Valores literales con beta/delta,
  uno/dos componentes, nodos totalmente ausentes y batches 1,2,1/reset.
- RED de dos rangos CPU/Gloo: **2 failed, 1 passed**, 3 deselected, 52 warnings,
  15.78 s (huber-ddp-red.log). Huber scalar/spin daba 4/6.5 frente a 4.3/6.95.
  El tercer caso MSE existente pasó: no se omite ni se redefine su prueba.
- Parche: reutilizar los acumuladores existentes de objetivo y momentos sobre
  los términos Smooth L1. Pérdida Huber/su alias y medias por rama reducen sumas
  y conteos válidos por época/rango; beta/delta y los gradientes del paso no
  cambian. No se aplica esta política a pérdidas Huber físicas/composite, que
  tienen denominadores diferentes y siguen pendientes.
- GREEN enfocado: **285 passed**, 211 warnings, 2.22 s
  (huber-epochs-focused.log). DDP scalar/spin MSE/Huber: **4 passed**,
  52 warnings, 19.90 s (huber-ddp-green.log); todos los rangos comprueban pérdidas,
  medias por rama y estadísticas globales contra valores literales, sin padding.
- Gate completo CPU: **950 passed, 2 skipped**, 6722 warnings, 104.30 s,
  exit 0 (huber-epochs-full-cpu.log). Los objetivos/denominadores especiales de
  3.1 y los otros frentes conservan todo su alcance; no se marca completo el goal.

### Sublote 3.5.a — CLI base y extras de comandos

- Reanudación solicitada tras agotar cuota; get_goal volvió a informar **active**
  en esta continuación. HEAD inicial 234102b; seis archivos del lote CLI propios
  sin commit. Sin cambios en entrenamientos, autorun ni entornos compartidos.
- RED instalado real previo: graph2mat --help fallaba sin requests
  (base-cli-help-red.log). RED de aislamiento Typer: la importación de Graph2Mat
  cambiaba la ayuda de una aplicación ajena (base-cli-markdown-red.log).
- Parche mínimo: importar HTTPError al ejecutar predict, YAML al parsear opciones
  y Jinja al generar plantillas. Mensajes de extra requerido para YAML/Jinja;
  dicts existentes no necesitan YAML. Requests declarado en server/tools, Jinja
  en analysis y YAML en siesta. Se elimina el parche global de API privada Typer;
  su rich_markup_mode nativo conserva la ayuda Markdown de los comandos.
- Las primeras simulaciones daban doce fallos de fixture, no de CLI: bloquear
  pandas impedía importar xarray/Sisl antes del comando. pandas es dependencia
  transitiva del perfil base sisl[viz], por lo que no se simula su ausencia.
  Se deshizo su import perezoso innecesario para conservar anotaciones públicas.
- RED aislado auténtico para YAML/Jinja: **2 failed, 18 deselected**, 3.83 s
  (base-cli-real-consumers-red.log), al reinstaurar temporalmente sus imports
  originales. Restaurado el parche solo después de comprobar ambos fallos.
- GREEN enfocado inicial: **20 passed**, 47 warnings, 16.92 s
  (base-cli-final-focused.log). Doce ayudas, parser YAML real, generación CLI
  de Lua con opciones literales, ausencia de extras y aislamiento de otras apps.
- Wheel actual construida con build isolation, exit 0 (cli-wheel-build.log),
  sha256 868c2bc7835bea7a1596326a6879ba84c8124662c8894aaaff6d780639e1a752.
  Instalada con dependencias base y pytest en nuevo venv
  /tmp/g2m-cli-wheel.XcBn5L/venv (cli-base-install.log), sin alterar los anteriores.
  Compilado Cython real y módulo desde site-packages; Torch, requests, YAML,
  Jinja, FastAPI, uvicorn y Cython runtime realmente ausentes. graph2mat --help
  exit 0; pip check exit 0. Logs cli-installed-base-profile.log y cli-installed-help.log.
- El primer gate de pruebas instalado falló antes de Graph2Mat porque el script
  suponía importlib.util cargado por importlib.abc. El entorno compartido ocultaba
  ese error de fixture. Añadido import explícito de util, sin cambio de producto.
  Gate instalado final: **16 passed, 4 deselected**, 8.28 s
  (cli-installed-base-tests-final.log). No es una simulación de instalación.
- Suite CPU completa del producto: **967 passed, 2 skipped**, 6722 warnings,
  116.38 s (base-cli-full-cpu.log), exit 0; precede únicamente los dos imports
  explícitos añadidos a los scripts de prueba. La CI base incluye el ejecutable
  graph2mat --help y las regresiones base; no se afirma ejecución remota.
  Scripts definitivos en el entorno de referencia: **20 passed**, 47 warnings,
  16.84 s (base-cli-final-test-imports.log), exit 0.
- 3.5 sigue abierto: cliente independiente del servidor/ML, rangos/versiones,
  constraints reproducibles y perfil completo limpio. El resto del goal y la
  revisión final mantienen todo su alcance original.

### Sublote 3.4.b / 3.5.b / 3.9.f — mapas periódicos y cliente independiente

- Commit CLI anterior: **024fecb**. Este lote continúa sobre él, sin revisión
  delegada por tarea; la revisión independiente sigue reservada al goal completo.
- Cliente: RED **1 failed, 20 deselected**, 0.53 s
  (client-without-server-red.log), al importar Torch desde server_app para usar
  ServerClient. Las funciones públicas de servidor se resuelven perezosamente
  con __getattr__; el cliente conserva la API y no carga el stack de modelos.
  GREEN **30 passed**, 53 warnings, 16.04 s (client-without-server-green.log),
  incluidos nueve tests HTTP existentes y consulta real a un servidor HTTP local
  con ausencia simulada de ML/FastAPI/YAML/Jinja. Sin mocks de Requests.
- Mapas: la primera fixture rectangular se declaró erróneamente nsc=1; sus 23
  fallos precedían el callback y no prueban el bug. Se construyó una BasisMatrix
  real con nsc=[3,1,1] y los tres bloques, sin relajar validaciones de periodicidad.
- RED auténtico **21 failed, 2 passed**, 52 warnings, 0.76 s
  (periodic-plot-real-red.log). MAE resultaba [[8,0,0]] en vez de [[2,3,3]]:
  las imágenes se sumaban en la columna central. NaN contaminaba nodos o diluía
  aristas; dos componentes DM fallaban al intentar serializar errores como DM;
  layouts diferentes se sumaban en silencio. Dos casos completamente vacíos
  ya conservaban el mapa blanco y permanecen como caracterización.
- Diseño mínimo: reutilizar yield_from_batch y labels_to para mapas, conservando
  neigh_isc y máscaras simétricas de cada configuración. Comprobar identidad de
  tipos, índices/aristas/imágenes/nsc y shapes entre muestras y batches; no exigir
  igualdad de coordenadas MD. Medias por entrada usan conteos int64 de targets
  válidos, excluyendo filas parcialmente NaN. Errores no finitos en targets válidos
  se rechazan. Componentes se dibujan por separado; no se añade referencia atómica
  ni se interpreta el error como H/DM/S. Vacíos quedan blancos; reset por split/época.
- Primer parche aún dio **15 failed, 10 passed**, 233 warnings, 2.88 s
  (periodic-plot-green.log): conteos booleanos y una ruta NumPy que ignoraba
  reemplazos de labels. Corregidos conteos enteros; el callback usa labels_to,
  directamente con los arrays de error. GREEN **25 passed**, 233 warnings,
  2.72 s (periodic-plot-focused.log), con ambas integraciones Lightning existentes.
  Las pruebas solo capturan el límite de visualización; la conversión es real.
- Regresión adicional NumPy del wrapper matrix_from_data: **1 failed, 1 passed**,
  111 deselected, 47 warnings, 0.04 s (numpy-prediction-red.log). El diccionario
  _data compartido no reflejaba labels reemplazadas por atributos; se devolvía
  la referencia en vez de la predicción. El wrapper crea su diccionario propio
  al reemplazar labels NumPy, sin mutar el original; Torch no cambia. Valores
  literales [[2,3],[3,4]], identidad y valores de las referencias preservados.
- GREEN cercano definitivo: **138 passed**, 358 warnings, 3.85 s
  (periodic-plot-numpy-focused.log). Gate completo CPU del producto: **993 passed,
  2 skipped**, 6722 warnings, 118.57 s (client-periodic-plot-full-cpu.log), exit 0.
- Wheel actual compilada (client-wheel-build.log), sha256
  a1311b2c33da274af335da6a7e92dbe66a3ec1ab20130beccbd8fed157942d55;
  instalada en otro venv limpio /tmp/g2m-client-wheel.uOFAJP/venv con base,
  Requests y pytest, sin Torch/e3nn/FastAPI/YAML/Jinja/Cython runtime.
  Cliente desde site-packages sin importar server_app, consulta HTTP real y ayudas:
  **17 passed, 4 deselected**, 9.31 s (client-installed-minimal-tests.log).
  pip check exit 0. Perfil es base+cliente, no una instalación completa ML.
- Añadido al smoke base el valor literal de la reconstrucción NumPy con
  predicciones y entrada preservada; el gate completo precede solo esas nuevas
  aserciones del script. Smoke instalado definitivo: **17 passed, 4 deselected**,
  9.12 s (client-installed-final-smoke.log). Corrección NumPy comprometida en
  **69a8f08**. 3.4 sigue pendiente de extrapolación, /info y formatos
  físicos H/EDM; 3.5 de rangos/constraints y perfil completo; 3.9 de gates/pérdidas,
  CLI --ckpt_path y oráculos científicos independientes. Revisión final pendiente.

### Sublote 3.4.c — canales, SOAP y bloques históricos

- Lote mapas/cliente comprometido en **015fc98**. La continuación trabaja solo
  sobre extrapolation.py, sus nuevas regresiones, extra/CI y documentación.
- RED canales/HistoryCopy: **15 failed**, 56 warnings, 0.12 s
  (series-channel-policy-red.log). DM/EDM se rellenaban con cero; H ignoraba su
  política y conservaba S fuera de las labels; all_dims sobrescribía la columna
  cero del llamador y rellenaba huecos tomando el no-nulo índice 0. HistoryCopy
  rechazaba kwargs reales del extrapolador antes de calcular nada.
- Parche: reutilizar OrbitalConfiguration.from_matrix y get_config_kwargs;
  all_dims pide raw_components explícitamente. Eliminar el truco de índices, no
  duplicar la extracción existente. H retiene 0 y DM/EDM NaN, canales y bloques
  ausentes; matrices originales y referencias no se mutan, incluida su resta.
  HistoryCopy acepta kwargs y conserva los valores/autoregresión anteriores en
  datos completos. GREEN servidor: **24 passed**, 57 warnings, 0.39 s
  (series-channel-policy-green.log), con nueve regresiones HTTP existentes.
- DScribe **2.1.2** y sparse **0.19.2** instalados solo en overlay temporal
  /tmp/g2m-soap-deps.o0Kjat; dependencias restantes ya presentes y de solo lectura.
  No instalación compartida. El import y SOAP reales funcionaron; el primer
  sondeo falló después por usar un orbital sin función radial. La fixture SOAP
  usa un SphericalOrbital real con radial exp(-r), sin sustituir el descriptor.
- La primera fixture SOAP anidaba una lista de orbitales y dio tres fallos antes
  de la función, no evidencia del bug. Corregida la lista, RED real **3 failed,
  1 passed**, 15 deselected, 52 warnings, 2.24 s (soap-periodicity-real-red.log):
  SOAP periódico daba 6.796562 frente a 25.140762; tags H_custom no eran especies
  químicas válidas; el error sin DScribe no indicaba el extra. La caracterización
  no periódica pasó. API primaria contrastada:
  https://singroup.github.io/dscribe/latest/tutorials/descriptors/soap.html.
- SOAP usa Z, PBC de cada configuración y el extra declarado
  graph2mat[extrapolation] (dscribe>=2.1.2,<3, Requires-Python>=3.9 verificado en
  METADATA). Una dependencia interna rota no se oculta. CI completa instala el
  extra para no omitir estos tests; no se afirma CI remota ejecutada.
- RED de vectores tras corregir SOAP: **6 failed, 6 passed**, 16 deselected,
  52 warnings, 0.33 s (soap-all-image-vectors-red.log). Mezclaba shifts ya en
  convención SIESTA con coordenadas Cartesianas (distancia 3.969887 en vez de .4)
  y usaba la celda nueva para historias antiguas (.9,.9,.9 en vez de .4,.4,.9).
  Parche: offsets enteros de sc_off por la celda de cada frame en nodos/aristas;
  sin redefinir la fórmula del descriptor. GREEN **37 passed**, 57 warnings,
  2.81 s (soap-image-vectors-green.log), con SOAP real, custom tags, parcial PBC,
  celda variable y oráculos literales de distancias/descriptor nativo.
- Historial: el primer oráculo eligió el representante inverso al declarado por
  el grafo simétrico; no se cambió su convención para obtener verde. Corregido a
  (1,0,+x), se retiró temporalmente el parche antes de repetir RED: **3 failed,
  1 passed**, 28 deselected, 52 warnings, 0.15 s (history-images-real-red.log).
  Faltaba una dirección en matrices no simétricas; imágenes fuera del nsc previo
  lanzaban IndexError. Ahora el stride depende de symmetric_matrix y esas imágenes
  cuentan como bloques ausentes, sin índices negativos/alias inventados.
- GREEN final cercano: **64 passed**, 57 warnings, 6.64 s
  (series-soap-final-focused.log): 32 regresiones de series/SOAP/historial, nueve
  HTTP y 23 mapas periódicos. Gate CPU completo con overlay DScribe presente:
  **1025 passed, 2 skipped**, 6726 warnings, 240.18 s
  (series-soap-full-cpu.log), exit 0. No se omiten los tests nuevos de SOAP.
- Este lote no termina 3.4: quedan /info, formatos físicos H/EDM, política explícita
  del Lua DM y extrapolación de bloques NaN/vacíos con historial incompleto y
  canales. Los otros frentes y la revisión final conservan su alcance íntegro.

### Inicio de diagnóstico 3.9.g — gates estáticos

- Herramientas instaladas en otro overlay temporal /tmp/g2m-quality-deps.UsKznt:
  Ruff 0.16.10, mypy 2.4.0 y pre-commit 4.6.2; entorno compartido intacto.
  Comprobaciones solo de lectura, no se aplicó autofix ni formateo a otros lotes.
- Baseline Ruff crítico E9/F63/F7/F82 con target py39: **4 errores**, exit 1
  (ruff-critical-baseline.log): ndarray no definido en OrbitalMatrix,
  OrbitalMatrixMetric no definido en processing, _re_setops_sub en el helper
  de globs y referencia a e fuera del except en el fallback FastAPI. Pendientes
  de regresión/parche, no se deshabilitan las reglas para obtener verde.
- El sondeo de tipos con mypy 2.4.0 no llegó al código: exit 2, esa versión ya
  no acepta --python-version 3.9 (mypy-core-baseline.log). Elegir una versión de
  checker compatible con la compatibilidad declarada; no subir el mínimo Python
  para hacer pasar el gate. Aún no hay gate de tipos verificado ni CI de calidad.

### Sublote 3.9.g.1 — nombres indefinidos con regresiones reales

- Extrapolación anterior comprometida en **187402e**. Sin cambios a entrenamientos,
  colas, g2m_autorun.py ni al entorno compartido.
- RED de ejecución: **7 failed, 134 deselected**, 52 warnings, 5.77 s
  (ruff-undefined-runtime-red.log). Cuatro patrones ZIP con clases de caracteres
  fallan en _re_setops_sub; get_type_hints falla en ndarray; dos subprocesses
  reproducen el NameError del fallback FastAPI, también cuando falta starlette.
- Parche mínimo: restaurar la sustitución de operadores del helper vendorizado,
  np.ndarray en la anotación y conservar la excepción como atributo de clase del
  fallback. Solo ausencia del paquete raíz fastapi activa el fallback; una
  dependencia interna rota se propaga. El error indica graph2mat[server].
- OrbitalMatrixMetric se importa bajo TYPE_CHECKING: resuelve la referencia del
  checker sin introducir el ciclo metrics→processing→metrics ni nuevos imports
  opcionales en el núcleo. Su import de ejecución sigue local a compute_metrics.
- GREEN cercano: **150 passed**, 178 warnings, 24.47 s
  (ruff-undefined-focused-green.log). Ruff E9/F63/F7/F82: exit 0, sin ignorar las
  cuatro incidencias. Gate completo CPU con DScribe presente: **1032 passed,
  2 skipped**, 6726 warnings, 139.92 s (static-names-full-cpu.log), exit 0.
- Mypy **1.19.1** compatible con --python-version 3.9 instalado solo en
  /tmp/g2m-mypy39-deps.Qvt5CT. Los dos primeros sondeos no llegaron al código:
  Torch actual usa match (>=3.10), NumPy actual usa type (>=3.12) en sus stubs.
  Son incompatibilidades entre stubs instalados y target, no pruebas del producto.
- Diagnóstico con imports externos opacos por módulo y follow_imports_for_stubs:
  **156 errores en 13 archivos**, 22 fuentes comprobadas
  (mypy39-core-scoped-baseline.log). No se desactivan errores del núcleo. Se
  contrasta también con stubs NumPy 1.26.4 compatibles con Python 3.9 en un
  overlay exclusivo del checker, no cambiando NumPy del producto ni sus tests.
  Ese sondeo conserva tipos NumPy reales: **210 errores en 13 archivos**, 22
  fuentes comprobadas (mypy39-numpy126-baseline.log). Algunos errores del sondeo
  opaco eran artefactos de aliases NumPy tratados como Any; el gate definitivo
  deberá conservar los stubs reales. No se afirma que los tipos pasen.
  Referencia de configuración: https://mypy.readthedocs.io/en/stable/config_file.html.
- El sondeo amplio Ruff E/F/B/I/UP/PLR0133 registra 1652 incidencias
  (ruff-full-baseline.log), principalmente líneas largas, imports/anotaciones.
  El sondeo semántico adicional conserva los detalles de redefiniciones y otros
  avisos en ruff-semantic-baseline.log. Aún no hay gate completo ni cierre 3.9.

### Sublote 3.4.d / 3.9.g.2 — /info JSON y redefiniciones

- Lote de nombres indefinidos comprometido en **75e4080**, con gate completo
  1032/2. /info se reproduce con procesadores/bases reales, no mocks.
- RED: **4 failed, 1 passed**, 9 deselected, 53 warnings, 1.01 s
  (server-info-real-red.log): bases genéricas/atómicas con radios escalares/arrays
  fallan al serializar NumPy o AtomicOrbital. La caracterización de nombres de
  rutas y operationId del frontend sí pasa antes del cambio.
- El primer parche de /info no bastó: el custom encoder de tabla devolvía un
  contenedor sin codificar sus hijos. **4 failed, 69 passed**, 57 warnings,
  26.53 s (server-info-routes-focused-green.log); no se registra como GREEN.
  Corrección: convertir explícitamente basis_table a {basis: [...]}, nombres
  cualificados de getters y reutilizar jsonable_encoder para arrays/scalars.
  Sin deepcopy de la tabla y sin mutar el procesador; restantes claves preservadas.
- Quitadas dos importaciones repetidas en extrapolación y renombradas cinco
  funciones frontend con name="get" explícito. La API conserva nombres e IDs
  de OpenAPI, verificados por el mismo test antes y después del cambio.
- GREEN cercano definitivo: **73 passed**, 57 warnings, 25.69 s
  (server-info-routes-focused-final.log). Ruff E9/F63/F7/F82/F811/B015/PLR0133:
  exit 0 (ruff-semantic-final.log), sin ignorar nombres ni redefiniciones.
  Gate completo CPU: **1037 passed, 2 skipped**, 6726 warnings, 138.90 s
  (server-info-full-cpu.log), exit 0, con DScribe presente.
  Gates configurados/ejecutados en CI y tipos siguen pendientes, igual que el
  resto del alcance de 3.1/3.4/3.5/3.9 y la revisión final del objetivo entero.

### Sublote 3.9.g.3 — contratos de tipos del núcleo y gates locales

- Reanudación en HEAD **2ca32dc**, rama g2m-v2; se conservan cambios y archivos
  ajenos. El gate completo previo 1037/2 no cubre este sublote sin commit.
- Checker Python 3.9 con NumPy 1.26.4 real: de **210 errores / 13 archivos** a
  **173 / 8**, luego **123 / 6**, sin desactivar errores propios del núcleo
  (core-types-basis-progress.log, core-types-sparse-progress.log).
- Contratos corregidos: bases canónicas tras saneamiento, referencias opcionales,
  mapas identidad, shape multicanal, registro de features, diccionario de bloques
  y helpers NumPy/Torch. Casts sobre invariantes ya validados; fórmulas intactas.
- RED real previo: igualdad con referencias None (dos casos) y FunctionDoc con
  func_role explícito. GREEN distingue ausencia de referencia y conserva el rol
  predeterminado. Caracterización Callable con __call__=None mantiene su dispatch.
- Runtime primer sublote: **671 passed, 1 skipped**, 4067 warnings, 33.56 s
  (core-types-basis-runtime.log). Segundo: **599 passed, 1 skipped**, 425 warnings,
  6.39 s (core-types-sparse-runtime.log), con datos Torch y ocho checkpoints.
- RED adicional **6 failed**, 114 deselected, 47 warnings, 0.11 s
  (core-types-processing-boundaries-red.log): neigh_isc sin edge_index,
  coordenadas sin procesador (tres campos) y reconstrucción sin referencias
  (tabla completa ausente o entrada None). Guardas compartidas rechazan datos
  inválidos con causa clara; referencias/etiquetas originales no se mutan.
- GREEN cercano **155 passed**, 184 warnings, 2.51 s
  (core-types-processing-green.log); mypy baja a **75 errores / 3 archivos**.
  Se corrigen los contratos de clases/arrays, retornos de factories conservan
  la subclase NumPy/Torch, tablas/grupos y strings de claves reales del readout.
  No se introducen imports ML obligatorios ni cambian las fórmulas/orden de bloques.
- RED batch vacío real de PyG: **1 failed**, 89 deselected, 47 warnings, 0.05 s
  (core-types-empty-metric-red.log): config_resolved devuelve None. Se inicializa
  el array antes de iterar, preservando resultados para batches no vacíos.
  GREEN datos **571 passed, 1 skipped**, 206 warnings, 3.78 s.
- La primera fixture de coeficientes omitía preprocessing_edges=None y falló
  antes del bug: no es RED válido (core-types-empty-coefficients-red.log).
  Fixture corregida con E3nnGraph2Mat real: **2 failed**, 10 deselected,
  62 warnings, 0.20 s (core-types-empty-coefficients-real-red.log). Reconstrucción
  vacía conserva dtype/device si hay tensors vacíos; sin ellos usa defaults Torch.
- GREEN enfocado final **703 passed, 2 skipped**, 2028 warnings, 23.06 s
  (core-types-final-focused.log). Sondeo de tipos **0 errores / 22 fuentes**
  (core-types-all-green.log), frente a 210 originales; sin ignores propios.
- Gates persistentes: Ruff 0.16.10 semántico sobre paquete/tests, mypy 1.19.1
  sobre todo core no-test con target py39, NumPy real 1.26.4; hooks y job quality.
  Ruling: conservar Black/isort y darles profile=black; Ruff vigila errores
  semánticos, no migra formato/estilo de todo el repo — satisface el gate de
  nombres/redefiniciones sin ampliar este lote a una reescritura mecánica; cuesta
  mantener formateadores separados. No se ignoran errores propios de mypy.
- Instalación de checkers real aislada /tmp/g2m-static-gates.YpKMAH/venv:
  exit 0 (core-types-quality-clean-install.log). Primer hook con PYTHONPATH del
  overlay falló por ejecutable Ruff ausente (core-types-ruff-hook.log); repetido
  sin overlay en venv limpio: Ruff y mypy **Passed**, exit 0
  (core-types-ruff-hook-clean.log, core-types-mypy-hook-clean.log). Comando directo
  del checker con pyproject también comprueba 22 fuentes y sale 0.
  CI remota queda configurada, no se afirma ejecutada.
- Wheel compilada con build isolation: exit 0 (core-types-wheel-build.log),
  sha256 9cbf4c8663d5ccfe8e8508f228a086e2efa6fa599157b35e1ab20e3f27970a52;
  venv base limpio /tmp/g2m-types-wheel.NmY0Ac/venv, sin Torch/PyG/e3nn/MACE/PL
  ni Cython runtime. pip check y graph2mat --help exit 0. **15 passed**, ocho
  deselected, 8.30 s (core-types-wheel-base-tests.log). Python -I confirma import
  desde site-packages y _sparse.so; H polarizado reconstruye literalmente
  [[2,.25],[.25,3]] y [[-1,.125],[.125,-2]], por el conversor compilado.
- Gate CPU completo del árbol congelado: **1050 passed, 2 skipped**, 6726 warnings,
  137.83 s, exit 0 (core-types-full-cpu.log), DScribe presente. Las omisiones
  conocidas son fixture H local y GPU compile, no las nuevas regresiones.
  Los otros requisitos de 3.1/3.4/3.5/3.9 y la revisión final siguen abiertos.

## Cierre del goal

Releer el objetivo, el roadmap y los IDs arriba. Para cada requisito registrar
archivo/prueba/comando y salida real; comprobar cobertura, no solo color verde.
Solo después de completar todos, pasar revisión final, corregir hallazgos y ejecutar
todos los gates se podrá marcar el goal completo. Antes, seguir activo.
