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

## Cierre del goal

Releer el objetivo, el roadmap y los IDs arriba. Para cada requisito registrar
archivo/prueba/comando y salida real; comprobar cobertura, no solo color verde.
Solo después de completar todos, pasar revisión final, corregir hallazgos y ejecutar
todos los gates se podrá marcar el goal completo. Antes, seguir activo.
