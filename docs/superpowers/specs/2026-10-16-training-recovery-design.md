# Recuperación de entrenamientos al ceder la GPU

Estado: **diseño aprobado**. El usuario aprobó el enfoque el 2026-10-16 mediante
«Apruebo»: recuperar desde la última época completa, con checkpoint atómico
separado del mejor `val_loss` y contabilidad del trabajo repetido. Después aprobó
este documento mediante «Listo, apruebo». Autoriza redactar el plan; no aprueba
por anticipación su ejecución ni trabajos científicos todavía no presentados.

Repositorio: `/home/christian/repositorios/graph2mat`, rama `g2m-v2`, checkout
actual. Sin commits ni worktrees nuevos. Evidencia inicial:
[preflight](../../../auditorias_mejoras/preflight_h2oclu_recuperacion_20261016.md).

## 1. Objetivo y alcance

Evitar que cada cesión a cómputo ajeno reinicie el entrenamiento desde cero.
Reutilizar Lightning y el supervisor existente, restaurando un estado completo
de una época terminada. No llamar continuidad exacta a una carga de pesos.
Conservar por separado la selección científica del menor `val_loss`.

La recuperación será opt-in en los runners experimentales de esta campaña.
El entrenamiento normal de Graph2Mat no cambia de comportamiento ni de defaults.
Primera integración: la comparación fixed/cost de 6.4 en h2oclu, todavía sin
trabajos ni presupuesto científico autorizados. El protocolo científico y su
presupuesto se concretarán después de certificar la recuperación.

No se modifica ni se relanza la campaña 7.2 cerrada. Su runner y artefactos
congelados quedan intactos; se reutilizan sus primitivas operativas, no su
checker exclusivo de gradientes de actualización de aristas. Fuera de alcance:
autorun, fonones, a-SiO2, dependencias compartidas, código/datasets externos,
DDP, recuperación intraépoca exacta y migración de checkpoints históricos.

## 2. Alternativas

1. **Última época completa, elegida y aprobada.** Estado de entrenamiento y
   selección verificables; se repite el fragmento no confirmado de la época.
   Limita pérdidas de progreso sin serializar colas de prefetch ni procesos.
2. Checkpoint intraépoca. Requiere posición y estado de sampler, batches
   prefetched, workers y aleatoriedad asociada. No se implementa por aproximación
   ni se afirma equivalencia sin una prueba que cubra todo ese estado.
3. Reinicio con pesos del último/best checkpoint, sin optimizador/RNG. No cumple
   el objetivo y no se ofrece como fallback silencioso.

## 3. Componentes y reutilización

Un módulo experimental pequeño en `auditorias_mejoras/training_recovery.py`
contendrá el callback de recuperación, validación/publicación de generaciones
y adaptación de orden de datos. Sus pruebas vivirán en el mismo directorio.
El runner de la nueva campaña 6.4 incorporará estas piezas; no se crea otro
daemon ni otro formato de cola.

Reutilización obligatoria:

- `bench_pipeline_gpu.guarded_command`, `memory_fraction`, telemetría de GPU
  y `process_tree_memory` para supervisar y medir procesos propios.
- `run_readout_comparison.process_tree`, `gpu_processes` y comprobación de hashes.
- Lightning `save_checkpoint`/`fit(ckpt_path=...)`, estado nativo de callbacks,
  optimizador, scheduler, progreso y precision plugin/scaler cuando exista.
- MatrixDataModule y `_CostBatchSampler` actuales, incluyendo su proxy y regla
  de singleton oversize. No reescribir empaquetado ni preprocesado.
- `OrbitalMatrixCLI` cuando sea necesaria la misma construcción desde config;
  verificar que sus defaults de checkpoint no sustituyan overrides explícitos.

Cambios de código previstos confinados a `auditorias_mejoras`. Si aparece un
defecto de producto que impida este contrato, demostrarlo primero y exponer el
cambio de alcance antes de modificar `src`. No extender APIs públicas por
anticipación ni adaptar callbacks de campañas cerradas que rechazan resume.

## 4. Estado recuperable y frontera certificada

Cada checkpoint completo incluye pesos/buffers, optimizador, scheduler,
scaler/precision plugin si procede, contadores y loops de Lightning, y estado
de todos los callbacks que influyen en entrenamiento o selección.
El callback adicional guarda RNG de Python, NumPy, Torch CPU y CUDA del único
dispositivo, seed base, próxima época y contadores de muestras/actualizaciones
confirmados. Telemetría acumulativa de intentos no se rebobina con el modelo.

Solo se publica una generación después de consumir todos los batches previstos
de la época, terminar su validación cuando corresponda, actualizar el scheduler
y recoger la selección científica correspondiente. Se verifica la cobertura
contra el plan de muestras de la época. Un callback de fin de época no es por
sí solo prueba de que todos estos estados hayan quedado consistentes.

Para la receta h2oclu actual: Adam, optimización automática, un dispositivo,
CosineAnnealingLR por época, sin gradient accumulation y precisión fp32. La
implementación deberá comprobar el orden real de hooks de Lightning: scheduler
no-plateau al último batch y callbacks monitorizados después del hook del
módulo. El callback de recuperación se ejecuta después de la selección best.
Configuraciones cuya frontera no esté certificada, incluido scheduler plateau,
no se admiten silenciosamente. El scaler se restaura mediante Lightning y se
comprueba en smoke AMP; eso no convierte otras recetas AMP en científicamente
validadas ni altera fp32 de la campaña.

No se publica como época completa un corte por max_steps/timeout en mitad de
sus muestras. Si aún no existe una generación completa, una cesión permite
reiniciar desde la inicialización pareada congelada, registrando todo el trabajo
anterior como repetido. Si había generaciones publicadas y ninguna es válida,
se detiene el trabajo: no se oculta corrupción reiniciando desde cero.

Al restaurar, imports, construcción de objetos, compilación y sanity checks no
pueden desplazar el RNG efectivo del entrenamiento recuperado. Su restauración
final ocurre antes del primer batch entrenado y después de los preparativos que
consumen aleatoriedad. No se aplica un seed_everything incondicional después.

## 5. Orden de datos, workers y pareado

Adaptación exclusivamente experimental de MatrixDataModule: misma permutación
por `(seed, época)` en ambos brazos, generador de RandomSampler independiente
del RNG del modelo y generador de DataLoader separado. Reutilizar loaders,
collate y sampler de coste existentes. El brazo fijo limita estructuras por
batch; el de coste añade el presupuesto existente. Mismo orden concatenado de
sample_id y mismas muestras por época, aunque cambien agrupación y updates.

Ambos brazos recargan el loader cada época para que construcción, longitud y
workers sigan el mismo protocolo. En coste ya es obligatorio; en fijo es un
override experimental declarado, no una modificación del default público.
Generadores, plan y siguiente época se verifican contra la huella congelada.
La restauración no hereda el estado accidental de un iterador del intento viejo.

Contrato de workers: dataset cacheado determinista, sin augmentations aleatorias,
rotating pools ni samplers personalizados. Se prueban 0 y 4 workers, incluyendo
prefetch. No se serializan procesos ni se promete equivalencia para transforms
estocásticas en workers. Si se detectan, se rechaza esa configuración en vez de
presentar como exacta una recuperación incompleta.

La nueva comparación requiere referencias fixed nuevas bajo este mismo protocolo.
Las históricas no prueban igualdad de trayectoria ni son tiempos pareados.
Los pesos/buffers iniciales completos de cada seed se congelan y se comparan
por hash entre fixed/cost. Un resume no vuelve a sustituirlos por inicialización.

## 6. Publicación atómica, integridad y retención

Un solo escritor por job, protegido mediante bloqueo local. Rutas limitadas al
directorio propio del job; rechazar escapes/symlinks a destinos externos.
Cada generación contiene checkpoint de entrenamiento, recibo JSON con tamaños
y SHA256 y copia protegida del checkpoint científico seleccionado, si existe.
El recibo incluye huella de receta, datos/splits/bases, fuentes, versiones,
seed, plan de batching, estado de frontera y progreso confirmado.

Publicación:

1. Crear generación temporal única dentro del mismo filesystem del job.
2. Guardar con Lightning el checkpoint completo y copiar el best seleccionado.
   Sin best todavía, registrar explícitamente ausencia; no inventar uno.
3. Verificar lectura CPU, campos obligatorios, huella, frontera, tamaños/hashes
   y consistencia de referencias a selección científica; fsync de archivos.
4. Publicar el directorio mediante rename atómico, persistir directorios y
   reemplazar atómicamente el índice de generaciones válidas. Temporales nunca
   cuentan como checkpoints recuperables.
5. Solo después del commit confirmado, conservar las dos generaciones válidas
   más recientes y retirar generaciones propias antiguas no referenciadas.
   No borrar artefactos de otras campañas ni el best científico en uso.
   Eliminar bests de trabajo huérfanos solo cuando no estén referenciados por
   ninguna generación retenida ni por la selección actual, dentro de rutas
   generadas y verificadas del job; no barrer directorios compartidos.

El índice o la última generación corruptos permiten buscar una generación
anterior íntegra y compatible, registrando el fallback y su pérdida de progreso.
Incompatibilidad de la receta/datos/fuentes solicitados detiene la ejecución;
no es un motivo para entrenar con otra configuración. No se cargan pickles de
procedencia desconocida: primero validar rutas propias y recibos contra la huella
del job, después cargar con la política de confianza de Lightning existente.

## 7. Selección científica protegida

El ModelCheckpoint de menor `val_loss` mantiene un directorio estable del job,
independiente de logs y carpetas de intento. Restaurar score, best_k_models,
paths y demás estado de callback, no solo best_model_path.

Cada generación conserva su propio best porque ModelCheckpoint puede eliminar
el best anterior antes de que se confirme la siguiente generación de recuperación.
Al retroceder, verificar/restablecer ese artefacto únicamente en su ruta propia
esperada, para que el estado restaurado no apunte a un archivo eliminado.
Un archivo existente inesperado o incompatible no se sobrescribe silenciosamente.
El best de una época no confirmada no sustituye el de la generación restaurada.

Si el último checkpoint confirma ya todo el presupuesto y el proceso cayó antes
de publicar fit.json, finalizar desde esa evidencia sin volver a entrenar ni
necesitar otro avance de GPU. Selección científica y resultado final se validan
antes de liberar una dependencia de cola.

## 8. Cesión y errores

El padre detecta cómputo ajeno usando las primitivas existentes. Reutiliza cierre
de OWN process group: TERM, espera máxima actual de 10 s, KILL si hace falta.
No espera a que termine una época ni a un guardado nuevo para ceder. Si un guardado
en curso se interrumpe/falla, conserva la última generación confirmada y libera
recursos propios. La cota es del protocolo supervisor, no garantía de latencia
del driver/SO ante un kernel no interrumpible.

Esperar GPU libre antes de reintentar, comprobar hashes y elegir generación
válida; pasar ckpt_path al fit real. Sin señales ajenas. Conservar allocator≤55%,
reserva inicial30%, aborto propio si libre<25% o GPU≥80°C y watchdog de progreso
300 s piloto/900 s completo. No prometer reserva dura ante asignaciones ajenas
entre muestras. Mantener bootstrap de extensiones y PYTHONPATH absoluto.

No añadir dumps periódicos de faulthandler. Errores de integridad/configuración,
OOM, pérdida no finita o falta de progreso se registran y fallan de forma visible;
solo la cesión reconocida habilita reintento automático. No cambiar autorun,
reactivar trabajos ajenos ni convertir fallos científicos en bucles infinitos.

## 9. Contabilidad y medición

Por intento: motivo de salida, tiempos de espera/setup/compilación/fit/guardado/
validación/cierre, generación usada/publicada, época/updates/muestras al inicio
y final observados, trabajo confirmado y trabajo descartado/repetido.
Contadores confirmados salen del checkpoint; observados, del heartbeat. Si el
último heartbeat precede al cierre, reportar intervalo o cota de trabajo perdido,
no un número exacto ficticio. Sumar todos los intentos; no llamar GPU-hours de
utilización a duración de procesos supervisados.

Allocated/reserved, telemetría total de VRAM/temperatura y RSS/PSS incluyen coste
de recuperación. Diferenciar entrenamiento útil, repeticiones y overhead frente
a calendario con esperas y evaluación. Comparación 6.4 conserva tres seeds
pareadas, recetas/splits congelados y selección por val_loss; este documento no
fija todavía trabajos, max_batch_cost ni presupuesto científico.

## 10. Pruebas y aceptación

- RED→GREEN: interrupción tras una época completa y varios batches de la
  siguiente; demostrar que hoy no recupera y después sí, repitiendo interrupciones.
- CPU, procesos nuevos: continuo vs interrumpido/reanudado con iguales muestras
  admitidas, orden, pérdidas, parámetros/buffers, Adam, cosine, loops, callbacks
  y RNG Python/NumPy/Torch. Ejercitar aleatoriedad del modelo en una fixture pequeña
  y además MatrixMACE real, fixed/cost y loaders de 0/4 workers.
  Comparar continuidad por separado en cada brazo: no exigir los mismos pesos
  finales entre fixed/cost, que deliberadamente agrupan muestras de modo distinto.
- En fixture CPU determinista exigir igualdad exacta; para rutas numéricas que
  necesiten tolerancia, fijarla y justificarla con control continuo repetido antes
  de atribuir diferencias a resume. No relajar tolerancias para hacer pasar.
- Corrupción/truncado/index inválido, guardado fallido y terminación durante
  publicación: fallback anterior íntegro; incompatibilidad debe bloquear. Cubrir
  primera época sin checkpoint y caída tras último commit antes de fit.json.
- Crear un best temprano y validaciones posteriores peores: reanudar no pierde
  score, selección ni archivo. Repetir con mejora no confirmada y fallback.
- Ownership y temporización: cierre acotado, temporales ignorados, retención
  limitada a artefactos propios y ningún envío de señales a cómputo ajeno.
- Smoke GPU/AMP acotado, solo si libre y mediante disparador propio controlado;
  demostrar checkpoint/optimizer/scaler/restauración y contabilidad. Sin matar
  procesos ajenos ni presentar el smoke como entrenamiento científico.
- Pruebas afectadas, Ruff/mypy aplicables y suite CPU general. Distinguir los
  cuatro fallos históricos HistorySOAPs por dscribe ausente; no instalarlo.

El punto 1 solo cierra con implementación y evidencia dinámica. El Goal completo
no cierra hasta ejecutar, con autorización concreta, y documentar la comparación
científica 6.4, aunque su resultado sea negativo. Próxima puerta: revisión del
plan de implementación, preservando el método Native elegido anteriormente.
No hay código implementado ni campaña lanzada por aprobar este documento.
