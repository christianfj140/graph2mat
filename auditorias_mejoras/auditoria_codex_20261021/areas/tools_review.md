# Revisión independiente de herramientas y campañas — cierre 2026-10-22

**What this change does:** el rango auditado incorpora herramientas SIESTA, métricas de matrices y bandas, desglose del error, servidor/cliente y scripts de campañas. Esta revisión contrasta los ocho hallazgos de `tools.md` contra el código original y añade dos contraejemplos confirmados. No modifica producto ni certifica resultados históricos de campañas.

## Identidad y método

Objeto de revisión: HEAD original **`a66b46cb0955ca344693d50737c5e9952fc257fc`**, dentro del rango original de 267 commits `origin/g2m-v2..HEAD`. Código leído y ejecutado exclusivamente en `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/source`. Las ubicaciones siguientes corresponden a ese HEAD, no al checkout posterior ni a las correcciones de Claude. Los orígenes se contrastaron con el historial fijado a ese commit.

Se releen los contratos y las líneas responsables, sus llamadas y los tests conectados. Se conservan las reproducciones de la revisión anterior y se vuelven a ejecutar sus cuatro scripts. Dos reproducciones adicionales refuerzan T8 y X2 sin alterar las funciones originales. Para los runners se ejecutan cuerpos extraídos mediante AST; entrenamiento, telemetría, lanzamientos e I/O externos se sustituyen por dobles controlados. No se importan módulos de campañas del repositorio MD_vs.

Entorno: wrapper `finish_tools_review/review-python`, importaciones dirigidas a la copia congelada, extensiones de esa copia, `CUDA_VISIBLE_DEVICES=''`, sin bytecode y con límites de dos hilos para OMP/BLAS. Los archivos temporales, logs y cachés de esta ejecución quedan en `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/finish_tools_review/`. No se ejecutan SIESTA real, GPU, fits, consultas HTTP externas ni operaciones de cola. T3 usa únicamente un proceso Python local que copia una matriz sintética y sale con código 7.

## Dictamen de los ocho hallazgos originales

**Resultado: 8/8 confirmados; ninguna objeción que los retire.** Se mantienen las prioridades originales. P1 identifica resultados científicos silenciosamente erróneos en interfaces públicas; P2 identifica fallos bajo condiciones concretas que deben constar en el contrato.

| ID | Prioridad | Dictamen, ubicación y origen | Caso observado, corrección mínima y límite |
| --- | --- | --- | --- |
| T1 | P1 | **Confirmado.** `src/graph2mat/tools/siesta/delta.py:123`, `:127`; `a6f22224`. | Hamiltonianos físicamente iguales, `nsc=[3,1,1]`, con orden ±1 de `sc_off` permutado: la resta correcta es cero, pero devuelve **3.968458805 eV** a k=(0.23,0,0). Sigue siendo Hermítica. Alinear columnas por imagen física o rechazar layouts distintos. El repro no demuestra que lectores o campañas históricos hayan producido esa permutación. |
| T2 | P1 | **Confirmado.** `src/graph2mat/tools/analysis/metrics.py:56`, `:65`, `:81`; `041e0ad7`. | Referencia y predicción comparten spin-up; spin-down difiere **1 eV** por nivel. `gamma_metrics` devuelve cero en todas sus métricas de error. Rechazar spin polarizado sin semántica implementada, o seleccionar/agregar canales explícitamente. El caso demostrado es collinear; no implica que las campañas spin1 estén afectadas ni certifica una semántica SOC. |
| T3 | P2 | **Confirmado.** `src/graph2mat/tools/siesta/delta.py:89`; `a6f22224`. Política análoga en `src/graph2mat/tools/siesta/dm_pilot.py:51`. | Primera llamada escribe TSHS y falla con salida **7**. Segunda acepta ese archivo, con geometría desplazada **0.5 Å**, pese a tolerancia **0.0001 Å**. Publicar cache solo después de éxito y validación; validar el cache al reutilizarlo. La reutilización de un resultado terminado está permitida: el defecto es aceptar un artefacto del intento fallido. Repro ejecutado sobre `compute_h0`, no sobre `run_from_dm`. |
| T4 | P2 | **Confirmado con límite explícito.** `src/graph2mat/tools/siesta/dm_pilot.py:72`, `:73`; `ba087112`. | Con `S=I`, diagonales exactas y error off-diagonal **0.1 eV**, el denominador de gauge solo ve ceros y devuelve **NaN**. Calcular gauge sobre soporte fijo que incluya S y declarar soporte RMS. Se exige resultado finito; no se impone un denominador único: RMS es 0.1 eV sobre los dos errores o 0.070710678 eV sobre las cuatro posiciones. No revalida el piloto histórico 7.7. |
| T5 | P2 | **Confirmado.** `src/graph2mat/tools/siesta/runner.py:42`; `d28c65ba`. | `System.Label sample` produce `sample` mediante lector FDF instalado, pero `siesta` mediante `system_label`; los llamadores buscan otro nombre de salida. Usar el parser FDF disponible, con default solo si falta la propiedad. La reproducción ejecutada demuestra normalización del nombre; el problema de includes se respalda por lectura de la expresión regular, no por un segundo repro de includes. |
| T6 | P2 | **Confirmado en parámetro explícito.** `src/graph2mat/tools/analysis/metrics.py:228`, `:231`; `041e0ad7`. | Con `S=I` y `threshold=2`, `n_keep=0` y `[-0:]` retiene todas las bandas. Rechazar explícitamente subespacio vacío antes de normalizar. Este repro usa umbral 2, no el default 0.01; no demuestra fallo con una S normalizada y dicho default. |
| T7 | P2 | **Confirmado.** `src/graph2mat/tools/analysis/breakdown.py:131`; `041e0ad7`. | Una fila válida con cuatro elementos y error cero produce **ZeroDivisionError**. Definir participaciones nulas o indeterminadas al sumar error cero, conservando MAE y conteos. El test existente comprueba `structure_breakdown` exacto, pero no agrega esa fila. Lista vacía no forma parte del caso contado. |
| X1 | P2 | **Confirmado para presupuesto general.** `auditorias_mejoras/run_soc_campaign.py:257`, `:284`; `88c344e`. `auditorias_mejoras/run_magnetic_campaign.py:366`, `:394`; `d9b5904`. | En ambos cuerpos `run()` originales, diez muestras, batch 2 y solicitud de siete updates producen configuración de **una época/cinco batches**, pero JSON registra **siete updates**. Usar presupuesto por pasos y guardar `global_step` efectivo separado de la solicitud. Son dobles sin acumulación ni entrenamiento real; no se afirma incumplimiento de presupuestos históricos de 8000, cuya divisibilidad y muestra efectiva no se comprobaron aquí. |

Las consecuencias y propuestas de `tools.md` son compatibles con la evidencia, respetando los límites anteriores. T4 no debe convertirse en una discusión sobre un valor RMS único, T6 no debe describirse como fallo del umbral habitual y X1 no permite atribuir costes incorrectos a campañas concretas sin sus artefactos.

## Nuevos hallazgos confirmados

### T8 — P2 — El desglose periódico asigna enlaces cercanos a categorías lejanas con coordenadas sin envolver

**Ubicación:** `src/graph2mat/tools/analysis/breakdown.py:65`, `:68`, `:74`; efecto en `:99`, `:103` y `:107`. **Origen:** `041e0ad7`, dentro del rango auditado.

**Qué hace:** `min_image_distances` promete la distancia a la imagen periódica más cercana. `structure_breakdown` utiliza esa distancia para shells y para declarar elementos fuera de `R_i + R_j`.

**Problema:** enumera solo traslaciones de −1 a +1 y utiliza coordenadas originales, sin envolverlas ni exigir que estén envueltas. En celda ortogonal de 10 Å, periódica en x, con átomos en x=0 y x=31 Å, devuelve **21 Å**. La imagen x−3L está a **1 Å**; al representar el segundo átomo en x=1 Å, la misma función devuelve 1 Å.

El repro adicional usa `nsc=[7,1,1]`, con hoppings almacenados en las imágenes físicas **−3 y +3**, a distancia real 1 Å; no depende de asignar un hopping espurio entre átomos separados 31 Å. H es Hermítica en Γ y fuera de Γ. Tras escribir y leer TSHS, el desglose real asigna **0.2 eV** de error a `beyond_range` y `shell_6.0-99.0`, y **0 eV** a `shell_0.1-1.6`. Cada orbital tiene radio 2 Å, de modo que el enlace está dentro del alcance combinado de 4 Å.

**Corrección mínima:** calcular la imagen mínima desde desplazamientos periódicos reducidos, con algoritmo válido para las celdas admitidas; o rechazar explícitamente geometrías fuera del dominio que el algoritmo soporte. Envolver coordenadas corrige el caso ortogonal reproducido, pero no certifica por sí solo la imagen mínima de celdas arbitrariamente sesgadas. La restricción actual no consta en el contrato.

**Si se omite:** el diagnóstico atribuye error a colas lejanas y elementos supuestamente espurios aunque proceda de enlaces cortos. No se demuestra que las geometrías históricas estuvieran sin envolver, ni que sus cifras de shells/alcance estén afectadas. No se amplía el hallazgo a otros algoritmos periódicos no reproducidos.

**Evidencia:** `reproduce_t8.py/.log` y `reproduce_t8_periodic_hopping.py/.log` en el directorio de cierre.

### X2 — P2 — El experimento energético omite la guardia térmica y de VRAM cuando no tiene hijos activos

**Ubicación:** `auditorias_mejoras/run_energy_loss_experiment.py:117`; contrato en `:72` y `:104`. **Origen:** `4dc39833`, dentro del rango auditado.

**Qué hace:** `train_all` promete arrancar cada entrenamiento únicamente cuando `_room()` lo permita. La guardia comprueba concurrencia, temperatura CPU/GPU y margen de VRAM antes de añadir otro proceso.

**Problema:** la condición `(not running or _room(...))` evita consultar la guardia con la lista local de procesos vacía. El primer entrenamiento arranca aunque `_room(0, 3)` rechace las condiciones. Lo mismo puede ocurrir tras quedar vacía esa lista en una iteración posterior. `run()` espera al speed probe, pero no llama a `_room` antes de entrar en `train_all`.

La reproducción inicial registra `room_would_allow=false`, **cero llamadas** a `_room` y **un arranque** de `control/seed0`, con paralelismo 3. La reproducción reforzada ejecuta también el cuerpo original de `_room` contra tres telemetrías simuladas: exceso de temperatura GPU, exceso de temperatura CPU y VRAM insuficiente. En los tres casos la guardia real devuelve falso, pero `train_all` registra el arranque sin invocarla. `start` solo captura argumentos y `sleep` interrumpe la simulación; no se crea proceso de entrenamiento ni se consulta hardware.

**Corrección mínima:** exigir `_room(len(running), parallel)` para todo arranque, también cuando `running` esté vacío. Mantener la espera y la reevaluación cuando la guardia deniegue el permiso.

**Si se omite:** puede iniciarse entrenamiento sin los márgenes térmicos o de memoria que este runner declara comprobar. La eventual supervisión del daemon no cumple la promesa local de comprobarlos antes del arranque. El script permite compartir GPU expresamente: **no se reclama exclusividad ni bloqueo de cualquier PID ajeno**. No se demuestra un OOM, sobretemperatura real ni perjuicio en la campaña histórica. La regla del traspaso sobre comprobar recursos antes de nuevos procesos concuerda con este contrato; no es necesario extenderla a contratos de otros runners para confirmar el defecto.

**Evidencia:** `reproduce_energy_guard.py/.log` y `reproduce_energy_guard_contract.py/.log` en el directorio de cierre.

## Reproducciones y validación

Todos los siguientes comandos terminaron con salida **0**, ejecutados desde `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/finish_tools_review/`:

```text
./review-python reproduce_tools.py
./review-python reproduce_updates.py
./review-python reproduce_t8.py
./review-python reproduce_energy_guard.py
./review-python reproduce_t8_periodic_hopping.py
./review-python reproduce_energy_guard_contract.py
```

Los resultados se guardan con el mismo nombre y sufijo `.log`. Los asserts exigen que aparezca el defecto conocido: salida cero del reproducer confirma el contraejemplo, no que producto esté corregido. `reproduce_tools.py` conserva además el control de `ServerClient.predict` con geometría pasada como objeto o como ruta: ambos devuelven coordenadas correctas en este entorno. La sospecha previa sobre esa ruta queda descartada, sin generalizarla a pruebas de red o carga.

Se ejecutaron también los tests originales conectados, sin añadir ni modificar tests:

```text
review-python -m pytest -q -p no:cacheprovider --basetemp=<cierre>/pytest-tmp \
  src/graph2mat/tools/analysis/tests/test_analysis_metrics.py \
  src/graph2mat/tools/analysis/tests/test_breakdown_attribution.py \
  src/graph2mat/tools/siesta/tests/test_delta.py \
  src/graph2mat/tools/siesta/tests/test_runner.py \
  src/graph2mat/tools/siesta/tests/test_label_noise_dm_pilot.py
```

Resultado: **30 passed, 55 warnings, 0.72 s**, salida 0; log `finish_tools_review/pytest_original.log`. Incluyen aritmética Δ con spin1/collinear y cambio de `nsc`, cache tras éxito, rechazo de geometría movida tras éxito del proceso, gauge, métricas/bandas y desglose normal. Pasan pese a los contraejemplos: no ejercitan permutación de `sc_off`, spin-down incorrecto en métricas, artefacto dejado por proceso fallido, error solo off-diagonal, etiquetas FDF normalizadas, subespacio vacío, agregación exacta ni coordenadas desplazadas varias celdas. Tampoco cubren los dos cuerpos de presupuesto o la guardia del runner energético.

## Cobertura y límites del cierre

La revisión focal relee `tools/siesta/{runner,delta,dm_pilot}.py`, `tools/analysis/{metrics,breakdown}.py`, sus llamadas y los cinco módulos de tests anteriores, junto con los cuerpos pertinentes de los tres runners. El trabajo heredado revisó además herramientas migradas, servidor/serialización, cliente y contratos críticos de recuperación/publicación enumerados en `tools.md`; este cierre no presenta esa cobertura más amplia como una segunda ejecución exhaustiva de todo el servidor o de todos los benchmarks.

No se modifican ni se reabren campañas, preparaciones o colas. No se vuelven a medir precisión, coste, OOD, convergencia, temperatura ni VRAM reales. No se verifican aquí presupuestos efectivos de campañas cerradas ni se trasladan estos casos sintéticos a sus resultados. Las comprobaciones globales de entorno y tests del servidor pertenecen al cierre coordinado de la auditoría; no añaden un bug de lógica a estos diez hallazgos.

**Veredicto:** mantener T1–T7/X1 con los límites precisados y añadir T8/X2, ambos P2. Corregir T1/T2 antes de considerar certificadas aritmética periódica genérica y métricas collineares; resolver los demás antes de usar los casos que activan sus fallos. Esta revisión no aplica arreglos ni juzga las correcciones posteriores de Claude.

**No comprobado:** campañas históricas reales, ejecución GPU/SIESTA, red/carga y validez de geometrías arbitrariamente sesgadas. Ningún fallo de campaña o de hardware se deduce de este informe.
