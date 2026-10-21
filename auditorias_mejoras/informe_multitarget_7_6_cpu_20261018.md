# 7.6 — integración multitarget H/S/DM/EDM en CPU

Fecha:2026-10-18. Checkout `<graph2mat>`, rama
`g2m-v2`, HEAD270d17d, cambios previos preservados. Diseño y plan aprobados
por el usuario con «Apruebo» en sus respectivas solicitudes. Ejecución Native.
[Diseño aprobado](../docs/superpowers/specs/2026-10-18-multitarget-7-6-design.md)
y [plan ejecutado](../docs/superpowers/plans/2026-10-18-multitarget-7-6.md).

**Estado final:** implementación7.6 cerrada y verificada en CPU; revisión
independiente y tres Important corregidos con regresiones RED→GREEN. Suite
completa posterior:2892passed/20skipped/0fallos,43146warnings,845.34s, salida0.
No hay etiquetas científicas compatibles certificadas ni campaña autorizada:
se cierra el contrato técnico, no una mejora de precisión ni el problema OOD.
4.1 permanece cerrado según su informe propio, no se ha reabierto.

## Qué se implementa

Un backbone MACE y el readout multicomponente existente, no cuatro MACE.
Descriptor version1 con orden H/S/DM/EDM, spin real1/2 y slices exactos;
cuatro componentes no polarizadas o siete colineales para el conjunto completo.
H y S obligatorios, DM/EDM opcionales. Matrices directas, sin H0/DM0 implícitas.
Selección opt-in, pérdidas/pesos/unidades separados y máscaras por componente.
Geometría/base/órdenes orbitales e imágenes y gauges de H/EDM se contrastan.

| Target | Canales físicos | Unidad | Loss inicial | Observaciones |
|---|---|---|---|---|
|H|1/2 reales|eV|MSE o gauge-MSE; energía opcional|Incluye ceros estructurales|
|S|1 escalar|Adimensional|MSE|Incluye ceros estructurales|
|DM|1/2 reales|Ocupación sisl|MSE; N/Mulliken/energía opcionales|Soporte archivado finito, restringible por componente|
|EDM|1/2 reales|eV|MSE; energía opcional|Soporte archivado finito, restringible por componente|

H gauge y objetivos físicos opcionales usan S_ref. Las reconstrucciones y
NetCDF autocontenidos usan S_pred. No se repara S, no se recortan autovalores,
no se aplica el threshold numérico de H/DM/EDM a S y no se recuperan bloques
fuera del grafo. La energía4.4–4.5 sigue siendo perturbación sobre la referencia
completa, no energía exacta del writer ni identidad entre predicciones.

FDF/TSHS/TSDE se prueban con ficheros temporales reales y fixtures sintéticas
declaradas, nunca etiquetas científicas derivadas de una predicción. La lectura
comparte validadores y lectores con4.4–4.5. TSDE no proporciona el overlap.
Un cero almacenado es observado; soporte DM/EDM ausente no lo es. Si sisl
comparte soporte entre spins no se reconstruye una distinción perdida: el
bundle admite máscaras explícitas que solo restringen observaciones reales.

Named views diferenciables en `matrix_predictions` sobreviven a backward,
checkpointing, callbacks y save/load. La inferencia desde geometría obtiene el
descriptor de config/checkpoint, sin etiquetas como features. Layouts con igual
anchura pero nombres distintos se rechazan; no basta comparar shapes.

## Evidencia reproducible por contrato

Logs, ledger, baseline numérico previo y benchmark:
`.superpowers/sdd/2026-10-18-multitarget-7-6/`. Se conservan porque no hay commits.
Los conteos de gates se solapan; no se deben sumar como pruebas independientes.

| Contrato | Evidencia principal | Gate por tarea (incluido en el gate final) |
|---|---|---|
|Descriptor/unidades/spin, ceros/soporte, FDF real, errores|`core/data/tests/test_multitarget.py`; task-1-tests.log|258passed con regresiones; módulo ampliado a37passed después|
|Dataset, caché, batches2/3átomos H/He, misma anchura distinta semántica, CLI spin|`bindings/torch/data/tests/test_multitarget_dataset.py`, `test_cli.py`; task-2-tests.log|174passed|
|Suma de losses independientes, peso0/NaN, máscaras por spin, error por muestra, física fraccionaria/periódica|`bindings/torch/tests/test_multitarget_loss.py`; task-3-tests.log|242passed con regresiones físicas|
|Un backbone, gradientes de cada target, lean/full/checkpoint/sum/mean, label leakage|`tools/lightning/tests/test_multitarget_training.py`|forward único y gradientes finitos/no nulos; nueva suite incluida en gate381|
|Paso Trainer CPU, preflight muestra posterior antes del update, epochs particionadas, checkpoint candidato y legacy|mismo módulo; task-4-tests.log|381passed con regresiones de métricas/stages/CLI|
|Legacy exacto|`test_legacy_numerics_keys_and_checkpoint_without_new_fields`|88claves, digest y valores literales capturados antes de modificar MatrixMACE; recarga sin campos nuevos|
|H/S/DM/EDM reconstruidas, S_pred distinta de S_ref, 13/4orbitales H/O y periodicidad|`core/data/tests/test_multitarget_reconstruction.py`|25controles; s/p/d/f, spin1/2, dirigido/media arista, espacio completo representado|
|Writers reales y protección de salida, CSV por target y máscara de spin, plot y HTTP rechazado|`test_multitarget_callbacks.py`, tests servidor; task-5-tests.log|402passed con regresiones; sin skips|
|YAML y predict CLI real sin etiquetas|`docs/examples/multitarget_predict.yaml`, task-6-doc-example.log|1passed; checkpoint matching, runner CPU real|
|Ruff y whitespace|22archivos Python tocados, `git diff --check`|pasan tras fixes y al cierre|

### Gate completo final y procedencia

Comando de reproducción de abajo ejecutado tras **todos** los cambios Python,
incluidos los fixes de revisión. Log autoritativo `task-6-tests.log`:
**2892passed,20skipped,43146warnings,845.34s, exit0**. No hubo nuevos cambios
de producto ni tests durante/después; solo documentación. El snapshot de los
22archivos de código/tests tocados en7.6 y seis archivos previos queda en
`final-source-sha256.txt`, verificado al terminar. HEAD270d17d sigue igual;
ningún commit, cambio de rama ni instalación.

Las20omisiones son explícitas, no fallos ocultos:1ejemplo externo4.HSX ausente,
1compile inductor que exige GPU,9tests coeficientes CUDA y9tests de fusión
con supervisor GPU opt-in no autorizado. No se certifica lo omitido. La suite
previa `full-suite.log` (2880passed/20skip) era anterior al último ajuste JSON
y a los fixes; se conserva como historia, no es el gate final. Las advertencias
contabilizadas no se presentan como cero ni como certificación de dependencias.

Ruff de los22archivos Python tocados, `git diff --check` y comprobación de
los seis hashes previos pasan. Parseo de14módulos productivos con gramática
Python3.9 correcto; runtime real fue Python3.12.3/Torch2.11.0, no certificación
adicional de runtime3.9. Contrato4.1 anterior permanece cerrado y sus tests
también están en esta suite general; no se lanzó otra auditoría4.1.

Los logs y workspace se conservan porque no hay commits y son evidencia
durable. No ejecutar limpieza de ignorados si se quiere preservar este registro.

No se relajan tolerancias para aprobar resultados: matrices físicas y niveles
bien condicionados se contrastan en float64 con rtol/atol1e-10 y controles
analíticos más estrictos. Los negativos detectan S singular, S incorrecta,
particiones con denominadores distintos, NaN observado, máscara mal tipada,
igual-ancho con otra semántica y pérdida física sin partners.

La referencia numérica legacy sin entrenar se capturó con seed766: nodos
`[0.039056375364447476]*2`, arista `[-0.0073608053886730015]`. El test persistente
usa esos literales y el digest de claves, no necesita el artefacto ignorado.
PyG usa backward reentrant para checkpointing; el test no exige la operación
autograd.grad que ese mecanismo explícitamente no soporta.

## Coste CPU exploratorio, no mejora de velocidad

Dos átomos, dos orbitales1s y una media arista, spin1, float64, dos hilos;
modelo pequeño2x0e+2x1o. Misma geometría/hiperparámetros, sin pérdidas físicas,
sin optimizer update ni entrenamiento científico. Cinco warmups,20rondas
alternando orden de brazos. Mediana ± MAD en ms:

| Fase | H-only | H/S/DM/EDM | Razón conjunta/H |
|---|---:|---:|---:|
|Preparación CPU|0,955 ±0,005|4,281 ±0,019|4,48|
|Forward|1,515 ±0,027|1,616 ±0,019|1,07|
|Forward+loss+backward completo|3,903 ±0,044|5,108 ±0,065|1,31|

Parámetros26632→26701; backbone17477 en ambos. Aumento pequeño de parámetros
en esta fixture no implica igual VRAM: crecen salidas/gradientes y posiblemente
auxiliares. Host compartido con pruebas CPU; costes orientativos, no un benchmark
aislado ni evidencia de aceleración. No se midieron GPU, VRAM, precisión, bandas
ni OOD. No extrapolar este sistema diminuto a w90/6x6 u otros materiales.
Medición repetida después de las correcciones de revisión: `bench_cpu.json`
y `bench_cpu-final.log`; la medida previa permanece en
`bench_cpu-before-review.json`. La validación de soporte archivado añade coste
de preparación; se publican los valores finales, no los anteriores al fix.

## Revisión independiente y correcciones

Una revisión Native fresca (gpt-6-astra/high), reanudada con el mismo agente
tras interrupción de cuota; no una segunda revisión. Cero Critical, tres
Important y cero Minor. Informe original resumido en `final-review.md` del
workspace. Los tres hallazgos se revalidaron como Important por sus efectos,
no solo por el texto del diseño. Una única pasada de corrección:

| Hallazgo | Corrección mínima y regresión |
|---|---|
|Cero DM/EDM unilateral almacenado pasa Hermiticidad numérica; la orientación decide si se observa|Comprobar también soporte contra transposición con imágenes opuestas antes del packing. `test_archived_zero_support_must_be_hermitian`: DM/EDM, dirigido/media arista, molecular/periódico y ambas orientaciones. Rechazo, nunca rellenar ceros desconocidos.|
|Ajuste scaleshift proyecta NaN de una máscara parcial válida y contamina todos los canales|Fallar explícitamente antes de modificar buffers si hay entradas no observadas o no finitas; no imputar ni implementar otro estimador de coeficientes. `test_output_scaleshift_rejects_partial_labels_without_mutation`: NaN o placeholder finito en máscara falsa, lote incompleto posterior, estado idéntico, forward/loss/backward finitos.|
|Selector en logger legacy genera cabecera de seis columnas con filas de cuatro|Rechazar `matrix_target` en esa ruta antes de abrir/truncar fichero. `test_legacy_samplewise_selector_rejected_before_truncating`: conserva centinela y handles vacíos. La ruta legacy sin selector mantiene schema.|

`review-fixes-red.log`:11failed esperados; mismo comando después,
`review-fixes-green.log`:11passed/80deselected,187warnings,2.07s.
`review-fixes-focused-valid.log`:176passed,2653warnings,29.19s, con regresiones
del readout. Un primer comando focal apuntó a un módulo inexistente y salió4
sin ejecutar pruebas; no es un fallo funcional ni un gate aprobado.

La decisión sobre scaleshift limita una combinación, no el soporte general
de máscaras: `output_scaleshift=False` es el default intacto y admite pérdidas
parciales; `True` requiere etiquetas completas para su proyección de bloques.
Un estimador masked de coeficientes, especialmente con orbitales múltiples,
necesitaría otro contrato; no se sustituyó por una imputación a cero.

Los aspectos que el reviewer no juzgó se resuelven explícitamente en las
Rulings del ledger: suite nueva a cargo del autor; hardware/backends y precisión
científica no certificados; SK C1/C2 preservado; preflight no promete atomicidad
contra fallos arbitrarios de E/S; cambios previos y4.1 fuera de re-auditoría.
No menores diferidos.

## Reproducción

Desde graph2mat, reutilizar entorno existente sin instalar:

```bash
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=/tmp/g2m-server-tests.IcPjmO:/tmp/g2m-http-runtime.LOcBGk:/tmp/g2m-soap-deps.o0Kjat
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
G2M_PY=python
"$G2M_PY" -m pytest src -q -rs --tb=short --disable-warnings
"$G2M_PY" .superpowers/sdd/2026-10-18-multitarget-7-6/bench_cpu.py
```

Los temporales son dependencias existentes, no parte de una nueva instalación;
si no existen, informar del recurso ausente. El Python del otro entorno se usa
solo como herramienta; no se leen sus repositorios ni sus datasets.

## Límites e intervenciones no realizadas

- No campaña científica ni recurso H/S/DM/EDM compatible certificado en este
  alcance. El inventario limitado no demuestra ausencia absoluta de etiquetas.
- Sin defaults promovidos, GPU, colas/autorun, commits, worktrees, dependencias,
  artefactos congelados, otros proyectos o fonones modificados.
- SK conserva su límite C1/C2 anterior y no acepta el packing completo4/7;
  estándar y low-rank sí se verifican. Sin extensión SK/SOC por inferencia.
- Scaleshift requiere etiquetas completamente observadas. Con soporte parcial,
  dejar `output_scaleshift=False`; error temprano, no NaN ni cambio silencioso.
- Solo NetCDF por target nuevo-modo; HTTP genérico rechaza el modelo conjunto.
  No hay transacción de batch contra fallos arbitrarios de disco o del writer:
  la garantía demostrada es preflight de datos/formato/rutas antes de escribir.
- Samplewise nuevo-modo certifica MSE y H-gauge explícitas; no todos los metrics
  legacy arbitrarios. Plot selecciona un target, no mezcla unidades.
- El CSV se inicializa al primer uso, no en su constructor: permite rechazar
  un callback nuevo-modo inválido sin truncar un fichero existente.
- No solución del problema OOD ni mejora de precisión demostrada. S_pred puede
  ser no-SPD y requerir rechazar un espectro posterior, nunca repararlo aquí.

## Siguiente experimento propuesto, no autorizado ni lanzado

Primero autorizar un recurso con snapshots coincidentes H/S/DM/EDM reales,
bases/ocupaciones/gauges certificados y splits ID/OOD congelados. No reutilizar
sin prueba un baseline histórico de ΔH para comparar este modo de H directa.

Comparación mínima pareada: H-only directa frente a H/S/DM/EDM, semillas0/1/2,
mismo backbone, datos/splits, updates, selección de checkpoint y receta H.
Seis fits nuevos más evaluación; pesos fijados solo con train/validación ID,
nunca ajustados sobre OOD. Un control H+S intermedio sería otra ampliación a
aprobar, no un trabajo implícito. Antes del lote: piloto de coste/VRAM y techo
de presupuesto explícitos; aquí no hay ETA GPU fundada.

Medir H en gauge común, niveles/gap con S_ref para aislar la calidad de H y,
por separado, observables desplegables con S_pred. Registrar SPD de S_pred,
error S, DM, EDM, población total/atómica y energía referenciada; distinguir
energía real de la matriz exportada y contribuciones fuera del grafo. Parear
semillas, publicar resultados por caso y regresiones OOD, sin promoción por
bajar únicamente la propia loss. Se necesitan autorización de datos/GPU,
receta/pesos y presupuesto antes de preparar o encolar esa campaña.
