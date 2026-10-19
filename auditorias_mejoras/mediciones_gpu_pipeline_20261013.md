# Medición GPU del pipeline: 5.3, 6.1, 6.2 y 6.4

Estado a 2026-10-14: comparación v2 completada, verificada y revisada.
Repositorio `graph2mat`, rama `g2m-v2`, HEAD `270d17d`. El usuario autoriza
utilizar la GPU compartida conservando al menos un 20 % de VRAM libre.

## Alcance y decisiones previas

La métrica primaria es tiempo de pared del trabajo completo, incluyendo lectura,
collate, transferencia y cálculo. Se informan por separado tiempos diagnósticos
con eventos CUDA, carga CPU y reconstrucción sparse. Inicialización y primer paso
se separan del régimen estable. Los picos allocated/reserved incluyen parámetros,
gradientes y Adam; cada variante/ronda tiene un proceso propio.

- **6.1:** datos reales congelados w90/6x6 y receta de la campaña anterior. Acceso
  histórico a snapshots sin copia, acceso seguro actual y dataset lazy. Comparación
  principal con cuatro workers; diagnóstico adicional con cero workers para
  separar aislamiento/copia de prefetch. Cuatro workers usan `pin_memory=True`;
  cero workers, `pin_memory=False`. Comparar estos dos regímenes cambia ambos
  ajustes; comparar safe/legacy dentro de cada régimen aísla las copias.
  El acceso histórico es una referencia de
  rendimiento de solo lectura, no una alternativa segura para usuarios.
- **6.2:** misma inferencia GPU y reconstrucción CPU a CSR, sustituyendo solamente
  las dos funciones Python vectorizadas por sus definiciones Git previas exactas.
  Medir también setup: el componente H+S usa la conversión CSR multicomponente.
  La ruta escalar compilada permanece idéntica. Esta optimización no modifica
  kernels de forward/backward; no se atribuirá ahorro de VRAM sin evidencia.
- **6.4:** mismo orden y mismas muestras por época; batching fijo frente a presupuesto
  de coste. Grafeno de tamaño uniforme funciona como control negativo. Fixture
  heterogénea periódica s/p/d/f de 12 grafos, 4–128 puntos, ya usada en CPU, permite
  medir el pico con tamaños distintos. La unidad primaria es una época completa,
  porque cambian número de batches y updates; esta prueba no demuestra equivalencia
  de optimización ni precisión.
- **5.3:** mismo modelo y datos con MSE H-only, flag apagado/encendido; comprobar
  pérdida y todos los gradientes en CUDA antes de cronometrar. Medir además el
  control gauge H+S con ambos flags: su fallback conocido no ofrece aceleración
  de coeficientes. No cambiar la pérdida de las recetas científicas existentes.

Semilla fija 20261013, fp32, TF32 desactivado, sin compile/checkpointing, dos hilos
por proceso. Orden invertido en la segunda ronda de cada pareja y rotado solo
entre parejas: cada variante precede a cada otra exactamente dos veces en cuatro
rondas. Los hashes de pesos iniciales
y tensores de entrada deben coincidir dentro de cada comparación pertinente.
Conservar muestras crudas, mediana, MAD, min/max y razones pareadas. Cuatro rondas
son una primera medida de rendimiento, no un ensayo de precisión de entrenamiento.

Ruling: se reutilizan los kernels Cython instalados del entorno CUDA, cargados
explícitamente, y el código Python del checkout actual. Evita medir por accidente
la alternativa escalar Python al retirar el finder editable. Se registran ruta y
SHA256 de ambos binarios; ningún paquete del entorno compartido se modifica.

## Recursos y límites

La GPU tiene contextos ajenos presentes. Un supervisor propio consulta cada
0,5 s memoria global, utilización, temperatura, clocks y potencia. El allocator
del hijo se limita al menor de 55 % de memoria total y memoria libre inicial
menos 30 % del total. El supervisor termina únicamente su propio grupo de procesos
si la memoria libre baja del 25 % o la GPU alcanza 80 °C. El margen del 25 % anticipa
el suelo solicitado del 20 %; ningún monitor puede impedir una asignación
instantánea realizada por otro usuario. Se registra el mínimo observado.

Los tiempos con contextos ajenos se presentan con esa limitación. Orden alterno,
dispersión y telemetría permiten valorar estabilidad, pero no garantizan exclusividad
del dispositivo. No se modifica la cola ni se detienen procesos ajenos.

## Fuentes del método

- [Semántica CUDA y ejecución asíncrona de PyTorch](https://docs.pytorch.org/docs/stable/notes/cuda.html).
- [Reset de picos del allocator](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.reset_peak_memory_stats.html).
- [Límite de memoria por proceso](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.set_per_process_memory_fraction.html).

## Ejecución

Runner: `auditorias_mejoras/bench_pipeline_gpu.py`. Artefactos aislados e ignorados
por Git en `auditorias_mejoras/experimentos_speed_probe_pipeline_20261013_*`.
Primero smoke con una ronda, un warmup y dos unidades medidas; después cuatro
rondas, diez warmups y treinta unidades medidas. Para 6.4 una unidad es una época.

```bash
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python \
  auditorias_mejoras/bench_pipeline_gpu.py run \
  --out auditorias_mejoras/experimentos_speed_probe_pipeline_20261014_v2 \
  --rounds 4 --warmup 10 --steps 30
```

## Registro de ejecución

- Las pruebas CPU del presupuesto de VRAM, análisis pareado y orden equilibrado
  para 2/4/5 variantes pasan: 6 tests.
- Smoke de caché real w90/6x6, reconstrucción sparse, batching homogéneo/heterogéneo
  y coeficientes completado. Oráculo de reconstrucción exacta sobre la misma
  predicción en cada proceso. Los hashes de matrices de forwards CUDA distintos
  no se utilizan como prueba de igualdad bit a bit entre procesos.
- 5.3 CUDA fp32, pérdida MSE: ruta de coeficientes activa, pérdida igual a la ruta
  matricial dentro de tolerancia; error máximo absoluto de gradiente
  `9,3132e-10` en w90 y `8,7311e-11` en 6x6. Gauge confirma fallback en ambos.
- Mínimo de memoria libre en los smokes: 72,58 % del total.
- Antes de congelar el runner se liberaron explícitamente los temporales de la
  comprobación numérica y se vació una sola vez la caché CUDA, antes del warmup.
  Así el oráculo previo no infla la memoria reserved de la variante certificada.
- La primera comparación completa `experimentos_speed_probe_pipeline_20261014_v1`
  se interrumpió: el orden rotado e invertido cada ronda se cancelaba con dos
  variantes y no equilibraba todas las parejas. Sus cifras no se emplearán para
  las conclusiones finales. Se conservan artefactos para trazabilidad.
- Repetición corregida iniciada el 2026-10-14 en
  `experimentos_speed_probe_pipeline_20261014_v2`: cuatro rondas, diez warmups y
  treinta unidades medidas. El manifiesto congela hash del runner y del source.
  También se registran pérdidas de todas las unidades y se comprueba finitud de
  parámetros/gradientes finales fuera del tiempo medido y después de capturar picos.
- Entorno: Python 3.12.3, Torch 2.11.0+cu130, e3nn 0.4.4, MACE 0.3.15,
  Lightning 2.6.1, PyG 2.7.0, NumPy 2.4.4, SciPy 1.17.1 y sisl 0.16.4;
  RTX 5090, driver 595.84, 32607 MiB según nvidia-smi.

Verificación final: 112 registros completos, 28 grupos de cuatro rondas, 30 unidades
medidas cada uno. Pérdidas finitas, rutas esperadas, oráculos sparse, cobertura de
muestras y mismos binarios nativos. Hash del runner/source coincide con el manifiesto.
Seis tests CPU pasan. Revisión independiente sin hallazgos Critical/Important.
La tanda v2 duró 1706,29 s (**28 min 26 s**), sin contar preparación/smokes/v1.
Mínimo global libre **75,824 %**, máxima temperatura **56 °C**; ningún aborto del
supervisor. Terminado el runner, GPU de nuevo al 0 % con 32096 MiB libres.

## Resultados v2 finales

Cuatro rondas por variante, 30 unidades medidas tras 10 de warmup. Los tiempos
de la tabla son medianas de las medianas por ronda. La razón es la mediana de
las cuatro razones pareadas, por lo que no siempre coincide con el cociente
de los tiempos tabulados. MAD = desviación absoluta mediana entre rondas.

### 6.1: seguridad de la caché y carga

| Datos / régimen | Histórico ms/paso | Seguro ms/paso | Razón pareada (MAD) | Pico allocated / reserved, ambos |
|---|---:|---:|---:|---:|
| w90, 4 workers, pinning | 131,43 | 134,44 | 1,011 (0,020) | 698,90 / 858 MiB |
| 6x6, 4 workers, pinning | 174,66 | 171,38 | 0,981 (0,026) | 5117,68 / 6540 MiB |
| w90, 0 workers, sin pinning | 53,88 | 57,34 | 1,061 (0,002) | 698,90 / 858 MiB |
| 6x6, 0 workers, sin pinning | 121,99 | 123,17 | 1,011 (0,001) | 5117,68 / 6542 MiB |

Con cuatro workers no hay cambio de velocidad concluyente frente al acceso
histórico: dispersión y rangos pareados incluyen ausencia de cambio. Sin workers
se ve el coste de seguridad: aproximadamente +6,1 % en w90 y +1,1 % en 6x6.
No hay ahorro de VRAM. El acceso histórico no preserva aislamiento al mutar una
muestra; no se recomienda recuperarlo.

La caché existente sí evita la carga repetida: lazy con cuatro workers da 183,28
ms/paso en w90 y 431,99 en 6x6, frente a 134,44 y 171,38 con caché segura. Esta
comparación cuantifica una opción que ya existía, no una aceleración nueva de 6.1.

El régimen combinado de cero workers sin pinning es mucho más rápido aquí que
cuatro workers con pinning. No atribuir toda la diferencia al número de workers ni
extrapolarla a datasets grandes/lazy. Las recetas guardadas permanecen intactas.

### 6.2: forward GPU sin gradientes y reconstrucción CSR

| Datos | Histórico ms/unidad (MAD) | Vectorizado ms/unidad (MAD) | Razón pareada (MAD) | Pico allocated / reserved, ambos |
|---|---:|---:|---:|---:|
| w90 | 39,48 (0,05) | 30,65 (0,16) | 0,776 (0,007) | 187,30 / 294 MiB |
| 6x6 | 116,04 (0,27) | 37,73 (0,18) | 0,326 (0,002) | 1402,19 / 2618 MiB |

Mejora reproducida de aproximadamente **22,4 % en w90 y 67,4 % en 6x6** para
forward + transferencia de salida + reconstrucción CSR. No es una aceleración
de los kernels GPU ni del paso de entrenamiento. Forward CUDA queda en
12,51/12,53 ms (w90) y 26,89/26,85 (6x6), histórico/vectorizado; la reconstrucción
CPU baja de 21,35 a 12,54 ms y de 84,95 a 7,00, respectivamente. No cambia VRAM.

Hay un coste de inicio real: setup H+S pasa de 747,06 a 778,41 ms en w90 y de
1739,92 a 2040,07 en 6x6 (+4,2 %/+17,3 % por cociente de medianas). Incluye
preprocesado, construcción de datasets y snapshots con caché del SO caliente;
no se oculta esta regresión bajo el resultado favorable del writer.

Las pérdidas, reconstrucción y hashes se comprueban fuera del tiempo medido.
`first_unit_ms` significa primera unidad tras el preflight, no arranque CUDA
completamente frío. No se ha cronometrado por separado toda la construcción del
modelo. La RAM registrada es RSS del proceso principal, no suma de workers ni
pico total del host. No hay evaluación nueva de precisión, AMP o DDP GPU.

### 6.4: batching por coste

| Datos | Fijo ms/época (MAD) | Coste ms/época (MAD) | Razón pareada (MAD) | Allocated fijo → coste | Reserved fijo → coste |
|---|---:|---:|---:|---:|---:|
| w90 | 117,41 (2,85) | 116,55 (0,49) | 0,990 (0,022) | 698,90 → 698,90 MiB | 858 → 858 MiB |
| 6x6 | 123,08 (0,07) | 123,28 (0,07) | 1,002 (0,002) | 5117,68 → 5117,68 MiB | 6542 → 6542 MiB |
| Fixture heterogénea | 378,65 (1,84) | 856,88 (4,66) | 2,264 (0,004) | 2783,28 → 1596,13 MiB | 3130 → 1754 MiB |

En grafeno uniforme conserva exactamente los mismos lotes: sin ahorro de VRAM y
tiempos compatibles con ausencia de beneficio práctico. En la fixture heterogénea
el ahorro es **42,65 % allocated y 43,96 % reserved**, a cambio de **2,26× tiempo
por época**. El presupuesto 100656 permite singleton para la estructura de 128
puntos que lo excede. Las doce muestras aparecen una vez por época en ambos casos.

El fijo hace tres updates por época y el presupuesto, siete: 90 frente a 210 en
las 30 épocas medidas, siempre 360 muestras. No son trayectorias de optimización
equivalentes ni un experimento de precisión. La reducción demuestra que el control
de coste puede recortar picos con grafos desiguales; no demuestra ahorro en cualquier
material ni un presupuesto óptimo. Se mantiene como opción para ajustar memoria.

### 5.3: entrenamiento en coeficientes

Régimen de esta comparación: cero workers, sin pinning, targets preparados en cada
collate CPU. MSE usa H-only; gauge conserva H+S y predice H, como en la campaña.

| Datos / pérdida | Matriz ms/paso (MAD) | Flag activado ms/paso (MAD) | Razón pareada (MAD) | Allocated matriz → flag | Reserved matriz → flag |
|---|---:|---:|---:|---:|---:|
| w90, MSE | 57,39 (0,70) | 73,33 (1,30) | 1,282 (0,019) | 698,84 → 693,94 MiB | 858 → 858 MiB |
| 6x6, MSE | 121,75 (0,10) | 235,82 (0,52) | 1,938 (0,003) | 5118,41 → 5115,20 MiB | 6542 → 6732 MiB |
| w90, gauge | 57,56 (0,13) | 58,02 (0,33) | 1,008 (0,008) | 698,90 → 698,90 MiB | 858 → 858 MiB |
| 6x6, gauge | 123,13 (0,05) | 123,24 (0,04) | 1,000 (0,0005) | 5117,68 → 5117,68 MiB | 6542 → 6542 MiB |

La ruta MSE es correcta en CUDA fp32: pérdida y todos los gradientes cotejados
contra la reconstrucción matricial, error máximo absoluto **6,985e-10** en v2.
Pero aquí **regresa 28,2 % en w90 y 93,8 % en 6x6**. Allocated baja solo 0,70 %
y 0,063 %; reserved de 6x6 sube 2,9 %. No es una optimización recomendable en
este régimen.

La mayor diferencia está en preparar targets durante collate: carga/collate CPU
pasa de 5,03 a 18,58 ms en w90 y de 3,39 a aproximadamente 116,2 ms en 6x6.
Forward/loss CUDA se mantiene aproximadamente en 16,71/17,64 ms y 28,10/28,25;
backward, 32,90/34,34 y 89,06/89,78, matriz/coeficientes. El generador de targets
recorre y valida los bloques en CPU (`coefficient_targets.py`, `__call__`).
Una evaluación posterior de caché de targets o de otros regímenes de workers
debe ser independiente; no se presupone que produzca una ganancia GPU.

Gauge confirma fallback matricial en todos los casos: diferencias de tiempo
pequeñas, sin cambio de memoria y sin aceleración en coeficientes. No modifica
las recetas de precisión ni el default del flag.

## Conclusión y alcance pendiente

- **6.2:** beneficio favorable y estable en reconstrucción CSR tras forward GPU;
  conservar la advertencia del setup H+S más lento y VRAM intacta.
- **6.4:** beneficio VRAM demostrado solamente en la fixture heterogénea, con
  coste temporal importante; precisión con batching distinto sin evaluar.
- **6.1:** aislamiento correcto con pequeño coste visible en CPU síncrona;
  no atribuir ahorro de memoria GPU ni una caché nueva a estas modificaciones.
- **5.3:** certificado CUDA fp32 correcto, pero rendimiento desfavorable con
  proyección CPU síncrona y ausencia de beneficio con gauge. Sigue opt-in.

Quedan fuera AMP, DDP CUDA, restauración/fit completo con Trainer CUDA, precisión
de entrenamientos largos, pico agregado de RAM de todos los workers, arranque
completamente frío, máximo batch admisible, I/O del archivo y cálculo SCF/H0.
Estas limitaciones no se presentan como mediciones completadas. No se cambiaron
defaults, recetas conservadas ni la cola de autorun.
