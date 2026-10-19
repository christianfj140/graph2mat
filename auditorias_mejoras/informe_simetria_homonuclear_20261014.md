# Perfil por pares y backward para 6.7 — 2026-10-14

Estado: ocho sondas principales terminadas y análisis recomputado. Alcance aprobado: medir la oportunidad de
6.7, no implementar una arquitectura distinta. Repositorio Graph2Mat, rama
`g2m-v2`; sin cambios de código de producto, defaults, datasets o cola.

## Pregunta y método

¿Cuánto trabajo de las operaciones de aristas corresponde a pares
homonucleares con `symm_transpose`, en forward y backward?

Se reutiliza `bench_setup_df.df_child`: fixture periódica sintética con especies
A (s+p+d, 9 orbitales) y B (s+p+f, 11), 16/72 puntos por grafo, batch de cuatro,
dos componentes simétricos, MACE de 16 canales, lmax3, correlación1 y dos
interacciones. Readout `E3nnLinearTSQNodeBlock`/`E3nnLinearEdgeBlock`. Pesos y
entradas iniciales reproducibles; Adam, fp32 y TF32 desactivado. No es una
validación de precisión científica ni un material TMD real/multi-zeta/SOC.

Diez warmups y treinta pasos medidos por proceso, cuatro rondas alternando
tamaños; después dos pasos instrumentados. Los tiempos/picos sin profiler se
separan de sus trazas. Eventos CUDA alrededor del forward y del backward
permiten medir sus intervalos completos, incluidos huecos de lanzamiento.
No confundir esos intervalos con suma de duración de kernels.

Durante el profiler se etiquetan por separado las operaciones A–A, A–B y B–B.
Se registran filas efectivas y aristas físicas: los pares homonucleares reciben
las dos direcciones apiladas, pero **una sola llamada**, como ya hacía el código.
Los heteronucleares no tienen ese `symm_transpose`.

Cada kernel se cuenta una sola vez. `External id` enlaza el kernel con su
operación CPU; los flujos explícitos `fwdbwd` enlazan la operación forward con
su backward, incluso en otro hilo de autograd. Esto evita el problema del
informe anterior: el scope CPU exterior de `loss.backward()` no incluye los
workers. No se suman tiempos inclusivos de categorías anidadas. Los kernels
sin enlace se registran como no atribuidos, nunca se imputan a un par.

Referencia metodológica: [integración de autograd en el profiler de PyTorch
2.11](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/profiler/README.md#autograd-integration).
La documentación confirma la correlación mediante secuencia/hilo; estas
trazas exportan sus endpoints `fwdbwd`, que usamos directamente para no
confundir números de secuencia pertenecientes a hilos distintos.

## Instrumentación auxiliar y verificaciones

Sonda desechable, no API de producto, en
`experimentos_speed_probe_symmetry_20261014_tools/profile_symmetry.py` y su
`test_profile_symmetry.py`. Artefactos ignorados por Git; no sustituye a una
implementación de 6.7. Cada ejecución conserva manifiesto de hashes de fuente,
helpers y runner, copia del runner, resultados, logs, traces y telemetría.

- RED inicial: faltaba el analizador; dos pruebas CPU fallan. GREEN: ambas
  pasan. Verifican correlación entre hilos, conteos literales sin duplicados,
  exclusión de `gpu_user_annotation` y rechazo de flujos incompletos.
- Prueba real CUDA con dos pares y matrices pequeñas: el forward y backward
  de ambos tienen atribución positiva; todos los kernels se contabilizan.
  Tres pruebas pasan bajo supervisor. El primer intento detectó únicamente
  una desigualdad estricta de sumas float (diferencia ~1e-18ms); se usa margen
  absoluto de 1e-9ms en esa comprobación contable, sin relajar equivalencia
  numérica del modelo ni tolerancias de predicción.
- Primer smoke del modelo falló por la instrumentación del constructor de
  Lightning y su inspección de hiperparámetros. Se retiró ese wrapper; los
  eventos se colocan en `MatrixMACE.forward`, sin modificar el constructor.
  Segundo smoke: ambos tamaños completan y conservan los tres pares esperados.
- Reutilizado el supervisor existente: allocator como máximo55%, reserva
  inicial30%, aborto del grupo propio si libre<25% o temperatura>=80°C.
  Comprobación de procesos CUDA ajenos antes de cada hijo; nunca detenerlos.

## Límites de interpretación

La fracción homonuclear medida es una oportunidad de cómputo, no ahorro ya
obtenido. Incluso si se pudiera eliminar una dirección sin cambiar la función,
el coste de proyectar/simetrizar pesos, gestionar canales y ejecutar backward
podría anularlo. Como las direcciones ya están apiladas, reducir filas no
elimina necesariamente kernels ni overhead de CPU. Una mitad del coste de la
operación tampoco equivale a una mitad del paso del modelo ni de su VRAM.

No se implementa ni se aprueba por estas medidas ninguna parametrización de
pesos nueva. Una implementación posterior requiere diseño, demostración de
equivalencia de salidas/gradientes y benchmark pareado end-to-end.

## Resultados definitivos

Artefactos: `experimentos_speed_probe_symmetry_20261014_v1/`. Ocho procesos
completos, cuatro rondas por tamaño; 10 warmups, 30 pasos medidos y 2 pasos de
profiler por proceso. `summary.json` se genera con la sonda auxiliar
`experimentos_speed_probe_symmetry_20261014_tools/analyze.py`.

El análisis volvió a parsear las ocho trazas y verificó hashes de todos los
archivos del manifiesto, copia del runner, tiempos finitos, número de pasos,
un único stream de kernels por trace, y suma contable frente a todos los
kernels crudos. Pesos iniciales y entradas coinciden entre rondas y con las
sondas d/f del informe anterior; no se cambió el modelo para obtener estos
porcentajes. Los scopes de par cubren **la operación TP+lineales**, no apilado,
promedio/transpuesta, cambio de base ni el backward anterior de MACE.

### Fracción homonuclear

Mediana de cuatro razones de ronda. Cada denominador es la suma de duraciones
de kernels correspondiente, **no tiempo de pared**; el total incluye forward,
backward y otros kernels del paso (pérdida/optimizador).

| Puntos por grafo, batch4 | Homonuclear / forward | Homonuclear / backward | Homonuclear / todos los kernels | Todas las operaciones de arista / todos los kernels |
|---|---:|---:|---:|---:|
| 16 | 24,71% | 16,82% | 19,56% | 32,16% |
| 72 | 27,83% | 17,89% | 21,65% | 38,38% |

Rangos entre rondas del porcentaje homonuclear total: 19,552–19,563% y
21,647–21,656%. Es repetibilidad de cómputo sobre los mismos datos/pesos
iniciales, no un intervalo de confianza de precisión ni semillas científicas.

### Desglose por par

Tiempos de kernels por paso, mediana de cuatro rondas; cada trace contiene dos
pasos y se divide por dos. Las aristas incluyen imágenes periódicas y los cuatro
grafos del batch. Para A–A/B–B, las filas efectivas son el doble de las aristas
físicas por el apilado de direcciones. A–B no aplica ese apilado adicional.

| Puntos por grafo | Par | Aristas físicas | Filas efectivas | Forward ms | Backward ms |
|---|---|---:|---:|---:|---:|
| 16 | A–A | 32 | 64 | 0,983 | 1,245 |
| 16 | A–B | 128 | 128 | 1,344 | 1,524 |
| 16 | B–B | 32 | 64 | 1,015 | 1,207 |
| 72 | A–A | 144 | 288 | 1,746 | 1,933 |
| 72 | A–B | 576 | 576 | 3,112 | 2,696 |
| 72 | B–B | 144 | 288 | 1,958 | 1,881 |

Cobertura del enlace kernel→CPU: 100% en las trazas de16 puntos; al menos
99,9988% del tiempo de kernels en las de72. En estas últimas queda un kernel
de0,8µs por trace sin `External id`, conservado en la categoría no atribuida.
No se inventa su propietario ni se omite del denominador total.

### Intervalos y memoria de la referencia

Estas cifras describen el modelo actual instrumentado con eventos CUDA,
**no son una comparación antes/después de implementar6.7**. Forward excluye
la pérdida; backward incluye el intervalo de `loss.backward()`. Los intervalos
incluyen huecos entre lanzamientos, a diferencia de las sumas de kernels.

| Puntos por grafo | Pared ms/paso (MAD) | Forward evento CUDA ms | Backward evento CUDA ms | Allocated / reserved MiB |
|---|---:|---:|---:|---:|
| 16 | 110,35 (0,33) | 33,61 | 72,24 | 577,38 / 612 |
| 72 | 110,75 (0,22) | 33,37 | 72,28 | 2509,80 / 2682 |

Suma de kernels por paso bajo profiler: 22,76/34,72ms para16/72 puntos.
Proviene de pasos distintos a los intervalos sin profiler: no restarla de
110ms para certificar un desglose del tiempo de CPU, ni derivar de ella una
cota end-to-end exacta. Tampoco comparar estos tiempos absolutos con el informe
anterior como si se hubiese optimizado el modelo.

## Conclusión y siguiente decisión

**6.7 tiene una oportunidad de cómputo no despreciable en esta fixture**, a
diferencia del cambio de base6.5: las operaciones homonucleares ocupan cerca
de20–22% de la suma de tiempos de kernels del paso, incluyendo backward.
Esto justifica estudiar un diseño equivalente acotado, no promoverlo ni
dar por conseguida una aceleración.

Como escenario meramente aritmético, ahorrar exactamente la mitad de ese
coste equivaldría a9,78%/10,83% de la suma de kernels. **No es una predicción
ni una cota demostrada del tiempo de pared**: reducir filas no garantiza
reducir el tiempo GPU a la mitad; los lanzamientos ya están fusionados y la
proyección de pesos tiene coste propio. No hay ahorro de velocidad o VRAM
medido para6.7, porque todavía no existe esa implementación.

Siguiente paso propuesto: diseño específico para `E3nnLinearEdgeBlock`
homonuclear, con prueba algebraica del intercambio/transpuesta de coeficientes
y pesos, compatibilidad de checkpoints, y comparación salidas/gradientes antes
de un benchmark pareado. No ampliar a readouts condicionados/low-rank/SK por
anticipación. La parametrización concreta y su implementación quedan sin
aprobar; estas sondas no autorizan el cambio de arquitectura.

## Cierre de verificación y recursos

- Gate CPU de los analizadores y supervisores: **18 pasan, 1 omitido** (CUDA
  desactivada). Gate CUDA pequeño bajo supervisor: **3 pasan**.
- Ocho hijos principales, dos smokes corregidos y análisis final terminan con
  exit0. El primer smoke fallido y la primera comprobación contable fallida
  permanecen conservados, sin mezclarlos con las medidas definitivas.
- Telemetría principal: 384 muestras, mínimo global libre **88,214%**, máximo
  **45°C**, ningún aborto. GPU al terminar:0% de uso,32096MiB libres.
- No se cambió código de producto; no se recompilaron wheels ni se repitió la
  suite general en esta tarea diagnóstica. Su último estado y sus fallos están
  registrados en `informe_setup_hs_df_20261014.md`; no se presentan como un
  nuevo gate verde. Tampoco se certificó precisión científica, otra arquitectura
  ni materiales reales a partir de estas sondas.
