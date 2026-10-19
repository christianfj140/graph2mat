# Preparación H+S y perfil d/f — 2026-10-14

Estado: parche, medidas, gates focalizados y revisión independiente terminados;
suite general no verde por los motivos registrados abajo. Alcance aprobado en el chat: optimización equivalente del
arranque H+S y perfil diagnóstico d/f. Graph2Mat/g2m-v2, base270d17d. Sin
defaults/colas/recetas/splits nuevos. Cambios previos del usuario preservados.

## Parche acotado

`_csr_to_block_dict_components` resolvía los duplicados con `np.unique` dentro
de cada bloque. Ahora agrupa una vez, ordena establemente por grupo/fila/columna
y conserva la última entrada de cada coordenada. Usa los primeros índices de
los grupos originales para conservar el orden de inserción del diccionario,
incluso si una entrada temprana queda sustituida por un duplicado posterior.
No cambia el kernel escalar Cython, ni la reconstrucción inversa ni los vecinos.

RED: seis tests nuevos detectan seis llamadas a unique donde se exige una;
GREEN:95 tests sparse, incluidos valores literales/oráculo por entrada, fill
NaN/entero, dtype real/complejo, capacidad sisl, fantasmas, imágenes periódicas,
duplicados y orden. Python3.9/Torch2.4.1 CPU: los mismos95 pasan.

## Método de medida

Runner `bench_setup_df.py`, tests `test_bench_setup_df.py`. Referencia exacta del
método desde Git270d17d, ejecutada en un namespace separado y activada solo en
el hijo baseline. Misma configuración congelada H+S de w90/6x6, caché segura,
cero workers, mismas etiquetas y hashes de todos los tensores train/val.
Cada variante/ronda usa proceso propio; comparar constructor+setup, no imports,
hashing ni teardown. Cuatro rondas en orden equilibrado. Memoria before/after
RSS/PSS y máximo RSS del proceso, incluida distinción entre pico total e
incremento sobre imports. Caché del sistema operativo no fría; pruebas CPU
concurrentes de dos hilos limitan extrapolar tiempos absolutos.

El profiler d/f reutiliza la fixture periódica de pruebas: especies sintéticas
con s+p+d (9 orbitales) y s+p+f (11),16/72 puntos,4 grafos por batch,C2 simétrico.
MACE16canales,lmax3,corr1,dos interacciones y readouts lineal+TSQ/lineal.
No se inventa dataset científico: etiquetas sintéticas para trabajo de cómputo.
Es un diagnóstico, no validación en TMD real, bases múltiples zeta o SOC.

Pasos sin instrumentar:10warmups+30medidos,4rondas en procesos propios, fp32,
TF32 apagado, Adam. Guardar picos allocated/reserved antes de perfilar. Después
dos pasos de profiler con anotaciones de forward/readout/node/edge/change-of-basis,
backward y Adam; scopes inclusivos, no sumar categorías solapadas ni interpretar
tiempo de kernel como tiempo de pared. No entrenamientos largos ni comparación
de precisión. Límite allocator/supervisor reutilizado de `bench_pipeline_gpu.py`:
objetivo20% libre,30% de margen inicial, aborto propio<25% o>=80°C, cleanup del
grupo propio con TERM/KILL; sin interferir con procesos ajenos.

## Smoke y corrección del parser

Smoke de4procesosCPU+2GPU termina. Setup6x6 preliminar2245→1849ms; no es
conclusión final. Torch2.11 emite entradas con el mismo nombre para CPU y
`gpu_user_annotation`. El primer parser sobrescribía scopesCPU con CUDA y daba
porcentajes imposibles. Trace confirma ambas categorías; test específico RED
reproduce sobrescritura, GREEN filtra `device_type=CPU` y usa su GPU time
agregado. No se usarán las cifras de perfil de ese smoke como resultado.

Artefactos definitivos en `experimentos_speed_probe_setup_df_20261014_v1`:
manifiesto, runner congelado, JSON/log por proceso y Chrome traces. Hashes de
source/runner/entradas y extensiones nativas conservados. Ningún paquete del
entorno compartido se instala ni actualiza.

## Registro final de verificación

- Fuente:95 pruebas sparse pasan; floor3.9:95 pasan.
- Runner:2 pruebas pasan (emparejamiento de outputs y parser CPU/CUDA).
- Gate final con archivos estabilizados: **161 pasan**, 67,74s, incluyendo
  95 pruebas sparse, 16 de ambos runners y 50 de raíz. Registro en
  `/tmp/g2m-setup-df-final-gate.log`; CUDA desactivada y dos hilos CPU.
- Suite completa source+root+runner: **2550 pasan, 25 omitidos, 5 fallan**,
  857,18s, en `/tmp/g2m-setup-df-full-suite.log`. Cuatro fallos son los casos
  HistorySOAPs ya conocidos por ausencia de `dscribe`. El quinto corresponde
  al test del parser: la suite había recogido su versión anterior, con eventos
  ficticios sin `device_type`, antes de corregir el parser y actualizar el test
  durante esa ejecución. El traceback muestra ese atributo ausente. La versión
  final del test pasa en el gate nuevo, además de la comprobación GPU real
  `parser_check`. No se presenta la ejecución general como verde ni se suman
  ejecuciones distintas para inventar un nuevo total de suite completa. No se
  volvió a ejecutar toda la suite después de estabilizar el runner; el código
  de producto no cambió durante aquella ejecución.
- Builds aislados offline con constraints existentes, wheels3.9/3.12 completas:
 95pruebas sparse instaladas pasan en cada una, desde fuera del checkout y con
 su propia extensión nativa. Dependencias reutilizadas en solo lectura, wheel
 propia primera en PYTHONPATH, finder editable desactivado. No resolución nueva
 de todas las dependencias. Entornos temporales propios de la entrega anterior.
- No nuevos cambios d/f en el producto: primero medir y recomendar.
- Revisión independiente final: **0 hallazgos críticos, importantes o menores**.
  El revisor reprodujo 95 pruebas sparse y las 2 del runner; comprobó hashes,
  resúmenes, resultados de los procesos CPU/GPU y límites del supervisor. No
  reconstruyó las wheels ni certificó precisión científica o el backward.
- `git diff --check` sin errores. GPU al cierre sin carga de las sondas:
  0% de utilización, 32096MiB libres, 36°C. Ningún default o trabajo encolado
  cambiado, ninguna dependencia del entorno compartido modificada.

Wheels instaladas verificadas (95 pruebas sparse por perfil):

- Python3.12: `651df857342b3cd8d1af0a0906e011976e8d2450ea907e1b1d0beca3487e954c`.
- Python3.9: `38a9257c1f3cb0df212caad8fc5570c0d4d87b81d8e25a40beb2e34db9c780dd`.

## Resultados finales de setup

16 procesos completos,4rondas×2variantes×2sistemas. Resumen recalculado coincide
con JSON; mismos hashes de outputs de todos los tensores train/val en cada par.
No se cronometran hashes ni comparaciones de equivalencia.

| Datos | Original ms | Parche ms | Razón pareada (MAD) | Pico RSS original/parche MiB | Incremento RSS de setup original/parche MiB |
|---|---:|---:|---:|---:|---:|
| w90 | 844,44 | 829,26 | 0,9761 (0,0094) | 902,94 / 902,65 | 48,41 / 48,40 |
| 6x6 | 2273,38 | 1873,06 | 0,8267 (0,0043) | 918,16 / 917,60 | 63,63 / 63,41 |

Medianas de ronda y de razones pareadas, no necesariamente cociente de las
medianas. Las cuatro razones favorecen candidato en ambos sistemas. **−2,4%w90
y−17,3%6x6 en preparación H+S**, sin ahorro de RAM demostrado. PSS observada
varía con cuántos procesos comparten páginas; no interpretar su dispersión como
ganancia del algoritmo. La estructura de arrays temporales cambia, pero el
pico del proceso queda prácticamente igual. CPU-only: ninguna afirmación de
VRAM, precisión o aceleración por paso de entrenamiento.

## Perfil d/f definitivo

Ocho procesos,16/72puntos×4rondas,4grafos por batch,10warmups/30pasos sin profiler.
Pesos iniciales y entradas reproducibles por tamaño, código y runner coinciden
con manifiesto al terminar; runner medido archivado. Dos pasos de profiler por
proceso separados del tiempo/pico principales; categorías CPU normalizadas.

| Puntos por grafo | ms/paso (MAD) | Allocated / reserved MiB | Cambio de base nodo+arista / GPU forward | Operación aristas / GPU forward |
|---|---:|---:|---:|---:|
| 16 | 114,62 (0,37) | 577,38 / 612 | 0,267% | 41,34% |
| 72 | 115,18 (0,17) | 2509,80 / 2682 | 0,177% | 51,27% |

Los porcentajes son tiempos de kernels inclusivos atribuidos al forward, **no
porcentajes del paso total** ni cotas de mejora end-to-end. Hay mucho tiempo de
lanzamientos/CPU; no equiparar porcentaje de kernels con porcentaje de pared.
El coste de TP/operaciones del readout domina al cambio de base en esta fixture.
No justifica implementar6.5 sparse por anticipación. 6.7 podría merecer perfil
por par/tipo y diseño, pero la fracción homonuclear ahorrable y su backward no
están aislados: no prometer mitad de FLOPs ni una ganancia de paso determinada.

**Límite del backward:** la anotación CPU de `g2m/backward` no contiene todos los
kernels lanzados por workers de autograd y devuelve GPU time casi cero. Ese
valor de los artefactos v1 **no se usa como medida de backward**. El parser actual
lo marca `None`, con test RED→GREEN; su tiempo CUDA global sin instrumentar sí
incluye forward+backward+Adam. Repetición pequeña `parser_check` valida el nuevo
campo sin reinterpretar los tiempos crudos originales. No se ha certificado un
desglose GPU completo del backward.

Mínimo global libre88,214%, máxima45°C, ningún aborto del supervisor. Son
medidas de una fixture, no de TMD real, bases multi-zeta grandes u otras recetas.
Se conservan las trazas (aproximadamente100MiB cada una) para inspección; no
modificaciones de arquitectura ni entrenamientos científicos.
