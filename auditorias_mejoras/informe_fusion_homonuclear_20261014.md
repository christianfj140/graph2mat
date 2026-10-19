# 6.7 — Fusión opt-in de pesos homonucleares

Implementación y mediciones:2026-10-14–15. Cierre documental:2026-10-15.

## Alcance y referencia

Trabajo en `graph2mat/g2m-v2`, HEAD inicial
`270d17d1f204293a059baa09e8f1b3756981e2bf` con cambios anteriores sin commit.
El diff inicial está conservado en
`.superpowers/sdd/2026-10-14-homonuclear-weight-fusion-6-7/baseline.patch`.
Diseño y plan aprobados del 2026-10-14, ejecución Native. Producto limitado a
`edge_operations.py` y `matrixblock.py`; sin cambios de defaults, datos, cola,
dependencias compartidas ni campañas científicas.

La opción existente de configuración se usa así:

```yaml
model:
  edge_block_readout_kwargs:
    fuse_homonuclear: true
```

Por defecto es `False`. Solo se certifica la clase exacta
`E3nnLinearEdgeBlock`, simetrización de extremos, fórmula `ij`, Q fijo y
compatible y ortonormal, sin output scale/shift. Pesos efectivos: `(W + R W)/2`, con
intercambio de canales de entrada, permutación/signo de salida y fase CG.
La evaluación utiliza N filas en vez de 2N, sin cambiar los Parameters
originales, sus nombres, su orden ni los estados de Adam. Los mapas derivados
son buffers no persistentes; no se cachean pesos efectivos entre pasos.

El certificado se repite después de cargar Q y los hijos. Copias/deepcopy no
retienen callbacks asociados al objeto original. Autocast dentro del readout,
pesos puros fp16/bf16, operaciones personalizadas, affine y mapas incompatibles
conservan la ruta anterior. AMP exterior se certifica con `readout_fp32=True`.
La mutación arbitraria directa de Q fuera de constructor/carga no constituye
una API de recertificación automática.

## Verificación

- Referencia CPU previa:147 tests pasan.
- Operación: escalar literal24.75, fase CG negativa, cinco bases s/p/d/f,
  segmentos repetidos, componentes1/2, N0/1/3, uno/dos orígenes; salidas y
gradientes a inputs y pesos originales. Gradcheck float64.
- Tres APIs comunes; hooks de TP demuestran N filas frente a2N.
- Fallbacks, Q incompatible/cargado/entrenable, dtype, deepcopy y claves
  estrictas de checkpoint. Perturbación Q float64 de1e-8 rechazada mediante
  tolerancia del dtype, no tolerancia fija fp32.
- Integración CPU:204 pasan/10 omitidos; modo single/sum/mean, agrupaciones,
  layout/sin layout, checkpointing, coeficientes sin reconstrucción densa,
  O(3), intercambio de extremos, checkpoint Lightning y Adam con historia
  no vacía y weight decay.
- Fuente final Python3.9:204 pasan/9 omitidos. Wheels finales Python3.9/3.12:
  265 pasan/9 omitidos cada una. Ruff sin errores; mypy23 archivos
  sin errores, usando herramientas aisladas ya existentes.
- CUDA obligatoria:8 pruebas pasan (fp32,fp16-mixed,bf16-mixed; comparación
  de valores/pérdidas/todos los gradientes, Trainer fit/validación/resume y
  fallback de pesos puros de baja precisión). Incluida compilación real:
  **9 pruebas CUDA pasan**, repetidas después de la corrección de revisión en99,41s.
  Primera preparación/forward/backward compilados306,05s; no se contabilizan
  como tiempo por paso ni como ganancia de la fusión.

Las tolerancias iniciales permanecen: float64 rtol1e-10/atol1e-12;
fp32 rtol5e-5/atol5e-6; AMP rtol1e-2/atol1e-3. Los logs CUDA guardan errores
por tensor, no solo una bandera de éxito.

## Empaquetado

Builds offline con constraints existentes, instalados exclusivamente en
venvs propios de `/tmp/g2m-fusion-wheel.hX03US` y probados desde fuera del
checkout, con extensión nativa de su wheel. Dependencias existentes
reutilizadas en solo lectura.

| Wheel | SHA256 |
|---|---|
| Python3.12 final v3 | `b6aed53ae581a86217d356142e4663c434761ec12aa87516f97234869dca44b2` |
| Python3.9 final v3 | `4c488c13682b4cfeee5b9b225a6b94a94aefdfd49148ee923182fe02f090d5f7` |

Cada wheel final:265 pasan/9 omitidos. Importaciones y extensión nativa
apuntan a su propio site-packages, no al checkout editable.

Primera suite general congelada:2646 pasan/34 omitidos/4 fallos en961,50s.
Los cuatro son exclusivamente
`test_real_extrapolation_with_missing_onsite_and_edge_blocks[False/True-1/2-HistorySOAPs]`,
por ausencia histórica de `dscribe`. No se instaló ese paquete ni se ocultaron
los fallos. Después se añadió el rechazo de Q escalado; su regresión pasó
RED→GREEN y se repitieron fuentes, wheels y CUDA. Tras la corrección final de
revisión, suite general congelada: **2653 pasan/34 omitidos/los mismos4 fallos
históricos**, en896,61s. No hay fallos nuevos ni se presenta la suite como
totalmente verde.

La restricción adicional de Q ortonormal conserva el contrato de
`forward_with_coefficients`: la ruta histórica proyecta la matriz mediante Q;
con Q escalado, eso no coincide con devolver coeficientes sin esa proyección.
El certificado comprueba filas de Gram una por una en CPU, sin construir una
matriz K×K. Para Q no ortonormal se mantiene la ruta legacy y su rendimiento.

## Medición predefinida

Primaria: pared del paso completo `df72`, incluidos proyección diferenciable,
backward y Adam. Cuatro rondas pareadas, orden equilibrado, procesos nuevos,
10warmups+30pasos; controles `df16`, `w90`, `6x6`. Igualdad de pesos, batch y
receta, preflight fuera del tramo medido, sin profiler durante el timing.
Reportar mediana/MAD/rango y las cuatro razones; mejora estable únicamente
si las cuatro razones son<1. Resultado mixto: inconcluso.

Supervisor existente: allocator≤55%, reserva inicial30%, aborta únicamente
el grupo propio si libre<25% o temperatura≥80°C. No se lanza con cómputo
CUDA ajeno. Kernels del perfil anterior no son una medida de aceleración.

El preflight usa el mismo contexto grad-enabled que el paso medido, retira
sus hooks y libera los dos grafos de validación; limpia la caché **una sola
vez antes del warmup**, nunca entre pasos medidos. Los picos se reinician
después de10warmups y se leen antes del profiler. Conserva pérdidas iniciales,
hashes de pesos/entradas/receta y opciones completas de fusión. El resumen
rechaza duplicados, rondas ausentes, abortos, NaN/inf y pares incompatibles.

Sondas fallidas conservadas, excluidas del análisis:

- Smokev1: especialización TorchScript extra de `no_grad` >5min en un TP;
  solo se detuvo el hijo propio. Un hijo diagnóstico con traceback corroboró
  la ubicación. Preflight grad-enabled eliminó la demora.
- Smokev2: faltaba el preflight en `training_step`, que llama directamente a
  MatrixMACE; hook trasladado al modelo interno. Smokev3:8hijos válidos.
- CUDAv2:8 pasan/1 falla al buscar `nvcc` por un PATH sin permisos. Se resolvió
  únicamente para hijos propios con `PATH=/usr/bin:/bin`; sin cambiar el venv.
- El primer test opcional de compile no invocaba compilación; no se aceptó como
  prueba. Se corrigió, se verificaron callables compilados y se ejecutó Inductor
  real con `max-autotune-no-cudagraphs`. No se activaron CUDA graphs.

## Resultado pareado final:32 procesos

Datos crudos y manifiesto:
`experimentos_speed_probe_homonuclear_20261014_v1/summary.json` y
`manifest.json` (176 hashes de fuentes/helpers/runner, diff inicial y configs).
Smoke finalv4:8hijos válidos, sin fallbacks inesperados. Cada hijo final ejecutó
10warmups+30pasos; cuatro rondas por variante/sistema, sin suite CPU concurrente.
Hardware RTX5090; Python3.12.3, Torch2.11.0+cu130, CUDA13.0, e3nn0.4.4,
Lightning2.6.1. fp32, TF32 desactivado, sin torch.compile en estas mediciones;
su certificación opcional CUDA es una prueba separada.

Tiempos: mediana de las medianas por ronda. Cambio porcentual: mediana de
las razones **pareadas**, no razón entre las dos medianas marginales.
VRAM en MiB, allocator PyTorch; no equivale a memoria total del dispositivo.

| Sistema | Pared off→on, ms | Cambio pareado | Allocated off→on, MiB | Reserved off→on, MiB |
|---|---:|---:|---:|---:|
| d/f16, batch4,C2 |112,09→112,61|+0,31%|577,38→489,71 (−15,18%)|612→512 (−16,34%)|
| **d/f72, primaria**, batch4,C2 |112,42→113,27|+0,49%|2509,80→2100,86 (−16,29%)|2682→2210 (−17,60%)|
| w90, control real |66,02→65,91|−0,07%|693,836→693,843 (≈igual)|890→888 (−2MiB)|
| 6x6, control real |125,98→125,71|−0,13%|5123,708→5123,402 (≈igual)|6516→6516|

| Sistema | Las cuatro razones on/off | Mediana | MAD | Rango |
|---|---|---:|---:|---|
| d/f16 |0,989132;1,016078;1,003641;1,002493|1,003067|0,006793|0,989132–1,016078|
| d/f72 |0,998036;0,994738;1,020856;1,011670|1,004853|0,008466|0,994738–1,020856|
| w90 |0,996266;1,006958;1,002301;0,981785|0,999284|0,005346|0,981785–1,006958|
| 6x6 |0,999937;0,998222;0,997505;0,999256|0,998739|0,000857|0,997505–0,999937|

Primaria d/f72: **no pasa el criterio de aceleración**, dos razones<1 y
dos>1; rendimiento inconcluso/neutro, no una regresión concluyente de0,49%.
d/f16 y w90 también mixtos. 6x6 cumple formalmente cuatro razones<1, pero
−0,13% es demasiado pequeño para recomendarlo como aceleración práctica o
extrapolarlo a otras cargas. No se cambia el criterio a posteriori ni se
selecciona ese control como nueva primaria.

En cambio, la reducción de VRAM d/f es consistente en las cuatro rondas,
incluyendo metadatos: **1,146MiB adicionales** en cada modelo d/f;
0,00449MiB en cada control s/p. No hay ahorro práctico demostrado en w90/6x6.
Construcción completa off/on: d/f16 3,333/3,335s, d/f72 3,331/3,350s,
w90 2,594/2,604s, 6x6 2,610/2,631s. Incluye inicialización/certificación;
no es una medida aislada del coste del certificado.

### Diagnóstico de fases, no sumas de scopes

| Control | Datos CPU off/on, ms | H2D CUDA off/on | Forward+loss CUDA off/on | Backward CUDA off/on | Adam CUDA off/on |
|---|---:|---:|---:|---:|---:|
| w90 |6,043/5,989|1,398/1,398|19,596/19,485|36,628/36,770|0,737/0,731|
| 6x6 |5,303/5,365|0,916/0,921|29,246/29,048|90,205/90,227|0,163/0,163|

Son medianas de eventos/medidas de fases, no descomposición aditiva exacta
de las medianas de pared. d/f usa batch ya transferido; su evento CUDA de
paso completo da112,060/112,581ms (16) y112,390/113,234ms (72).
El helper d/f heredado no aísla el backward con eventos: queda **no medido
por separado**, no se sustituye por su scope exterior incompleto. Sus dos
pasos de profiler se ejecutan después de guardar tiempos/picos, nunca dentro
del tramo de30pasos.

Preflight: error absoluto máximo de predicción≤8,95e-7; pérdidas iniciales
pareadas y hashes de pesos/entradas/receta coincidentes. Las diferencias
bit-a-bit no se exigen. Telemetría de32hijos: mínimo libre66,62%, máximo58°C;
sin abortos de memoria/temperatura. El chequeo de cómputo ajeno ocurre antes
de cada lanzamiento; no es un bloqueo que impida a otro usuario iniciar
procesos entre muestras del supervisor.

## Conclusión y límites

**6.7 implementado y certificado como opción para ahorrar VRAM d/f, no
recomendado como aceleración general.** Mantener `fuse_homonuclear=False` por
defecto. No se afirma mejora de precisión, generalización, inferencia aislada
ni materiales científicos d/f. Las cuatro rondas miden repetición de cómputo,
no incertidumbre de entrenamientos científicos.

Revisión independiente final terminada, sin Critical ni Minor. Un Important:
inputs con distinto número de filas podían hacer broadcasting y cambiar los
valores/errores de la ruta histórica. Se añadieron seis regresiones (tres APIs,
uno/dos orígenes), observadas RED→GREEN. Ahora se exigen entradas2D compatibles
y un número común de filas; en caso contrario se conserva literalmente el
fallback. Gate general, CUDA, floor3.9 y ambas wheels repetidos después del fix.

**Proveniencia temporal:** los32 tiempos y picos de esta campaña corresponden
al snapshot anterior a ese guard de shapes. El fix añade solo comparaciones
de metadatos Python; no cambia operaciones tensoriales para los inputs válidos
medidos. No se ha repetido el benchmark de pared después del fix y no se afirma
que los hashes archivados correspondan al árbol final. La equivalencia final
sí está probada de nuevo en CPU/CUDA/wheels. Logs y rulings completos en
`.superpowers/sdd/2026-10-14-homonuclear-weight-fusion-6-7/`.
