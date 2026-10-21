# Diagnóstico del gap OOD de h2oclu — 2026-10-17

## Resultado y decisión

El mecanismo numérico dominante en estas matrices es el error aprendido dentro
del subespacio de pequeño solapamiento, amplificado al resolver el problema
generalizado. No es únicamente un error grande de los orbitales HOMO/LUMO:
estados intrusos pueden cambiar la ordenación de los niveles ocupados.

En las 18 predicciones OOD, sustituir **solo el residual dentro del subespacio
λ(S)<0,01** por su valor exacto reduce el error medio del gap de **3,044974 a
0,270305 eV**. Son controles oráculo con etiquetas, no un nuevo modelo ni una
mejora desplegable. Se conservan las 736 direcciones, la misma S, referencia,
ocupación y gauge. No se aplica ortogonalización truncada.

Recomendación mínima: estudiar una validación comparable de la **opción 4.7 ya
existente**, antes de más arquitectura o pesos por bloques. No implementarla de
nuevo. Sus resultados históricos no prueban que funcione con la receta cosine
actual. No se entrenó, encoló ni cambió ninguna pérdida o default en este trabajo.

## Alcance y procedencia

- Repositorio `graph2mat`, rama `g2m-v2`, HEAD
  `270d17d1f204293a059baa09e8f1b3756981e2bf`; checkout previo preservado.
- Seis fits cerrados de `experimentos_batching_h2oclu_20261016_science_v1`,
  fixed/cost × seeds 0/1/2; cuatro estructuras ID de selección y tres OOD.
- Los 627 hashes congelados y versiones se verifican antes y después. Se
  comprueban además manifest, seis resultados/recibos, checkpoints seleccionados
  y 42 NPZ de H físico. No se carga un modelo ni se ejecuta inferencia nueva.
- Especies, geometría y orbitales se leen del FDF y su base `.ion.xml` congelada,
  no de los Z placeholder del TSHS. H/O tienen respectivamente 5/13 orbitales,
  con orden explícito `(n,l,m,zeta)` por átomo guardado en el JSON.
- Se verifican coordenadas, celda, orbitales por átomo, nsc=(1,1,1), spin no
  polarizado y S idéntica entre Href/H0. Las cuatro etiquetas Δ ID conservadas
  verifican Δ=Href−H0; OOD no tiene etiqueta Δ separada y se calcula en memoria.
  La identidad de reconstrucción del H predicho guardado está respaldada por el
  evaluador congelado y sus recibos; aquí se contrasta algebraicamente, no se
  pretende repetir la inferencia que lo produjo.
- H en eV por la lectura nativa de sisl; los espectros recalculados coinciden
  con todos los niveles originales a 1e-8 eV, y las métricas H/gap/ventana también.
- Puertas crudas de Hermiticidad ≤1e-8 para Href/Hpred/H0/S antes de proyección
  simétrica; λmin(S)>1e-10. No se impone esa misma razón al residual diminuto,
  cuyo denominador cambia por cancelación. Una regresión cubre esa distinción.
- NetCharge=0; OOD contiene 32 O+64 H, 256 electrones y 128 niveles ocupados.
  Todos los resultados son **condicionales a S de referencia**, no validación
  de S predicha ni de bandas dispersivas o derivadas.

Los scripts de entrenamiento, artefactos originales y sus métricas no se editan.
La ejecución final del diagnóstico tarda **26,96 s de pared CPU**, con BLAS
limitado a dos hilos. No es un benchmark de entrenamiento, memoria o GPU.

## Método de atribución y comprobaciones

Se usa el mismo gauge de la campaña:
`c = <S,Hpred−Href>/<S,S>`, sobre la unión exacta Href/Hpred/S, y
`R = Hpred−Href−cS`. El soporte no se diluye con ceros implícitos.

Para `S=U diag(λ) Uᵀ`, `D=diag(λ)^(-1/2) Uᵀ`, se calculan
`Aref=D Href Dᵀ` y `T=D R Dᵀ`. Es equivalente a Löwdin salvo una transformación
ortogonal: se conservan **todos** los autovalores positivos de S.
λ<0,01 solo define un proyector diagnóstico L y su complemento Q.

Dos descomposiciones distintas:

1. En la base de autovectores de S, las energías cuadráticas de `LTL`, los
   bloques cruzados y `QTQ` son aditivas en `||T||F²`.
2. Para máscaras atómicas/especies/familias se publica
   `<T,D Rgrupo Dᵀ>/||T||F²`. Estas contribuciones **con signo** suman uno;
   pueden ser negativas o mayores que uno por cancelación. No son porcentajes
   causales del error del gap. Se publican además las fracciones de `||R||F²`.

El control de cada grupo evalúa `Aref+T−Tgrupo`, con la misma c y ventana
predeclarada HOMOref−2/LUMOref+2 eV; no realinea cada intervención. Los cambios
del gap son no lineales y no se suman como si fueran errores elementwise.

Pesos de estados: `Tr(P_L P_estados)/rango(P_estados)` en el espacio
ortonormal. Las selecciones incluyen clústeres degenerados completos con
tolerancia 1e-8 eV; no se interpretan autovectores arbitrarios por separado.
La fuga ocupada se define como `Tr(P_ref,no-ocupado P_pred,ocupado)`;
es un peso en equivalentes de estados, no una carga Mulliken ni un número
entero de electrones. Si el gap fuese degenerado, esa atribución ocupada no
se publicaría como única.

Hay controles analíticos de referencia exacta, desplazamiento cS, cancelación
de H0, amplificación por pequeño S, sumas de particiones, conservación de
dimensión, reordenación ocupada y rotación de subespacios degenerados.

Comprobación adicional independiente sobre las 42 intervenciones LL: se
reconstruye el Hamiltoniano reparado en la base atómica original mediante
`D^-1 (T−T_LL) D^-T + Href` y se resuelve por Cholesky `L^-1 H L^-T`, no por
el marco de autovectores de S usado en el diagnóstico. Todos conservan la
dimensión original; diferencia máxima de error del gap **2,84e-13 eV**.
Los hashes de entradas/resultados/fuentes siguen intactos después del contraste.

## Referencias y H0

| Estructura | Orbitales | λmin(S) | Dimensión L | Gap referencia eV | Gap H0 eV |
|---|---:|---:|---:|---:|---:|
| ID n4_000 | 92 | 0,002980 | 13 | 4,559661 | 8,996523 |
| ID n4_001 | 92 | 0,003151 | 13 | 4,769597 | 8,541316 |
| ID n8_002 | 184 | 0,002277 | 32 | 5,924497 | 8,925793 |
| ID n8_003 | 184 | 0,002514 | 32 | 5,779445 | 8,614750 |
| OOD n32_000 | 736 | 0,001292 | 137 | 4,974187 | 7,981717 |
| OOD n32_001 | 736 | 0,001256 | 141 | 5,185933 | 7,974740 |
| OOD n32_002 | 736 | 0,001512 | 136 | 4,532587 | 7,991556 |

Se comparan H0, `Δref=Href−H0` y `Δpred=Hpred−H0`. La identidad
`Δpred−Δref=Hpred−Href` se cumple con defecto máximo **4,44e-16 eV**.
H0 compartida cancela exactamente: **no hay un error independiente de H0 que
pueda sumarse al error de aprendizaje**. Sí hay una corrección física que
aprender: los gaps H0 difieren 2,79–3,46 eV de la referencia OOD.

La norma de Frobenius del residual H0 alineado y transformado por D es
36,23/36,09/38,96 eV en las tres estructuras OOD. En n32_001, el residual
aprendido es 170,78–327,04 eV en esa norma, aunque mejore la fidelidad
elementwise. Esto muestra que una corrección pequeña en la métrica de entrada
puede introducir errores espectrales mayores que los de H0; no demuestra que
H0 sea una mejor predicción física global.

## Resultado numérico por estructura, brazo y semilla

`LL%` es la fracción de `||T||F²` en LTL. `Error LL exacto` es el error absoluto
del gap tras reparar **solo ese residual**, no tras quitar direcciones de S.

| Trabajo | OOD | Gap eV | Error original eV | LL% | Error LL exacto eV | Fuga ocupada, estados |
|---|---|---:|---:|---:|---:|---:|
| fixed_s0 | 000 | 0,0433 | 4,9309 | 85,50 | 0,0986 | 3,060 |
| fixed_s0 | 001 | 0,0509 | 5,1350 | 88,22 | 0,0576 | 4,199 |
| fixed_s0 | 002 | 0,0778 | 4,4548 | 86,05 | 0,4392 | 1,722 |
| cost_s0 | 000 | 0,1791 | 4,7951 | 83,82 | 0,1115 | 1,067 |
| cost_s0 | 001 | 0,0468 | 5,1391 | 87,43 | 0,0246 | 1,210 |
| cost_s0 | 002 | 3,1382 | 1,3944 | 85,54 | 0,5025 | 0,077 |
| cost_s1 | 000 | 0,8518 | 4,1224 | 84,95 | 0,1280 | 0,227 |
| cost_s1 | 001 | 0,1618 | 5,0242 | 87,68 | 0,2793 | 2,216 |
| cost_s1 | 002 | 5,0531 | 0,5205 | 85,19 | 0,5275 | 0,033 |
| fixed_s1 | 000 | 2,8476 | 2,1265 | 83,97 | 0,1567 | 0,097 |
| fixed_s1 | 001 | 0,2050 | 4,9809 | 86,69 | 0,2914 | 2,097 |
| fixed_s1 | 002 | 5,0378 | 0,5052 | 85,12 | 0,5199 | 0,037 |
| fixed_s2 | 000 | 5,1070 | 0,1328 | 89,03 | 0,1203 | 0,046 |
| fixed_s2 | 001 | 0,0531 | 5,1328 | 89,43 | 0,1798 | 1,065 |
| fixed_s2 | 002 | 5,1186 | 0,5860 | 85,78 | 0,5868 | 0,041 |
| cost_s2 | 000 | 5,0833 | 0,1091 | 88,91 | 0,0942 | 0,049 |
| cost_s2 | 001 | 0,0387 | 5,1472 | 89,34 | 0,1753 | 1,095 |
| cost_s2 | 002 | 5,1050 | 0,5724 | 85,71 | 0,5724 | 0,040 |

Los 24 registros ID y todas las descomposiciones/controles individuales están
en el JSON final; no se omiten los casos donde una intervención empeora.

## Qué explica la amplificación y qué no

Medias de registros, con el mismo peso por estructura y fit:

| Métrica | ID selección, 24 registros | OOD, 18 registros |
|---|---:|---:|
| H-MAE alineado, meV | 5,538 | 5,670 |
| Norma operador de R, eV | 0,596 | 2,623 |
| Norma operador de T, eV | 15,671 | 64,586 |
| Error absoluto gap original, eV | 0,07106 | 3,04497 |
| RMSE niveles original, eV | 0,06127 | 0,78543 |
| Fracción cuadrática LL | 73,93% | 86,57% |
| Fracción cuadrática cruzada L/Q | 22,67% | 12,39% |
| Fracción cuadrática QQ | 3,40% | 1,04% |
| Error gap con LL exacto, eV | 0,06114 | 0,27031 |
| RMSE niveles con LL exacto, eV | 0,05647 | 0,12648 |
| Error gap reparando LL y L/Q, eV | 0,04947 | 0,27656 |
| Fuga ocupada media, estados | 0,00267 | 1,02086 |

La reparación LL reduce el error medio OOD del gap un 91,12%, pero solo mejora
15/18 casos; los tres no mejorados están en n32_002 y ya tienen errores de
aproximadamente 0,5–0,6 eV. Reparar todas las contribuciones que tocan L mejora
17/18 casos. Estas cifras no son una garantía de un método entrenable.

Reparar solo el bloque cruzado deja el error medio en 2,96881 eV; reparar solo
QQ, en 2,92341 eV. La evidencia localiza el mecanismo especialmente en LL,
no demuestra que todo el residual fuera de LL sea irrelevante.

**Contraejemplo a una explicación demasiado simple:** hay predicciones con
86–89% de energía residual en LL que conservan un gap razonable. La fracción
no basta para predecir el fallo: importan signo, orientación y mezcla con los
estados de referencia. Tampoco basta la condición de S, que es idéntica entre
brazos/semillas de una misma estructura.

El salto ID→OOD tampoco se debe exclusivamente a una S más condicionada:
la norma operador del residual sin transformar ya aumenta 0,596→2,623 eV,
pese a H-MAE parecido. Crecen la estructura/coherencia del error y el tamaño
del sistema, no solo su amplificación. Este contraste no aísla causalmente
el tamaño del cambio de entornos. La proyección del residual a la ventana
de estados de referencia tiene norma Frobenius media 0,267 eV ID y 1,646 eV
OOD, mucho menor que el residual transformado completo: mirar únicamente
primer orden en esa ventana puede no detectar intrusos procedentes de fuera.

En n32_001, los dos niveles frontera tienen solo 0,34–3,65% de peso en L.
Sin embargo, la traza ocupada sobre L pasa de 0,1479 en la referencia a
1,0924–3,6950, y la fuga del espacio ocupado es 1,065–4,199 estados. Hay
129–132 niveles por debajo de HOMOref−0,05 eV frente a 128 ocupados nominales.
Por tanto, un intruso puede quedar ocupado y desplazar qué niveles ocupan
los índices 127/128: el gap pequeño no exige que esos dos autovectores sean
los de mayor peso en L. No debe describirse todo el fallo como «un estado
aislado dentro del gap».

## Especies, familias y bloques

| Partición OOD | Contribución con signo a `||T||F²`, media | Error gap oráculo medio, eV |
|---|---:|---:|
| onsite | 55,07% | 3,75463 |
| offsite | 44,93% | 2,66812 |
| H-H | 36,35% | 1,43529 |
| H-O | 36,23% | 2,47710 |
| O-O | 27,42% | 2,87492 |

Reparar onsite no es una solución universal: ayuda mucho a seed0 en n32_000/001,
pero puede cerrar gaps previamente razonables en otras semillas. Reparar
offsite muestra el comportamiento opuesto en varios casos. Esto contradice
una recomendación automática de aumentar solo el peso onsite.

El proyector L contiene aproximadamente 52% de peso orbital en O-2p (las dos
zetas), 19% en O-2s, 26% en H-1s y 3% en polarización H-2p; O-3d aporta ≈0,2%.
Estos pesos describen el subespacio, **no el error**. La atribución del error
es distinta: medias con signo destacadas H1s(z1)-H1s(z1) 23,16%,
O2p(z1)-O2p(z1) 16,77%, O2s(z1)-O2s(z1) 15,61%,
O2p(z1)-O2p(z2) −14,27% y O2p(z2)-O2p(z2) 11,78%.
Los pares con m se agrupan por familia radial/angular completa, sin privilegiar
una orientación arbitraria. No aparece una única familia a corregir siempre:
por ejemplo, O2p(z1)-O2p(z2) llega a −136,9% en cost_s1/n32_002 por cancelación.

Los diez pares de bloques atómicos de mayor contribución absoluta, sus especies
y los pesos por átomo se guardan para cada registro; índices cero-based.
Los tres primeros se reparan por separado y conjuntamente. En n32_001, los
tres primeros acumulan solo 8,28–24,12% de la contribución con signo y sus
identidades cambian: (22,66)/(92,92)/(28,30) para fixed_s0,
(29,29)/(36,36)/(6,29) para seed1 y (14,14)/(13,13)/(28,28) para seed2.
No hay un bloque atómico culpable universal.

No aditividad explícita: en cost_s1/n32_001, sumar los tres cambios individuales
del gap da −0,00999 eV, pero la reparación conjunta cambia −0,10386 eV.
Reparar los tres bloques principales deja un error OOD medio de 3,75942 eV,
peor que el original, pese a reducir algunas contribuciones matriciales.

## Hipótesis, evidencia contraria y límites

| Hipótesis | Evidencia | Dictamen |
|---|---|---|
| Unidades, ocupación, solver o gauge explican por sí solos el colapso | FDF neutral/128 ocupados, reproducción Cholesky previa y recálculo actual; cS analítico conserva gap | No respaldada |
| Error independiente de H0 se suma al de la red | H0 compartida cancela a 4,44e-16 eV | Refutada en esta representación |
| Error LL amplificado y reordenación ocupada son mecanismo dominante | Reparación full-basis LL 3,045→0,270 eV; fuga ocupada y pesos de subespacios | Respaldada para estas matrices |
| Basta corregir onsite, una especie o tres bloques dominantes | Oráculos por grupo pueden empeorar; atribuciones cambian entre semillas | Refutada como regla universal |
| Condición de S o porcentaje LL predicen todo el fallo | Misma S y gaps distintos; fracciones LL altas también con gaps razonables | Insuficiente |
| Una pérdida 4.7 arreglará la campaña actual | Solo antecedentes históricos, sin fits comparables actuales con esa opción | Pendiente, no demostrado |
| El batching causa el mecanismo | Aparece en fixed y cost; las seis parejas no aíslan arquitectura/objetivo/datos | No demostrado |

Evidencia **diagnostic-only**: 3 semillas, 3 estructuras OOD correlacionadas ya
observadas, validación usada para selección y lineage incompleto. Las 18 filas
no son 18 réplicas independientes; no se inventa un IC poblacional con ellas.
Se describen rangos y variabilidad, sin escoger un ganador post hoc.

MAJOR para afirmar generalización física resuelta: gap sigue fallando en las
predicciones sin oráculo. MODERATE: no se ha aislado por qué la optimización,
la representación local o los datos permiten ese residual; el mecanismo
numérico no es una causa única del entrenamiento. No se auditó ni reentrenó
la campaña histórica 4.7 para volver a validar sus cifras.

## Siguiente intervención mínima, solo propuesta

La opción existente usa `spectral_window="spectral_null.npz"` y
`gauge_invariant_mse(spectral_weight=...)`; no hace falta una nueva API ni un
corrector de bloques. La receta de los seis fits actuales no activa ese término.
El antecedente 4.7 en este dataset hace más informativo comprobar su transferencia
a esta receta que inventar otra arquitectura o recortar la base.

Para avanzar hace falta autorización nueva de un protocolo y sus trabajos:
comparación pareada de tres candidatos 4.7 con los fixed_s0/1/2 guardados **solo
si se verifica comparabilidad completa**, conservando datos/splits, muestras,
inicializaciones, cosine, selección y evaluación full-basis. El peso histórico
1e-4 es un candidato previamente observado, no un óptimo independiente.
No ajustar pesos ni elegir checkpoints mirando OOD. Si la referencia guardada
no puede reutilizarse limpiamente, pedir autorización de los baselines nuevos
antes de lanzarlos. No se ha preparado ni encolado esa campaña aquí.

## Reproducción y verificaciones

Script: [diagnose_h2oclu_ood.py](diagnose_h2oclu_ood.py).
Pruebas: [test_h2oclu_ood_attribution.py](test_h2oclu_ood_attribution.py).
Resultado autoritativo: [results_v3.json](diagnostico_h2oclu_ood_20261017/results_v3.json).
`results.json` y `results_v2.json` son iteraciones anteriores conservadas, no
la fuente de las conclusiones finales.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:auditorias_mejoras \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python \
auditorias_mejoras/diagnose_h2oclu_ood.py \
--output auditorias_mejoras/diagnostico_h2oclu_ood_20261017/reproduction.json
```

El destino debe ser nuevo, dentro de auditorias y fuera de la campaña original;
el CLI rechaza sobreescritura. El JSON registra hashes de entradas y del script.
SHA256 script: `46a3469ff6c7bf3a2751cb5efffe24ab1727abb08a5d2a182b6a2930622d5f3e`.
SHA256 resultado v3: `4729fa1f34a113d143d9e9d0aa55e4fe150c97926ffd0f478f6a6747975511dc`.

Verificación nueva, no reutilización del gate histórico:

- Once controles propios pasan. Con los21tests del evaluador/campaña existente:
  **32 passed**,17,71s,1051warnings de la pila existente.
- `pytest src tests -q --tb=short --disable-warnings`, con el mismo entornoCPU:
  **2653 passed /34 skipped /4 failed**,878,07s,40653warnings.
  Este comando no incluye auditorias; no equiparar su recuento con2789 de
  baterías históricas de otro alcance.
- Los cuatro fallos son exactamente los conocidos de
  `src/graph2mat/tools/server/tests/test_extrapolation.py::test_real_extrapolation_with_missing_onsite_and_edge_blocks`:
  `[False-1-HistorySOAPs]`, `[False-2-HistorySOAPs]`,
  `[True-1-HistorySOAPs]`, `[True-2-HistorySOAPs]`.
  Todos por `ModuleNotFoundError: No module named 'dscribe'`;
  no se instaló ni se ocultó la dependencia. **La suite no está toda verde.**
  [Log completo](diagnostico_h2oclu_ood_20261017/full-suite-src-tests.log).
- Ruff de `src tests` y los dos archivos nuevos pasa; `git diff --check` limpio.
  Los627hashes congelados y los hashes de resultados/fuentes del diagnóstico
  permanecen intactos después de los cálculos y del contraste independiente.

Sin dependencias instaladas, cambios productivos, GPU, commits, fonones o
modificaciones de autorun. Los smokes de entrenamiento sintético de la suiteCPU
no son nuevos fits científicos ni alteran la campaña. Diagnóstico cerrado;
la corrección aprendida y el resto del roadmap siguen pendientes de decisión.
