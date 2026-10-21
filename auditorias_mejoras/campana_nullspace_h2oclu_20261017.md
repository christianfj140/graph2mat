# Validación exploratoria de 4.7 en h2oclu — 2026-10-17

## Cierre verificado — 2026-10-18

**Campaña terminada y auditada, sin promoción de defaults.** Los tres IDs
efectivos `:bootstrap1` están DONE, sin FAILED ni proceso propio vivo; tres
fits de2000épocas/64000muestras/8000updates y21matrices nuevas completos.
La conclusión es una mejora espectral exploratoria con deterioro de H, no
una solución general del problema OOD. El resto del documento conserva el historial.

### Requisitos y evidencia

| Afirmación | Prueba de cierre | Resultado |
| --- | --- | --- |
| Procedencia y receta comparables | 925hashes originales+5de recuperación, versiones, entradas ordenadas y estados iniciales congelados; lectura validada de resultados | Verificado; no se regeneraron baselines |
| Presupuesto y selección | 2000épocas×32muestras,4updates/época; best nativo monitoriza val_loss básico, con hparams de entrenamiento1e-4 | Tres candidatos completos; epochs seleccionadas1839/1829/1829 |
| Integridad de resultados y recuperación | hashes de checkpoints/resultados/recibos/42NPZ; fit0/best0 originales conservados; dos prefijos de cola intactos | 21matrices nuevas y21referencias; tres IDs efectivos DONE |
| Reconstrucción y convenciones físicas | helper congelado H0+ΔH; referencias FDF/TSHS/H0 y etiquetas Δ, especies H/O, orbitales/orden/geometría/celda, spin/ocupación y S compatibles | S de referencia, unidades eV, neutral/no polarizado;128ocupados y736orbitales OOD |
| Métricas reproducibles | recalcular métricas completas de las42matrices guardadas, incluyendo p95/p99, soporte y desgloses | Coinciden con resultados originales, sin sobrescribirlos |
| Espectros sin truncamiento | ruta independiente Cholesky+eigvalsh, controles analíticos exacto/cS y todos los niveles de las42matrices | Discrepancia máxima2,28795e-12eV; ninguna dirección eliminada |
| Incertidumbre y regresiones | medias por estructura/seed, razón de medias y27resamples pareados ordenados; lista de todas las regresiones>5% | Coinciden con summary, sin semillas excluidas |
| Coste y límites | seis fits/telemetrías, allocated/reserved, PSS y muestras ausentes explícitas | No comparación causal de velocidad/RAM por no ser contemporáneos |

La reproducción completa y el inventario de982archivos verificados quedan en
`experimentos_nullspace_h2oclu_20261017_v1/closure_audit_20261018.json`.
Incluye42filas por estructura/brazo/seed, métricas/desgloses/soporte, metadatos
orbitales y las regresiones explícitas. `audit_h2oclu_nullspace.py` no infiere,
entrena, escribe en cola ni cambia matrices/checkpoints. Lee las matrices físicas
guardadas; la suma H0+Δ se certifica por la fuente de evaluación congelada y sus
artefactos, no mediante una segunda inferencia. El chequeo Cholesky es numéricamente
independiente del eigensolver generalizado original.

### Resultado primario y compromisos

Medias de tres semillas pareadas0/1/2; dentro de cada seed, cuatro casos ID o
tres OOD. Niveles: RMSE de la ventana definida por la referencia, no todo el espectro.

| Split / error | Fixed | 4.7 | Cambio | IC95 de razón4.7/fixed |
| --- | ---: | ---: | ---: | ---: |
| OOD H alineado,meV | 5,703058 | 6,471390 | +13,47% | 1,0196–1,2705 |
| OOD niveles alineados,meV | 928,487766 | 285,497586 | −69,25% | 0,2489–0,4082 |
| OOD gap absoluto,eV | 3,109454 | 1,192056 | −61,66% | 0,1231–0,5567 |
| ID H alineado,meV | 5,596454 | 6,712047 | +19,93% | 1,1744–1,2258 |
| ID niveles alineados,meV | 63,457823 | 61,541357 | −3,02% | 0,8042–1,1951 |
| ID gap absoluto,meV | 75,495455 | 75,056710 | −0,58% | 0,7330–1,1957 |

| Seed | H OOD fixed→4.7,meV | Niveles OOD fixed→4.7,meV | Gap error OOD fixed→4.7,eV |
| --- | ---: | ---: | ---: |
| 0 | 4,842593→5,268152 | 1760,911241→535,888800 | 4,840239→2,898157 |
| 1 | 6,357157→6,386977 | 673,420129→145,905092 | 2,537560→0,261930 |
| 2 | 5,909422→7,759041 | 351,131929→174,698866 | 1,950562→0,416080 |

La reducción de gap/niveles medios aparece en las tres semillas, pero H medio
empeora en las tres. La mejora del gap se produce en8/9predicciones OOD; los
niveles mejoran7/9 y H solo1/9. Hay regresiones>5% en5/9casos de H,1/9de
niveles y1/9de gap. En seed2/caso000, el errorgap empeora0,132808→0,383818eV
y niveles68,27→147,36meV; la media no oculta esa regresión.

Todos los casos siguientes son OOD n32; las etiquetas000/001/002 se refieren a
`h2oclu_test_n32_…`. Son736niveles por matriz y128ocupados en cada caso.

| Seed/caso | H4.7,meV | Niveles4.7,meV | Gap referencia,eV | Gap predicho4.7,eV | Errorgap4.7,eV |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0/000 | 4,0238 | 510,89 | 4,974187 | 1,743546 | 3,230641 |
| 0/001 | 4,9134 | 920,94 | 5,185933 | 0,163931 | 5,022002 |
| 0/002 | 6,8673 | 175,84 | 4,532587 | 4,974416 | 0,441829 |
| 1/000 | 4,8283 | 107,42 | 4,974187 | 4,809440 | 0,164746 |
| 1/001 | 6,4681 | 153,28 | 5,185933 | 5,016609 | 0,169324 |
| 1/002 | 7,8645 | 177,02 | 4,532587 | 4,984305 | 0,451718 |
| 2/000 | 5,8717 | 147,36 | 4,974187 | 4,590369 | 0,383818 |
| 2/001 | 8,7862 | 200,98 | 5,185933 | 4,606631 | 0,579302 |
| 2/002 | 8,6192 | 175,76 | 4,532587 | 4,817707 | 0,285120 |

Persiste una subestimación severa del gap en seed0/caso001:5,185933→0,163931eV.
Por tanto, ni el errorgap medio1,192eV ni su reducción permiten declarar resuelto
OOD. El test ya fue observado y el linaje histórico es incompleto; tres seeds y
27resamples tienen poca potencia. Evidencia de validación interna exploratoria,
condicional a S de referencia, no claim general ni test final independiente.

Defectos brutos máximos: Hpred2,63e-10, Href2,51e-10, Sref4,78e-10; todos pasan
el gate1e-8 antes de la simetrización explícita. λmin(S)≥0,00125637 y condición
máxima≈4180,48. No se aplicó corte a S ni se sustituyeron métricas originales.
Se conserva soporte unión exacta Href/Hpred/Sref, umbral0; los auxiliares p95/p99,
Frobenius y soporte se publican en las42filas, no se seleccionan post hoc.

### Coste observado, sin atribución causal

| Seed | Pared fit4.7,min | Allocated4.7,MiB | Reserved4.7,MiB | PSS pico árbol,MiB |
| --- | ---: | ---: | ---: | ---: |
| 0 | 67,22 | 1909,15 | 2356 | 9368,75 |
| 1 | 75,11 | 1909,64 | 2564 | 12852,38 |
| 2 | 79,78 | 1909,52 | 2726 | 13489,77 |

Pared de los tres fits3,70191h, no GPU-hours de kernels activos; no incluye la
espera humana/recuperación del evaluador. Último fit terminó2026-10-18 a00:30:13CEST;
la evaluaciónCPU final vino después. Tres evaluacionesCPU suman13,15s. Cada fit
tuvo un intento exitoso y no hubo reentrenamiento de seed0 por el fallo de import.

Medias fixed→4.7: allocated1889,95→1909,44MiB, reserved2546,67→2548,67MiB,
pared4722,10→4442,29s y PSS14174,13→11903,63MiB. Los tiempos/PSS no son
contemporáneos: no atribuir aceleración ni ahorro de RAM al término4.7. No hay
ahorro VRAM demostrado; allocated aumenta≈1,03% y reserved≈2MiB. Telemetría4.7:
libre mínimo observado88,08%, temperatura máxima58°C; no garantía entre sondeos.
Se excluyen explícitamente las4muestras PSS ausentes de candidatos y5de baselines,
no se sustituyen por cero; los picos son muestreados, no cotas duras.

### Decisión y cierre

Conservar4.7 como opción para explorar fidelidad espectral en bases difusas, con
el coste en H documentado. No promover como mejora general: falla la condición
de ausencia de regresiones OOD>5% en H/niveles. No barrer pesos, cambiar selección
ni añadir entrenamientos tras observar este test. La campaña aprobada queda
cerrada incluso con este resultado mixto; no implica cerrar el roadmap completo.

Gates:33focales de campaña/recuperación pasan en39,03s con preload real;
suite general más reciente2653pasan/34omitidos/4HistorySOAPs por falta de dscribe,
942,67s, con fuentes productivas actuales idénticas. No se repite toda la suite
por cambios exclusivamente de auditoría/documentación. Ruff/diff verificados.
Un foco invocado directamente con probe_site sin bootstrap reprodujo el fallo
conocido del lanzador original:1fallo/32pasan; se conserva el log. El wrapper
reparado es el punto de entrada certificado, sin editar fuentes congeladas.
El fallo focal se llama
`test_prepare_rejects_missing_authorization_before_reading_or_creating`:
la importación del módulo congelado precedía al bootstrap. Los cuatro fallos
generales corresponden a `src/graph2mat/tools/server/tests/test_extrapolation.py`:

- `test_real_extrapolation_with_missing_onsite_and_edge_blocks[False-1-HistorySOAPs]`.
- `test_real_extrapolation_with_missing_onsite_and_edge_blocks[False-2-HistorySOAPs]`.
- `test_real_extrapolation_with_missing_onsite_and_edge_blocks[True-1-HistorySOAPs]`.
- `test_real_extrapolation_with_missing_onsite_and_edge_blocks[True-2-HistorySOAPs]`.

Reproducir la auditoría, desde Graph2Mat y con destino nuevo:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=auditorias_mejoras/probe_site:src:auditorias_mejoras OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python auditorias_mejoras/audit_h2oclu_nullspace.py \
  --out auditorias_mejoras/experimentos_nullspace_h2oclu_20261017_v1/closure_audit_repeat.json
```

El script rechaza un destino existente. No repetir prepare/enqueue ni restaurar
los IDs históricos fallidos/bloqueados. El siguiente bloque se decidirá según
las carencias reales del roadmap, no ampliando esta campaña automáticamente.

## Recuperación autorizada — 2026-10-17,21:55CEST

El usuario pidió «Corrige y relanza». El primer fit terminó2000épocas,
64000muestras/8000updates,4033,31s; la entrada falló después, al importar la
evaluaciónCPU. Con probe_site, callback→graph2mat entraba en _sparse.py fuente
antes de recovery_api.compiled_extensions, provocando ausencia de cython.
Reproducción en entorno exacto: import directo exit1, preload nativo exit0.
No fue OOM, fallo científico del entrenamiento ni necesidad de instalar paquetes.

Se añade `relaunch_h2oclu_nullspace.py` sin modificar ningún archivo congelado.
Precarga extensiones instaladas antes de importar el launcher original y evalúa
en el mismo proceso preparado; utiliza guarded_fit original solo cuando falta
fit.json. Evaluación original valida presupuesto/selección, rechaza parciales y
reutiliza resultados completos.33focales pasan42,19s, Ruff/diff limpios. Nueva
suite general en curso; evidencia previa2653pasan/34skip/4HistorySOAPs, sin src edits.

Recibo adicional `relaunch_receipt.json` fija código nuevo, manifest/recibo
originales, fit0 y checkpoint0. Tres appends1412→1415/prefijo intacto, IDs
efectivos `experimentos_nullspace_h2oclu_20261017_v1:null_s{0,1,2}:bootstrap1`,
encadenados; state/entradas viejas intactos. Son recuperaciónCPU0+dos fits
pendientes, no tres nuevos entrenamientos ni nuevas referencias.

Bootstrap0 DONE, siete matrices verificadas, todos los espectros válidos;
fit/checkpoint0 conservan exactamente sus hashes, sin reentrenamiento. A21:55:11
bootstrap1 supervisor1053391 arrancado y bootstrap2 pendiente; daemon sin reinicio.
Auditoría21:56:40:31épocas/992muestras confirmadas por RecoveryStore de seed1,
125updates/1000muestras observadas, fitGPU1053418 vivo; GPU29158/32607MiB
libres40°C.925hashes originales+5adicionales y prefijo de cola pasan.
Evidencia `experimentos_nullspace_h2oclu_20261017_v1/relaunch_audit.json`.
Seed0 exploratoria OOD, medias de tres casos: H4,842593→5,268152meV (+8,79%),
niveles1760,911241→535,888800meV (−69,57%), gap error4840,238721→2898,157243meV
(−40,12%). Validación H+17,05%, niveles−21,64%, gap error+20,83%. Compromiso mixto,
gap OOD todavía grande, solo una semilla; no conclusión de mejora general,
no variar el candidato fijo ni promover defaults. Quedan dos semillas y cierre.

## Protocolo fijado antes de los resultados

El usuario autorizó «Lo autorizo. Hazlo» y aprobó el plan Native mediante
«Apruebo». Se autorizan exactamente tres candidatos nuevos `null_s0/1/2`, no
entrenamientos de referencia ni barridos. Objetivo: comprobar si la pérdida
4.7 existente reduce el error del gap molecular OOD con la receta cosine actual.
No se implementa otra pérdida ni se cambia ningún default.

Plan: `docs/superpowers/plans/2026-10-17-h2oclu-nullspace-validation.md`.
Raíz científica: `experimentos_nullspace_h2oclu_20261017_v1/`.
Estado final: tres candidatos entrenados y evaluados, auditoría cerrada el2026-10-18.
Los IDs originales siguientes son historial; usar sus equivalentes `:bootstrap1`.
IDs fijados: `experimentos_nullspace_h2oclu_20261017_v1:null_s0`,
`experimentos_nullspace_h2oclu_20261017_v1:null_s1` y
`experimentos_nullspace_h2oclu_20261017_v1:null_s2`, en ese orden secuencial.

## Comparabilidad

Se reutilizan exclusivamente `fixed_s0/1/2` de
`experimentos_batching_h2oclu_20261016_science_v1/`. SHA256 del manifest:
`da7a0e3a1268cb9504f2ede4dd0b3cfe1df263294cb5ff21324949e26ac0ddca`.
Se congelan también resultados, recibos, generaciones de recuperación, mejores
checkpoints y 21 matrices físicas de esos tres baselines. Sus 627 hashes originales
y versiones deben seguir pasando; si no, detener, sin lanzar referencias nuevas.

Se conservan 32 train, 4 validation y 3 OOD, geometrías, etiquetas Δ, H0,
especies, bases, pseudos, soporte y orden orbital. Se reutilizan los estados
iniciales completos originales, no solo las semillas. Batch8, 2000 épocas,
64000 exposiciones/8000 actualizaciones, MACE original FP32 compilado,
cache, cuatro workers no persistentes, Adam nativo y cosine2000/eta_min1e-5.
El orden histórico se conserva mediante `runs_json` nativo, congelado por hash.
El root y glob de bases originales no cambian. Antes de preparar los fits se
comparan bit a bit todos los tensores nativos de las 36 muestras; solo se añade C.

Tratamiento único: `spectral_weight=1e-4` en train. Un callback experimental
derivado de ModelCheckpoint pone temporalmente el peso a cero en validación,
manteniendo el min-val_loss de gauge básico cada cinco épocas. Restaura el
peso tras validación y excepciones, sin alterar hparams ni el modelo base.

Los 36 proyectores se calculan exclusivamente con S de referencia de train/val:
C=U_low/sqrt(λ), λ<0,01, padding cero hasta 32 columnas (rangos observados4–32).
Se exige floor1e-3 inactivo y CᵀSC=I para columnas no nulas. Se registra/rechaza
defecto relativo bruto de S>1e-8; solo después se usa explícitamente (S+Sᵀ)/2.
No se generan proyectores OOD ni se elige peso con los resultados nuevos.

## Evaluación y decisión

Cada fit se evalúa en CPU con el helper físico congelado de la campaña anterior:
Hpred=H0+ΔHpred, eV, soporte unión, gauge cS y espectro generalizado full-basis
con S de referencia. Se conservan las 736 direcciones de cada caso OOD; 0,01
del entrenamiento NO es un corte en la evaluación ni cambia la referencia.
Se guardan siete matrices por candidato: cuatro ID y tres OOD; total21 nuevas.

Primaria: error absoluto del gap OOD, media de estructuras por semilla y luego
media de semillas. Publicar también H-MAE alineado y RMSE de niveles alineados,
incluyendo todas las regresiones por semilla y estructura. Razón candidato/base,
ratios pareados y CI95 de los 27 remuestreos pareados ordenados de tres semillas.
Señal robusta requiere límite superior del CI<1; señalar regresiones>5% en H o
niveles aunque mejore el gap. Fallos de hermiticidad/S no se convierten en ceros.
Métricas auxiliares, p95/p99 y soporte quedan en los resultados individuales.

El test OOD ya fue observado y el linaje histórico es incompleto: evidencia
exploratoria y condicional a S de referencia, no test final independiente.
No inferir derivadas, fonones o EPC. No promoción automática de defaults,
aunque el resultado sea favorable. Un control oráculo con etiquetas no es
una mejora desplegable; este experimento sí prueba entrenamiento real.

Referencia fixed conservada (meV, media de tres estructuras OOD por seed):

| seed | H alineado | RMSE niveles | error gap |
| --- | ---: | ---: | ---: |
| 0 | 4,842593 | 1760,911241 | 4840,238721 |
| 1 | 6,357157 | 673,420129 | 2537,560430 |
| 2 | 5,909422 | 351,131929 | 1950,562107 |

## Coste, ejecución y límites

Se mantienen las guardas existentes: cesión ante cómputo ajeno, allocator≤55%,
reserva inicial30%, cierre solo del grupo propio si libre<25%, GPU≥80°C o
watchdog, y recuperación desde época completa. No garantizan20% libre entre
sondeos ante asignaciones externas. No modificar ni reiniciar autorun.

El recibo propio de cola conservará IDs únicos, tres appends con lock y hash
del prefijo anterior; los candidatos se encadenan detrás de dependencias vivas.
Cada entrada hace fit→evaluaciónCPU→summary antes de liberar la siguiente.
Un fallo bloquea dependientes; no relanzar con otra receta automáticamente.

Los resultados conservan pared, allocated/reserved, PSS, intentos y cesiones
con el ámbito del runner. El coste histórico de las referencias no es una
medida contemporánea: no afirmar ahorro causal de velocidad/VRAM.
Estimación inicial4–6h exclusivas para tres fits, más gates y espera ajena
sin cota. Baselines78,70min/fit; overhead actual de4.7 aún no medido.

## Evidencia técnica y trazabilidad

Workspace conservado: `.superpowers/sdd/2026-10-17-h2oclu-nullspace-validation/`.
Preflights_v1/v2/v3/v4 son preparación/ensayos CPU, nunca semillas científicas.
Los errores previos de simetrización y formato CLI quedan registrados; las
correcciones no relajan tolerancias ni cambian producto/receta.
Smoke real v4: dos épocas, cuatro muestras, cuatro workers, interrupción tras
época1, recuperación completa y carga por LitMACEMatrixModel base pasan.
val_loss seleccionado0,047870177775621414 frente a cálculo básico independiente
0,047870174050331116 (FP32). En los checkpoints nuevos el callback normaliza
solo el metadato redundante basis_table=None: tras resume el CLI lo convertía
en texto no recargable; la tabla real superior, los pesos y el peso1e-4 se conservan.

Incidente de test negativo: un selector de archivo erróneo añadió siete bytes
a la extensión compartida `_sparse.so`. Se retiraron exclusivamente esos bytes
tras verificar que el prefijo coincidía con el hash congelado original;
SHA restaurado `b2ea222ff9a7891fd1c4daebcccce11777a90a4365e15fc7c8bd43ce524f896b`.
El test ahora modifica solo su referencia temporal. No queda cambio de dependencia.

Revisión independiente única Native terminada: 0Critical/1Important/1Minor.
Contraste con scipy.linalg.eigh independiente del generador NumPy en las36S:
max|CᵀSC−I|=2,36e-13 y diferencia máxima de CCᵀ≤4,17e-11; padding/floor,
coordenadas y cuentas orbitales compatibles. El Important detectó sobreescritura
al repetir run/eval: ahora se reutiliza resultado completo verificado y se
rechaza cualquier destino de evaluación parcial, preservándolo sin reescribir.
Una evaluación CPU interrumpida requiere decisión explícita sobre un intento
nuevo; no se sobreescribe automáticamente ni se repite entrenamiento terminado.

El Minor sobre semillas espectrales excluidas se reclasificó Important por su
efecto sobre la población analizada: summary identifica paired_seeds y cada
exclusión con seed, brazo, estructura y motivo. Se publican regresiones y fallos,
no solo las parejas que sobrevivan. Ambos hallazgos entraron en una sola pasada
RED→GREEN, sin segunda revisión. No quedan mejoras menores diferidas.

Gates focales definitivos:26tests nuevos pasan41,84s; conjunto nuevo+recuperación+
campaña+diagnóstico150pasan/5CUDAguarded omitidos297,75s. Ruff src/tests y los
tres archivos Python nuevos limpio; diff--check limpio. Manifest científico
`de03dcd283564d1504905dca1a189af1fac8829c0281a5bc83dbe07d2aa8ece3`,925archivos
congelados y36comparaciones bit a bit pasan.

Suite general fresca:2653pasan/34omitidos/4fallos,910,30s. Los cuatro son exactamente
test_real_extrapolation_with_missing_onsite_and_edge_blocks[False/True-1/2-HistorySOAPs],
por ausencia de dscribe; ningún fallo nuevo. No instalar ni corregir esa dependencia.
El conjunto general no incluye los tests de auditorias; no sumar recuentos históricos.

## Encolado verificado — 2026-10-17,18:42CEST

`queue_receipt.json` complete=True, tres IDs únicos y comandos exactos verificados
contra la cola real;1409→1412entradas y SHA256 del prefijo anterior intacto.
null_s0 no tenía dependencias vivas; null_s1 espera null_s0 y null_s2 espera null_s1.
Daemon1756952 conservado sin cambios/reinicio. Supervisor912675 y fitGPU912694
vivos, comandos propios y relación de parentesco correctos; GPU31228/32607MiB
libres,35°C al arrancar. Confirmación de PID no equivale a resultado científico.
Auditoría18:43:19:24épocas/768muestras confirmadas por RecoveryStore,
96updates observadas, mejor checkpoint protegido presente;925hashes verificados
después del arranque. GPU29472/32607MiB libres,38°C. Evidencia reproducible del
estado en `experimentos_nullspace_h2oclu_20261017_v1/submission_audit.json`.

No repetir prepare/enqueue ni editar fuentes/configs/C congelados mientras corren.
Consultar recibo/IDs filtrados, progress/fit/result/summary y procesos reales;
no imprimir el mapa histórico completo de pids. Resultados y auditoría de las
21matrices candidatas quedan pendientes hasta terminar los tres trabajos;
no confundir gates, CPU smoke o arranque con beneficio físico.
