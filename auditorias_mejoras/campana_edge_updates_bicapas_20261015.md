# 7.2 — Actualización de aristas entrenando con bicapas

**Cierre2026-10-16:** los seis trabajos y sus evaluaciones terminaron. Medias
OOD de tres semillas: tbgH+16,8%, bandas−7,5%; grhbn_blH−5,3%, bandas−9,5%.
La mejora media de bandas no es consistente entre semillas; en grhbn_bl las
bandasID empeoran en las tres (+35,4% de media). Opción conservada, **sin
promoción ni cambios de defaults**. Resultados y límites completos al final.

**Recuperación2026-10-15:** primer intento fallido a los1290pasos por SIGSEGV
en el volcado periódico de diagnóstico. Causa reproducida bajo gdb, timer
retirado sin tocar modelo/datos. Control2500pasos/340,38s pasa. Un retry del
mismo `tbg_s0` encolado y única dependencia corregida; resto intacto.
RetryPID2974768 confirmado,4750pasos observados; otros cinco pendientes.
GateCPU2662pasan/34omitidos/4fallos históricos `dscribe`;39tests afectados pasan.
[Diagnóstico, pruebas y recibo de recuperación](diagnostico_relanzamiento_72_20261015.md).

Autorización del usuario: «Apruebo 7.2 en ambas bicapas, seis trabajos
(recomendado)». Fecha:2026-10-15. Repositorio `graph2mat`, rama `g2m-v2`.
Es una extensión acotada del flujo experimental existente, no una arquitectura
nueva. Sin cambios de defaults, fonones, autorun ni entrenamientos de referencia.

## Pregunta y contrato previo

¿Una capa residual de actualización de aristas mejora la precisión cuando
el entrenamiento ya contiene acoplamientos intercapa? La extrapolación anterior
monocapa→bicapa no demostraba una mejora general de7.2; este experimento cambia
el dominio de entrenamiento, no utiliza ese resultado para elegir hiperparámetros.

| Afirmación | Evidencia prevista | Estado inicial |
|---|---|---|
| La opción funciona y sus pesos reciben gradientes | Piloto40pasos por material, pérdida/gradientes finitos y activos; lectura/predicción de checkpoint | Preflight; no prueba de precisión |
| Mejora H o bandas en estos dominios | Tres semillas0/1/2, referencia histórica comparable, misma evaluación actual para referencia/candidato | Pendiente de entrenamientos |
| Acelera o ahorra VRAM | Requeriría referencias nuevas pareadas bajo igual carga y protocolo | No se afirma; no están autorizadas |
| Generaliza universalmente | Requeriría materiales y pruebas independientes adicionales | Fuera del alcance |

## Datos, recetas y selección congelados

- `tbg`:474train/60validation; referencia
  `runs/bilayer/n474__ts{0,1,2}__g2mv2_fp32_lin_gaugeh_delta`.
- `grhbn_bl`:66train/9validation; referencia
  `runs/grhbn_bl/grhbn_bl75__ts{0,1,2}__g2mv2_fp32_lin_gaugeh_kl_delta_u16k`.
  Es bicapa grafeno/hBN, no el conjunto anterior de tres capas.
- Único cambio científico: `edge_update_layers:0→1`. Modelos/hyperparámetros
  efectivos de los checkpoints históricos se hacen explícitos en las configs;
  se mantienen batch32, fp32, pérdida gauge, ΔH, Adam y cosine,533/5333épocas.
- Misma selección: checkpoint de menor `val_loss`, sin ajustar por OOD.
  No se cambia el DataModule ni se introduce un sampler nuevo. Mismas semillas
  no implican pesos iniciales o secuencias de batches bit-a-bit idénticos: la
  nueva arquitectura consume aleatoriedad adicional.
- Manifest de cada dataset y hashes de sus artefactos verifican train/validation
  y OOD; los enlaces históricos se cotejan con los mismos archivos Δ/H0.
  Bases, configs, checkpoints, fuentes y extensiones nativas quedan congelados.
  No se generan etiquetas Δ/H0 ni se escribe sobre los datasets externos.

## Evaluación y criterio

Se reevalúan los seis checkpoints históricos, sin reentrenarlos, con el mismo
código actual y dispositivo CPU que los candidatos. ID usa la validación
histórica completa. OOD usa registries no vistos para `tbg` y moiré22° para
`grhbn_bl`; se conserva el filtro histórico `registry_seen_in_training`.

Primarias, en meV: `H_MAE_gauge_meV` y `band_rmse_aligned_meV`, tanto ID como OOD.
H se reconstruye sumando el H0 existente; bandas usan el solapamiento de
referencia. Se conservan CSVs por estructura, incluidos intra/intercapa.
Resumen por tres semillas: medias, razón de medias candidato/referencia y
razones pareadas por semilla con rango. Resultados incompletos se rotulan
`partial`, nunca se completan con otras semillas o recetas.

Los OOD ya se han observado en campañas anteriores: no son test independiente.
Tres semillas permiten una comparación exploratoria, no una promoción automática
de defaults ni una conclusión sólida sobre diferencias pequeñas. Se informarán
regresiones, incluso si otra métrica mejora. No se toca el default.

## Operación nocturna

Runner: `run_bilayer_edge_updates.py`; tests: `test_bilayer_edge_updates.py`.
Orden aprobado: tbg0→grhbn0→tbg1→grhbn1→tbg2→grhbn2, secuencial, en el daemon
existente. Añadir solo seis entradas mediante `append_queue`; conservar bytes
anteriores, no reactivar pendientes pausados o con ancestros fallidos.

Límite del allocator≤55%; presupuesto calculado con30% de reserva inicial.
Supervisor aborta solo su grupo si libre<25% o GPU≥80°C; cede a cómputo ajeno
sin enviarle señales. El daemon conserva su supervisión térmica/CPU existente.
Falta de progreso:300s en piloto,900s en entrenamiento→aborto propio registrado.
Es protección muestreada, no garantía contra asignaciones ajenas entre muestras.
Un error no dispara otra campaña ni referencias nuevas; los siguientes trabajos
encadenados no se lanzan si su dependencia falla.

Cada trabajo guarda checkpoint seleccionado, tiempos y telemetría propios,
reevaluación de referencia/candidato, `result.json`; después actualiza
`summary.json`. La cola persistente no depende de que el chat siga abierto;
no equivale a una supervisión humana/IA continua y necesita el equipo encendido.

Estimación inicial histórica:9–12h, pues las seis referencias sumaron9,23h.
No es una comparación de velocidad: referencias corrieron bajo otra carga.
Se revisará la previsión con pilotos y el primer entrenamiento completo.

## Registro de preflight

- Ocho tests del wrapper pasan, incluidos hashes/receta/ancestría de cola,
  resumen parcial, ownership, watchdog y entorno absoluto para subprocess.
- v1: preparación inicial, sin lanzamiento; v2: rechazo del glob absoluto antes
  de entrenar, corregido utilizando `root_dir` histórico y patrones relativos.
- v3: bloqueo sin progreso; TERM no bastó, se terminó únicamente su grupo propio
  identificado con PGID=SID=PID. Causa no determinada; no se presenta como arreglada.
  No se modificaron permisos de ptrace ni procesos ajenos. Se añadió watchdog.
- v4: ambos pilotos GPU40pasos pasan. Tbg reserved1886MiB; hBN3518MiB.
- v5: GPU tbg pasa; smoke CPU detecta PYTHONPATH relativo inválido al cambiar
  de cwd. Entorno absoluto/CPU explícito corregido RED→GREEN; ningún trabajo encolado.
- v6: preparación y ambos pilotos finales completos,40pasos por material;
  gradientes activos, pérdidas finitas, lectura de checkpoints y predicción
  H/bandas ID/OOD de referencia/candidato correctas. Tbg reserved1886MiB,
  hBN3518MiB. Duraciones de piloto51,50/72,14s, incluyendo setup y validación;
  no extrapolarlas linealmente como duración del entrenamiento.
- Ocho tests del wrapper y16 del benchmark6.7:24pasan. Ruff y diff--check limpios.
- **Encolados exactamente seis trabajos**, con dependencias secuenciales y sin
  dependencias vivas externas en ese momento. Cola1396→1402entradas; verificado
  SHA256 de todos los bytes anteriores. Recibo con IDs y hash previo conservado.
  Los68pendientes antiguos y82fallidos se mantienen intactos. Autorun existente,
  PID1756952, no reiniciado ni modificado.
- Lanzamiento confirmado: `tbg_s0` PID2858309, hijo del daemon; entrenamiento
  completo en progreso, heartbeat780pasos observado. Los otros cinco están
  encolados, no lanzados.8.306hashes congelados y prefix de cola verificados
  de nuevo después del lanzamiento. Estado histórico puntual: ese intento
  falló después; consultar la recuperación al inicio de este documento.

Campaña final encolada:
`auditorias_mejoras/experimentos_edge_updates_bilayer_20261015_v6/` (ignorada por
git). Evidencia autoritativa: `manifest.json`, `pilots.json`, `queue_receipt.json`,
`*_s*/result.json`, `summary.json`. No editar fuentes congeladas mientras corre.

## Resultados finales — 2026-10-16

Seis IDs efectivos DONE (para tbg_s0 se usa `:retry1`), seis `result.json`, seis
checkpoints best-val_loss presentes, pasos completos7995/15999 y actualización
de aristas activa. Verificados de nuevo8.306hashes congelados. El resumen se
recalculó desde cada `result.json`: mismas medias, razones y semillas0/1/2 que
`summary.json`; ambos materiales tienen `status=complete`.

Todas las cifras siguientes son medias de tres semillas, en meV. H es
`H_MAE_gauge_meV`; bandas, `band_rmse_aligned_meV`. Razón=media candidato/media
referencia; no media de razones ni selección de la mejor semilla.

| Material | Población | H referencia→7.2 | Cambio H | Bandas referencia→7.2 | Cambio bandas |
|---|---|---:|---:|---:|---:|
| tbg | ID |1,268→1,374|+8,35%|3,174→3,112|−1,95%|
| tbg | OOD |0,957→1,117|+16,77%|1,984→1,834|−7,53%|
| grhbn_bl | ID |1,728→1,756|+1,61%|4,068→5,510|+35,45%|
| grhbn_bl | OOD |0,991→0,938|−5,34%|16,338→14,783|−9,51%|

### Variación y explicación del cambio respecto al resumen parcial

Razones pareadas de bandasOOD por semilla0/1/2:

- tbg:1,135/0,577/1,222; mejora solo una de tres. H OOD:
  0,755/1,295/1,619; empeora en dos, con regresión media16,8%.
- grhbn_bl:1,568/1,008/0,621; mejora solo una de tres. La tercera semilla tiene
  referencia24,870meV y candidato15,453meV, por lo que cambia el balance medio
  parcial de +19,7% con dos semillas a−9,5% con tres. No significa que las dos
  primeras hayan mejorado. H OOD:1,054/0,928/0,877, mejora en dos de tres.
- BandasID grhbn_bl:1,684/1,075/1,355, peores en las tres semillas. Es una
  regresión consistente que no debe ocultarse tras la mediaOOD favorable.

### Coste observado, no comparación de velocidad

Fits exitosos:3039,91s tbg (tres×≈17min) y28181,48s grhbn_bl
(tres×≈2h36min); total **8,67h de duración de fits exitosos**.
Picos allocated/reserved: tbg≤1279,32/2074MiB; grhbn_bl≤3029,61/3530MiB.
No se comparan estos picos con medidas históricas de otro tipo para inferir
ahorro de VRAM, ni la pared con referencias bajo otra carga para inferir velocidad.

La campaña conservó35intentos:6exitosos,28cesiones a cómputo ajeno y el fallo
inicial del timer. Los intervalos de telemetría suman **14,26h supervisadas**,
incluyendo intentos interrumpidos/fallido, pero excluyendo esperas antes de los
fits, evaluaciónCPU y los controles diagnósticos. No son GPU-hours de utilización
efectiva ni la duración total de calendario. VRAM libre mínima85,60%, GPUmáx59°C
en esos intervalos; los procesos ajenos no se interrumpieron. Las cesiones
reinician desde cero y explican pérdida de trabajo/variabilidad de la duración.

### Auditoría y decisión

- Evidencia válida: implementación operativa y comparación exploratoria
  controlada de estas dos recetas/dominios, con referencias y evaluación comunes.
- MODERATE: tres semillas con gran variación y OOD ya observado; no es test
  independiente ni una certificación estadística de diferencias pequeñas.
- MAJOR para una afirmación de mejora general: H OOD tbg+16,8% y bandasID
  grhbn_bl+35,4%, además de bandasOOD peores en dos semillas por material.
- Conclusión:7.2 no muestra una mejora conjunta/robusta que justifique promoción
  general en estas bicapas. Se conserva como opción. Sin defaults nuevos,
  referencias nuevas ni selección de otra receta tras observar OOD.
- No respaldado: mejora universal, de derivadas/fonones, de velocidad o VRAM.
- Siguiente paso de mayor valor: no gastar otra campaña por iniciativa propia;
  si el usuario quiere profundizar, acordar antes un objetivo específico y
  validación independiente. La recuperación por checkpoint frente a cesiones
  es una mejora operativa posible, no implementada ni requerida para interpretar
  estos seis resultados completos.
