# Etiquetas w90 para el comparativo 7.6 — auditoría CPU

Fecha: 2026-10-18. Graph2Mat, `g2m-v2`, HEAD270d17d. Usuario autorizó
explícitamente leer solo el recurso indicado con «Hazlo». No se reabre el Goal
técnico 4.1+7.6 ni se autoriza por ello entrenar, usar GPU o modificar datos.

## Dictamen

**Las 1500 muestras tienen etiquetas compatibles con 7.6**, con procedencia y
consistencia numérica comprobadas en CPU. No se certifica mejora ML, OOD general
ni independencia estadística de los splits. Se preservan sus particiones
manuales, con alarmas informativas por un duplicado train/validación y bloques
MD compartidos; no se modifican ni se bloquean automáticamente entrenamientos.

Recurso autorizado:
`/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1500`.
No se leyeron otros datasets ni el código del otro repositorio. Los enlaces
que escapan de esta raíz se rechazan antes de abrirlos. Entradas FDF con
`%include` no se siguen sin una autorización adicional de sus rutas.

## Evidencia numérica

Script: `audit_multitarget_w90.py`. Resultado completo:
`audit_multitarget_w90_20261018.json`; progreso/diagnóstico:
`audit_multitarget_w90_20261018.log`. Salida0,32.588s, un hilo CPU.
Python3.12.3, NumPy2.4.4, SciPy1.17.1, sisl0.16.4.

| Comprobación | Resultado |
|---|---|
|Etiquetas originales|1500 TSHS/TSDE/HSX; H/S desde TSHS y DM/EDM archivadas desde TSDE|
|Procedencia/integridad|13512 archivos únicos contrastados con hashes declarados cuando existen y vueltos a hashear al finalizar; ninguno cambió|
|SCF original|300/300,900/900,300/300 pasos MD convergidos, con Job completed; los RUN.out por snapshot son cabeceras, no la prueba de convergencia|
|Geometría|Dos C, ocho orbitales; máximo FDF–TSHS nativo:2.6484e−10Å en posiciones y2.5401e−10Å en celda|
|Base/órdenes|Bases y pseudopotenciales locales con hashes coincidentes; ORB_INDX contra orden atómico/especie/n/l/m de la base ion.xml, todas las muestras|
|Spin/packing|No polarizado, H/S/DM/EDM:4canales; máscaras completas en2048etiquetas por muestra (512 por target)|
|Gauge/unidades|Validadores existentes contrastan Ef de TSHS/TSDE a1e−8eV; H/EDM en eV, S adimensional, DM ocupación sisl|
|S(k), espacio completo|400 puntos de la malla20×20×1 por muestra:600000 matrices S(k); mínimo autovalor0.1644395031, máximo condicionamiento15.20763778; sin cortes ni reparación|
|Electrones periódicos|Tr(DS) por suma de imágenes:7.999999999999982–8.000000000000018; coincide con8electrones de los logs, sin factor2 adicional|
|Consistencia energética referenciada|Máximo residuo por orbital de HD−SE:9.0367e−10eV;0fallos de soporte físico desconocido|

La integridad es relativa a los manifiestos archivados y a la comparación
antes/después, no autenticidad frente a una autoridad externa. Los hashes de
las copias/enlaces de split se contrastaron con los de sus muestras originales.
Los13512 son rutas físicas únicas: enlaces internos pueden apuntar al mismo
archivo, no son muestras independientes adicionales. HSX se hasheó, pero los
valores entrenables auditados proceden de TSHS/TSDE, no de una comparación
numérica exhaustiva TSHS–HSX. No se certificaron ocupaciones banda a banda ni
idempotencia; no imponer McWeeny a partir de esta comprobación.
La base Ghost-H está archivada, pero las geometrías auditadas contienen solo
dos C; no se certifica aprendizaje de átomos ghost con estos datos. La carga
de etiquetas sobre el grafo no demuestra reconstrucción exacta de todas las
colas DM/EDM fuera de su soporte. Antes del experimento debe medirse ese techo
de representación y distinguir energía referenciada de la matriz exportada.

Primera ejecución: una guarda del auditor comparó Z del TSHS sin base con Z
de la geometría FDF. El reader asigna especies placeholderH cuando TSHS no
provee esa información; no era una discrepancia de las matrices originales.
Se comprobó la causa, se añadió regresión RED→GREEN y se valida geometría y
número de orbitales con TSHS, y especies/orden con FDF+ORB_INDX+ion.xml.
No se relajó la tolerancia geométrica1e−8Å ni se modificaron datos/producto.
Log inicial conservado en `audit_multitarget_w90_20261018_initial.log`.

## Splits congelados: advertencias, no redistribución

`frozen_split_manifest.json`:1198train,150validación,150test. Dos exclusiones
explícitas por hueco temporal:md_1198 ymd_1349. Hash declarado del split:
`9d56b02ab44e339296e04b4a1f9c3a788f56a2dc85969e87970c10481479e98f`.
Se registra ese identificador y el SHA del fichero completo; no se afirma
haber reproducido su algoritmo externo de canonicalización.

| Bloque de origen | Train | Validación | Test |
|---|---:|---:|---:|
|150K,300frames|300|0|0|
|300K,900frames|898|1|0|
|450K,300frames|0|149|150|

Un grupo de matrices TSHS+TSDE idénticas bit a bit cruza train/validación:
**md_0 ymd_300 en train, md_1200 en validación**, todos frame0 de sus bloques.
TSHS contiene H/S y TSDE DM/EDM, por lo que coincide el cuarteto completo.
No se detecta duplicado exacto train/test ni validación/test con ese criterio.
Esto no excluye semejanza geométrica o correlaciones temporales más generales.

Train/validación comparten el bloque300K; validación/test comparten450K.
El hueco de un frame no demuestra decorrelación. Test450K está fuera de las
temperaturas de train, pero450K ya aparece en validación y no constituye una
trayectoria independiente para selección. Publicar como comparativo sobre
el split histórico, no como certificación OOD general. No se movió/excluyó
ninguna muestra ni se creó otra partición por iniciativa del auditor.

## Comparativo propuesto — todavía sin piloto ni lanzamiento

Dos brazos nuevos, semillas0/1/2 (seis fits), mismos1498snapshots/splits y
mismo presupuesto por optimizer updates. H-only **directa**, no un checkpoint
histórico delta, frente H/S/DM/EDM directas; `sub_point_matrix=False` en ambos.

Arquitectura de partida recuperada del config/hparams local de la campaña
`experimentos_readout_w90_6x6_20261010_v1/attempts/train-w90-baseline-seed1/attempt-001`:
48x0e+48x1o+48x2e+48x3o,3interacciones,correlation3,max_ell3,10Bessel,
cutoff polinómico6, keep_last_layer_irrepsTrue, edge_hidden4x0e+4x1o+4x2e,
E3nnLinearTSQNodeBlock/E3nnLinearEdgeBlock,leanTrue,avg_num_neighbors1,
batch32,fp32,lr1.8e−3,compileFalse,checkpointingFalse.
Es una propuesta de reutilización, no garantía de que todos los defaults
implícitos coincidan: antes de ejecutar se deben resolver/guardar configs
completas actuales y comprobar inicialización del backbone compartido.
No se siguieron las rutas externas de bases/datos escritas en aquel config:
para este experimento la base sería la local ya verificada de este dataset.

H usaría gauge-MSE con S_ref en ambos brazos, edge_weight1 como la receta
histórica. DM/S/EDM usarían MSE separada. Física auxiliar apagada para aislar
la supervisión multitarea; sin nuevas identidades entre predicciones, long_range,
contexto, coefficient-only ni scaleshift. Nada se cambia por defecto en la
librería. La evaluación separaría H con S_ref de reconstrucciones con S_pred.
Antes del piloto se deben congelar también puntos k y definición de niveles/gap;
para w90 no basta el espectro en Γ. La propuesta de bandas debe incluir K y
comprobar allí S_ref/S_pred sin cortes, registrando fallos de SPD predicha en
vez de ocultarlos. La positividad de referencia certificada aquí corresponde
a la malla SCF20×20×1, no a todo punto continuo de la zona de Brillouin.

Mi propuesta de pesos, **no una decisión ya aprobada**, es H=1 y coeficientes
auxiliares fijos por cociente de varianzas de etiquetas exclusivamente train:

| Target | STD train en su unidad | Peso propuesto |
|---|---:|---:|
|H|1.510937292|1|
|S|0.140910988|114.9749324|
|DM|0.104007648|211.0388115|
|EDM|0.853530979|3.133677534|

No es una normalización nueva del modelo ni ajuste con OOD; solo constantes
explícitas de loss calculadas de los momentos archivados en el JSON.
Es una heurística por escala de etiquetas, no garantía de equilibrio de
gradientes ni de mejora H; debe aprobarse como parte de la receta antes del lote.

Checkpoint propuesto por la misma H gauge de validación en ambos brazos,
mediante callbacks por experimento (`val_loss` H-only y
`val/target/hamiltonian/loss` multitarget), no por totales de pérdidas distintos.
No modificar el monitor default de la librería ni cerrar1.15 por esta elección.
Ni test ni observables OOD se usarían para elegir pesos/checkpoint.

Presupuesto candidato:4000optimizer updates por fit, mismo orden de muestras
por semilla y scheduler por update. **No copiar4000épocas** del YAML histórico:
con1198muestras/batch32 una época son38updates y cambiaría mucho la exposición.
El mecanismo concreto de scheduler y todos los hiperparámetros se deben fijar
en el diseño/plan de campaña antes de construir el runner. Ningún fit antiguo
se da por comparable ni se reutiliza como baseline sin demostrarlo.

Pendiente de autorización: diseño final/receta/pesos, recurso OOD independiente
si se exige esa conclusión y piloto GPU con presupuesto explícito. Propuesta
de piloto:20warmups+100updates medidos, tope15min, antes de autorizar el lote
según coste/VRAM observado. Aquí no hay ETA GPU fundada ni trabajos encolados.

## Reproducción y comprobaciones del auditor

Desde Graph2Mat, sin escribir en el dataset:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python \
  auditorias_mejoras/audit_multitarget_w90.py > /ruta/nueva/en/graph2mat/resultado.json
```

El script no recibe rutas de salida ni escribe fuentes: JSON a stdout,
progreso a stderr; elegir una ruta de salida nueva propia. La autorización de
lectura permanece limitada al ROOT fijo del script. Tres controles iniciales
RED→GREEN (ruta con symlink externo, alteración de hash, convergencia SCF por
paso) y la regresión de geometría:4passed,45warnings,0.46s. Ruff correcto.
Suite global posterior más estos cuatro controles:
**terminada:2896passed/20skipped/0fallos,43146warnings,850.60s, exit0**.
`audit_multitarget_w90_20261018_tests.log` es el gate terminal; las20omisiones
son1ejemplo externo ausente,1inductorGPU,9coeficientesCUDA y9supervisorGPU de
fusión no autorizado. No se certifica lo omitido. Ruff de script/tests,
`git diff --check`, rama g2m-v2 y28hashes de producto/tests anteriores correctos.
Solo documentación se actualizó tras ese gate. El primer comando exploratorio
de geometría no se presenta como una prueba del dataset fallida: la regresión
del auditor y sus resultados quedan descritos arriba.

SHA-256 del script ejecutado:
`1ac9b914b7f8a6ddbb7198708cd1bf51804fff02ab9cc5db6f945fd45aa6c463`.
SHA-256 del resultado JSON:
`509d94308a37d26d552449a41dd5506d2f46cdf5f58bef125b638bc220ef40f0`.

Fuente productiva y los28hashes del cierre7.6 intactos. Ninguna GPU, inferencia
de checkpoints, entrenamiento científico, cola/autorun, dependencia, commit,
worktree, corrección del modelo o dato de otro recurso fue utilizado/modificado.
