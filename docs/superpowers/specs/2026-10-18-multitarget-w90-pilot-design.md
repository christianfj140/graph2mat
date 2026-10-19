# Piloto de coste H-only frente a H/S/DM/EDM en w90

Fecha: 2026-10-18. Estado: diseño aprobado por el usuario con «Apruebo» el
2026-10-19; plan Native aprobado también el2026-10-19, ejecución en curso. Ejecución Native
en el checkout actual de Graph2Mat, rama `g2m-v2`, sin commits ni worktrees.

## 1. Propósito y autoridad

El usuario respondió «Hazlo» a la propuesta de aprobar el protocolo y el
presupuesto de un piloto antes de decidir seis entrenamientos completos.
Este documento concreta esa propuesta, no presume autorización del lote.
El usuario aprobó este diseño el2026-10-19. La autorización corresponde
únicamente al piloto descrito, después de revisar su plan de implementación
y verificar sus controles; no a los seis fits definitivos.

Pregunta del piloto: ¿cuánto cuestan, con la misma exposición y receta, las
actualizaciones H-only y multitarget, y funcionan ambas rutas en CUDA?
No puede demostrar precisión final, convergencia, mejora causal de velocidad,
ahorro de VRAM ni generalización OOD. No reabre el Goal técnico 4.1+7.6.

No se entrenarán las seis variantes definitivas, buscarán hiperparámetros,
añadirán brazos H+S ni cambiarán defaults, pérdidas, checkpoints previos,
datos, splits, dependencias o autorun. Sin fonones ni otros proyectos.

## 2. Recurso autorizado e integridad

Único dataset externo, solo lectura:
`/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1500`.
Usar su `material_basis`, no las rutas externas del YAML histórico. Reutilizar
`checked_path` y `verified_file` de `auditorias_mejoras/audit_multitarget_w90.py`.
Rechazar enlaces externos e includes FDF antes de cargarlos; no abrir fuentes
del otro repositorio. El intérprete/dependencias instaladas ya usadas siguen
siendo herramientas, no autorización para explorar nuevos datos.

Congelar los 1198/150/150 IDs del manifiesto, sus exclusiones y su hash
declarado. No redistribuir muestras. Mantener las alarmas por md_1200 de
validación idéntica a md_0/md_300 de train y por bloques MD compartidos.
Son advertencias, no un bloqueo automático ni permiso de corregir splits.

Contrastar fuentes con el certificado CPU existente y registrar hashes antes
y después. Registrar también HEAD, rama, hashes del código productivo actual,
scripts, configuraciones, versiones y estado dirty. Cualquier cambio de fuente
durante el piloto invalida la comparación; no reparar ni repetir solo un brazo
silenciosamente. Todo artefacto nuevo irá en un directorio nuevo y no
sobrescribible bajo `auditorias_mejoras/` de Graph2Mat.

## 3. Dos brazos y receta fija

Decisiones propuestas por el agente y aprobadas con este diseño para el piloto:

| Campo | Ambos brazos |
|---|---|
|Semilla del piloto|0; no iniciar las semillas 1/2|
|Objetivo|Matrices directas, `sub_point_matrix=False`; no H0 ni DM0|
|Backbone|48x0e+48x1o+48x2e+48x3o; 3 interacciones; correlation3; max_ell3|
|Radial/readout|10 Bessel, cutoff polinómico6; edge hidden4x0e+4x1o+4x2e; TSQ linear node y linear edge históricos|
|Resto de arquitectura|keep_last_layer_irreps=True, lean=True, avg_num_neighbors=1; simétrica, basis_grouping=point_type|
|Opciones nuevas|edge_update_layers=0, readout_per_interaction=False, pair_envelope=False; sin contexto/long_range/coefficient-only/scaleshift/cueq|
|Precisión/ejecución|fp32; compile=False; gradient_checkpointing=False; 1 GPU, sin acumulación de gradiente|
|Datos|batch32, drop_last=False, store_in_memory=True; loader_threads=0, pin_memory=True; sin augmentations ni cost batching|
|Optimizer|Adam del modelo existente; lr0.0018, amsgrad=True, weight decay5e-7 en sus grupos actuales; betas(0.9,0.999), eps1e-8, sin clipping|
|Scheduler|CosineAnnealingLR, T_max4000, eta_min1e-5; una llamada DESPUÉS de cada optimizer.step, nunca por época|

Se parte de la arquitectura del config local de
`experimentos_readout_w90_6x6_20261010_v1/attempts/train-w90-baseline-seed1/attempt-001/`.
No se afirma identidad con sus defaults implícitos ni comparabilidad con sus
resultados antiguos. Resolver y guardar todos los argumentos efectivos de
modelo/datos/optimizer, incluidos los no enumerados; deben coincidir entre
brazos salvo las diferencias de targets descritas aquí. No instalar paquetes.
`loader_threads=0` es una elección para reducir variables del piloto, no una
recomendación de rendimiento para producción.

H-only: etiquetas H+S (`h_and_overlap`, dos componentes), una salida H,
loss `gauge_invariant_mse`, edge_weight1, overlap_weight0. Multitarget: descriptor
H/S/DM/EDM, spin1, cuatro salidas; H usa la misma gauge-MSE y S/DM/EDM usan
elementwise-MSE, todos con edge_weight1. Cualquier peso físico auxiliar,
incluido spectral/electron/Mulliken/energy, queda en cero.

Pesos explícitos multitarget: H=1, S=114.9749324374332,
DM=211.0388115121933, EDM=3.133677534317286. Proceden exclusivamente de las
varianzas train archivadas; son una heurística de escala, no equivalencia de
gradientes ni mejora demostrada. No se ajustarán tras ver el piloto o test.

Inicializar el backbone una vez en CPU y copiar únicamente su estado compartido
por nombres/forma a ambos modelos. Comprobar igualdad exacta de esos tensores
y de features en eval antes de entrenar. No copiar por posición ni cargar
parcialmente un checkpoint histórico. Los heads de diferente anchura no son
idénticos; registrar esa limitación. Comprobar que las dos implementaciones de
H loss dan el mismo resultado sobre predicciones H idénticas y etiquetas reales.

Generar un orden train independiente del RNG de construcción de modelos,
por semilla/época, y reutilizarlo entre brazos; registrar los IDs por update.
1198/batch32 da 38 updates por época, último batch14. El piloto recorre el
orden completo, no repite artificialmente un batch fijo de tamaño32.

## 4. Preflight CPU: soporte y espectro

Reutilizar lectores, packing, collate y reconstrucción de Graph2Mat. Comprobar
antes de GPU que el orden H/S/DM/EDM, las máscaras completas y las referencias
necesarias son los del certificado; ninguna etiqueta entra como feature.

En los 1498 ejemplos representados, reconstruir las etiquetas mediante el
writer en memoria, sin threshold, y comparar con las matrices originales,
por target: error máximo/Frobenius y norma de las contribuciones fuera del
soporte del grafo. Guardar también N y HD−SE reconstruidos frente al original.
Esto cuantifica el suelo de representación, no una corrección del dataset ni
una promesa de que DM/EDM exportadas conservan las identidades exactamente.
Errores de geometría/órdenes/no finitos bloquean. Una cola finita omitida se
documenta, no se oculta ni cambia el cutoff para obtener un mejor resultado.

Congelar para la futura evaluación dos conjuntos, en coordenadas recíprocas:
la malla SCF20×20×1 exacta leída del recurso y el camino Γ(0,0,0) →
K(1/3,2/3,0) → M(1/2,1/2,0) → Γ, con50 intervalos uniformes por segmento y
extremos compartidos una sola vez (151 puntos). K procede del BandLines del
RUN.fdf real, usando1/3 exacto en lugar de su redondeo decimal. No usar Γ solo.
Guardar arrays/pesos y hashes: cualquier detalle de convención de malla debe
resolverlo el lector existente y quedar explícito, no sustituirse por otra malla.

Verificar S_ref sobre la unión de ambos conjuntos para todos los ejemplos,
en espacio completo, sin cortar/regularizar autovalores. Reproducir espectros
de referencia y su suelo tras reconstrucción. Hermiticidad relativa≤1e-8;
Cholesky/eigh sin reparación y mínimo autovalor estrictamente positivo.
Un fallo de SPD o hermiticidad de referencia **detiene todo el intento antes
de lanzar GPU**, no solo su interpretación espectral. Informar el ID y la
causa concreta; no excluir el caso ni continuar como un benchmark de coste
sin nueva decisión del usuario.

La definición futura de gap será un observable de llenado nominal N=8,
no una ocupación DFT banda a banda: cuatro bandas doblemente ocupadas,
gap directo e5(k)−e4(k) y gap indirecto firmado min(e5)−max(e4) en cada
conjunto, sin truncar a cero. Las ocupaciones reales son fraccionarias;
no llamar a este gap molecular HOMO–LUMO ni imponer idempotencia.
Los niveles se compararían tras un único cS por estructura, con el mismo
producto escalar empaquetado edge_weight1 usado por H loss, no un ajuste por k.

El preflight solo usa referencias; no selecciona hiperparámetros con test.
El piloto no infiere ni evalúa predicciones de test. En una campaña posterior
H(S_ref) sería la comparación principal común; H(S_pred) y observables DM/EDM
serían diagnósticos separados. Fallos de SPD de S_pred se contarían con IDs,
sin reparación ni desaparición de sus denominadores.

## 5. Ejecución GPU acotada

Ejecutar secuencialmente H-only y multitarget, semilla0, cada uno con
20 optimizer updates de calentamiento +100 medidos. Ambos avanzan el
scheduler completo de4000 desde el update1, sin comprimirlo a120. No son
checkpoints reutilizables para el lote ni entrenamientos finales de precisión.

Tope **15min en total para ambos brazos**, desde lanzar el primer proceso
hijo hasta terminar ambos, incluyendo arranque, validación y esperas internas.
Preflight CPU anterior separado; limpieza puede añadir como máximo10s.
Si el plazo no permite completar la pareja, conservar parciales y parar.
No aumentar el tope, reducir modelo/batch, reintentar tras OOM ni lanzar
automáticamente una variante adicional. Si GPU está ocupada, informar y
detenerse sin encolar, esperar indefinidamente ni tocar procesos ajenos.

Usar hijos/procesos separados para liberar allocator/contexto entre brazos.
Reutilizar `gpu_state`, `memory_fraction` y `process_tree_memory` de
`bench_pipeline_gpu.py`; no cambiar ese benchmark histórico. La supervisión
mínima propia añadirá plazo máximo y comprobación de procesos de cómputo
ajenos, sin importar ni modificar autorun. Una falta de permisos de telemetría
o de identificación de procesos impide el lanzamiento.

Protecciones: GPU0 sin otros procesos de cómputo; temperatura inicial<75°C;
fracción del allocator=min(0.55, libre/total−0.30), al menos512MiB de margen
asignable. Muestrear cada0.5s; abortar si libre<25%, temperatura≥80°C,
aparece cómputo ajeno, falla la telemetría o se agota el plazo. La terminación
TERM/KILL se limita al grupo del hijo propio, nunca al escritorio ni a otro
job. Liberar nuestra reserva exige terminar el proceso, no pausarlo con STOP.
Estas protecciones buscan conservar al menos20% libre con margen, pero no
garantizan exclusión frente a asignaciones externas entre muestras; explicitar
ese límite en el resultado.

Medición mediante el training_step/backward/Adam reales, no solo forward.
Desactivar logger externo/checkpoint de piloto; conservar métricas internas
de la ruta entrenada. El mecanismo mínimo podrá usar el loop directo del
benchmark existente, sin modificar módulos productivos. Una pasada de
validación completa150 muestras al final de cada brazo comprobará finitud
y equivalencia de la métrica H, sin selección de checkpoint; medirla aparte
y mantenerla dentro del tope15min. No usar test en el piloto.

Por update medido: IDs/tamaño, wall monotónico sincronizado CUDA antes/después,
tiempos de carga/H2D y cómputo si son separables sin cambiar el pipeline,
loss finita, LR y contador de optimizer. No vaciar cache entre updates.
Tras20 warmups sincronizar y resetear solo contadores de pico, no optimizer
ni scheduler. Registrar picos warmup y medidos por separado: allocated y
reserved de PyTorch, máximo uso total/mínimo libre NVML/nvidia-smi, y PSS/RSS
del árbol propio. Contexto CUDA/driver no equivale a allocated PyTorch.

## 6. Entregables, verificaciones y cierre

Implementación posterior mínima: script experimental en `auditorias_mejoras`,
sus tests focales, directorio propio de artefactos y un informe. Reutilizar
helpers; no framework de colas, API nueva ni instalación de dependencias.

Pruebas pequeñas: rutas/hashes protegidos; IDs/splits preservados; equivalencia
H-loss; copia exacta del backbone; mismo orden/último batch; scheduler por
update; límites de plazo/VRAM/temperatura/procesos ajenos sin matar procesos
reales; fallo no finito y no-clobber. Smoke CPU real de ambos brazos y
regresión focal multitarget existente, no repetir toda la suite por docs.
Las pruebas CUDA y el piloto solo después de aprobar diseño/plan.

Artefactos: configuraciones efectivas, manifiesto y hashes, evidencia preflight,
IDs/contadores/LR, logs, telemetría y resumen JSON. Registrar resultado completo,
abortado o bloqueado, nunca convertir un parcial en comparación completa.
Dar medianas/p95/rango de tiempo, throughput ponderado por muestras, tiempo
de preparación/validación y picos de memoria. La pareja es exploratoria:
orden H→joint no contrabalanceado, sin repeticiones independientes ni IC
causal. No declarar mejora estable basándose en estos100updates.

Estimar pared por fit de4000updates por brazo a partir de sus datos medidos,
separando coste de validación cada200updates (y al final), arranque/CPU y
margen explícito de incertidumbre. Una sola validación no caracteriza todos
los checkpoints/guardados: dar rango, no una hora garantizada de finalización.
Los tres fits por brazo, semillas0/1/2, serían una nueva autorización expresa
con presupuesto total tras este informe. Se seleccionarían por la misma H
gauge de validación, no por loss total joint; ese monitor es solo de campaña.

Cerrar tras el informe de coste y recomendación de viabilidad, actualizando
roadmap/traspaso sin promover defaults ni declarar resuelto OOD. No prolongar
el piloto buscando una mejora ni lanzar los seis fits sin preguntar.

## 7. Revisión del diseño

Builder check independiente completado sobre esta propuesta: una pregunta
relevante, resuelta aquí por elección conservadora del agente. Un fallo
espectral de referencia detiene el piloto completo antes de GPU (§4).
No queda ninguna pregunta adicional del revisor. Esta revisión no sustituye
la aprobación del usuario ni demuestra ejecución o rendimiento del piloto.
