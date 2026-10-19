# Protocolo propuesto de siguientes validaciones

Actualización2026-10-18 posterior:4.1 cerrado para el contrato aprobado, sin
cambios productivos/defaults; focal261 y suiteCPU2732pasan/20skip declarados.
Evidencia en `informe_reconstruccion_no_ortogonal_20261018.md`. El siguiente
paso es aprobar el diseño7.6 escrito, no rehacerH+S ni reabrir campañas4.7.
No se autorizan fits/datasets nuevos. Los estados inferiores que aún sitúan
4.1 como pendiente son historia sustituida por esta actualización.

Estado2026-10-18: validación4.7 h2oclu terminada/auditada, tres candidatos y
21matrices nuevas frente a21fixed, sin promoción. GapOOD−61,66% y niveles−69,25%,
pero H OOD+13,47%/ID+19,93%; ningún experimento nuevo en este cierre. Informe
`campana_nullspace_h2oclu_20261017.md`. El siguiente bloque de decisión continúa
siendo4.1, seguido por objetivos adicionales7.6 cuando haya contrato/datos
compatibles. H+S/TSHS con S y readout_per_interaction/sum/mean ya existen;
no presentarlos como código ausente.1.15 sigue siendo una decisión de política
predeterminada, no ausencia de soporte a ModelCheckpoint personalizado.

Estado actualizado2026-10-17: seis trabajos6.4 DONE/evaluados; comparación
exploratoria por muestras cerrada, sin promoción. Cost reduce allocated15,66%,
reserved25,55%,hostPSS10,24%,pared3,77%; precisión OOD mixta y gap aún incorrecto.
No dataset/default modificado ni nuevos trabajos. [Campaña final](campana_batching_h2oclu_20261016.md),
[auditoría de cierre](cierre_goal_recuperacion_batching_20261017.md).

## 6.4: batching por coste y precisión

Actualización2026-10-16: dataset/observables y plan técnico aprobados: **h2oclu**,
splitmanual32/4/3 congelado, H gauge y niveles/gap moleculares (no bandas
dispersivas). Recuperación técnica cerrada,115tests focales y GPU/AMP/compile.
Seis pilotos finalesv4,5épocas/tworounds propios terminados; candidato326283
reduce allocated10,85% en ambas rondas, reserved1812→1814MiB (sin ahorro).
Propuesta concreta6fits0/1/2 fixed/cost,2000épocas/64000muestras;8000/8139updates.
4–7h exclusivas estimadas, más evaluación/esperas. Seis trabajos autorizados
y encolados; [protocolo aprobado](propuesta_batching_h2oclu_20261016.md).
Hallazgo: scheduler histórico era LRconstante, no cosine del YAML. Comparación
aprobada activa cosine2000 explícito en ambos brazos, con aprobación de
esa receta efectiva. La vista igualtiempo permanece separada/no implementada,
cero trabajos adicionales propuestos por ahora. Los párrafos inferiores son
el planteamiento anterior; dataset, observables y presupuesto de la propuesta
actual sustituyen sus puntos todavía no resueltos. La comparación aprobada por
muestras está terminada; los seis trabajos igualtiempo del planteamiento inferior
no se aprobaron ni ejecutaron. No reabrirlos automáticamente al cerrar este Goal.

Disponibles localmente: configs y muestras congeladas de w90/6x6 de la campaña
`experimentos_readout_w90_6x6_20261010_v1`. Cada familia es uniforme y ya funciona
como control negativo de memoria. La fixture s/p/d/f heterogénea es sintética:
sirve para contrato y picos, no para concluir precisión científica.

Candidato a evaluar: conjunto real de tamaños distintos con base y convenciones
compatibles. Unir w90/6x6 sería un dataset nuevo y podría cambiar el dominio;
no hacerlo automáticamente. Faltan aprobación del conjunto y metadata de grupos
independientes para congelar splits y listar trabajos ejecutables.

### Comparación predefinida

1. Congelar datos, etiquetas, matriz objetivo, base, grupos de origen y splits
   ID/OOD. Guardar hashes y política de selección de checkpoint.
2. Referencia: batching fijo. Candidato: presupuesto de coste elegido mediante
   piloto de memoria, sin escogerlo por su resultado OOD.
3. Al menos tres seeds pareadas con pesos iniciales idénticos y las mismas
   muestras. Conservar pérdida, Adam, scheduler y receta científica. Registrar
   batches, updates, muestras, tiempo y recursos por época.
4. Dos vistas, sin fingir que son equivalentes: (a) igual presupuesto de muestras
   con distinto número de updates; (b) igual presupuesto de tiempo/compute,
   incluyendo overhead y reportando las muestras realmente vistas. Definir el
   scheduler en la unidad de presupuesto correspondiente, antes del lanzamiento.
   Igualar solo épocas no aísla el cambio de trayectoria de optimización.
5. Métricas primarias: H alineado y bandas alineadas ID/OOD, en unidades de la
   campaña; auxiliares: loss, wall time, allocated/reserved, RSS/PSS, updates.
   No elegir retrospectivamente otra métrica para declarar ganador.
6. Ratios por seed, resumen y bootstrap pareado con IC95, indicando poca potencia
   con tres seeds. Promoción: límite superior IC95 de razón <1 en la métrica
   primaria acordada y ninguna regresión OOD >5 % en H ni bandas alineadas.
   Una reducción de memoria puede justificar opción operativa, no default de
   precisión. No ampliar seeds sin nueva autorización de campaña.

### Lista de trabajos condicionada

- Piloto de carga/memoria: fijo y coste, una seed, pocas unidades; sin selección
  por OOD. Estima duración y fija presupuesto candidato.
- Comparativo por muestras: fijo/coste × seeds 0,1,2 = seis entrenamientos.
- Comparativo por tiempo: fijo/coste × seeds 0,1,2 = seis entrenamientos, salvo
  reutilización de referencias estrictamente equivalentes documentada.
- Evaluación ID/OOD sobre checkpoints seleccionados con el mismo criterio.

No son órdenes de cola: dataset, presupuesto, receta, dependencias e IDs no están
aprobados. Pedir autorización con la lista resuelta y duración medida del piloto
antes de encolar. No se puede estimar tiempo total extrapolando 28 minutos de
sondas a entrenamientos de precisión.

## Ampliaciones independientes, por orden de decisión

1. **4.1:** definir qué significa reconstrucción no ortogonal completa en la API
   y qué observables certifican la reconstrucción; leer el soporte H+S existente
   antes de proponer nueva representación. Diseño propio y aprobación.
2. **4.4/4.5 DM:** dataset con S, DM y referencias de carga/Mulliken compatibles;
   validar spin/ocupaciones y soporte periódico antes de entrenar. No imponer
   McWeeny a ocupaciones fraccionarias. Pérdidas ya implementadas no equivalen a
   beneficio experimental demostrado.
3. **7.6:** DM/EDM como targets adicionales solo con etiquetas coincidentes por
   geometría y convención energética explícita. Reutilizar H+S, sin inventar EDM.
4. **8.1 y 8.3:** spin/SOC y especies/ghosts requieren caso concreto y diseños
   propios; no añadir flexibilidad especulativa al pipeline ahora.
5. **3.13:** dos GPU reales para comunicación, métricas y parámetros sin gradiente.
   DDP CPU no sustituye esa prueba; la máquina actual solo anuncia una GPU.

1.15, residual opcional 3.8 y cualquier cambio de defaults siguen requiriendo
decisión del usuario. Los readouts ya comparados no se relanzan automáticamente;
7.7 permanece NO_GO. Trabajo de fonones excluido.
