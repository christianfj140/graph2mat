# Propuesta 6.4 — h2oclu fixed/cost — 2026-10-16

**Cierre2026-10-17:** seis trabajos aprobados terminados y auditados. [Campaña
final](campana_batching_h2oclu_20261016.md): memoria favorable, precisión OOD mixta,
sin promoción. Este documento conserva contrato/previsiones anteriores; los
estados inferiores «incompleto/encolado» son históricos. Vista igualtiempo sigue
separada y sin jobs autorizados; no es requisito nuevo para esta campaña cerrada.

Estado actualizado: **seis trabajos científicos autorizados**, preparación de
lanzamiento completada: **seis encolados2026-10-16 23:40CEST**.
Usuario: «Autorizo esos 6 trabajos», después «Continua».
Incluye la receta cosine explícita presentada abajo. No autoriza trabajos extra.
Manifest final propio: `experimentos_batching_h2oclu_20261016_v4/manifest.json`.
v1 es diagnóstico interrumpido por fallo de medición; v2, medida previa a fixes
de revisión. v3 precede el guard final de resolución/orden de enlaces; v4 repite
todos los pilotos con fuentes definitivas. Ninguno es campaña científica ni
semilla adicional; inicializaciones y costes idénticos entre v2/v3/v4.

## Contrato científico previo a predicciones

Hipótesis: limitar coste de batch puede reducir pico de VRAM sin pérdida de
muestras; precisión/tiempo pueden mejorar o empeorar por distinto agrupamiento
y número de updates. Comparar fixed nuevo vs cost nuevo; no reutilizar tiempos
ni checkpoints históricos como referencias pareadas.

Dataset aprobado sin cambios:32train/4validation/3testOOD, tamaños6/12/24átomos
en train y96en OOD. No monómeros, resplits ni nuevas etiquetas. Datos Δ existentes,
H0 y referencias físicas congelados; evaluación **H=H0+ΔH predicha**, no comparar
MAE del residual con MAE de H físico. Mismos inputs, bases/pseudos H/O y orden
de orbitales. VersiónSIESTA homogénea en39salidas; Smin0,00125637 y condición
máxima4180,48. No generar S/H0/Δ ni modificar scripts externos.

Primarias por estructura y medias de estructuras/seed, separadas ID(validation)
y OOD(test): H-MAE alineado al gauge y error de niveles/gap moleculares.
Validación es selección, no testID independiente. TestOOD ya observado,
lineage explícito incompleto y snapshots correlacionados: evidencia exploratoria.
No declarar independencia de39estructuras ni promoción general.

Política de evaluación predeclarada, sin escoger por scores nuevos:

- S solo referencia, no predicha: diagnósticos espectrales condicionales a S.
- Soporte H: unión del soporte de referencia y reconstrucción predicha, incluyendo
  S para gauge; no promedio sobre todos los ceros implícitos. Reportar también
  soporteF1/falsosceros, MAE/RMSE/Frobenius/p95/p99, on/offsite y pares de especie.
- Gauge físico c=⟨S,Hpred−Href⟩/⟨S,S⟩ sobre ese soporte, Hpred−cS; guardar c.
- Reportar defecto hermítico crudo de H/S. Eigensolver solo si defecto relativo
  ≤1e-8 y λmin(S)>1e-10; proyectar simetría únicamente después de reportarlo.
  Resolver `scipy.linalg.eigh(H,S)` enfloat64, Γ. Sin canonización1e-2 que
  eliminaría direcciones de este S positivo; sensibilidad umbral se informa
  como diagnóstico separado, no se elige tras ver errores.
- Ocupación neutral no polarizada: Nocc=(6NO+NH)/2; gap=LUMO−HOMO. Alinear
  niveles con el mismo c matricial. Ventana de referencia desdeHOMO−2eV hasta
  LUMO+2eV, incluyendo ambos por construcción; ocupados completos auxiliares.
  No usar ventana±2eV alrededor de EF que puede caer dentro de gap vacío.
- Semillas pareadas0/1/2, idénticos pesos/buffers iniciales congelados, orden
  seed/época común. Razones por seed, media y bootstrap pareadoIC95 (poca
  potencia con3seeds); no ampliar réplica sin permiso. Sin cambio de defaults.

## Receta autorizada

MACE histórico3interacciones/correlación3/lmax3/48x0e+48x1o+48x2e+48x3o,
readout linealnode/edge, keepLast, n_output_components1, gauge_invariant_mse,
AdamLR0,0018, FP32, compileTrue, batch_size8, storeTrue,4workers.
Recuperación nueva: loader recargado por época, workers no persistentes y
generadores separados; igual en ambos brazos. Selección best-val_loss cada5épocas.

**Diferencia efectiva importante:** histórico tuvo LRconstante aunque YAML decía
cosine. Propuesta activa explícitamente cosine por época,T_max2000,eta_min1e-5
sobre cinco grupos Adam nativos. Diferencia aprobada con los seis trabajos; no
presentarlo como preservación del scheduler histórico. Mantener LRconstante
necesitaría ampliar el contrato técnico, hoy certificado solo con cosine.

## Pilotos y trabajos por resolver

Techo proxy fijo=sumaochomayorescostes train=435044. Candidatos217522/326283.
Seis pilotos técnicos:fixed y dos costes,seed0,5épocas, dos rondas inversas,
sin evaluar OOD. Inicialización idéntica y recuperación activa. Mayor presupuesto
que reduzca allocated en ambas rondas; si ninguno, resultado operativo nulo y
menor presupuesto propuesto sin declarar ahorro. Resultado: **seleccionado326283**.

| Brazo | Allocated, rondas0/1 MiB | Reserved MiB | Updates/5épocas |
|---|---:|---:|---:|
| fixed |1606,35 / 1606,35|1812 / 1812|20 / 20|
| cost217522 |1206,35 / 1206,35|1540 / 1540|25 / 25|
| cost326283 |1432,02 / 1432,02|1814 / 1814|21 / 21|

Allocated−10,85% en candidato seleccionado; **reserved+2MiB**, sin ahorro de
VRAM reservada ni libre total demostrado para ese candidato. Alternativa menor
ahorra allocated−24,90%/reserved−15,01%, pero hace más updates; criterio previo
elige mayor budget que baja allocated en ambas rondas. No cambiar criterio al
ver reserved. No hay resultado de precisión de estos cinco epochs.

Seis pilotos finales completos en341,32s supervisados,160muestras por piloto.
Setup22,99–23,89s; primer epoch20,42–20,99s, mezclado con compile lazy;
Publicación de recuperación1,81–2,43s/5épocas (no todo guardado nativo); wall de
época puede incluir best al terminar validación. Épocas posteriores: fijo1,027/1,125s medios y
candidato1,132/1,025s. Dos rondas son insuficientes para afirmar velocidad; CPUcompartida
con gates y caché de compilación reutilizada. No aislar compilación por resta.
GPUlibre mínimo29634/32607MiB, Tmax53°C. RSSsuma37,4–37,9GiB sobrecuenta páginas
compartidas de workers; PSSpico4,50–4,67GiB, no evidencia de ahorroRAM.
Telemetría completa por intento y hashes externos antes/después intactos.

Campaña primaria autorizada, sin ampliaciones:

| Trabajos | Seeds | Muestras por trabajo | Scheduler | Selección |
|---|---|---:|---|---|
| fixed_s0/fixed_s1/fixed_s2 |0/1/2|64000 (2000×32)|cosine2000porépoca|menorval_loss,cada5|
| cost_s0/cost_s1/cost_s2 |0/1/2|64000 (2000×32)|cosine2000porépoca|menorval_loss,cada5|

**Trabajos autorizados:** los seis de la tabla, cost326283,
2000épocas cada uno, sin early stopping, nuevos baselines fixed. Inicializaciones
seed0/1/2 congeladas en manifest; mismo orden concatenado por época.
Proxy de train total859916porépoca. Plan sampler nativo simulado2000épocas:
fixed8000updates/seed; cost8139/seed (4–5batchesporépoca), +1,74%; cada brazo
ve64000muestras. El budget217522 haría10030updates/seed y no se selecciona.

Estimación de ejecución exclusiva: **4–7h para los seis fits**, más evaluaciónCPU
y esperas ajenas sin cota. Cálculo central≈4,8h extrapolando épocas posteriores
más guardados, no57s×400. Solo cuatro épocas warm por ronda: intervalo operativo,
no benchmark confiable ni promesa nocturna. Registrar coste real, reanudar desde
frontera y no ampliar jobs/seeds al fallar. Listado y receta cosine explícita
aprobados y encolados. Recibo completo/prefijo verificado y estado real en
[informe de campaña](campana_batching_h2oclu_20261016.md). No ampliar ni repetir
fits; si se reanuda el agente, consultar recibo y estado antes de actuar.

Igualar muestras no iguala actualizaciones ni tiempo: publicar
todo, incluyendo setup/compilación no aislada, checkpointing, cesiones/repeticiones,
espera y evaluaciónCPU. Tiempo supervisado no son GPU-hours de utilización.

Vista igual tiempo/compute permanece separada: no se obtendrá igualando épocas
ni pasos. No hay seis trabajos secundarios aprobados ni scheduler temporal
certificado. Antes de proponerlos, fijar unidad/presupuesto y criterio de parada
al ceder GPU; no inferir permiso para ampliar callback o campaña por este plan.
Por ahora se propone **cero trabajos adicionales** de esa vista, dejando esa
decisión explícita para el usuario después del piloto. Goal sigue incompleto.
