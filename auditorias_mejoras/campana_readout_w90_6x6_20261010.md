# Comparación baseline / 7.1 / 7.4 — 2026-10-10

Alcance aprobado por el usuario: reutilizar experimentos históricos de grafeno
w90 y 6x6 y encolar la comparación cuando la GPU esté disponible. No cambia
defaults, pérdidas físicas ni `g2m_autorun.py`; no ejecuta trabajo de fonones.

## Campaña

Directorio local (artefactos grandes excluidos de Git):
`auditorias_mejoras/experimentos_readout_w90_6x6_20261010_v1`.
ID de cola: `readout:experimentos_readout_w90_6x6_20261010_v1`.
Estado: **encolada, pendiente**. Confirmadas una única entrada, conservación
exacta de los bytes previos de la cola y daemon PID 1756952 activo. Recibo:
`experimentos_readout_w90_6x6_20261010_v1/queue_receipt.json`.
En ese instante había 105 entradas listas por delante y la GPU estaba ocupada.
No se promete inicio ni finalización durante esta noche.

18 entrenamientos: 2 sistemas × 3 readouts × semillas 0/1/2.

| Sistema | N train | Batch | Épocas | Actualizaciones |
| --- | ---: | ---: | ---: | ---: |
| w90 | 64 | 32 | 4000 | 8000 |
| 6x6 | 8 | 16 | 16000 | 16000 |

Readouts: `E3nnLinearEdgeBlock`, `E3nnLowRankEdgeBlock(rank=4)` y
`E3nnSlaterKosterEdgeBlock(rank=4)`. Todos FP32, compile/checkpointing desactivados.
Mismos backbone, onsite, etiquetas H/S, pérdida gauge, H0, optimizador, scheduler,
splits y selección del mejor checkpoint por validación de las recetas históricas.
Las variantes tienen distinta capacidad; no se afirma equivalencia matemática.

`manifest.json` conserva configs, rutas resueltas de splits, versiones y SHA256
de fuentes, bases, matrices, H0 y sidecars estructurales, incluidos XV. Rechaza
fugas de geometrías entre train/validation/test de cada sistema. No crea H0 ni
ejecuta SIESTA. Datos históricos y código de evaluación externos solo se leen.
Un cambio en archivos congelados invalida el lanzamiento/resultados: no se
actualizan hashes automáticamente para aceptar cambios.
Manifiesto final comprobado: 7548 archivos congelados, incluidos 309 XV.

## Orden y mediciones

1. Seis smokes GPU (2 pasos, semilla 0).
2. 72 procesos frescos de medición: 2 sistemas × 3 variantes × 12 rondas,
   rotando las seis permutaciones. Lote fijo real; 10 pasos de calentamiento
   y 30 medidos. Backbone/onsite cargados estrictamente del mismo estado y
   orden de muestras con generador independiente del constructor del readout.
3. Los 18 entrenamientos completos, secuenciales.
4. Evaluación CPU de validation/test y OOD registry, AB-MD y moiré. Los H0
   están precalculados; no se modifica su caché.

Telemetría cruda: forward con pérdida, backward y paso con Adam; sincronización
CUDA en los límites. Excluye collate/H2D y comprobaciones de gradientes.
Picos allocated/reserved separados, reseteados tras warmup. Guarda dtype, GPU,
versión Torch y configuración TF32. `probe_summary.json` aparece antes de los
entrenamientos largos: mediana/MAD de paso y razones pareadas de fases/VRAM.
Verifica igualdad de hashes de lote y estados comunes antes de comparar.

Primaria científica: H-MAE alineada al gauge en test, definición histórica en Γ.
`precision_summary.json`: razones pareadas de 3 semillas, media geométrica e
IC95 por bootstrap pareado (10000 remuestreos; semilla 20261010). Promoción exige
razón e IC superior <1 y ningún empeoramiento OOD pareado >5% en H alineada o
bandas alineadas. Tres semillas dan potencia limitada. Conserva resultados
por estructura/semilla y comparación SK/low-rank. No cambia defaults automáticamente.

## Ejecución desatendida

La cola existente recibe una sola entrada idempotente, sin dependencias de
trabajos ajenos fallidos. Reserva declarada 22000 MiB más los márgenes existentes
del autorun. El adaptador espera a que no haya procesos compute GPU ajenos.
Si aparece uno durante un entrenamiento, termina únicamente su propio grupo
de procesos y reintenta en otro directorio; conserva el intento interrumpido.
Esto no es aislamiento exclusivo de hardware: existe la latencia del sondeo
de 2 segundos. No puede impedir que otro usuario lance trabajo entre sondeos.
Un proceso Python ajeno mantiene un contexto CUDA desde hace días: esta política
estricta espera también a que ese contexto se libere, aunque su utilización sea
cero. No se excluye ni termina automáticamente ese proceso.

Reanudar tras un fallo requiere investigar su log y ejecutar de nuevo la misma
campaña; etapas con marcador completado no se repiten. No se promete recuperación
automática de errores científicos, entradas cambiadas o caída del daemon.
El equipo debe permanecer encendido y sin suspensión. El chat no necesita seguir
abierto una vez confirmada la entrada y el daemon.

Comprobar resultados locales:

```bash
cd /home/christian/repositorios/graph2mat
ls auditorias_mejoras/experimentos_readout_w90_6x6_20261010_v1/completed/
```

El log del autorun está bajo el directorio histórico
`Comparison/results/dataset_design_curves_v1/g2m_improvements/autorun/`;
usar el ID de esta campaña, sustituyendo `:` por `__`.
Cada intento guarda además su propio `train.log`, config, CSV y telemetría.

## Verificación previa

- Tests específicos del adaptador, incluyendo cola append-only/idempotente,
  deriva de fuentes durante espera GPU, rutas por intento, emparejamiento de
  pesos/orden de datos y veto científico.
- Cuatro fits reales CPU de dos pasos: w90 baseline/low-rank/SK y 6x6 SK,
  checkpoints y hashes comunes w90 idénticos.
- Recarga + predicción real CPU + restauración H0 + métricas de una estructura
  para las tres variantes w90.
- Probe real CPU SK w90: 40 pasos, 30 muestras medidas y hash fijo de lote.
- Política real de checkpoint completo: un `best-epoch0-step1.ckpt` por `val_loss`
  en fit CPU reducido, conservando la configuración del entrenamiento largo.
- Tests específicos: 14 passed en Python 3.12 y 14 passed en Python 3.9;
  Ruff y `git diff --check` sin errores.
- Evidencia de integración: `/tmp/g2m-campaign-cpu-m_0hosdg/`.
  Copia duradera en el directorio de campaña `preflight_cpu/`; logs de pruebas
  y ciclos RED/GREEN en `verification/`.
- Revisión independiente: sin hallazgos críticos; corregida congelación de
  helpers del predictor y geometrías XV antes del manifiesto final.

La primera suite completa usó por error `PYTHONPATH=src`: diez pruebas DDP
cargaban la instalación antigua al cambiar su cwd. Se interrumpió esa ejecución
(1757 passed, 10 failed, 2 skipped); log `/tmp/g2m-campaign-full-suite.log`.
La prueba DDP aislada pasa al usar la ruta absoluta. Se relanzó toda la suite con
`PYTHONPATH=/home/christian/repositorios/graph2mat/src`: **2610 passed, 2 skipped,
36534 warnings, 862.35 s**, ningún fallo. Log completo conservado en
`verification/g2m-campaign-full-suite-absolute.log` dentro de la campaña.

Pruebas afectadas por ese comando incorrecto, todas de `test_ddp_cpu.py`:

- `test_two_cpu_ranks_cost_batches_keep_coverage_padding_and_epoch_metrics`
  [1-ordered, 2-shuffled, 1-variable, 1-single, 1-workers].
- `test_two_cpu_ranks_train_real_mace_for_multiple_steps` [all_types, missing_types].
- `test_two_cpu_ranks_irreps_training_preserves_dense_updates_and_epoch_metrics`
  [all_types, missing_types, missing_labels].

## Límites

Los smokes CPU no son resultados científicos ni mediciones GPU. Precisión,
velocidad GPU y VRAM siguen sin resultado hasta ejecutar la campaña.
No se mide máximo batch admisible ni AMP/compile; no se declara mejora de gaps
ni error por distancia/especie no calculado por el evaluador heredado.
No hay plazo garantizado: la espera de GPU y la diagonalización CPU OOD pueden
dominar el tiempo total. Las predicciones se conservan y consumen disco.

## Resultado (2026-10-11)

Campaña completa (6 smokes, 72 sondas, 18 entrenamientos, evaluación CPU). `precision_summary.json`,
`probe_summary.json`. Razón frente a baseline, media geométrica pareada de 3 semillas:

| | w90 test H (IC95) | 6x6 test H (IC95) | bandas OOD w90 | bandas OOD 6x6 | paso w90 / 6x6 |
|---|---|---|---|---|---|
| 7.1 rango bajo | 0,83 (0,72–1,07) | 0,94 (0,80–1,08) | ×1,11–1,31 | ×1,37–1,64 | +13 % / +1 % |
| 7.4 Slater–Koster | **0,73 (0,60–0,93)** | 0,91 (0,84–1,01) | ×1,14–1,34 | ×1,51–1,57 | +6 % / +3 % |

H fuera de distribución apenas cambia (0,92–1,08); las bandas alineadas en registry, AB-MD y moiré empeoran
en todos los casos. Ninguno cumple la regla de promoción; defaults sin cambios, ambos siguen opt-in.

## Extensión a bicapas: `tbg` y `grhbn_bl` (2026-10-13)

Se completó la comparación entrenada en la misma familia física que la prueba:
baseline, `lowrank` (7.1), `sk` (7.4) y `nodemix` (1.9), semillas 0/1/2,
para `tbg` (3 × 4 trabajos) y `grhbn_bl` (5 trabajos previos + 7 que completan
la matriz de 12, tres semillas por variante). Configuración y datos congelados descritos
en el traspaso; los resultados son `result.json` en
`MD_vs_AtomicDisplacement/Comparison/results/dataset_design_curves_v1/runs/{bilayer,grhbn_bl}`.

Resumen con la misma convención de `g2m_bilayer_benchmark.py --summary`:
media aritmética de tres semillas y cociente entre medias de cada variante y
baseline emparejada del mismo material. H está alineado al gauge; las bandas son
RMSE alineado del moiré objetivo. No se estimaron intervalos de confianza.

| Material / variante | Validación H (meV; razón) | Moiré H (meV; razón) | Bandas alineadas (meV; razón) | Tiempo medio / pico GPU máximo |
|---|---:|---:|---:|---:|
| `tbg` baseline | 1,266; 1,00 | 0,957; 1,00 | 1,982; 1,00 | 11,4 min / 2696 MiB |
| `tbg` low-rank | 1,356; 1,07 | 1,097; 1,15 | 8,203; 4,14 | 22,4 min / 2572 MiB |
| `tbg` SK | 1,254; 0,99 | 0,976; 1,02 | 7,687; 3,88 | 33,8 min / 3138 MiB |
| `tbg` nodemix | 1,277; 1,01 | 1,066; 1,11 | 9,271; 4,68 | 22,8 min / 3222 MiB |
| `grhbn_bl` baseline | 1,728; 1,00 | 0,991; 1,00 | 16,338; 1,00 | 173,2 min / 4626 MiB |
| `grhbn_bl` low-rank | 1,853; 1,07 | 1,241; 1,25 | 14,363; 0,88 | 177,9 min / 4608 MiB |
| `grhbn_bl` SK | 1,577; 0,91 | 0,863; 0,87 | 16,606; 1,02 | 187,0 min / 4128 MiB |
| `grhbn_bl` nodemix | 1,443; 0,84 | 0,943; 0,95 | 14,692; 0,90 | 171,0 min / 4714 MiB |

Interpretación: la conclusión provisional («la capacidad adicional no ayuda si
el dominio está cubierto y Δ ya es bueno») **no se confirma de forma general**.
En `tbg` los readouts no mejoran H y multiplican por 3,88–4,68 el error de
bandas. En `grhbn_bl`, nodemix mejora las medias H y bandas; SK mejora H, con
bandas neutras; low-rank empeora H pese a mejorar bandas. El efecto depende del
material y la métrica. Son resultados exploratorios de tres semillas sin IC95;
no justifican promoción ni cambio de defaults. Los tiempos y picos son los
registrados por cada entrenamiento; no son una comparación controlada de
rendimiento entre campañas/materiales.
