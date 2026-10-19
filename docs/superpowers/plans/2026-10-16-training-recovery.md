# Training Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task, preserving Native execution previously selected by the user. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recuperar entrenamientos interrumpidos desde épocas completas sin perder estado de optimización ni selección científica, y preparar la propuesta ejecutable de validación 6.4 sin lanzar la campaña.

**Architecture:** Un módulo experimental de recuperación reutiliza checkpoints de Lightning, loaders de Graph2Mat y supervisor de GPU existentes. Generaciones atómicas retienen estado completo/RNG y una copia protegida del best; un runner nuevo integra recuperación y preparación fixed/cost sin modificar campañas cerradas.

**Tech Stack:** Python 3.12 del entorno compartido existente, Torch 2.11, Lightning 2.6.1, sisl 0.16.4, pytest, fcntl, pathlib, hashlib, json, os; sin dependencias nuevas.

**Spec:** [diseño aprobado](../specs/2026-10-16-training-recovery-design.md), aprobado por «Listo, apruebo» el 2026-10-16.

## Global Constraints

- Repositorio: `/home/christian/repositorios/graph2mat`, rama `g2m-v2`, checkout actual.
- Sin commits ni worktrees nuevos. Preservar todos los cambios previos sin limpiar el árbol.
- El entrenamiento normal de Graph2Mat no cambia de comportamiento ni de defaults.
- No se modifica ni se relanza la campaña 7.2 cerrada. Su runner y artefactos congelados quedan intactos.
- Fuera de alcance: autorun, fonones, a-SiO2, dependencias compartidas, código/datasets externos, DDP, recuperación intraépoca exacta y migración de checkpoints históricos.
- Cambios de código previstos confinados a `auditorias_mejoras`. Si aparece un defecto de producto, demostrarlo y exponer el cambio de alcance antes de modificar `src`.
- Conservar allocator≤55%, reserva inicial30%, aborto propio si libre<25% o GPU≥80°C y watchdog de progreso300 s piloto/900 s completo.
- Reutiliza cierre de OWN process group: TERM, espera máxima actual de 10 s, KILL si hace falta. Sin señales ajenas ni dumps periódicos de faulthandler.
- La primera receta certificada usa Adam, CosineAnnealingLR por época, un dispositivo, optimización automática, sin gradient accumulation y fp32. Rechazar fronteras no certificadas.
- Dataset aprobado: h2oclu, splits históricos32/4/3 intactos; H alineado y niveles/gap moleculares. Tres seeds pareadas previstas. Sin monómeros adicionales ni resplits.
- Entrenamientos científicos: presentar trabajos y presupuesto concretos para aprobación antes de lanzar o encolar. Este plan no los autoriza.

## Review Focus

1. Symlink de un checkpoint/best o directorio de recuperación a otra campaña: rechazar sin escribir/borrar fuera del job; Task1.
2. Generación publicada pero índice perdido por caída entre renames: recuperar la última generación íntegra, sin aceptar temporales; Task1.
3. CLI, compilación o sanity check consumen RNG antes del primer batch recuperado: restaurar después de esos preparativos; Tasks3/5.
4. Mejor checkpoint borrado por mejora no confirmada y posterior rollback: recuperar copia protegida y score anteriores; Tasks3/5.
5. Proceso termina después del último commit y antes de fit.json: finalizar desde evidencia sin entrenamiento adicional; Tasks4/5.

## Archivos, comandos y contratos compartidos

Crear:

- `auditorias_mejoras/training_recovery.py`: store, callback y adaptación determinista de loaders.
- `auditorias_mejoras/test_training_recovery.py`: contratos de store, loaders, callback y fallos.
- `auditorias_mejoras/run_cost_batching_comparison.py`: prepare, fit, pilot y supervisor para la nueva campaña, sin enqueue científico en este plan.
- `auditorias_mejoras/test_cost_batching_comparison.py`: integración, contabilidad, permisos y preflight.
- `auditorias_mejoras/recovery_certification.py`: child de certificación CPU/GPU y exportación de estado/trace; no campañas científicas.
- `auditorias_mejoras/test_recovery_certification.py`: pruebas entre procesos nuevos.
- `auditorias_mejoras/informe_recuperacion_entrenamiento_20261016.md`: evidencia y límites.
- `auditorias_mejoras/propuesta_batching_h2oclu_20261016.md`: trabajos/budgets que requerirán autorización.

Actualizar únicamente documentos propios: este plan, diseño si requiere aclaración aprobada, roadmap, traspaso y protocolo6.4. Outputs en directorios nuevos dentro de `auditorias_mejoras`; usar el patrón existente de `.gitignore`, sin ignorar fuentes/tests/informes.

Comandos desde Graph2Mat. En ejemplos, `G2M_RECOVERY_PY` identifica solo este entorno:

```bash
G2M_RECOVERY_PY=/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:auditorias_mejoras OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "$G2M_RECOVERY_PY" -m pytest auditorias_mejoras/test_training_recovery.py -q --tb=short
```

Para hijos CUDA: PYTHONPATH absoluto a `auditorias_mejoras/probe_site`, `src` y
`auditorias_mejoras`; `bench_pipeline_gpu.compiled_extensions()` antes de importar
Graph2Mat/e3nn. No cambiar el entorno compartido ni sus permisos.

API experimental acordada; nombres exactos para las tareas siguientes:

- `RecoveryStore(root: Path, fingerprint: dict)`: huella JSON inmutable de inputs/receta/fuentes/versiones.
- `RecoveryStore.lock()`: context manager flock exclusivo, no bloqueante.
- `RecoveryStore.publish(trainer, *, boundary: dict, best_path: Optional[Path]) -> Path`: generación publicada.
- `RecoveryStore.latest() -> Optional[dict]`: recibo validado con checkpoint y fallback, o None si nunca hubo generación.
- `RecoveryStore.restore_best(receipt: dict) -> Optional[Path]`: best propio validado, o None si no existía.
- `RecoveryDataModule`: misma firma que MatrixDataModule, sin flags públicos nuevos; captura `recovery_seed` desde torch.initial_seed al construirse y adapta train_dataloader/val_dataloader.
- `loader_plan(loader) -> list[list[int]]`: plan estable sin consumir RNG global.
- `capture_rng() -> dict`: Python/NumPy/TorchCPU/CUDA; `restore_rng(state: dict) -> None`, sin reseed posterior.
- `RecoveryCallback(store: RecoveryStore, seed: int, sample_ids: list[str])`: subclase ModelCheckpoint, estado propio y frontera.

Recibo JSON: `format_version=1`, huella canónica, `next_epoch`, `global_step`,
`samples_committed`, `plan_sha256`, `epoch_complete`, `checkpoint`, `best` o null,
hash/tamaño de cada archivo y `fallback` con motivos de rechazos recuperables.
Checkpoint: campos nativos de Lightning más `g2m_recovery` con esos contadores,
huella y RNG; no sustituir loops nativos por contadores inventados.

## Task 1: Store atómico y recuperación de generaciones

**Files:** crear `training_recovery.py` y `test_training_recovery.py`.
**Consumes:** Lightning `trainer.save_checkpoint(path)`, SHA256 existente, huella JSON del job.
**Produces:** `RecoveryStore` y recibo validado según contrato compartido.

- [x] **1.1 RED:** crear fixtures `saved_trainer` (save_checkpoint guarda payload con campos nativos completos y g2m_recovery), `store`, `boundary` y `best_path`. El payload incluye optimizer_states, lr_schedulers, callbacks y loops, no solo pesos. Añadir pruebas de dos commits y corrupción del último:

```python
def test_corrupt_latest_falls_back(store, saved_trainer, boundary):
    first = store.publish(saved_trainer, boundary=boundary, best_path=None)
    later = dict(boundary, next_epoch=boundary['next_epoch'] + 1,
                 global_step=boundary['global_step'] + 4)
    saved_trainer.advance(later)
    second = store.publish(saved_trainer, boundary=later, best_path=None)
    (second / 'state.ckpt').write_bytes(b'truncated')
    receipt = store.latest()
    assert Path(receipt['checkpoint']).parent == first
    assert receipt['fallback']
```

- [x] **1.2 Verificar RED:** comando base, `-k 'corrupt_latest or atomic or fingerprint or symlink or retention'`; fallos deben deberse al contrato ausente, no al entorno/import.
- [x] **1.3 GREEN:** implementar lock, rutas own fail-closed, JSON canónico, save a temporal único, hash/tamaño/carga CPU, recibo y fsync, rename de generación e índice atómico. Publicar solo frontera completa. `latest` escanea generaciones publicadas si índice falta/corrupto, ignora temporales y valida fingerprint antes de deserializar. Incompatibilidad del job detiene; cero generaciones devuelve None; generaciones existentes todas inválidas lanzan ValueError.

`saved_trainer.advance(boundary)` actualiza epoch/global_step/loops y g2m_recovery
de modo consistente con la nueva frontera; no fabricar un payload contradictorio
para probar un store que debe rechazar precisamente esa contradicción.

Usar esta primitiva local para durabilidad, sin modificar el write_json de campañas congeladas:

```python
def fsync_path(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
```

No tratar como commit una publicación fallida antes del rename de generación.
Una generación ya renombrada e íntegra es recuperable aunque falle la actualización
del índice; su frontera no puede depender de que exista latest.json.

- [x] **1.4 Verificar GREEN:** añadir índices truncados, caída en cada fase de publicación mediante monkeypatch de os.replace/fsync, root/generación/best symlink, firma incompatible, doble escritor y tres commits. Dos generaciones válidas retenidas; ningún archivo sentinel externo borrado. `restore_best` valida destino propio/hash, repone copia ausente, rechaza archivo inesperado. Sin best temprano devuelve None.
- [x] **1.5 Registrar:** logs RED/GREEN y hashes de archivos cambiados en el workspace de recuperación; marcar tarea. No commit.

## Task 2: Loaders pareados por época sin RNG global compartido

**Files:** ampliar `training_recovery.py` y `test_training_recovery.py`.
**Consumes:** MatrixDataModule, `_CostBatchSampler`, `RecoveryStore` únicamente para metadata posterior.
**Produces:** `RecoveryDataModule`, `loader_plan`, permutación seed/época y generadores de loader independientes.

- [x] **2.1 RED:** fixture con configs heterogéneas de `bench_batching_coste.inputs`, convertida a RecoveryDataModule sin cambiar las etiquetas. Parametrizar workers0/4 y épocas0/1/2. Comprobar mismo orden concatenado fixed/cost y que consultar longitud/plan/prefetch no consume otro orden:

```python
def test_same_samples_and_order_in_both_arms(fixed_module, cost_module):
    fixed = loader_plan(fixed_module.train_dataloader())
    cost = loader_plan(cost_module.train_dataloader())
    flatten = lambda plan: [index for batch in plan for index in batch]
    assert flatten(fixed) == flatten(cost)
    assert sorted(flatten(fixed)) == list(range(len(fixed_module.train_dataset)))
    assert len(cost) >= len(fixed)
```

- [x] **2.2 Verificar RED:** comando base con `-k 'order or loader or prefetch or sampler'`.
- [x] **2.3 GREEN:** reutilizar `super().train_dataloader()` y su batch_sampler/collate; fijar generador del sampler estándar a `(seed + epoch) % 2**64` antes de materializar su plan y generador distinto del loader a `(seed + epoch + 2**32) % 2**64`. Recargar cada época en ambos brazos. Fijar generador separado de validación; no reseed del modelo. Rechazar dataset no determinista, rotating_pool, sampler personalizado, DDP o configuración no cacheada. `loader_plan` devuelve copia del plan cost estable; para fijo usa generador clonado, sin avanzar el generador que consumirá el loader.

```python
epoch = self.trainer.current_epoch if self.trainer is not None else 0
loader = super().train_dataloader()
sampler = loader.batch_sampler.sampler
sampler.generator = torch.Generator().manual_seed((self.recovery_seed + epoch) % 2**64)
loader.generator = torch.Generator().manual_seed((self.recovery_seed + epoch + 2**32) % 2**64)
```

`recovery_seed` se captura al construir el módulo, no de un RNG cambiado durante fit.
No cambiar un plan cost ya materializado: crear loader nuevo antes de consultarlo.

- [x] **2.4 Verificar GREEN:** loader recién creado para época2 produce mismo orden que uno continuo que llegó a época2; las épocas distintas no repiten siempre la misma permutación. Hash de RNG global antes/después de iterar el loader idéntico en dataset determinista. Datos/collate equivalentes al módulo base, oversize singleton y cobertura intactos. Marcar metadata/huella de la política para no afirmar equivalencia histórica.
- [x] **2.5 Registrar:** resultados y restricción de transforms estocásticas, sin commits.

## Task 3: Callback de frontera, RNG y selección best protegida

**Files:** ampliar `training_recovery.py` y `test_training_recovery.py`.
**Consumes:** store Task1, plan/loader Task2, Lightning callbacks/optimizer/scheduler.
**Produces:** `RecoveryCallback`, `capture_rng`, `restore_rng`, checkpoints completos por época.

- [x] **3.1 RED:** toy LightningModule con Linear+Dropout, pérdida que utiliza además random.random y np.random, Adam y CosineAnnealingLR; dataset determinista con índices visibles. Validación controlada con mejor score temprano y scores posteriores peores. Interrumpir en época1 después de dos batches, reconstruir Trainer/model/datamodule y reanudar desde store.latest. Parametrizar validación cada1 y cada5 épocas. Deben coincidir estados, pérdidas y selección.

```python
def test_rng_roundtrip():
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(state)
    assert random.random() == expected[0]
    assert np.random.random() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], rtol=0, atol=0)
```

- [x] **3.2 Verificar RED:** comando base con `-k 'rng or callback or boundary or best'`; registrar la regresión de progreso/estado, no solamente un import inexistente.
- [x] **3.3 GREEN:** RecoveryCallback subclasifica ModelCheckpoint para participar en los hooks monitorizados. Configurar el best val_loss antes de recovery, con dirpath estable `job/best`. Capturar contadores/plan completo y RNG; publicar en on_train_epoch_end solo después del best y scheduler certificado. Guardar estado adicional en g2m_recovery y restaurarlo en on_load_checkpoint/on_train_start después de preparativos. Usar estado nativo de Lightning para optimizador/scheduler/scaler/loops.

```python
def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
```

Comprobar automáticamente un dispositivo, Adam, cosine por época, acumulación1,
optimización automática y cobertura completa antes de publicar. No serializar
state_dict de recuperación con campos faltantes por tratarse de un primer save.
No publicar max_steps que corta la época. Estado RNG se toma del momento de la
frontera, no del callback inicial ni de la inicialización pareada.

- [x] **3.4 Verificar GREEN:** prueba de mejora del best seguida de fallo antes de commit; fallback repone archivo/score previos. El guardado actual fallido conserva generación anterior. Dos interrupciones y primera interrupción antes del primer commit. Probar consumo RNG artificial en preparación/sanity y rechazo de plateau/accumulation/DDP antes de entrenar. Contadores confirmados no incluyen batches descartados; checkpoint contiene todos los estados requeridos.
- [x] **3.5 Registrar:** frontera exacta de hooks soportada y tolerancias; no afirmar replay intraépoca.

## Task 4: Runner resumible, supervisor y contabilidad acumulativa

**Files:** crear `run_cost_batching_comparison.py`, `test_cost_batching_comparison.py`; ampliar pruebas del store si hace falta, sin tocar wrappers congelados.
**Consumes:** RecoveryStore/DataModule/Callback Tasks1–3 y supervisor existente.
**Produces:** `guarded_fit(config: Path, job_root: Path, pilot=False) -> dict`, `fit(args) -> None`, `attempt_accounting(start, end, receipt) -> dict`; CLI `fit --config --root --attempt --fraction [--checkpoint]` y `pilot --root`.

- [x] **4.1 RED:** test de integración simula una única cesión y dos ejecuciones del hijo; el segundo comando debe incluir el checkpoint de la última generación y no el best. El primer intento registra trabajo descartado; sumas no se sustituyen por el tiempo del último fit.

```python
def test_resume_uses_recovery_not_selected_best(supervised_run):
    result = supervised_run(yield_after_batches=6)
    assert len(result['attempts']) == 2
    assert result['attempts'][1]['resume_checkpoint'].endswith('/state.ckpt')
    assert result['attempts'][0]['repeated_samples_lower_bound'] > 0
    assert result['seconds_all_attempts'] >= result['seconds_successful_fit']
```

Fixture `supervised_run` usa subprocess real corto CPU con comandos controlados y
primitivas GPU simuladas; producción nunca trata ese simulador como cómputo real.

- [x] **4.2 Verificar RED:** pytest del nuevo test runner, `-k 'resume or accounting or final_commit or ownership'`.
- [x] **4.3 GREEN:** mantener lock por job durante supervisor completo. Antes de cada intento verificar huella/artefactos, seleccionar recuperación válida y restaurar best protegido. Si recibo confirma todo el presupuesto, escribir fit.json desde evidencia sin GPU. En otro caso arrancar hijo con ckpt_path en config efectiva; validar config ya parseada de CLI, callback order y ubicaciones de best estables. Reutilizar guard y ForeignGPU mediante wrapper local de gpu_state y finally de restauración, sin editar bench_pipeline_gpu.

```python
receipt = store.latest()
checkpoint = None if receipt is None else receipt['checkpoint']
command = [sys.executable, str(Path(__file__).resolve()), 'fit',
           '--root', str(job_root), '--config', str(config),
           '--attempt', str(attempt), '--fraction', str(fraction)]
if checkpoint is not None:
    command += ['--checkpoint', str(checkpoint)]
```

El hijo fija ckpt_path de fit, no llama load_from_checkpoint para empezar un
entrenamiento nuevo. No reutilizar ComparisonCallback, que rechaza resume.
Heartbeat recoge global_step, época, batch_idx y muestras observadas; atomicidad
no implica que heartbeat sea exacto al caer el proceso. Registrar cotas si falta
última observación. Tiempos de todos los intentos y de recuperación se conservan
fuera del estado rebobinado; errores distintos de ForeignGPU fallan, no retry.

- [x] **4.4 Verificar GREEN:** ausencia de checkpoint al primer intento, múltiples cesiones, checkpoint corrupto y error de configuración. Final commit sin fit.json no llama al guard ni a CUDA. PID/PGID propios con TERM/KILL; señal ajena prohibida; release≤10 s más sondeo bajo condiciones controladas. Guard fallido durante save deja generación previa. Entorno absoluto funciona con cwd externo. Errores guard/OOM/NaN/watchdog visibles y sin bucle automático. Registrar allocated/reserved, host RSS/PSS y sobrecoste de guardado/copia.
- [x] **4.5 Registrar:** prueba del bug original de reinicio desde cero y su corrección, conservando stdout/returncodes/telemetría. Sin queue writes.

## Task 5: Certificación entre procesos, GPU acotada y cierre técnico

**Files:** crear `recovery_certification.py`, `test_recovery_certification.py` e informe técnico; ampliar tests Tasks1–4 para fallos hallados.
**Consumes:** APIs Tasks1–4, fixtures training_inputs/data_module_for y comparación recursiva existente de test_coefficient_cuda_training.
**Produces:** CLI `recovery_certification.py --root PATH --device cpu|cuda --precision 32-true|16-mixed|bf16-mixed`; estado/trace/checkpoints de control y reporte verificable.

- [x] **5.1 RED:** crear child que exporta parámetros/buffers, optimizer, scheduler, loops, callback best, RNG y trace con sample_ids reales/pérdidas. CPU toy con Dropout/random/NumPy y MACE real (fixture s/p/d/f pequeña, coefficient_training=False, dtype float64, Adam+cosine). Para MACE repetir las dos estructuras de training_inputs hasta12 muestras y mantener índices distintos; batch_size3, coste que produzca más batches; asignar val_dataset no vacío para certificar también selección. Ejecutar continuo4 épocas vs dos interrupciones dentro de épocas con mismo max_epochs4/huella; cada ejecución en proceso nuevo, workers0/4, fixed/cost.

```python
def test_cpu_continuity(certified_cpu_pair):
    continuous, resumed = certified_cpu_pair
    assert continuous['admitted_sample_ids'] == resumed['admitted_sample_ids']
    assert continuous['global_step'] == resumed['global_step']
    assert continuous['next_epoch'] == resumed['next_epoch'] == 4
    assert continuous['best_tensor_sha256'] == resumed['best_tensor_sha256']
    assert resumed['repeated_samples_lower_bound'] > 0
    assert continuous['parameter_sha256'] == resumed['parameter_sha256']
```

Además de hashes, comparar recursivamente tensores/estados y pérdidas por batch
admitido. No exigir hashes binarios de todo el pickle (incluyen metadata variable):
comparar hash de tensores del best y su score, y SHA256 de su copia protegida
frente al archivo de origen dentro del mismo job.

- [x] **5.2 Verificar RED:** pytest test_recovery_certification CPU; demostrar un fallo real al omitir resume o RNG en un control negativo, y GREEN con implementación completa. Verificar actual sample_ids de fixtures, no solo índices esperados del sampler. Retener trazas de batches descartados separadas.
- [x] **5.3 GREEN y GPU:** ejecutar por el supervisor existente, solo libre, dos interrupciones propias controladas mediante heartbeat. Smoke fp32/fp16/bf16 y MACE real pequeño con readout_fp32=True cuando corresponda. Comparar pesos, Adam, scheduler, scaler y estado CUDA RNG. CPU fixture determinista exige exactitud; si otra ruta necesita tolerancia, medir primero repetición continua y fijar criterio antes del resume. Nunca relajar tras ver resultados fallidos. No usar ni señalizar procesos ajenos como disparador.
- [x] **5.4 Gates:** ejecutar comandos siguientes, conservando logs íntegros/exit codes:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:auditorias_mejoras OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "$G2M_RECOVERY_PY" -m pytest auditorias_mejoras/test_training_recovery.py auditorias_mejoras/test_cost_batching_comparison.py auditorias_mejoras/test_recovery_certification.py auditorias_mejoras/test_bilayer_edge_updates.py auditorias_mejoras/test_bench_pipeline_gpu.py -q --tb=short
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:auditorias_mejoras OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "$G2M_RECOVERY_PY" -m pytest src tests auditorias_mejoras/test_training_recovery.py auditorias_mejoras/test_cost_batching_comparison.py auditorias_mejoras/test_recovery_certification.py -q --tb=short
/tmp/g2m-quality-deps.UsKznt/bin/ruff check src tests auditorias_mejoras/training_recovery.py auditorias_mejoras/run_cost_batching_comparison.py auditorias_mejoras/recovery_certification.py auditorias_mejoras/test_training_recovery.py auditorias_mejoras/test_cost_batching_comparison.py auditorias_mejoras/test_recovery_certification.py
/tmp/g2m-static-gates.YpKMAH/hooks/repoxop7ia2e/py_env-python3.12/bin/python -m mypy --config-file pyproject.toml
git diff --check
```

Los cuatro HistorySOAPs por dscribe ausente son fallos históricos: identificar
casos exactos y explicar; cualquier fallo adicional requiere diagnóstico. No
instalar paquetes ni presentar la suite como completamente verde.

- [x] **5.5 Revisión Native:** una revisión independiente final del cambio completo,
  conforme a executing-plans/requesting-code-review; no implementadores por tarea.
  Corregir Critical/Important con regresión y rerun afectado. Sin merge/commit/push.
  Informe registra estados restaurados, límites, trabajo recuperado/repetido,
  latencia de cesión, overhead, tolerancias y hashes. Si GPU ocupada, preparar
  todo CPU y declarar smoke pendiente; no cerrar punto1 sin su evidencia requerida.

## Task 6: Preparación de h2oclu y propuesta de campaña6.4

**Files:** ampliar runner/tests Task4, crear propuesta h2oclu; actualizar protocolo,
roadmap y traspaso. No alterar el manifest externo ni las recetas históricas.
**Consumes:** recuperación certificada y preflight h2oclu existente.
**Produces:** CLI `prepare --root PATH` y `pilot --root PATH`; manifest propio con
hashes, configs/initializaciones pareadas, resultados de piloto y propuesta sin cola.

- [x] **6.1 RED:** preparar sobre fixtures locales pequeñas; rechazar referencia,
H0, Δ, basis o versión faltantes/cambiados, salida fuera de Graph2Mat y cualquier
intento de escribir junto a datos externos. Comprobar que prepare/pilot no
importan ni invocan funciones externas que generan etiquetas o escriben cola.

```python
def test_prepare_is_readonly_and_freezes_pair(prepared_campaign):
    before, after, manifest = prepared_campaign
    assert before == after  # hashes de dataset/recetas/cola externas
    assert manifest['split_counts'] == {'train': 32, 'validation': 4, 'test': 3}
    for seed in (0, 1, 2):
        pair = manifest['initial_states'][str(seed)]
        assert pair['fixed_sha256'] == pair['cost_sha256']
```

- [x] **6.2 Verificar RED:** pytest del runner con `-k 'prepare or readonly or frozen or initialization'`; fixtures no dependen de datos externos presentes para pasar la suite.
- [x] **6.3 GREEN:** resolver39 FDF/TSHS/H0,36 Δ y bases/pseudos/configs históricos;
congelar hashes propios y versiones actuales. Config base desde receta histórica,
root_dir/globs relativos preservados y outputs propios. Mismos parámetros iniciales
completos por seed y huella validada antes de cada intento. No reutilizar baselines
históricos para comparar tiempo/precisión de este nuevo protocolo. Mantener
checkpoint menor val_loss y evaluar H físico reconstruido, no confundirlo con Δ.

```python
counts = Counter(row['split'] for row in manifest_rows)
if counts != {'train': 32, 'validation': 4, 'test': 3}:
    raise ValueError('h2oclu frozen split mismatch')
fixed_proxy_ceiling = sum(sorted(training_costs, reverse=True)[:8])
pilot_budgets = sorted({max(1, int(fixed_proxy_ceiling * ratio)) for ratio in (.5, .75)})
```

Costes calculados únicamente sobre train con proxy existente. Pilotos de memoria:
fixed y esos dos presupuestos, seed0, cinco épocas completas, dos rondas en orden
inverso, recuperación activa y mismas inicializaciones. No evaluar predicciones
OOD para escoger presupuesto. Registrar setup/compile/checkpointing/fit y por época,
muestras/updates, allocated/reserved y RAM. Elegir el mayor presupuesto que reduzca
allocated en ambas rondas; si ninguno lo hace, informar resultado operativo nulo
y proponer el menor para estudiar el tradeoff, sin declarar ahorro. No esperar
indefinidamente GPU libre sin entregar primero el trabajo independiente terminado.

- [x] **6.4 Verificar GREEN:** preparar h2oclu real en directorio nuevo propio;
smoke de lectura/restore y piloto acotado solo si GPU libre. Hashes externos antes/
después idénticos. No lanzar entrenamientos científicos por pasar este gate.
- [x] **6.5 Propuesta concreta para el usuario:** presentar seed0/1/2 fixed/cost,
presupuesto seleccionado, muestras y updates previstos, tiempos medidos e intervalo
de duración total incluyendo overhead/repeticiones. Comparación por muestras:
2000 épocas de32 muestras en cada brazo, cosine2000, selección best-val_loss cada5;
seis entrenamientos nuevos. Presentar por separado la vista de tiempo/compute:
no llamarla equivalente ni sustituirla por igualar épocas. Su presupuesto y
scheduler deben quedar explícitos en el protocolo científico antes de pedir
autorización de esos trabajos adicionales. Proponer hasta seis trabajos de esa
segunda vista con su propia receta y coste, sin asumir que ya están autorizados.
Primarias aprobadas: H gauge y niveles/gap ID/OOD; fijar ventana/soporte/
generalized eigensolver y umbrales por referencia física, no por scores de modelos.
Declarar lineage incompleto y test ya observado. Solicitar aprobación del listado
resuelto antes de escribir cola o ejecutar ciencia; si falta una decisión, pedirla.
- [x] **6.6 Handoff:** registrar recuperación técnica cerrada solo si Task5 pasó;
6.4 científico preparado/no ejecutado y Goal incompleto. Referenciar manifest,
pilotos, tests y propuesta, sin inflar porcentaje de roadmap ni declarar mejora.

## Auditoría de cobertura y handoff de ejecución

- Spec§1–3 y restricciones: Tasks1–6, sin cambios a src ni campañas cerradas.
- Spec§4: Tasks3/5 (estado completo, frontera, primera/última época, RNG y scaler).
- Spec§5: Tasks2/5/6 (orden, workers, prefetch e inicialización pareada).
- Spec§6–7: Tasks1/3/5 (atomicidad, corrupción, compatibilidad, retención y best).
- Spec§8–9: Tasks4/5 (ownership, release acotado, retry selectivo y coste completo).
- Spec§10: Tasks1–5, con resultado científico todavía condicionado a autorización.
- Review Focus1–5 tienen regresiones asignadas; revisión final única en Task5.

Estado: **plan técnico ejecutado y verificado Native**, mediante «Apruebo» del usuario.
La aprobación de este plan habilita implementación y certificaciones técnicas,
no encolado/ejecución de la campaña científica. Tras aprobación, leer spec+plan
y executing-plans; tareas1–6cerradas con TDD/revisión/gates, sin commits.
Informe final y propuesta científica en auditorias_mejoras.115tests focales pasan,
GPU/AMP/compile pasan; general2744pasan/39omitidos y solo4HistorySOAPs históricos.
Campaña científica6.4 y Goal no cerrados: seis trabajos concretos esperan permiso.
