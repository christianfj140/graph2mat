# 3.12 — Guardrail de splits, solo aviso

Fecha: 2026-10-19. Repositorio `graph2mat`, checkout actual de `g2m-v2`.
Autorización del usuario: «Haz 3.12», referida a la alarma no bloqueante descrita
en la conversación. No ejecuta el antiguo diseño de manifiesto/generador ni
valida estrictamente particiones que el usuario ya construyó manualmente.

## Contrato implementado

`MatrixDataModule.setup` comprueba únicamente los inputs de train/val/test que
ya están disponibles, después del reparto histórico y antes de caché/pool.
No carga un split adicional, no lee etiquetas/matrices y no ejecuta la alarma
en `setup("predict")`. `split_guardrail_report` expone el informe más reciente.

Las evidencias se mantienen separadas:

- Fuentes reutilizadas: ruta canónica, incluidos alias symlink; ZIP por archivo
  y miembro exacto, no por carpeta. Para un ZIP en memoria, objeto del archivo
  y miembro exacto; lectores sin nombre distintos no se equiparan.
- Mismo objeto input en memoria: identidad de objeto, no igualdad numérica.
- `sample_id`, `sha256` o `group_id` explícitos compartidos entre splits.
  Un grupo denota una familia correlacionada definida por el usuario, no una
  trayectoria inferida de nombres. Los hashes son declaraciones, no resultados
  de una lectura adicional ni prueba de integridad certificada por el guardrail.

Las anotaciones opcionales se declaran en el JSON existente:

```json
{
  "train": ["frame0/RUN.fdf"],
  "val": ["frame1/RUN.fdf"],
  "split_metadata": {
    "frame0/RUN.fdf": {"sample_id": "f0", "group_id": "MD1"},
    "frame1/RUN.fdf": {"sample_id": "f1", "group_id": "MD1"}
  }
}
```

Las claves de fuente pueden ser relativas a root_dir o absolutas originales,
también al copiar root a un temporal; se conservan nombres léxicos de symlinks.
Para inputs en memoria: `sample.metadata["split_guardrail"]`, con precedencia
sobre la anotación externa. Identificadores: cadenas no vacías; sha256:64hex.

El ejemplo anterior avisa de un grupo compartido, pero usa exactamente los
splits originales. Los avisos usan logging, no warnings convertibles en error
mediante `-Werror`. El informe no se vuelve a emitir si no cambia; sí se revisa
la metadata en setups posteriores. Las identidades originales sobreviven a
Subset, InMemoryData y RotatingPoolData; el pool se evalúa sobre su dataset padre
completo, no solo los elementos activos. Teardown final libera las referencias.

## Interpretación y límites

- `overlap_detected`: evidencia conocida compartida, por tipo y pareja de splits.
- `not_evaluable`: falta información de grupos, metadata inválida o dataset no
  inspeccionable sin consumirlo. No demuestra contaminación.
- `no_known_overlap`: grupos explícitos disjuntos entre splits comparables;
  **nunca certifica independencia física/estadística**.
- `not_comparable`: menos de dos splits no vacíos inspeccionables. La metadata
  malformada sigue generando un diagnóstico incluso en este caso.
- `independence_certified` siempre es falso. Se informa cobertura de grupos,
  entradas inválidas, splits no inspeccionables y ejemplos acotados por alarma.

No compara geometrías, no usa umbrales de similitud, no prueba separación
distribucional ni detecta copias renombradas sin identidades/hashes declarados.
No verifica hashes ni metadatos declarados. No bloquea, elimina, reordena,
reparte de nuevo, cambia semilla/fracción/defaults o modifica checkpoint policy.
No auditó los datos científicos ni mide mejora de precisión, tiempo o VRAM.

## Verificación

Foco ampliado: **89passed,0fallos**,55warnings,6.32s. Incluye21casos de la nueva
alarma más regresiones existentes de splits, stages, temporales y componentes.
RED inicial:13fallos por funcionalidad ausente. Revisión independiente única:
3Important y1Minor; cuatro regresiones reprodujeron los defectos, más control
sin copia que ya pasaba. Correcciones mínimas: ZIP sin filename, alias léxico,
raíz original y diagnóstico de metadata inválida con un único split.
RED de revisión:4failed/17passed; GREEN ampliado:89passed.

Se conservan listas/identidades y reparto histórico seed32/10% (val[8,15] para
20inputs), tanto lazy como cached. Se prueba ausencia de lectura mediante
dataset trampa, ZIP en memoria abierto, JSON opcional, edición de anotaciones,
etapas predict/fit/test y parent completo del pool. Ruff y `git diff --check`
pasan. La batería completa CPU posterior termina con **3226passed/26skip/
0fallos**,48155warnings,1226.05s (20min26s), salida0. Incluye las21pruebas
nuevas y todas las regresiones del checkout; no se toma del gate interrumpido.
Las omisiones no son pruebas aprobadas de GPU ni una matriz de entornos nuevos.

Comando focal reproducible desde la raíz de graph2mat:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m pytest -q \
src/graph2mat/tools/lightning/tests/test_split_guardrail.py \
src/graph2mat/tools/lightning/tests/test_datamodule_split.py \
src/graph2mat/tools/lightning/tests/test_datamodule_stages.py \
src/graph2mat/tools/lightning/tests/test_datamodule_temporary_roots.py \
src/graph2mat/tools/lightning/tests/test_matrix_components.py
```

Logs completos y ledger: `.superpowers/sdd/2026-10-19-split-guardrail/`.
El gate general usa los overlays de tests ya disponibles (servidor/HTTP/SOAP),
sin instalar dependencias ni modificar el entorno del producto:

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH=/tmp/g2m-server-tests.IcPjmO:/tmp/g2m-http-runtime.LOcBGk:/tmp/g2m-soap-deps.o0Kjat \
python -m pytest -q
```

Se corrigió un comando focal que nombraba un archivo inexistente antes de
ejecutar el gate correcto; no fue un fallo de producto. El primer gate general
se interrumpió para incorporar las regresiones de revisión (191passed/3skip,
KeyboardInterrupt); no cuenta como suite completa aprobada.

Sin GPU, entrenamientos científicos, colas/autorun, nuevos datasets, fonones, SOC,
multi-GPU, commits, worktrees ni cambios de rama. Los cambios anteriores del
usuario y los artefactos científicos se conservan; el nuevo hook cambia
`data.py` respecto a las fuentes congeladas de la campaña7.6, no sus resultados.
