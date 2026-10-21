# 7.1 — Readout de aristas de rango bajo

Estado: implementación opcional **integrada y verificada en CPU**. Evaluación GPU,
AMP CUDA y precisión científica **pendientes por decisión del usuario**.

## Diseño e integración

`E3nnLowRankEdgeBlock` proyecta cada irrep de entrada a `rank` canales en
ambos extremos (proyecciones independientes), acopla canales alineados por
todos los caminos angulares admitidos y mezcla los resultados hacia las
multiplicidades de salida. Cada operación/par aprende una MLP radial sin
biases, con SiLU y gates tanh por camino/canal, más un gate para el término
lineal a ancho original. Las dos direcciones radiales se promedian.

El rango limita la capacidad bilineal; no es algebraicamente equivalente al
TP completo. Conserva caminos angulares, no la libertad de todos sus pesos.
El término lineal no pasa por el cuello de rango. Ningún default cambia:
selección por la clase existente `edge_block_readout` y kwargs opcionales
`edge_block_readout_kwargs={"rank": 4}`. La nueva opción se añade al final de
la firma para preservar argumentos posicionales. Checkpoints previos omiten
esa opción y mantienen las operaciones/claves anteriores.

Reutiliza preprocessing, kwargs radiales, pairing, simetrización, layout,
reconstrucción y ruta de coeficientes existentes; no añade dependencias.
Con agrupación distinta de point_type se comparten pesos según el agrupador
ya existente, no se promete un conjunto independiente por especie.

Se requieren una/dos fuentes pareadas (node/edge), rank entero positivo y
radiales 0e. Se rechazan explícitamente tipos/dimensiones/fuentes incorrectos.
Un radial cero anula ambos términos, incluso con features nodales no nulas.
`MatrixMACE` rechaza explícitamente `hamiltonian_local_context` con este readout:
su contexto de especies/nodos no se anula en el corte. El guard inspecciona los
módulos construidos, también con readouts personalizados, en vez de depender
solo del selector Lightning. Otros radiales/custom preprocessing deben respetar
el contrato de anularse en el corte; no se certifican todas sus combinaciones.

## Verificación

- Referencia previa: `test_hamiltonian_readout.py`, 9 pasan, 156 warnings,
  0.77 s, exit 0.
- RED: 26 pruebas de operación fallan por export ausente, y seis de integración
  por kwargs Lightning ausente. Después GREEN de la operación: 26 pasan.
- En integración se corrigieron fixtures, no rutas ajenas de producto:
  no existen pares B-B en el batch (gradientes None legítimos, tipos agrupados
  reordenados), el layout vive fuera del store PyG y el checkpoint normal
  necesita la metadata real del DataModule.
- Enfocadas: **37 pasan**, 2022 warnings, 25.59 s, exit 0, Python 3.12.
- Piso real: **37 pasan**, 953 warnings, 32.72 s, exit 0, Python 3.9.
- Tras el guard de contexto: **61 pasan**, 2715 warnings, 33.48 s, exit 0,
  Python 3.12; **61 pasan**, 1261 warnings, 43.62 s, exit 0, Python 3.9.
  Son 42 pruebas nuevas y 19 regresiones existentes de contexto Hamiltoniano.
- Ruff semántico en los archivos nuevos/modificados, mypy-core, compilación
  Python 3.9 y diff --check: exit 0, repetidos tras el guard. Black aplicado
  solo a archivos nuevos. mypy-core certifica el core, no todo MACE/e3nn.
- Primera suite completa: 10 fallan, 2236 pasan, 2 omitidos, 28122 warnings,
  609.99 s, exit 1. El comando usó `PYTHONPATH=src` relativo: los procesos DDP
  cambian cwd y cargaron una wheel antigua sin `_CostBatchSampler` ni
  `coefficient_training` (ruta site-packages visible en los tracebacks).
  La ruta absoluta corrige las diez pruebas afectadas (10 pasan/31 deselect,
  76.18 s, exit 0). Segundo full: 2246 pasan, 2 omitidos, 28122 warnings,
  649.90 s. Es histórico, anterior al guard de contexto de la revisión. No se parchea
  producto ajeno para ocultar un error del comando.
- Parser Lightning real: selecciona la clase pública y kwargs rank=4.
- **Suite final del source tras la corrección: 2251 pasan, 2 omitidos,
  28815 warnings, 751.75 s, exit 0**. No se usa el full histórico anterior como
  prueba del source final. Log: `full-cpu-final.log` en el directorio SDD propio.
- Tras commit `5d607b3`, `task-done` repitió las **42 pruebas nuevas**:
  2715 warnings, 37.99 s, exit 0; rango real `7225ca4..5d607b3` en el ledger.

Las pruebas cubren una fórmula escalar independiente con pesos fijados,
rotaciones/reflexiones O(3), múltiples fuentes y orden de kwargs, radiales
cero/continuidad, autograd frente a diferencias finitas, batches vacíos,
intercambio homonuclear y coeficientes C1/C2. MatrixMACE real cubre agrupadores
point_type/max/basis_shape, readouts single/sum/mean, gradientes, layouts,
entrenamiento de dos pasos y checkpoints candidato/legacy. Oráculo orbital
sisl con bases s/p/d/f verifica H(R)=H(-R)^T, rotación/inversión y permutación
en una cadena heteronuclear periódica; filas f no triviales.

## Revisión y decisiones de alcance

Una revisión independiente final contrastó los cinco focos del plan y ejecutó
las 37 pruebas originales (25.48 s, 2022 warnings). Sin hallazgos Critical ni
Minor. Detectó un Important: con contexto Hamiltoniano, reutilización de nodos
y envelope por par, conservar features nodales y desplazar aristas fuera del
corte daba max|H_edge|=0.05483048634328883; sin contexto, cero exacto. Reproducción
CPU propia confirmó ambos valores. No se afirma regresión del readout anterior.

Se rechaza la combinación nueva en el límite MatrixMACE, sin modificar el
contexto ni la física del modelo anterior. Cinco regresiones RED fallaron por
no lanzar ValueError (9.28 s); cubren readout single/per-interaction, reuse_nodes
apagado/encendido, y API directa con módulo de contexto/readout personalizado.
La referencia con contexto sigue aceptada. No se implementa un envelope nuevo
por adelantado ni una segunda variante.

Decisiones registradas: elección de rango inicial delegada por el prompt;
corregir PYTHONPATH del comando DDP, no producto ajeno; revisión única antes
del cierre, seguida por nuevos gates/full del source final. No hay Minor
aplazados. El revisor no juzga independientemente suite completa, piso 3.9,
Ruff/mypy (los verifica el implementador), GPU/AMP/VRAM/cuEquivariance/compile,
precisión/bandas/OOD/rank óptimo, rendimiento con datos reales, combinaciones
long-range/custom preprocessing ni checkpoints históricos externos. Su prueba
física justo alrededor del corte perdió el handle y no cuenta como evidencia;
las pruebas de operación propias cubren cero y continuidad radial. No extender
esas evidencias a todas las configuraciones opcionales.

Comandos de reproducción (desde el repositorio):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/tmp/g2m-python312-full.LHyni1/venv/bin/python -m pytest -q \
  src/graph2mat/bindings/e3nn/modules/tests/test_low_rank_edge_readout.py \
  src/graph2mat/tools/lightning/tests/test_low_rank_edge_model.py \
  --disable-warnings --tb=short

# Mismo comando con /tmp/g2m-python39.D7TJAp/full-venv/bin/python para el piso.
# Añadir test_hamiltonian_context.py del mismo directorio e3nn para las 61 pruebas.
# Suite completa: mismo entorno CPU y python -m pytest -q --disable-warnings --tb=short.
/tmp/g2m-static-gates.YpKMAH/venv/bin/pre-commit run ruff-check --all-files
/tmp/g2m-static-gates.YpKMAH/venv/bin/pre-commit run mypy-core --all-files
git diff --check
```

## Mediciones CPU: beneficio dependiente del tamaño/arquitectura

Python 3.12.3, Torch 2.11.0+cpu, e3nn 0.4.4, MACE 0.3.15,
Lightning 2.6.1; fp32, dos threads. Fixtures propias congeladas C2 s/p/d/f,
periódicas dirigidas, 16/72 puntos (96/432 aristas). Misma secuencia, backbone
y readout de nodos, rank 4, semilla 20261009; tres warmups y 12 rondas alternas.
El paso incluye clone del batch, forward, pérdida, backward y Adam; excluye
carga/collate. Forward/backward se miden también por separado. No son tiempos
de un entrenamiento científico. Host no aislado; suite CPU concurrente durante
parte de las mediciones. Se guardan mediana, min/max, MAD, versiones y hashes en
`resultados_readout_rango_bajo_cpu.json`.

| Mediana CPU (ms), referencia → candidato | 16 puntos | 72 puntos |
|---|---:|---:|
| Forward modelo completo | 42.639 → 28.942 | 133.113 → 52.814 |
| Backward modelo completo | 89.872 → 70.678 | 206.795 → 117.681 |
| Paso completo modelo | 142.172 → 107.995 | 359.644 → 179.918 |
| Paso operación aislada | 1.702 → 2.068 | 3.203 → 3.299 |

En el modelo sintético: paso ≈24 % / 50 % menor (1.32× / 2.00×).
Parámetros totales **526016 → 356400** (−32.2 %), operaciones de arista
**197120 → 27504** (−86.0 %). No convertir esos porcentajes en ahorro de VRAM.

La operación aislada es **más lenta** en el paso: +21.5 % / +3.0 %, pese a
45440 → 2896 parámetros (−93.6 %). Es otra forma de entrada/salida, no un
desglose del tiempo del modelo completo: los módulos pequeños y gates añaden
overhead. No se selecciona solo el caso favorable ni se promete velocidad
universal. Las arquitecturas producen predicciones distintas por diseño.

Fuente medida SHA256:
`2eed3f3a89bce0cc2a542e1c65d4645ee843d53b9a46fb68c2fa3e7cb32146d4`.
Hashes de fixtures, modelos iniciales y versiones completos en el JSON.
El guard de contexto se añadió después de medir; no se repitió el benchmark:
las fixtures usan contexto desactivado y la operación medida conserva ese hash.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/tmp/g2m-python312-full.LHyni1/venv/bin/python \
  auditorias_mejoras/bench_readout_rango_bajo.py --points 72 --rank 4 --rounds 12
# Repetir con --points 16. El script valida outputs/gradientes finitos y fixture inmutable.
```

## Lo que no demuestra este trabajo

No se midió GPU, pico CUDA allocated/reserved, AMP CUDA ni precisión comparada.
Los dos pasos de smoke verifican optimización, no generalización. Cambiar la
capacidad puede degradar H/bandas aunque use menos parámetros. Defaults
anteriores preservados; usar el candidato solo como experimento hasta ejecutar
`protocolo_gpu_precision_7_1.md`, que separa sonda GPU y seis entrenamientos
pareados (tres semillas por arquitectura), con márgenes previos y veto OOD.

## Archivos propios

Commit local de código/tests: **`5d607b3`** (base `7225ca4`), cinco archivos,
641 líneas añadidas/2 retiradas. Documentación, benchmark y resultados quedan
en un commit local separado; consultar `git log -1 --oneline -- auditorias_mejoras/informe_readout_rango_bajo.md`.

- `src/graph2mat/bindings/e3nn/modules/edge_operations.py`: operación y export
  mediante el `__all__` existente; preprocessing/exports superiores reutilizados.
- `src/graph2mat/models/mace.py`: rechazo de contexto incompatible.
- `src/graph2mat/tools/lightning/models/mace.py`: kwargs opt-in serializables.
- `src/graph2mat/bindings/e3nn/modules/tests/test_low_rank_edge_readout.py` y
  `src/graph2mat/tools/lightning/tests/test_low_rank_edge_model.py`: 42 regresiones.
- En `auditorias_mejoras/`: `bench_readout_rango_bajo.py`,
  `resultados_readout_rango_bajo_cpu.json`, `plan_readout_rango_bajo.md`, este
  informe, `protocolo_gpu_precision_7_1.md`, `roadmap.md` y
  `roadmap_precision_rendimiento.md`.

Los logs de ejecución propios se conservan en
`.superpowers/sdd/plan_readout_rango_bajo/` (ignored). No se modifica autorun,
colas, datos originales, entornos compartidos ni cambios ajenos. Sin push/merge.
