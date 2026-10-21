# Cierre del pipeline y compatibilidad — 2026-10-14

Plan: `docs/superpowers/plans/2026-10-14-cierre-pipeline-validacion.md`.
Base `270d17d1f204293a059baa09e8f1b3756981e2bf`, rama g2m-v2. En curso;
no interpretar tareas iniciadas como completadas. Defaults, cola y entorno
compartido intactos. Tests y artefactos pertenecen al checkout Graph2Mat.

## 1. Diagnóstico previo al parche

Runner existente ampliado con factorial workers 0/4 × pinning false/true.
La opción `loader` conserva caché segura, H+S, receta, pesos y muestras;
`persistent_workers` true con workers y no aplicable sin ellos. La receta
productiva no cambia. Nueve tests CPU del runner pasan (tres nuevos RED→GREEN).

El supervisor muestrea cada 0,5 s su árbol propio, excluye procesos ajenos y
registra RSS principal/hijos/suma simultánea y PSS agregada cuando está accesible.
Las lecturas /proc son consecutivas, no snapshot atómico; procesos que terminan
entre lecturas se omiten. RSS duplica páginas compartidas, PSS las prorratea;
los picos muestreados son límites inferiores del máximo instantáneo.

Smoke: ocho procesos completos, una ronda, un warmup y dos pasos; no usar sus
tiempos fríos como resultado principal. Artefactos:
`experimentos_speed_probe_pipeline_20261014_loader_smoke`.

Perfil independiente: doce procesos, tareas coeff/sparse, una ronda, warmup1,
steps2, flag `--profile`. Artefactos y perfiles `.partial.*.prof`:
`experimentos_speed_probe_pipeline_20261014_profile`. Sus tiempos perfilados
no se mezclan con las sondas de régimen estable.

### Causa seleccionada para optimización

En 6x6 MSE, collate perfilado tarda 155 ms y projector 148 ms. Hay 9010 llamadas
a `_valid_label_rows`, 27034 reducciones `any`, 18021 `reshape`, 9010 `zeros`
y 9010 conversiones `double`. w90 repite el patrón con 1024 bloques, projector17ms.
`dict.setdefault` evalúa incluso el valor descartado: crea un residual escalar
por bloque, aunque solo existen dos grupos. La validación NaN también recorre
cada bloque y lanza pequeñas operaciones Torch repetidas.

Hipótesis acotada: validar filas por tipo node/edge de una vez y consultar una
suma prefija; crear cada entry una vez y convertir a double después de agrupar.
Sin cache persistente, sin targets obsoletos, sin cambiar algebra ni objetivos.
La prueba estructural existente añadida falla en cuatro combinaciones antes del
parche: espera dos validaciones, observa una por bloque. El rendimiento final
debe compararse contra el método original exacto, no contra una reimplementación.

### Setup H+S: localizado, no modificado

En 6x6 vectorizado, setup perfilado2,813s: `csr_to_block_dict`1,817s y
`_csr_to_block_dict_components`1,797s, con 149464 llamadas `np.unique` por los
bloques/última escritura. Lectura/configuración2,203s (incluye esa conversión),
snapshot/deepcopy~0,261s. Confirma que no es un kernel GPU. Se selecciona
solo projector para esta iteración, como exige el plan; una segunda optimización
sparse necesita investigación y aceptación separadas.

## 2. Comandos y recursos

Python CUDA: `python`.
Usar runner para medidas: carga Python actual y los mismos binarios nativos
instalados, registrando sus hashes; no instala paquetes.

```bash
python auditorias_mejoras/bench_pipeline_gpu.py run --tasks loader \
  --out auditorias_mejoras/experimentos_speed_probe_pipeline_20261014_loader_v1 \
  --rounds 4 --warmup 10 --steps 30
```

VRAM/temperatura bajo la protección existente: allocator≤55%, margen inicial30%,
aborto propio con libre<25% o >=80°C, objetivo20% libre. No es un bloqueo de
asignaciones ajenas. GPU RTX5090 de32607MiB. No se encolan trabajos.

Una primera suite CPU con `probe_site` provoca fallos de procesos secundarios:
su Python intenta importar `_sparse.py` y no tiene Cython runtime. Se conserva
el log, pero no se presenta como regresión productiva ni suite válida. Los
tests CPU usan luego el editable normal, verificado en este checkout y con
extensiones nativas instaladas. `probe_site` queda limitado al runner que las
carga explícitamente. Hay 24 CPU lógicas; las sondas iniciales coinciden con
pruebas CPU de dos hilos y esto limita extrapolar sus tiempos absolutos.

## 3. Puertas científicas

Diseño 3.12: `docs/superpowers/specs/2026-10-14-trajectory-splits-design.md`.
Protocolo 6.4/física: `protocolo_siguientes_validaciones_20261014.md`.
Son propuestas para aprobación, no implementaciones ni órdenes de entrenamiento.

Estado del plan: tareas1–4 ejecutadas con los límites de entorno y suite aquí
declarados; tareas5–6 entregadas como diseño/protocolo, pendientes de aprobación
para implementación/campaña. Cambios sin commit por preferencia del plan; se puede
preparar un commit selectivo tras aprobación, sin incluir archivos previos ajenos.

## 4. DataLoader: resultados factoriales completos

32 procesos, cuatro rondas equilibradas, 10 warmups/30 pasos. Medianas de
medianas por ronda; PSS/RSS son máximos muestreados del árbol, no del host entero.
Todos los hashes de entradas/pesos emparejados y rutas certificadas; resumen
recalculado ordenando las rondas coincide con el JSON guardado.

| Datos | Workers | Pinning | ms/paso | RSS árbol MiB | PSS árbol MiB |
|---|---:|---|---:|---:|---:|
| w90 | 0 | no | 62,61 | 2221 | 2015 |
| w90 | 0 | sí | 63,83 | 2221 | 2016 |
| w90 | 4 | no | 165,91 | 5286 | 2215 |
| w90 | 4 | sí | 142,35 | 5297 | 2326 |
| 6x6 | 0 | no | 124,65 | 2219 | 2042 |
| 6x6 | 0 | sí | 124,65 | 2221 | 2044 |
| 6x6 | 4 | no | 167,27 | 5200 | 2134 |
| 6x6 | 4 | sí | 174,45 | 5204 | 2141 |

En este dataset pequeño y cacheado, cero workers gana incluso aislando pinning.
No extrapolar a lazy o datasets grandes; no cambiar defaults. Pinning tiene
efecto pequeño con cero workers, y con cuatro depende del sistema. RSS de
~5,2GiB con workers no equivale a cinco GiB físicos privados: PSS queda
~2,1–2,3GiB. No comparar estos tiempos absolutos como mejora frente a v2:
son otro experimento y había pruebas CPU concurrentes. Mínimo libre76,407%,
máximo57°C; sin abortos de recursos.

## 5. Compatibilidad CUDA/AMP

`test_coefficient_cuda_training.py`: nueve tests GPU pasan, 21,49s, bajo
`bench_pipeline_gpu.py validate`. Artefactos `experimentos_speed_probe_pipeline_20261014_cuda_v2`.

| Precisión | Error loss máximo | Error gradiente absoluto máximo | Fit/validación/resume/Adam |
|---|---:|---:|---|
| fp32 | 5,9605e-8 | 1,8626e-9 | pasa |
| fp16 mixed | 0 | 2,6971e-6 | pasa, incluido scaler |
| bf16 mixed | 0 | 8,6317e-5 | pasa |

AMP usa `readout_fp32=True`. El control sin ese requisito conserva fallback;
gauge conserva reconstrucción matricial en las tres precisiones. Se comparan
pesos, estado Adam, global_step/epoch, métrica de validación y estado del plugin
de precisión tras dos épocas continuas frente a una+restore+una.
Tolerancias predefinidas fp32 rtol/atol5e-6; AMP rtol1e-2/atol1e-3, sin relajarlas.
El máximo relativo por tensor de gradiente es3,348e-7/0,002413/0,01310,
respectivamente; la tolerancia absoluta también participa en el gate.

La primera ejecución tuvo un fallo del propio test, no de Graph2Mat: esperaba
la clave legacy `native_amp_scaling_state`; Lightning2.6 guarda `MixedPrecision`.
Inspección del connector, corrección para leer qualname real del plugin y
comparación explícita de su scaler. Registro v1 conservado (8 pasan/1 falla);
v2 íntegramente verde. No se certifica precisión física nueva, AMP arbitrario,
otras versiones CUDA ni DDP GPU.

## 6. Gates de código y empaquetado

- Parche productivo limitado a `coefficient_targets.py`, sin API/cache nuevas.
- Prueba estructural cuatro fallos antes del parche; después252 pruebas de foco
  pasan, incluyendo pérdida/gradientes, signed, componentes, NaN, ghost y workers.
- Python3.9.25/Torch2.4.1 CPU:124 pruebas source del projector/collate pasan.
- Wheels3.9/3.12 construidas offline con `uv build --wheel`, backend PEP517 y
  constraints de build existentes. `python -m build` no estaba instalado.
- SHA256 wheel312: `994e5bfce485d1fc04595cf08d51f1d5cdc9b2f86f18430e88a55181871d4fc2`.
- SHA256 wheel39: `0a76844d30134fa8e71ce1098f5fb77861788f8ede8f2dadc36c38ca1257e08f`.
- Instalación propia en `/tmp/g2m-pipeline-wheel.TZE6nf/py{312,39}`; dependencias
  existentes expuestas explícitamente en PYTHONPATH de solo lectura, con wheel
  propia antes de ellas y finder editable desactivado. No se modifican paquetes
  del entorno compartido ni se afirma resolución limpia de todas las dependencias.
- Primer intento de pytest instalado con `--import-mode=importlib` importó dos
  veces un módulo y duplicó el registro de converters. Se repite usando modo
  pytest normal/`--pyargs`, sin cambios al producto para adaptar el test runner.
- Resultado instalado: **252 pasan** en Python3.9/Torch2.4.1 CPU (124,97s)
  y **252 pasan** en Python3.12/Torch2.11 (CPU,120,33s). Se ejecutan tests
  incluidos en cada wheel, desde fuera del repositorio, con su código y
  extensiones nativas propios, sin importar el editable.
- Tests de raíz del repositorio: **50 pasan**, 73,90s.

## 7. Parche de targets: comparación final

Artefactos `experimentos_speed_probe_pipeline_20261014_coeffopt_v1`:40 procesos,
cinco variantes ×dos sistemas ×cuatro rondas,10warmups/30pasos.531s (8min51s,
sin preparación). Orden equilibrado por parejas, pesos/inputs idénticos dentro
de cada familia MSE/gauge y mismos binarios nativos. Resumen recalculado coincide;
hashes de runner/source coinciden con manifiesto tras terminar. Referencia:
`__call__` exacto de `CoefficientMatrixMSEProjector` desde commit270d17d, sustituido
solo dentro de su hijo; constructor y resto del modelo idénticos al candidato.
No usar la tanda anterior v2 como referencia numérica de esta optimización.

| Datos | Coef. original ms/paso | Coef. optimizado | Matricial | Razón optimizado/original (MAD) | Razón optimizado/matricial (MAD) |
|---|---:|---:|---:|---:|---:|
| w90 | 85,52 | 72,56 | 62,44 | 0,859 (0,007) | 1,135 (0,033) |
| 6x6 | 243,43 | 149,41 | 123,55 | 0,614 (0,001) | 1,213 (0,004) |

Medianas de medianas de ronda para tiempos, mediana de razones pareadas para
efectos: **−14,1%/−38,6%** respecto a coeficientes originales. Collate baja
20,80→9,59ms y122,97→29,30ms, w90/6x6. No son medidas de precisión científica.

Allocated original/optimizado idéntico:693,94MiB w90 y5115,20MiB6x6. Reserved
idéntico858/6732MiB. Referencia matricial698,84/5118,41MiB allocated,
858/6542MiB reserved. **El parche no ahorra VRAM.** La suma prefija añade un
array CPU proporcional al número de filas observadas; quitar temporales por
bloque y convertir después del stack reduce otro trabajo/almacenamiento CPU.
No se promete ahorro de RAM ni una caché persistente.

Pérdida y todos los gradientes MSE GPU verificados por proceso. Error absoluto
máximo de gradiente1,1642e-9 w90/1,1642e-10 6x6. Control gauge mantiene fallback,
razón de paso flag/reference1,014(MAD0,009) y1,001(MAD0,0002), sin beneficio
recomendable. Mínimo global libre75,824%, temperatura máxima58°C; cero abortos.

Conclusión: conservar optimización equivalente del projector, pero **no promover
coefficient_training como aceleración**: aún cuesta+13,5%/+21,3% frente a matriz
en este régimen. No cambiar receta/pérdida para aprovecharlo. Una solución con
cache persistente o gauge proyectado requiere diseño y pruebas propios; no se
implementa especulativamente. Las rondas iniciales coinciden con pruebas CPU;
reportar dispersión y limitar extrapolación a otros datasets/máquinas.

## 8. Suite completa y revisión final

Source CPU: **2479 pasan,25 omitidos,4 fallos**,766,14s. Los cuatro son las
combinaciones False/True ×1/2 de
`test_real_extrapolation_with_missing_onsite_and_edge_blocks[...-HistorySOAPs]`,
por ausencia de `dscribe`; coinciden con los antecedentes del traspaso. El test
de mensaje de dependencia opcional sí pasa en el entorno correcto. No instalar
ese extra compartido ni alterar tests para ocultarlos. No declarar toda la
suite verde: no hay regresiones nuevas en las rutas tocadas.

Revisión independiente de solo lectura (`pipeline_closure_review`, Astrahigh):
0Critical/1Important/0Minor. Hallazgo: el supervisor podía dejar workers propios
vivos si moría el líder o si un worker ignoraba TERM. Corregido con limpieza
incondicional TERM/wait/KILL del grupo propio en finally, también al salir el
líder. Dos reproducciones CPU fallan antes y pasan después;14tests del runner
pasan. Nueva validación CUDAv3 con supervisor corregido:9pasan21,18s.
Artefactos de esa repetición en `experimentos_speed_probe_pipeline_20261014_cuda_v3`.

El runner medido se archivó antes del fix en `coeffopt_v1/runner-measured.py`;
su hash coincide con el manifiesto, aunque el runner actual incorpora cleanup.
No cambió el código productivo después de las medidas/wheels/suite completa.
Gate combinado root/runner/foco tras cleanup: **316 pasan**,170,60s; no se
repiten otros12min de rutas productivas no afectadas. No segunda revisión delegada: los tests RED→GREEN
comprueban el hallazgo. Sin hallazgos menores diferidos.

## 9. Decisiones de ejecución y coste si fueran incorrectas

- Checkout indicado/árbol sucio, sin worktree ni commits automáticos: conserva
  preferencia del usuario; exige revisión selectiva antes de integrar.
- Implementación Native, solo una revisión independiente final: cubre lo explicado
  y aceptado; coste extra de un contexto, sin delegación de tareas productivas.
- Ledger directo porque el extractor espera `Task` y el plan usa `Tarea`:
  conserva documento español; menor automatización de seguimiento.
- Baseline con probe_site inválida para subprocess: repetir con editable normal;
  coste de ejecución adicional, evita atribuir23fallos de imports al producto.
- `uv build` aislado en lugar de módulo build ausente: mismo backend/constraints,
  comando distinto. Wheels reutilizan dependencias existentes de solo lectura:
  valida artefacto instalado, no una nueva resolución limpia de dependencias.
- Gate postreview enfocado tras fix exclusivo de auditoría: evita repetir suite
  productiva no modificada; no oculta sus cuatro fallos por dscribe ausente.
- Rendimiento final/suite que el reviewer no pudo certificar mientras corrían:
  verificados por ejecutor al terminar, con40registros/hashes/resumen y log completo.
- Splits: solo diseño sin metadata suficiente, por tanto3.12 sigue pendiente.
- DDP GPU/otras combinaciones AMP/precisión física/OOD: sin evidencia, no se
  certifican. No extrapolar equivalencia numérica a mejora científica.
- Diffs históricos del roadmap/campaña preservados: el árbol sucio no constituye
  un commit único revisado; no atribuir todos sus cambios a esta entrega.
