# 6.7 — Homonuclear Weight Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for Native execution, or superpowers:subagent-driven-development if explicitly chosen by the user. Execute task-by-task; steps use checkbox syntax.

**Goal:** Evaluar una orientación en los bloques homonucleares certificados sin cambiar la función, los parámetros originales ni los checkpoints.

**Architecture:** Proyectar diferenciablemente los pesos TP/Linear existentes y evaluar N filas, en vez de apilar 2N. El bloque e3nn comparte una ruta de coeficientes entre sus tres APIs; la ruta histórica permanece por defecto y ante cualquier caso no certificado. Los pesos efectivos no se cachean entre pasos.

**Tech Stack:** Python >=3.9, Torch >=2.4.1, e3nn0.4.4 y rango admitido, MACE, Lightning, pytest; sin nuevas dependencias.

**Spec:** [Diseño aprobado](../specs/2026-10-14-homonuclear-weight-fusion-6-7-design.md), aprobado mediante «Apruebo» el2026-10-14.

**Estado del plan:** aprobado con «Hazlo» el 2026-10-14; ejecución Native iniciada en el checkout solicitado, con registro local y sin commits automáticos.

## Global Constraints

- Único repositorio: `/home/christian/repositorios/graph2mat`, rama `g2m-v2`.
- Trabajar en el checkout indicado; no crear otro worktree.
- `fuse_homonuclear=False` por defecto; ninguna promoción automática.
- No cambiar core/NumPy, DataModule, CLI, exports, vecinos, datasets/splits ni colas.
- No trabajar en fonones ni modificar `g2m_autorun.py` u otros repositorios.
- Preservar cambios existentes y checkpoints/configs sin la nueva opción.
- No sustituir Parameter, cambiar sus nombres/formas/orden, comprimir pesos ni alterar grupos de weight decay.
- Sin `detach`, `.data`, modificación in-place de pesos ni caché de pesos proyectados entre pasos.
- No instalar, actualizar ni desinstalar paquetes en el entorno compartido.
- No campañas científicas nuevas; únicamente pruebas y sondas cortas aprobadas por este plan.
- GPU solo libre de cómputo ajeno: allocator como máximo55%, reserva inicial30%, aborto propio con libre<25% o>=80°C.
- No parar procesos ajenos ni encolar si la GPU está ocupada: informar del recurso pendiente.
- Sin commits automáticos en este worktree sucio. Conservar unidades de revisión mediante diffs y logs; commits solo si el usuario los pide.

## Review Focus

1. Escalas/bias o una subclase no lineal: nunca tratarlos como TP+Linear certificados; fallback (Tasks1–2).
2. Q cargado, entrenable o no compatible y deepcopy: recertificar y no reutilizar metadatos obsoletos (Tasks2–3).
3. Varias componentes, segmentos repetidos y grafos vacíos: no mezclar canales ni producir una pérdida distinta por reshaping (Tasks1–3).
4. Hooks y kwargs privados/globales: preservar llamadas normales de módulos y errores originales; no perder un origen de features (Tasks1–3).
5. Autocast, Adam histórico y compilación: mantener gradientes a pesos originales, fallback en precisión no certificada y ausencia de sincronizaciones nuevas (Tasks3–4).

## Mapa de archivos

| Archivo | Responsabilidad |
|---|---|
| `src/graph2mat/bindings/e3nn/modules/edge_operations.py` | Certificado de pesos y cálculo TP/Linear proyectado de la clase exacta `E3nnLinearEdgeBlock` |
| `src/graph2mat/bindings/e3nn/modules/matrixblock.py` | Opción, certificado Q, carga/fallback y selección común de coeficientes |
| `src/graph2mat/bindings/e3nn/modules/tests/test_homonuclear_fusion.py` (nuevo) | Oráculos, proyección, APIs de bloque, gradientes, hooks y fallbacks |
| `src/graph2mat/tools/lightning/tests/test_homonuclear_fusion_model.py` (nuevo) | MACE/Lightning, restauración, Adam, CUDA/AMP y compilación |
| `auditorias_mejoras/bench_homonuclear_fusion.py` (nuevo) | Supervisor reutilizado, validación CUDA y comparación pareada opt-in |
| `auditorias_mejoras/test_bench_homonuclear_fusion.py` (nuevo) | Emparejamiento y rechazo de medidas no comparables |
| `auditorias_mejoras/informe_fusion_homonuclear_20261014.md` (nuevo) | Evidencia de implementación y beneficio separados |
| `auditorias_mejoras/roadmap.md` | Estado real y enlace a resultados, sin borrar historial |

No crear otro módulo de producción para dos transformaciones privadas ni un protocolo extensible de operaciones. Si el alcance de producto necesita superar los dos archivos previstos, explicar el motivo antes de hacerlo.

## Referencia y comandos

HEAD al redactar: `270d17d1f204293a059baa09e8f1b3756981e2bf`, con cambios previos **sin commit**. HEAD no identifica por sí solo la referencia: guardar diff y hashes de fuentes al empezar, sin incluir archivos ajenos en commits.

Intérprete CUDA existente:
`/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python`.
Intérprete floor3.9 existente:
`/tmp/g2m-python39.D7TJAp/full-venv/bin/python`.
Verificar que todavía existen e importan el source correcto antes de usarlos.

En los comandos siguientes, ejecutar siempre desde el checkout Graph2Mat.
Para pytest CPU usar CUDA desactivada y dos hilos. No aplicar `probe_site` a
la suite general: su eliminación del finder editable se reserva a las sondas
que precargan explícitamente extensiones nativas con `compiled_extensions()`.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python -m pytest \
  src/graph2mat/bindings/e3nn/modules/tests/test_coefficient_only_block.py \
  src/graph2mat/tools/lightning/tests/test_coefficient_matrix_training.py -q --tb=short
```

Esperado: referencia certificada pasa; registrar conteos y cualquier fallo real.
Última suite amplia histórica:2550 pasan/25omitidos/5fallos; cuatro HistorySOAPs
sin `dscribe` y un test de profiler recogido antes de modificarlo, posteriormente
revalidado en un gate de161 pruebas. No convertir ese historial en un nuevo
resultado verde. No modificar archivos mientras la futura suite completa corre.

---

## Task 1: Certificado y evaluación de pesos en la operación

**Files:** modificar `edge_operations.py`; crear `test_homonuclear_fusion.py`.

**Interfaces:**

- Consume `TensorProduct.instructions`, `weight_views`, `weight_view_for_instruction`, `irreps_in1/2/out`, `weight`, y sus equivalentes de Linear.
- Produce `E3nnLinearEdgeBlock._prepare_homonuclear_fusion(index, sign) -> Optional[str]`: recibe mapa completo de coeficientes, devuelve None si queda certificado o un motivo de rechazo.
- Produce selector privado `forward(_homonuclear=False, **tuple_kwargs)`. False mantiene el cálculo anterior; True exige preparación y evalúa TP/Linear con pesos proyectados.
- Metadatos en buffers no persistentes de los módulos existentes; sin Parameter nuevo. Definir nombres `_homonuclear_index` y `_homonuclear_sign`; placeholders vacíos cuando no hay certificado para no introducir atributos ausentes en la compilación.

- [ ] Guardar referencia de rama/diff/hashes y ejecutar el comando CPU previo. Leer todos los callers de `E3nnLinearEdgeBlock` y firmas antes de cambiarlo.
- [ ] Escribir primero un test literal escalar que detecte proyección incorrecta, sin reutilizar el proyectador como oráculo:

```python
def test_scalar_projection_has_literal_result():
    operation = E3nnLinearEdgeBlock("2x0e", "1x0e").double()
    with torch.no_grad():
        operation.tensor_products[0].weight.copy_(torch.tensor([1., 2., 3., 4.]))
        operation.linears[0].weight.zero_()
    index = torch.tensor([0])
    sign = torch.tensor([1.], dtype=torch.float64)
    assert operation._prepare_homonuclear_fusion(index, sign) is None
    a = torch.tensor([[2., 1.]], dtype=torch.float64)
    b = torch.tensor([[3., 4.]], dtype=torch.float64)
    actual = operation(_homonuclear=True, node_feats=(a, b))
    # TP normalizado: forward23.5, reverse26.0, media24.75.
    torch.testing.assert_close(actual, torch.tensor([[24.75]], dtype=torch.float64))
```

- [ ] Ejecutar el test y observar RED por la preparación/selección aún inexistentes. Añadir un caso `1o × 1o → 1e` con T=-I: la fase de intercambio es negativa y la proyección debe conservar el producto no nulo, no cancelarlo.
- [ ] Añadir helper **solo de tests** `explicit_coefficients(block, **pairs)`: evaluar la operación original en ambos órdenes, reconstruir cada matriz con Q, transponer explícitamente e incluir el factor1/2. Para comparar coeficientes, proyectar esa matriz con Q. No llamar a la nueva preparación para generar valores esperados.
- [ ] Construir mapas por instrucción con los factores ya normalizados. Antes de indexar, comprobar salida por segmento/m, signos±1 constantes, involución y parejas de caminos. Rechazar modos no `uvw`, pesos no compartidos/internos, bias, Q/mapas incompatibles y factores inversos distintos/no positivos. Ejemplo del núcleo TP tras validar condiciones:

```python
flat = torch.arange(tp.weight_numel, dtype=torch.long)
lookup = {(r.i_in1, r.i_in2, r.i_out): n for n, r in enumerate(tp.instructions)}
indices, signs = [], []
for instruction in tp.instructions:
    reverse = lookup[instruction.i_in2, instruction.i_in1, instruction.i_out]
    view = tp.weight_view_for_instruction(reverse, weight=flat).transpose(0, 1)
    permutation, output_sign = output_maps[instruction.i_out]
    phase = (-1) ** (tp.irreps_in1[instruction.i_in1].ir.l
                    + tp.irreps_in2[instruction.i_in2].ir.l
                    - tp.irreps_out[instruction.i_out].ir.l)
    indices.append(view[..., permutation].reshape(-1))
    signs.append((phase * output_sign).reshape(1, 1, -1)
                 .expand(instruction.path_shape).reshape(-1))
```

`output_maps` es la lista de permutación/signos por multiplicidad, extraída
del mapa de coeficientes por la preparación. Concatenar índices/signos en el
orden original de `weight_views`; comprobar mapa biyectivo e involutivo. Para
Linear conservar instrucción `(i,o)`, permutar solo su multiplicidad de salida
y no añadir la fase CG. Si no hay pesos, usar buffers vacíos sin concatenar
una lista vacía. Preparar en CPU y mover buffers al dispositivo/dtype del peso.

- [ ] Evaluar los módulos existentes con pesos explícitos, sin otro TP/Linear ni cambio de irreps. Núcleo diferenciable:

```python
def projected(layer):
    return (0.5 * layer.weight
            + 0.5 * layer.weight[layer._homonuclear_index] * layer._homonuclear_sign)

term = tp(a, b, weight=projected(tp)) + linear(a + b, weight=projected(linear))
```

Sumar los términos de los orígenes en el mismo orden de `_ordered_inputs`.
Consumir el keyword privado antes de contar/ordenar entradas. Mantener
`operation(...)`, `tp(...)` y `linear(...)` para preservar hooks. No usar el
keyword para alterar permanentemente el estado del módulo.

- [ ] Ampliar RED→GREEN a `s/sp/spd/spf/2s2p1d1f`, componentes1/2, features con segmentos repetidos, N=0/1/3, origen nodo/mensaje/ambos. Comparar cada gradiente con cotangente aleatoria usando `torch.autograd.grad(..., allow_unused=True)`; conservar patrón None y ceros.
- [ ] `gradcheck` pequeño sobre entradas y pesos **originales**: usar `torch.func.functional_call` con el dict de parámetros como argumentos de gradcheck; no pasar solo los pesos proyectados. En float64 comprobar rtol/atol1e-10/1e-12; en fp32 usar5e-5/5e-6 y registrar errores máximos antes de cambiar tolerancias.
- [ ] Verificar que cada rechazo deja preparación desactivada y borra metadatos previos; no capturar cualquier excepción de programación como fallback. Ejecutar `python -m pytest src/graph2mat/bindings/e3nn/modules/tests/test_homonuclear_fusion.py -q` con el intérprete y entorno CPU definidos arriba, más las regresiones de operaciones existentes.

**Salida revisable:** proyección correcta a nivel de operación, función normal intacta y parámetros originales con gradientes. La selección desde MACE todavía no está conectada.

## Task 2: Bloque común, fallback y ciclo de carga

**Files:** modificar `matrixblock.py`; ampliar `test_homonuclear_fusion.py`.

**Interfaces:**

- Consume `_prepare_homonuclear_fusion` de Task1, mapas RTP ya existentes y Q efectivo.
- Añade argumento de constructor `fuse_homonuclear: bool=False`, consumido por el bloque.
- Produce `_configure_homonuclear_fusion() -> None`, ejecutado al construir y tras cargar estado; y `_fused_homonuclear_coefficients(args, kwargs) -> Optional[Tensor]`, común a las tres APIs.
- Estado del bloque `_fused_homonuclear` y motivo de rechazo; no buffers persistentes nuevos.

- [ ] Escribir RED: construir un bloque `symm_transpose=True`, fórmula `ij`, operación exacta Linear, y `fuse_homonuclear=True`. Cargar `state_dict` de una referencia sin flag. Comparar las tres APIs con `explicit_coefficients`; observar el fallo del flag todavía reenviado indebidamente a la operación.
- [ ] Añadir prueba de trabajo real mediante pre-hook de cada TP: con N=3, flagon ve3 filas por llamada y flagoff6. Afirmar que hooks de operación y TP se ejecutan; evitar deducir ahorro a partir de un atributo booleano.
- [ ] Consumir la opción antes de `operation_kwargs`, validar bool y preparar solamente cuando el tipo es exactamente `E3nnLinearEdgeBlock`, `symm_transpose=True`, `symmetry=="ij"` y `output_scaleshift=False`.
- [ ] Validar T contra Q efectivo en CPU: involución, signos, igualdad `Q.transpose(1,2)` con Q permutado/firmado, dimensiones y Q fijo. Expandir índices por componente mediante offsets; no duplicar/mutar las etiquetas.
- [ ] Registrar hook de carga que recertifique **después** de cargar Q y pesos de hijos. El callback usa el módulo recibido, no una closure ligada al objeto original que sobreviva a deepcopy. Un Q no certificado desactiva fusión; no afirmar que se ha certificado coefficient-training para ese Q. Reutilizar el fallback existente de mapas ausentes cuando corresponda, sin reparar otros formatos de checkpoint por anticipación.
- [ ] Conectar la misma selección a las tres APIs; esquema del cálculo común:

```python
coefficients = self._fused_homonuclear_coefficients(args, kwargs)
if coefficients is not None:
    return self.coefficients_to_block(coefficients)  # forward
return super().forward(*args, **kwargs)
```

En `forward_coefficients`, devolver directamente `coefficients`; en
`forward_with_coefficients`, devolver `(coefficients_to_block(coefficients), coefficients)`.
Con None, ejecutar literalmente la ruta anterior. No promediar/transponer
de nuevo la salida ya fusionada. Tensor vacío es éxito, no un sentinel falso.

- [ ] En el helper común, conservar fallback cuando los pesos no sean float32/float64 o haya autocast no certificado dentro del readout. Con readout fp32, el contexto exterior AMP puede seguir activo para MACE, pero el contexto efectivo del readout debe estar desactivado. Consultar solo metadatos/contexto; ninguna reducción de tensores o copia host en cada paso.
- [ ] Emitir aviso una vez por bloque candidato rechazado, solo si se solicitó el flag y `symm_transpose=True`; no avisar por cada bloque heteronuclear que deliberadamente conserva su ruta.

- [ ] Añadir fallbacks: `output_scaleshift` no trivial; `symm_transpose=False`; `ij=ji`/fórmula no estándar; SimpleEdge, LowRank/SK y una subclase de Linear cuyo forward añade una no linealidad; argumentos globales/posicionales; mapas ausentes o modificados. Flagoff debe mantener valores y errores históricos.
- [ ] Añadir un TP real con normalización `path` y varianzas de entrada distintas en caminos inversos: comprobar rechazo por factor distinto, no fingirlo alterando solo texto de instrucciones compiladas.
- [ ] Cargar Q mezclado entre copias de `1o` de una base `sp`, con rotación no conmutante con T; confirmar que se desactiva fusión y la salida matricial coincide con referencia legacy. Cargar después Q compatible y confirmar recertificación. Añadir Q entrenable como rechazo explícito.
- [ ] Comprobar cambios de dtype/dispositivo, parámetros idénticos y buffers derivados no persistentes; ejecutar foco de ambas tareas y `test_coefficient_only_block.py`.

**Salida revisable:** las tres APIs usan una orientación solo cuando corresponde; todos los casos fuera del contrato conservan la referencia.

## Task 3: Modelo completo, checkpoints y Adam

**Files:** crear `test_homonuclear_fusion_model.py`. Producto solo se ajusta en los dos archivos previstos si una regresión identifica allí la causa.

**Interfaces:** consume opción y APIs de Task2; reutiliza `training_inputs` y `data_module_for` de `test_coefficient_matrix_training.py`. Produce pruebas CPU de modelo completo y restauración; no modifica esos helpers existentes.

- [ ] Escribir comparación pareada de MACE antes de ajustar cualquier plumbing. Preparar referencia y candidato con semilla idéntica, estado idéntico y mismo batch:

```python
reference, batch, samples = training_inputs(
    coefficient_training=False, edge_block_readout=E3nnLinearEdgeBlock)
candidate, candidate_batch, _ = training_inputs(
    coefficient_training=False, edge_block_readout=E3nnLinearEdgeBlock,
    edge_block_readout_kwargs={"fuse_homonuclear": True})
candidate.load_state_dict(reference.state_dict(), strict=True)
expected = reference(batch.clone())
actual = candidate(candidate_batch.clone())
for key in ("node_labels", "edge_labels"):
    torch.testing.assert_close(actual[key], expected[key], rtol=1e-10, atol=1e-12)
```

Construir inicialmente en float64 mediante la fixture de dtype por defecto;
no construir Q float32 y castear después como supuesto oráculo float64.

- [ ] Comparar pérdidas y todos los gradientes por nombre; entradas con ambas especies, aristas periódicas y un tipo ausente. Probar layout precalculado y sin layout, `single/sum/mean`, componentes1/2, agrupaciones `point_type/basis_shape/max` y checkpointing. Las agrupaciones sin certificado deben demostrar fallback, no pasar solo porque no había aristas elegibles.
- [ ] Probar coefficient-training con targets existentes: loss/gradientes iguales y prohibir reconstrucción densa durante su paso certificado; `forward_with_coefficients` no evalúa dos TP por orientación. O(3) y permutación de extremos con features/bases reales, sin exigir que un bloque de arista individual sea una matriz simétrica.
- [ ] Verificar mismo `state_dict().keys()`, shapes y orden de `named_parameters`, sin buffers de metadatos persistentes. Cargar on→off y off→on con `strict=True`, config antiguo sin clave, `deepcopy` y checkpoint Lightning propio.
- [ ] Verificar la compilación TorchScript de los submódulos donde la ruta anterior ya la soporta, incluyendo el argumento de pesos explícitos y los buffers derivados. No ampliar soporte TorchScript del módulo raíz por anticipación; ningún fallo nuevo con el flag apagado es aceptable.
- [ ] Comparar continuidad y resume de Adam:2pasos continuos frente a1+save+load+1, a tolerancia float64, incluyendo `step`, `exp_avg`, `exp_avg_sq`, weight decay y patrones de parámetros no usados. Comparar también off/on partiendo del mismo estado Adam histórico no vacío. No recomprimir estados a pesos efectivos.
- [ ] Ejecutar todos los tests nuevos y las familias `test_coefficient_matrix_training.py`, `test_coefficient_target_collate.py`, `test_low_rank_edge_model.py`, `test_slater_koster_model.py` y `test_edge_pairing_and_safety.py`. Registrar cualquier fallo por nombre; no parchear un archivo ajeno para ocultarlo.

**Salida revisable:** integración pública ya existente conserva modelo/checkpoint/optimizador y semántica física; no se ha medido beneficio GPU todavía.

## Task 4: Certificación CUDA/AMP, compilación y empaquetado

**Files:** ampliar `test_homonuclear_fusion_model.py`; crear `bench_homonuclear_fusion.py` con acción `validate`; informe de cierre.

**Interfaces:** consume modelo certificado CPU y supervisor `guarded_command(command, log_path, telemetry_path, env)`, `memory_fraction(total, free)`, `gpu_state()` y `compiled_extensions()` de `bench_pipeline_gpu.py`.

- [ ] Escribir primero tests marcados CUDA con las precisiones `32-true`, `16-mixed`, `bf16-mixed`; skip cuando CUDA/bf16 no existe o no está `G2M_GPU_GUARDED=1`. Reutilizar el patrón de `test_coefficient_cuda_training.py`, no ejecutar GPU al importar la suite.
- [ ] En cada precisión, comparar off/on con `readout_fp32=True`: predicciones, loss y todos los gradientes. fp32:rtol5e-5/atol5e-6; AMP:rtol1e-2/atol1e-3 como límite inicial del contrato existente, reportando errores por tensor. Comprobar que la ruta fusionada sí se usó y no pasó por fallback inadvertidamente.
- [ ] Con autocast activo dentro del readout o parámetros fp16/bf16 puros, demostrar fallback. El guard consulta dtype/contexto, no copia tensores a CPU. No afirmar certificación de esas rutas por una salida finita.
- [ ] Ejecutar Trainerfit/validación/resume cortos y verificar Adam/scaler. Añadir compilación opcional con `torch_compile=True`: comparar outputs/gradientes tras warmup; registrar tiempo/coste de compilación por separado. Si requiere CUDA graphs inestable, no habilitarlo ni tratar un crash como éxito.
- [ ] Implementar `validate` con el mismo supervisor; comprobar procesos CUDA ajenos antes del hijo y usar límite calculado, nunca establecer fracción fija mayor que el presupuesto. El hijo precarga extensiones e importa Graph2Mat antes de e3nn. Ejecución prevista:

```bash
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python \
  auditorias_mejoras/bench_homonuclear_fusion.py validate \
  --out auditorias_mejoras/experimentos_speed_probe_homonuclear_20261014_cuda_v1
```

- [ ] Ejecutar suite general **con fuentes estabilizadas**, sin editar tests/runners mientras corre:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python -m pytest src tests -q --tb=short
```

Registrar los cuatro casos históricos `test_real_extrapolation_with_missing_onsite_and_edge_blocks[False/True-1/2-HistorySOAPs]` si continúa ausente `dscribe`; no instalarlo en el entorno compartido. Cualquier otro fallo se diagnostica y se informa antes del benchmark final.

- [ ] Ejecutar Ruff y mypy-core con sus configuraciones existentes; no ampliar ignores. Floor3.9: tests nuevos y de coeficientes relevantes desde source con el intérprete citado, CUDA desactivada y `PYTHONPATH=src`.
- [ ] Construir wheels offline con `uv build --wheel --offline --python` para ambos intérpretes y `--build-constraints constraints/build-py39.txt`/`build-py312.txt`, en directorios temporales propios. Instalar solo en venvs propios, nunca en el compartido. Ejecutar desde fuera del checkout con `--pyargs` los nuevos tests y las regresiones de bloque/modelo; confirmar importación y extensión nativa de la wheel. Conservar hashes, comandos y conteos por perfil; no usar `--import-mode=importlib` para simular aislamiento.

**Salida revisable:** equivalencia GPU/AMP certificada dentro del alcance y wheels verificadas; recursos ausentes o compilación no disponible quedan declarados, nunca como pruebas aprobadas.

## Task 5: Comparación de rendimiento y cierre independiente

**Files:** completar `bench_homonuclear_fusion.py`; crear `test_bench_homonuclear_fusion.py`, informe y actualizar roadmap. Documentar la opción en el docstring del bloque y del readout, sin crear documentación de otra API.

**Interfaces:**

- CLI `run --rounds N --warmup N --steps N --out PATH`; variantes `off/on`, sistemas `df16/df72/w90/6x6`.
- CLI interna `child --system SYSTEM --variant off|on --round N --warmup N --steps N --fraction F --output PATH`, ejecutada únicamente por el supervisor.
- Reutiliza `bench_setup_df.df_child(args)` para fixture d/f y `bench_pipeline_gpu.child(args)` con `task="coeff", variant="gauge_dense"` para controles reales, sin cambiar esos runners anteriores.
- CLI `validate` de Task4 y `summarize(records)` nuevo: agrupa por sistema, empareja por ronda, rechaza pesos/entradas/configs científicos diferentes y devuelve ratios pared/VRAM. Excluir solo la diferencia de flag del hash de receta comparada; conservar configs completos de ambas variantes.

- [ ] Escribir RED del resumen con datos literales: off10ms/on8ms produce0.8; distinto hash de pesos/entradas, una ronda faltante o un registro abortado deben rechazar comparación. No aceptar silenciosamente la intersección de rondas.
- [ ] Inyectar el flag en la construcción usando una factoría local del hijo que deje intacto `LitMACEMatrixModel.__init__` (Lightning inspecciona su firma). Núcleo de la factoría:

```python
Original = mace_module.LitMACEMatrixModel
def factory(*args, **kwargs):
    options = dict(kwargs.get("edge_block_readout_kwargs") or {})
    options["fuse_homonuclear"] = variant == "on"
    kwargs["edge_block_readout_kwargs"] = options
    return Original(*args, **kwargs)
mace_module.LitMACEMatrixModel = factory
```

Definir `mace_module` importando `graph2mat.tools.lightning.models.mace` y
`variant` como elección del hijo. La sustitución vive solo en el proceso de
sonda; no editar el constructor ni un archivo de entrenamiento/config original.
Capturar el modelo construido para comprobar elegibilidad, pesos iniciales y
filas reales mediante preflight fuera del tramo medido; retirar hooks de conteo
antes del timing. No sustituir la operación por un cálculo de media carga que
no produzca la misma predicción.

Para los controles reales, conservar `variant="gauge_dense"` en los argumentos
del runner reutilizado: `off/on` es la variante de fusión del wrapper, no una
variante reconocida por `bench_pipeline_gpu.child`. Preparar explícitamente:

```python
probe_args = argparse.Namespace(
    task="coeff", system=args.system, variant="gauge_dense", round=args.round,
    warmup=args.warmup, steps=args.steps, fraction=args.fraction,
    output=args.output, profile=False)
pipeline.child(probe_args)
```

`pipeline` es `bench_pipeline_gpu`. Para d/f, llamar a `bench_setup_df.df_child`
con `points={"df16":16,"df72":72}[args.system]` y los mismos campos round,
warmup, steps, fraction y output. Añadir al resultado original los campos
`system`, `fusion_variant` y ronda para el resumen, sin reescribir sus tiempos.

- [ ] Crear directorio de salida exclusivo, manifiesto HEAD+diff/hashes de source/helpers/runner, configs/hashes de entradas, semilla, versiones y copia del runner. Verificar que no cambian durante la campaña. Cada variante/ronda en un proceso nuevo, orden equilibrado mediante `variant_order`.
- [ ] Smoke primero:1ronda,1warmup,2pasos, todos los sistemas. Rechazar outputs no equivalentes, fallback inesperado o error/aborto; conservar los logs fallidos en otro directorio, no mezclarlos con medidas definitivas.
- [ ] Campaña corta:4rondas×2variantes×4sistemas=32hijos,10warmups+30pasos por hijo. Solo optimización breve para medir cómputo, no entrenamientos científicos. Comandos previstos:

```bash
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python \
  auditorias_mejoras/bench_homonuclear_fusion.py run --rounds 1 --warmup 1 --steps 2 \
  --out auditorias_mejoras/experimentos_speed_probe_homonuclear_20261014_smoke_v1
/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python \
  auditorias_mejoras/bench_homonuclear_fusion.py run --rounds 4 --warmup 10 --steps 30 \
  --out auditorias_mejoras/experimentos_speed_probe_homonuclear_20261014_v1
```

- [ ] Pared primaria = paso completo `df72`; incluir proyección de pesos, backward y Adam. Reportar por separado datos/H2D en controles reales, CUDA forward/backward, allocated/reserved, memoria de metadatos, y construcción/preparación. Picos antes del profiler, sin vaciar caché entre pasos. Perfilar kernels solo después del timing; no sumar scopes anidados ni usar el viejo scope exterior incompleto del backward.
- [ ] Comprobar mismas muestras/pesos/pérdidas iniciales por par; ratios de cada par y mediana/MAD/rango de4rondas. Aceleración recomendada solo con mediana<1 y las cuatro razones<1; resultado mixto=inconcluso. Cualquier regresión de memoria/control queda visible. No afirmar mejora de precisión ni cambiar default con esas cifras.
- [ ] Ejecutar gates finales con archivos congelados y `git diff --check`; revisar diff de esta tarea separado de los cambios anteriores. Solicitar **una revisión independiente final** del rango/archivos atribuibles a6.7, su matemática, tests y análisis; el revisor no modifica archivos. Resolver hallazgos importantes mediante RED→GREEN y repetir gates afectados. Esta es la única delegación prevista para Native.
- [ ] Completar informe con código/tests, comandos, números observados, fallos, versiones/wheels, análisis pareado, tradeoffs y alcance no certificado. Actualizar6.7 como implementado/certificado o pendiente de recursos, y beneficio favorable/neutro/regresión por separado. Conservar el perfil anterior como historia.

**Salida revisable:** implementación opt-in verificada y beneficio medido sin promesas universales, o resultado negativo documentado. Ninguna campaña/default nuevo.

## Auto-revisión del plan y entrega

- [x] Diseño leído completo; alcance de producto limitado a dos archivos.
- [x] Las cinco clases de Review Focus tienen pruebas asignadas.
- [x] Interfaces privadas definidas antes de usarse; no nuevo protocolo público.
- [x] Referencia independiente, gradientes originales, fallback y checkpoint cubiertos.
- [x] Implementación, recursos y beneficio tienen estados separados.
- [x] Revisión independiente final prevista; no subagentes de implementación por defecto.
- [x] Usuario revisa/aprueba este plan y confirma ejecución Native mediante «Hazlo».

No ejecutar Tasks1–5 antes de la última casilla. La aprobación previa del
diseño no aprueba este documento ni los pasos todavía no presentados.
