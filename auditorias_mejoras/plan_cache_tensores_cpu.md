# Caché CPU existente con ownership seguro — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL `superpowers:executing-plans`;
> TDD directo, mismo checkout y una revisión al final del goal completo.

**Goal:** Reutilizar los tensores de `store_in_memory` sin contaminar accesos
futuros, y retirar el coste de representación medido en la ruta lazy.

**Architecture:** `InMemoryData` ya guarda TorchBasisMatrixData procesados;
no otra caché ni flag. Snapshot por deepcopy al materializar y deepcopy por
acceso, incluyendo metadata/NumPy; mantiene shapes/dtypes/NaN. El chequeo de
NetCharge/FDF se limita a paths (str/os.PathLike/zipfile.Path), no repr de
BasisConfiguration, datos PyG ni Geometry. Rebuild explícito si cambian fuentes
o procesamiento; no invalidación automática ni persistencia en disco.

**Tech Stack:** Python>=3.9, copy.deepcopy, Torch/PyG/NumPy/sisl existentes.

**Spec:** `roadmap_precision_rendimiento.md` Task2, 6.1 y su contrato de ownership.
Este lote no implementa 6.2/6.4 ni decide ganancias GPU o precisión.

## Global Constraints

- Solo Graph2Mat/g2m-v2/mismo checkout, preservar archivos ajenos.
- Sin GPU ajena, autorun, colas, datasets originales, fonones ni shared envs.
- API/flag `store_in_memory=False` intactos; sin nueva opción/dependencia/export.
- Objetos devueltos pasan a ser independientes, no identidad compartida mutable.
- Sin copiar archivos ni cerrar readers del caller; lifecycle existente conserva ownership.
- No afirmar reducción VRAM ni paso GPU con una medida de acceso CPU sintética.

## Review Focus

- Input ya PyG: el snapshot no debe compartir arrays/tensores/metadata con el caller.
- Metadata mutable/arrays NumPy: Data.clone no basta; usar deepcopy sin excepciones silenciosas.
- Workers fork/spawn/persistentes: transferir/mutar muestra no cambia la copia local siguiente.
- ZIP/FDF/spectral_C: el filtro de path no debe retirar NetCharge ni sidecars.
- Store-memory/stages/Subset y CPU-target collate: mismo contenido, batches y pérdidas.
- Tamaño explícito cero debe cargar cero muestras; límites negativos/mayores que
  dataset fallan antes de materializar, no cargar todo por `size or len(dataset)`.

## Diseño y evidencia previa

Subproyecto acotado sobre dataset existente, no arquitectura nueva. La orden
explícita del goal pide concretar/ejecutar sin aprobaciones rutinarias; este plan
no expande el alcance. Alternativas: caché lazy por índice añade invalidación y
duplica InMemoryData; clone PyG no copia arrays NumPy/objetos no Tensor. Elegimos
deepcopy: mayor coste que acceso legacy compartido, pero ownership completo.

Perfil previo CPU con muestras propias periódicas s/p/d/f, C2, 16/72 puntos:
legacy comparte mutable; deepcopy ~0.14ms frente a lazy ~5.9/22.8ms. cProfile
atribuye la mayoría a dataclass repr/arrayprint, por str(item). No es ganancia
implementada: no medir antes y después de este parche como si la caché no existiera.
Primer intento de fixture usó ndarray3D inválido para CSR; corregido a BasisMatrix,
sin cambio de producto. Logs cache-data-profile-16/72 en el registro temporal.
RSS bruto incluye allocator/runtime y no equivale a tensor_bytes exactos.

### Task 1: snapshot, acceso aislado y filtro de paths

**Files:** Modify `src/graph2mat/bindings/torch/data/dataset.py`;
Modify `src/graph2mat/core/data/table.py` y tests/test_table.py únicamente para
preservar estado runtime/radios en deepcopy de AtomicTable, sin cambiar pickle;
Create `src/graph2mat/bindings/torch/data/tests/test_tensor_cache.py`;
Modify `src/graph2mat/tools/lightning/tests/test_datamodule_stages.py`,
`docs/data_loading.rst` y roadmaps; benchmark propio en auditorias_mejoras.

**Interfaces:** Consume TorchBasisMatrixDataset(input_data,processor),
InMemoryData(dataset,size=None), MatrixDataModule(store_in_memory=True).
Produce el mismo tipo de muestra con storage independiente; no API nueva.

- [x] Step 1: RED sample real periódico s/p/d/f C1/2, sym/dir y entradas
  BasisConfiguration/PyG. Comparar con lazy, mutar tensores y nested metadata
  NumPy; siguiente acceso idéntico al oráculo. Cambiar input después del snapshot
  tampoco cambia caché; `.to('meta')` no transforma snapshot. Código central:
  ```python
  cache = InMemoryData(dataset)
  expected = deepcopy(cache[0])
  sample = cache[0]
  sample.point_labels.add_(7)
  sample.metadata['custom']['array'][0] = -9
  torch.testing.assert_close(cache[0].point_labels, expected.point_labels)
  np.testing.assert_array_equal(cache[0].metadata['custom']['array'], expected.metadata['custom']['array'])
  ```
  Expected: fallos por shared storage/metadata, no fixtures inválidos.
  Incluir `InMemoryData(dataset,size=0)` vacío, negativos/exceso ValueError antes
  de leer y truncación válida; cambiar selección `size or len` a `size is None`
  y validar `0<=size<=len(dataset)` antes de cargar para delimitar RAM explícita.
- [x] Step 2: RED perfil repr reproducible con clase de configuración cuyo
  `__repr__` falla si se invoca para comprobar suffix; dataset real se construye.
  Implementar guard path y deepcopies usando import existente:
  ```python
  if isinstance(item, (str, os.PathLike, zipfile.Path)) and str(item).endswith('.fdf'):
      # bloque NetCharge existente, sin cambios
  self.data_objects = [deepcopy(dataset[i]) for i in range(size)]
  return deepcopy(self.data_objects[index])
  ```
  Expected: GREEN contrato, paths FDF siguen cargando charge/spectral_C.
- [x] Step 3: workers0/2 con spawn y fork si disponibles, persistentes dos
  iteraciones. Transform in-place de tensor/metadata dentro de wrapper consume
  caché sin acumulación; comparar batches con literal original+una transformación.
  Snapshot pickle sin readers ZIP. Fixture ZIP/FDF real existente más cache,
  archivo caller sigue legible tras materializar/close del dataset y después
  caché sigue legible con archive caller cerrado. Expected: exit0 sin procesos vivos.
- [x] Step 4: RED store-memory actual por stage/Subset; transferir sample a meta
  y mutar labels no cambia siguiente acceso, con setters CPU-target existentes.
  Ejecutar lifecycle, ZIP, split/stages, collate/train irreps nuevos. Expected:
  pérdida/targets/defaults iguales, no cache-specific constructor flag.
  Borde encontrado: deepcopy Atomic usaba pickle y reconstruía referencias
  on-site7→0. RED core/real cache-reconstrucción; método __deepcopy__ copia dict
  runtime con memo, restauraR como pickle existente, no modifica estado portable.
  Verificar radial/no radial y no deserializar file_contents durante copia.
- [x] Step 5: benchmark CPU pareado antes/después con fixture común y warmup;
  alternar legacy repr/cache-compartida y candidato usando subclase local,
  no checkout ni mutación de source. Medir inicialización, access+collate,
  tensor_bytes y RSS en procesos aislados; registrar coste de copias como regresión
  frente al acceso cache legacy. Métrica primaria lazy access, no regresión
  de elementos exactos, tolerancia temporal 10% para esa ruta. Expected: datos
  reales medidos; sintético no prueba paso GPU/VRAM ni física/precisión.
- [x] Step 6: floor3.9/Torch2.4.1, full source CPU final, Ruff/mypy/compile39/diff;
  docs snapshot/invalidation/RAM, commit explícito y task-done con nuevos tests
  cache+dataset_resources+datamodule_stages. Expected: exit0 leído, 6.1 evaluación
  GPU queda recurso ausente si sigue ocupada; siguiente bloque 6.2 perfilado.
