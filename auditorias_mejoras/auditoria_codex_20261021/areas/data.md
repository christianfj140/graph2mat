# Auditoría de procesamiento, datos y núcleo — 2026-10-21

## Alcance y estado

Revisión de solo lectura de `g2m-v2`, HEAD `a66b46cb0955ca344693d50737c5e9952fc257fc`. Se revisó el cambio respecto a `origin/g2m-v2` (`a5e81345`) y el código conectado. No se editaron fuentes, pruebas, configuraciones productivas ni datos externos. No se ejecutaron trabajos GPU.

La rama añade optimizaciones de agrupación, objetivos en coeficientes, referencias físicas y multitarea, entradas magnéticas y ampliación del radio del grafo. Esta revisión verifica especialmente que esa preparación preserve el significado de las matrices y las entradas que consume el modelo.

Se confirman seis defectos: cinco ya existían en la base `origin/g2m-v2`; uno fue introducido en los 267 commits. La corrección reciente del resorting se encuentra presente y pasa el oráculo independiente descrito abajo. Ninguno de los seis defectos se atribuye a esa corrección.

Entorno utilizado exclusivamente para las reproducciones:

```text
/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/audit-python
```

El wrapper importa la copia aislada de HEAD bajo `source/src`, con ambas extensiones Cython recompiladas desde el código actual. `probe.log` registra las rutas de ambos `.so`; no se usaron las extensiones obsoletas del entorno editable. CUDA deshabilitada, OMP/BLAS limitados a dos hilos y escritura de bytecode deshabilitada. Todos los ejemplos son sintéticos pequeños; no hay cálculos SCF, entrenamiento ni datos de terceros.

## Hallazgos confirmados

### DATA-01 — P1 — La resta del baseline mezcla ejes y puede alterar la matriz polarizada reconstruida

- **Ubicación:** `src/graph2mat/core/data/matrices/basis_matrix.py:105`; operación inversa en `src/graph2mat/core/data/processing.py:1397`.
- **Escenario:** `sisl.DensityMatrix(spin="polarized")`, `n_matrix_components=2`, tabla atómica con ocupaciones iniciales no nulas y `sub_point_matrix=True`, que es el valor predeterminado del procesador. Dos orbitales s por átomo bastan. Es una ruta ordinaria de `BasisMatrixData.new` y `TorchBasisMatrixDataset`; no requiere las referencias físicas opcionales.
- **Esperado:** restar una referencia definida para cada componente y volver a sumarla sobre los mismos ejes, o rechazar explícitamente esa combinación si el baseline polarizado no está soportado. Convertir las etiquetas de vuelta debe recuperar la DM original.
- **Real:** el bloque tiene forma `(n,n,2)` y la referencia `(n,n)`. La resta se alinea por los dos últimos ejes, mientras la suma inversa usa `point_matrix_flat[:,None]`. Para `n=2`, ambas operaciones se ejecutan pero no son inversas. Con baseline diagonal `(2,1)`, las diagonales originales son spin 0 `[0.4,1.4]` y spin 1 `[0.6,1.6]`; al reconstruir se obtienen `[0.4,2.4]` y `[2.6,1.6]`. Para `n=4`, la resta aborta con `ValueError: operands could not be broadcast together with shapes (4,4,2) (4,4)`.
- **Evidencia reproducible:** `data/probe_baseline.py` y `data/probe_baseline.log` dentro del directorio temporal. Comando: `audit-python data/probe_baseline.py`. Incluye tamaños 1, 2 y 4; el tamaño 1 vuelve correctamente y sirve como control.
- **Origen:** heredado de `origin/g2m-v2`. El soporte de componentes y la suma inversa aparecen en `1d52abb0` (`feat: add collinear-spin component support for Hamiltonian pipeline`), que es ancestro de la base. El cambio de `6cf523f7` a `_get_block` conserva la resta anterior; no introduce este problema.
- **Consecuencia:** una conversión de ida y vuelta con etiquetas exactas altera poblaciones espín-resueltas y carga total. Otras bases usuales abortan antes de entrenar. La ruta física opcional ya rechaza baseline colineal, pero esa guardia no protege el procesamiento ordinario.
- **Propuesta mínima:** extender esa prohibición al procesador general si no existe una definición validada del baseline por espín. Si se admite expresamente la misma referencia para cada canal, restar `reference[...,None]` y mantener idéntica convención en la reconstrucción; la definición física del baseline debe quedar explícita.

### DATA-02 — P1 — Cambiar `nsc` en `from_config` consulta bloques de otra imagen

- **Ubicación:** `src/graph2mat/core/data/processing.py:2047` y `:2050`.
- **Escenario:** configuración con matriz cuyo `nsc=[3,1,1]`; llamada pública `BasisMatrixData.from_config(config, processor, nsc=[5,1,1])`. Una cadena con un átomo por celda y acoplamientos a las imágenes ±1 basta. Esta opción tiene consumidores reales en `tools/server/tests/test_extrapolation.py:251` y `:418`.
- **Esperado:** ampliar el contenedor de superceldas conserva el bloque asociado a cada desplazamiento físico. Alternativamente, rechazar una sustitución incompatible antes de devolver etiquetas.
- **Real:** se calculan `neigh_isc` en la nueva `Lattice`, pero `block_dict` sigue usando los índices de la `nsc` original. Para exactamente los mismos vecinos y desplazamientos, las etiquetas `[7,3]` pasan a `[0,7]`: un bloque válido se vuelve cero y otro se toma de la imagen opuesta. La guardia de alcance solamente comprueba que las imágenes caben, condición satisfecha en este ejemplo.
- **Evidencia reproducible:** sección `PROBE nsc_override` de `data/probe.py` y `data/probe.log`. Comando: `audit-python data/probe.py`. Usa H real, un orbital por átomo, celda de 2 Å, radio 1.1 Å y acoplamientos distintos en ±1; sin solapamiento ni espectros.
- **Origen:** heredado de `origin/g2m-v2`. La opción `nsc` y la consulta de etiquetas con índices de la nueva celda ya existían; la conversión a `sisl.Lattice` procede de `c586aaa6` y el flujo es anterior a los 267 commits. El remapeo de `sc_off` añadido por `928dee0` corrige otra situación, no esta sustitución posterior de `nsc`. Hay un agravamiento dentro de los 267 commits: `_get_block` (`6cf523f7`) transforma la consulta de una clave ausente en fill cero; en este ejemplo esa consulta habría abortado antes. La reutilización incorrecta de índices de imágenes es anterior.
- **Consecuencia:** objetivos y matrices físicamente incorrectos sin error, aunque el grafo mantenga su geometría y la predicción sea exacta para las etiquetas obtenidas.
- **Propuesta mínima:** usar los desplazamientos enteros para obtener índices de la celda original al extraer etiquetas y conservar los índices nuevos solo para el contenedor de salida. Como medida más pequeña, rechazar `nsc != config.matrix.nsc` cuando haya matriz.

### DATA-03 — P1 — La guardia para bases del mismo Z acepta bases de igual dimensión y distinta simetría

- **Ubicación:** `src/graph2mat/core/data/processing.py:1978`; identificación por Z en `src/graph2mat/core/data/configuration.py:469`.
- **Escenario:** dos especies C con el mismo Z: una con tres orbitales s y otra con una capa p completa. Ambas tienen tres funciones, pero las representaciones son `3x0e` y `1x1o`. La tabla contiene la primera especie.
- **Esperado:** si estas especies no están soportadas, rechazarlas antes de asociar las etiquetas a una única representación. Si se soportan, asignar tipos y bases diferentes.
- **Real:** la guardia compara solo el número total de orbitales y devuelve tipos `[0,0]`. La segunda especie se interpreta como `3x0e`. La reconstrucción contiene una sola especie, con tres orbitales s en ambos átomos. No se lanza error.
- **Evidencia reproducible:** sección `PROBE same_size_basis` de `data/probe.py` y `data/probe.log`. Imprime las dos bases reales, la base interpretada y las bases de la H reconstruida. La H usa solo diagonales reales para que el fallo no dependa de soporte complejo o de fases periódicas.
- **Origen:** el colapso por Z es heredado. `de43c1a` introduce en los 267 commits la guardia por dimensión y documenta que estas especies no están soportadas; esa corrección es incompleta para dimensiones iguales. No se atribuye a la guardia una regresión respecto a la base.
- **Consecuencia:** las filas y columnas se asignan a irreps equivocadas. Una capa p rota de forma distinta de tres escalares; la igualdad de tamaño no asegura compatibilidad física ni equivariancia. No se presenta esta comparación deliberadamente incompatible como validación científica de precisión.
- **Propuesta mínima:** comparar también la secuencia de capas/irreps de cada especie real con la tabla, antes de construir etiquetas; mantener el rechazo explícito mientras no se soporte una identidad de especie independiente de Z. La identidad radial completa sigue siendo una limitación adicional y no se certifica aquí.

### DATA-04 — P2 — `NetCharge` en un FDF incluido se sustituye silenciosamente por cero

- **Ubicación:** `src/graph2mat/bindings/torch/data/dataset.py:178` y `:179`.
- **Escenario:** FDF raíz válido con `%include charge.fdf`, donde el fichero incluido declara `NetCharge 2.0`; carga normal del dataset con `load_labels=False`.
- **Esperado:** `data.total_charge == 2`, usando la misma semántica FDF que SIESTA/sisl.
- **Real:** `sisl.get_sile(root).get("NetCharge")` devuelve `2.0`, pero el dataset obtiene `tensor([0.])` porque busca una línea mediante regex solo en el fichero raíz.
- **Evidencia reproducible:** sección `PROBE included_charge` de `data/probe.py` y `data/probe.log`; FDFs completos conservados en `data/include_case/`. Comando: `audit-python data/probe.py`. No se lee ninguna matriz ni se ejecuta SIESTA.
- **Origen:** introducido en los 267 commits por `88b3beec`, al añadir la entrada de carga para `long_range="qeq"`. La modificación posterior `6227edfb` adapta la ruta ZIP pero mantiene el parser parcial.
- **Consecuencia:** QEq recibe un sistema neutro cuando la estructura declara carga +2; `models/mace.py:773` pasa esa entrada al solver, y `models/long_range.py:115` impone la restricción con ella. El error afecta la condición física del modelo, no solo metadata.
- **Propuesta mínima:** leer `NetCharge` mediante el lector FDF ya disponible. Conservar cero solo cuando el parámetro esté ausente según ese lector.

### DATA-05 — P2 — La conversión de matrices complejas de un componente falla en Cython actual

- **Ubicación:** `src/graph2mat/core/data/_sparse.py:91` y `:94`; selección de esta ruta en `src/graph2mat/core/data/sparse.py:140`.
- **Escenario:** `sisl.Hamiltonian(dtype=np.complex128)` de un componente y dos átomos, con H Hermítica `[[1,0.25+0.5j],[0.25-0.5j,2]]`. El conversor genérico acepta este tipo y la función está anotada con `cython.numeric`.
- **Esperado:** preservar el dtype complejo y sus entradas al crear los bloques; si el contrato no admite matrices complejas, rechazarlas con un mensaje explícito antes de convertir.
- **Real:** la función inicializa cada bloque con `np.full(..., fill_value)` real, sin dtype derivado de `data`. Al asignar un escalar complejo la extensión Cython actual aborta con `TypeError: float() argument must be a string or a real number, not 'complex'`. La ruta multicomponente sí usa `np.result_type(data.dtype,np.float64)`.
- **Evidencia reproducible:** sección `PROBE complex_single` y traceback de `data/probe.log`; código en `data/probe.py`. Se registra expresamente el `.so` actual. No se ha probado aquí la manifestación de este caso ejecutando `_sparse.py` como Python puro.
- **Origen:** heredado; la creación sin dtype aparece en `31865ea0` y ya está en `origin/g2m-v2`. `bb01435` corrige el truncamiento por fill entero en Python, pero no añade preservación del dtype complejo escalar.
- **Consecuencia:** la API de entrada falla para una H compleja válida de sisl incluso antes de acceder al modelo. Este hallazgo no solicita implementar SOC ni altera el aplazamiento de esa funcionalidad.
- **Propuesta mínima:** inicializar los bloques con un dtype que preserve el dtype de entrada y pueda representar el fill, como ya hace la ruta multicomponente. Si el soporte es deliberadamente solo real, declarar y validar esa limitación en la frontera pública.

### DATA-06 — P2 — `load_labels=False` no se respeta para configuraciones y muestras ya preparadas

- **Ubicación:** `src/graph2mat/core/data/processing.py:1946` y `:1948`; consumidor en `src/graph2mat/bindings/torch/data/dataset.py:171`.
- **Escenario:** `TorchBasisMatrixDataset([configuration], processor, load_labels=False)` con una `OrbitalConfiguration` que contiene H. También ocurre con una `TorchBasisMatrixData` ya preparada.
- **Esperado:** una muestra para inferencia sin etiquetas, conforme al argumento público `load_labels`; el objeto fuente debe conservarse.
- **Real:** para la configuración se calculan y devuelven etiquetas `tensor([42.])`. Para `TorchBasisMatrixData`, `new` devuelve el objeto original (`got is item`), también con etiquetas. El argumento solo llega a las ramas que crean una configuración nueva desde una matriz/ruta/geometría.
- **Evidencia reproducible:** sección `PROBE labels_off` de `data/probe.py` y `data/probe.log`; incluye ambas clases de entrada.
- **Origen:** heredado. Los retornos anticipados proceden de `49a76003` y ya existen en `origin/g2m-v2`; el dataset actual sigue pasando correctamente la bandera.
- **Consecuencia:** consumidores que exigen muestras sin objetivos reciben objetivos y, para muestras preparadas, una referencia mutable al original. No se afirma que los modelos actuales lean esas etiquetas como features: la reproducción demuestra incumplimiento del contrato y aliasing, no fuga de etiquetas durante entrenamiento.
- **Propuesta mínima:** cuando `labels=False`, construir una vista/contenedor sin `matrix` o sin etiquetas sin modificar el original. Evitar el retorno temprano de la muestra original si el argumento requiere cambiar su contenido.

## Cobertura sin nuevos hallazgos confirmados

- Se inspeccionaron `core/data/{processing,sparse,_sparse,configuration,basis,table,formats,neighborhood,node_feats}.py`, `core/data/matrices/basis_matrix.py`, `core/data/matrices/physics/orbital_matrix.py`, `bindings/torch/data/{dataset,coefficient_targets,formats,data}.py`, `bindings/torch/load.py`, `core/modules/{graph2mat,matrixblock,_labels_resort}.py` y `bindings/torch/modules/graph2mat.py`, con consumidores y pruebas focales conectados.
- Oráculo independiente de bloques verdaderos: **60 combinaciones pasan, cero fallos**. Cruza cinco ordenaciones de 3–4 especies H/C/N/O, simetría sí/no, agrupación `point_type`/`basis_shape`/`max` y 1/2 componentes. Las bases tienen tamaños 1, 4 y 5; C/N comparten forma. Cada operación devuelve el bloque de referencia del nodo/arista que realmente recibe; la salida completa debe coincidir en forma y valores con las etiquetas. Se incluyen imágenes periódicas y cambios de orientación por agrupación. Artefactos: `data/grouping_oracle.py`, `data/common.py`, `data/grouping_oracle.log`; última línea `fails 0`.
- Ese oráculo se adaptó de pistas previas permitidas, eliminando su importación del parche de extensiones antiguas. La ejecución se hizo contra las extensiones actuales del coordinador; no se aplicó ningún parche productivo ni monkeypatch al código auditado.
- Se comprobaron por lectura los arreglos de `ncol`, la preservación de fill en bloques ausentes, signos de transposición SOC en serialización, identidad para tipos con signo, filtros de `max`, layout precalculado, cierres ZIP/pool, copias CPU y campos de checkpoint nuevos. Las pruebas existentes específicas se examinaron; su resultado de ejecución corresponde al coordinador, que gestiona la suite general.
- La preservación del target y la semántica de S, máscaras y canales se contrastaron con tests de orientación de coeficientes, soporte físico, spin colineal, imágenes no estándar y bases de igual Z. No se detectó otro defecto confirmado en `coefficient_targets.py` por inspección; este agente no ejecutó un barrido independiente de todos los objetivos físicos o de coeficientes.

## Sospechas y límites que no son hallazgos confirmados

- La conversión de momentos vectoriales usa la misma matriz que las posiciones y comenta que el cambio de base es una rotación propia. Los nombres personalizados de `basis_convention` también permiten determinante −1. No se ha construido una comparación completa del modelo y sus convenciones de vector axial en ese caso; queda sospecha sin confirmar, no defecto contado.
- `_get_symmetric_unique_edge_mask` puede recurrir a elegir un subconjunto que cuadre con el número de etiquetas. El flujo estándar auditado mantiene las parejas canónicas y pasa el oráculo. No se ha reproducido un consumidor válido que active esa recuperación y asigne bloques incorrectamente.
- La guarda de igual Z tampoco certifica identidad radial, pseudopotencial o ajustes SIESTA. DATA-03 demuestra un desacuerdo angular explícito; no se extrapola a todos los casos de igual dimensión ni se afirma validación de bases físicas completas.
- No se midieron prestaciones con datasets grandes, carreras bajo carga prolongada, GPU ni observables espectrales. Los hallazgos de matrices se prueban en su representación; no se atribuye a sus cifras un error de bandas, energía o electron-fonón no medido. Las matrices H sintéticas sin S representan ejemplos ortogonales; los resultados no certifican un problema generalizado H/S.

## Cómo repetir y criterio final

Desde `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/`:

```bash
./audit-python data/probe.py
./audit-python data/probe_baseline.py
./audit-python data/grouping_oracle.py
```

Los scripts de defectos capturan las excepciones deliberadamente para mostrar todas las reproducciones; que su proceso termine con código cero no significa ausencia de defectos. Los logs conservados muestran los valores y errores concretos. El oráculo positivo imprime el recuento de fallos.

**Veredicto:** corregir DATA-01, DATA-02 y DATA-03 antes de considerar preservadas las matrices en las combinaciones afectadas; DATA-04 debe corregirse antes de usar QEq con FDFs incluidos. DATA-05 y DATA-06 requieren corrección o límites explícitos del contrato. La auditoría no implementa ninguna corrección ni promueve cambios de comportamiento.
