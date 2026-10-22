# Auditoría de modelos MACE y módulos e3nn — 2026-10-21

## Alcance y resultado

La revisión corresponde exclusivamente a `a66b46cb0955ca344693d50737c5e9952fc257fc` y a los 267 commits posteriores a `origin/g2m-v2` (`a5e81345`). Se revisaron `models/mace.py`, `models/long_range.py`, los módulos conectados de `bindings/e3nn`, su integración con Lightning y sus contratos/pruebas. El cierre de las reproducciones se realizó el 22 de octubre sobre la copia original archivada; los cambios posteriores del checkout principal quedan fuera de este informe.

Se confirman **tres defectos introducidos dentro de los 267 commits**: dos errores físicos en la opción de largo alcance y una incompatibilidad entre el checkpointing de activaciones y el entrenamiento con MACE congelado. No se atribuyen estos defectos a las correcciones posteriores de la rama principal ni se evalúan aquí esas correcciones.

Las comprobaciones usaron CPU, geometrías sintéticas y fixtures del mismo HEAD. No se modificaron fuentes, pruebas ni dependencias; no se lanzaron GPU, entrenamientos, campañas ni colas. Entorno de reproducción: Python 3.12.3, PyTorch `2.11.0+cu130`, e3nn `0.4.4`, CUDA oculta y dos hilos por biblioteca. `environment.log` verifica que `graph2mat` se importa desde la copia archivada, no desde el checkout editable actual.

## E1 — P1: la suma electrostática depende de la base elegida para la misma red

**Localización:** `src/graph2mat/models/long_range.py:29`; consumidores en `:61` y `:82`.

**Escenario real:** una celda periódica no ortogonal o no reducida. `_kvectors` promete todos los vectores recíprocos con `0 < |k| <= kmax`, pero limita cada índice entero con `ceil(kmax / |b_i|)`. Esa cota solo es suficiente cuando los vectores recíprocos son ortogonales: en una base sesgada, componentes grandes de signo opuesto pueden cancelarse y producir un vector corto. La función omite esos términos sin aviso.

**Oráculo:** las celdas por filas `A = diag(10,10,10) Å` y `A' = M A`, con

```text
M = [[10,1,0],
     [ 9,1,0],
     [ 0,0,1]],  det(M) = 1,
```

describen exactamente la misma red. Se mantienen las mismas posiciones cartesianas `[[0,0,0],[2,1,0.5]] Å`, cargas `[1,-1]`, smearing `2 Å` y tolerancia `1e-8`. Un cambio entero unimodular de base no cambia el potencial ni la matriz de interacción.

```text
                                  base cúbica          misma red, base sesgada
vectores devueltos                 1356                 38
potencial en átomo 0 (eV/e)        -5.782316365157627    -7.850100614579900
potencial en átomo 1 (eV/e)         5.782316365157627     7.850100614579900
error máximo de potencial                               2.067784249422262
error máximo de ewald_matrix (eV/e²)                     2.462968901702652
```

Una enumeración independiente usa la identidad `k·a_i = 2π n_i` y la cota completa `|n_i| <= kmax |a_i| / (2π)`. Recupera los mismos **1356 vectores** y la misma matriz en ambas bases, con `rtol=atol=1e-12`. El control cúbico coincide con la implementación. Un cambio de base más moderado `[[1,1,0],[0,1,0],[0,0,1]]` ya reduce el recuento a 1204 y produce un error de potencial de `5.4552e-7 eV/e`; el ejemplo fuerte demuestra que el defecto no queda acotado por la tolerancia solicitada.

**Consecuencia:** `long_range="charges"` puede incorporar un campo espurio por la representación de la celda. `ewald_matrix`, que alimenta `long_range="qeq"`, comparte el mismo defecto; no se ha cuantificado aquí el error resultante de las cargas Qeq ni el de una campaña entrenada.

**Origen:** `6834407fe1ab7387a54e00d7f8d7c59b81bde559`, que introduce `_kvectors` y su cota actual. Está dentro de los 267 commits. `88b3beec` añade Qeq y reutiliza el kernel; no origina el defecto.

**Propuesta mínima:** enumerar con una cota válida para celdas generales y filtrar por la norma de `k`, o reducir la base con una transformación exacta antes de enumerar. Añadir una regresión de invariancia ante el cambio unimodular anterior. No basta con ampliar arbitrariamente el rango por un factor fijo.

**Reproducción:** `e3nn/repro_long_range.py`, `finish_e3nn/long_range.log`, `finish_e3nn/reciprocal_oracle.py` y `finish_e3nn/reciprocal_oracle.log` dentro del directorio temporal indicado abajo.

## E2 — P1: el potencial escalar de largo alcance se suma solo a H↑

**Localización:** `src/graph2mat/models/mace.py:788`; entrada pública en `src/graph2mat/tools/lightning/models/mace.py:310` y `:334`.

**Escenario real:** ruta legacy colineal con etiquetas `[H↑, H↓, S]`, `n_matrix_components=3`, `n_output_components=2`, `long_range=True` y pérdida gauge. `LitMACEMatrixModel` acepta esta configuración. La función `add` modifica exclusivamente la primera columna cuando la salida tiene varias componentes.

**Esperado:** para la corrección electrostática escalar que declara el módulo,

```text
ΔHσ,ij = (V_i + V_j)/2 * S_ij,
```

ambos canales H reciben el mismo término. El potencial no define un campo de intercambio dependiente del spin. La columna S se usa como referencia geométrica y no debe recibir esa corrección.

**Reproducción pública:** se construye el modelo Lightning anterior sobre el fixture de dos especies, se fija `H↑=H↓=0` y `S=I`, y se comparan los mismos pesos con el corrector activado y desactivado. S es positiva definida y las dos ejecuciones conservan geometría, base, orden orbital y soporte. Las cargas se hacen no nulas con una inicialización determinista de su cabeza, sin entrenamiento.

```text
up_correction_max       = 2.2411062717437744 eV
down_correction_max     = 0.0 eV
up_error_vs_VS          = 1.1920928955078125e-07 eV
down_error_vs_VS        = 2.2411062717437744 eV
```

La reproducción directa de `MatrixMACE` también lo confirma (`15.3984 eV` de corrección máxima en el primer canal y cero en el segundo). Esas magnitudes corresponden a pesos sintéticos; no describen un error medido en un checkpoint entrenado.

**Consecuencia:** la opción introduce una separación artificial entre spins. Incluso un potencial constante común, que debería añadir `c S` a ambos canales, se convierte en un desplazamiento relativo físico entre ellos. El defecto es distinto del gauge independiente por spin documentado en `metrics.md`; la comparación aquí usa directamente las matrices de salida y no depende de esa pérdida.

**Alcance preciso:** las rutas nuevas `matrix_targets`, `spin_orbit` y `magnetic_moments` rechazan expresamente `long_range` y no son el caso denunciado. El caso reproducido es la ruta colineal legacy que permanece aceptada. Los tests actuales de largo alcance solo comprueban una componente de salida y no cubren este contrato.

**Origen:** `6834407fe1ab7387a54e00d7f8d7c59b81bde559` introduce la suma exclusiva a la columna cero sobre una API que ya admitía múltiples componentes. Defecto dentro de los 267 commits, no heredado de `a5e81345`.

**Propuesta mínima:** aplicar el mismo término a todos los canales H declarados y conservar cualquier canal auxiliar. Si la opción se limita deliberadamente a un Hamiltoniano escalar, esa restricción debe validarse antes del forward; aceptar dos canales H y corregir uno produce una matriz físicamente diferente de la declarada.

**Reproducción:** `e3nn/repro_lightning_options.py` y `finish_e3nn/lightning_options.log`; control directo en `e3nn/repro_long_range.py`.

## E3 — P2: el checkpointing desconecta el readout cuando MACE está congelado

**Localización:** `src/graph2mat/models/mace.py:572`; política de etapas aceptada en `src/graph2mat/tools/lightning/model.py:1346`.

**Escenario real:** `gradient_checkpointing=True` junto con una etapa `freeze=["mace"]`, `train=["readout"]`. Es la configuración prevista para entrenar el readout conservando las características de un backbone. La API acepta ambas opciones; el checkpointing implica `lean_mace`.

**Causa:** `_checkpointed_readout` usa `checkpoint(..., use_reentrant=True)` con `node_feats` como único argumento tensor. Al congelar MACE y no pedir derivadas geométricas, esas características no requieren gradiente. El checkpoint reentrante ejecuta el readout sin construir el grafo de sus parámetros y devuelve etiquetas desconectadas, aunque los pesos del readout siguen siendo entrenables.

**Reproducción:** se aplica una etapa real de `LitMACEMatrixModel`, se verifica que todos los parámetros MACE están congelados y se calcula la misma pérdida sobre las mismas etiquetas y pesos:

```text
checkpoint=False: loss=0.0007639014511369169, requires_grad=True
                  14 tensores del readout con gradiente no nulo
checkpoint=True:  loss=0.0007639014511369169, requires_grad=False
                  backward -> RuntimeError:
                  element 0 of tensors does not require grad and does not have a grad_fn
```

El control directo de `MatrixMACE` reproduce el mismo corte y obtiene 11 tensores con gradiente no nulo sin checkpointing. El fallo no es una diferencia tolerable de precisión: el entrenamiento de esa etapa no puede efectuar el backward.

**Origen:** `bd1d64d3b5128d71591e523bc306350c8d2f2e4f` introduce el checkpoint reentrante del readout sin comprobar si su entrada requiere gradiente. La política de congelación por etapas ya existe en `a5e81345`. Defecto introducido dentro de los 267 commits; los cambios posteriores para coefficient-only conservan esa llamada.

**Propuesta mínima:** ejecutar el readout normalmente cuando sus entradas no requieran gradiente, para que autograd siga sus parámetros entrenables. La solución debe conservar el checkpointing válido cuando MACE sí se entrena. Añadir una regresión que combine las opciones públicas anteriores y compruebe gradientes del readout, no solo valores de salida. Sustituir globalmente el modo reentrante exige validar aparte la limitación TorchScript documentada; no es necesario para este caso.

**Reproducción:** `e3nn/repro_freeze_checkpoint.py`, `finish_e3nn/freeze_checkpoint.log`, y la segunda mitad de `e3nn/repro_lightning_options.py` con su log nuevo.

## Controles y zonas sin otro hallazgo confirmado

- Se reejecutaron los tests existentes de largo alcance, Qeq, etapas y equivalencia de `lean_mace`/checkpointing con backbone entrenable: **13 passed, 27 deselected, 678 warnings, 6.30 s**, salida 0. La selección incluye Madelung rocksalt, rotación y batch, fórmula `VS`, neutralidad, carga total Qeq, estacionariedad Qeq y gradientes de MACE/readout sin congelación. Son controles positivos; no cubren los tres casos nuevos anteriores.
- La revisión conectada de los bloques e3nn abarcó transformación de coeficientes a matrices, componentes y simetría por transposición, operaciones de nodos/aristas, reutilización de características, readouts por interacción, simplificación de irreps, actualizaciones de aristas y envolvente radial. No quedó otro defecto con caso concreto confirmado en esta área. Los resultados independientes de ordenación/agrupación de etiquetas corresponden al informe `data.md`; no se cuentan otra vez aquí.
- Los readouts de bajo rango y Slater–Koster se contrastaron con sus restricciones de base, simetría y coeficientes. Las opciones incompatibles que ya se rechazan explícitamente no se clasifican como fallos. Tampoco se exige una ganancia científica por el mero hecho de ofrecer esas arquitecturas.
- En contexto Hamiltoniano y rutas magnéticas/SOC se revisaron los contratos de entrada, forma de componentes y opciones excluidas. La inversión temporal, la hermiticidad y la interpretación física tienen tests dedicados en el árbol original. Su existencia y lectura no certifican todas las combinaciones ni sustituyen una validación de observables; no se ejecutó aquí una campaña magnética adicional.
- Se contrastó el camino de compilación que preserva claves de `state_dict` y el límite documentado de cuEquivariance. No se midieron CUDA, compilación con Inductor, kernels cuEq, velocidad ni VRAM. No hay una conclusión nueva de rendimiento.

## Límites investigados que no se cuentan como defectos nuevos

**`readout_fp32=True` con un modelo convertido a float64.** El control float64 sin esa opción funciona; al activarla, el cast explícito de características a float32 encuentra pesos float64 y falla en TorchScript. Esa implementación y su documentación ya están en `a5e81345`; la opción se describe para ejecutar el readout en fp32 bajo precisión mixta. No se ha encontrado una promesa de compatibilidad con pesos float64, ni se ha probado una receta certificada que necesite esta combinación. Se registra como límite de combinación, **no como cuarto hallazgo ni regresión de los 267 commits**. Evidencia: `e3nn/repro_dtypes.py` y `finish_e3nn/dtypes.log`.

**Autocast CPU BF16.** El modelo base falla con `both inputs should have same dtype` antes del readout, con y sin `readout_fp32` y con o sin largo alcance. La llamada aislada al corrector de largo alcance sí funciona bajo ese autocast. `docs/data_loading.rst` ya declara el fallo basal CPU BF16 y que la elegibilidad de coefficient training no certifica AMP de extremo a extremo. Por tanto, estos fallos no se atribuyen al corrector ni se presentan como una nueva regresión e3nn. CUDA AMP no se comprobó.

Las comparaciones físicas de E1 y E2 son diagnósticos algebraicos reproducibles: usan las mismas geometrías, bases, orden orbital, unidades y soporte dentro de cada comparación. En E2 se usa S de referencia idéntica, deliberadamente fijada a I para aislar la semántica de spin; no se afirma precisión conjunta H/S ni se resuelve un problema espectral con S predicha. No se extrapolan los errores sintéticos a bandas, DOS, OOD, semillas o resultados de campañas previas. No se declara ausencia global de defectos.

## Artefactos y comandos

Raíz temporal: `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d`. Los scripts anteriores se preservan en `e3nn/`; el cierre genera exclusivamente logs, un oráculo independiente y procedencia en `finish_e3nn/`. `source/` conserva el archivo íntegro del HEAD original. El wrapper `audit-python` desactiva el finder editable mediante el bootstrap y usa las extensiones recompiladas para ese árbol.

Desde esa raíz:

```bash
./audit-python e3nn/repro_long_range.py
./audit-python e3nn/repro_freeze_checkpoint.py
./audit-python e3nn/repro_lightning_options.py
./audit-python e3nn/repro_dtypes.py
./audit-python finish_e3nn/reciprocal_oracle.py
```

Desde `source/`, con el wrapper anterior como intérprete:

```bash
../audit-python -m pytest -q \
  src/graph2mat/tools/lightning/tests/test_long_range.py \
  src/graph2mat/tools/lightning/tests/test_training_stages.py \
  src/graph2mat/bindings/e3nn/modules/tests/test_edge_pairing_and_safety.py \
  -k 'long_range or ewald or equilibrate or qeq or training_stage or lean_mace_matches'
```

**Dictamen:** corregir E1 y E2 antes de usar largo alcance para las configuraciones afectadas, y E3 antes de combinar checkpointing con una etapa de readout sobre MACE congelado. La auditoría del área queda cerrada para el HEAD original; este informe no evalúa los arreglos posteriores.
