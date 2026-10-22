# Revisión de arreglos Claude: datos, modelos y e3nn — 2026-10-22

## Alcance y resultado

Se revisó la copia congelada de `17e246f` con los ocho ficheros modificados que ya estaban presentes al crearla, contrastando los cambios desde `a66b46c` con sus consumidores y pruebas. La copia compartida es `/tmp/graph2mat-codex-review-20261022-f80m50ec/source`; su intérprete CPU es `../review-python`. Los cambios propios de producto y pruebas se hicieron exclusivamente en esa copia. El checkout original solo recibe este informe. No se instalaron dependencias ni se usaron GPU, SCF, colas o campañas.

Los arreglos de Claude resuelven DATA-01, DATA-02, DATA-04 y E1–E3 en las reproducciones originales. Quedaban **cuatro problemas confirmados**: DATA-03, DATA-05 y DATA-06 seguían presentes; la comprobación de acuerdo entre modelo y datos de `2caeb69` también era incompleta. Los cuatro se corrigieron en la copia con pruebas que fallaron antes del cambio. No se modificó la limitación `readout_fp32=True` con pesos float64.

La revisión del commit posterior `d3d7fe5` se limitó a lectura por hash, porque Claude continuó trabajando en el checkout original durante esta fase. Ese cambio no se incorporó a ciegas a la copia congelada. Sus hunks y los propios son independientes; deben preservarse ambos al integrar.

## Problemas pendientes corregidos

### R1 — P1 — DATA-03: igual dimensión no implica igual base angular

**Antes:** una configuración con dos especies C, `3x0e` y `1x1o`, seguía aceptándose como si ambas fueran `3x0e`. La guardia de `from_config` comprobaba el número total de funciones. También aceptaba una `BasisConfiguration` cuyas capas estaban en otro orden que las de la tabla, aunque la dimensión coincidiera.

**Cambio:** `core/data/processing.py:2004` contrasta la secuencia ordenada de capas `(l,paridad)` de las bases presentes con la entrada correspondiente de la tabla. Reutiliza `_shell_sequence`; no confunde `2x0e` con dos capas consecutivas `0e+0e`. La comprobación se aplica cuando hay una matriz, conserva el rechazo previo por dimensión y no convierte esta guardia en soporte completo de especies distintas con el mismo Z.

**Después:** ambos casos incompatibles lanzan `ValueError` antes de extraer etiquetas. Los casos de base coincidente y las conversiones vecinas siguen pasando. No se certifica igualdad radial, de pseudopotencial o de ajustes SIESTA: el rechazo añadido cubre la incompatibilidad angular demostrada y el orden de capas.

### R2 — P2 — DATA-05: la conversión escalar pierde o rechaza entradas complejas

**Antes:** el `.so` actualizado seguía fallando con `TypeError: float() argument must be a string or a real number, not 'complex'`. Al ejecutar el mismo `_sparse.py` como Python, en lugar de fallar descartaba la parte imaginaria con `ComplexWarning`.

**Cambio:** `core/data/_sparse.py:27` calcula una vez `np.result_type(np.asarray(data).dtype, np.float64)` y lo usa al crear los bloques. Es la misma promoción que ya usa la ruta multicomponente. Se recompiló únicamente esta extensión en la copia compartida, con Cython y GCC ya disponibles; no se sustituyó ninguna extensión del checkout original.

**Después:** `complex64` y `complex128` preservan exactamente las entradas sintéticas, tanto en Python como en Cython. La conversión pública de una H Hermítica `[[1,0.25+0.5j],[0.25-0.5j,2]]` recupera la misma matriz. Los fills reales y los casos multicomponente vecinos siguen pasando. Este arreglo preserva datos numéricos; no promete entrenar modelos con targets complejos ni amplía el contrato SOC.

### R3 — P2 — DATA-06: `load_labels=False` no retiraba los targets de objetos ya preparados

**Antes:** una `OrbitalConfiguration` con matriz devolvía H/S aunque se solicitara `load_labels=False`. Una `TorchBasisMatrixData` preparada se devolvía por identidad, con sus targets y como referencia mutable al objeto fuente.

**Cambio:** `core/data/processing.py:1954` copia la muestra preparada antes de retirar `point_labels`, `edge_labels` y sus filtros. Para configuraciones, crea un contenedor separado con `matrix=None` antes de construir el grafo. Se conservan campos auxiliares y metadata ajenos a esos targets; no se modifica el original ni se añaden etiquetas a las features.

**Después:** las salidas NumPy y PyG carecen de targets; la configuración fuente mantiene su matriz y la muestra preparada conserva sus etiquetas originales, `total_charge` y metadata. La prueba también verifica que `load_labels=True` sigue preservando las columnas H/S, incluida S no diagonal. Se mantienen las rutas existentes que cargan S explícitamente para largo alcance: no se inventa soporte de `long_range` sin su entrada S.

Las muestras preparadas se copian profundamente para conservar sus campos auxiliares sin alias mutable. No se ha medido el coste de esa copia con muestras grandes. En PyG, un campo ausente y un campo retirado con `None` no se exponen necesariamente como atributo; las pruebas usan la ausencia efectiva del target, no imponen una representación nueva al contenedor.

### R4 — P1 — `2caeb69`: tipos iguales podían ocultar representaciones modelo/datos distintas

**Antes:** `_check_data_agreement` rechazaba tipos o especies reordenados, pero aceptaba tablas con el mismo tipo y dimensión que representaban `3s` frente a `p`, otro orden de capas o convenciones cartesianas/esféricas distintas. El modelo podía asociar sus pesos a otra representación.

**Cambio:** `tools/lightning/models/mace.py:376` conserva la comparación de tipos de Claude y añade secuencias angulares ordenadas y la transformación de convención. La comprobación acepta multiplicidades escritas de forma equivalente y aliases de convención con la misma transformación.

**Después:** los tres casos incompatibles se rechazan; las tablas explícitas equivalentes se aceptan. Los tests reales de checkpoint legacy/actual, restauración y predicción CPU siguen pasando. No se exige identidad radial completa que `PointBasis` no representa. Esta guardia no certifica por sí sola toda la procedencia física de una base.

## Dictamen por arreglo de Claude

| Commit o cambio congelado | Dictamen en los ficheros de esta revisión | Evidencia y límite |
|---|---|---|
| `ba9b440` — referencia por componentes | Correcto para el contrato existente de una referencia compartida por canal. | La resta usa `reference[...,None]`, inversa de la suma existente. Reproducción original DM spin2 con 1, 2 y 4 orbitales vuelve a la matriz original. No redefine físicamente la referencia de spin. |
| `5a4d900` — potencial común y política H/S | Correcto en las rutas verificadas. | Ambos H reciben `VS`; S predicha queda intacta. Prueba adicional con `sisl.Hamiltonian(spin="polarized", orthogonal=False)` y procesamiento real `h_and_overlap`; guardia Lightning rechaza las políticas sin S. |
| `a492fbf` — signo de QEq | Coherente con la convención electrónica ahora declarada. | `NetCharge>0` significa electrones retirados; las cargas electrónicas suman `-NetCharge`. Test existente obtiene `[-1,2]` para entradas `[1,-2]`. El solver genérico conserva su restricción explícita. |
| `4b5194b` — envolvente del contexto | Correcto en la ruta revisada. | La misma envolvente multiplica features radiales y canales escalares de contexto antes de concatenarlos; prueba real suprime el bloque cerca de su radio por pareja. No se añade una certificación de convergencia con el radio. |
| `624101e` — checkpointing con partes congeladas | Correcto; no se sustituyó globalmente por checkpointing no reentrante. | El helper omite el checkpoint reentrante si ningún argumento requiere gradiente. Comparaciones de valores y gradientes pasan con MACE completo congelado y con embedding congelado. La etapa pública produce los mismos 14 tensores con gradiente no nulo con checkpoint off/on. |
| `4dd8f9c` — orden de capas y tabla atómica `max` | Correcto para los casos anunciados; no cubría DATA-03. | Rechaza agrupaciones con capas en otro orden, conserva agrupaciones válidas y construye la tabla máxima sintética como tabla genérica. El rechazo de configuración/tabla se añade en R1. |
| `67aa8e0` — coeficientes en aristas inversas | Correcto en las combinaciones verificadas. | El proyector crea specs para tipos inversos homonucleares no simétricos, salvo el tipo cero que coincide consigo mismo. Pasan casos de 2/3 especies, tres agrupaciones, matrices simétricas/no simétricas y equivalencia con MSE de matrices. |
| `fb236ca` — FDF NetCharge | Corrige el caso local anunciado. | Se usa el lector FDF sisl: nombres equivalentes, `%include` y ausencia. La reproducción original devuelve `2.0`. ZIP mantiene el límite declarado de buscar solo en el propio miembro; no se certifican includes dentro del ZIP. |
| `f4fe6f7` — `nsc` distinta | Correcto como rechazo explícito. | El caso original `[3,1,1]→[5,1,1]` falla antes de devolver etiquetas erróneas; omitir `nsc` conserva `[7,3]`. No implementa un remapeo general de imágenes. |
| `55dbbde` — modo compile | Correcto como guardia del modo que ya se había descartado por segfault. | `reduce-overhead` se rechaza antes de compilar. No se ha medido ni certificado Inductor/CUDA en esta revisión CPU. |
| `f3b4d29` — pruebas de opciones | Pruebas útiles, sin otro fallo confirmado en las seleccionadas. | Cubren opciones rechazadas, equivariancia de bloques y gestión de recursos ZIP. El test cuEq se omite por dependencia ausente; no equivale a una certificación cuEq CPU/GPU. |
| `2caeb69`, parte de modelo | Incompleto inicialmente; completado por R4. | La ordenación de globs pertenece a la otra revisión. Aquí se comprueban tipo, capas, orden y transformación de convención modelo/datos. |
| Dirty — cota recíproca Ewald | Matemáticamente correcta; resuelve E1. | Cota `|n_i|≤kmax|a_i|/(2π)`. Oráculo independiente recupera 1356 vectores en las tres bases equivalentes y potenciales iguales hasta redondeo. |
| Dirty — guardias MACE y mensajes e3nn | Correctas para las combinaciones comprobadas. | Rechazan `mp_cutoff` sin embedding radial del readout, simplificación de irreps con readouts por interacción y opciones de mensajes con un preprocesador que no las consume. Tests de aceptación/rechazo pasan. No se cambió arquitectura ni default. |
| Posterior `d3d7fe5` | Compatible con los cambios propios; preservar su código y test al integrar. | Elimina la elección arbitraria subset-sum de representantes simétricos. Afecta `_get_symmetric_unique_edge_mask`, mientras R1/R3 afectan `new`/`from_config`; no hay dependencia entre esos hunks. Leído por hash, no ejecutado en la copia congelada. |

## Evidencia CPU y límites

Los artefactos están en `/tmp/graph2mat-codex-review-20261022-f80m50ec/data_models/artefactos/`.

- `data_red.log`: **10 fallos** antes de los arreglos R1–R3, con las causas esperadas. Incluye pérdida imaginaria Python y excepción Cython. Durante la verificación se corrigieron dos supuestos del propio fixture PyG: acceso a campos ausentes y lectura por clave de `node_attrs`; no se añadió comportamiento productivo para satisfacer esos supuestos.
- `model_agreement_red.log`: **3 fallos, 1 control pasa** antes de R4.
- `regressions_final.log`: **15 pasan**, 110 warnings, 1.18 s, salida 0. Incluye los 14 casos de regresión/control de R1–R4 y el control adicional de largo alcance con H/S colineal real.
- `data_neighbors_final2.log`: **239 pasan**, 66 warnings, 1.83 s, salida 0. Cubre procesamiento, bases incompatibles, sparse/fill/ncol, agrupación atómica, recursos de datasets y referencias físicas.
- `model_agreement_green.log`: **28 pasan**, 271 warnings, 2.96 s, salida 0. Incluye tipos/especies, tablas explícitas y round trips reales de checkpoints CPU.
- `claude_models_controls.log`: **104 pasan, 2 omitidos**, 2810 warnings, 25.78 s, salida 0. Las omisiones corresponden a Inductor que exige GPU y cuEq no instalado. Incluye largo alcance/QEq, gradientes con congelación, envolvente de contexto, equivariancia, guardias y orientación de coeficientes.
- `complete_reciprocal_oracle.log`: las bases cúbica, moderadamente sesgada y fuertemente sesgada contienen **1356 vectores** y dan la misma matriz electrostática. En la base fuerte el potencial vuelve a `[-5.782316365157636, 5.782316365157636] eV/e`, eliminando el desacuerdo original de `2.067784 eV/e`.
- `long_range_oracle.log`: potencial, gradientes respecto a cargas, cargas QEq y gradientes respecto a electronegatividad/dureza son invariantes bajo el cambio de base, con `rtol=atol=1e-11`. La enumeración observada pasa de 3375 candidatos y 0.73 ms en la base cúbica a 260625 candidatos y 5.49 ms en la base fuerte. Son mediciones CPU aisladas, no un benchmark de entrenamiento; la cota correcta puede encarecer bases muy sesgadas y no se implementa aquí reducción de red.
- `lightning_options_final.log`: ambos spins reciben `2.2411063 eV` de corrección máxima sintética, con error `1.1921e-7 eV` frente a `VS`; la etapa congelada mantiene `requires_grad=True` y 14 tensores con gradiente no nulo. La magnitud proviene de pesos sintéticos, no de un checkpoint entrenado.

No se suman estos gates como si fueran tests disjuntos: existe solapamiento. Los warnings son de dependencias, TorchScript y multiprocessing; no se afirma una ejecución sin warnings. Ruff no está instalado en el intérprete de esta copia; no se instaló para esta tarea. La comprobación de whitespace y parseo se registra aparte. La suite completa y el gate de la fuente integrada corresponden al coordinador, porque otros agentes están modificando sus áreas en la misma copia.

Los controles de matrices son diagnósticos algebraicos con geometría, base, orden, soporte y unidades fijos. La H compleja de conversión es Hermítica por construcción; en largo alcance se usa la misma S de referencia positiva y no diagonal. No se ha evaluado S predicha, bandas, DOS, extrapolación, semillas, precisión de una campaña, VRAM ni ganancias GPU. Ningún resultado convierte una prueba de software en una certificación científica global.

## Entrega para integración

Solo hay cambios propios de producto en `core/data/{processing,_sparse}.py` y `tools/lightning/models/mace.py`; las pruebas nuevas están en `core/data/tests/test_review_data_models_regressions.py` y `tools/lightning/tests/test_review_model_regressions.py`. Los demás arreglos de Claude se conservan. `own.patch` y `verification.json` registran los cinco ficheros para el merge; el `.so` local es únicamente un artefacto de validación y no debe integrarse como binario.

**Dictamen:** arreglos revisados aceptables para los contratos y controles indicados, con R1–R4 incorporados. No queda otro fallo confirmado pendiente en esta revisión de área. Falta el gate completo de la fuente final tras integrar los trabajos concurrentes; no se certifican los cambios posteriores no incluidos en la copia salvo la compatibilidad por lectura de `d3d7fe5`.
