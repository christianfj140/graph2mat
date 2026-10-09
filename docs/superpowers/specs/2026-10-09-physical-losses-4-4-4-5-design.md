# Diseño: objetivos físicos opcionales 4.4–4.5

Fecha: 2026-10-09. Repositorio Graph2Mat, rama `g2m-v2`, mismo checkout.
Estado: alcance conversacional aprobado; **diseño escrito pendiente de revisión**.
Antecedente: `auditorias_mejoras/inventario_objetivos_fisicos.md`, commit `61ed294`.

## 1. Intención, éxito y límites

El usuario quiere adelantar mejoras potenciales de precisión, velocidad o VRAM.
Este bloque añade pérdidas que puedan aportar señal física verificable, no
promete una mejora ML. Se cierra su implementación solo con integración real,
pruebas CPU, costes medidos y protocolo experimental; la evaluación de precisión
queda expresamente pendiente. No se trabaja en 7.4, fonones, autorun, GPU,
entrenamientos largos, datasets originales ni entornos compartidos.

Alcance aprobado: carga/Mulliken para DM y consistencia energética usando
H/S/DM/EDM reales, contracciones periódicas y agrupación atómica. McWeeny no
es un objetivo exacto aplicable a las muestras fraccionarias inventariadas.
No se inventarán matrices ni ocupaciones a partir de una predicción H-only.

Hay dos referencias utilizables inicialmente: grafeno y grafeno–hBN AA. La
bicapa AA solo sirve como diagnóstico mientras su manifiesto siga rechazándola
por falta de `electronic_convergence.json`. Las comprobaciones numéricas
existentes cubren tres muestras; todavía no todos los snapshots.

## 2. Enfoque seleccionado y alternativas

Se amplían `elementwise_mse` y `gauge_invariant_mse` mediante un ayudante común
de contracciones. No se crea un framework de pérdidas, un readout nuevo ni
otro ensamblador denso. Los términos se seleccionan con pesos cero por defecto.
Las referencias auxiliares se preparan en CPU y comparten el orden de etiquetas
y la selección de aristas del procesador existente.

Alternativas descartadas para este bloque:

- Diagonalizar la predicción en muchos k y ajustar su número de electrones:
  más coste, posibles derivadas problemáticas y conteo tautológico si se fija N.
- Plegar las matrices a Γ: contradice el conteo periódico observado en datos.
- Nueva arquitectura o reactivar el término espectral descartado: fuera de
  este alcance y sin nueva evidencia experimental.

## 3. Convenciones y fórmulas

Se soportan matrices **reales**, no polarizadas o colineales. Las componentes
spin de DM ya incorporan sus ocupaciones: una componente no polarizada cuenta
ambos spins; dos componentes colineales se suman, sin factor adicional de dos.
SOC, no colinealidad y coeficientes complejos se rechazan explícitamente.

S es la referencia de overlap, nunca una S predicha para satisfacer el objetivo.
Para A un átomo y las imágenes R, se usa la contracción de matrices hermíticas:

```
C_A(X,Y) = sum_(i in A,j,R,spin) X_ij,spin(R) Y_ji,spin(-R)
         = sum_(i in A,j,R,spin) X_ij,spin(R) Y_ij,spin(R), para matrices reales
p_A(D,S) = C_A(D,S), con S compartida entre los canales spin
N(D,S) = sum_A p_A(D,S)
u_A(H,D) = C_A(H,D)
v_A(S,E) = C_A(E,S), E = EDM
```

H/EDM se leen en eV; población en electrones; residuos de energía en eV por
átomo. Las poblaciones son Mulliken, no cargas netas ni cargas de Bader.
La identidad de referencia es `u_A(H_ref,D_ref) = v_A(S_ref,E_ref)`.
Se comprueban spin, geometría, base, celda, imágenes y cero de energía común.
La referencia energética se rechaza si su residuo máximo supera 1e−6 eV por
átomo; no se desplaza EDM de forma implícita para hacerla pasar.

### 3.1 Carga y Mulliken: observable de la matriz realmente representada

El grafo contiene bloques G; la predicción reconstruida es cero fuera de G,
como en la conversión actual. Para DM residual se restaura D0 en los bloques
on-site antes de evaluar la población, nunca se interpreta delta-D como DM.
La referencia de población se calcula con **toda la DM/S archivada**.

```
delta_p_A = p_A(D_pred_representada,S_ref) - p_A(D_ref,S_ref)
L_N,s = [(sum_A delta_p_A) / n_atoms_s]^2
L_M,s = mean_(A in s) delta_p_A^2
```

No se multiplica por N ni se divide por un N inferido de `q0`. Se normaliza
el error de conteo por número de átomos para comparar tamaños sin ocultar el
error total: el log también comunica delta-N sin normalizar, en electrones.
Ambos términos pueden elegirse por separado, aun cuando comparten contracciones.
Fuera de G se resta la contribución **conocida** de la referencia, no se inventa
una etiqueta cero. Esto evita confundir una pérdida exacta con su versión
proyectada cuando el cutoff excluye contribuciones pequeñas.

### 3.2 Energía: consistencia de la perturbación representada

Una comprobación adicional encontró contribuciones H–DM fuera del grafo actual:
la identidad usando solo G tiene residuos de aproximadamente 3e−6, 1.4e−5 y
4.4e−6 eV por celda en las tres muestras. No se puede exigir cero a esa identidad
truncada ni cambiar el cutoff/readout silenciosamente.

Se elige un objetivo de **perturbación sobre la referencia completa**: fuera de
G se conserva la matriz de referencia al definir este observable de entrenamiento.
Esto no altera la matriz que el writer devuelve y no se presentará como su
energía total exacta. Su error fuera del grafo se mide aparte en el protocolo.

```
r_A,ref = u_A(H_ref,D_ref) - v_A(S_ref,E_ref)
H predicha: r_A = r_A,ref + C_A(delta_H_G,D_ref)
EDM predicha: r_A = r_A,ref - C_A(delta_E_G,S_ref)
L_E,s = mean_(A in s) r_A^2
```

Las diferencias se toman respecto de etiquetas reales. La matriz no predicha
se mantiene fija y E_ref procede del TSDE: no se construye EDM a partir de H
predicha. Una perturbación genérica produce pérdida y gradiente no nulos.
La ruta de consistencia para DM usa el mismo contrato:
`r_ref + C_A(delta_D_G,H_ref)`, sin añadir otro mecanismo ni imponer idempotencia.

Con `gauge_invariant_mse`, delta-H es la ya alineada `delta_H - c_s S` **en G**,
con el mismo c, pesos y canales que el término matricial existente. No se
recalcula una alineación distinta para la energía. Desplazar conjuntamente
H_ref/H_pred por aS y E_ref por aD conserva el objetivo; añadir aS a H_pred
también se elimina con la alineación. Con `elementwise_mse` se conserva el
gauge de referencia; un desplazamiento unilateral de H es error físico.

La suma se agrupa por átomo antes de elevar al cuadrado, no por orbital. Así
se conserva la invariancia frente a rotaciones de la base dentro de cada átomo.

## 4. Interfaces mínimas

### 4.1 Selección y datos auxiliares

- `MatrixDataModule(..., physical_labels=None)` y el mismo argumento final en
  `TorchBasisMatrixDataset`. Acepta una lista con `"charge"` y/o `"energy"`;
  None/lista vacía conserva la ruta actual sin lecturas ni campos adicionales.
- Charge necesita DM de referencia y S real. Energy necesita H/S/DM/EDM reales
  y compatibles. No se cargan las matrices de energía al pedir solo charge.
- Primer soporte file-backed: FDF local. Se reutilizan lectura de base,
  conversión sparse→bloques y aplanado del procesador, no scripts de otro repo.
  ZIP solicitado con este modo nuevo se rechaza claramente; la ruta ZIP antigua
  sigue funcionando con el modo apagado. La API directa prepara datos por el
  mismo ayudante a partir de matrices sparse reales y la malla k explícita,
  sin inventar un fichero. No se admiten arrays auxiliares sueltos como
  sustituto de esa preparación y validación física.
- Los componentes físicos de DM/EDM se separan explícitamente de S al preparar
  este modo opcional. No se cambia la interpretación de datos/checkpoints legacy.
  Se registra EDM como `OrbitalMatrix`, sin warning de fallback ni baseline ficticia.
- Se soporta D0 no polarizada existente. DM colineal con `sub_point_matrix=True`
  se rechaza inicialmente: la resta/restauración legacy no tiene aquí una
  convención colineal certificada. Colineal con baseline apagada sí se prueba.

El contrato tensorial usa campos `physical_*`, concatenables por PyG, sin
`index` en nombres que provocarían desplazamientos automáticos. Contiene:

- auxiliares node/edge con el orden exacto de las etiquetas: S escalar y los
  H/D/EDM requeridos, con componente spin explícita;
- población atómica completa de referencia y r_A,ref completos;
- baseline on-site de DM cuando aplica y descriptor del tipo/spin/unidades;
- procedencia y máscaras que distinguen valores observados, ceros estructurales
  del formato sparse y valores realmente desconocidos.

Las referencias son targets exclusivos de la pérdida, nunca features de MACE,
contexto del readout ni entradas a una red de cargas; se prueba esa separación.

La ausencia de una matriz no se convierte en ceros. Tampoco se convierte un
NaN arbitrario en cero. Los ceros estructurales de una matriz sparse presente
solo tienen esa semántica explícita de representación; no certifican la DM
continua no truncada. Si una etiqueta desconocida tiene peso físico no nulo,
se rechaza el objetivo. Con peso exactamente cero puede excluirse antes del
producto, sin producir `NaN*0` ni falsear una población observada.

### 4.2 Cálculo y pérdida

`elementwise_mse` y `gauge_invariant_mse` reciben al final de sus argumentos
opcionales `electron_weight=0.0`, `mulliken_weight=0.0`, `energy_weight=0.0`.
Los pesos deben ser finitos y no negativos. Electron/Mulliken solo para DM;
gauge solo para H. Los términos se añaden a la pérdida matricial sin cambiar
sus denominadores previos. Pesos cero no exigen datos auxiliares.

Una función común calcula índices de propietarios atómicos a partir de los
layouts/selección de aristas existentes. `TypeLayoutCollate` los prepara en CPU
cuando se necesitan y los mueve por el mecanismo de layout actual. NumPy y
Torch comparten el contrato y la fórmula; no se desconecta autograd ni se
reconstruyen matrices densas durante el paso de entrenamiento.

Un bloque on-site aporta una sola vez. Una arista simétrica representada una
sola vez aporta a **ambos extremos**, incluso si son imágenes del mismo átomo.
En aristas dirigidas completas solo aporta al extremo origen. Este factor no
es `edge_weight`, que continúa siendo una ponderación de la pérdida matricial.
Los batches pueden mezclar tamaños/especies pero no convenciones de spin
incompatibles en un tensor. Se prueban grafos sin aristas y bases vacías parciales.

Cada pérdida física es la media de sus L_s por estructura. Lightning agrega
sumas/conteos mediante `_EpochWeightedLoss`; no promedia RMSEs ni mezcla sus
denominadores con el número de elementos matriciales. Los contadores y stats
son desprendidos de autograd; las contracciones que forman la pérdida no lo son.

### 4.3 Validación temprana y compatibilidad

En preparación/arranque de fit se validan también las pérdidas de etapas
futuras: matrices disponibles, pesos, target, spin, baseline, forma/componentes
y contrato de collate. Un término solicitado sin referencias falla antes del
primer paso de optimización. La API directa vuelve a validar su contrato.
No hay desactivación silenciosa ni fallback a una pérdida sin física.

Para S singular/mal condicionada: se valida S(k) en float64 sobre la malla SCF
de referencia declarada (Γ para una referencia molecular explícita). Se rechaza
lambda_min<=0 o lambda_min/lambda_max<=1e−12, sin recorte de autovalores ni
regularización. Esta comprobación CPU se hace al preparar las referencias, no
por paso; no certifica puntos ajenos a esa malla. El ayudante de API directa
recibe S sparse y k explícitos y pasa la misma validación antes del collate;
no acepta una marca de confianza que el caller use para saltarse ese paso.

Checkpoints anteriores no requieren campos ni parámetros nuevos. No se modifica
el número de salidas, el readout Hamiltoniano, MACE ni los defaults científicos.
El modo coefficient-only solo se usa cuando los pesos físicos son cero; con
física se necesita la ruta matricial existente, nunca se pierde el término.
Los módulos NumPy siguen importables sin Torch. Piso Python 3.9 preservado.

## 5. Verificación que debe preceder al cierre

RED→GREEN con oráculos analíticos independientes: S no ortogonal, imágenes
contrapuestas, factores spin, población y conservación de carga, errores de
energía no triviales, rotaciones por átomo, gauge común, intercambio de extremos,
hermiticidad, gradcheck y diferencias finitas. Deben detectar específicamente
el plegado incorrecto a Γ y el factor incorrecto de media arista.

Regresiones de NaNs/máscaras/matrices ausentes, grafos heterogéneos, varios
componentes, filas estructuralmente nulas, baseline, S singular/mal condicionada,
ocupaciones fraccionarias sin idempotencia, configuraciones incompatibles,
pesos cero, métricas NumPy, stages, checkpoint legacy y candidato.

Smoke CPU propio Lightning con datos reales leídos sin escritura; cada término
por separado debe participar en la pérdida y producir gradiente finito no nulo.
Datos/condiciones que no permitan un término se reportan, no se rellenan.
No se sustituye la validación real por fixtures sintéticas. Se contrastan también
las poblaciones contra `sisl.DensityMatrix.mulliken` y los objetivos contra las
referencias completas antes/después de aplanar a G.

Gates: enfocados, regresiones, suite CPU completa, Ruff, mypy según gate existente,
Python 3.9 real/compile y `git diff --check`, con PYTHONPATH absoluto. Revisión
del diff y corrección de hallazgos importantes antes de reclamar integración.

## 6. Coste y protocolo experimental

Por paso: O(elementos de etiqueta × canales) en productos/scatter y O(átomos)
en reducciones; sin diagonalizaciones ni matrices densas orbital×orbital.
Los auxiliares aumentan RAM/transferencia y pueden aumentar VRAM; no se declara
ningún ahorro. La validación de S cuesta O(n_k n_orb^3) y lecturas adicionales
al preparar datos: se medirá por separado. `store_in_memory` reutiliza las
referencias preparadas; la lectura on-demand puede repetir ese coste.

Benchmark CPU propio: warmup separado, rondas alternadas, misma predicción y
trabajo, mediana/MAD, versiones, threads, tamaños/componentes y carga del host.
Separar carga/validación, collate, pérdida+backward y paso completo Lightning.

Protocolo posterior: hashes/splits congelados, referencia preservada, mismo
modelo/presupuesto, tres semillas pareadas; control, N, Mulliken y energía por
separado antes de combinaciones. Medir matrices, población total/atómica, energía
de perturbación y **observables de la matriz reconstruida completa**, bandas,
gap y OOD pertinentes. Desglosar contribuciones fuera de G. No promover una
pérdida que empeore OOD aunque baje su propio residuo de entrenamiento.

## 7. Archivos previstos y condición de salida

Producto: `core/data/metrics.py`, `core/data/matrices/__init__.py`, ayudante
acotado de preparación/contracción en `core/data/physical.py`, integración en
`bindings/torch/data/dataset.py`, `tools/lightning/data.py` y
`tools/lightning/model.py`. Reutilizar `processing.py`/layout existente; cambiar
esa interfaz solo si una regresión demuestra que es imprescindible.
No modificar readouts ni añadir dependencias.

Tests vecinos del núcleo/Torch/Lightning, script propio de benchmark CPU,
informe de resultados/protocolo y ambos roadmaps. El plan posterior fijará
los pasos RED→GREEN y comandos concretos una vez aprobado este diseño.

El cierre exige todos los objetivos aplicables integrados/verificados CPU,
límites y costes explícitos, revisión final y protocolo preparado. El informe
distinguirá no aplicable, bloqueado por recurso, CPU verificado y experimental
pendiente. Este documento, por sí solo, no cumple ese cierre.
