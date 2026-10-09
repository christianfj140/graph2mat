# 7.4 — Slater–Koster neuronal con residual equivarante: diseño CPU

Estado: **propuesta pendiente de aprobación del usuario**. No autoriza código
de producto. Goal creado el 2026-10-09; base inspeccionada `8c8dc3e`, rama
`g2m-v2`, checkout `/home/christian/repositorios/graph2mat`.

## 1. Intención y contrato

El usuario quiere adelantar mejoras potenciales de precisión, velocidad o VRAM.
Este goal incorpora **una** primera variante opcional de 7.4, comprueba su
corrección científica e integración en CPU y mide su coste CPU. No promete
ganancia: implementación, evaluación de precisión y evaluación GPU son hitos
separados. Readout actual, defaults, APIs anteriores y checkpoints legacy se
conservan. No se cambia el backbone MACE ni se reabre 4.4–4.5 o 7.1.

Restricciones: mismo checkout, sin worktree; preservar cambios ajenos; Python
>=3.9; sin dependencias nuevas, GPU, entrenamientos comparativos largos,
fonones, autorun, colas, otros repositorios, push/merge ni procesos ajenos.
Las pruebas y benchmarks usan fixtures propias, no necesitan etiquetas reales.

## 2. Contexto comprobado y alternativas

`roadmap.md` mantiene 7.4 pendiente. Task 4 de
`roadmap_precision_rendimiento.md` exige una sola variante y un plan separado.
La parte física 4.4–4.5 tiene cierre CPU, no evaluación de precisión.

Puntos de extensión existentes:

- `bindings/e3nn/modules/edge_operations.py`: operaciones por arista;
  `E3nnLowRankEdgeBlock` de 7.1 ya ofrece residual bilineal/lineal con gates
  radiales y `rank` configurable. No es SO(2) ni Slater–Koster.
- `bindings/e3nn/modules/matrixblock.py`: RTP, coeficientes, reconstrucción,
  escalado e intercambio/transposición; no duplicar esas rutas.
- `core/modules/graph2mat.py`: `_data_get_edge_args` entrega datos pareados
  por dirección; layouts y agrupaciones ya existen.
- `models/mace.py`: calcula vectores periódicos y radiales en `_readout`,
  pero actualmente solo publica `edge_attrs` y `edge_feats` al readout.
- `tools/lightning/models/mace.py`: selectores `edge_block_readout` y
  `edge_block_readout_kwargs` ya permiten elegir una clase opcional.
- `core/data/basis.py` y `processing.py`: convenciones orbitales y conversión
  cartesiana/base existentes. No transformar de nuevo posiciones ya convertidas.

**A. SO(2) general en marco de enlace.** Mayor oportunidad de reemplazar
acoplamientos y reconstrucción densos; también exige una nueva representación
de features locales, operaciones SO(2) completas, manejo de paridad y auditoría
de libertad de marco. Una implementación solo diagonal no representa un entorno
anisótropo general. Es mayor cambio que la primera entrega propuesta.

**B. Slater–Koster neuronal + residual existente — recomendada.** Añade un
prior angular de dos centros y conserva anisotropía mediante el residual de
7.1. Reutiliza coeficientes/RTP e integración existentes. Es menor cambio,
pero mantiene reconstrucción y residual: **no** obtiene automáticamente la
reducción de complejidad de un readout SO(2) directo. Puede ser más lento que
7.1 solo; la comparación debe mostrarlo.

Se propone B exclusivamente. No construir eSCN/QHNetV2 ni una segunda opción
SO(2) en este goal. La elección es una recomendación, no una aprobación.

## 3. Modelo matemático

Para cada par ordenado de capas radiales completas a/b, con momentos la/lb,
la rama SK predice `min(la, lb)+1` integrales por componente real:
`V_ab,q`, q=0..min(la,lb): sigma, pi, delta y phi cuando corresponda.
Capas con igual l pero distinto índice radial **no** comparten integral por
accidente. C1/C2 significan uno/dos canales orbitales reales independientes;
no representan rotaciones de spin ni SOC.

En el eje de referencia interno `n0=(0,1,0)`, compatible con la convención
de rotaciones Y de e3nn instalado, el bloque axial de una integral unitaria
es `P_ab,0[la,lb]=1`; para q>0 tiene unos en
`[la-q,lb-q]` y `[la+q,lb+q]`. Los demás elementos son cero.
Esta definición fija signos y normalización de V en la base interna; no
importar tablas cartesianas sin convertirlas. Debe comprobarse contra
rotaciones de e3nn y oráculos orbitales independientes.

El bloque SK es el transporte orbital de esa matriz axial hacia la dirección
del enlace. En lugar de escoger un eje transversal y diferenciar ángulos Euler,
se implementa la expresión equivalente en coeficientes globales:

```
L = |la-lb|, |la-lb|+2, ..., la+lb
A_ab,L = sum_q T_ab,Lq * V_ab,q
c_ab,LM(n) = A_ab,L * Y_LM(n)
c_total = c_SK + c_residual
B = coefficients_to_block(c_total)  # ruta existente
```

`T` es constante. Se calibra al construir la operación: proyectar cada
`P_ab,q` con el Q/RTP existente, extraer la amplitud paralela a `Y_L(n0)` y
comprobar tanto la reconstrucción como la ausencia de componentes prohibidas.
La asignación incluye par de capas, instancia de irrep y componente C1/C2;
no inferirla solo por dimensión o por la multiplicidad agregada de l.
No se crea un mapa global denso KxK. Los buffers nuevos solo pertenecen a la
variante opt-in y respetan `.to(dtype/device)` y checkpoint.

El residual usa `E3nnLowRankEdgeBlock`, sin modificar su definición ni
renombrarlo como arquitectura nueva. Produce el resto de coeficientes,
incluidas contribuciones de entorno fuera de la familia axial SK. No se
afirma universalidad ni equivalencia al FCTP completo a rango finito.

### 3.1. Libertad de marco y O(3)

`P_ab,q` entrelaza las rotaciones sobre el eje; por tanto, transportar el
bloque con R o con R seguida de cualquier giro sobre n0 da el mismo resultado.
La expresión con Y evita construir R en producción: no tiene cartas,
ejes transversales auxiliares ni singularidad angular en los polos.

Para orbitales naturales, paridad del bloque es `(-1)^(la+lb)`. Los L usados
cumplen `(-1)^L=(-1)^(la+lb)`, garantizando inversión de la rama SK. El
residual conserva su contrato O(3). Pruebas deben certificarlo; el argumento
no sustituye evidencia. No recortar L=6 para pares f-f.

### 3.2. Entradas, conditioning y cutoff

La operación nueva, propuesta como `E3nnSlaterKosterEdgeBlock`, consume
tuples `node_feats`/`edge_messages`, `edge_feats` y `edge_vectors`, ordenadas
por los mecanismos existentes. `edge_vectors` son vectores **dirigidos**
incluyendo shifts periódicos, en el mismo sistema de coordenadas que features
y orbitales. No inferir dirección desde canales radiales.

Un MLP pequeño de ancho 32 por operación produce V a partir de radiales
promediadas y canales 0e de ambos extremos de las fuentes disponibles. Solo
0e entra al conditioning; no componentes cartesianas, 0o ni etiquetas.
El residual recibe las fuentes originales. `rank` conserva su significado
de 7.1, valor inicial 4. No añadir otro catálogo de opciones sin necesidad.

Multiplicar V por `tanh(sum_k radial_k**2)` garantiza V=0 cuando la entrada
radial envolvente es cero, incluso con biases o nodos no nulos. La rama
residual ya cumple ese contrato. No garantizar cutoff para radiales arbitrarias:
MatrixMACE debe usar el embedding/envelope existente y las pruebas comprobarlo.
La derivada del bloque también debe anularse en el corte suave.

Vectores no finitos, longitud cero o direcciones pareadas inconsistentes
son errores explícitos. Longitud positiva arbitrariamente pequeña no se
trunca silenciosamente. Self-image periódico con shift no nulo se admite;
el onsite permanece en el readout de nodos actual. Lote vacío devuelve formas
correctas sin reducción inválida. No usar clamp como sustituto de validar cero.

## 4. Integración y compatibilidad

Alcance inicial: orbitales reales de paridad natural, capas completas s/p/d/f,
múltiples capas radiales, pares homo/heteronucleares, C1/C2, agrupaciones
`point_type`/`basis_shape`/`max`, readout único o por interacción sum/mean.
Se mantienen layouts, intercambio y ensamblado periódico existentes.

Cambios previstos, únicamente después de aprobación:

1. Una operación nueva y su export en `bindings/e3nn/modules/`; reutilizar
   extracción escalar y residual existentes cuando resulte compatible.
2. `matrixblock.py`: proporcionar metadatos de bases/Q/componentes a la nueva
   operación mediante parámetros de constructor explícitos. Extensión aditiva,
   solo para firmas que los soliciten; ningún argumento obligatorio nuevo
   ni buffers añadidos al readout legacy. Certificar la correspondencia Q/irreps.
   La advertencia de irreps inalcanzables debe considerar la nueva fuente
   angular SK; no afirmar que toda la salida residual se vuelve alcanzable.
3. `models/mace.py`: publicar `edge_vectors` del cálculo existente y extender
   validaciones a la nueva operación real, no solo al selector Lightning.
4. Selector/configuración existentes: comprobar import class path, kwargs,
   guardado/recarga y ausencia de cambios en defaults. No crear otro selector.
5. Pruebas vecinas, benchmark reproducible, informe y protocolo experimental.

No soportados inicialmente: orbitales de paridad no natural, l>3, complejo,
no colineal/SOC o simetrías internas distintas del bloque offsite `ij`.
Rechazar al construir, no truncar ni hacer fallback silencioso. El fallback
de reconstrucción de coeficientes existente para RTP no certificados se conserva.

`hamiltonian_local_context=True` se rechaza: concatena canales tras el envelope
y puede abrir colas radiales. Conservar el guard de 7.1 y cubrir custom readouts.
Tampoco permitir shifts escalares de arista no nulos que reabran el cutoff;
escalados por irrep son compatibles si se prueba la ruta de ajuste existente.
No ampliar gradient checkpointing, compile o AMP: restricciones existentes
se conservan y las combinaciones no certificadas se documentan, sin prometer
compatibilidad nueva. No se ejecuta CUDA/AMP CUDA.

## 5. Criterios de aceptación

- RED→GREEN observable para oráculos y contratos nuevos.
- SK aislada con integrales fijadas: s-s=Vss, s-p=Vsp*n y
  p-p=Vpi*I+(Vsigma-Vpi)*n*n^T, convirtiendo bases explícitamente.
  Los oráculos no llaman al constructor de T ni al forward de producción.
- Pares s/p/d/f y capas repetidas: comparación con transporte orbital
  independiente en varias direcciones; verificar cada integral q y cada
  componente, signos, normalización y rango del mapa T.
- Residual apagado/activado por pesos controlados: demostrar aportación SK
  y anisotropía fuera de la familia axial. No esconder SK=0 permanente.
- Rotaciones propias, reflexiones/inversión, intercambio y permutaciones;
  rotaciones alrededor del enlace; dos marcos auxiliares de test, ejes y
  direcciones próximas a polos. El forward sin cartas debe coincidir.
- Gradcheck/diferencias finitas en posiciones/vectores, features y parámetros
  con pesos no triviales; pruebas de continuidad angular y cutoff con nodos
  no nulos. Revisar tolerancias según precisión real de Q, no aflojarlas para
  ocultar fallos. Separar error float32 y float64.
- Periódicas con imágenes opuestas, self-images no nulas, lotes vacíos y
  heterogéneos, layouts y modos/grupos/C1/C2 declarados; errores negativos.
- Coeficientes/reconstrucción/pérdidas existentes consistentes, dos pasos de
  entrenamiento CPU y reload estricto. Defaults legacy conservan claves y
  predicciones con pesos idénticos; ninguna migración implícita de checkpoints.
- Pruebas enfocadas y suite completa `pytest src/graph2mat tests -q --tb=short`,
  CPU forzada, PYTHONPATH absoluto y límites de hilos. Gates estáticos/Python
  3.9 aplicables. Registrar comandos, versiones, conteos y omisiones reales.
  El gate histórico 2477/2 no cuenta como resultado de 7.4.

## 6. Medición y protocolo posterior

Benchmark CPU con mismo backbone, fixture, dtype, hilos y alcance:
readout completo actual, 7.1 solo (ablación) y SK+residual propuesto. No son
arquitecturas de igual capacidad y no se exige igualdad de predicciones.
Warmup, rondas alternadas, mediana/MAD; parámetros, forward, backward y paso
completo. Distinguir operación aislada y modelo, inicialización y ejecución.
Memoria de host solo bajo una metodología reproducible, identificando lo que
mide; no llamarla VRAM. Guardar resultados crudos, comandos y estado Git.

Preparar sin ejecutar protocolo GPU/VRAM/precisión: al menos semillas 0/1/2,
splits congelados y presupuesto comparable, comparación de las tres variantes,
H raw/alineado al gauge, bandas/gap y OOD pertinentes cuando haya referencias.
No inventar datos ni sustituir métricas físicas por pérdidas sintéticas.
La selección/promoción queda vetada si empeora OOD según umbrales fijados antes
de ejecutar; no se cambian defaults por ganar un microbenchmark.

## 7. Estado y cierre

- [x] Crear goal, confirmar rama/base y conservar cambios ajenos.
- [x] Inspeccionar puntos de extensión y comparar dos alternativas.
- [x] Escribir propuesta y revisar alcance/contradicciones sin producto.
- [ ] Aprobación explícita del usuario de este diseño escrito.
- [ ] Plan escrito aprobado y ejecución Native.
- [ ] Implementación, tests, benchmark y gates CPU con evidencia nueva.
- [ ] Informe, resultados crudos, protocolo GPU/precisión y roadmap actualizado.

El goal de implementación CPU no se cierra por esta propuesta. 7.4 global
seguirá pendiente de evaluaciones GPU/precisión aunque el código CPU se cierre.
Ante inviabilidad del mapa, fallo científico persistente o necesidad de ampliar
alcance, informar evidencia y pedir dirección; no cambiar a SO(2) por sorpresa.

## 8. Fuentes y límites de la evidencia

La arquitectura propuesta es una decisión de ingeniería para este repositorio,
no una reproducción ni una mejora experimental establecida por estos trabajos:

- [Slater y Koster, 1954](https://journals.aps.org/pr/abstract/10.1103/PhysRev.94.1498):
  origen del modelo de integrales de dos centros; página/abstract consultados,
  texto completo con acceso restringido. No se copiaron tablas del artículo.
- [Passaro y Zitnick, eSCN, 2023](https://proceedings.mlr.press/v202/passaro23a.html):
  reducción SO(3)→SO(2); sus ganancias no se atribuyen a esta variante SK+RTP.
- [Yu et al., QHNetV2, 2026](https://proceedings.mlr.press/v306/yu26y.html):
  readout/operaciones locales SO(2), alternativa más amplia no implementada aquí.
- [e3nn, armónicos esféricos](https://docs.e3nn.org/en/stable/api/o3/o3_sh.html):
  referencia de equivariancia/normalización. La web estable describe 0.5.1;
  el código local inspeccionado es 0.4.4 y manda para convenciones concretas.

No se han ejecutado tests, benchmarks ni simulaciones de esta arquitectura
durante la fase de diseño. Las afirmaciones matemáticas anteriores son
requisitos a demostrar con los oráculos indicados, no resultados ya obtenidos.
