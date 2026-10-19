# Diseño 7.6: objetivos conjuntos H/S/DM/EDM

Fecha: 2026-10-18. Graph2Mat, rama `g2m-v2`, checkout actual.
Estado: diseño aprobado por el usuario («Apruebo», 2026-10-18), tras presentar
este documento. Plan en `../plans/2026-10-18-multitarget-7-6.md` aprobado
después con «Apruebo»; ejecución Native cerrada y verificada en CPU. Este bloque pertenece al Goal
conjunto; su aprobación no autoriza campañas científicas ni GPU.

Resultado técnico:2892passed/20skipped/0fallos en suite final, revisión única
con tres Important corregidos RED→GREEN. Informe final:
`auditorias_mejoras/informe_multitarget_7_6_cpu_20261018.md` (raíz del repo).
Decisiones verificadas: SK conserva C1/C2; fitting scaleshift exige etiquetas
completas; CSV al primer uso y selector legacy rechazado. No certificación
científica, nuevos defaults ni campaña; detalles/costes en informe y ledger.

## 1. Intención y éxito

Añadir un modo opcional de aprendizaje de varias matrices del mismo grafo,
con un solo backbone y salidas identificadas por objetivo, pérdidas separadas
y reconstrucción física explícita. No basta un tensor con más columnas sin
semántica de unidades, spin, máscaras, gauge y soporte.

El éxito técnico es cargar/preparar objetivos compatibles, entrenar un paso
sintético CPU con gradientes correctos, guardar/recargar y reconstruir cada
salida, sin alterar la ruta legacy. No exige ni autoriza entrenamientos
científicos, GPU, nuevos datasets, dependencias, otros proyectos, commits,
worktrees, defaults, cola/autorun ni trabajo de fonones.

## 2. Estado existente y datos accesibles

- H+S ya se cargan y predicen mediante componentes semánticas explícitas.
  El bloque 4.1 certifica ese contrato, no implementa multitarea DM/EDM.
- MatrixMACE procesa el grafo una vez y entrega node_labels/edge_labels.
  Los readouts por interacción agregan capas: no son objetivos DM/EDM.
- Los readouts ya soportan `n_matrix_components`; se reutiliza esa capacidad
  en lugar de duplicar MACE o crear un ensamblador orbital nuevo.
- `physical.prepare_physical_data` y `load_physical_data` preparan un target
  H, DM o EDM por muestra, con otras matrices como referencias auxiliares.
  `physical_*` no son targets conjuntos ni deben entrar en features.
- Existen fixtures analíticas moleculares/periódicas y fraccionarias para
  tests. No se inventarán etiquetas científicas de entrenamiento a partir
  de la diagonalización de una predicción ni productos elementwise.
- Inventario de nombres sin seguir enlaces, incluyendo ignorados y excluyendo
  .git/.venv:11914HSX,3TSHS,8468FDF; ningún .TSDE/.DM/.EDM regular enumerado con
  esos sufijos exactos. Hay enlaces de campañas hacia datos externos que no
  se siguieron. No se inspeccionaron ZIP ni todas las variantes de formato;
  este conteo no prueba ausencia absoluta, pero tampoco certifica un conjunto
  H/S/DM/EDM real compatible accesible dentro del alcance actual.
- Los inventarios históricos de grafeno/hBN en otro repositorio no autorizan
  leer ese otro repositorio en este Goal. La validación científica queda
  pendiente de un recurso autorizado, sin bloquear las pruebas sintéticas.

## 3. Enfoque propuesto y alternativas

**Enfoque aprobado:** reutilizar el readout multicomponente
existente con un descriptor fijo de objetivos y vistas nombradas. Cada objetivo
dispone de sus componentes de salida y parámetros de salida correspondientes;
se comparten backbone y maquinaria equivarante, no las etiquetas ni su pérdida.
El packing es una representación interna versionada, no una interpretación
física de todas las columnas como Hamiltoniano.

Alternativas descartadas: entrenar modelos completos independientes (no es
multitarea de backbone compartido), concatenar canales sin descriptor (mezcla
semánticas), o un framework general de tareas/heads arbitrarios (innecesario
para estos cuatro objetivos). No añadir MoE ni una arquitectura nueva.

## 4. Contrato de objetivos y compatibilidad

Modo opt-in `matrix_targets`, por defecto None: la ruta actual no lee matrices
adicionales, no cambia retornos/hyperparameters requeridos ni nombres de pesos
de checkpoints legacy. Los nombres admitidos son `hamiltonian`, `overlap`,
`density_matrix`, `energy_density_matrix`. Rechazar desconocidos y duplicados.

El descriptor normaliza un orden canónico H,S,DM,EDM, versión1, canales físicos
por objetivo y slices exactos. Se conserva en datos/config/checkpoint y se
contrasta en carga/preflight: el mismo número de columnas no prueba que dos
layouts sean compatibles. La nueva ruta exige H y S, y permite añadir DM,
EDM o ambas; el caso H/S sin adicionales también debe ser válido. No se
admite un target S-only, DM-only o EDM-only por esta nueva ruta: ya existen
caminos de un objetivo y no se duplican aquí.

Inicialmente matrices reales no polarizadas o colineales. S tiene un canal
escalar; H,DM,EDM comparten uno o dos canales físicos. Para el conjunto completo
son cuatro componentes no polarizadas o siete colineales. No confundirlos con
SOC, spin raw o cuatro/siete Hamiltonianos. H/EDM en la convención eV de sisl,
S adimensional y DM con su convención de ocupación explícita, sin factor2
inventado en la población. No imponer idempotencia a ocupaciones fraccionarias.

Decisión aprobada: el modo nuevo aprende matrices directas, con
`sub_point_matrix=False`; no construye ni resta H0/DM0 ni acepta un delta como
si fuera una matriz física completa. Las recetas legacy con sus baselines
conservan su camino sin cambios. Soportar baselines independientes por target
sería una ampliación posterior explícita, no una resta implícita compartida.

Geometría, especies/base y orden orbital, celda, imágenes, spin y cero energético
deben coincidir entre matrices. Reutilizar validadores/lectores físicos donde
sean aplicables. TSDE no contiene S: adjuntar la S real del mismo snapshot,
no convertir los ceros que deja un lector sin overlap en etiquetas.
DM/EDM son matrices archivadas o fixtures de test explícitas, no H*DM elementwise.

## 5. Carga, máscaras y grafo

Usar el mismo grafo/base/layout de bloques para todos los objetivos, mediante
las conversiones existentes. Evitar graphs/backbones independientes por target.
El soporte puede diferir en los archivos: no tratar coeficientes DM/EDM
desconocidos como ceros observados. Conservar máscaras por objetivo/componente,
incluida su simetría entre imágenes. H/S conservan sus ceros estructurales
según el contrato existente. Un NaN arbitrario no se transforma en cero.

Dos entradas autorizadas para esta API: FDF local con matrices correspondientes
o un bundle explícito de objetos sisl compatibles, útil también para tests.
Solo abrir archivos resueltos dentro del recurso autorizado; no seguir enlaces
hacia otros repositorios por el hecho de estar presentes en un overlay.
Esta es una restricción de los accesos del agente durante este Goal, no una
raíz absoluta que deba codificarse en la librería. La API recibe rutas del
caller; aquí se verifica con fixtures/temporales propios, sin explorar datos
externos ni ampliar permisos por inferencia.
ZIP del modo nuevo se rechaza claramente inicialmente; legacy ZIP no cambia.

Todos los targets seleccionados deben existir al preparar etiquetas. Una
matriz completa ausente falla temprano, incluso si su peso es cero; para no
pedirla se elimina de matrix_targets. Los coeficientes individualmente
desconocidos se excluyen de la pérdida mediante su máscara antes de restar o
multiplicar, sin NaN*0 ni ocultar errores de predicción sobre entradas conocidas.
Si un target con peso positivo no tiene observaciones válidas para una muestra,
rechazarla con estructura/target/motivo, no alterar silenciosamente el promedio.

Inferencia desde geometría no lee H/S/DM/EDM de referencia como features. El
descriptor se obtiene de la config/checkpoint, no de etiquetas de test. El
batch puede mezclar tamaños y especies compatibles, no layouts/spins distintos.

## 6. Pérdidas y física opcional

Pesos por target explícitos, finitos/no negativos, al menos uno positivo;
registrar pérdidas/métricas separadas por objetivo y denominadores reales.
No una MAE/MSE anónima que mezcle eV, S y ocupaciones por número de columnas.
Los pesos son coeficientes en las unidades convencionales declaradas, no
normalizaciones automáticas aprendidas ni ajustadas usando OOD.

H usa `gauge_invariant_mse` o `elementwise_mse` según selección explícita; S,
DM y EDM usan su propia pérdida matricial existente compatible con máscaras.
Gauge de H usa S de referencia, nunca S predicha para reducir artificialmente
su propio residual. Reutilizar las métricas y agregación por elementos/estructura
existentes, con el mismo denominador que cada pérdida en solitario. La suma
ponderada de pérdidas por target debe coincidir con su cálculo independiente.

Pesos cero no evalúan el término ni propagan NaNs o gradientes de ese objetivo;
el backbone sigue recibiendo las contribuciones de los targets activos.

Reutilizar opcionalmente 4.4–4.5 por objetivo: N/Mulliken en DM y energía de
perturbación en H/DM/EDM con los partners de referencia certificados existentes.
S permanece referencia para esas pérdidas, aunque otro head la prediga. No
introducir una identidad nueva entre las cuatro matrices predichas ni afirmar
que la energía sea la exacta de la matriz truncada del writer. Los coeficientes
físicos de ese objetivo siguen apagados por defecto y sus preflights continúan
obligatorios; partners ausentes/incompatibles producen error temprano.
Un acoplamiento no lineal entre todas las predicciones sería otro diseño, no
una consecuencia implícita de reutilizar estas pérdidas.

## 7. Salidas, reconstrucción y checkpoint

Exponer predicciones nombradas por target además del packing interno necesario
para reutilizar readouts. Cada vista es diferenciable; ninguna copia NumPy
interrumpe autograd en el cálculo de pérdidas. No interpretar el tensor packed
como `sisl.Hamiltonian` de cuatro/siete canales.

Reconstrucción con selección explícita de objetivo: H se combina con S predicha
para un objeto Hamiltonian físico; S produce Overlap; DM/EDM mantienen sus
componentes físicas y la convención de overlap elegida explícitamente. En
este modo, reconstrucciones físicas autocontenidas usan S predicha por defecto
del contrato nuevo, nunca S de referencia a escondidas. Las métricas de
entrenamiento con S_ref se distinguen de espectros con S_pred. No reparar una
S_pred no SPD ni prometer espectros válidos universalmente.

Writer nuevo-modo exige elegir target y formato compatible antes de escribir,
sin volcar componentes adicionales como spin ni descartarlas silenciosamente.
No ampliar endpoints HTTP multi-target en este bloque; el servidor debe
rechazar un layout no admitido antes de publicar salida mal interpretada.
La serialización de un objetivo explícitamente separado puede usar los writers
existentes. Los checkpoints guardan descriptor y pesos/config de pérdidas;
round-trip candidato y carga legacy se prueban por separado.

En la ruta nueva se rechazan de entrada las combinaciones aún no certificadas
que asumen un Hamiltoniano único o S en última columna: long_range,
hamiltonian_local_context y entrenamiento coefficient-only. También rechazar
etapas de entrenamiento que intenten sustituir la pérdida multitarea por una
single-target incompatible. No desactivarlas silenciosamente y no restringir
su uso en configuraciones legacy. El plan comprobará las otras optimizaciones
existentes contra este contrato y no afirmará compatibilidad sin probarla.

## 8. Archivos, aceptación y recursos

Reutilizar `configuration.py`/`processing.py`/sparse y la preparación física;
un helper acotado de descriptor/carga puede vivir en `core/data/multitarget.py`,
sin registro genérico extensible. Integrar opt-in en dataset/DataModule/CLI,
readout MatrixMACE y loss/logging Lightning, además de reconstrucción/writer.
El plan fijará las firmas después de aprobar este contrato y antes de editar.

Tests CPU deben probar lectura FDF/bundle controlados, igualdad de grafos y
orbitales, orden H/S/DM/EDM con spin1/2, formas/slices/máscaras heterogéneas,
matriz ausente, NaNs/inf, pesos cero/negativos, física fraccionaria/periódica,
gradientes finitos no nulos por target activo y hacia backbone, suma de pérdidas
independientes, un paso sintético real, checkpoint/save-load, export nombrado y
errores por incompatibilidad. Probar que cambiar etiquetas no cambia features
ni el forward en modo eval. Comparar ruta legacy antes/después, no solo shapes.

Resultado técnico con fixtures no significa entrenamiento científicamente
validado. Medir overhead CPU de preparación y forward/loss/backward pequeño
si se implementa, sin atribuirlo a GPU. No se garantiza mejora de precisión y
predecir matrices adicionales puede aumentar VRAM y coste.

Cerrar implementación solo con gates y evidencia; actualizar roadmap/traspaso
separando implementado, CPU verificado, bloqueado por etiquetas y campaña
pendiente. Proponer después un experimento pareado single-target/multitarea,
con receta/dataset/splits/observables/presupuesto por aprobar; no lanzarlo.
