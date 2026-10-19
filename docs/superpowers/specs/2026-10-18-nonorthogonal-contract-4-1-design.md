# Diseño 4.1: certificar el contrato no ortogonal existente

Fecha: 2026-10-18. Graph2Mat, rama `g2m-v2`, checkout actual.
Estado: diseño aprobado explícitamente por el usuario («Apruebo», 2026-10-18),
referido a este documento. Cierre técnico2026-10-18: pruebas/documentación,
sin cambios productivos; evidencia en el informe de reconstrucción4.1.
Pertenece al Goal 4.1 + 7.6 solicitado por el usuario. Este documento delimita
solo 4.1; el diseño de 7.6 se presentará después de resolver este contrato.

## 1. Intención y condición de éxito

Certificar qué conserva Graph2Mat al pasar de un Hamiltoniano no ortogonal
a sus etiquetas y de estas a una matriz y un archivo físico. El éxito no
requiere nueva arquitectura: exige una especificación comprobable y pruebas
independientes de los observables `H(k)c = epsilon S(k)c`.

El usuario autorizó CPU, fixtures y cambios mínimos después de aprobar el
diseño. No autorizó GPU, entrenamientos científicos, dependencias, commits,
worktrees, cambios de defaults o datos ni reabrir campañas anteriores.

## 2. Evidencia actual y carencia delimitada

En HEAD `270d17d1f204293a059baa09e8f1b3756981e2bf`, con cambios existentes
preservados, ya hay:

- Lectura explícita `matrix_component_policy="h_and_overlap"` en
  `core/data/configuration.py` y reconstrucción en `core/data/sparse.py`:
  `(H,S)` no polarizada o `(H_up,H_down,S)` colineal.
- Recorrido NumPy/Torch y slicing de batches en `core/data/processing.py`.
- Oráculos periódicos independientes para campos reales s/p/d/f y Bloch sums
  en `core/data/tests/test_periodic_hermiticity_oracles.py`.
- Serialización de H/S y metadatos H-only en `MatrixWriter`.
- Contrato HTTP/NetCDF/TSHS, gauge y escritura atómica en
  `tools/server/_serialization.py`, documentado en `docs/server.rst`.
- Restauración de referencia y pruebas de no duplicar S en
  `tools/server/extrapolation.py` y `tests/test_physical_serialization.py`.

La carencia demostrada es de certificación integrada: los oráculos periódicos
comparan componentes y H(k)/S(k), pero no cierran por sí solos el contrato de
espectros generalizados antes/después del recorrido y los writers.
No se ha demostrado que falte un lector, escritor ni una API pública nueva.

Baseline CPU ejecutado: 52 pruebas correctas; un módulo omitido porque FastAPI
no está instalado en el intérprete usado. No se presenta HTTP como verificado
en este arranque ni se infiere una regresión a partir de esa ausencia.

## 3. Enfoque recomendado y alternativas

Recomendación: ampliar los oráculos vecinos y la documentación existente,
sin cambiar código productivo salvo una carencia reproducida por una prueba.
Si aparece un fallo de contrato, documentar causa y corrección mínima; si
requiere nueva API, ampliar el diseño y solicitar aprobación antes de editar.

Alternativas descartadas para este alcance: otro lector H/S, inyección implícita
de S de referencia, ortogonalización correctora, recorte de S, regularizador
de positividad o cambios de arquitectura. Ninguna se necesita para certificar
la representación actual.

## 4. Contrato propuesto

### 4.1 Representación y soporte

Para matrices reales no polarizadas o colineales y componentes semánticamente
declaradas, conservar especies/tipos, geometría, orden de orbitales y canales,
celda, imágenes y cada coeficiente representable en el grafo. Probar el orden
mediante campos independientes distintos por átomo, orbital, canal e imagen.

El contrato es sobre el soporte representado por el grafo, no sobre bloques
físicos fuera del cutoff. Las fixtures deben caber enteramente en ese soporte;
no ampliar radios ni inventar coeficientes para mejorar una reconstrucción.
La ruta simétrica usa los partners transpuestos de imágenes opuestas. La ruta
dirigida no adquiere una simetrización silenciosa. No mutar datos de entrada.

H y sus niveles se expresan en eV; S es adimensional. Las conversiones de
unidades pertenecen a los lectores/escritores, no a una reescala nueva aquí.

### 4.2 S de referencia frente a S predicha

Reconstruir etiquetas H+S conserva la S etiquetada. Reconstruir predicciones
H+S usa la S suministrada por esas predicciones. H-only sigue siendo H-only:
tener S en las etiquetas de una pérdida no adjunta S automáticamente al output.
El servidor sigue rechazando H-only como archivo físico no ortogonal salvo
una ruta existente que restaure explícitamente una referencia compatible.
`orthogonal_basis=True` es declaración del caller, no una reparación de S.

Estas pruebas certifican conservación, no precisión ni positividad universal
de S aprendida. El writer actual comprueba finitud y diagonal positiva, no
definición positiva global. No reforzar ese guard ni cambiar su coste en 4.1.

### 4.3 H directa y referencias residuales

El procesador genérico no construye H0 de SIESTA ni suma H0 automáticamente;
`sub_point_matrix` no es ese H0 y está desactivado para Hamiltonianos.
En las rutas de extrapolación existentes con `m_0`, certificar por separado:
H-only delta restaura H de referencia y su S explícita compatible; H+S delta
restaura una vez ambas contribuciones. No interpretar S como canal de H.
La referencia representa matrices dadas por el caller: esta certificación no
valida que una referencia sea físicamente el H0 de densidades superpuestas.

### 4.4 Archivos, energía y espectros

Certificar NetCDF del servidor con Ef explícito: sisl devuelve H-Ef*S al leer;
comparar en ese gauge y verificar el desplazamiento uniforme de niveles.
TSHS del servidor solo se certifica con Ef=0, preservando el rechazo existente
de Ef distinto de cero. Probar HSX en la ruta MatrixWriter existente sin
atribuirle guards que solo posee el servidor. Documentar requisitos de geometría
y base que el lector del formato necesite para preservar el orden orbital.

No ampliar formatos ni prometer soporte SOC/no colineal físico: los campos raw
pueden conservar componentes numéricas, pero no autorizan inferir su semántica.

## 5. Aceptación verificable

1. Extender el oráculo periódico existente: para Γ y dos k no nulos, comparar
   los niveles del objeto reconstruido con el problema generalizado construido
   de los campos analíticos independientes. Cubrir ambos modos de simetría y
   spin no polarizado/colineal, con bases heterogéneas y todos los orbitales.
2. Añadir un caso analítico de dos orbitales con S no diagonal positiva definida:
   niveles exactos conocidos y transformación por Cholesky como contraste.
   Comprobar H+cS: todos los niveles cambian c, diferencias y gap no cambian.
   No comparar autovectores individuales en subespacios degenerados.
3. Contrastar matrices y niveles antes/después de los writers certificados
   y de la restauración explícita de referencia existente. Reutilizar fixtures
   y tests HTTP, con funciones de predicción deterministas, sin entrenar modelos.
4. Incluir ruta Torch/batch con tamaños diferentes para evitar contaminación
   entre estructuras; verificar que no se modifican etiquetas ni referencias.
5. Fijar tolerancias por representación: en memoria float64, inicialmente
   rtol=1e-10/atol=1e-10 eV para espectros bien condicionados, con tolerancias
   matriciales de los oráculos existentes. Para archivos, justificar por
   precisión del formato, no adaptar tolerancias al error observado. El plan
   fijará los valores de cada writer antes del RED→GREEN.
6. Si S no es positiva definida en un caso negativo, el control espectral
   debe fallar sin regularizarla ni eliminar direcciones. No es un requisito
   nuevo de rechazo del writer ni un test de exactitud del modelo.

## 6. Archivos afectados y compatibilidad

Tests previstos: los oráculos periódicos del núcleo, tests de componentes
Lightning y serialización física del servidor. Documentación: secciones de
`docs/data_loading.rst` y `docs/server.rst`, roadmap y traspaso. Reutilizar helpers
locales; no crear un nuevo framework de diagnóstico ni nuevas exports/configs.

Callers productivos, defaults, losses, checkpoints y datasets deben permanecer
sin cambios si las pruebas nuevas confirman el contrato actual. Una corrección
aprobada deberá preservar esas interfaces y pasar sus regresiones relevantes.

## 7. Entorno, cierre y enlace con 7.6

Comando baseline, desde el repositorio y sin GPU:

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python -m pytest \
 src/graph2mat/core/data/tests/test_spin_collinear.py \
 src/graph2mat/core/data/tests/test_periodic_hermiticity_oracles.py \
 src/graph2mat/tools/lightning/tests/test_matrix_components.py \
 src/graph2mat/tools/server/tests/test_physical_serialization.py \
 -q -rs --tb=short --disable-warnings
```

Resultado inicial: 52 passed, 1 skipped (módulo FastAPI), 52 warnings, 0.66 s
de pytest; no equivale al tiempo total de arranque del proceso.

Actualización de entorno tras aprobación, 2026-10-18: los extras ya existen
en `/tmp/g2m-server-tests.IcPjmO`, `/tmp/g2m-http-runtime.LOcBGk` y
`/tmp/g2m-soap-deps.o0Kjat`. Su uso mediante PYTHONPATH, conservando el mismo
intérprete y checkout, permite ejecutar las cuatro rutas del comando anterior:
**125 passed, 53 warnings, 3.86 s**, sin omisiones ni instalaciones. La primera
ejecución con solo HTTP dio 117 passed/8 failed; diagnóstico con las excepciones
del TestClient visibles identificó exclusivamente la falta de `dscribe` en el
path. Añadir su directorio preexistente resuelve esos ocho casos sin editar
producto. No confundir el primer resultado con una regresión H/S.

El plan verificará si ya existe un entorno compatible con los extras HTTP,
sin instalaciones ni acceso a otros proyectos. Si falta, pedir al usuario ese
recurso y dejar explícitamente pendiente la certificación HTTP, sin dar 4.1
por completo. Los tests del núcleo pueden avanzar independientemente.

Cerrar 4.1 requiere todos los observables aprobados certificados y documentación
de límites, no una mejora de entrenamiento. Después presentar el diseño 7.6:
inventario de etiquetas accesibles dentro del alcance, targets/pesos/pérdidas,
compatibilidad single-target y pruebas CPU. Los inventarios históricos que
mencionan datos de otro repositorio no autorizan leerlos en este Goal.
