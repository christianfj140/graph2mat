# Revisión y corrección de herramientas tras Claude — 2026-10-22

## Alcance y resultado

Se revisaron los cambios de herramientas en `2c3b840`, `eca05a8`, `1e12f10`, `c6c4850` y `17e246f`, sus consumidores y tests. La implementación se realizó exclusivamente en la copia compartida `/tmp/graph2mat-codex-review-20261022-f80m50ec/source`, basada en `17e246f318174b8a6c743b6c31ea01c4b740ed4b` y los cambios locales de Claude ya presentes al crearla. No se modificaron las fuentes del checkout original. La integración posterior y la suite global corresponden al agente raíz.

Las correcciones de Claude resuelven spin-down en métricas collineares, la pérdida de imágenes lejanas de H0 y los artefactos dejados por nuevas ejecuciones SIESTA fallidas. Quedaban defectos reproducibles en alineación de imágenes, gauge disperso, casos límite del análisis, includes y bloques FDF, y reutilización de caches incompatibles. Se corrigieron con pruebas RED→GREEN. La nueva ruta SOC funciona con un oráculo Hermítico de ocho componentes reales; también se corrigió su manejo de la representación equivalente de cuatro componentes complejos de sisl.

No se ejecutaron GPU, SIESTA real, entrenamientos ni operaciones de cola. Las ejecuciones de SIESTA son el fixture Python local que copia una matriz sintética. No se modificaron scripts históricos de campañas, defaults de entrenamiento ni `tools/lightning`; X1/X2 y los cambios del writer Lightning son responsabilidad de los otros agentes.

## Dictamen por commit de Claude

| Commit | Dictamen | Comprobación y trabajo adicional |
| --- | --- | --- |
| `2c3b840` | Correcto para métricas collineares y conservación del `nsc` mayor; incompleto para aritmética genérica. | El oráculo spin-down produce MAE **250 meV**, RMSE **500 meV** y error espectral **707.106781 meV**. Se conservan el gauge común y el rechazo explícito de configuraciones no soportadas. T1 seguía presente con `sc_off` permutado y `nsc` igual; ahora se remapean columnas por traslación física. Los tests de H0 lejano y corte de todos los canales siguen pasando. |
| `eca05a8` | Publicación tras éxito y separación de rutas CLI correctas; cache incompleto. | Los intentos fallidos dejan de publicarse. Pero `run_from_dm` seguía retornando cualquier TSHS existente y el wrapper no verificaba entradas ni procedencia. Ahora hay un recibo mínimo de hashes y validación previa al rename en H0/DM. Caches legadas o modificadas generan error y permanecen intactas. |
| `1e12f10` | Sin objeción confirmada en el alcance revisado. | Se trazó la correspondencia descriptor global del paso p−1 → label del paso p, con pasos ausentes. Los tests de `HistoryCopy`, bloques ausentes y extrapolación real incluidos en el gate del servidor pasan. No fue necesario cambiar su implementación. |
| `c6c4850` | Ruta de ocho componentes reales y reconstrucción de S comprobadas; dos fallos de almacenamiento corregidos. | Oráculo SOC independiente Hermítico en Γ y k=(0.23,0,0), con S no trivial, valida proyección TRS y roundtrip `.nc`/`.TSHS` + H0. `combine_hamiltonians` y `project_time_reversal` fallaban al recibir los cuatro componentes complejos equivalentes de sisl. Se usa su conversión de almacenamiento certificada por el mismo oráculo. La proyección rechaza S compleja en vez de descartar su parte imaginaria. |
| `17e246f` | Fixes de cliente y ampliación de supercelda de atribución conservados; FDF incompleto. | `system_label` normalizaba puntuación, pero no resolvía `%include`. Ahora usa el lector FDF. Las reescrituras textuales rechazan referencias externas que no pueden resolver ni copiar con seguridad. `noise_fdf` ya no sigue desplazando bloques numéricos después de `%endblock` sin nombre. `DM.InitSpin` se sustituye con las reglas completas de etiquetas FDF. Los tests del cliente por ruta y la atribución con imágenes adicionales pasan. |

Se leyó además el diff de `2df88f8` (posterior a la copia): codifica valores en las consultas Lua modificadas. No se encontró objeción estática en ese diff. No se ejecutó Lua ni se incorporó ese commit a esta copia. Los commits posteriores del checkout original requieren la integración y validación coordinadas; este informe no los presenta como parte del gate sobre `17e246f`.

## Fallos pendientes de la auditoría original

| ID | Estado anterior en la copia de Claude | Corrección aplicada y ubicación final |
| --- | --- | --- |
| T1 | Mismo `nsc` con `sc_off` distinto seguía sumando imágenes diferentes. | `siesta/delta.py:152`, `:185`: columnas de cada CSR se alinean con las traslaciones físicas de la geometría destino; se conserva el `nsc` máximo. Se rechazan geometrías o conteos orbitales incompatibles y no se mutan entradas. El overlap añadido a una predicción H-only utiliza el mismo remapeo. |
| T2 | Corregido por Claude para el caso collinear demostrado. | Control independiente numérico de ambos canales en `analysis/tests/test_audit_regressions.py`. No se añade semántica SOC a estas métricas. |
| T3 | Nuevos fallos de proceso ya no se publicaban, pero un archivo legado o un cache de otras entradas se reutilizaba. | `siesta/runner.py:152`, `siesta/delta.py:111`, `siesta/dm_pilot.py:45`: recibo de FDF reescrito, pseudos, extra/DM y salida; validación de matriz/geometría/layout conocido antes de publicar y al reutilizar. No se borran caches legadas o incompatibles. |
| T4 | `aligned_error` seguía devolviendo NaN con error solo off-diagonal. | `siesta/dm_pilot.py:70`: gauge con la norma completa de S; RMS sobre la unión de posiciones almacenadas en H, H_ref y S. El caso de dos errores de 0.1 eV sobre cuatro posiciones produce **0.070710678 eV**. Se alinea físicamente el soporte periódico; el piloto rechaza spin no unpolarizado en esta función. |
| T5 | Puntuación corregida; includes sin resolver y reescrituras potencialmente ineficaces. | `siesta/runner.py:44`, `:51`: lectura de etiqueta mediante sisl; rechazo claro de `%include` y `<` en reescrituras sin contexto. `siesta/delta.py:79` elimina el bloque efectivo anterior de `DM.InitSpin` aunque su etiqueta contenga varios separadores. `siesta/label_noise.py:33` respeta `%endblock` sin nombre. |
| T6 | `n_keep=0` seguía seleccionando todos los estados mediante `[-0:]`. | `analysis/metrics.py:277`: `ValueError` explícito para subespacio vacío. Repro con S=I y umbral **2**, sin cambiar el default 0.01. |
| T7 | Agregación de error cero seguía dividiendo entre cero. | `analysis/breakdown.py:129`: participaciones de error **0.0** para suma total cero, manteniendo MAE y conteos correctos. |
| T8 | MIC limitada a ±1 seguía confundiendo enlaces cortos en coordenadas sin envolver o celdas no reducidas. | `analysis/breakdown.py:63`: `ase.geometry.find_mic`, dependencia ya existente. Oráculos: 31 Å en celda 10 Å → **1 Å**; sin PBC → **31 Å**; celda sesgada no reducida con b=(51,1,0) → **√0.5 Å**, sin limitar la búsqueda a las imágenes vecinas. |

Las rutas de la tabla son relativas a `src/graph2mat/tools/` en la copia congelada de cierre. T4 declara el soporte RMS elegido; no se extrapolan los nuevos valores a la campaña 7.7. T6 no se presenta como fallo del umbral habitual. Ninguno de estos contraejemplos se atribuye a resultados históricos sin sus artefactos.

## Cache: contrato y pruebas

El recibo `.graph2mat-siesta.json` contiene exclusivamente hashes de entradas y salida. Un cambio de DM, FDF efectivo, pseudopotencial o bytes de salida impide la reutilización. La ausencia o invalidez del recibo también la impide. Se exige elegir un directorio nuevo; no se elimina el cache previo ni se relanza implícitamente sobre él. La consulta de un cache antiguo no requiere localizar un ejecutable SIESTA antes de rechazarlo.

H0 y DM validan que el archivo sea una matriz legible y finita, su geometría/celda y, cuando son conocidos por el llamador, conteos orbitales y spin. La validación ocurre en el directorio parcial antes del rename. El fallo de lectura de una TSHS corrupta y una salida con átomos desplazados 0.5 Å no dejan directorio final. La variante SOC del cache se prueba con inicialización no magnética y modificación posterior del tipo de spin en FDF.

**Límite de formato:** TSHS no guarda especies ni identidad completa de base. Se comprobó con una matriz de carbono: Z=[6,6] antes de escribir y Z=[1,1] tras leer TSHS. Comparar directamente Z de ese lector contra FDF rechazaría datos válidos. Se conservan los hashes del FDF y los pseudopotenciales, y se documenta que versión de SIESTA, identidad de base y orden orbital siguen siendo requisitos del usuario. El recibo no certifica versión del ejecutable ni reemplaza procedencia científica externa.

`_sha256` procesa bloques de 1 MiB y evita `hashlib.file_digest`, ausente en Python 3.9/3.10 que el proyecto admite. Una prueba elimina esa API del entorno para verificar la ruta compatible. No se ejecutó aquí un intérprete Python 3.9 real.

## SOC y soporte físico comprobado

`siesta/tests/test_audit_soc_physics.py` construye bloques spinor explícitos, sin usar las funciones de conversión de Graph2Mat para generar el oráculo. El acoplamiento es `t I + i(gx σx + gy σy + gz σz)`, con bloque inverso adjunto y traslaciones ±1. S tiene diagonal unidad y acoplamientos no nulos. Se comparan H(k), S(k) y Hermiticidad en Γ y fuera de Γ, antes y después de proyección y serialización.

Se comprueban ocho componentes reales y cuatro complejas, así como la adición de H0 a una predicción sin S para `.nc` y `.TSHS`. El residuo TRS conocido de un pequeño término Zeeman es **1e−4/0.3**, y la proyección reproduce el Hamiltoniano TRS explícito. La proyección sigue siendo una transformación solicitada explícitamente: estos tests no certifican que sea físicamente admisible proyectar datos magnéticos reales ni establecen un umbral nuevo para campañas.

Las fixtures SOC previas de Claude generan avisos de onsite no Hermítico. No se usan esos ejemplos como prueba de validez espectral: el nuevo oráculo Hermítico es la evidencia física independiente. Los cambios al callback `MatrixWriter` de Lightning se revisan y validan en la otra área, sin edición desde este trabajo.

## Tests, RED→GREEN y controles

Tres módulos nuevos:

- `analysis/tests/test_audit_regressions.py`: imágenes permutadas, geometría incompatible, gauge off-diagonal, subespacio vacío, agregación exacta, MIC general y oráculo collinear.
- `siesta/tests/test_audit_cache_fdf.py`: includes, etiquetas/bloques FDF, caches idénticos/modificados/legados, integridad, publicación y hash compatible con Python 3.9.
- `siesta/tests/test_audit_soc_physics.py`: oráculo SOC, almacenamiento real/complejo, S, formatos físicos y cache SOC.

El test existente `server/tests/test_server_app.py:55` se actualiza para esperar **exactamente** `edge_cutoff_padding=0.0` y `magnetic_moments=None`, junto con todo el payload anterior. No se relaja a una comparación parcial: las cuatro variantes del test D1 habían fallado con esos dos campos legítimos adicionales.

| Evidencia en `tools/artefactos/` | Resultado observado |
| --- | --- |
| `red_core.log` | **7 failed, 2 passed**: fallos responsables antes del primer parche. |
| `green_core.log` | **49 passed, 59 warnings**, 0.97 s, con regresiones próximas. |
| `red_cache_fdf_d1.log` | **16 failed, 2 passed** antes de los fixes de cache/FDF/D1. |
| `red_noise_endblock.log` | **1 failed**: el desplazamiento alcanzaba el bloque posterior `DM.InitSpin`. |
| `green_cache_fdf_d1.log` | **49 passed, 56 warnings**, 1.52 s. |
| `red_soc_storage.log` | **2 failed, 4 passed**: representación compleja en proyección y mezcla de almacenamiento. |
| `red_soc_complex_overlap.log` | **1 failed**: ausencia de rechazo explícito para S compleja. |
| `green_soc_storage.log` | **7 passed, 46 warnings**, 0.11 s. |
| `red_hash_python39.log` | **1 failed** al retirar `hashlib.file_digest`, antes de sustituirlo por hashing por bloques. |
| `green_final_focal.log` | **22 passed, 46 warnings**, 0.41 s, después de corregir el hash. |
| `green_legacy_preserved.log` | Prueba específica de conservación byte a byte de cache legado tras el último refuerzo de su assert. |
| `gate_tools.log` | **252 passed, 389 warnings**, 11.06 s: análisis, SIESTA y servidor, incluidos HistoryCopy y serialización física. Ejecutado antes del último cambio de compatibilidad del hash, cubierto después por el gate focal de 22. |
| `gate_spectral.log` | **3 passed, 46 warnings**, 0.03 s: tests conectados de `spectral_targets`. |

Los comandos usan `review-python -m pytest -q -p no:cacheprovider` con las rutas nombradas en cada log. No se ejecuta suite global desde este subtrabajo. `git diff --check` sobre los archivos propios pasa. Los diez archivos Python propios se parsearon correctamente. Ruff no está instalado en el intérprete de revisión ni disponible en PATH; el intento y su error quedan en `ruff.log`, sin instalar dependencias.

## Entrega y límites

Fuentes congeladas: **11 archivos** propios, enumerados con SHA-256 en `tools/artefactos/closure_sources.json`: siete archivos existentes, tres módulos nuevos de tests y `docs/examples/benchmarks/README.md`. Los tres tests nuevos deben incluirse expresamente en la integración porque no pertenecen al índice base de la copia. No se han realizado commits ni cambios de estado del repositorio original.

La documentación explica caches legadas, directorios nuevos, includes no admitidos para reescrituras textuales y soporte RMS. No se extiende el soporte físico de métricas a SOC, del piloto DM a spin no certificado ni de la proyección TRS a sistemas magnéticos. Sin SIESTA real no se certifican el SCF de H0, la compatibilidad de ejecutables/bases ni resultados científicos de campañas; las conclusiones son de corrección de representación y software bajo las compatibilidades indicadas.

**Veredicto:** fixes de Claude parcialmente correctos; los fallos restantes reproducidos de esta área quedan corregidos en la copia y protegidos por tests focales. Integración y suite CPU global final pendientes del agente raíz, incluyendo los commits posteriores de Claude y los cambios de las otras áreas.

## Revisión cruzada posterior del coordinador

Antes de integrar, el coordinador añadió el cambio de base unimodular fuerte de M3 a la prueba de distancias: celda `[[100,1,0],[99,1,0],[0,0,10]]`, separación cartesiana `[0,0.04,0]` y PBC completo. El atajo de `ase.geometry.find_mic` daba una distancia incorrecta. Evidencia: `evidencia/breakdown_mic_red.log`, un fallo y nueve controles pasan.

El cierre final de T8 usa `general_find_mic`, evitando ese atajo; sin PBC conserva directamente la norma cartesiana. El gate de análisis da **26 passed** en `evidencia/breakdown_mic_green.log`. Los hashes de entrega del subtrabajo anterior describen su snapshot anterior a este refuerzo. El manifiesto de integración del coordinador contiene la versión final. La suite completa se registra en el informe principal de esta revisión.
