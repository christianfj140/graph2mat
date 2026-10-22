# Revisión independiente de los arreglos de Claude — 22 de octubre de 2026

## Dictamen

Los arreglos de Claude solucionan numerosos casos reales. Su cierre requería correcciones adicionales: el control de base modelo/datos era incompleto; las cachés todavía podían reutilizar otras entradas; la nueva ruta SOC fallaba con almacenamiento complejo equivalente; y el arreglo de evaluación de etapas aplicaba la pérdida de una etapa futura a checkpoints tempranos.

La revisión cubre **32 commits**, desde `a66b46c` hasta `e1e7e4b9f4c516a5e9011b2273a4f6a7e829ac27`, y los ocho cambios locales de Claude presentes al reanudar. Estos últimos terminaron incorporados en sus commits posteriores. [Dictamen por commit](commits_revisados.md). La auditoría original de 267 commits permanece separada: describe el estado anterior a estas correcciones.

Los parches propios resuelven los 16 hallazgos originales que seguían abiertos o tenían una corrección incompleta. Los otros nueve quedan resueltos por Claude en las reproducciones verificadas. Esto cierra los **25 contraejemplos confirmados**; no demuestra ausencia de otros bugs ni certifica campañas científicas completas.

**Entrega terminada:** 31 archivos aplicados al checkout original tras verificar la suite y la integridad de la fuente. HEAD de Claude conservado, sin commits ni push propios. Las fuentes aplicadas coinciden byte a byte con las probadas. Parche completo: [`evidencia/correcciones_codex.patch`](evidencia/correcciones_codex.patch); comprobación: [`integration_verified.json`](evidencia/integration_verified.json).

## Estado de los hallazgos originales

| IDs | Cierre verificado |
|---|---|
| DATA-01, DATA-02, DATA-04 | Arreglos de Claude: baseline multicomponente, rechazo de `nsc` incompatible y lectura FDF de carga. |
| DATA-03, DATA-05, DATA-06 | Correcciones propias: comparar capas angulares/orden, preservar valores complejos y respetar `load_labels=False` sin modificar la muestra fuente. |
| M1, M3 | Correcciones propias: un gauge común para ambos spins; imagen mínima mediante reducción de celda. |
| M2 | Arreglo de Claude: gauge espectral con `G=CᵀSC` real. |
| E1, E2, E3 | Arreglos de Claude: enumeración Ewald completa, potencial común en todos los H y gradientes del readout con MACE congelado. |
| L1 | Corrección propia sobre el arreglo de Claude: recuperar la etapa guardada, sin congelar parámetros antes del registro DDP. |
| L2 | Arreglo de Claude verificado con dos rangos CPU: un CSV por rango. |
| L4 | Corrección propia: rechazar una métrica de selección que ya no se actualiza en la validación actual. |
| T1, T3, T4, T5 | Correcciones propias: remapear imágenes físicas, verificar cachés y su salida antes de publicar, gauge con toda S y reglas FDF coherentes. T3/T5 completan arreglos parciales de Claude. |
| T2 | Arreglo de Claude: diagnósticos colineales observan ambos spins. |
| T6, T7, T8 | Correcciones propias: rechazar subespacio vacío, admitir predicción exacta y calcular distancias periódicas generales. |
| X1, X2 | Correcciones propias: presupuesto exacto de pasos con checkpoint disponible y guardia de temperatura/VRAM también en el primer lanzamiento. |

D1 también queda corregido: las cuatro expectativas JSON conservan la comparación estricta e incorporan los dos campos legítimos añadidos al procesador.

## Comprobaciones adicionales y correcciones de interpretación

- Se completa el acuerdo entre las bases de modelo y datos: tipos iguales no bastan para distinguir `3s` de `p`, otro orden de capas o otra convención.
- La suma de Hamiltonianos y la proyección TRS admiten las representaciones equivalentes de sisl con ocho componentes reales o cuatro complejas. Un oráculo spinor Hermítico independiente comprueba H(k), S(k), serialización y reconstrucción con H0. Se rechaza S compleja cuando la proyección necesita S real.
- Se corrige el cierre de bloques FDF sin nombre y la sustitución completa de `DM.InitSpin`. Las reescrituras textuales rechazan includes que no pueden resolver ni trasladar; el lector de `SystemLabel` sí los resuelve.
- Se acota la afirmación «todos los entrenamientos históricos usaron LR constante»: solo la avalan los checkpoints CLI inspeccionados. Los runners 6.4/4.7 y 7.6 administraban su scheduler explícitamente. No se alteran cifras históricas ni se repite la sonda GPU.

La revisión cruzada también detectó dos insuficiencias de los primeros parches propios. Se corrigieron antes de la entrega: `max_steps` corto podía impedir toda validación/checkpoint, y el atajo público MIC de ASE podía fallar todavía en una celda muy sesgada. Quedan protegidas por Trainer CPU real y por el mismo cambio unimodular usado en el contraejemplo de métricas, respectivamente.

## Evidencia y alcance

La suite completa `pytest src tests -q -ra` sobre la fuente integrada produjo **3599 passed, 25 skipped, 0 failed**, en **971.28 s**, con 32089 advertencias registradas. [Log completo](evidencia/pytest_integrated.log) y [JUnit](evidencia/pytest_integrated.xml). No se suman los gates focales como si fueran ejecuciones disjuntas. Las omisiones incluyen GPU/supervisor, cuEq ausente, fixtures SIESTA reales no distribuidos y combinaciones expresamente no soportadas.

Entorno: Python 3.12, PyTorch `2.11.0+cpu`, Lightning `2.6.1`, sisl `0.16.4`, ASE `3.29.0`, e3nn `0.4.4` y MACE `0.3.15`; perfil completo con FastAPI/dscribe. [Versiones, imports y hashes de extensiones](evidencia/integrated_environment.json). Se comprobaron 429 entradas de fuente/configuración/documentación sin alteraciones durante el gate. Los cinco enlaces absolutos históricos se preservan en el checkout y no son parte de las pruebas portables.

[Datos/modelos/e3nn](data_models.md), [Lightning y métricas](lightning_metrics.md), [herramientas](tools.md) y [scripts/documentación](experimentos_documentacion.md) registran los fallos antes del parche, controles y límites. Los artefactos de área se conservan bajo `evidencia/areas/`; los logs generales y manifiestos de integración están en `evidencia/`.

Se trabajó en copias aisladas para preservar el trabajo concurrente de Claude. El merge usa tres versiones: fundación `17e246f` más sus ocho cambios locales, parches propios y HEAD final de Claude. Se conserva su eliminación de la inferencia arbitraria de aristas, las guardias DDP/scaleshift y sus pruebas nuevas. Solo se sustituyen los hooks y tests que imponían incorrectamente la última etapa. Los cinco symlinks históricos excluidos de las copias no se eliminan del checkout.

Ruff sobre `src tests` y `git diff --check` pasan. Mypy presenta **los mismos ocho errores de la fundación**, sin diagnósticos nuevos normalizando números de línea. No se ocultan ni se afirma que ese gate sea verde. [Resultados de calidad](evidencia/quality_results.json), [Mypy anterior](evidencia/mypy_baseline.log) y [actual](evidencia/mypy_final.log). Las extensiones Cython usadas para validar se compilaron desde la fuente integrada, con hashes y rutas registrados; no se copiaron binarios a Git ni se modificaron instalaciones compartidas.

Las cachés legadas sin recibo o con entradas/salida distintas se rechazan y permanecen intactas: requieren un directorio nuevo. TSHS no conserva identidad de especies/base completa; los hashes de FDF/pseudos y controles disponibles no sustituyen procedencia científica. El CSV DDP requiere reunir los archivos por rango. Lua recibió revisión estática; su runtime no se certifica aquí. Las combinaciones GPU/Inductor/cuEq y los resultados científicos históricos conservan sus límites previos.

No se ejecutaron GPU, SCF, campañas científicas ni cambios de colas. Los entrenamientos de pruebas son ejemplos sintéticos CPU. No hay commits ni push propios. Tras cambiar `_sparse.py`, una instalación editable con extensión previamente compilada necesita recompilarse para usar esa corrección; esta revisión conserva intacta la instalación compartida de MD_vs.
