# Auditoría de herramientas, servidor y scripts de campaña — 2026-10-21

## Alcance y método

Revisión de HEAD `a66b46cb0955ca344693d50737c5e9952fc257fc`, con atribución al rango de 267 commits `origin/g2m-v2..HEAD`. Se leyeron las herramientas SIESTA/analysis migradas y sus tests, CLI, serialización física, servidor, cliente y flujos de extrapolación modificados. La revisión adicional de scripts experimentales se concentró en presupuestos, selección/procedencia de checkpoints, soporte periódico, recuperación y publicación de resultados; no constituye una reejecución de campañas.

Las reproducciones usan matrices sintéticas pequeñas en CPU. La copia aislada contiene el código de HEAD y ambas extensiones Cython compiladas desde ese código. No se ejecutaron SIESTA, HTTP externo, GPU, entrenamientos ni modificaciones de colas. El proceso que simula un fallo de SIESTA es un script local de Python que copia una matriz sintética y termina con código 7.

Reproducciones iniciales: `/tmp/graph2mat-codex-audit-20261021-1n6tqn2d/tools/reproduce_tools.py` y `reproduce_updates.py`. Resultados conservados en `../evidencia/reproduce_tools.log` y `../evidencia/reproduce_updates.log`. Los scripts finalizados se incorporan al paquete de evidencia de la auditoría. Un código de salida cero del reproducer indica que se han observado los contraejemplos esperados; no significa ausencia de bugs.

## Hallazgos confirmados

### T1 — P1 — La aritmética de ΔH suma imágenes distintas cuando `sc_off` está permutado

**Ubicación:** `src/graph2mat/tools/siesta/delta.py:123` y `:127`. **Origen:** `a6f22224`, dentro de los 267 commits.

`combine_hamiltonians` compara `nsc` y spin, pero suma directamente columnas CSR sin comprobar la identidad física de las imágenes. Dos Hamiltonianos Hermíticos idénticos, con dos átomos y `nsc=[3,1,1]`, pueden almacenar ±1 en distinto orden. En ese caso `a-b` debe ser cero. A `k=[0.23,0,0]`, el método devuelve una matriz con módulo máximo **3.968458805 eV**, mientras `a.Hk-b.Hk` es exactamente cero. La matriz incorrecta sigue siendo Hermítica: comprobar solo Hermiticidad o Γ no detecta el error.

La API acepta objetos sisl directamente. La reproducción no afirma que todos los lectores TSHS produzcan ese orden ni que las campañas previas lo hayan usado. El defecto está en el contrato público de aritmética y afecta combinaciones válidas de objetos con imágenes no estándar.

**Propuesta mínima:** alinear ambos patrones por desplazamientos `sc_off`, o rechazar layouts diferentes antes de operar. Comprobar también geometría y orden orbital; el repro actual demuestra únicamente el desacuerdo entre imágenes.

### T2 — P1 — Las métricas públicas pueden declarar perfecta una predicción con spin-down incorrecto

**Ubicación:** `src/graph2mat/tools/analysis/metrics.py:56`, `:65` y `:81`. **Origen:** `041e0ad7`, dentro del rango.

Las llamadas `Hk/eigh` sin selector de spin observan solo el primer canal collinear. No hay rechazo de matrices polarizadas ni opción para pedir/combinar ambos canales. En una H de dos orbitales con `S=I`, la referencia tiene niveles `[-0.5,0.5]` para ambos spins. La predicción conserva spin-up y cambia spin-down a `[-1.5,1.5]`. `gamma_metrics` devuelve **H_MAE=0, H_RMSE=0, error espectral=0 y rel_Frob=0**, aunque spin-down difiere en **1 eV** por nivel.

No se afirma que las recetas spin1 usadas para el roadmap estén afectadas. El problema aparece al aplicar la nueva API de métricas a matrices polarizadas admitidas por la rama. Los docstrings no restringen estas funciones a spin1 ni declaran que miden solo spin-up.

**Propuesta mínima:** rechazar explícitamente spin polarizado/SOC mientras no exista semántica certificada, o exponer selector y agregación física de todos los canales. La corrección de la pérdida gauge M1 es independiente: arreglar esa pérdida no hace que estas métricas observen spin-down.

### T3 — P2 — El cache de H0 acepta el archivo de un proceso fallido y evita comprobar geometría

**Ubicación:** `src/graph2mat/tools/siesta/delta.py:89`. **Origen:** `a6f22224`, dentro del rango. `run_from_dm`, en `tools/siesta/dm_pilot.py:52`, repite la política de reutilizar por mera existencia; la reproducción se ejecutó sobre `compute_h0`.

Un proceso puede escribir TSHS y luego fallar. La primera llamada propaga correctamente `CalledProcessError`, pero deja el archivo. La segunda retorna ese archivo antes de comprobar éxito, integridad o geometría. El repro simula salida 7 después de escribir una TSHS válida con un átomo desplazado **0.5 Å**. La segunda llamada la acepta pese a `geometry_tol=0.0001 Å` y sin ejecutar el proceso de nuevo.

La reutilización de resultados terminados es comportamiento documentado; el hallazgo es certificar implícitamente como resultado reutilizable un artefacto dejado por un intento fallido, saltándose una validación prometida por la misma API. No requiere modificar entradas entre llamadas.

**Propuesta mínima:** publicar el cache solo tras finalizar y validar, mediante recibo o renombrado de un resultado temporal; al leer cache, validar al menos su integridad y geometría. Mantener el fallo anterior como fallo y no reutilizar su salida como resultado terminado.

### T4 — P2 — El error alineado del piloto DM devuelve NaN para un error exclusivamente off-diagonal

**Ubicación:** `src/graph2mat/tools/siesta/dm_pilot.py:72`. **Origen:** `ba087112`, dentro del rango.

El denominador del gauge usa S solo en posiciones donde `H-Href` es no nulo. Para una H no ortogonal de sisl con `S=I`, diagonales exactas y hoppings con error `0.1 eV`, todas esas posiciones tienen `Sij=0`: se calcula `0/0`. `aligned_error` devuelve **NaN** aunque la norma de S es positiva y el error tiene un valor finito. El óptimo físico es `c=0`; su RMS es `0.1 eV` sobre los dos errores no nulos, o `0.070710678 eV` si se usa el patrón completo de cuatro posiciones. La elección del denominador RMS debe declararse; ninguna da NaN.

**Propuesta mínima:** calcular el gauge sobre un soporte fijo que incluya S y su norma completa, no solo posiciones de diferencia no nula. Declarar también el soporte del RMS para no cambiar el denominador según dónde el error sea exactamente cero. Este repro no revalida ni invalida cuantitativamente el piloto histórico 7.7.

### T5 — P2 — El parser de `SystemLabel` no sigue la semántica FDF del resto del runner

**Ubicación:** `src/graph2mat/tools/siesta/runner.py:42`. **Origen:** `d28c65ba`, dentro del rango.

Con el FDF válido `System.Label sample`, el lector sisl instalado devuelve `sample` al pedir `SystemLabel`; el runner devuelve **`siesta`**. La normalización de puntos, guiones y subrayados ya está implementada en `_fdf_label`, pero no se usa aquí. Tampoco se resuelven includes. H0, ruido, piloto DM y staging buscan entonces un archivo con nombre distinto al escrito por SIESTA.

**Propuesta mínima:** leer `SystemLabel` con el parser FDF disponible, igual que las otras propiedades físicas. Usar `siesta` solo si la propiedad está ausente. El fallo de `NetCharge` en includes se documenta por separado como DATA-04 porque altera directamente la carga impuesta a QEq.

### T6 — P2 — La ortogonalización canónica retiene todos los estados cuando ninguno supera el umbral

**Ubicación:** `src/graph2mat/tools/analysis/metrics.py:228` y `:231`. **Origen:** `041e0ad7`, dentro del rango; la misma expresión existe en evaluadores SOC/magnéticos experimentales.

Si `n_keep=0`, `argsort(w)[-n_keep:]` equivale a `[0:]`, no a una selección vacía. Para `S=I` y umbral explícito `2.0`, la función devuelve las dos bandas completas aunque ninguna dirección supera el umbral. Si S tiene direcciones negativas, puede además intentar normalizarlas con raíz de un número negativo.

Este caso no aparece con el umbral habitual `0.01` y una S normalizada con diagonal unidad; no se extrapola a esas campañas. Es un fallo del parámetro público y de la política de exclusión, no evidencia de un problema en todas las evaluaciones.

**Propuesta mínima:** comprobar `n_keep>0` y rechazar la proyección vacía con un error explícito. No cambiar silenciosamente de criterio ni conservar todas las direcciones descartadas.

### T7 — P2 — El desglose agregado falla para una predicción exacta

**Ubicación:** `src/graph2mat/tools/analysis/breakdown.py:131`. **Origen:** `041e0ad7`, dentro del rango.

`summarize_breakdown` divide la suma de cada categoría por el error total sin comprobar si es cero. Una fila válida con cuatro elementos observados y error alineado cero produce **`ZeroDivisionError: float division by zero`**. Una predicción exacta o un cambio puro de gauge son entradas normales de diagnóstico.

**Propuesta mínima:** definir explícitamente shares nulos o indeterminados cuando el error total es cero, preservando MAE cero y conteos. La lista vacía necesita su propio contrato, pero no forma parte del repro contado.

### X1 — P2 — Los runners SOC/magnético registran updates solicitados que no ejecutan

**Ubicación:** `auditorias_mejoras/run_soc_campaign.py:257` y `:284`; `auditorias_mejoras/run_magnetic_campaign.py:366` y `:394`. **Origen:** `88c344e` y `d9b5904`, dentro del rango.

Ambos convierten el presupuesto en épocas con división entera y escriben después el número solicitado como `updates`. Se ejecutan sus cuerpos `run()` originales mediante AST, sustituyendo únicamente dependencias de entrenamiento/I/O por dobles sintéticos: diez muestras, batch 2, solicitud 7 updates. Configuran **una época = cinco updates**, pero el resultado registra **siete**. Si el presupuesto es menor que una época pueden excederlo por `max(1,...)`. No se ejecuta ni modifica código del repositorio MD_vs.

El repro no afirma que el presupuesto de 8000 updates de una campaña concreta se haya incumplido: depende de la divisibilidad y la muestra efectiva de esa campaña. Demuestra que `--updates N` y la procedencia guardada no son fiables para presupuestos generales.

**Propuesta mínima:** usar `max_steps` o contar/guardar el `global_step` realmente completado, con presupuesto solicitado y efectivo separados. Evitar etiquetar la solicitud como coste observado.

## Cobertura, controles positivos y límites

- Lectura completa de herramientas migradas `tools/siesta/{runner,delta,label_noise,dm_pilot}.py`, `tools/spectral_targets.py`, `tools/analysis/{metrics,breakdown,attribution,kb_tail}.py`, CLI delta y fixtures/tests conectados. El conflicto floor/gauge espectral está en M2; no se duplica aquí.
- Lectura de cambios y flujos conectados en `tools/server/{server_app,extrapolation,_serialization,api_client,__init__}.py`, CLI/request y plantillas Lua modificadas. Se revisaron la publicación atómica sin overwrite, metadatos serializados, preservación de S/canales y rechazo explícito de HTTP para multitarget/SOC/momentos. No se confirmó otro defecto del servidor en esta revisión. No se hizo prueba de carga ni auditoría de una exposición de red pública.
- Se comprobó una sospecha sobre `ServerClient.predict` con ruta de geometría y salida DM: **descartada en el entorno actual**, porque objeto y ruta devolvieron las coordenadas correctas. El control positivo permanece en el log; no se cuenta como hallazgo.
- En scripts históricos se inspeccionaron especialmente `training_recovery.py` (fronteras, hashes, generaciones, selección y RNG), `recovery_certification.py`, `run_readout_comparison.py` (append y publicación), `run_bilayer_edge_updates.py`, `run_cost_batching_comparison.py`, `run_h2oclu_batching_science.py`, `h2oclu_batching_metrics.py`, `run_h2oclu_nullspace.py`, `pilot_multitarget_w90.py`, `run_multitarget_w90_six.py`, `close_multitarget_w90.py` y runners SOC/magnético. La profundidad varía: se trazaron contratos críticos, no todas las líneas de cada benchmark.
- El protocolo de recuperación tiene restricciones deliberadas a un dispositivo, Adam/cosine y épocas completas. No se cuentan como bugs combinaciones que rechaza expresamente. No se relanzaron certificaciones, preparaciones, pilotos, fits ni colas ya cerrados.
- La revisión de evidencia científica es documental y de fórmulas. No se atribuyen los contraejemplos sintéticos a cifras concretas publicadas en el roadmap sin inspeccionar los artefactos de esas campañas.

**Veredicto:** corregir T1/T2 antes de considerar certificadas aritmética periódica genérica y métricas collineares. Resolver el resto antes de usar las combinaciones que activan sus fallos. No se ha aplicado ningún arreglo.
