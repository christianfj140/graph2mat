# Certificación de reconstrucción no ortogonal — 4.1

Fecha: 2026-10-18. Repositorio Graph2Mat, rama `g2m-v2`, mismo checkout.
HEAD: `270d17d1f204293a059baa09e8f1b3756981e2bf`, con cambios previos conservados.
Estado: **4.1 cerrado para el contrato aprobado**, con controles, regresiones
CPU y revisión independiente verificados. 7.6 permanece pendiente de aprobación.

## Alcance aprobado y cambio efectivo

El usuario aprobó diseño y plan de 4.1. Se certifica el contrato H+S existente,
no una representación nueva, una mejora de entrenamiento ni solución OOD.
Cambios efectivos: cuatro archivos de pruebas y documentación. No se cambió
código productivo, defaults, pérdidas, checkpoints ni datos. Sin GPU, fits,
autorun/cola, instalaciones, commits o worktrees. 7.6 no se implementa por
inferir aprobación de este bloque.

## Requisitos y evidencia directa

| Requisito | Prueba/observable |
| --- | --- |
| Especies, orbitales y componentes periódicos | `test_periodic_components_and_bloch_hermiticity_against_independent_fields`: campos independientes H/O, bases s/p y s/d/f, imágenes 0 y ±1, simétrica/dirigida, igualdad de coeficientes |
| Niveles no ortogonales sin recorte | El mismo oráculo: todos los niveles en Γ, kx=0.17 y -0.31 contra `scipy.linalg.eigh` sobre Bloch sums independientes, spin no polarizado/colineal |
| Espectro exacto y gauge | `test_nonorthogonal_analytic_levels_and_gauge_shift`: H=[[1.375,-0.125],[-0.125,1.375]], S=[[1,0.25],[0.25,1]], niveles [1,2] eV, contraste Cholesky y H+0.7S → [1.7,2.7]; NumPy/Torch, simétrica/dirigida, no mutación |
| Controles que detectan un S incorrecto | `test_nonorthogonal_controls_detect_wrong_or_singular_overlap`: S=I no reproduce [1,2]; S singular conserva coeficientes y falla al resolver, sin eliminar direcciones |
| Predicciones S distintas de la etiqueta y batches | `test_h_and_overlap_batch_predictions_keep_distinct_structures`: PyG real con dos/tres átomos, S predicha distinta por grafo, salidas tuple/generator y espectros contra arrays previos, spin 1/2 y ambos modos de simetría, entradas intactas |
| Gauge de NetCDF | `test_http_prediction_roundtrips_real_matrix_spin_overlap_and_fermi`: subida/local, spin 1/2, matrices H−0.5S y todos los niveles desplazados exactamente −0.5 eV |
| TSHS no ortogonal | `test_http_nonorthogonal_tshs_zero_fermi_preserves_generalized_levels`: subida/local, spin 1/2, H/S y niveles conservados con Ef=0; el guard existente rechaza Ef distinto de cero |
| Callback real HSX | `test_matrix_writer_h_and_overlap_hsx_preserves_generalized_levels`: MatrixWriter con batch/predicciones reales, S no diagonal, spin 1/2, lectura con geometría/base explícitas; niveles [1,2] y [1.5,2.5] eV; sin mutación |
| Referencias residuales | Pruebas existentes `test_extrapolated_h_only_delta_restores_explicit_reference_overlap` y `test_h_overlap_delta_and_reference_recover_s_without_doubling_it` ampliadas con espectros; esta última usa delta-S no nula y detecta S duplicada |
| H-only, raw y formatos no compatibles | Regresiones existentes de componentes, metadatos H-only, rechazo raw físico y guardas HTTP de Ef/formato/no overwrite, sin cambiar sus contratos |

Los espectros se comparan como conjuntos ordenados completos, no mediante
autovectores arbitrarios de subespacios degenerados. Las pruebas periódicas
certifican reconstrucción en memoria; los casos de formato nuevos son
moleculares. No extrapolar automáticamente la cobertura cruzada a todos los
formatos, bases y k imaginables.

## Tolerancias y semántica física

En campos analíticos se conserva igualdad exacta donde ya se exigía; H(k)/S(k)
atol=2e-14. Niveles y nuevos round-trips float64: rtol=1e-10, atol=1e-10 eV
(S adimensional). Valores fijados en el plan, no ajustados al resultado.
HSX del sisl instalado escribe versión2/dtype=float64 por defecto; no se asumió
un formato float32 ni se relajó su tolerancia. Torch recibe coeficientes
diádicos exactos en FP32 y efectúa el control de gauge/predicciones en FP64.

H/S reconstruidas cubren el soporte del grafo, no coeficientes más allá del
cutoff. S de etiquetas no se adjunta a una predicción H-only; H+S predicha
usa su propia S. El writer verifica precondiciones, no positividad definida
global de una S aprendida. No se introduce cutoff de S, regularización,
simetrización correctora ni nueva pérdida.

El procesador no construye H0 de densidades atómicas superpuestas. El baseline
aislado `sub_point_matrix` no es ese H0 y se desactiva para Hamiltonianos.
En extrapolación se certifica restaurar una matriz explícita compatible una
vez, sin afirmar que su contenido sea físicamente ese H0.

## Comandos reproducibles y resultados

Usar, desde este repositorio, el intérprete preexistente
`/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python`.
Solo se utiliza como herramienta; no se trabaja en ese otro proyecto.

```bash
env CUDA_VISIBLE_DEVICES= \
 PYTHONPATH=/tmp/g2m-server-tests.IcPjmO:/tmp/g2m-http-runtime.LOcBGk:/tmp/g2m-soap-deps.o0Kjat \
 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 /home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python -m pytest \
 src/graph2mat/core/data/tests/test_processing.py \
 src/graph2mat/core/data/tests/test_spin_collinear.py \
 src/graph2mat/core/data/tests/test_periodic_hermiticity_oracles.py \
 src/graph2mat/tools/lightning/tests/test_matrix_components.py \
 src/graph2mat/tools/server/tests/test_physical_serialization.py \
 -q -rs --tb=short --disable-warnings
```

Resultado focal final, después de corregir los hallazgos de revisión y de
explicitar metadatos geométricos: **261 passed, 0 skipped, 63 warnings, 4.56 s**.
Log: `.superpowers/sdd/2026-10-18-nonorthogonal-contract-4-1/focused-final.log`.
Son 16 casos parametrizados añadidos y ampliaciones de oráculos ya existentes;
el total también incluye las regresiones del archivo test_processing.

Regresiones acotadas: **1323 passed, 1 skipped, 2181 warnings, 37.29 s**:
`pytest src/graph2mat/core/data/tests src/graph2mat/tools/lightning/tests/test_matrix_components.py src/graph2mat/tools/lightning/tests/test_physical_training.py src/graph2mat/tools/server/tests -q -rs --tb=short --disable-warnings`.
Suite CPU completa: **2732 passed, 20 skipped, 40974 warnings, 822.22 s**:
`pytest src -q -rs --tb=short --disable-warnings`. Mismo prefijo env/intérprete
que el comando anterior; logs `regressions.log` y `full-suite-final.log` en el
workspace del plan. No hubo fallos. El número de warnings se conserva, no se
presenta una ejecución sin advertencias.

Las20omisiones están declaradas: un ejemplo externo ausente en
`test_h_dimension.py`, una prueba inductor que exige GPU, nueve pruebas CUDA
de coeficientes y nueve de fusión homonuclear que requieren supervisor GPU
opt-in. No se usa ninguna omisión para sustituir un observable CPU de4.1.

La suite completa fue lanzada después del pase de fixes de revisión. Durante
su ejecución se añadieron cinco assertions de metadatos al oráculo periódico;
esa última definición se ejecutó en el focal final261correcto. No hubo cambios
productivos entre ejecuciones ni cambios en el número de tests por ese refuerzo.
La primera suite desfasada fue interrumpida deliberadamente y su log
`full-suite.log` NO es el gate de cierre; el resultado completo autoritativo es
`full-suite-final.log`. Los registros Task1/Task2 son snapshots anteriores,
no sustituyen el focal final.

Ruff de los cuatro archivos Python y diff-check pasan después de las ediciones
de tests. Los seis SHA-256 de cambios previos del usuario fijados en el ledger
coinciden al comprobarlos de nuevo. HEAD no cambió; no hay commits.

### Diagnósticos iniciales conservados

- Sin extras en PYTHONPATH: 52 pasan y un módulo HTTP se omite por FastAPI.
- Con HTTP pero sin SOAP: 117 pasan y ocho casos DM-history fallan. El trace
  mostrado con excepciones TestClient activadas identifica dscribe ausente.
  Reutilizar su directorio preexistente da baseline completo125/0omitidos.
- Primeras pruebas nuevas:167 pasan/dos fallan por fixture que entregaba
  ndarray a datos Torch. Se corrigió solo la fixture para entregar Tensor FP64;
  no se cambió producto ni se presenta como un fallo de reconstrucción.

No se instaló ni modificó ningún paquete. Los directorios temporales son
preexistentes; si desaparecen se debe proporcionar un entorno compatible, no
aceptar omisiones silenciosas como certificación HTTP.

## Revisión, mutaciones y conclusión

Una revisión Native independiente encontró0Critical,2Important,0Minor. Los dos
Important eran huecos de pruebas, no defectos productivos:

1. Comparaciones matriciales antiguas dejaban rtol=1e-7 implícito. Una rotación
   por congruencia de H/S de ángulo2e-8 conserva espectros, pero cambia S≈1e-8;
   el test pasaba antes del fix. Con rtol=1e-10 explícito, la misma mutación
   falla en la comparación de S, como debe. Log `mutation-readback.log`.
2. Las predicciones del batch solo cambiaban diagonales. Una mutación que
   sustituía predicted edges por reference edges pasaba antes del fix. Ahora
   se cambian aristas por estructura/canal con valores diádicos y S SPD, y esa
   mutación falla (S offsite0.0625 frente0.1875). Log `mutation-stale-edges.log`.

Las mutaciones solo se aplicaron en memoria de procesos diagnósticos separados;
no se editó producto ni matrices científicas. Esos logs contienen fallos
**esperados** del test mutado, no fallos de la suite correcta. Después, los
dos módulos afectados pasan110tests y el focal final261; la suite completa
posterior a los fixes también termina correcta. No se requiere otra revisión
ni se difieren menores.

Decisiones registradas: revisar diff del checkout en vez de un rango vacío
porque se prohíben commits (riesgo mitigado con lista/hash de cambios previos),
y conservar las exclusiones explícitas del contrato (riesgo de extrapolar
garantías fuera de cobertura). No se atribuyen mejoras ML/VRAM ni garantías
SPD universales, SOC, soporte fuera del grafo o formatos×bases×k exhaustivos.

No faltó implementación productiva demostrable para este contrato.4.1 cierra
con pruebas/documentación, no con un lector duplicado ni un cambio de modelo.
El Goal completo4.1+7.6 NO está terminado: el diseño7.6 está escrito y revisado
para implementabilidad, pero necesita aprobación, plan e implementación propia.
No se han medido precisión de aprendizaje, velocidad ni VRAM.
