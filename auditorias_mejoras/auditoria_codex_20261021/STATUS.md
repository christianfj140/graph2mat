# Trabajo terminado — 22 de octubre de 2026

Las dos auditorías originales y la revisión posterior de Claude están terminadas. El usuario autorizó también corregir lo incorrecto; los parches ya están aplicados al checkout original. No hay trabajo de esta solicitud esperando aprobación.

## Auditoría original

Snapshot `a66b46cb0955ca344693d50737c5e9952fc257fc`: exactamente 267 commits sobre `origin/g2m-v2`. Resultado: 25 defectos funcionales confirmados (19 introducidos, 6 heredados), más D1 de tests. Roadmap: 81 puntos y 217 IDs contrastados sin duplicados; 55 OK, 9 discrepancias documentales, 5 parciales, 6 pendientes autorizados, 4 históricos/experimentales y 2 excluidos. Los huecos de alcance no se implementaron automáticamente ni se reabrieron campañas.

La suite histórica inicial dio 3341 passed, 38 skipped, 5 failed por preparación de entorno. El gate posterior, sobre el mismo snapshot y entorno corregido, dio 172 passed y 4 failed D1. Esos logs se conservan; no representan el resultado de la entrega final.

## Revisión y correcciones posteriores

Revisados 32 commits de Claude hasta `e1e7e4b9f4c516a5e9011b2273a4f6a7e829ac27`, preservando también los ocho cambios locales que existían al reanudar y terminaron en sus commits. Nueve contraejemplos originales quedan resueltos por Claude y 16 mediante correcciones adicionales propias. Se corrigieron además problemas de acuerdo angular entre modelo/datos, almacenamiento SOC complejo y reescritura FDF, y una generalización documental incorrecta sobre los schedulers históricos.

La revisión cruzada reforzó los primeros parches propios: Trainer CPU real garantiza checkpoint con presupuestos cortos y la imagen mínima general también cubre bases muy sesgadas. Se eliminaron los hooks de 2248f52 que imponían una etapa futura al evaluar un checkpoint temprano; permanecen las guardias DDP y scaleshift.

## Verificación final y entrega

Suite completa: **3599 passed, 25 skipped, 0 failed**, 971.28 s, 32089 warnings. Ruff sobre `src tests` y diff check pasan. Mypy conserva exactamente ocho errores de la fundación; sin errores nuevos. No se afirma una validación científica global ni soporte de las combinaciones omitidas.

Fuente combinada probada en `/tmp/graph2mat-codex-review-20261022-f80m50ec/integrated-source`, con ambas extensiones Cython recién compiladas y wrapper `integrated-python`. Se verificó que sus 429 entradas de fuente/configuración/documentación no cambiaron durante el gate. Después se aplicaron únicamente los 31 archivos propios al checkout original, comprobando coincidencia byte a byte, HEAD de Claude intacto y conservación de los cinco symlinks históricos. No volver a ejecutar el script de integración: el parche ya está aplicado.

Informe principal: `revision_claude_20261022/README.md`. Evidencia persistente: `revision_claude_20261022/evidencia/`, incluidos JUnit, log, manifiestos, parche completo y artefactos focales. Respaldo previo a la aplicación: `/tmp/graph2mat-codex-review-20261022-f80m50ec/upstream-before-integration`.

No se lanzaron GPU, SIESTA real, campañas científicas, cambios de cola, commits ni push. Solo se entrenaron ejemplos sintéticos CPU de tests. La instalación compartida de MD_vs y sus binarios no se alteraron: una instalación editable debe recompilar `_sparse` para usar esa corrección. Las cachés SIESTA legadas sin recibo se conservan y requieren un directorio nuevo.
