# Preparación del comparativo científico 7.6

Fecha: 2026-10-18. Petición del usuario: «Haz esto pero en la rama g2m-v2»,
en respuesta a la propuesta de certificar datos H/S/DM/EDM y preparar la
comparación H-only directa frente a multitarget. No se reabre el Goal técnico
4.1+7.6, ya cerrado; esta preparación no certifica beneficio científico.

## Estado de rama y preservación

Cambio no forzado de `physics-prior` a `g2m-v2`, autorizado expresamente por el
usuario. HEAD final `270d17d1f204293a059baa09e8f1b3756981e2bf`.
El hash de `git diff --binary` antes/después del cambio coincide: los cambios
locales rastreados se conservaron. Los 28 archivos del manifiesto
`.superpowers/sdd/2026-10-18-multitarget-7-6/final-source-sha256.txt` también
coinciden. Sin stash, merge, commits, worktrees ni restauraciones forzadas.
Los dos commits propios de `physics-prior` quedan en esa rama, sin mezclarlos.

## Inspección local ejecutada

Inventario de nombres con `rg --files --hidden --no-ignore`, sin seguir enlaces
de directorio, excluyendo `.git`, `.venv`, `.superpowers`, build/dist y cachés.
Se buscaron TSHS/TSDE, DM/EDM, NetCDF, ZIP y TAR. La única entrada con esos
patrones fue `auditorias_mejoras/evidencia_slater_koster_cpu.tar.gz`: su índice
contiene 55 entradas, ninguna con sufijo de matriz TSHS/TSDE/DM/EDM/HSX/NetCDF.
Se leyó el índice, sin extraer ni modificar archivos.

Los nombres abundantes bajo `.superpowers` corresponden a artefactos de tests;
no deben presentarse como un dataset científico ni como nuevos snapshots.
Esta búsqueda no es una prueba absoluta de ausencia de datos: no inspecciona
contenedores arbitrarios, todas las variantes de nombres ni fuentes externas.
Sí implica que no se ha certificado un recurso local para el comparativo.
No se usaron predicciones ni fixtures sintéticas como etiquetas científicas.

## Candidato histórico, todavía no abierto

**Actualización posterior:** usuario autorizó esa lectura con «Hazlo».
Recurso auditado en CPU,1500muestras compatibles,13512hashes intactos; detalles
y avisos de splits en [certificación actual](certificacion_multitarget_w90_20261018.md).
Las frases «todavía no abierto»/«falta autorización» siguientes describen el
estado previo, no el actual. No autorizan GPU ni el lote científico.

El documento local `inventario_objetivos_fisicos.md` registra el recurso:

`$MDVS_ROOT/Comparison/datasets/graphene_w90_joint/graphene_w90_phase1_iid1500`

Ese inventario histórico contabiliza 1500 tripletes TSHS/TSDE/HSX, con
`benchmark_ready=true`. Solo comprobó numéricamente una muestra del conjunto;
no certifica hoy todos los snapshots, la convergencia ni la independencia de
splits. Es el primer candidato propuesto por disponer de etiquetas archivadas,
no una selección definitiva ni una verificación nueva de su estado.

La raíz está fuera de Graph2Mat. No se han listado ni leído esos datos durante
esta tarea; falta autorización explícita de lectura de ese recurso. Tampoco
se han seguido los overlays/enlaces que salen del repositorio.

## Siguiente paso condicionado

1. Autorizar lectura sola del recurso anterior. No modificar código, datos,
   manifiestos ni resultados de ese otro repositorio; sin GPU ni SIESTA.
2. Certificar matrices coincidentes, hashes/procedencia, geometría/base/orden
   orbital, spin, soporte observado y gauge H/EDM con los helpers existentes.
   Revisar convergencia y los splits históricos sin rehacerlos automáticamente.
3. Presentar selección ID/OOD, receta/pesos, criterio de checkpoint y protocolo
   de comparación H-only directa/multitarget. No usar un baseline delta como
   sustituto de la referencia directa ni añadir H+S como tercer brazo implícito.
4. Acordar piloto GPU y presupuesto antes del lote de seis fits pareados
   (dos brazos, semillas 0/1/2), medir H/niveles/gap con S_ref y S_pred por
   separado, calidad/SPD de S, DM/EDM y observables físicos aplicables.

Estado actual: rama corregida, lectura autorizada y etiquetas del recurso
auditadas. Una duplicación exacta train/validación y bloques MD compartidos
se notifican solo como alarma; splits intactos. Propuesta de comparativo
documentada, no receta/presupuesto de campaña aprobados. Sin OOD independiente
certificado, piloto, entrenamientos, colas ni medición GPU/VRAM. Defaults intactos.
Gate completo posterior del auditor:2896passed/20skip/0fallos,850.60s,exit0;
Ruff/diff correctos y28hashes del cierre técnico intactos.
