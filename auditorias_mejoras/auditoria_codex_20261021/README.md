# Auditorías independientes de Graph2Mat

La auditoría original está cerrada sobre `a66b46c`: los 267 commits que el 21 de octubre estaban por delante de origin/g2m-v2. Se terminó el 22 tras recuperar el trabajo interrumpido por cuota.

- [Auditoría de bugs](bugs.md): 25 defectos funcionales confirmados —19 introducidos y 6 heredados— y una regresión de tests. Incluye prioridad, origen, evidencia y límites.
- [Roadmap frente a implementación](areas/roadmap.md): dictamen y discrepancias. [Tabla completa](areas/roadmap_tabla.md): 81 puntos y 217 IDs originales, sin omisiones.
- [Revisión posterior de correcciones de Claude](revision_claude_20261022/README.md): terminada sobre sus 32 commits hasta e1e7e4b. Arreglos adicionales aplicados en 31 archivos; los 25 contraejemplos originales quedan cerrados en las rutas verificadas. Suite CPU completa: **3599 pasan, 25 omitidas**. Ruff pasa; Mypy conserva ocho errores previos. Detalles y límites en el informe.

El roadmap original se clasifica en 55 puntos OK, 9 discrepancias documentales, 5 parciales reales, 6 pendientes autorizados, 4 históricos/experimentales y 2 excluidos. Esto distingue existencia de código, resultados medidos, promoción de defaults y decisiones del usuario; una opción sin beneficio general no se convierte en código sobrante por ese motivo.

La auditoría original no modificó producto ni lanzó GPU, SIESTA, fits o colas. La autorización posterior permite corregir bugs: esos cambios y su verificación se registran aparte. No hay commits ni push propios.
