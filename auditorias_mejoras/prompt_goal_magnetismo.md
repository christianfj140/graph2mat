# Prompt de goal — Graph2Mat: magnetismo (momentos como entrada y SOC sin TRS)

Copiar el bloque siguiente a una sesión en modo goal si el objetivo no está ya
activo. Si está activo, reanudarlo: no crear otro ni repetir tareas terminadas.

```text
Implementa y evalúa en modo goal el soporte magnético de Graph2Mat: momentos
magnéticos por átomo como entrada, representación completa del spinor con SOC
sin imponer inversión temporal (8 canales) y spin colineal con momentos.

REPOSITORIO Y REFERENCIA
Trabaja solo en <graph2mat>, rama g2m-v2, en ese
checkout. Lee completamente auditorias_mejoras/roadmap_magnetismo.md: fija
diseño, contratos, métricas, límites y criterios de cierre. Lee la fila 8.1 de
auditorias_mejoras/roadmap.md y docs/superpowers/specs/2026-10-19-soc-8-1-design.md.
HEAD al preparar el plan: d1c2283 o posterior; comprueba el real.

ORDEN
0. Arranque: inventario, cola g2m_autorun, referencia de tests enfocada.
1. F0 Fe bcc en SIESTA (CPU): pseudo FR, red, malla k, marco de DM.InitSpin,
   TSHS magnético con SOC, comprobaciones T y C4 en datos reales,
   metaestabilidad de patrones, H0 con momentos, S, colas, decisión sobre B'.
2. Representación de 8 canales en datos, con oráculo de Pauli independiente.
3. Momentos en los datos: lectura Mulliken, batch, cachés, ZIP, checkpoints.
4. Modelo: capa magnética local tras MACE, bloque de 8 canales, inversión
   temporal exacta por doble pasada, colineal con momentos, CLI y rechazos.
5. F4 dataset Fe (A colineal, B SOC con eje aleatorio, B' solo si F0 lo aprueba).
6. F5 campañas GPU: A Δ y Δ sin momentos; B Δ, Δ sin momentos y directo;
   3 semillas por brazo; métricas de D5, ablación y OOD.

AUTORIZACIÓN
El usuario autorizó (2026-10-21, «Lanza el goal y hazlo todo end to end»): cálculos SIESTA de F0 y F4 en E-cores
con control térmico, descarga del pseudo Fe FR del tarball PseudoDojo con
verificación sha256, y campañas GPU de F5 por g2m_autorun.

TRABAJO
Ejecuta sin preguntas rutinarias entre tareas. Escribe el plan detallado de
cada tarea antes de modificar código. Reutiliza el código SOC existente.
Test RED real → parche mínimo → GREEN → regresiones y suite. Oráculos
independientes para convenios; nunca validar un convenio solo con el código
que lo implementa. Preserva APIs, Python >=3.9, checkpoints y defaults; todo lo
nuevo opt-in. Registra por sublote commits, decisiones, comandos, tests,
medidas y siguiente acción en roadmap_magnetismo.md, y los resultados en la
fila 8.1 de roadmap.md. Reanuda desde ese registro y git log.

MEDIDAS
Métricas de D5 fijadas antes de ver resultados: H por grupo de canales,
bandas alineadas, momentos recalculados desde H (μB y ángulo), ablación sin
canales impares, OOD. Mismo presupuesto y splits congelados por brazo; tres
semillas. Separar implementación verificada de beneficio medido.

LÍMITES
Sin merge, push ni PR. Commits locales pocos y con sentido. No commitear en
MD_vs_AtomicDisplacement (sí editar sus scripts de bench y escribir en runs/).
GPU solo por g2m_autorun, ≥20 % de VRAM libre, sin OOM; CPU <80 °C. No tocar
procesos de otros usuarios ni trabajos ajenos de la cola; nunca pkill -f sobre
líneas que contengan rutas de scripts. Sin dependencias nuevas.

CIERRE
Revisión independiente final, correcciones verificadas, suite y gates sin
fallos nuevos. Rechazo motivado con evidencia cierra una tarea; «aplazado» o
falta de recurso no. Si falta un dato o una autorización imprescindible,
continúa con lo independiente e informa. Entrega por tarea: implementación,
pruebas, medidas, commits y límites.
```
