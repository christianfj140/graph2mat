# Prompt de goal — Graph2Mat: precisión, velocidad y VRAM

Copiar el bloque siguiente a una sesión en modo goal si el objetivo no está ya
activo. Si está activo, reanudarlo: no crear otro ni repetir tareas terminadas.

```text
Implementa y evalúa en modo goal los candidatos de precisión, velocidad y VRAM
de Graph2Mat. Demuestra las ganancias o documenta resultados neutros/negativos.

REPOSITORIO Y REFERENCIA
Trabaja solo en /home/christian/repositorios/graph2mat, g2m-v2, en ese checkout.
Lee completamente auditorias_mejoras/roadmap_precision_rendimiento.md y
auditorias_mejoras/roadmap.md; lee el cierre de goal_ingenieria.md.
El roadmap específico fija los contratos, límites y criterios de cierre.
HEAD inicial: d338d1a62113af6f0fa1e04df08802ad32abcbf8; comprueba el real.
3.1/3.4/3.5/3.6/3.9 ya están cerrados. Referencia histórica: 1541 pasan,
2 omitidos CPU; no presentarla como verificación nueva.

ORDEN
0. Inventario real y referencia enfocada reproducible.
1. 3.14 necesario → 5.3: compatibilidad de coeficientes y entrenamiento en
irreps con targets proyectados, sin reconstrucción innecesaria. Preservar
pesos/gradientes del MSE certificado; media por operación no es MSE por elemento.
MAE/Huber/gauge necesitan reconstrucción/fallback, no basta Q ortogonal.
2. 6.1/6.2/6.4: caché de tensores CPU segura, conversiones sparse vectorizadas
y batching por coste. Sin densificación global, muestras perdidas ni lifecycle roto.
3. 7.1: primera variante opcional de TP de aristas canal a canal o rango bajo;
equivarianza, simetría, cortes y gradientes comprobados.
4. 4.4–4.5: objetivos pertinentes con S, N_e/Mulliken, Tr(DS), McWeeny bajo
hipótesis válidas y coherencia EDM/DM/H. Resolver solo lo necesario de 4.1.
Después, primera variante 7.4: SO(2) o Slater–Koster neuronal con residuo.
No inventar DM/EDM ni imponer idempotencia a ocupaciones fraccionarias.

TRABAJO
Ejecuta directamente sin preguntas rutinarias entre bloques. Diseña/planifica
cada subproyecto antes de modificarlo. Reutiliza código/dependencias existentes.
Test RED real → parche mínimo → GREEN → regresiones y suite completa.
Cubrir componentes/H+S, readout por interacción, agrupación/orientación,
s/p/d/f, periodicidad, NaN/vacíos, workers/ZIP y DDP según aplique.
Preserva APIs, Python >=3.9, checkpoints y defaults; variantes científicas opt-in.
Registra por sublote commits, decisiones, comandos/logs, tests, medidas y siguiente
acción en roadmap_precision_rendimiento.md. Reanuda desde registro/git log.

MEDIDAS
Separar implementación verificada de beneficio demostrado. GPU: referencia y
candidato con mismo modelo/batch/dtype, warmup, rondas alternadas y dispersión;
allocated/reserved, RAM y coste inicial. Precisión: smokes y comparación pareada,
splits congelados, presupuesto/checkpoints iguales y al menos tres semillas.
Fijar métrica primaria y tolerancia antes; medir física/OOD pertinente.

LÍMITES
Preserva archivos ajenos. Sin fonones, otros repositorios, push/merge/publicación.
No tocar g2m_autorun.py, colas, entrenamientos activos ni datasets originales.
Entornos/salidas propios; no alterar el entorno compartido ni ocupar GPU de
trabajos ajenos. GPU ocupada: avanzar CPU. Datos/checkpoints existentes en lectura.
Commits locales explícitos permitidos. No rehacer Δ, normalización, edge updates,
checkpointing/compile; no ampliar a ingeniería general salvo dependencia necesaria.

CIERRE
Revisión independiente final, correcciones verificadas, suite y gates reales.
Rechazo motivado con evidencia cierra la evaluación; “aplazado” o falta de recurso
no. Continúa bloques independientes; informa si falta autoridad/dato imprescindible.
Completo solo con los criterios del roadmap específico satisfechos. Entrega por
candidato implementación, P/T/V, aceptado/rechazado, commits, pruebas y límites.
```
