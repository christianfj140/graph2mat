# Traspaso entre agentes — g2m-v2 (2026-10-13)

Escrito para: el agente (Codex u otro) que continúe este trabajo en el repositorio `graph2mat`, rama
`g2m-v2`, sin acceso al chat anterior. Todo lo que no está aquí ni en `roadmap.md` se perdió con la
conversación; si algo parece faltar, pregúntalo al usuario en vez de suponerlo.

## 1. Dónde está cada cosa

- Código y estado del roadmap: `/home/christian/repositorios/graph2mat`, rama `g2m-v2` (HEAD `ae65fbc` o posterior).
  `auditorias_mejoras/roadmap.md` es la fuente de verdad: cada punto tiene su estado, medidas y fecha.
- Resultados de la campaña 7.1/7.4: `auditorias_mejoras/campana_readout_w90_6x6_20261010.md` y
  `experimentos_readout_w90_6x6_20261010_v1/precision_summary.json` (directorio ignorado por git).
- Diagnóstico del error de bandas fuera de distribución:
  `/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/results/dataset_design_curves_v1/g2m_improvements/ood_band_attribution/README.md`.
- Worktree auxiliar en uso: `/home/christian/repositorios/graph2mat-wt-cpu` (rama `g2m-v2-cpu`, ya fusionada en `g2m-v2`).
  Lo importan las sondas y el experimento de energía encolados; no lo borres mientras haya trabajos suyos pendientes.
- Entorno: `/home/christian/repositorios/MD_vs_AtomicDisplacement/.venv/bin/python` (graph2mat editable apunta al
  checkout principal). Para ejecutar el código de otro árbol hay que quitar el finder editable:
  `PYTHONPATH=auditorias_mejoras/probe_site:<árbol>/src:auditorias_mejoras`.

## 2. Reglas que no están en el código

1. **Cola de GPU** (`.../g2m_improvements/autorun/queue.jsonl`, daemon `g2m_autorun.py`, PID en `state.json`):
   - Añade solo al final y con `run_readout_comparison.append_queue` (bloqueo + conserva los bytes previos).
     Para reordenar, copia de seguridad y reescritura atómica bajo `flock`, abortando si el fichero cambió.
   - Los trabajos "estrictos" (campaña, sonda de velocidad, experimento de energía) ceden la GPU a cualquier
     otro proceso y se quedan parados. Encola lo nuevo con `"after"` = ids vivos (pendientes sin ancestro fallido).
   - El daemon ya vigila temperatura (pausa con SIGSTOP) y OOM (reencola y baja la concurrencia). Respétalo:
     trabajos nuevos con varios procesos deben comprobar `g2m_autorun.gpu()/cpu_temp()` antes de arrancar más.
2. La campaña congeló hashes de `src/graph2mat/**` y de módulos del repo MD_vs_AtomicDisplacement
   (p. ej. `g2m_improvements_benchmark.py`). Ya terminó; los trabajos encolados posteriores usan el árbol actual.
   Registra variantes nuevas con scripts envoltorio (`g2m_bench_grouping.py`, `g2m_bench_neutral.py`,
   `g2m_bilayer_readouts.py`) en lugar de editar ese fichero.
3. No pongas defaults nuevos sin medir fuera de distribución: criterio de promoción = razón con IC95 superior < 1 y
   ninguna regresión OOD > 5 % en H alineado ni en bandas alineadas.
4. Suite de tests: `pytest src` da 2475 pasan, 16 omitidos y 4 fallos conocidos
   (`test_extrapolation` HistorySOAPs: falta el paquete `dscribe`; ya fallaban antes).

## 3. Estado de lo encolado (2026-10-13 10:48)

Quedan 7 trabajos, todos del comparativo de bicapa (`g2m_bilayer_readouts.py`, material `grhbn_bl`, receta
`fp32_lin_gaugeh_kl_delta_u16k` + 7.1/7.4/nodemix): `…_lowrank:2`, `…_sk:0,1,2`, `…_nodemix:0,1,2`.
Resultados en `runs/grhbn_bl/<prefijo>__ts<seed>__g2mv2_<variante>/result.json`
(claves `val_metrics` y `ood_grhbn_moire_unseen`). Los 12 de `tbg` (carpeta `runs/bilayer`, prefijo `n474`) están hechos.

## 4. Resultados ya registrados (resumen)

- 7.1 y 7.4: mejoran H en test (7.4 en w90 ×0,73, IC95 0,60–0,93) pero empeoran las bandas fuera de distribución
  (×1,1–1,6). No promovidos. 7.7 (DM → H en un paso): NO_GO.
- 1.9 `nodemix`: peor en grafeno, pero mejora H en TMD, BaTiO3 y ZnO (bandas de las láminas de ZnO peor). El usuario
  pidió expresamente **no** hacerlo default.
- 7.5 `max`, 4.5 energía, 1.3 y 1.11: no se promueven.
- Atribución OOD: 83–100 % del error de bandas de 6x6 viene de acoplamiento intercapa.
- Comparativo con entrenamiento de bicapa (tbg, 3 semillas): 7.1/7.4/nodemix no mejoran H (±15 %) y las bandas
  empeoran ×3,8–4,7 frente a la referencia Δ (1,98 meV). grhbn_bl: parcial (7.1 con 2 semillas: bandas ×0,78,
  H ×1,10–1,14); faltan 7.4 y nodemix.

## 5. Siguiente paso

1. Cuando los 7 trabajos terminen, resumir grhbn_bl con el mismo código que tbg (media de semillas, razón frente a la
   referencia) y registrar ambos en `roadmap.md` (puntos 7.1, 7.4, 1.9) y en `campana_readout_w90_6x6_20261010.md`.
2. Conclusión provisional: la capacidad extra del readout no ayuda cuando el dominio está cubierto y la referencia Δ ya
   es muy buena. Si se confirma con grhbn_bl, proponer al usuario cerrar 7.1/7.4/1.9 como opciones sin promover.

## 6. Decisiones que son del usuario (no las tomes solo)

1.15 (monitor de checkpoint independiente de la pérdida, cambia un default), 3.8 (residual opcional en
`E3nnInteraction`), reanudar a-SiO2 (la caché de templados está en `bench/sio2/glass_cache`; el vidrio de 216 átomos
no pasa la puerta Si–Si ≥ 2,8 Å con enfriamiento de 20 ps; v6 con 100 ps quedó parado por decisión del usuario),
cualquier cambio de default.

## 7. Ficheros sin commitear en MD_vs_AtomicDisplacement

Repositorio con muchos cambios del usuario sin commitear; no commitees nada ahí sin preguntar. Son nuestros:
`Comparison/scripts/{g2m_bench_grouping,g2m_bench_neutral,g2m_bilayer_readouts,g2m_dm_pilot,g2m_ood_band_attribution}.py`,
`tests/test_g2m_bench_materials.py` (nuevo) y cambios en `Comparison/scripts/g2m_bench_materials.py`
(semillas disjuntas de SiO2, templados en paralelo con protecciones de temperatura/VRAM/OOM, `quench_fs`).

## 8. Estilo

El usuario escribe en español; las respuestas en el chat van en estilo telegráfico (AGENTS.md), pero commits,
documentos y código en prosa normal. Informa de resultados con cifras y di explícitamente lo que no se midió.
