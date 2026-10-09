# Batching de entrenamiento por coste declarado — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL `superpowers:executing-plans`.
> Ejecución directa/TDD, mismo checkout, sin agentes por tarea; revisión final
> independiente solo al cerrar el goal completo. **Verificado CPU; commit/task-done por registrar.**

**Goal:** Opt-in para limitar un proxy explícito de coste por batch sin perder
muestras, romper métricas/lifecycle o producir diferente número de pasos DDP.

**Architecture:** Añadir solo `max_batch_cost: Optional[int]=None` al final del
constructor MatrixDataModule. None conserva exactamente los loaders actuales.
Proxy por muestra: C*(sum_i n_i² + sum_directed_edges n_i*n_j), con tamaños de
basis_table y componentes del procesador, no valores de labels. No es una cota
VRAM: ignora activaciones/backbone/allocator. Prepass una vez sobre el dataset
train efectivo (después del split), cache solo de enteros por identidad de
dataset; reconstruir al cambiarlo. No materializar otra caché de muestras.

Sampler interno BatchSampler usa sampler existente, greedy en su orden, y
`batch_size` sigue siendo límite adicional de estructuras. Muestra que supera
budget produce singleton documentado; última muestra nunca se descarta.
RotatingPoolData dinámico incompatible: error antes de leer, no costes obsoletos.
Validation/test/predict conservan loaders actuales; no opciones extra.

**Tech Stack:** Python>=3.9, Torch/PyG/Lightning/NumPy existentes.

**Spec:** roadmap_precision_rendimiento.md Task2/6.4 y docs/data_loading.rst.
Preparado mientras source/tests6.2 estaban congelados. Task1 iniciado después
del cierre fullCPU/commit/task-done de6.2, BASE903a132.

## Global Constraints

- Graph2Mat/g2m-v2/mismo checkout; preservar APIs/checkpoints/defaults.
- Sin GPU ajena/autorun/colas/fonones/shared envs/datasets originales.
- Un presupuesto de proxy no promete evitar OOM, ni mejora de precisión/VRAM.
- Empaquetar distinto cambia agrupación, pasos por época y trayectoria de
  optimización. Comparaciones posteriores con presupuesto de updates pareado.
- No silently_drop/pad: registrar DDP padding y contar realmente lo evaluado.
- No nuevas dependencias, sampler de pares ni generalización a custom samplers.
- Ruling tras RED real: opt-in activa con warning el ajuste público de Lightning
  reload_dataloaders_every_n_epochs=1 y captura época antes de len/reinyección;
  sin ello Lightning trunca épocas posteriores más largas. Datasets/costes reutilizados.

## Review Focus

- DDP: greedy por shard puede dar pasos distintos y bloquear all-reduce.
- __len__ debe coincidir con iter, sin reshuffle cada vez que Lightning lo consulta.
- Cobertura/determinismo/vacíos/ghost/oversize, Subset/stages/ZIP/workers0/2.
- Defaults idénticos y flag checkpoint aditivo; collate de coeficientes preservado.
- Pair loss depende de partners en mismo batch: rechazar combinación conocida
  con pair_difference_mse antes de forward, también tras cambio de loss por etapa.

## Evidencia / decisiones de diseño

Perfil propio readonly cost-batching-profile.json en registro temporal, C2
dirigido s/p/d/f,4/16/72/128puntos: costes5592/22368/100656/178944;
extracción desde sample mediana .0115/.0118/.0129/.0137ms, lazy config1.68/
1.40/2.92/3.59ms (primera llamada, no benchmark comparativo). Tensorbytes
23392/93424/420240/747056. Secuencia desigual12muestras: batches fijos5 tienen
proxymax313152 y tensorbytesmax1309384; boceto greedybudget100656 da oversize
singleton128, max178944/748104bytes, cobertura exacta. **Es boceto, no sampler
implementado ni evidencia GPU.** Coste I/O del prepass no medido aún.

Alternativas: límite solo n_edges ignora tamaños orbitales; ordenar por tamaño
cambia aún más agrupación y añade heurística. Greedy del orden sembrado mantiene
alcance mínimo. No inferir capacidad libre de GPU ni autoajustar budgets.

Fuente primaria local revisada: Lightning2.6.6 utilities/data.py reinserta
DistributedSampler en BatchSampler y conserva kwargs capturados si se construye
dentro del hook *_dataloader. lightning_fabric llama set_epoch sobre
batch_sampler.sampler, no sobre BatchSampler directamente. Verificar también
floor3.9 instalado, no asumir API de otra versión.

DDP propuesto: para DistributedSampler estándar, simular con la misma clase
Torch/seed/epoch/drop_last los shards de todos los ranks sin collectives, calcular
max batches y repetir batches locales cíclicamente hasta igualar pasos. Conserva
padding de samples del sampler nativo, añade padding explícito de batches.
No copiar internals de Torch. Custom distributed samplers no certificables:
error explícito en opt-in, default intacto. Cache plan por epoch; __len__ y
__iter__ usan mismo plan. Single RandomSampler con generator privado sembrado
desde torch.initial_seed+epoch. Trainer retiene el plan durante toda la época,
incluso si prefetch agota el sampler; siguiente loader/epoch lo reemplaza.
Standalone sin Trainer invalida al terminar iterator, no por len.
Probar reinyección real Lightning, no solo sampler aislado.

### Task 1: sampler, integración train y verificación distribuida

**Files:** Modify src/graph2mat/tools/lightning/data.py;
Modify src/graph2mat/tools/lightning/model.py solo guard de pair loss si necesario;
Create tools/lightning/tests/test_cost_batching.py;
Modify tools/lightning/tests/test_ddp_cpu.py;
docs/data_loading.rst, benchmark/informe en auditorias_mejoras y roadmaps.

**Interfaces:** Consume dataset train/Subset/sampler/BasisTable existentes;
produce batches normales PyG/TypeLayoutCollate. No export público nuevo.

- [x] Step 1: task-start/brief/BASE; releer callers/stages/DDP/pair_loss y validar
  diseño mínimo con APIs locales de ambos entornos antes de editar producto.
- [x] Step 2: RED greedy budget/límite batch_size/cobertura/último/oversize/ghost,
  enteros positivos (rechazar bool/float/no finitos) antes de I/O, len estable,
  reproducibilidad misma semilla/época y variación entre épocas; dataset intacto.
- [x] Step 3: parche mínimo sampler interno, cache plan y prepasscosts train;
  RED/verde default loader igual, flag opcional guardado, Subset y collation
  matrix/coeficientes, workers0/2 spawn/fork, ZIP/stages/resource ownership.
- [x] Step 4: DDP real2ranks, costes muy desiguales y número impar de muestras;
  misma longitud por rank/epoch sin hang, reinyección/set_epoch reales, muestras
  originales cubiertas, padding declarado y métricas SSE/count igual al literal
  de todos los batches efectivamente procesados. Control negativo debe detectar
  el camino viejo/fijo si el entrenamiento no usa cost sampler. Retirarlo.
- [x] Step 5: paired_loss/reemplazo loss por etapas incompatible falla antes
  del backbone, no parejas separadas silenciosamente; rotatingpool y custom
  samplers no certificados fallan con explicación sin cambiar default.
- [x] Step 6: benchmark mismo conjunto congelado, layout/dtype/model y total
  trabajo, medir prepass/inicialización/RAM/collate y dispersión de costes;
  GPU solo libre, allocated/reserved/paso/throughput y oversize documentados.
  Primaria reducción de pico del proxy sin pérdida de muestras; no regresión
  numérica de predicciones para batch partitions (float64 tolerancia1e-10),
  métricas epoch por elementos. No concluir precisión con smoke ni equivalencia.
  CPU: pico proxy313152→178944, cobertura exacta; collate1.307→2.557ms,
  inicialización.153→17.625ms. Equivalencia MACE real C1/C2 simétrico/dirigido.
  GPU con cómputo ajeno: recurso ausente; no beneficio VRAM/velocidad GPU demostrado.
- [ ] Step 7: focused/floor3.9/fullCPU final source/tests congelados;
  Ruff/mypy-core/compile39/diff, docs/roadmaps, commit explícito y task-done real.
  Luego7.1; medidas GPU/precisión siguen abiertas si faltan recursos/datos.
  FinalCPU2209pasan/2omitidos/26518warnings/634.88s,exit0; focused40 y floor40
  verdes. Source/tests congelados; benchmark repetido con hash finaled975e57.
