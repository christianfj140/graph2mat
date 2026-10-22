# Dictamen por commit de Claude

Los 24 commits presentes al reanudar y los ocho posteriores se revisan sobre el HEAD final `e1e7e4b`. «Correcto» se limita al contrato y las pruebas descritos; no certifica todas las combinaciones ni campañas.

| Commit | Dictamen | Evidencia y límite |
|---|---|---|
| `f819bae` — docs(audit): bug and roadmap-vs-code audit of g2m-v2 after the upstream merge | Informe histórico | [Se contrasta con la auditoría independiente; la interpretación global de LR requiere D2.](README.md) |
| `81b18ca` — fix(cli): attach lr_scheduler to the model's optimizer when the config has none | Correcto en controles CPU | [Scheduler conectado al optimizador del modelo; rechazos de reanudación y etapas incompatibles probados.](lightning_metrics.md) |
| `69af9aa` — bench: constant vs cosine learning-rate probe on the w90 fp32_lin recipe | Runner conservado; alcance documental corregido | [No se repite su sonda ni se extrapola a runners con scheduler explícito.](experimentos_documentacion.md) |
| `ba9b440` — fix(data): subtract the point reference from every component of multi-component blocks | Correcto | [Referencia por componente; roundtrip de DM polarizada.](data_models.md) |
| `5a4d900` — fix(long_range): shift every H column and require h_and_overlap data | Correcto | [Ambos H reciben VS, S intacta y política comprobada.](data_models.md) |
| `2c3b840` — fix(tools): spin-polarized matrices in the analysis tools; keep every H0 image | Correcto parcialmente; completado | [Spin-down y mayor nsc correctos; faltaba alinear sc_off con igual nsc.](tools.md) |
| `d2082af` — docs(roadmap): cheap constant vs cosine learning-rate probe on w90 | Documentación corregida | [Las muestras CLI inspeccionadas no representan todas las campañas históricas.](experimentos_documentacion.md) |
| `4b5194b` — fix(mace): apply the pair envelope to the Hamiltonian-context edge channels | Correcto | [Envolvente también actúa en los canales del contexto.](data_models.md) |
| `75e75d2` — fix(metrics): gauge-invariant spectral loss for non-orthonormal columns (4.7) | Correcto | [Proyección con G real restaura invariancia de gauge bajo el floor de S.](lightning_metrics.md) |
| `2caeb69` — fix(lightning): sorted globs; model and data basis types must match | Correcto parcialmente; completado | [Globs ordenados; acuerdo de base necesitaba capas, orden y convención además del tipo.](data_models.md) |
| `6509fad` — fix(callbacks): H-only predictions of (H, S) labels in MatrixWriter and SamplewiseMetricsLogger | Correcto en controles CPU | [Writer y métricas H-only/H+S, metadata y colisiones probados.](lightning_metrics.md) |
| `f4fe6f7` — fix(data): reject an explicit nsc different from the matrix's in from_config | Correcto como rechazo | [nsc explícita incompatible falla antes de extraer etiquetas.](data_models.md) |
| `eca05a8` — fix(siesta): SIESTA results only land when the run succeeds; safe H0 keys | Correcto parcialmente; completado | [Publica tras éxito y separa rutas; se añade verificación de entradas/salida y cache legado.](tools.md) |
| `fb236ca` — fix(data): read NetCharge with fdf label rules and %include | Correcto | [NetCharge mediante lector FDF, incluidos aliases e includes locales.](data_models.md) |
| `a492fbf` — fix(long_range): electronic charges in qeq add up to -NetCharge | Coherente con convención declarada | [Cargas electrónicas suman -NetCharge.](data_models.md) |
| `624101e` — fix(mace): gradient checkpointing with frozen parts | Correcto | [Checkpoint reentrante se omite si entradas no requieren gradiente; readout mantiene gradientes.](data_models.md) |
| `4dd8f9c` — fix(table): basis grouping checks shell order; max grouping of atomic tables | Correcto en su alcance | [Orden de capas y group(max) protegidos; faltaba DATA-03 en configuración/tabla.](data_models.md) |
| `67aa8e0` — fix(coefficients): reverse same-species edges of non-symmetric matrices | Correcto en combinaciones verificadas | [Aristas inversas homonucleares y equivalencia con pérdida matricial.](data_models.md) |
| `b52c73b` — fix(callbacks): per-sample metrics under DDP cover every rank | Correcto | [Dos procesos CPU: cuatro identidades exactas al reunir CSV por rango.](lightning_metrics.md) |
| `1e12f10` — fix(server): HistoryCopy pairs each label with the previous step's matrix | Sin objeción confirmada | [Correspondencia descriptor/label trazada; HistoryCopy y consumidores pasan.](tools.md) |
| `c6c4850` — feat(siesta): spin-orbit Delta-learning without MD_vs_AtomicDisplacement (8.1) | Ruta real correcta; compleja corregida | [Oráculo SOC independiente; mezcla de almacenamiento y proyección compleja corregidas.](tools.md) |
| `55dbbde` — fix(mace): reject torch.compile mode reduce-overhead (6.8) | Correcto como rechazo | [reduce-overhead se rechaza explícitamente; sin certificación CUDA/Inductor.](data_models.md) |
| `f3b4d29` — test: cover roadmap options that had no test | Pruebas útiles | [Se conservan; dependencias opcionales y combinaciones excluidas mantienen sus omisiones.](data_models.md) |
| `17e246f` — fix(tools): fdf label rules, robust noise fdf, client geometry, attribution supercell | Correcto parcialmente; completado | [Cliente y atribución conservados; lector FDF, includes y cierre de bloques reforzados.](tools.md) |
| `2248f52` — fix(lightning): training stages under DDP, standalone evaluation loss, scaleshift error | Corrección de evaluación incorrecta; sustituida | [DDP/scaleshift conservados; la pérdida se restaura desde la etapa guardada, no la última prevista.](lightning_metrics.md) |
| `d3d7fe5` — fix(data): do not guess symmetric edge representatives from label counts | Correcto como rechazo | [Se preservan código y test que eliminan la inferencia arbitraria por subset-sum.](README.md) |
| `2df88f8` — fix(siesta): percent-encode query values in the Lua server client | Sin objeción estática | [Valores URL codificados; no se certifica ejecución Lua en esta revisión.](tools.md) |
| `e2d204a` — fix(models): reject options that would be ignored; Ewald k-vectors for skewed cells | Correcto en controles CPU | [Ewald equivalente y gradientes invariantes; guardias de opciones no consumidas conservadas.](data_models.md) |
| `90361cc` — test(e3nn): autocast fallback warning test does not depend on earlier tests' filters | Correcto | [Filtro always permite comprobar el contador propio de warnings sin depender del registro global.](README.md) |
| `d598b65` — docs(audit): fix status of every finding; roadmap text brought up to date | Limpieza útil; no cierra todo el roadmap | [Se conservan sus cambios; D2 acota LR y el informe independiente mantiene huecos documentales/de alcance.](README.md) |
| `d32c955` — test(magnetic): MatrixWriter writes spin_orbit_full predictions (round trip) | Prueba conservada | [Roundtrip de MatrixWriter con spin_orbit_full en la suite integrada.](README.md) |
| `e1e7e4b` — docs: MatrixWriter writes spin-orbit predictions; only the server still refuses them | Documentación coherente con writer | [Aclara escritura SOC sin S y rechazo del servidor; límites físicos se conservan.](README.md) |
