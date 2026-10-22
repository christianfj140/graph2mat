# Revisión de scripts y afirmaciones de experimentos — 2026-10-22

## X1: presupuesto de updates

Los runners SOC y magnético no estaban incluidos en los arreglos de Claude y conservaban la división entera que ejecutaba un número de updates distinto del registrado. La corrección usa techo de épocas y `max_steps` exacto, además de rechazar presupuestos no positivos antes del preflight. Mantiene el presupuesto predeterminado de 8000, sin reinterpretar resultados históricos.

La revisión cruzada detectó una insuficiencia del primer parche propio: con un presupuesto menor que una época, Lightning podía parar sin validar y no producir ningún checkpoint `best`, que el runner externo exige. La validación ahora se programa en pasos globales: `min(n_updates, steps_per_epoch * max(1, epochs // 200))`, con `check_val_every_n_epoch=None`. Mantiene la cadencia elegida para épocas completas y garantiza una validación dentro del presupuesto corto. El checkpoint elegido sigue siendo el mejor de las validaciones realizadas; no se exige que sea el último estado entrenado.

Las pruebas ejecutan los cuerpos originales de `run` sin importar la inicialización del arnés externo. La frontera `run_training` usa un Trainer CPU real y un modelo lineal sintético: presupuestos 2, 7 y 10 deben ejecutar exactamente esos pasos y producir un único checkpoint best existente. Los presupuestos 0 y −1 se rechazan antes de entrenar. La extracción AST conserva las anotaciones pospuestas para Python 3.9. No se ejecutó un intérprete 3.9 real.

Evidencia: `campaign_red.log` registra 11 fallos del presupuesto/guardia originales; `campaign_checkpoint_red.log` añade dos fallos reales por falta de checkpoint antes de ajustar la validación. `campaign_checkpoint_green.log`: **12 pruebas pasan**, incluidos seis entrenamientos sintéticos CPU y ambos controles de la guardia. No se lanzó entrenamiento científico ni se modificaron campañas existentes.

Los archivos históricos `result.json` y `train_done.json` se conservan. El arreglo no acredita retrospectivamente los updates de sus ejecuciones ni relanza resultados cacheados. Esos artefactos mantienen su procedencia anterior.

## X2: primera guardia de energía

El primer `start` omitía `_room` cuando no había entrenamientos propios. Ahora todos los arranques requieren esa comprobación, preservando la posibilidad de compartir GPU cuando la telemetría lo permita. Las regresiones comprueban rechazo y permiso sin iniciar procesos GPU. No se modificó daemon ni cola.

## D2: alcance de la afirmación del learning rate

`69af9aa`/`run_lr_schedule_probe.py` y `d2082af`/`roadmap.md` generalizaban «todos los entrenamientos históricos» a partir del fallo CLI con `optimizer: null`. Esa extensión contradice runners explícitos: `run_cost_batching_comparison.py:594–606` instala `CosineAnnealingLR` mediante `configure_with_scheduler` y desactiva `auto_configure_optimizers`. `training_recovery.py` rechaza checkpoints sin `lr_schedulers`. El cierre 6.4/4.7 usa ese protocolo y 7.6 administra su scheduler directamente.

La documentación limita la afirmación a los checkpoints y recetas CLI inspeccionados. No altera cifras ni declara equivalencia entre campañas con protocolos distintos. El informe histórico de Claude se conserva como artefacto, con esta corrección posterior de interpretación. Su sonda w90 de 1000 épocas no prueba superioridad general de ningún schedule y no se repite.

## Alcance de commits

`69af9aa`: runner experimental separado, sin efecto en código productivo. `d2082af`: resultados conservados; se corrige la generalización de alcance. `d598b65`: limpieza documental conservada; no sustituye este contraste. Los arreglos de CLI de `81b18ca` los valida el informe Lightning/métricas. No se lanzaron SCF, GPU ni ajustes científicos, ni se tocaron colas.
