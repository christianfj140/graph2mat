# Protocolo posterior 7.4: precisión y GPU — NO EJECUTADO

El goal actual solo implementa y verifica CPU. Ninguna cifra CPU demuestra
mejora de precisión, velocidad GPU ni VRAM. Mantener readout default sin cambios.

## Preflight bloqueante antes de autorizar el experimento

Todavía **no listo**: faltan dataset/material concreto y manifests de splits
congelados para esta comparación. No reutilizar nombres de datasets como si
fueran manifests comprobados. Preparar y aprobar:

- Rutas y SHA256 de configuraciones, H y base orbital; S/referencias de bandas
  necesarias para el evaluador. Spin C1/C2, unidades, H0 y gauge explícitos.
- Manifests train/validation/test y OOD con hashes, sin fuga entre trayectorias,
  materiales o estructuras relacionadas. Documentar qué distribución es OOD.
- Receta única congelada: backbone, cutoff, preprocessing, agrupación, pérdidas,
  normalización, optimizador, scheduler, batch efectivo y selección de checkpoint
  por validación. Mismo H0 y mismo gauge entre variantes. No alterar 4.4–4.5.
- Presupuesto primario en pasos y muestras vistas idéntico, junto al tiempo
  real consumido. Registrar convergencia y coste; igual tiempo no sustituye
  igual trabajo. Material, scheduler y splits no pueden elegirse tras ver test.
- Evaluador científico con oráculos y referencias; bandas/gap y OOD no se
  marcan evaluados si faltan referencias. Sin OOD medible, no hay promoción.
- Autorización separada de entrenamiento/GPU, dispositivo y recursos libres.
  No tocar autorun, colas ni procesos ajenos para obtenerlos.

## Tres variantes emparejadas

1. Baseline `E3nnLinearEdgeBlock`, readout actual.
2. Ablación 7.1 `E3nnLowRankEdgeBlock`, rank=4.
3. 7.4 `E3nnSlaterKosterEdgeBlock`, rank=4, SK + mismo residual 7.1.

Semillas **0/1/2** emparejadas: mismas particiones, orden de muestras y pesos
iniciales de backbone/onsite por carga estricta, no solo igual seed. Conservar
los estados iniciales, configs resueltas, versiones y SHAs. Las arquitecturas
tienen distinta capacidad: no son modelos matemáticamente equivalentes.
Usar errores por semilla/material, no seleccionar la mejor semilla.

## Métricas y decisión predefinida

Primaria: **MAE de H alineada al gauge en test**, misma definición/pesos por
elemento entre variantes. Registrar también H raw, error por bloque/especie y
distancia, bandas/gap y OOD cuando haya referencias. Gauge del evaluador debe
estar fijado antes de test y no esconder errores espectrales.

Para cada candidato y su baseline, calcular razones pareadas por semilla y
material y su media geométrica. Promoción exige razón primaria **<1**, evidencia
de incertidumbre compatible con mejora y **ninguna degradación OOD pareada >5 %**.
Reportar todas las parejas, dispersión e intervalos con método fijado antes de
medir. Tres semillas no garantizan potencia estadística: si intervalos incluyen
ausencia de mejora, resultado **inconcluso**, no promoción. Comparar también
7.4 contra 7.1 para atribuir el coste/beneficio de SK. No ocultar peores bandas
o gaps bajo una MAE agregada menor. Si baseline primario es cero, la razón no
se define: informar diferencias absolutas y no aplicar promoción automática.

## Rendimiento GPU/VRAM (pendiente)

Separar dos experimentos: **batch fijo idéntico** y **máximo batch admisible**.
Mantener dtype y trabajo, fixtures/split congelados, versiones CUDA/Torch/e3nn/
MACE/driver, GPU, hilos, entorno y carga concurrente registrados. Comparar primero
float32; AMP solo tras autorización y validación numérica propias. No asumir
compatibilidad de esta variante con compile, checkpointing o AMP por herencia.
Sus restricciones existentes siguen vigentes; no están certificadas aquí.

Por variante: construcción fuera de medida, warmup hasta estado estable,
inicialización Adam, sincronización antes/después de forward, backward y paso
completo; medir fases separadas y conservar muestras crudas, mediana y MAD.
Rotar el orden de las tres variantes y alternar reverso. Documentar si incluye
copias, loss, zero_grad y actualización de pesos. Comprobar loss/gradientes
finitos fuera de timing y que ambas ramas de 7.4 aportan.

Para memoria GPU, proceso fresco por variante/alcance, sincronizar y resetear
picos después del warmup. Registrar por separado peak **allocated** y
**reserved**, con el suelo de parámetros/optimizador y el batch efectivo.
No inferir VRAM desde RSS host ni desde bytes de parámetros. Buscar máximo
batch con política de OOM explícita y procesos propios aislados; distinguir
throughput de latencia por paso. No cambiar batch para favorecer una variante
en la comparación a batch fijo.

## Estado de esta entrega

Preparado: este protocolo. Ejecutado: únicamente pruebas/benchmark CPU
sintéticos descritos en `informe_slater_koster_cpu.md`. Pendiente: dataset/splits
experimentales, precisión científica, bandas/gap, OOD y GPU/VRAM. Ni 7.4 global
ni Task 4 completo se cierran con el cierre del goal CPU.
