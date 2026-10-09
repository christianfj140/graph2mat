# 7.1 — Evaluación GPU y precisión aplazada

Estado: **protocolo preparado, no ejecutado**. El usuario aplazó expresamente
GPU; no lanzar estos experimentos como parte del goal de implementación CPU.
Necesita autorización posterior y GPU sin trabajos ajenos. No detener procesos.

## Candidato y referencia congelados

- Referencia: `graph2mat.bindings.e3nn.E3nnLinearEdgeBlock`.
- Candidato: `graph2mat.bindings.e3nn.E3nnLowRankEdgeBlock`, `rank=4`.
- Una primera comparación, no búsqueda retrospectiva de rank en el test.
  Si se quiere estudiar otro rank, será un experimento separado preregistrado.
- Mismo commit, entorno, dtype, backbone, readout de nodos, número de
  interacciones, normalización, pérdida, batch y datos. Solo cambia el readout
  de aristas. Desactivar compile/checkpointing en ambas primeras sondas;
  estudiar sus combinaciones después, separadas de este efecto principal.
- Guardar versión Python/Torch/CUDA/e3nn/MACE/Lightning, driver/GPU, commit y
  estado dirty, hashes de configuración/datos/splits y pesos iniciales.
- Copiar a un directorio experimental propio los manifiestos de datos/splits
  existentes, sin modificar datasets ni colas. No usar el test para seleccionar
  la configuración. Inventariar los datos antes de generar comandos concretos.

## Velocidad y VRAM: procedimiento

1. Comprobar `nvidia-smi` y los procesos antes de cada tanda; si hay cómputo
   ajeno, posponer, no ocupar la GPU. Registrar hora, utilización, temperatura
   y clocks al inicio/final de cada ronda; descartar y repetir una pareja si
   aparece carga concurrente o throttling asimétrico.
2. Primera sonda: fixtures sintéticas congeladas de
   `bench_readout_rango_bajo.py`, 16 y 72 puntos, C2, fp32, rank 4. Después:
   un batch real congelado de `w90` y otro de `6x6` con sus recetas iguales.
   No extrapolar una forma/tamaño a todas las bases.
3. Ejecutar cada arquitectura en un proceso propio, secuencialmente: solo
   ella, su batch y su optimizador residen en CUDA. Mantener ambos modelos en
   CUDA contaminaría el pico de VRAM de uno con el otro. Fijar la semilla y
   copiar pesos backbone/on-site de referencia al candidato cuando las formas
   coincidan; no pretender cargar los pesos de un TP completo en el factorizado.
4. Separar inicialización/transferencia/primer paso (incluido Adam) del steady
   state. Warmup mínimo 10 pasos, comprobar estabilidad; 12 rondas pareadas
   alternando referencia→candidato y candidato→referencia. Cada ronda contiene
   30 pasos con la misma secuencia de batches y el mismo número de updates.
5. Medir carga/collate, H2D, forward, backward, optimizador, paso completo e
   inferencia por separado. CUDA events para tramos GPU; reloj de pared y
   sincronización antes/después de la ventana completa para throughput.
   No introducir `.item()`/copias CPU ni sincronizaciones dentro del trabajo
   temporizado. Forward+backward sueltos no sustituyen la medida del paso.
6. Tras warmup, `torch.cuda.synchronize()` y
   `torch.cuda.reset_peak_memory_stats()`. Registrar memoria inicial y picos
   absolutos `max_memory_allocated()` / `max_memory_reserved()` en bytes/MiB;
   incluir modelo, gradientes y estados de Adam. No llamar `empty_cache()` entre
   pasos; usar un proceso nuevo para cada arquitectura/ronda y reportar también
   el coste del primer paso. Esos picos son del proceso, no `nvidia-smi` global.
7. Reportar mediana, min/max y MAD de ms/paso, estructuras/s, aristas/s y picos;
   razón pareada por ronda. Métrica primaria de rendimiento: tiempo de pared
   steady-state por paso completo; secundaria: pico allocated. Una diferencia
   comparable a la dispersión se etiqueta inconclusa. Más parámetros pequeños
   o más kernels pueden anular el ahorro del TP; no ocultar regresiones.
8. GPU/AMP: antes de medir bf16/AMP, verificar contra fp32 de **la misma
   arquitectura** salidas, pérdida y gradientes finitos. La igualdad entre
   arquitecturas distintas no es un criterio válido. Definir tolerancias con
   el ruido AMP de la referencia antes de comparar; registrar fallback si una
   operación no es compatible. No declarar AMP validado por una prueba CPU.

## Precisión: comparación separada

- Usar primero `6x6` con su receta de referencia y splits congelados, y sus
  evaluaciones OOD existentes de bicapa/moiré. Mantener Δ/H0, gauge, S auxiliar,
  dtype y selección de checkpoint como estén definidos en esa receta; no
  combinar 7.1 con cambios nuevos de pérdida o de datos.
- Semillas pareadas **0, 1, 2**, mismo presupuesto de updates, optimizador,
  schedule, batches, validaciones y criterio de checkpoint. No igualar tiempo
  de pared sacrificando pasos en una variante. Primero smoke propio corto en
  CPU/GPU autorizada; después los seis entrenamientos separados y completos.
- Métrica primaria: H-MAE alineado de gauge en test ID, en meV, cuando la
  receta usa gauge. Secundarias: H bruto, bandas/autovalores alineados y OOD,
  desglose onsite/intercapa/especie/l, tiempo hasta precisión objetivo y fallos.
  Un material/objetivo sin gauge usa su H-MAE físico acordado, no una métrica
  inventada después de ver resultados. No añadir etiquetas DM/EDM inexistentes.
- Fijar **antes del candidato** el margen de no regresión ID: mayor de 5 % de
  la media de referencia y dos desviaciones estándar de esa referencia entre
  las tres semillas. Conservar los valores por semilla y diferencias pareadas;
  tres semillas son una primera señal, no evidencia de universalidad.
- Bandas OOD tienen veto a recomendar un cambio de default si empeoran fuera
  de la variación de referencia; calcular de la misma forma su margen antes
  de entrenar el candidato. Mejor H-ID no demuestra mejores bandas-OOD.
- Una reducción de VRAM/tiempo con pérdida de precisión se presenta como
  tradeoff. No cambiar defaults por una sonda sintética o un único material.
- Extender a una base heteronuclear con d/f cuando haya datos propios adecuados;
  las pruebas de simetría s/p/d/f no sustituyen esa evaluación científica.

## Configuración y siguiente procedimiento

Uso Python (la selección existente permanece igual por defecto):

```python
from graph2mat.bindings.e3nn import E3nnLowRankEdgeBlock
from graph2mat.tools.lightning.models.mace import LitMACEMatrixModel

model = LitMACEMatrixModel(
    basis_table=table,
    # Resto de kwargs copiados sin cambios de la receta de referencia.
    edge_block_readout=E3nnLowRankEdgeBlock,
    edge_block_readout_kwargs={"rank": 4},
)
```

En `MatrixMACE`/`E3nnGraph2Mat` directamente: `edge_operation=E3nnLowRankEdgeBlock`
y `edge_operation_kwargs={"rank": 4}`. Se necesitan `edge_feats_irreps` y
features radiales escalares 0e; el modelo MACE estándar ya las proporciona.
Mantener `hamiltonian_local_context=False`: MatrixMACE rechaza explícitamente
esa combinación con rango bajo porque el contexto no se anula en el corte.

Siguiente paso autorizado **CPU** y reproducible, no prueba GPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/tmp/g2m-python312-full.LHyni1/venv/bin/python \
  auditorias_mejoras/bench_readout_rango_bajo.py --points 72 --rank 4 --rounds 12
```

Para GPU, después de nueva autorización: inventariar/configurar el entorno CUDA
y los batches reales, implementar el runner con aislamiento por proceso descrito
arriba, ejecutar primero la sonda y luego el protocolo científico. El intérprete
CPU temporal anterior no se presenta como un entorno CUDA ni como permanente.
