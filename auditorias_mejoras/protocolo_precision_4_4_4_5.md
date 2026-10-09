# Protocolo experimental posterior: objetivos físicos 4.4–4.5

Estado: preparado, **no ejecutado**. La implementación y los smokes CPU no
demuestran mejor precisión, bandas, OOD, velocidad ni VRAM. Defaults apagados.
Este protocolo no autoriza lanzar GPU ni entrenamiento comparativo largo.

## Referencia y elegibilidad

Usar los manifiestos congelados originales; no reescribirlos ni reasignar frames
excluidos. Registrar listas ordenadas y SHA-256 de train/validation/test/OOD,
configuración, base, archivos científicos y revisión Git antes de entrenar.

| Dataset | SHA-256 de `frozen_split_manifest.json` | Elegibilidad |
| --- | --- | --- |
| Grafeno `graphene_w90_phase1_iid1500` | `93d436f847680fe88b0543ef2420845194ef916e20b682dfe651132f93731d3b` | Manifiesto ready; certificar cada muestra física antes del experimento |
| Grafeno–hBN `graphene_hbn_aa_md30` | `6e03f23c49a15762a644534c16a5efc6394a9955e161bae3ff65d49c3ca7fa19` | Manifiesto ready; H-energy bloqueado en muestra 0 por DM desconocida dentro de G |
| Bicapa `bilayer_graphene_aa_md30` | `a3c4e7e6f40a359e7c577401391c40d65d6b8c202b6860711433797528457ff2` | No-ready: falta `electronic_convergence.json`; solo diagnóstico, no benchmark |

Los smokes actuales cubren únicamente `MD_steps/0` de grafeno y grafeno–hBN.
No certifican todos los snapshots. Aplicar preflight completo a la selección
congelada, incluyendo futuras etapas y validación. Guardar rechazos por muestra,
sin rellenar unknown a cero, corregir S ni cambiar el gauge silenciosamente.
Si se excluyen muestras incompatibles, congelar una selección común nueva para
todos los brazos y documentar su sesgo; no seleccionar según resultados ML.

## Brazos, semillas y presupuesto

Por target aplicable, usar semillas pareadas **0, 1, 2**:

1. Control: pérdida matricial actual, tres pesos físicos cero.
2. N: `electron_weight>0`, solo DM; demás pesos físicos cero.
3. Mulliken: `mulliken_weight>0`, solo DM; demás pesos físicos cero.
4. Energy: `energy_weight>0`, H, DM o EDM con H/S/DM/EDM verificadas; demás cero.

H y EDM no reciben brazos de carga inventados. No combinar objetivos antes de
evaluarlos por separado. No añadir el término espectral descartado a estos
brazos: si se estudia otra hipótesis espectral, hacerlo en un diseño separado.
McWeeny/idempotencia exacta no aplica a las ocupaciones fraccionarias actuales.

Dentro de cada pareja: mismo checkpoint inicial por semilla, arquitectura/base,
componentes, dtype, datos, orden de batches, límites de coste, optimizer,
scheduler, clipping, número de actualizaciones y política de selección/stopping.
Elegir pesos solo con train/validation; no mirar test/OOD para afinarlos. Registrar
unidades y escala de cada peso: electrones²/átomo², electrones² y eV² por átomo
no son magnitudes intercambiables. `.3` en el smoke es un peso diagnóstico,
**no una receta recomendada**.

## Métricas y controles

Publicar por semilla, material, split y tamaño, junto con mediana y dispersión:

- MAE/RMSE matricial raw; H también alineada con el gauge definido y c registrado.
- Error total ΔN en electrones y error por átomo; poblaciones Mulliken atómicas.
- Residuo energético de perturbación, conservando la referencia fuera de G.
- Observables de la matriz realmente escrita: restaurar D0 si procede,
  reconstruir todas las imágenes representadas, adjuntar S real y contabilizar
  separadamente el truncamiento fuera de G. No llamarlos equivalentes al término
  energético de entrenamiento.
- Bandas en malla/camino k congelados, alineación común, error en ventana de E_F,
  estados ocupados/desocupados y gap cuando tenga sentido físico; no usar solo Γ
  para el grafeno primitivo ni atribuir al término energético garantía espectral.
- OOD con referencia real y base compatible: otro tamaño/celda, acoplamientos
  intercapas u otro material solo si las etiquetas y el modelo lo admiten.
- Controles por rotación de base atómica, permutación y cambio común H→H+aS,
  EDM→EDM+aDM; raw H y alineada tienen interpretaciones distintas.

Registrar error dentro/fuera de G, masks, spin y ocupaciones. No multiplicar DM
no polarizada otra vez por dos; sumar canales colineales explícitos. SOC,
no colinealidad, complejos y baseline DM colineal siguen rechazados.

## Coste y decisión

Medir tiempo de carga/preflight, entrenamiento y evaluación con trabajo igual;
bytes auxiliares, RSS y, únicamente cuando se autorice GPU, pico VRAM mediante
medida CUDA real y warmup independiente. No inferir VRAM desde RSS ni comparar
coefficient-only con ruta matricial para atribuir coste al término físico.

No recomendar/promover un término si empeora OOD aunque baje su propia pérdida.
Conservar todos los controles y las tres semillas, no solo la mejor corrida.
Un resultado inconcluso mantiene defaults apagados. El siguiente paso autorizado
ahora es preparar/revisar esta evaluación; su ejecución larga requiere decisión
posterior del usuario y recursos/datasets elegibles.
