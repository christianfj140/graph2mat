# Objetivos físicos 4.4–4.5: implementación y evidencia CPU

2026-10-09, rama `g2m-v2`, mismo checkout. **Cierre final en curso**: están
implementados y probados los términos; suite final/revisión independiente aún
deben terminar. No se declara cerrada toda 4.4, 4.5 experimental ni Ola 4.

## Implementado y límites científicos

`elementwise_mse` y `gauge_invariant_mse` admiten `electron_weight`,
`mulliken_weight`, `energy_weight`, todos cero por defecto. Dataset/DataModule
aceptan `physical_labels=("charge",)` y/o `("energy",)`; `None` preserva la ruta
legacy, sin lecturas auxiliares. Un objetivo activo necesita referencias
preparadas, no arrays arbitrarios ni etiquetas inventadas.

- Carga: DM predicha representada en G, D0 restaurada cuando corresponde,
  contra poblaciones completas de referencia; N normalizada por átomos antes
  de elevar al cuadrado, Mulliken agrupada por átomo.
- Energía: perturbación de H, DM o EDM sobre referencia completa, conservada
  fuera de G. **No es la energía total exacta de la matriz truncada del writer**.
  EDM lleva signo negativo; H gauge usa exactamente el delta alineado de la
  pérdida base, sin recalcular c.
- Contracciones por imagen opuesta real, sumando canales spin sin factor 2
  adicional. Media arista aporta a ambos extremos, incluida autoimagen.
  No se ensamblan matrices densas ni se diagonaliza durante la pérdida.
- Preparación valida S(k), base, geometría, spin, malla y gauge. FDF opt-in:
  solo malla MP explícita, diagonal, positiva y sin offset. Otras mallas:
  API directa con k/pesos explícitos. ZIP físico no certificado; ZIP legacy
  permanece soportado. Predicción sin labels no lee auxiliares.
- Unknown DM/EDM no equivale a cero observado; solo se excluye cuando la
  contraparte es exactamente cero, antes de multiplicar. H-energy exige DM
  observada en toda fila donde H predicha puede variar.
- H/S archivadas tienen asimetría numérica acotada: `1e-8*max(1,maxabs)`.
  No se simetrizan ni alteran coeficientes. Partners forward/reverse separados
  conservan la orientación real, a coste de memoria auxiliar. S se certifica
  tras esa cota; positividad sobre parte hermítica y λmin mayor que cota de
  error antisimétrico, λmin/λmax>1e−12. Sin clipping/regularización.
- McWeeny/idempotencia exacta no aplica a ocupaciones fraccionarias. SOC,
  no colineales, complejos y baseline DM colineal se rechazan explícitamente.

Lightning rechaza pesos/configuraciones incompatibles en construcción y datos
incompatibles con cualquier etapa física antes de forward/primer optimizer
step. Física fuerza ruta matricial; pesos cero conservan certificado por
coeficientes. Epoch/DDP acumulan cada media con su denominador: elementos para
base, estructuras para física y SSE/count para RMSE de ΔN. Checkpoints anteriores
sin kwargs conservan parámetros y predicciones. Auxiliares no entran en features.

El preflight usa `_data_source` de Lightning antes de crear iteradores; si deja
de estar disponible, error explícito. Indexa todos los datos train/val usados,
incluidas futuras etapas; streams irreversibles rechazados. Snapshots cacheados
se reutilizan; on-demand paga una pasada CPU completa por rank y los pools
rotatorios validan sus fuentes completas. No oculta ese coste como cache gratis.

## Datos reales, solo lectura

Fuentes `MD_steps/0/RUN.fdf` de grafeno `graphene_w90_phase1_iid1500` y
grafeno–hBN `graphene_hbn_aa_md30`, bajo la raíz autorizada del
[inventario](inventario_objetivos_fisicos.md). SHA-256 FDF/TSHS/TSDE/HSX
antes/después iguales. TSDE no contiene geometría/S independientes; se suministra
la base FDF/TSHS y se lee sin inyectar S para preservar soporte CSR observado.
No se certifican 1560 snapshots ni convergencia electrónica por leer matrices.

| Muestra | N completo | Carga N/M sobre DM | Energy sobre H / DM / EDM |
| --- | ---: | --- | --- |
| Grafeno | 8 | dos smokes válidos | tres smokes válidos |
| Grafeno–hBN | 16 | dos smokes válidos | **H bloqueado** / DM válido / EDM válido |

Los nueve casos válidos tienen término físico inicial positivo, gradiente de
`total-base` finito/no nulo y dos pasos Lightning CPU, tanto control como
candidato desde idénticos pesos iniciales/seed13. Gradiente L1 de carga:
grafeno 104.016; hBN 71.299 (N) y 73.949 (M). Energy: grafeno H 640.705,
DM 5289.397, EDM 902.454; hBN DM 34641.289 y EDM 527.485. Son diagnósticos,
no ganancias de precisión; el modelo diminuto no cubre todo el span high-l.

Oráculo independiente une entradas sparse por imagen opuesta antes de agrupar
átomos; contrasta población con `sisl.DensityMatrix.mulliken("atom")`, identidad
H–DM/EDM–S y observable G reconstruido. Máximo residuo atómico full ≈1.61e−11
eV en grafeno y 5.78e−10 eV en hBN. Cola de población fuera de G: ≤1.78e−15
en estas dos muestras; no demuestra que sea cero en otros grafos. Los tests
analíticos incluyen explícitamente truncamiento no nulo.

**Recurso concreto para H-energy en hBN/0:** DM observada en las filas del grafo
actualmente ausentes del TSDE; haría falta una referencia con ese soporte, o
otro grafo certificado con alcance científico justificado. No rellenar a cero.
Bicapa AA permanece excluida de benchmark por faltar `electronic_convergence.json`.

## Costes CPU pareados

Intel Core Ultra 9 285K, float64, dos threads Torch/OMP/OpenBLAS/MKL, seed13,
3 warmups y 12 rondas alternadas AB/BA. Etiquetas exactamente iguales (incluidos
NaN), arquitectura/pesos/predicciones/estado inicial iguales. Control matricial,
no coefficient-only. Los controles de carga seleccionan los mismos canales
físicos originales, leyendo S para geometría pero sin certificación auxiliar.

| Muestra / término | `loss+backward` control → física, ms | `Trainer.fit` un paso control → física, ms | Bytes auxiliares muestra / batch1 |
| --- | --- | --- | --- |
| Grafeno N | 0.111 → 0.543 | 17.820 → 22.215 | 25648 / 33608 |
| Grafeno M | 0.113 → 0.556 | 17.910 → 22.107 | 25648 / 33608 |
| Grafeno energy DM | 0.114 → 0.539 | 19.202 → 22.513 | 34576 / 42536 |
| Grafeno energy H | 0.113 → 0.534 | 19.135 → 22.574 | 34320 / 42280 |
| Grafeno energy EDM | 0.113 → 0.529 | 18.983 → 22.658 | 34320 / 42280 |
| hBN N | 0.210 → 0.931 | 30.146 → 35.885 | 281439 / 450407 |
| hBN M | 0.202 → 0.911 | 31.449 → 36.559 | 281439 / 450407 |
| hBN energy DM | 0.210 → 0.915 | 29.541 → 34.625 | 471483 / 640451 |
| hBN energy EDM | 0.235 → 0.997 | 29.753 → 35.079 | 468523 / 637491 |

Medianas: un paso `Trainer.fit` cuesta **+16.3 % a +24.7 %** en estos smokes,
incluyendo setup/logging/optimizer/preflight preparado. No equivale a un paso
steady-state de entrenamiento grande. `loss+backward` aislado ≈4.2–4.9×
una base diminuta; incremento absoluto ≈0.4–0.7 ms. Carga/preparación/certificación:
grafeno ≈21–28 ms frente a ≈7–8 ms, hBN ≈82–105 ms frente a ≈19–20 ms.

[JSON completo](resultados_objetivos_fisicos_cpu.json) incluye cada muestra
cruda, mediana/MAD, primera invocación, cached-access, collate, hashes de
scripts/fuentes/Git, versiones, CPU/threads y carga del host. Carga media host
≈5.5–9.1, smokes/benchmarks CPU concurrentes: resultados locales, no entorno dedicado.
Primera invocación no vacía caches del sistema operativo. Aux bytes incluyen
tensores, layout y payload k, no overhead Python. RSS pico del proceso completo:
grafeno 752516 KiB, hBN 843620 KiB, **no asignación pareada, no VRAM**.
No se midió GPU ni se demuestra ahorro de tiempo/memoria con estas pérdidas.
Estas cifras sustituyen las mediciones anteriores a la corrección del preflight.
Se repitieron ambos smokes y benchmarks sobre producto `371f5a2`, exit0 los
cuatro. `git_dirty=true` corresponde únicamente a documentación tracked en
edición (diff SHA-256 `34b8b2dfc91badee708ae58a32a570972f8c57689a6e1672ab34cac15d8e1edf`);
`git diff HEAD -- src tests` y ambos scripts estaba vacío antes de medir.

## Verificación y reproducción

Todos los comandos usan `CUDA_VISIBLE_DEVICES=''`, `PYTHONPATH` absoluto al
`src` de Graph2Mat y `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=2`.
Python CPU existente 3.12.3/Torch2.11 y piso 3.9.25/Torch2.4.1; sin instalar.

- Task1: 2277 tests de src +29 de raíz pasan, 2 omisiones conocidas.
- Task2: 2348 pasan, 2 omisiones, suite completa src+tests.
- Task3 (`78606d5`): **2418 pasan, 2 omisiones**, 683.38 s, exit0.
- Task4 regresiones ampliadas: **234 pasan**, 92.60 s, exit0; piso3.9 pérdidas
  +integración **118 pasan**, 20.32 s, exit0. Full gate Task4: **2466 pasan,
  2 omitidos**, 710.02 s, exit0 (producto `8cfbc85`, anterior a fix final).
- Gate físico posterior al fix `371f5a2`: **226 pasan** en 3.12 (25.52 s)
  y **226 en 3.9** (26.44 s),
  incluyendo herramientas CPU; exit0 ambos. Ruff `src tests` y ambos scripts,
  mypy existente (23 archivos) y compileall3.9 pasan, exit0.
- Smokes de ambos datasets y ambos benchmarks: exit0. Autoverificación literal
  7/300 y 1/800; 5 tests de contratos de scripts pasan. Fallos RED y correcciones
  registrados: máscaras gauge in-place, certificado, epoch denominators,
  preflight futuro, checkpoint sin DataModule, peso fraccionario y recurso ausente.

Ejecutar `smoke_objetivos_fisicos.py --fdf PATH --output NUEVO.json` y
`bench_objetivos_fisicos.py --fdf PATH --warmup 3 --rounds 12 --output NUEVO.json`.
Ambos rechazan output existente/dentro del dataset; salir no cero ante un fallo
inesperado. H-energy unknown se informa bloqueado, no como prueba válida.

Commits propios de producto: `5215fe7`, `1719d6a`, `ae04e79`, `78606d5`,
`8cfbc85`, `371f5a2`; scripts/evidencia `ba9ea01`. Cambios previos/ajenos conservados; sin
push, merge, fonones, 7.4, autorun ni modificación de archivos científicos.

## Pendiente experimental

[Protocolo de tres semillas pareadas](protocolo_precision_4_4_4_5.md): control,
N, Mulliken y energy por separado, splits congelados, presupuesto igual,
matrices raw/alineadas, bandas, carga, observables writer y OOD. No promover un
término que empeore OOD aunque reduzca su propia pérdida. Defaults siguen cero.

## Revisión independiente y corrección final

Una revisión fresca `gpt-6-astra` (high), Native con un único revisor, examinó
el rango `0b3ad55..ba9ea01`, código conectado, scripts, JSON y documentación.
Comprobó los cinco focos: malla FDF sin fallback Γ; unknown frente a cero;
etapas futuras; aristas dirigidas/medias/autoimágenes; agregación por estructura
invariante al reparto. Recalculó medianas/MAD/ratios y comprobó hashes de scripts.
No ejecutó por su cuenta la suite completa ni nuevas lecturas de datos ajenos.

Un hallazgo **Important**, ninguno Critical/Minor: el preflight pasaba etiquetas
como predicciones y no contrastaba el contrato del modelo ni kind/spin entre
muestras. Tres regresiones (`later_spin`, `later_kind`, `model_components`)
fallaron antes de la corrección (exit1, 3 failed, 2.62 s). Ahora se inspeccionan
los componentes declarados por los readouts/modelo, se validan formas de salida
sintéticas sin forward y kind/spin homogéneos en train/validation. Un modelo
personalizado opt-in debe declarar `n_matrix_components` en sí mismo o kwargs;
se rechaza si no puede inspeccionarse. Predicción H-only con S solo como etiqueta
y spin colineal válido conservan pruebas positivas. Defaults cero no cambian.

Revisor declinó juzgar precisión/bandas/OOD/convergencia, GPU/VRAM, elegibilidad
de todos los snapshots, recuperación de soporte DM hBN/geometría independiente
de TSDE y trabajo fuera de alcance. Se mantienen respectivamente como experimentos
pendientes, recursos ausentes o exclusiones explícitas; no como beneficios probados.
Menores aplazados: ninguno. Gate final de la corrección todavía en curso.
