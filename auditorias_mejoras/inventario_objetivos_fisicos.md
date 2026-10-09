# Inventario de etiquetas para 4.4–4.5

Fecha: 2026-10-09. Checkout inspeccionado: `g2m-v2`, HEAD `b5a4d26`.
Estado: evidencia preliminar de datos y convenciones; **no implementación,
no diseño aprobado, no cierre de 4.4–4.5**.

## Alcance y autorización

El objetivo inicialmente limitaba la búsqueda al repositorio Graph2Mat. El
usuario indicó después que los materiales están en
`/home/christian/repositorios/MD_vs_AtomicDisplacement`. Se inspeccionaron allí
los datasets en lectura sola. No se modificaron datos, manifiestos, entornos,
colas, entrenamientos ni código de ese repositorio. No se ejecutaron fonones,
GPU, SIESTA ni entrenamientos.

El checkout de Graph2Mat conserva un config H-only archivado en
`Comparison/results/diagnostics/h2o_hamiltonian/edge_rep_fix_rerun/workspace_view`.
Sus directorios de muestras y base contienen enlaces al repositorio indicado
por el usuario: no constituían prueba de ausencia de etiquetas. La búsqueda
acotada siguiente localizó conjuntos persistentes con etiquetas adicionales.

## Datos localizados y profundidad de comprobación

Raíz común:
`/home/christian/repositorios/MD_vs_AtomicDisplacement/Comparison/datasets`.

| Conjunto, relativo a la raíz | Snapshots numéricos en `MD_steps` | Tripletes TSHS/TSDE/HSX presentes | Manifiesto `benchmark_ready` |
| --- | ---: | ---: | --- |
| `graphene_w90_joint/graphene_w90_phase1_iid1500` | 1500 | 1500 | true |
| `bilayer_graphene_AA_md30/bilayer_graphene_aa_md30` | 30 | 30 | false |
| `graphene_hBN_AA_md30/graphene_hbn_aa_md30` | 30 | 30 | true |

Se contaron entradas y presencia de archivos en esos tres conjuntos, no todos
los materiales del repositorio. Se leyeron y comprobaron numéricamente las
matrices de **`MD_steps/0` de cada conjunto**, no los 1560 snapshots completos.
Se contrastaron los nueve SHA-256 de TSHS/TSDE/HSX de esas tres muestras con
sus respectivos `metadata.json`: los nueve coinciden. Esto no sustituye una
validación exhaustiva de integridad, geometrías, SCF o independencia temporal.

La bicapa AA tiene `validation_status="invalid"` y el error
`electronic_convergence.json is missing`. Su validación de artefactos sí pasa.
Sus matrices sirven como diagnóstico de lectura/convenciones, pero no se
presentan como dataset científicamente habilitado para un benchmark. Recurso
concreto pendiente: informe de convergencia electrónica para esa configuración.
No se generó ni reparó dicho informe.

## Etiquetas presentes, ausentes y no verificadas

- **H/S:** presentes en TSHS/HSX. La lectura numérica usó expresamente TSHS;
  no se compararon aquí todos los valores contra HSX.
- **DM/EDM:** ambas legibles desde TSDE usando sisl 0.16.4. La lectura adjuntó
  la geometría y el overlap reales leídos del mismo snapshot. No se obtuvieron
  etiquetas de una diagonalización de la predicción.
- **Overlap de TSDE:** no está almacenado en ese formato. El lector instalado
  deja esa componente a cero si no se proporciona `overlap`; esos ceros no son
  etiquetas físicas y no pueden usarse para las pérdidas nuevas.
- **Spin:** las tres muestras son no polarizadas, una componente física más S.
  La DM ya contiene la ocupación total de ambos spins: no se multiplica otra
  vez por dos al contraerla con S. No se validó un dataset colineal o SOC real.
- **Unidades:** H y EDM leídas por sisl están en eV; S es adimensional. Las
  contracciones DM–S producen población electrónica y H–DM/EDM–S producen eV.
  No se extrapola esta convención a otros formatos/lectores sin comprobarlos.
- **Electrones:** 8, 16 y 16, respectivamente, por la DM de referencia; hay
  números coincidentes en las salidas SIESTA (para hBN, en `RUN.out` del
  conjunto). No se sustituye el número real por `geometry.q0` sin validación.
- **Periodicidad:** celda y conexiones de imágenes presentes; los `nsc` de
  estas muestras son [5,5,1], [5,7,1] y [7,7,1].
- **Puntos k:** los FDF declaran Monkhorst–Pack 20×20×1 sin desplazamiento.
  La comprobación independiente usó los 400 puntos completos, cada peso
  1/400; las salidas de grafeno/bicapa informan 211 puntos reducidos por TRS.
  No se cargó ni verificó una tabla explícita de pesos irreducibles archivada.
- **Ocupaciones por estado:** no se localizó un archivo de ocupaciones en las
  tres carpetas de muestra. Sí consta `ElectronicTemperature 0.075 eV`.
  Una comprobación diagnóstica con H/S de referencia y Fermi–Dirac confirma
  ocupaciones fraccionarias; no se convierte esa reconstrucción en etiqueta
  inventada ni en una justificación para imponer idempotencia.
- **Gauge:** las muestras leídas tienen su Fermi en cero. Sus TSHS y TSDE
  guardan niveles de Fermi originales iguales entre sí: −5.699742409836231,
  −5.408602692866281 y −5.538212533415436 eV. Usarlos otra vez como potencial
  químico sobre H ya desplazada da números incorrectos de electrones.

Los lectores y las convenciones de Mulliken se contrastaron con su código
instalado y las referencias primarias de
[TSDE](https://sisl.readthedocs.io/en/latest/api/io/generated/sisl.io.siesta.tsdeSileSiesta.html),
[DensityMatrix](https://sisl.readthedocs.io/en/v0.16.0/api/generated/sisl.physics.DensityMatrix.html)
y [EnergyDensityMatrix](https://sisl.readthedocs.io/en/v0.16.0/api/generated/sisl.physics.EnergyDensityMatrix.html).

## Contrastes físicos ejecutados

Comprobaciones CPU, Python 3.12.3, sisl 0.16.4, NumPy/SciPy; sin utilizar una
implementación nueva de pérdidas de Graph2Mat. Se leyeron matrices mediante
`fdf.read_hamiltonian(order=['TSHS'])`, `read_overlap(order=['TSHS'])` y los
lectores DM/EDM de TSDE con `geometry` y `overlap` explícitos.

Para las matrices reales y hermíticas de estas muestras:

```
N = sum_(i,j,R) D_ij(R) S_ji(-R) = sum_(i,j,R) D_ij(R) S_ij(R)
p_A = sum_(i in A,j,R) D_ij(R) S_ij(R)
e_A = sum_(i in A,j,R) H_ij(R) D_ij(R)
     = sum_(i in A,j,R) S_ij(R) E_ij(R)
```

Aquí `E` es EDM, no un autovalor. La segunda forma del conteo usa realidad y
hermiticidad; no autoriza extender un producto sin conjugación a SOC complejo.
Las sumas recorren las imágenes originales, antes de plegar a Γ. En grafos
simétricos con solo media arista habrá que redistribuir ambos extremos; no se
ha implementado todavía esa adaptación.

| Diagnóstico de `MD_steps/0` | Grafeno | Bicapa AA | Grafeno–hBN AA |
| --- | ---: | ---: | ---: |
| Átomos / orbitales | 2 / 8 | 4 / 16 | 4 / 34 |
| N por suma directa | 7.999999999999995 | 16.000000000000020 | 16.000000000000092 |
| N por integral discreta de Tr[D(k)S(k)] | 7.999999999999990 | 16.000000000000020 | 16.000000000000103 |
| Tr[D(Γ)S(Γ)], **no es el conteo periódico** | 7.998131894950259 | 16.028686600670902 | 16.584079804970970 |
| Tr(HD), eV por celda | −71.73586988074231 | −147.8522206328203 | −135.7351877817243 |
| Tr(SE), eV por celda | −71.73586988071018 | −147.85222063292716 | −135.73518778069158 |
| Máximo residuo energético por orbital, eV | 8.97e−10 | 9.22e−10 | 3.21e−9 |
| Estados fraccionarios en la malla completa, tolerancia 1e−6 | 48 | 60 | 102 |

Se construyeron D(k) y S(k) independientemente mediante las matrices por imagen
y `exp(2πi k·R)`, no mediante el ensamblado Γ de Graph2Mat. La integral coincide
con la suma directa; sustituirla por Γ no es físicamente correcto.

Se aplicaron además transformaciones ortogonales aleatorias por átomo
(NumPy `default_rng(43)`, QR). Las poblaciones por orbital cambiaron hasta
0.0460, 0.00520 y 0.7529 electrones, respectivamente. Las poblaciones por
átomo se conservaron con residuo máximo 1.78e−15, y las energías por átomo
con residuo máximo 2.85e−14 eV. Por tanto, un objetivo escalar de población o
energía debe agrupar por átomo: no penalizar arbitrariamente componentes
orbitales dependientes de la orientación de la base.

Se comprobó también el cambio común de referencia energética
`H -> H + 0.37 S`, `E -> E + 0.37 D`: conserva el residuo energético a la
precisión anterior. Desplazar H dejando EDM fija no es la misma operación.

S(k) es definida positiva en los dos puntos examinados, Γ y
(0.173,0.287,0), en las tres muestras; el menor autovalor observado en hBN
es 5.15e−4 en Γ. **No es una certificación de buen condicionamiento en todo
el dataset ni en toda la zona de Brillouin.** No se aplicó regularización.

## Aplicabilidad, sin cerrar objetivos

- N/Mulliken: las etiquetas permiten contrastar una **DM predicha** con la
  referencia, usando S real. No aportan esa señal a un modelo que solo predice
  H si el número de electrones se fija automáticamente al diagonalizarlo.
- Consistencia H–DM–EDM: hay referencias independientes para estudiar un
  término sobre H o EDM predicha, manteniendo las demás matrices de referencia
  fijas. Debe respetar el gauge elegido y la agrupación atómica.
- McWeeny/idempotencia: **no aplicable a estos datos como etiqueta exacta**,
  dadas las ocupaciones fraccionarias confirmadas. Tampoco puede imponerse a
  una DM(k) reconstruida de coeficientes espacialmente truncados sin demostrar
  antes las hipótesis necesarias.
- La pérdida espectral de primer orden ya existente no se rehizo ni reactivó.
- La integración de etiquetas auxiliares, máscaras, restauración del residuo
  atómico, batching, gradientes, checkpoints y normalización sigue pendiente.
  El registro de matrices aún trata EDM mediante el fallback genérico.
- No se han medido costes de pérdidas nuevas, precisión, velocidad ni VRAM.
  Los resultados anteriores prueban convenciones en muestras, no mejoras ML.

## Referencias preservadas

SHA-256 de `frozen_split_manifest.json`, sin reescribirlos:

```
graphene: 93d436f847680fe88b0543ef2420845194ef916e20b682dfe651132f93731d3b
bilayer_AA: a3c4e7e6f40a359e7c577401391c40d65d6b8c202b6860711433797528457ff2
graphene_hBN_AA: 6e03f23c49a15762a644534c16a5efc6394a9955e161bae3ff65d49c3ca7fa19
```

Sus conteos train/validation/test declarados son 1198/150/150, 22/3/3 y
22/3/3. Son menores que el total localizado: no se redistribuyeron los
snapshots restantes ni se supuso por qué quedaron fuera.

## Próxima decisión y límites de esta entrega

Está pendiente la aprobación conversacional del alcance propuesto, después
la revisión del diseño escrito y del plan antes de tocar producto, según la
guía de diseño activa. El goal sigue abierto. Este inventario no sustituye
RED→GREEN, smoke Lightning, suite completa, gates de tipos/Python 3.9,
benchmark CPU pareado ni protocolo experimental de tres semillas.
