# 6.7 — Una orientación mediante proyección de pesos

Estado: **diseño aprobado por el usuario el 2026-10-14**, mediante «Apruebo»
en respuesta a la solicitud de revisión. La aprobación autoriza redactar el
plan, no ejecutar un plan aún no revisado. No se ha escrito código de producto.

Repositorio: `<graph2mat>`, rama `g2m-v2`.
Referencia de coste: [perfil por pares y backward](../../../auditorias_mejoras/informe_simetria_homonuclear_20261014.md).

## 1. Objetivo y límites

Evaluar una sola orientación de cada bloque que hoy usa `symm_transpose`,
conservando la función de `E3nnLinearEdgeBlock`, los gradientes respecto a sus
parámetros originales y las entradas, y la representación de checkpoints.
No reducir la capacidad, cambiar las irreps ni crear una parametrización nueva.

La operación homonuclear ocupa el 19,56%/21,65% de la suma de tiempos de kernels
en la fixture d/f de 16/72 puntos. No es porcentaje de pared ni ahorro demostrado.
Las dos orientaciones ya se apilan en una llamada: aquí se busca reducir sus
filas de 2N a N, no prometer la eliminación de la mitad de los lanzamientos.

Fuera de alcance: defaults, colas, campañas científicas, datasets/splits,
readouts low-rank/SK/condicionados, kernels CUDA nuevos, dependencias nuevas,
compresión de parámetros y soporte especulativo de cualquier TensorProduct.

## 2. Alternativas y elección

1. **Proyección diferenciable de los pesos existentes, recomendada.** Reutiliza
   los TP y Linear actuales con su argumento de pesos explícitos. Conserva
   parámetros, nombres, formas y optimizador; añade trabajo de proyección.
2. Parametrizar solamente pesos simétricos/antisimétricos independientes.
   Puede reducir parámetros, pero obliga a transformar checkpoints y estados
   de Adam y puede cambiar la optimización. Se descarta para esta tarea.
3. Optimizar solamente la evaluación apilada actual. Ya hay una llamada por
   par; no elimina por sí mismo la evaluación de ambas orientaciones.

No se construye otro TP con irreps simplificadas ni instrucciones reordenadas:
eso podría cambiar sus normalizaciones y el significado de pesos guardados.

## 3. Contrato matemático

### 3.1 Función actual y transpuesta de coeficientes

Para un origen de features, sean `a,b` los extremos, `W` los pesos bilineales
y `U` los lineales. Sin `output_scaleshift`:

\[
F_{W,U}(a,b)=\operatorname{TP}_{W}(a,b)+L_U(a+b).
\]

Hay una suma de esta expresión por origen (`node_feats` y/o `edge_messages`).
La simetrización actual es, en el espacio de coeficientes:

\[
S_{W,U}(a,b)=\tfrac12\{F_{W,U}(a,b)+T F_{W,U}(b,a)\}.
\]

`T` es la permutación con signo ya certificada por los RTP. Si `Q_k` es la
matriz del coeficiente k, `Q_k^t=s_k Q_{\pi(k)}` y
`(Tc)_k=s_k c_{\pi(k)}`; `T²=I`. Para varias componentes, T actúa por separado
en cada componente, nunca intercambia H/S u otros canales físicos.

El certificado necesario es más fuerte que comprobar dimensiones: en cada
segmento de salida de e3nn, la permutación debe actuar solo sobre multiplicidades,
conservar el índice magnético m y tener un signo constante dentro de cada irrep.
No basta con que las bases tengan igual número de orbitales.

### 3.2 Transformación bilineal

Sea una instrucción `r=(i,j,o)` de conexión `uvw`, con pesos `W_r[u,v,w]` y
factor normalizado de camino `alpha_r`. Su instrucción inversa es
`r'=(j,i,o)`. La salida correspondiente a multiplicidad w se transpone a
`pi_o(w)`, con signo `s_o(w)`. El intercambio de las entradas del acoplamiento
angular aporta `epsilon=(-1)^(l_i+l_j-l_o)`.

Definir la transformación de pesos:

\[
(R W)_r[u,v,w]
=s_o(w)\,\epsilon\,
\frac{\alpha_{r'}}{\alpha_r}
W_{r'}[v,u,\pi_o(w)].
\]

Entonces, por intercambio del acoplamiento angular y cambio de índices:

\[
\operatorname{TP}_{RW}(a,b)=T\operatorname{TP}_W(b,a).
\]

**Primer alcance certificado:** instrucciones FullyConnected `uvw`, pesos
compartidos internos, entradas con las mismas irreps y factores de caminos
inversos iguales y no nulos. La razón alpha es entonces 1 y R es una permutación
con signo. Factores diferentes, caminos sin pareja o una transpuesta que salga
de su segmento de irreps no se aproximan: conservan la ruta original.

Con `R²=I`, definir `P(W)=(W+RW)/2`. Por linealidad respecto a W:

\[
\operatorname{TP}_{P(W)}(a,b)
=\tfrac12\{\operatorname{TP}_W(a,b)+T\operatorname{TP}_W(b,a)\}.
\]

### 3.3 Término lineal y resultado

Como `a+b=b+a`, basta proyectar la salida del operador lineal. Para cada
instrucción Linear `(i,o)`, `R_L(U)[u,w]=s_o(w)U[u,pi_o(w)]`; la permutación
permanece dentro del mismo segmento y conserva su normalización. Sin bias:

\[
L_{(U+R_LU)/2}(a+b)=\tfrac12\{L_U(a+b)+T L_U(a+b)\}.
\]

Por tanto, una evaluación de TP y Linear sobre N filas con sus pesos
proyectados produce exactamente S en aritmética real. Aplicar Q después
produce el mismo bloque orbital que el promedio original de 2N filas.
No debe ejecutarse además la simetrización exterior: sería redundante y
anularía el ahorro buscado.

### 3.4 Gradientes y optimizador

No se sustituye ni modifica in-place el Parameter W o U. Se calcula una
expresión diferenciable nueva en cada forward y se pasa como peso explícito
al módulo existente. Por la regla de la cadena:

\[
\nabla_W\mathcal L=P^t\nabla_{P(W)}\mathcal L,
\]

que coincide con derivar la expresión simetrizada original. Lo mismo vale
para U y para las entradas; aquí P es además autoadjunto por ser R una
permutación con signo involutiva. La identidad cubre cualquier objetivo
diferenciable aplicado a la misma salida, no solo MSE.

Se conservan parámetros redundantes, weight decay y estados históricos de
Adam. No usar `detach`, `.data`, `copy_` sobre pesos, ni caché de pesos
proyectados entre pasos. En coma flotante cambia el orden de operaciones:
se exige equivalencia con tolerancias, no identidad bit a bit ni trayectorias
numéricas idénticas después de entrenamientos largos.

## 4. Integración mínima propuesta

### 4.1 Selección y ubicación

Opción booleana `fuse_homonuclear=False` en `E3nnIrrepsMatrixBlock`, consumida
por el bloque y no reenviada indiscriminadamente a constructores de operaciones.
Se usa la ruta de configuración existente:

```yaml
model:
  edge_block_readout_kwargs:
    fuse_homonuclear: true
```

No añadir otro flag a Lightning/MACE ni exportar una nueva clase. El default
False conserva la ruta actual, incluidos checkpoints/configs sin la opción.

La operación elegible será **exactamente** `E3nnLinearEdgeBlock`, no cualquier
subclase que pueda haber sobreescrito el forward con un término no lineal.
El bloque prepara el certificado usando sus mapas de transpuesta; la operación
contiene el cálculo interno que evalúa sus TP/Linear con pesos proyectados.
La selección interna se pasa en la llamada normal del módulo (un keyword
privado, consumido antes de ordenar los pares), para conservar sus hooks de
forward. No invocar directamente un método alternativo saltándose
`Module.__call__`, ni mutar un flag compartido entre llamadas/batches.
No modificar `MatrixBlock` genérico, NumPy ni la selección de vecinos/aristas.

### 4.2 Un único camino certificado

`forward`, `forward_coefficients` y `forward_with_coefficients` deben consumir
el mismo cálculo de coeficientes fusionados. La primera reconstruye una vez;
la segunda no reconstruye; la tercera devuelve ambos sin recalcular TP.
No crear tres implementaciones independientes de la proyección.

La elegibilidad usa el `symm_transpose` real del bloque, no una comparación
de números atómicos. Los bloques heteronucleares/no simétricos siguen iguales.
Si una agrupación de bases produce bloques cuadrados simetrizados, solo será
elegible cuando su mapa efectivo pase el mismo certificado y sus tests de
integración; no se deduce elegibilidad de `basis_shape`/`max` por su nombre.

Solo se fusionan llamadas con pares de tensores de entrada compatibles con
los orígenes declarados. Argumentos globales, firmas personalizadas o caminos
no certificados mantienen la llamada histórica, incluidos sus errores de
validación originales; no se captura cualquier excepción para esconder bugs.

### 4.3 Metadatos y checkpoints

Construir índices/signos planos por TP y Linear a partir de las instrucciones
existentes y del mapa T. No formar matrices densas de transformación de pesos
ni una matriz K×K de coeficientes. Complejidad y memoria O(número de pesos).

Metadatos derivados en buffers **no persistentes**; conservar las mismas
claves, shapes y orden de Parameter en `state_dict`/optimizer. No crear nuevos
TP/Linear ni consumir números aleatorios para preparar la ruta.

El certificado debe comprobar biyectividad/involución, m/signos por irrep,
parejas de instrucciones y normalización. Al cargar un estado se recertifica
contra el **Q cargado**, no solo contra el Q del constructor. Si un checkpoint
contiene un Q no compatible con la permutación firmada, se usa la ruta original.
Los cambios de dispositivo/dtype mueven metadatos con el módulo. La mutación
directa arbitraria de Q fuera del constructor/carga no se presenta como API
soportada de invalidación automática.

### 4.4 Fallbacks del primer alcance

- Flag apagado, `symm_transpose=False` o fórmula distinta de `ij`.
- `output_scaleshift=True`. No suponer que una escala diagonal conmuta con T:
  en general habría que transformar también escala y shift; no se implementa
  esa ampliación en esta primera versión.
- Operación distinta de la clase exacta indicada, pesos externos por arista,
  modos de conexión distintos de `uvw`, bias u otros contratos no certificados.
- Mapa RTP ausente, no firmado, no involutivo o incompatible con Q/irreps.
- Dtypes/contextos de autocast no certificados. Primer objetivo: CPU
  float64/fp32 y CUDA fp32; AMP únicamente con el readout realmente en fp32,
  como en el contrato existente. No dar por certificado fp16/bf16 puro.

Fallback correcto antes que acelerar una función diferente. Informar una vez
del motivo cuando se pidió el flag pero un bloque candidato no es certificable;
no emitir avisos ni efectuar comprobaciones tensor→host en cada forward.

## 5. Prueba realizada y pruebas aún requeridas

### 5.1 Exploración de requisitos, no implementación

Se comprobaron en CPU con e3nn 0.4.4/Torch 2.11.0:

- 44 identidades de intercambio de `o3.wigner_3j`, para l de entrada 0–3 y todos
  los l de salida permitidos: error máximo 7,22e-16 en float64.
- Cinco bases (`s`, `sp`, `spd`, `spf` y `2s2p1d1f`), con 1/2 componentes: Q
  transpuesto coincide con su permutación firmada, con error máximo 1,56e-15.
- En esos diez casos, T conserva segmento de salida y m y su signo es uniforme
  por irrep; todos los caminos `uvw` del TP tienen su inverso con el mismo
  factor normalizado, usando features con segmentos repetidos y multiplicidades
  distintas (`2s+3p+1d+2f+1s`).

Registro de la repetición final: `/tmp/g2m-6-7-design-prerequisites.log`, exit0.

La primera invocación importó e3nn antes de Graph2Mat y encontró el problema
conocido de `slice` en `torch.load(weights_only=True)` de e3nn 0.4.4. Se repitió
con el orden de importación habitual de Graph2Mat, que ya registra ese global
seguro. No se desactivó weights_only ni se modificaron paquetes.

Estas comprobaciones respaldan los requisitos de la derivación. **No son
tests de una ruta fusionada, gradcheck ni certificación GPU de 6.7**, porque esa
ruta aún no se ha implementado. No se utilizó GPU en esta tarea de diseño.

### 5.2 Aceptación necesaria para implementar/certificar

- Referencia independiente: dos evaluaciones originales de la operación y
  transpuesta explícita de la matriz; no usar la misma proyección como oráculo.
- Salidas, coeficientes y gradientes de **todas** las entradas y parámetros;
  gradcheck pequeño en float64 y comparación float32. Incluir una pérdida
  con cotangentes aleatorias, no solo norma cuadrada de la salida.
- s/p/d/f, bases repetidas, segmentos de features repetidos,1/2componentes,
  solo nodos, solo mensajes y ambos orígenes;0/1/Naristas.
- O(3), intercambio de extremos, grafos periódicos homo/heteronucleares y
  flags apagados. Verificar que un bloque elegible evalúa Nfilas y no2N.
- MACE real con una/varias lecturas por interacción, coefficient-only,
  checkpointing y agrupaciones que pasen certificación; unsupported conserva
  fallback. No asumir que una prueba de operación cubre el modelo completo.
- Checkpoints antiguos/nuevos `strict=True`, flag on/off, resume de Adam,
  Q cargado incompatible, cambio de dtype/dispositivo y copia del modelo.
- Hooks de forward del bloque/operación/TP conservados en las rutas que ya
  los soportan; la selección privada no se interpreta como otro par de features.
- CUDA fp32 y AMP con readout fp32: salidas/gradientes y pasos finitos.
  Compilación opcional sin introducir sincronizaciones host en la ruta caliente.
- Gates de regresión del proyecto y wheels Python3.9/3.12 proporcionales al
  cambio. Registrar fallos históricos por separado; no inventar una suite verde.

## 6. Evaluación de beneficio, después de equivalencia

Comparación opt-in on/off con pesos, batch, dtype y entorno idénticos. Primero
smoke; después cuatro rondas equilibradas,10warmups y30pasos, en la misma
fixture d/f16/72. Añadir controles w90/6x6 con sus datos/configs congelados,
sin redefinir splits ni entrenar modelos científicos nuevos.

Métrica primaria: pared del paso completo de la referencia d/f72. Reportar
ratios pareados, mediana/MAD/rango, forward/backward por eventos, kernels,
VRAM allocated/reserved y coste de metadatos/proyección. No excluir la
proyección ni su backward del tramo medido. La memoria adicional de índices,
signos y pesos temporales puede anular un ahorro de activaciones.

Recomendar la ruta como aceleración solo si la mediana pareada y las cuatro
razones de ronda favorecen al candidato; documentar cualquier regresión en
los controles y el tradeoff de memoria. Si las razones se solapan con1,
resultado inconcluso, no éxito. Si empeora, mantener la referencia y reportar
el resultado negativo. En ningún caso se cambia el default automáticamente.

GPU solo libre de cómputo ajeno, usando el supervisor existente: allocator
como máximo55%, reserva inicial30%, aborto propio con libre<25% o>=80°C.
Sin tocar procesos ajenos ni encolar; nuevas campañas científicas requieren
aprobación aparte. No prometer el9,78%/10,83% aritmético de kernels como mejora
de pared, ni que la precisión vaya a aumentar.

## 7. Alcance de archivos y estado de aprobación

Producto previsto: `bindings/e3nn/modules/edge_operations.py` y
`bindings/e3nn/modules/matrixblock.py`. Tests junto a las operaciones, bloques
y modelos Lightning; documentación de la opción y mediciones en auditorías.
No se prevé cambiar core/NumPy, el DataModule, la CLI ni exports. Si la lectura
de callers demuestra que hace falta ampliar ese alcance, explicarlo antes
de hacerlo; no introducir un protocolo genérico por anticipación.

El diseño aprobado no es una implementación terminada. Siguiente puerta:
revisión/aprobación del plan concreto y confirmación del método de ejecución.
Ningún commit, default, cola o código de producto cambiado por preparar el diseño.

## Fuentes técnicas

- [NIST DLMF34.3(ii)](https://dlmf.nist.gov/34.3#ii): fase al intercambiar
  columnas de los símbolos3j; aquí aplicada al acoplamiento angular entero.
- [API oficial TensorProduct de e3nn](https://docs.e3nn.org/en/stable/api/o3/o3_tp.html):
  conexión `uvw`, pesos explícitos y vistas por instrucción. El diseño se
  contrastó además con `_tensor_product.py`, `_sub.py` y `_linear.py` de la
  versión0.4.4 instalada, no suponiendo que la documentación estable sea esa
  misma versión.
- Código local: `irreps_tools.ReducedTensorProducts`,
  `E3nnIrrepsMatrixBlock`, `E3nnLinearEdgeBlock` y construcción de bloques en
  `Graph2Mat`. La identidad de pesos propuesta es una derivación para estas
  clases, no una garantía publicada por e3nn sobre cualquier readout.
