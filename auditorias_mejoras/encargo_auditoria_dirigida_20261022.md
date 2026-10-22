# Encargo: auditoría dirigida de corrección física de `g2m-v2`

Fecha: 2026-10-22. Para un auditor que no haya escrito el código (Codex u otro agente).

## Resultado esperado: puede que no haya bugs, y eso está bien

Esta auditoría **no tiene que encontrar nada**. Ya hubo dos auditorías completas
(`auditoria_bugs_20261021.md` y `auditoria_codex_20261021/`) y sus arreglos están aplicados y
revisados. Si las comprobaciones pasan, el resultado correcto es decirlo: «comprobado X con el
método Y; no se encontró ningún fallo». Ese informe vale tanto como uno con hallazgos.

- No conviertas limitaciones conocidas, opciones de estilo o mejoras posibles en «bugs» para
  tener algo que entregar.
- No rebajes el umbral de evidencia: un hallazgo solo cuenta si hay una reproducción mínima que
  falla y una causa identificada.
- Una sospecha que no consigas reproducir se anota como «no reproducido», no como hallazgo.
- Si una zona no se pudo comprobar (falta un dato, tarda demasiado), dilo; no lo rellenes.

## Objetivo

Comprobar, con un método distinto a las auditorías anteriores, que las partes donde un fallo
daría **números físicamente incorrectos pero creíbles** son correctas. Las auditorías
anteriores leyeron código. Esta usa **oráculos y propiedades**: invariancias, idas y vueltas,
y comparación con cálculos independientes.

## Alcance

Solo estas zonas, en este orden de prioridad:

1. **Gauge y pérdidas con S**: `core/data/metrics.py` (`gauge_invariant_mse`, término espectral,
   espacio nulo de S), `tools/analysis/metrics.py` (métrica c*).
2. **Espín-órbita y magnetismo**: `core/data/spin_orbit.py`, conversión de canales en
   `core/data/configuration.py` y `core/data/sparse.py`, `bindings/e3nn/modules/spin_orbit.py`
   y `magnetic.py`, `models/mace.py` (doble pasada m/−m).
3. **Herramientas Δ**: `tools/siesta/delta.py` (H0, `combine_hamiltonians`, `add_h0`,
   proyección de inversión temporal), `tools/siesta/dm_pilot.py`.
4. **Largo alcance**: `models/long_range.py` y su uso en `models/mace.py`.
5. **Reconstrucción etiquetas ↔ matriz**: `core/data/processing.py`, `core/data/sparse.py`,
   `_sparse.py`, con políticas, `nsc`, `edge_cutoff_padding` y `basis_grouping`.
6. **Código posterior a la revisión de Codex**: `git log e1e7e4b..g2m-v2 -- src` (parches de
   Codex, mypy, 3.11, guardrail en Python 3.9, docstrings, marcador `slow`).

Fuera de alcance: rendimiento, estilo, documentación, scripts de campañas, servidor, y repetir
lo que ya cubren los tests existentes con el mismo método.

## Método: comprobaciones sugeridas

Cada una es una propiedad que debe cumplirse. Basta con hacer las que tengan sentido; no hace
falta hacerlas todas ni inventar más.

**Gauge y S**
- `gauge_invariant_mse(pred + c·S) == gauge_invariant_mse(pred)` para c arbitrario, con 1 y 2
  componentes de espín (un único c común) y con aristas y nodos parcialmente sin etiqueta.
- Un desdoblamiento rígido de espín (+c·S en ↑, −c·S en ↓) **sí** se penaliza.
- El término espectral no cambia con c·S ni con un cambio de base de las columnas dentro de la
  ventana.

**Espín-órbita y magnetismo**
- Ida y vuelta SIESTA (8 reales o 4 complejas) → canales → SIESTA, exacta, en un TSHS SOC real
  (Bi2Se3 en `$G2M_DATA_ROOT`, solo lectura) y en uno sintético.
- H(k) reconstruido es hermítico en k aleatorios, y la inversión temporal se cumple en el
  modelo `spin_orbit` (canales impares nulos tras la proyección).
- Equivariancia del modelo: girar la geometría y los momentos rota el spinor de H según la
  representación correcta (D(R) orbital ⊗ SU(2) de espín).
- Magnetismo: F(m) y F(−m) cumplen la paridad de inversión temporal exacta declarada.

**Herramientas Δ**
- `add_h0` sobre la etiqueta Δ = H − H0 (`write_delta_sample`) devuelve H exacto con datos
  reales, incluso con `nsc` distintos entre H y H0.
- `combine_hamiltonians` con el mismo `nsc` y `sc_off` permutado da el mismo resultado que con
  `sc_off` canónico.
- Proyección TRS: idempotente, y no cambia un H que ya cumple la inversión temporal.

**Largo alcance**
- La suma de Ewald coincide con una suma directa convergida en una celda pequeña, incluida una
  celda oblicua.
- Invariancia del potencial frente a traslación y rotación rígidas; cargas neutras con
  `"charges"`; con `"qeq"`, suma de cargas electrónicas = −NetCharge.

**Reconstrucción**
- Etiquetas → matriz → etiquetas exacta para cada política, con y sin
  `edge_cutoff_padding`, y con `basis_grouping` `point_type`, `basis_shape` y `max`.
- Una permutación de átomos de la estructura de entrada permuta la matriz de salida y no cambia
  nada más.

## Reglas

- **Solo CPU.** Nada de GPU ni de entrenamientos largos; la cola de GPU no se toca.
- **SIESTA real: no.** Usar TSHS/DM ya calculados, en solo lectura. No escribir en
  `/mnt/research-data` ni modificar `MD_vs_AtomicDisplacement`.
- **No cambies código de producto ni hagas commits o push.** Las reproducciones van en una
  carpeta propia (`auditorias_mejoras/auditoria_dirigida_<fecha>/`). Si propones un arreglo,
  descríbelo; aplicarlo es decisión del usuario.
- Entorno: el venv de `MD_vs_AtomicDisplacement/.venv` con `graph2mat` editable (la extensión
  `_sparse` está recompilada) o uno propio; anota versiones.
- Respeta las reglas de temperatura: CPU por debajo de 80 °C.

## Criterio para que algo cuente como hallazgo

Un hallazgo necesita las cuatro cosas:

1. una reproducción mínima, ejecutable, que falla;
2. la causa en el código (fichero y línea);
3. el efecto: qué número sale mal y en qué combinación de opciones admitida;
4. la gravedad: **grave** (resultado físico incorrecto en silencio en una combinación
   admitida), **media** (fallo acotado o error ruidoso) o **menor**.

Lo demás se clasifica aparte, sin contarlo como bug: limitación ya documentada (ver
`roadmap.md`, «Estado vigente», y `CHANGELOG.md`), sospecha no reproducida, mejora posible.

## Entrega

Un informe corto `informe.md` en la carpeta de la auditoría:

1. **Resumen** en tres líneas. Si no hay hallazgos, decirlo así: «No se encontraron fallos en
   las comprobaciones realizadas».
2. **Tabla de comprobaciones**: zona, propiedad comprobada, método, resultado (pasa / falla /
   no comprobado y por qué).
3. **Hallazgos**, si los hay, con los cuatro puntos del criterio.
4. **Otras observaciones** (limitaciones, sospechas no reproducidas), separadas.
5. Entorno y comandos para repetir las comprobaciones.

No hace falta un informe largo: las comprobaciones que pasan se listan en la tabla, sin más.

## Cuándo parar

Cuando se hayan hecho las comprobaciones razonables de las seis zonas, o antes si el tiempo
no da. Una zona sin hallazgos tras sus comprobaciones está terminada; no hay que seguir
buscando en ella.
