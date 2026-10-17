# Información física en graph2mat: estado del arte y plan de pruebas

Fecha: 2026-10-17. Objetivo final: constantes de acoplamiento electrón-fonón (e-ph) a partir
de H predicho. Este documento es un estudio de viabilidad, no una implementación.

> **Naturaleza del documento:** primer barrido rápido y amplio (Sonnet), pensado para detectar
> indicios. Una segunda pasada con un modelo más potente debería: (1) leer los textos
> completos de los artículos de §2, que aquí solo se vieron por resúmenes; (2) revisar el
> código de Delta Learning y gauge una vez subido; (3) concretar E0 como script y las
> interfaces de §7.

## 0. Límites de este estudio (leer primero)

- **Código visto.** Este clon tiene 60 commits en total y `g2m-v2` coincide con
  `origin/g2m-v2` (`a5e8134`, 2026-09-27). No aparecen los ~200 commits locales que
  mencionas, ni el **Delta Learning**, ni la **invariancia gauge** (no hay ningún "gauge" en
  `src/`; el gauge solo figura como punto 2.4 del plan). Tampoco están `MD_vs_AtomicDisplacement`
  ni `Comparison/results`. Lo más probable es que esos commits estén en tu máquina o en el repo
  del benchmark. Todo lo que digo sobre la interacción con Delta Learning y gauge es por tanto
  **condicional**: depende de qué línea base usa exactamente tu Delta Learning.
- **Literatura.** arXiv y nature.com están bloqueados desde este entorno (403 del proxy). He
  trabajado con resúmenes de búsqueda, no con los textos completos. Las cifras de abajo son
  las que daban esos resúmenes; no las he contrastado con los artículos. Las marco con
  "(sin verificar)" cuando vienen de una reseña de terceros.
- **Ninguna recomendación está medida.** Son hipótesis ordenadas por coste y por riesgo.

## 1. Lo que el repo ya tiene (y que condiciona qué merece la pena)

- `sub_point_matrix` (`core/data/matrices/basis_matrix.py`): resta a las etiquetas on-site el
  bloque de átomo aislado por especie y lo suma al reconstruir. Es la línea base física más
  simple, y solo existe para nodos.
- `output_scaleshift` (`bindings/e3nn/modules/matrixblock.py`): escala por irrep y
  desplazamiento 0e on-site ajustados con las etiquetas de entrenamiento. Es una línea base
  aprendida/ajustada, dentro del readout.
- `HamiltonianLocalContext`: features escalares invariantes (especie, distancias, momentos
  direccionales). Es información derivada de la geometría, no física nueva.
- `initial_node_feats` solo admite `OneHotZ`.
- El conversor bloque ↔ coeficientes irreducibles ya existe (`block_to_coefficients`, y las
  últimas commits añaden metadatos de coeficientes). Esto es la pieza clave para inyectar
  cualquier matriz externa (S, H0) como feature equivariante.

## 2. Qué dice la literatura

| Trabajo | Idea | Qué aporta aquí |
|---|---|---|
| HAMSTER, Schwade et al. (Nat. Commun. 2026; arXiv 2508.20536) | Tight-binding ajustado a la celda unidad + corrección ΔML con pocos cálculos DFT | Eficiencia de datos; error en autovalores < 50 meV a 400 K en CsPbBr3 (resumen de búsqueda). Es el artículo que recordabas |
| NextHAM (arXiv 2509.19877, ICLR 2026) | Entrada: H⁽⁰⁾ construido con la densidad de átomos aislados superpuestos (sin diagonalizar). La red predice la corrección. Pérdida en espacio real y recíproco | Es el análogo más cercano a "Delta Learning con prior físico en la misma base". Una reseña de terceros cita una reducción del 96 % en el objetivo de algunos bloques (sin verificar). El problema de estados fantasma por mal condicionamiento de S motiva la pérdida recíproca |
| Sun, Aradi et al. (JCTC 2023) | Corrige H y S de DFTB retropropagando contra la DOS de DFT | Δ-learning sobre H y S, no solo sobre energías |
| Irle 2019, EquiDTB, MLΔTB | Δ-learning sobre DFTB / TB empírico | Confirman la receta, sobre todo en energías y fuerzas |
| HamGNN (Zhong et al., arXiv 2302.00439) | Predice H y ∇H; EPC = fórmula analítica con H, ∇H y S | Es el patrón e-ph de referencia con red equivariante |
| DeepH-DFPT (PRL 2024, arXiv 2401.17892) | Derivadas por autodiferenciación para EPC | Alternativa a la diferencia finita |
| MLEPI (arXiv 2602.23084) | EPC por diferencias finitas, probado en grafeno bicapa | Cercano a tu caso `MD vs AtomicDisplacement` |
| DEQHNet (arXiv 2406.03794), self-consistency training (arXiv 2403.09560) | Autoconsistencia dentro de la red o de la pérdida | Mucha ingeniería y exigen un constructor de Fock diferenciable; no recomendado aquí |

Dos huecos honestos: no he encontrado un artículo que añada una pérdida explícita sobre
**derivadas** de H respecto a las posiciones (la literatura supervisa derivadas con diferencias
finitas de DFT, o las obtiene por autodiff del modelo), ni uno que cuantifique cuánto mejora
**EPC** (no solo H) un prior físico.

## 3. Principio de diseño: distinguir información de sesgo inductivo

- **Sesgo inductivo.** Forma funcional restringida (Slater–Koster, dos centros). No añade
  información que MACE no pueda calcular de la geometría; solo reduce lo que hay que aprender.
- **Información nueva.** Cantidades exactas y costosas de aprender que sí se pueden calcular:
  S, la energía cinética T, la parte no local (KB), el potencial de átomo neutro, y H construido
  con la densidad atómica (primer paso de SIESTA, sin autoconsistencia).

Con H = T + V_NL + V_NA + V_H[ρ] + V_xc[ρ], el primer paso de SIESTA contiene exactamente
todo salvo el apantallamiento autoconsistente. Una red entrenada sobre H − H⁽⁰⁾ solo tiene que
aprender polarización y apantallamiento (más suave, de bajo l). Además H⁽⁰⁾ incluye los
bloques KB que hoy quedan fuera del grafo (plan, Fase 0.3 y 4.1).

## 4. Candidatos, ordenados por relación valor/coste

Valor y riesgo son estimaciones mías, no mediciones.

| # | Candidato | Información o sesgo | Coste de implementación | Coste en inferencia | Valor esperado | Riesgo principal |
|---|---|---|---|---|---|---|
| A | **H⁽⁰⁾ de SIESTA (átomos superpuestos) como línea base y feature** (estilo NextHAM) | Información | Medio: generalizar `sub_point_matrix` a nodos **y aristas** por estructura, guardar H⁽⁰⁾ como arrays alineados con las etiquetas | Una llamada a SIESTA sin diagonalizar por estructura | Alto | Depende de SIESTA en inferencia; hay que comprobar qué escribe SIESTA tras 1 iteración |
| B | **S como feature de arista** y base de pérdidas/métricas | Información | Bajo–medio: S ya está en el TSHS/HSX; convertir a coeficientes con `block_to_coefficients` | S se calcula solo de la geometría y la base | Medio (alto en multiespecie) | En grafeno monoespecie puede aportar poco |
| C | **Extended Hückel**: H⁽⁰⁾_ij = K · S_ij · (ε_i + ε_j)/2 con ε por especie/capa ajustado | Sesgo + S | Bajo si B está hecho | Nulo extra | Medio | Débil para π–σ y enlaces curvados; sirve como prueba barata de si "H ∝ S" explica algo |
| D | **Rama dos-centros neuronal (Slater–Koster) aditiva**: integrales V_{l l' m}(r) por par de especies, solo función de |r| y Y_L(r̂), más residuo de MACE | Sesgo | Medio: un readout extra sin pasar por MACE | Nulo extra | Medio | Mucha de esa función ya es alcanzable por MACE (correlation=1); mejora condicionamiento, no información |
| E | **Pérdida sobre derivadas / diferencias entre configuraciones emparejadas** (p. ej. H(R+δ) − H(R) del modelo frente a DFT) | Información de supervisión, no de arquitectura | Bajo–medio: dos forwards por par en el mismo batch, o `jvp` si e3nn lo soporta | Nulo | **Alto para e-ph** | Exige pares de configuraciones en el dataset |
| F | Pérdida espectral con S (bandas por k-puntos, estilo NextHAM) | Información | Medio–alto: diagonalización diferenciable, degeneraciones | Nulo | Medio | Gradientes inestables en degeneraciones; es el punto 4.2 del plan |
| G | Features de elemento (electronegatividad, nivel atómico) en lugar de OneHotZ | Información | Bajo | Nulo | Nulo en grafeno; útil para transferir entre especies | Solo importa si vas a multiespecie |
| H | Autoconsistencia (DEQ, Mulliken iterado, SCF diferenciable) | Información | Alto | Variable | Incierto | Necesita Fock diferenciable; no empezar por aquí |

Descartado de entrada: acoplar un código DFTB externo. La base de DFTB no coincide con la
numérica de SIESTA, y A da lo mismo en la base correcta.

## 5. Lo que cambia por tener e-ph como objetivo

- La señal física es la **diferencia** H(R+δ) − H(R), no H. Un error de H de 15 meV puede ser en
  buena parte un sesgo suave común a configuraciones próximas, que se cancela en la diferencia,
  o no. **Mide el error de la diferencia entre configuraciones emparejadas antes de decidir
  nada.** Un prior que baja el MAE absoluto pero no el de la diferencia no mejora g.
- En una base local de orbitales que se mueven con los átomos, una traslación rígida de todo el
  cristal deja H invariante. Las redes con vectores relativos lo cumplen por construcción, de
  modo que la **regla de suma acústica** no necesita restricción aparte. Conviene añadir un test
  numérico (Σ_átomos ∂H/∂u ≈ 0) como comprobación, no como mejora.
- Con A, ∂H/∂u se descompone en ∂H⁽⁰⁾/∂u + ∂ΔH/∂u. La primera parte es exacta si H⁽⁰⁾ se
  calcula en cada configuración, y la red solo tiene que acertar la derivada del residuo,
  suave. Es la razón principal por la que A puede valer más para e-ph que para H: es una
  hipótesis a medir con E0 (abajo).
- La referencia de energía de H⁽⁰⁾ puede diferir de la de H por estructura. Con tu invariancia
  gauge, el objetivo sería H − H⁽⁰⁾ − c·S. Si tu Delta Learning ya resta otra línea base, hay
  que decidir si A la sustituye o se apila sobre ella; sin ver ese código no puedo decirlo.

## 6. Experimentos propuestos (de menor a mayor coste)

Todos con el protocolo de `plan_mejoras.md` §1: `w90`, semillas emparejadas.

- **E0: oráculo, sin entrenar (horas).** Con el dataset existente y SIESTA con un paso:
  1. Norma y MAE de H − H⁽⁰⁾ frente a H − H_onsite_medio, desglosado como en la Fase 0.4
     (on-site, 1ª/2ª capa, pares lejanos).
  2. Lo mismo para las diferencias entre configuraciones emparejadas: ‖Δ(H − H⁽⁰⁾)‖ frente a
     ‖ΔH‖.
  3. Ajuste lineal H_ij ≈ κ·S_ij (+ε) por tipo de bloque, para evaluar B y C sin tocar la red.
  Esto decide si A, B y C merecen código. Si H − H⁽⁰⁾ no es claramente menor que H − H_onsite,
  se abandona A.
- **E1: línea base H⁽⁰⁾ (A).** Entrenar con objetivo H − H⁽⁰⁾, pérdida y métricas sobre el
  total. Mismo `w90`, 3 semillas. Métricas: H-MAE, bandas, DOS, y **error de ΔH emparejado**.
- **E2: A + S como feature de arista (B).** Aislar si la feature aporta algo más allá de la
  línea base.
- **E3: pérdida de diferencias (E)**, con y sin A. Es la prueba más directa de si la mejora
  llega a g.
- **E4: rama dos-centros (D)**, solo si E0 muestra que el condicionamiento es el cuello de
  botella (un residuo con colas de 1e-3–1e-5 eV mal escaladas).
- **E5: métrica final.** g_{mn}(k,q,ν) por diferencia finita sobre un modo, frente a DFT, para
  la mejor variante.

Criterio de mejora: el de §1 del plan (dos semillas emparejadas, mejora media > 2× la
diferencia entre semillas de la línea base).

## 7. Esbozo de implementación para A (si E0 sale bien)

1. Datos: campos opcionales `point_baseline` y `edge_baseline` en `BasisMatrixData`, con el
   mismo layout plano que `point_labels`/`edge_labels`. Hoy `to_flat_nodes_and_edges` solo
   resta el bloque on-site por especie; se generaliza a "restar baseline si existe".
2. Procesador: `MatrixDataProcessor.matrix_from_data` suma la línea base al reconstruir.
   `compute_metrics` compara siempre totales.
3. Modelo: el baseline entra además como feature de nodo/arista (vía `block_to_coefficients`)
   en `E3nnEdgeMessageBlock`. Interacción con `output_scaleshift`: ajustarlo sobre el residuo.
4. Proveedor de H⁽⁰⁾: interfaz enchufable (especie media, EHT, fichero SIESTA). Cache en disco
   versionada por hash de configuración, como ya propone la auditoría para el preprocesado.
5. Tests: equivariancia con baseline activo, ida y vuelta (restar y sumar) exacta, y que
   `deepcopy` siga funcionando.

## 8. Qué necesito de ti para pasar de estudio a código

1. **Subir lo que tienes en local** (por ejemplo `git push origin g2m-v2:g2m-v2-local`, a otra
   rama para no pisar `origin/g2m-v2`) y el repo del benchmark o al menos
   `Comparison/results/.../README.md`. Sin eso no puedo ver Delta Learning ni gauge.
2. **Qué línea base resta hoy tu Delta Learning** (¿H de equilibrio?, ¿átomo aislado?, ¿otro
   modelo?).
3. **Si aceptas SIESTA (un paso sin diagonalizar) en inferencia.** Si no, A queda fuera y el
   foco pasa a B, C, D y E.
4. **Si el dataset tiene pares emparejados** (desplazamiento atómico, no solo MD) para E.
