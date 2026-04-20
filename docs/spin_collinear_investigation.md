# Investigación técnica: soporte de spin colineal en Hamiltonianos

Este documento resume hallazgos del pipeline actual de `graph2mat` para matrices tipo Hamiltoniano y los cambios mínimos recomendados para soportar spin colineal (up/down), sin entrar en SOC ni caso complejo no hermítico.

## Hallazgos clave

- El pipeline actual está diseñado para **una sola componente escalar por elemento de matriz**.
- En lectura desde `sisl`, el convertidor a bloques toma explícitamente `spmat.data[:, 0]`, por lo que descarta cualquier componente extra (incluyendo spin colineal en matrices polarizadas).
- La representación interna (`BasisMatrix`, `BasisMatrixData`, labels planos, punteros por bloque, métricas) asume `point_labels` y `edge_labels` 1D.
- La reconstrucción de salida hacia `sisl.Hamiltonian` pasa por `scipy.csr` 2D y `sp_class.fromsp(geometry, csr)`, ruta que no preserva multicomponente de spin.

## Consecuencia

Para spin colineal se debe introducir explícitamente una dimensión de componentes (2) en labels, conversiones sparse y reconstrucción a `sisl`, evitando duplicar artificialmente la base orbital.

