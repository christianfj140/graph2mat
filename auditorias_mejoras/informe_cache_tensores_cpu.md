# 6.1 — caché CPU segura y coste de datos

Fecha2026-10-09. Subplan `plan_cache_tensores_cpu.md`; BASEbeb0098.
Implementación usa la caché existente `store_in_memory`/InMemoryData, no añade
caché ni flag. Deepcopy al materializar/acceder aísla tensores y metadata mutable;
rebuild si cambian fuentes/procesamiento. `data_objects` queda readonly por contrato.
Tamaño cero no materializa; negativos/exceso ValueError antes de leer.
Guard pathlike evita repr de matrices/configuraciones solo para detectar `.fdf`.
Dependencia AtomicTable: deepcopy propio conserva estado runtime (incluida la
referencia on-site), metadata y radios atomic/radial sin reconstruir por pickle.
El formato de checkpoints/pickle NO cambia. El camino anterior sustituía7→0
en una referencia configurada y cambiaba la reconstrucción física.

## Evaluación CPU pareada

Métrica primaria fijada antes: lazy access; no regresión temporal mayor de10%,
igualdad exacta de elementos/dtype. Fixture propia periódica s/p/d/f, dos
componentes crudos (sin interpretar S física), fp32, 16/72 puntos y96/432 aristas.
No dataset del usuario. Datos dirigidos y collate con `symmetric=False` en ambos
caminos: la primera medida usó por error el default simétrico del collate y sus
porcentajes quedan sustituidos por esta ejecución corregida.
Python3.12.3/Torch2.11.0cpu. Tres warmups, doce rondas alternadas, ocho repeticiones
por tramo/ronda; caché inicial/RSS en procesos nuevos. Source dataset SHA256:
`2650126914570197eff0071b14dc5fa71360ae650a7f3027805ff8d64b680809`.

| Mediana ms, base → candidato | 16 puntos | 72 puntos |
|---|---:|---:|
| Lazy acceso | 5.984 → 1.127 | 23.251 → 2.024 |
| Lazy acceso+collate2 | 12.601 → 2.727 | 47.248 → 4.547 |
| Caché acceso compartido → copia | 0.00021 → 0.145 | 0.00022 → 0.148 |
| Caché acceso+collate2 | 0.338 → 0.646 | 0.353 → 0.674 |
| Inicialización caché8muestras | 48.670 → 12.630 | 186.793 → 20.352 |
| RSS adicional KiB al materializar8 | 5044 → 5228 | 8744 → 8800 |
| Bytes de tensores/muestra, idénticos | 93424 | 420240 |

Dispersión lazy(min,max):16 base(5.828,6.112),candidato(1.098,1.180);
72 base(22.951,23.870),candidato(1.982,2.057). Collate-cache16 base(.332,.345),
candidato(.632,.666);72 base(.347,.381),candidato(.663,.719).
Igualdad exacta de tensores/batches comprobada dentro del benchmark.

Decisión: retirar repr es favorable en este CPU/input concreto (≈5.3×/11.5× acceso).
La caché YA EXISTÍA; no atribuir esa reutilización a un mecanismo nuevo.
Copiar para ownership **empeora** el acceso de la caché y collate2(≈+91% ambos,
~0.31–0.32ms adicionales por dos muestras), compromiso deliberado por seguridad,
con ~0.14–0.15ms/muestra aquí. No declarar
ganancia de cache sobre la versión compartida. RSS incluye runtime/allocator;
peak_delta0 indica pico previo no superado, no consumo cero. Carga host no aislada.
No extrapolar a I/O de ficheros, paso GPU/VRAM, materiales grandes o precisión.

## Reproducir

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/christian/repositorios/graph2mat/src \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/tmp/g2m-python312-full.LHyni1/venv/bin/python \
auditorias_mejoras/bench_cache_tensores_cpu.py --points 16
# Repetir --points 72. JSON con min/mediana/max y memoria de procesos nuevos.
```

Baseline local reproduce únicamente el camino anterior de BasisConfiguration
sin FDF/sidecars y caché compartida; no checkout ni mutación del módulo durante
medidas. Tests separados cubren FDF/ZIP/spectral_C, lectores sisl XV nativos,
componentes/direcciones, workers0/2 spawn/fork y cleanup de tmp propio.

## Pruebas y límites

- RED24fallos/11pasan/44.61s; GREEN38pasan/5.43s, salidas leídas.
- Floor source3.9/Torch2.4.1cpu152pasan/124.10s; dos controles XV/tmp adicionales
  separados2pasan/0.52s. No afirmar una única ejecución154 ni wheel actual.
- Ruff fuente/tests, mypy-core, compile39/diff verdes. HookRuff excluye auditorías;
  Ruff directo sobre benchmark **All checks passed**, compile39 también verde,
  no contar el hookSkipped comoPass.
- Primer fullCPU se interrumpió solo en PIDpropio430406 al descubrir el borde
  Atomic: exit2,1395pasan/2skip/429.14s; no suite completa. CoreRED3deproducto,
  fixturecache tuvo dos errores(typeH y BasisMatrix sin referencias), corregidos
  con table.types/DensityMatrix/convert_to. Control negativo en proceso propio
  elimina temporalmente __deepcopy__, sin tocar source: cache falla0vs7,1/0.03s.
  GREEN56/5.46s; REDradiosdistintos1/0.03s y GREENcertificado57/5.67s.
- Suite source CPU definitiva **2100pasan,2omitidos,26002warnings,629.52s,exit0**,
  cache-tensor-full-cpu-certified.log, con bordeAtomic/radios corregidos,
  sin modificaciones de source/tests durante ella. Floor source3.9 actualizado
  **171pasan,2322warnings,127.17s,exit0**, cache-tensor-python39-certified.log.
  Hash table.py final `88eb775edf7555ef116532e82ca49be9bdb2d4b2c648d5e64c41ed76834e385e`.
- Benchmark corregido: cache-tensor-benchmark-16/72-certified.json, exit0 ambos;
  stderr vacíos. Ruff directo/compile39 repetidos después del cambio, verdes.
- Logs `/tmp/g2m-precision-goal.A2Pbgk/cache-tensor-*`. Sin ownGPU/entornos ajenos.
- Producto bc54e91. Task-done real **57pasan,51warnings,5.02s,exit0**, rango
  beb0098..bc54e91, coreAtomic+cache+resources+stages; salida y ledger leídos.
- 6.1 implementación CPU cerrada; beneficio CPU lazy medido,
  coste cache regresivo documentado. GPU/VRAM aún sin recurso; goal sigue abierto.
  Siguiente: perfil sparse6.2 y después sampler6.4, no fonones.
