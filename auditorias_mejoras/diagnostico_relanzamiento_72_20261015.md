# 7.2 — Diagnóstico del fallo y recuperación de la campaña

Usuario: «Investiga que es y relanzalo»,2026-10-15. Alcance: corregir el runner
y recuperar los mismos seis trabajos aprobados, sin otras recetas, referencias
nuevas, defaults, fonones ni cambios de autorun.

## Fallo observado

`tbg_s0` empezó02:15:13 y terminó con error02:18:19. Último heartbeat:1290pasos
de7995; checkpoint parcialbest-val_loss en1275. Los cinco dependientes no se
lanzaron. Telemetría:180,6s, mínimo90,9%VRAM libre, máximo52°C; no señal de
agotamiento de VRAM o aborto térmico. Los8.306hashes congelados seguían intactos.

El log acababa a mitad de un volcado `Timeout (0:01:30)!`, sin excepción Python.
Ese texto era un temporizador periódico incondicional de diagnóstico: **no
demostraba que el entrenamiento estuviera bloqueado**. El runner tampoco
incluía el código de salida del hijo en su error, dificultando el diagnóstico.

## Reproducción y causa

Se ejecutó el mismo runner/config/seed bajo gdb, con límite diagnóstico2500pasos,
allocator0,55 y supervisor de VRAM/temperatura. Se reprodujo en90,64s:

```text
Thread 2 "python" received signal SIGSEGV, Segmentation fault.
#0 PyCode_Addr2Line
#2 _Py_DumpTracebackThreads
```

El hilo que falla es el del volcado de pilas, no el hilo del modelo/backward.
El entrenamiento había registrado570pasos. Esto localiza el fallo nativo de
este entornoPython3.12.3 en `faulthandler.dump_traceback_later(90, repeat=True)`,
añadido al wrapper como diagnóstico. No se atribuye a7.2 ni a un OOM, y no se
afirma que todos los entornosPython3.12 reproduzcan el problema.

gdb devuelve0 al terminar el script de comandos tras detenerse por SIGSEGV;
ese0 **no es éxito del entrenamiento**. Evidencia: señal y backtrace del
inferior, no el status del debugger. Al terminar quedaron cuatro workers del
inferior en su propio grupo; se verificaron comandos/UID/PGID2968583 y se
terminó únicamente ese grupo de diagnóstico. Sin señales ajenas ni cambios de
permisos/entorno compartido.

Evidencia conservada en
`experimentos_edge_updates_bilayer_20261015_diagnosis_v1/`:
`gdb-original.log`, telemetría, `gdb-status.json`, config diagnóstica,
`runner-before.py` (SHA256 original
`f7d24b812ab2d99ff59e2691151eea5b03106add577355faadf6547c17ea6b92`).

## Corrección mínima y prueba

- Retirados import, inicio y cancelación del volcado periódico. Se conserva
  íntegro el watchdog externo de heartbeat900s, VRAM libre<25%, GPU≥80°C,
  fracción calculada≤55% y cesión ante cómputo ajeno.
- El error del runner ahora incluye el código de salida del hijo.
- Regresión: `test_fit_does_not_start_unsafe_background_stack_dumps`, observada
  RED contra el runner anterior y GREEN después. Evita reinstalar el timer
  inseguro antes del startup, sin necesitar GPU en la suite.
- Nueve tests del wrapper y30de supervisión/benchmark:39pasan. Ruff y
  diff--check limpios. No se cambió código de producto Graph2Mat, datos,
  hiperparámetros ni selección de checkpoints.
- Control corregido **2500pasos completos en340,38s**, con la misma config
  diagnóstica: supera los1290pasos y180s del intento nocturno. Pérdida final
  finita8,328e-7, actualización de aristas activa; allocated1276,07MiB,
  reserved1966MiB. No constituye resultado científico ni comparación de velocidad.
- Suite generalCPU final: **2662pasan/34omitidos/4fallos históricos**,935,55s.
  Son exclusivamente
  `test_real_extrapolation_with_missing_onsite_and_edge_blocks[False/True-1/2-HistorySOAPs]`,
  por ausencia de `dscribe`. No se presenta como suite totalmente verde ni se
  instalaron dependencias. Logs completos en `full-cpu-after.log`.

## Recuperación ejecutada

Se reinicia `tbg_s0` desde su semilla original, no desde el checkpoint parcial:
se perdieron unos tres minutos y así no se introduce un protocolo nuevo de
restauración de RNG/optimización. El parcial se conserva, no se elimina.

Manifest original archivado; comprobados8.305hashes sin cambios y actualizada
únicamente la entrada del runner, con metadatos de recuperación y enlace al
control.8.306hashes verificados después. Los seis configs científicos permanecen
byte-a-byte idénticos.

Cola: conservar el primer IDfallido y añadir un IDretry único para ese mismo
trabajo; sustituir únicamente la dependencia del `grhbn_bl_s0` pendiente para
que espere al retry. Las otras cuatro dependencias no cambian. Operación bajo
`flock`, copia de seguridad y reemplazo atómico, abortando si cambian cola/estado;
append mediante `append_queue`. Comprobar bytes de todas las líneas ajenas y
el prefijo anterior a la campaña. Sin tocar daemon, state ni otros pendientes.

Recuperación encolada:1402→1403entradas, solo un retry nuevo:
`experimentos_edge_updates_bilayer_20261015_v6:tbg_s0:retry1`.
Solo cambia la dependencia de `grhbn_bl_s0` para esperar al retry; las otras
cuatro y todos los trabajos ajenos permanecen intactos. Cola anterior y manifest
guardados en el directorio diagnóstico; `relaunch_receipt.json` guarda pruebas
del reemplazo, append y conservación del prefijo anterior a la campaña.
No se han tocado código/estado del daemon ni defaults. Arranque confirmado:
retryPID2974768, hijo del daemon1756952; **4750pasos observados**, más allá del
punto del fallo inicial. Los otros cinco siguen encolados con sus dependencias.
SuiteCPU final completada y hashes/cola comprobados de nuevo: una dependencia
modificada y un retry añadido, resto de bytes y checkpoint parcial intactos.

Todavía no hay comparación científica final de7.2. El control y el reintento
demuestran que ya no aparece el fallo del temporizador en los tramos observados,
no garantizan ausencia de cualquier otro fallo durante los seis entrenamientos.
No se cambia la estimación histórica en una afirmación de aceleración.
