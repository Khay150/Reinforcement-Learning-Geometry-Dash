Niveles de entrenamiento: nacho_facil (peso 1), nacho_medio (peso 1), nacho_muy_dificil (peso 1), facil (peso 1), medio (peso 1), dificil (peso 1), muy_dificil (peso 1), muy_muy_dificil (peso 1), franco_facil (peso 1), franco_medio (peso 1), franco_dificil (peso 1), franco_techo (peso 1), franco_rutas (peso 1), franco_tunel (peso 1), franco_pasillos (peso 1), franco_sin_saltar (peso 1), santi_intermedio_1 (peso 1), santi_intermedio_2 (peso 1), santi_intermedio_dificil (peso 1), santi_dificil (peso 1)
Validación: santi_facil, level_1, nacho_dificil (held out)

## Modelo final: niveles de entrenamiento

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (e8_generalizacion) | 3 | 49 ± 8 | 70% ± 6 | -4.36 ± 0.88 | 1.94 | 0/3 |

## Modelo final: validación

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (e8_generalizacion) | 3 | 67 ± 0 | 80% ± 3 | -2.63 ± 0.04 | 2.12 | 0/3 |

## Mejor modelo (best_model.zip): niveles de entrenamiento

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (e8_generalizacion) | 3 | 61 ± 3 | 76% ± 4 | -3.05 ± 0.28 | 1.93 | 0/3 |

## Mejor modelo (best_model.zip): validación

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (e8_generalizacion) | 3 | 67 ± 0 | 81% ± 3 | -2.58 ± 0.10 | 1.99 | 0/3 |
