Niveles de entrenamiento: nacho_facil (peso 1), nacho_medio (peso 1), nacho_dificil (peso 1), nacho_muy_dificil (peso 1)
Validación: level_1 (held out)
Progresión: [nacho_facil] -> [nacho_medio] -> [nacho_dificil, nacho_muy_dificil]

## Modelo final: niveles de entrenamiento

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (grid_fixed_lv-nacho_progresion_prog_jp0.05) | 2 | 50 ± 0 | 65% ± 2 | -4.31 ± 0.01 | 1.95 | 0/2 |

## Modelo final: validación

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (grid_fixed_lv-nacho_progresion_prog_jp0.05) | 2 | 25 ± 25 | 48% ± 27 | -7.38 ± 2.62 | 2.08 | 0/2 |

## Mejor modelo (best_model.zip): niveles de entrenamiento

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (grid_fixed_lv-nacho_progresion_prog_jp0.05) | 2 | 65 ± 12 | 77% ± 9 | -2.61 ± 1.43 | 1.72 | 0/2 |

## Mejor modelo (best_model.zip): validación

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (grid_fixed_lv-nacho_progresion_prog_jp0.05) | 2 | 25 ± 25 | 54% ± 34 | -7.26 ± 2.69 | 1.77 | 0/2 |
