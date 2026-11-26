Niveles de entrenamiento: nacho_facil (peso 1), nacho_medio (peso 1), nacho_dificil (peso 1), nacho_muy_dificil (peso 1)
Validación: level_1 (held out)
Progresión: [nacho_facil] -> [nacho_medio] -> [nacho_dificil, nacho_muy_dificil]

## Modelo final: niveles de entrenamiento

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_progresion_prog_jp0.05) | 2 | 69 ± 6 | 76% ± 1 | -2.17 ± 0.66 | 1.56 | 0/2 |

## Modelo final: validación

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_progresion_prog_jp0.05) | 2 | 50 ± 50 | 68% ± 32 | -4.64 ± 5.33 | 2.00 | 1/2 |

## Mejor modelo (best_model.zip): niveles de entrenamiento

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_progresion_prog_jp0.05) | 2 | 76 ± 1 | 82% ± 4 | -1.36 ± 0.19 | 1.67 | 0/2 |

## Mejor modelo (best_model.zip): validación

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_progresion_prog_jp0.05) | 2 | 0 ± 0 | 49% ± 0 | -10.12 ± 0.06 | - | 0/2 |
