Niveles de entrenamiento: nacho_facil (peso 1), nacho_medio (peso 1), nacho_dificil (peso 1), nacho_muy_dificil (peso 1)
Validación: level_1 (held out)

## Modelo final: niveles de entrenamiento

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_jp0.05) | 2 | 68 ± 2 | 71% ± 3 | -2.38 ± 0.28 | 1.80 | 0/2 |

## Modelo final: validación

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_jp0.05) | 2 | 0 ± 0 | 33% ± 3 | -10.17 ± 0.04 | - | 0/2 |

## Mejor modelo (best_model.zip): niveles de entrenamiento

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_jp0.05) | 2 | 82 ± 8 | 85% ± 8 | -0.66 ± 0.82 | 1.62 | 0/2 |

## Mejor modelo (best_model.zip): validación

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_jp0.05) | 2 | 0 ± 0 | 35% ± 14 | -10.03 ± 0.03 | - | 0/2 |
