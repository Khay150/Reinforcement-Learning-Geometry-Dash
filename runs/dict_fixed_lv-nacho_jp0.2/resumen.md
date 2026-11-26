Niveles de entrenamiento: nacho_facil (peso 1), nacho_medio (peso 1), nacho_dificil (peso 1), nacho_muy_dificil (peso 1)
Validación: level_1 (held out)

## Modelo final: niveles de entrenamiento

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_jp0.2) | 2 | 51 ± 1 | 66% ± 1 | -5.22 ± 0.22 | 1.51 | 0/2 |

## Modelo final: validación

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_jp0.2) | 2 | 0 ± 0 | 60% ± 11 | -11.53 ± 0.24 | - | 0/2 |

## Mejor modelo (best_model.zip): niveles de entrenamiento

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_jp0.2) | 2 | 76 ± 1 | 82% ± 4 | -2.72 ± 0.21 | 1.38 | 0/2 |

## Mejor modelo (best_model.zip): validación

| agente | seeds | éxito (% fases) | progreso medio | retorno medio | saltos / oráculo (fases ganadas) | seeds que ganan en todas las fases |
|---|---|---|---|---|---|---|
| PPO (dict_fixed_lv-nacho_jp0.2) | 2 | 0 ± 0 | 39% ± 10 | -11.13 ± 0.60 | - | 0/2 |
