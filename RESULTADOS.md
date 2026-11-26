# Resultados experimentales

Resultados de los experimentos E1 a E8, con las corridas que los respaldan. La lista de experimentos sale de la
sección 18 de [`DOCUMENTACION.md`](DOCUMENTACION.md), más E8 (generalización). Las corridas se lanzaron con los scripts
de [`experimentos/`](experimentos/README.md); E6 es anterior.

## Cómo leer los números

- **Éxito:** porcentaje de las **10 fases de arranque** (x0 de 2.0 a 2.95) que el agente gana, con política
  determinista. Se reporta como media ± desvío entre seeds.
- **Mejor modelo** (`best_model.zip`): el de la evaluación periódica (cada 25k pasos) con más fases ganadas en los
  niveles de entrenamiento. **Modelo final** (`model.zip`): el del último paso.
- **s/o (saltos / oráculo):** saltos del agente dividido los del oráculo, sólo en fases ganadas. 1.0 es jugar óptimo;
  en `level_1` el oráculo usa 13 saltos.
- **1º 100%:** primera evaluación periódica en la que el agente gana las 10 fases, por seed.
- Cada corrida está en `runs/<corrida>/`:
  - `resumen.md`: tabla entre seeds;
  - `config.json`: todos los parámetros;
  - por seed, en `<algo>/seed<N>/`: `eval_history.csv` (evaluación periódica), `eval_fases_best.csv` (fases del mejor
    modelo), `eval_fases_best_val.csv` (niveles de validación) y los GIFs.
- **Salvo que se diga otra cosa:** PPO, 3 seeds, 300k pasos, `level_1`, recompensa por defecto (progreso + 1 al ganar −
  10 al morir) y fase de arranque fija en el entrenamiento.

## Resumen

| # | Pregunta | Respuesta corta | Corridas |
|---|---|---|---|
| E1 | ¿PPO o QR-DQN? | Los dos ganan `level_1`; PPO llega antes y QR-DQN salta un poco menos | `base_ppo`, `e1_qrdqn` |
| E2 | ¿Qué observación? | Con imagen + CNN aprende 2-3 veces más rápido; `dict` además es estable | `base_ppo`, `e2_obs_grid`, `e2_obs_dict` |
| E3 | ¿Castigo por salto fijo o con rampa? | Con muerte −10, un castigo de 0.1 no recorta los saltos: todos saltan ~2× el óptimo | `base_ppo`, `e3_fijo`, `e3_rampa` |
| E4 | ¿Memoriza la fase de arranque? | No: entrenado con una sola largada ya gana desde las 10 | `base_ppo`, `e4_random` |
| E5 | ¿Recompensa `salto_preciso`? | No se puede comparar: ninguna variante aprende los niveles de Santi | `e5_santi_jp`, `e5_santi_salto_preciso` |
| E6 | ¿Progresión de niveles? | No mejora frente a entrenar todo junto, y el olvido aparece igual | `dict_fixed_lv-nacho_*` |
| E7 | ¿La dificultad medida predice el aprendizaje? | Predice cuánto cuesta encontrar la meta; si se aprende depende del primer obstáculo | `e7_nivel_*` |
| E8 | ¿Generaliza a niveles nuevos? | Sí a uno fácil y a uno medio (100%); no al difícil (0%) | `e8_generalizacion` |

---

## E1. Algoritmos: PPO contra QR-DQN

**Configuración:** observación `features`, `level_1`. Sólo PPO y QR-DQN (A2C y DQN se descartaron del plan).

| Corrida | Éxito mejor | s/o mejor | Éxito final | 1º 100% (seeds) |
|---|---|---|---|---|
| PPO | 100 ± 0 | 2.08 | 100 ± 0 | 175k, 250k, 175k |
| QR-DQN | 100 ± 0 | 1.86 | 97 ± 5 | 275k, 300k, 200k |

**Conclusiones:**
- **Los dos aprenden a ganar `level_1` desde las 10 largadas** (mejor modelo, 3/3 seeds).
  → [`runs/base_ppo/resumen.md`](runs/base_ppo/resumen.md), [`runs/e1_qrdqn/resumen.md`](runs/e1_qrdqn/resumen.md)
- **PPO llega antes:** ~200k pasos de media contra ~258k de QR-DQN.
  → `runs/base_ppo/ppo/seed*/eval_history.csv`, `runs/e1_qrdqn/qrdqn/seed*/eval_history.csv`
- **QR-DQN salta un poco menos** (1.86× contra 2.08× el óptimo), pero su modelo final es menos estable (97 ± 5%: una
  seed pierde una largada al final).
  → [`runs/e1_qrdqn/resumen.md`](runs/e1_qrdqn/resumen.md)

## E2. Observaciones: `features`, `grid` y `dict`

**Configuración:** PPO, `level_1`. `features` es la misma corrida que la base de E1.

| Observación | Red | Éxito mejor | s/o mejor | Éxito final | 1º 100% (seeds) |
|---|---|---|---|---|---|
| `features` | MLP | 100 ± 0 | 2.08 | 100 ± 0 | 175k, 250k, 175k |
| `grid` | CNN | 100 ± 0 | 1.92 | 67 ± 47 | 125k, 125k, 50k |
| `dict` | CNN + MLP | 100 ± 0 | 2.23 | 100 ± 0 | 75k, 75k, 75k |

**Conclusiones:**
- **Con la imagen y una CNN, PPO aprende 2 a 3 veces más rápido.** `dict` gana las 10 largadas a los 75k pasos en las
  3 seeds; `features` tarda entre 175k y 250k.
  → `runs/e2_obs_dict/ppo/seed*/eval_history.csv`, `runs/base_ppo/ppo/seed*/eval_history.csv`
- **`grid` es rápida pero inestable:** el mejor modelo gana todo, pero el modelo final de una seed termina perdiendo
  (67 ± 47%). Le falta la velocidad vertical y la posición dentro del bloque (2-3 ticks seguidos ven la misma imagen).
  → [`runs/e2_obs_grid/resumen.md`](runs/e2_obs_grid/resumen.md)
- **`dict` es rápida y estable** (100% también con el modelo final).
  → [`runs/e2_obs_dict/resumen.md`](runs/e2_obs_dict/resumen.md)
- **En `level_1` no se nota la limitación de `features` con techos,** porque ese nivel no tiene columnas ambiguas. Es
  la razón por la que E7 usa sólo niveles sin techo y E8 usa `dict`.

## E3. Penalización por salto: sin castigo, fija y con rampa

**Configuración:** PPO, `features`, `level_1`. `c = 0.1` fijo contra `c = 0.1` con rampa (`--ramp`: arranca en 0 y
sube 0.01 cada vez que gana ≥ 80% de los últimos 20 episodios).

| Corrida | Éxito mejor | s/o mejor | Éxito final | c al final |
|---|---|---|---|---|
| sin castigo | 100 ± 0 | 2.08 | 100 ± 0 | 0 |
| c = 0.1 fijo | 70 ± 42 | 2.02 | 70 ± 42 | 0.1 |
| c = 0.1 con rampa | 100 ± 0 | 2.06 | 87 ± 19 | 0.05, 0.01, 0.06 |

**Conclusiones:**
- **Todos los agentes saltan alrededor del doble de lo necesario,** con o sin castigo. El mejor PPO (seed 0) gana
  `level_1` con **27 saltos**, contra 13 del oráculo.
  → [`runs/base_ppo/resumen.md`](runs/base_ppo/resumen.md); GIF: `runs/base_ppo/ppo/seed0/mejor_episodio_level_1_x0_2.00.gif`
- **Con muerte = −10, un castigo de 0.1 por salto no alcanza.** 14 saltos de más cuestan 1.4: para el agente, saltar
  siempre sigue siendo el seguro más barato.
- **El castigo fijo además empeora el aprendizaje:** una de las 3 seeds no llega a ganar las 10 largadas.
  → [`runs/e3_fijo/resumen.md`](runs/e3_fijo/resumen.md)
- **La rampa subió tarde y poco:** en 300k pasos `c` llegó a 0.05, 0.01 y 0.06; nunca a 0.1. No es una prueba justa de
  la rampa con `c = 0.1`.
  → `runs/e3_rampa/ppo/seed*/jump_penalty.csv`
- **Con un castigo más grande sí baja.** En los niveles de Nacho (E6, `dict`), `c = 0.2` lleva s/o de 1.62 a 1.38 con
  el mismo éxito.
  → [`runs/dict_fixed_lv-nacho_jp0.2/resumen.md`](runs/dict_fixed_lv-nacho_jp0.2/resumen.md) contra
  [`runs/dict_fixed_lv-nacho_jp0.05/resumen.md`](runs/dict_fixed_lv-nacho_jp0.05/resumen.md)

## E4. Fase de arranque fija contra aleatoria

**Configuración:** PPO, `features`, `level_1`. Fase fija = siempre x0 = 2.0 (la base); aleatoria = x0 sorteado en cada
episodio de entrenamiento. Las dos se evalúan en las mismas 10 fases.

| Entrenamiento | Éxito mejor | s/o mejor | Éxito final | 1º 100% (seeds) |
|---|---|---|---|---|
| fase fija | 100 ± 0 | 2.08 | 100 ± 0 | 175k, 250k, 175k |
| fase aleatoria | 100 ± 0 | 2.02 | 100 ± 0 | 200k, 225k, 175k |

**Conclusión:** el agente entrenado siempre desde x0 = 2.0 ya gana desde las 10 largadas: **no memoriza en qué tick
llega a cada obstáculo**. Sortear la largada no cambia el resultado ni la velocidad.
→ [`runs/base_ppo/resumen.md`](runs/base_ppo/resumen.md), [`runs/e4_random/resumen.md`](runs/e4_random/resumen.md)

## E5. Recompensa `salto_preciso` en los niveles de Santi

**Configuración:** PPO, `features`, los 5 niveles de Santi
([`niveles/santi.json`](niveles/santi.json), valida en `level_1`). Castigo fijo de 0.05 contra
`--rewards progress win death salto_preciso`.

| Recompensa | Éxito mejor (entren.) | Progreso | `level_1` (validación) |
|---|---|---|---|
| castigo fijo 0.05 | 7 ± 9 | 26% | 0% |
| `salto_preciso` | 3 ± 4 | 32% | 0% |

**Conclusión:** **resultado negativo.** Con `features` y 300k pasos ninguna de las dos recompensas aprende esos niveles,
así que no se pueden comparar. Coincide con E7, donde `santi_intermedio_1`, `santi_intermedio_2` y `santi_dificil`
tampoco se aprenden por separado.
→ [`runs/e5_santi_jp/resumen.md`](runs/e5_santi_jp/resumen.md),
[`runs/e5_santi_salto_preciso/resumen.md`](runs/e5_santi_salto_preciso/resumen.md)

## E6. Progresión de niveles

**Configuración:** PPO, `dict`, `c = 0.05`, **2 seeds**, los 4 niveles de Nacho, valida en `level_1`.
- Con progresión ([`niveles/nacho_progresion.json`](niveles/nacho_progresion.json)): fácil → medio → difícil + muy
  difícil. Pasa de etapa con 80% de éxito en los últimos 50 episodios; cada nivel pesa
  `base × máx(0.2, 1 − éxito)`.
- Sin progresión ([`niveles/nacho.json`](niveles/nacho.json)): los 4 desde el principio.

| Corrida | Éxito mejor | Éxito final | s/o mejor | `level_1` mejor / final |
|---|---|---|---|---|
| sin progresión | 82 ± 8 | 68 ± 2 | 1.62 | 0% / 0% |
| con progresión | 76 ± 1 | 69 ± 6 | 1.67 | 0% / 50 ± 50% |
| sin progresión, c = 0.2 | 76 ± 1 | 51 ± 1 | 1.38 | 0% / 0% |

Por nivel (mejor modelo): fácil, medio y difícil 100% en las dos; muy difícil 30% sin progresión y 5% con.

**Conclusiones:**
- **La progresión funciona como mecanismo:** llega a la última etapa en las 2 seeds (a los 97k y 120k pasos).
  → `runs/dict_fixed_lv-nacho_progresion_prog_jp0.05/ppo/seed*/progresion.csv`
- **No mejora frente a entrenar todo junto** (76% contra 82%).
  → [`runs/dict_fixed_lv-nacho_progresion_prog_jp0.05/resumen.md`](runs/dict_fixed_lv-nacho_progresion_prog_jp0.05/resumen.md),
  [`runs/dict_fixed_lv-nacho_jp0.05/resumen.md`](runs/dict_fixed_lv-nacho_jp0.05/resumen.md)
- **El olvido aparece aun sin progresión:** una seed llegó a 60% en `nacho_muy_dificil` a los 125k pasos y en la
  evaluación siguiente volvió a 0.
  → `runs/dict_fixed_lv-nacho_jp0.05/ppo/seed1/eval_history.csv` (columna `success_nacho_muy_dificil`)
- **Con 4 niveles de entrenamiento no generaliza a `level_1`:** el mejor modelo saca 0% en todas las corridas. El 50%
  del modelo final con progresión es 1 seed de 2, y gana con 26 saltos (el doble del óptimo). E8 muestra que con 20
  niveles sí generaliza.
- **Son 2 seeds:** son tendencias, no conclusiones firmes.

## E7. Dificultad medida contra desempeño

**Configuración:** PPO, `features`, **un nivel por corrida**, 200k pasos, 12 niveles sin techo (los que `features`
puede representar). La dificultad es la de [`scripts/difficulty.py`](scripts/difficulty.py), en el peor caso entre las
10 fases.

| Nivel | log10 P(azar) | Ventana mín. | Éxito mejor | 1ª victoria (entren.) | Dónde muere |
|---|---|---|---|---|---|
| `franco_facil` | −2.0 | 8 | 80 ± 0 | 2k | |
| `nacho_facil` | −3.7 | 7 | 100 ± 0 | 13k | |
| `level_1` | −4.6 | 2 | 53 ± 41 | 66k | (con 300k pasos, en E1: 100%) |
| `facil` | −6.2 | 7 | 0 | nunca | escalón de la columna 7 |
| `santi_intermedio_1` | −6.7 | 1 | 3 ± 5 | 67k | (progreso 86%) |
| `medio` | −6.8 | 3 | 70 ± 42 | 93k | |
| `nacho_medio` | −7.1 | 5 | 0 | nunca | pozo de 3 de la columna 9 |
| `dificil` | −8.2 | 2 | 0 | nunca | escalón de la columna 7 |
| `nacho_dificil` | −8.8 | 3 | 100 ± 0 | 27k | |
| `santi_intermedio_2` | −10.9 | 1 | 0 | nunca | |
| `nacho_muy_dificil` | −12.8 | 1 | 0 | nunca | |
| `santi_dificil` | −14.0 | 1 | 0 | nunca | (progreso 86%) |

Correlaciones de Spearman entre niveles (n = 12, valor p por permutaciones):

| Medida de dificultad | contra | ρ | p |
|---|---|---|---|
| log10 P(ganar al azar) | pasos hasta la 1ª victoria | −0.69 | 0.02 |
| log10 P(ganar al azar) | éxito del mejor modelo | +0.56 | 0.06 |
| ventana media | pasos hasta ganar el 50% en entrenamiento | −0.61 | 0.04 |
| trampas, cadenas de saltos | cualquier medida de desempeño | \|ρ\| ≤ 0.3 | ≥ 0.37 |

**Conclusiones:**
- **La probabilidad de ganar al azar predice cuánto cuesta encontrar la meta** (la primera victoria). Es la única
  relación clara; con el éxito final queda al borde de la significancia.
- **Las trampas y las cadenas de saltos no predicen nada** del desempeño del agente.
- **Las excepciones se explican por el primer obstáculo.** En `facil` y `dificil` el agente choca contra el escalón de
  la columna 7, y en `nacho_medio` cae en el pozo de la columna 9, en las 30 evaluaciones. Nunca llega a ver el resto
  del nivel, y la métrica mide el nivel entero.
  → `runs/e7_nivel_facil/ppo/seed*/eval_fases_best.csv` (columnas `status` y `death_x`), ídem `e7_nivel_dificil` y
  `e7_nivel_nacho_medio`
- **No son niveles imposibles:** el agente de E8 (con `dict` y 20 niveles) gana `facil` y `nacho_medio` al 100%. Es un
  problema de exploración con un solo nivel y `features`.
- **Tabla y gráficos completos:** `python scripts/difficulty_vs_performance.py 'runs/e7_nivel_*'` (salen en
  `figuras/dificultad_vs_desempeno/`).
  → `runs/e7_nivel_*/resumen.md`

## E8. Generalización a niveles nuevos

**Configuración:** PPO, `dict`, `c = 0.05`, 600k pasos.
- Entrena en 20 niveles con el mismo peso y valida en 3 que nunca ve: `santi_facil` (fácil), `level_1` (medio) y
  `nacho_dificil` (difícil) ([`niveles/generalizacion.json`](niveles/generalizacion.json)).
- Usa `dict` porque entre los 20 están los niveles con techo.

| | Éxito mejor | Éxito final | s/o mejor |
|---|---|---|---|
| 20 niveles de entrenamiento | 61 ± 3 | 49 ± 8 | 1.93 |
| validación (3 niveles) | 67 ± 0 | 67 ± 0 | 1.99 |
| — `santi_facil` (fácil) | 100 (3/3 seeds) | 100 | 2.04 |
| — `level_1` (medio) | 100 (3/3 seeds) | 100 | 1.94 |
| — `nacho_difícil` (difícil) | 0 | 0 | — (progreso 29-50%) |

En entrenamiento, el mejor modelo:
- **gana todo** en `facil`, `nacho_facil`, `nacho_medio`, `franco_facil`, `franco_medio`, `franco_dificil`,
  `franco_rutas`, `franco_tunel` y `franco_sin_saltar`;
- **gana en parte** `santi_intermedio_dificil` (90%), `franco_pasillos` (63%), `franco_techo` (60%), `dificil` y
  `medio` (33%), `santi_intermedio_1` (27%) y `nacho_muy_dificil` (7%);
- **no gana nunca** `muy_dificil`, `muy_muy_dificil`, `santi_dificil` ni `santi_intermedio_2`.

**Conclusiones:**
- **Generaliza a niveles nuevos fáciles y medios:** `santi_facil` y `level_1` se ganan desde las 10 largadas, en las 3
  seeds, sin haberlos visto nunca.
  → `runs/e8_generalizacion/ppo/seed*/eval_fases_best_val.csv`; GIF:
  `runs/e8_generalizacion/ppo/seed0/val_mejor_episodio_level_1_x0_2.00.gif`
- **No generaliza al difícil:** `nacho_dificil` no se gana (llega a entre 29% y 50% del nivel). Una seed lo ganó en
  una sola evaluación (a los 550k pasos) y lo perdió en la siguiente.
  → `runs/e8_generalizacion/ppo/seed1/eval_history.csv` (columna `val_success_nacho_dificil`)
- **La validación sale más alta que el entrenamiento** (67% contra 61%) porque entre los 20 de entrenamiento hay
  varios muy difíciles que no aprende.
- **Ve techos:** gana los niveles de Franco con techo o túnel (`franco_tunel`, `franco_rutas` 100%; `franco_techo`
  60%), que `features` no puede representar.
- **Comparado con E6** (4 niveles, `level_1` 0%), entrenar en más niveles es lo que permite generalizar.
  → [`runs/e8_generalizacion/resumen.md`](runs/e8_generalizacion/resumen.md)

---

## Conclusiones generales

1. **Algoritmo:** PPO y QR-DQN aprenden a ganar `level_1`; PPO es más rápido y QR-DQN salta un poco menos (E1).
2. **Representación:** pesa más que el algoritmo. Con la imagen y una CNN el aprendizaje es 2-3 veces más rápido, y
   `dict` (imagen + estado del cubo) es además estable (E2).
3. **Saltar de más:** ningún agente lo resolvió; todos saltan ~2× el óptimo. El castigo por salto tiene que pesar
   frente al costo de morir para que cambie algo (E3, E6 con `c = 0.2`).
4. **Varios niveles:** la progresión por etapas no mejoró y el olvido aparece igual (E6).
5. **Generalización:** dentro de un nivel no memoriza la largada (E4). Entrenado en 20 niveles, gana niveles nuevos
   fáciles y medios, pero no el difícil (E8).
6. **Dificultad:** la probabilidad de ganar al azar predice cuánto cuesta encontrar la meta; si un nivel se aprende
   depende sobre todo de su primer obstáculo (E7).

## Limitaciones

- **2-3 seeds por configuración:** los desvíos son grandes (por ejemplo, 70 ± 42%) y varias diferencias son tendencias.
- **200-600k pasos:** algunas cosas no tuvieron tiempo; por ejemplo, la rampa de E3 no llegó a `c = 0.1` y `level_1` en
  E7 (200k) no siempre se aprende.
- **Evaluación determinista en 10 fases:** un nivel puede pasar de 0% a 100% entre dos evaluaciones seguidas.
- **El mejor modelo se elige con los niveles de entrenamiento,** no con los de validación (en E6 el mejor modelo valida
  peor que el final).
- **E5 no permite comparar las recompensas,** porque ninguna aprende esos niveles.
