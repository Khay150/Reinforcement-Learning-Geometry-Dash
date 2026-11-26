# Documentación del proyecto: Geometry Dash con Aprendizaje por Refuerzo

Este documento explica el proyecto completo: **qué problema resuelve, los conceptos de Aprendizaje por Refuerzo
(RL) que usa, cómo está construido cada módulo y por qué se tomó cada decisión**. Para instalar y correr las cosas
paso a paso está [SETUP.md](SETUP.md); para una referencia rápida de comandos y la lista completa de opciones, el
[README](README.md).

## Índice

1. [El problema en una página](#1-el-problema-en-una-página)
2. [Conceptos de Aprendizaje por Refuerzo](#2-conceptos-de-aprendizaje-por-refuerzo)
3. [Arquitectura del repositorio](#3-arquitectura-del-repositorio)
4. [La física del juego (`gd_env/physics.py`)](#4-la-física-del-juego-gd_envphysicspy)
5. [El nivel (`gd_env/levels.py`)](#5-el-nivel-gd_envlevelspy)
6. [Niveles registrados](#6-niveles-registrados)
7. [El entorno Gymnasium (`gd_env/env.py`)](#7-el-entorno-gymnasium-gd_envenvpy)
8. [El oráculo: la solución óptima exacta (`gd_env/oracle.py`)](#8-el-oráculo-la-solución-óptima-exacta-gd_envoraclepy)
9. [Dificultad de un nivel (`gd_env/difficulty.py`)](#9-dificultad-de-un-nivel-gd_envdifficultypy)
10. [Visualización (`gd_env/render.py`)](#10-visualización-gd_envrenderpy)
11. [Los agentes: Stable-Baselines3, PPO, A2C, DQN, QR-DQN y las redes](#11-los-agentes-stable-baselines3-ppo-a2c-dqn-qr-dqn-y-las-redes)
12. [Cómo se arma el entrenamiento](#12-cómo-se-arma-el-entrenamiento)
13. [Evaluación: fases de arranque y métricas](#13-evaluación-fases-de-arranque-y-métricas)
14. [El costo de saltar: penalización, rampa, bonus por seguir vivo y salto preciso](#14-el-costo-de-saltar-penalización-rampa-bonus-por-seguir-vivo-y-salto-preciso)
15. [Guardado del mejor modelo](#15-guardado-del-mejor-modelo)
16. [Continuar entrenando un modelo](#16-continuar-entrenando-un-modelo)
17. [Scripts y archivos de salida](#17-scripts-y-archivos-de-salida)
18. [Resultados experimentales](#18-resultados-experimentales)
19. [Tests](#19-tests)
20. [Limitaciones y trabajo futuro](#20-limitaciones-y-trabajo-futuro)
21. [Glosario](#21-glosario)

---

## 1. El problema en una página

En Geometry Dash un cubo avanza solo hacia la derecha a velocidad constante. El jugador **sólo decide cuándo
saltar**. Si toca un pincho, choca de costado contra un bloque o cae en un pozo, pierde. Gana si llega al final.

El proyecto arma una versión minimalista y **determinista** del juego (el "modo cubo") y la usa como laboratorio de RL:

```
   ┌───────────────┐   acción (0 = nada, 1 = saltar)   ┌──────────────────────┐
   │    AGENTE     │ ────────────────────────────────▶ │       ENTORNO        │
   │ (PPO/A2C/DQN/ │                                   │  (GeometryDashEnv)   │
   │  QR-DQN)      │ ◀──────────────────────────────── │  física + uno o      │
   │  red MLP o CNN│                                   │  varios niveles fijos│
   └───────────────┘   observación + recompensa        └──────────────────────┘
```

Tiene cuatro piezas:

| Pieza | Qué es | Para qué sirve |
|---|---|---|
| **Entorno** (`gd_env/`) | Simulador del juego con la API estándar de Gymnasium, con 23 niveles registrados | Es el "mundo" con el que interactúa el agente |
| **Oráculo** (`gd_env/oracle.py`) | Un planificador que conoce la física y encuentra la solución perfecta | Prueba que cada nivel se puede ganar y da el techo de desempeño (13 saltos en `level_1`) |
| **Medida de dificultad** (`gd_env/difficulty.py`) | El mismo grafo de estados que el oráculo, analizado salto por salto | Dice qué tan difícil es un nivel **antes** de entrenar: precisión, trampas, saltos encadenados, P(ganar al azar) |
| **Agentes** (`agents/`, `scripts/`) | PPO, A2C, DQN (Stable-Baselines3) y QR-DQN (sb3-contrib) | Aprenden a jugar **sin** conocer la física, sólo por prueba y error |

La pregunta de fondo es: **¿cuánto se acerca un agente que aprende por prueba y error a la solución óptima que
calcula el oráculo?** Se mide en fases ganadas (¿termina el nivel?), en saltos usados (¿juega limpio o salta de más?)
y, cuando se entrena con varios niveles, en niveles que nunca vio (¿aprendió a jugar o memorizó?).

---

## 2. Conceptos de Aprendizaje por Refuerzo

### 2.1 El ciclo agente-entorno

En RL un **agente** interactúa con un **entorno** en pasos discretos. En cada paso `t`:

1. el agente recibe una **observación** `oₜ` (lo que ve del estado del mundo);
2. elige una **acción** `aₜ` según su **política** `π(a | o)`;
3. el entorno avanza, devuelve una **recompensa** `rₜ` (un número) y la observación siguiente.

Una secuencia completa desde el inicio hasta el final (ganar, morir o agotar el tiempo) es un **episodio**. Como el
cubo avanza a velocidad constante, un episodio ganado dura siempre lo mismo: **520 pasos (ticks)** en `level_1` (110
columnas) y **270** en los niveles de 60 columnas.

### 2.2 Proceso de decisión de Markov (MDP)

Formalmente el problema es un MDP: estados `S`, acciones `A`, una dinámica `P(s' | s, a)` y una recompensa `R(s, a, s')`.
La **propiedad de Markov** dice que el futuro depende sólo del estado actual, no de cómo se llegó a él.

En este juego el estado completo es `(t, x, y, vy, en_suelo)` más el nivel que se está jugando. La dinámica es
**determinista**: la misma acción en el mismo estado da siempre el mismo resultado. Eso simplifica mucho el problema y
permite construir el oráculo y medir la dificultad de forma exacta.

> **Observación ≠ estado.** El agente no ve el nivel entero: ve una ventana de 12 a 16 columnas hacia adelante
> (sección 7.2). Si la observación no alcanza para decidir bien, el problema pasa a ser *parcialmente observable*
> (POMDP). Por eso la observación `features` incluye la velocidad vertical: con sólo la posición no se puede saber si
> el cubo sube o baja.

### 2.3 Retorno y factor de descuento γ

El agente no maximiza la recompensa inmediata sino el **retorno**, la suma descontada de recompensas futuras:

```
Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + ...        con 0 ≤ γ < 1
```

`γ` (gamma) define cuánto importa el futuro. Con `γ = 0.99`, una recompensa a `k` pasos vale `0.99ᵏ`:

| pasos en el futuro | peso `0.99ᵏ` |
|---|---|
| 10 | 0.90 |
| 100 | 0.37 |
| 270 (fin de un nivel de 60 columnas) | 0.07 |
| 520 (fin de `level_1`) | 0.005 |

Una regla práctica es que el **horizonte efectivo** es `1/(1−γ) = 100` pasos. Es decir, el agente "ve venir" las
consecuencias de unos 100 ticks (≈ 20 tiles), no las del nivel entero. Por eso la recompensa por ganar (+1 al final)
influye poco al principio del nivel, y lo que realmente guía al agente son la recompensa por progreso y la
penalización por morir, que llegan pronto.

### 2.4 Funciones de valor

- **`V(s)`**: retorno esperado si se arranca en `s` y se sigue la política.
- **`Q(s, a)`**: retorno esperado si en `s` se hace `a` y después se sigue la política.

Si se conoce `Q`, la mejor política es elegir `argmaxₐ Q(s, a)`. DQN y QR-DQN aprenden `Q` directamente (métodos
**basados en valor**); PPO y A2C aprenden la política y usan `V` como referencia (métodos **actor-crítico**) para saber
si una acción salió "mejor o peor de lo esperado" (la **ventaja** `A(s, a) = Q(s, a) − V(s)`).

### 2.5 Exploración vs. explotación

Para descubrir que saltar antes de un pincho es bueno, el agente tiene que **probar** acciones cuyo resultado todavía
no conoce (explorar). Pero para ganar tiene que **usar** lo que ya sabe (explotar). Los algoritmos resuelven esto
de forma distinta:

- **PPO y A2C** tienen una política estocástica (muestrean acciones con probabilidades) y un bono de **entropía**
  (`ent_coef`) que la empuja a no volverse determinista demasiado pronto.
- **DQN y QR-DQN** usan **ε-greedy**: con probabilidad ε elige una acción al azar. ε baja de 1.0 a 0.05 durante el 30%
  inicial del entrenamiento.

Gran parte de los problemas observados en este proyecto son **de exploración**, no de diseño de la recompensa
(ver sección 14.3). La medida de dificultad lo cuantifica: la probabilidad de ganar apretando al azar (sección 9) es,
aproximadamente, la chance de que un agente recién inicializado vea una victoria.

### 2.6 Recompensa escasa vs. moldeada (reward shaping)

Si la única recompensa fuera "+1 al ganar", el agente casi nunca la vería: una política al azar gana `level_1` una
vez cada ~40 000 episodios (sección 9). Por eso se usa **reward shaping**: se agrega una recompensa intermedia por
progreso para que cada tile avanzado dé señal. La **penalización por salto** también es shaping: cambia *qué*
soluciones prefiere el agente (menos saltos) sin cambiar *cuáles* ganan. La recompensa de **salto preciso** (sección
14.6) va un paso más allá: el costo de saltar depende de si había algo que saltar.

En el código la recompensa está armada con **términos** que se prenden y apagan (sección 7.3), así cada experimento
puede decir exactamente qué piezas de shaping usó.

### 2.7 Asignación de crédito (credit assignment)

Cuando el agente muere en un pincho, ¿la culpa es del último paso o de no haber saltado 5 ticks antes? Repartir el
mérito o la culpa entre acciones pasadas es el problema de asignación de crédito. El descuento `γ`, los retornos de
varios pasos y GAE (sección 11.2) existen para resolverlo. En este juego hay ventanas de despegue muy chicas (2 ticks
en `level_1`, 1 tick en varios niveles nuevos) y **trampas**: saltos que aterrizan bien pero dejan el nivel imposible
más adelante (sección 9). Las dos cosas hacen que la asignación de crédito tenga que ser precisa.

### 2.8 Generalización

Un agente puede "memorizar" la secuencia exacta de acciones de un episodio sin entender el juego. Para detectarlo, el
proyecto evalúa en **fases de arranque** distintas (el cubo empieza un poco corrido, sección 13) y, con varios niveles,
en **niveles de validación** que nunca jugó durante el entrenamiento (sección 12.5). Si el agente aprendió a reaccionar
a los obstáculos, gana igual; si memorizó ticks o niveles, falla.

---

## 3. Arquitectura del repositorio

```
Reinforcement-Learning-Geometry-Dash/
├── gd_env/                 EL JUEGO (no depende de RL)
│   ├── physics.py          dinámica pura: gravedad, salto, colisiones, muerte, pinchos colgantes
│   ├── levels.py           23 niveles, registro LEVELS y resolución de conjuntos de niveles (--levels)
│   ├── env.py              GeometryDashEnv: la API de Gymnasium sobre la física
│   ├── rewards.py          Transition, términos de la recompensa (REWARD_TERMS), salto preciso, recompensas propias
│   ├── oracle.py           planificador exacto (programación dinámica) + OraclePolicy
│   ├── difficulty.py       dificultad de un nivel medida con la física (level_difficulty)
│   ├── render.py           frames, GIF del episodio y mapa del recorrido
│   └── __init__.py         exporta todo y registra "GeometryDash-v0"
├── agents/                 RL CON STABLE-BASELINES3
│   ├── common.py           algoritmos (ALGOS), entornos de entrenamiento, construcción y carga de modelos
│   ├── cnn.py              extractores CNN propios para las observaciones grid y dict
│   ├── training.py         train(...): toda la lógica de una corrida (validar, entrenar, guardar, evaluar, resumir)
│   ├── runs.py             utilidades de corridas: recompensa y niveles desde config.json, evaluar y grabar GIFs
│   ├── callbacks.py        rampa de la penalización, progresión de niveles y guardado del mejor modelo
│   └── evaluate.py         evaluación por fases, métricas, CSV y tabla resumen
├── scripts/                PUNTOS DE ENTRADA (línea de comandos)
│   ├── train.py            parser de argumentos que llama a agents.train(...)
│   ├── evaluate.py         re-evalúa modelos guardados
│   ├── record.py           GIF de un episodio (oráculo o modelo)
│   ├── run_oracle.py       resuelve el nivel con el oráculo y lo reproduce
│   └── difficulty.py       tabla de dificultad de los niveles (sin entrenar)
├── ejemplos/               recompensas propias para --reward-fn (reward_ejemplo.py, reward_santi.py)
├── niveles/                conjuntos de niveles para --levels (ejemplo.json, nacho.json, nacho_progresion.json,
│   │                       santi.json)
│   └── png/                una imagen por nivel registrado + _todos_los_niveles.png
├── tests/                  test_env.py, test_cnn.py, test_difficulty.py, test_spike_down.py
├── runs/                   resultados de los entrenamientos (una carpeta por corrida)
└── videos/                 GIFs de ejemplo
```

**Principio de diseño central:** la física vive en un único módulo de funciones puras (`physics.py`) y la usan
*tanto* el entorno *como* el oráculo *como* la medida de dificultad. Así el oráculo planifica sobre exactamente la
misma dinámica que experimenta el agente, y "el oráculo gana con 13 saltos" o "este salto tiene una ventana de 2
ticks" son afirmaciones sobre el entorno real, no sobre una aproximación.

El flujo de dependencias va en un solo sentido: `physics → levels/rewards → env/oracle/difficulty/render → agents →
scripts`. `gd_env` no sabe nada de RL; se podría usar con cualquier otra librería.

---

## 4. La física del juego (`gd_env/physics.py`)

### 4.1 Convenciones

- **Unidad de longitud = 1 tile.** El cubo mide 1×1.
- `x` = borde izquierdo del cubo, `y` = borde inferior. El piso es la fila 0 de la grilla y su cara superior está en
  `y = 1`, así que el cubo apoyado en el piso tiene `y = 1`.
- **Un tick = un paso de simulación = un paso del entorno.**

### 4.2 Parámetros (`Physics`)

| Parámetro | Valor | Significado |
|---|---|---|
| `vx` | 0.2 tiles/tick | avance horizontal, constante e incontrolable |
| `gravity` | 0.05 tiles/tick² | cuánto baja `vy` por tick |
| `jump_v` | 0.4 tiles/tick | velocidad vertical al despegar |
| `max_fall` | 0.6 tiles/tick | velocidad máxima de caída (< 1 para no atravesar tiles en un tick) |
| `spike_w`, `spike_h` | 0.3 | hitbox del pincho: un cuadrado de 0.3×0.3 centrado en el tile (abajo en un pincho común, arriba en uno colgante) |

Con estos números un salto en terreno plano dura **16 ticks**, sube **1.4 tiles** y avanza **3.2 tiles**.
Alcanza para pasar un pincho o un pozo de 2 tiles, y para subir a un bloque de altura 1.

### 4.3 Un tick, paso a paso (`step`)

```
1. Velocidad vertical: si la acción es 1 y el cubo está apoyado → vy = jump_v.
   Después se aplica gravedad: vy = max(vy − gravity, −max_fall).
2. Avance horizontal: x = x0 + vx·t  (se calcula desde el inicio, sin acumular error numérico).
   Si en la nueva x el cubo se solapa con un bloque → CRASH (choque de costado).
3. Movimiento vertical: y += vy. Si se solapa con un bloque:
   - bajando → aterriza encima (on_ground = True);
   - subiendo → golpea el techo con la cabeza y queda debajo (vy = 0, no muere).
4. Chequeos: pincho (común o colgante) → SPIKE; y < −1 → FELL (cayó al pozo); x ≥ meta → WIN; si no, OK.
```

Detalles importantes:

- **`x` depende sólo del tick.** No hay acción para frenar o acelerar. Esto es lo que convierte al espacio de estados
  en un grafo por capas (sección 8).
- **Saltar en el aire no hace nada.** La acción 1 equivale a "mantener apretado": si el agente sigue pidiendo 1, el
  cubo vuelve a saltar apenas aterriza.
- **Golpear un bloque con la cabeza no mata**, sólo corta la subida. Por eso funcionan los techos y las plataformas
  flotantes sin cambiar la física: lo que mata es chocar de costado o tocar un pincho.
- **Pinchos colgantes (`SPIKE_DOWN`).** Cuelgan de un bloque y apuntan hacia abajo. Su hitbox es la del pincho común
  espejada (0.3×0.3 en la parte **alta** del tile). Con un hueco de 1 tile debajo, el cubo pasa caminando pero muere
  si salta (la cabeza entra en la hitbox). Rozar la punta, como en el pincho común, no mata.
- **Morir es terminal y absorbente:** el cubo queda congelado donde murió.

---

## 5. El nivel (`gd_env/levels.py`)

El nivel es una grilla `grid[fila, columna]` con fila 0 = piso. Cada celda es `EMPTY` (0), `BLOCK` (1), `SPIKE` (2) o
`SPIKE_DOWN` (3, pincho colgante). `level_1()` mide 110 columnas × 7 filas; el cubo arranca en `x = 2` y gana al
llegar a `x = 106`:

```
......................................................###.....................................................
..............^.........^^........###...^...........#####.........^^.......###........^.........^....#........
##############################################..################################..############################
```

(Se omiten las 4 filas superiores, vacías. `#` bloque, `^` pincho, `v` pincho colgante, `.` vacío; un `.` en la última
línea es un pozo.)

Obstáculos en orden: pincho simple, doble pincho, escalón de 1, pincho, **pozo de 2** (x = 46), escalera 1→2, doble
pincho, plataforma baja, pozo de 2 (x = 80), dos pinchos y un último bloque.

Hay dos formas de definir niveles nuevos:

```python
from gd_env import parse_level, build_level
# 1) dibujándolo en ASCII (admite cualquier cosa: techos, plataformas flotantes, pinchos colgantes 'v')
lvl = parse_level("""
.......
..^..#.
###.###
""")
# 2) por listas de obstáculos (sólo cosas apoyadas en el piso: pinchos, columnas de bloques, pozos)
lvl = build_level(length=60, height=7, spikes=[14, 30], blocks=[(20, 1)], pits=[(40, 2)])
```

`parse_level` pone la meta `goal_margin = 4` columnas antes del final del dibujo: en un nivel de 60 columnas, la meta
está en `x = 56`. `to_ascii(level)` hace la conversión inversa (es lo que imprime `run_oracle.py --ascii` y
`difficulty.py --ascii`).

**Techos y plataformas flotantes.** Como el dibujo ASCII admite bloques en cualquier fila, se pueden armar techos
(`muy_dificil`), plataformas en el aire con pinchos colgando debajo (`franco_techo`, `franco_tunel`,
`franco_pasillos`), dos rutas a distinta altura (`franco_rutas`) o bloques y pinchos sueltos en el aire
(`muy_muy_dificil`). La física no necesitó cambios para eso (sección 4.3); lo único nuevo fue el pincho colgante.

Para poder usar un nivel desde `--levels`, se registra con un nombre en el diccionario `LEVELS` de
`gd_env/levels.py` (`get_level(nombre)` lo construye). Hoy hay 23, descriptos en la sección 6. La sección 12.5 explica
cómo se entrena con varios.

> La altura `H = 7` importa porque fija el tamaño de la observación `grid` (3×7×16) y la normalización de las alturas
> en `features`. Niveles con otra altura cambian la forma de la observación, y una red entrenada no podría usarlos.
> Por eso todos los niveles registrados miden 7 de alto y arrancan en `x = 2.0`.

---

## 6. Niveles registrados

Cada integrante del grupo diseñó una familia de niveles con una idea distinta. Todos tienen 7 filas; `level_1` mide
110 columnas y el resto 60. `tests/test_env.py` comprueba con el oráculo que **todos se ganan desde las 10 fases de
evaluación**. Hay una imagen de cada uno en [`niveles/png/`](niveles/png/) (línea verde = inicio, amarilla = meta) y
todos juntos en [`niveles/png/_todos_los_niveles.png`](niveles/png/_todos_los_niveles.png), pensadas para el informe y
la presentación. Para verlos en texto:
`python -c "from gd_env import get_level, to_ascii; print(to_ascii(get_level('franco_techo')))"`.

### 6.1 Las familias

- **`level_1` (TP).** El nivel original: un poco de todo (pinchos simples y dobles, escalón, pozos, escalera,
  plataforma baja). Es el de referencia y el que se usa como **validación** en los conjuntos de Nacho y de Santi.
- **Nacho (`nacho_*`): una habilidad por nivel.** `nacho_facil` = bajadas (caer de una plataforma es gratis: premia
  *no* saltar); `nacho_medio` = pozos anchos de 3 que obligan a despegar tarde, desde el borde; `nacho_dificil` =
  mesetas de altura 3 con pinchos arriba y huecos donde el cubo queda atrapado entre paredes; `nacho_muy_dificil` =
  ritmo, saltos en cadena con cadencia exacta. Están pensados para la **progresión** (sección 12.6).
- **Rugo (`facil`, `medio`, `dificil`, `muy_dificil`, `muy_muy_dificil`): escalera de dificultad.** De pinchos y
  escalones sueltos a escaleras altas con pinchos encima; `muy_dificil` es `dificil` con un **techo** de bloques
  arriba, y `muy_muy_dificil` es "pixel perfect": bloques y pinchos flotando, ventanas de 1 tick casi en todos los
  saltos.
- **Franco (`franco_*`): techos, rutas y no saltar.** `franco_facil`/`medio`/`dificil` son una escalera clásica;
  `franco_techo`, `franco_tunel` y `franco_pasillos` usan **pinchos colgantes** (bajo la plataforma saltar mata);
  `franco_rutas` tiene **dos rutas** que ganan (abajo por un túnel con 5 saltos, arriba con 7); `franco_sin_saltar` es
  un techo de pinchos en todo el nivel: **se gana sin saltar nunca** y cualquier salto mata. Sirve para ver si el
  agente aprende a no saltar cuando no conviene.
- **Santi (`santi_*`): pinchos en racha.** Cinco niveles con dificultad creciente (`santi_facil`,
  `santi_intermedio_1`, `santi_intermedio_2`, `santi_intermedio_dificil`, `santi_dificil`) con pinchos dobles y
  triples (`^^`, `^^^`), también al bajar de un escalón (`##^^^`). Son el banco de pruebas de la recompensa de
  **salto preciso** (sección 14.6), con `niveles/santi.json` (pesos crecientes 1 / 1 / 1.5 / 1.5 / 2, validación
  `level_1`).

### 6.2 Qué tan difíciles son

Números de `python scripts/difficulty.py` (peor caso entre las 10 fases de arranque; las columnas se explican en la
sección 9). "P azar" es `log10 P(ganar apretando al azar)`: −3 = uno de cada mil episodios.

| Nivel | Autor | Saltos mín. | Ventana mín. / media (ticks) | Saltos con trampas | Cadena más larga | P azar |
|---|---|---|---|---|---|---|
| `level_1` | TP | 13 | 2 / 6.3 | 3 | 3 | −4.6 |
| `nacho_facil` | Nacho | 6 | 7 / 7.8 | 2 | 2 | −3.7 |
| `nacho_medio` | Nacho | 8 | 5 / 6.0 | 0 | 3 | −7.1 |
| `nacho_dificil` | Nacho | 12 | 3 / 6.3 | 1 | 4 | −8.8 |
| `nacho_muy_dificil` | Nacho | 14 | 1 / 3.4 | 4 | 3 | −12.8 |
| `facil` | Rugo | 8 | 7 / 7.9 | 4 | 2 | −6.3 |
| `medio` | Rugo | 9 | 3 / 5.7 | 4 | 3 | −6.8 |
| `dificil` | Rugo | 14 | 2 / 5.8 | 2 | 6 | −8.2 |
| `muy_dificil` | Rugo | 15 | 1 / 4.2 | 4 | 8 | −10.3 |
| `muy_muy_dificil` | Rugo | 12 | 1 / 1.1 | 6 | 2 | −33.1 |
| `franco_facil` | Franco | 5 | 8 / 8.4 | 1 | 1 | −2.0 |
| `franco_medio` | Franco | 8 | 3 / 6.8 | 2 | 2 | −5.1 |
| `franco_dificil` | Franco | 11 | 3 / 5.5 | 2 | 6 | −10.7 |
| `franco_techo` | Franco | 5 | 3 / 5.2 | 3 | 2 | −17.3 |
| `franco_rutas` | Franco | 5 | 7 / 7.4 | 0 | 3 | −3.1 |
| `franco_tunel` | Franco | 6 | 8 / 8.3 | 1 | 2 | −11.6 |
| `franco_pasillos` | Franco | 8 | 3 / 6.0 | 1 | 3 | −15.7 |
| `franco_sin_saltar` | Franco | 0 | — | 0 | 0 | −80.7 |
| `santi_facil` | Santi | 6 | 8 / 8.3 | 2 | 2 | −3.4 |
| `santi_intermedio_1` | Santi | 7 | 1 / 5.1 | 2 | 2 | −6.7 |
| `santi_intermedio_2` | Santi | 9 | 1 / 4.0 | 0 | 2 | −10.9 |
| `santi_intermedio_dificil` | Santi | 10 | 3 / 6.4 | 2 | 2 | −11.9 |
| `santi_dificil` | Santi | 13 | 1 / 3.9 | 3 | 2 | −14.0 |

Algunas lecturas:

- Los nombres "fácil / difícil" de cada autor se ordenan bien **dentro** de cada familia, pero no son comparables
  entre familias: `santi_intermedio_1` tiene un salto de 1 tick, más preciso que cualquiera de `nacho_dificil`. La
  tabla da un criterio común.
- Las columnas miden cosas distintas: `franco_tunel` tiene ventanas anchas (8 ticks) pero P(azar) muy baja (−11.6),
  porque al azar el cubo salta bajo los pinchos colgantes; `muy_muy_dificil` es difícil en todo.
- `franco_sin_saltar` es el caso extremo: no hace falta ninguna precisión (0 saltos) y, sin embargo, al azar no se
  gana nunca (hay que no apretar durante 270 ticks: `0.5²⁷⁰ ≈ 10⁻⁸¹`). Es un problema de **exploración pura**.

### 6.3 Conjuntos de niveles (`niveles/*.json`)

| Archivo | Entrenamiento | Validación |
|---|---|---|
| `ejemplo.json` | 17 niveles con peso 1 (todos menos `franco_sin_saltar` y los de Santi) | `null` (uno de entrenamiento al azar: no mide generalización) |
| `nacho.json` | los 4 de Nacho, peso 1 | `level_1` (held out) |
| `nacho_progresion.json` | los 4 de Nacho, introducidos en 3 etapas | `level_1` (held out) |
| `santi.json` | los 5 de Santi, pesos 1 / 1 / 1.5 / 1.5 / 2 | `level_1` (held out) |

El formato está en la sección 12.5.

---

## 7. El entorno Gymnasium (`gd_env/env.py`)

### 7.1 La API de Gymnasium

[Gymnasium](https://gymnasium.farama.org/) es el estándar de facto para entornos de RL. Todo entorno expone:

```python
obs, info = env.reset(seed=..., options=...)                   # empieza un episodio
obs, reward, terminated, truncated, info = env.step(action)    # avanza un paso
```

- `terminated`: el episodio terminó por la dinámica del juego (ganó o murió).
- `truncated`: se cortó por tiempo (`max_steps`, por defecto 1.5× lo necesario: 780 ticks en `level_1`). Con la física
  actual no puede pasar, porque `x` avanza siempre, pero es parte de la API.
- `action_space = Discrete(2)` y `observation_space` describen qué entra y qué sale; SB3 los usa para construir la red.

`GeometryDashEnv` está registrado como `"GeometryDash-v0"`, así que también se puede crear con `gym.make`.

### 7.2 Observaciones (`obs_mode`)

**`"features"`** (la opción por defecto de `train.py`, con `MlpPolicy`): un vector de **28 números** (`4 + 2·lookahead`,
con `lookahead = 12`):

| índices | contenido |
|---|---|
| 0 | `y / H`: altura del cubo |
| 1 | `vy / jump_v`: velocidad vertical (sin esto no sabría si está subiendo o bajando) |
| 2 | `en_suelo` (0 o 1): si saltar ahora tiene efecto |
| 3 | `x mod 1`: posición dentro del tile (para el timing fino) |
| 4…27 | para cada una de las 12 columnas desde la del cubo: altura del bloque **más alto** / H, y si hay pincho (común o colgante) |

El agente "ve" **12 columnas hacia adelante = 12 tiles = 60 ticks** de anticipación. Un salto ocupa 3.2 tiles, así que
alcanza para preparar uno o dos saltos.

**`"grid"`** (`--obs grid`, con una CNN): una imagen `(3, 7, 16)` de 0s y 1s con tres canales: bloques, pinchos y
jugador. Es una ventana de 16 columnas que arranca en la del cubo. Los pinchos colgantes van en el **mismo canal** que
los comunes (la forma de la observación no cambió y los modelos viejos se siguen cargando); se distinguen por la
posición. No incluye la velocidad vertical, así que el agente tiene que deducir si sube o baja sólo con la posición:
el problema es parcialmente observable.

**`"dict"`** (`--obs dict`, CNN + MLP): `{"grid": ..., "vec": [y/H, vy/jump_v, en_suelo, frac(x)]}`, la grilla más el
vector de estado. Resuelve la falta de velocidad de `grid`.

`features` es información ya "masticada" (alturas y pinchos por columna); `grid` obliga a la red a **aprender** a
extraer esa información de una imagen. Comparar las dos mide cuánto cuesta aprender la representación (sección 11.6).

> **Limitación de `features` con techos.** De cada columna sólo ve la altura del bloque más alto y si hay *algún*
> pincho. Con un techo o una plataforma encima del cubo (`muy_dificil`, `franco_techo`, `franco_rutas`...) esa altura
> es la del techo: no ve el piso, el hueco libre, ni si el pincho cuelga o está en el piso. En `franco_rutas` las dos
> rutas se ven iguales. `grid` y `dict` sí ven todos los bloques. Esta diferencia es parte de lo que mide el
> experimento de observaciones (sección 18, E2).

### 7.3 Recompensa: términos que se prenden y apagan

La recompensa de cada tick es la suma de **términos**. Cuáles valen se elige por entrenamiento con `--rewards` (o
`rewards=[...]` en Python); la lista está en `REWARD_TERMS` de `gd_env/rewards.py`:

| Término | Qué suma en el tick | Parámetros (flag, valor por defecto) |
|---|---|---|
| `progress` | `progress_scale · Δx / (goal_x − start_x)`: cruzar todo el nivel suma `progress_scale` | `--progress-scale` (1) |
| `win` | `+win_bonus` en el tick en que gana | `--win-bonus` (1) |
| `death` | `−death_penalty` en el tick en que muere | `--death-penalty` (10) |
| `survive` | `+survive_bonus` cada `survive_interval` ticks vivo (o, con `survive_rate`, proporcional a la penalización por salto; sección 14.5) | `--survive-bonus` (0), `--survive-interval` (1), `--survive-rate` (0) |
| `jump_penalty` | `−c` en cada salto **efectivo**; con `--ramp`, `c` va cambiando (sección 14) | `--jump-penalty` (0), `--ramp` |
| `salto_preciso` | ajuste por saltar según lo que haya adelante (sección 14.6) | — |

Sin `--rewards` quedan prendidos los de siempre (`DEFAULT_TERMS` = todos menos `salto_preciso`), así las corridas
viejas se reproducen igual. Como `survive_bonus` y `jump_penalty` valen 0 por defecto, la recompensa por defecto es
**progreso + 1 al ganar − 10 al morir**. Un episodio perfecto sin penalización por salto suma **2.0** (1 de progreso y 1
de bonus). Morir resta 10, mucho más que todo lo que se puede ganar, así que evitar la muerte domina cualquier otra
consideración.

> Las corridas `runs/features_fixed*` son de antes de que el castigo por morir por defecto pasara de 1 a 10: se
> entrenaron con muerte = −1 (se nota en su retorno, por ejemplo −0.97 en `features_fixed_jp0.1`).

Cómo está implementado: `reward_terms(tr, terms, ...)` suma todos los términos prendidos **menos `jump_penalty`**, que
lo resta el entorno en `GeometryDashEnv.step` porque su valor cambia durante el entrenamiento (la rampa lo actualiza en
caliente). `default_reward(tr, ...)` es `reward_terms` con `progress`, `win`, `death` y `survive`.

**Reglas** (se chequean antes de entrenar, con un error claro; ver `validate_terms` y `train`):

- `jump_penalty` y `salto_preciso` son **excluyentes**: los dos fijan cuánto cuesta saltar y se pisarían. Tampoco se
  acepta `salto_preciso` con `--jump-penalty > 0` o `--ramp`.
- Un parámetro de un término apagado es un **error**, no se ignora en silencio (por ejemplo `--death-penalty 5` sin
  `death` en `--rewards`). La tabla término → parámetros es `TERM_PARAMS`.
- Términos desconocidos o una lista vacía son error.
- `--rewards` no se combina con `--reward-fn` (esa ya reemplaza toda la recompensa).
- `--survive-rate` y `--survive-bonus` son alternativas, y `--survive-rate` necesita `--jump-penalty > 0`.
- Los términos quedan en `config.json` (así `evaluate.py` re-evalúa con la misma recompensa) y en el nombre de la
  corrida (`..._rw-progress+win+death+salto_preciso`).

**Recompensa propia.** Toda la recompensa se puede reemplazar con una función `reward_fn(tr) -> float`, donde `tr`
es un `Transition` con lo que pasó en el tick:

| campo | significado |
|---|---|
| `tr.prev`, `tr.new` | estados antes y después del tick |
| `tr.status`, `tr.dead`, `tr.won` | resultado del tick |
| `tr.jumped` | hubo un salto efectivo |
| `tr.dx`, `tr.d_progress`, `tr.progress` | avance en el tick (en tiles y en fracción del nivel) y progreso acumulado |
| `tr.level` | el nivel, por si hace falta mirar obstáculos |

```python
from gd_env import Transition, default_reward

def mi_recompensa(tr: Transition) -> float:
    r = default_reward(tr, death_penalty=5.0)     # partir de la de por defecto...
    return r - 0.02 if tr.jumped else r           # ...y agregarle lo que haga falta
```

Se usa con `GeometryDashEnv(reward_fn=...)`, `train("ppo", reward_fn=...)` o desde la línea de comandos con
`--reward-fn archivo.py:funcion` (hay ejemplos en `ejemplos/reward_ejemplo.py` y `ejemplos/reward_santi.py`). Por
compatibilidad se acepta también la firma vieja `fn(prev, new, status, level)`. En `config.json` la función queda
guardada como texto `archivo.py:funcion`, así `evaluate.py` puede recargarla; una `lambda` o una función de un
notebook no se puede recargar, y en ese caso la re-evaluación usa la recompensa por defecto (sólo cambia la columna
`return`). La `jump_penalty` se suma igual aunque haya `reward_fn`.

### 7.4 Fase de arranque

`env.reset(options={"x0": 2.5})` hace que el cubo arranque en `x = 2.5` en lugar de 2.0. Como `x` avanza 0.2 por tick,
eso corre todos los obstáculos respecto de la grilla temporal: los ticks en que hay que saltar cambian.
Es una forma barata de tener **variantes del mismo nivel** sin diseñar niveles nuevos (sección 13).

### 7.5 Varios niveles

`level` puede ser una lista de niveles, con `level_weights` opcional (un peso por nivel). En cada `reset` el entorno
**sortea** uno con probabilidad proporcional a su peso, usando su propio generador aleatorio: así las 8 copias en
paralelo de PPO no juegan todas el mismo nivel a la vez. `reset(options={"level": "nacho_medio"})` fuerza un nivel
(lo usan la evaluación y los GIFs). `max_steps` se recalcula según el largo del nivel que salió.

Todos los niveles tienen que tener la misma altura: la forma de la observación depende de ella, y una red sólo acepta
una forma. Los pesos son un atributo común del entorno, así que se pueden cambiar en medio del entrenamiento con el
mismo mecanismo que usa la rampa de la penalización por salto (sección 14.4). Eso es lo que usa la **progresión de
niveles** (sección 12.6): empezar con peso 0 en los niveles difíciles e ir subiéndolo a medida que el agente gana los
fáciles.

### 7.6 `info`

Cada paso devuelve `info = {"x", "y", "t", "status", "progress", "success", "level"}`. `progress` va de 0 a 1 y `success`
es `True` sólo si ganó. El `Monitor` de SB3 registra `success`, `progress` y `level` de cada episodio de entrenamiento.

---

## 8. El oráculo: la solución óptima exacta (`gd_env/oracle.py`)

### 8.1 La idea

Como el entorno es determinista y conocemos su física, se puede **calcular** la mejor jugada en lugar de aprenderla.
La observación clave es que `x` depende sólo del tick `t`. Entonces, todos los estados alcanzables en el tick `t`
forman una **capa**, y cada acción lleva de la capa `t` a la capa `t+1`. El espacio de estados es un **grafo dirigido
acíclico (DAG) por capas**:

```
tick 0      tick 1        tick 2           ...      tick 520
 (s0) ──0──▶ (y=1,vy=0) ──0──▶ ...
   └───1──▶ (y=1.35,..) ──0──▶ ...                     ▶ WIN
                        (en el aire sólo existe la acción 0)
```

### 8.2 El algoritmo (programación dinámica)

Para cada capa se guarda, por cada estado alcanzable, la **mínima cantidad de saltos** con la que se llega y de qué
estado vino. Los estados con la misma clave `(t, y, vy, en_suelo)` se fusionan quedándose con el de menos saltos.
Las transiciones que mueren se descartan. Cuando alguna transición gana, se reconstruye hacia atrás el camino.

Es el mismo principio que el algoritmo de Dijkstra o que Viterbi: como el grafo es acíclico y por capas, basta un
recorrido hacia adelante. El número de estados es chico (unos 6 500 en todo `level_1`), así que resolver el nivel
tarda menos de una décima de segundo.

Resultado para `level_1`: **gana en 520 ticks con 13 saltos**, despegando en
`x = 13.2, 23.2, 32.2, 39.2, 45.8, 49.8, 52.2, 65.2, 73.2, 79.8, 85.2, 95.2, 99.2`
(`Oracle(level, Physics()).plan().jump_x`).

### 8.3 Para qué sirve

1. **Prueba que el nivel tiene solución** (y que todas las fases de evaluación la tienen, en los 23 niveles).
2. **Techo de desempeño**: el mínimo de saltos. Los agentes se reportan como "saltos / oráculo" (1.00 = óptimo).
3. **Política experta** (`OraclePolicy`, con el método `predict` igual que SB3): sirve para grabar GIFs de referencia
   y, a futuro, para clonado de comportamiento o DAgger. Sigue los cambios de nivel del entorno (un oráculo por nivel).
4. **Base de la medida de dificultad** (sección 9), que usa el mismo grafo para medir la precisión de cada salto.
5. `Oracle.is_winnable(state)` dice si un estado todavía se puede salvar.

> El oráculo **no es un agente de RL**: hace trampa leyendo el estado interno y conociendo la física. Los agentes de
> RL sólo ven observaciones y recompensas.

---

## 9. Dificultad de un nivel (`gd_env/difficulty.py`)

### 9.1 Para qué

Con 23 niveles de cinco autores hacía falta un criterio común para decir "este es más difícil que aquel" **sin
entrenar** (y para justificarlo en el informe). Lo que acordamos como dificultad es: **qué tan preciso tiene que ser
cada salto** y **cuántos saltos hay que encadenar**, más un resumen de **qué tan difícil es dar con la solución
explorando**. Todo se calcula con la física, de forma exacta.

```bash
python scripts/difficulty.py                          # tabla de todos los niveles registrados
python scripts/difficulty.py level_1 --detalle        # + tabla salto por salto
python scripts/difficulty.py mi_nivel.txt --ascii     # un nivel dibujado en ASCII, sin registrarlo
python scripts/difficulty.py niveles/santi.json       # los niveles de un .json (entrenamiento y validación)
```
```python
from gd_env import get_level, level_difficulty
d = level_difficulty(get_level("level_1"))      # d.summary() = la fila de la tabla; d.jumps = detalle por salto
```

### 9.2 Cómo se calcula

Igual que el oráculo, enumera **todos** los estados alcanzables desde el arranque (el DAG por capas de tick). Después,
de atrás para adelante, calcula para cada estado `s`:

- `J(s)`: la mínima cantidad de saltos para ganar desde `s` (infinito si ya no se puede ganar);
- `P(s)`: la probabilidad de ganar desde `s` apretando al azar (50% saltar / 50% no, en cada tick).

Con eso recorre un camino óptimo salto por salto, despegando en el **centro** de cada ventana, y mide:

| Métrica | Qué es | Por qué importa para RL |
|---|---|---|
| **saltos mínimos** | los mismos que el oráculo | techo de desempeño |
| **ventana** de un salto (ticks) | cuántos ticks de despegue distintos siguen ganando **sin gastar saltos de más**. 1 tick = 0.2 tiles. Se reporta la mínima (el salto más preciso) y la media | ventana chica = la política tiene que acertar el tick exacto; un cambio chico de pesos mueve el salto y mata (la inestabilidad de la sección 15.1) |
| **trampas** | ticks de despegue desde los que el cubo **aterriza vivo** pero el nivel ya no se puede ganar ("el primer salto determina los siguientes") | el error se paga después, no cuando se comete: asignación de crédito difícil (sección 2.7) |
| **saltos encadenados** | el salto `j+1` depende del `j` si cambiar cuándo se despega en `j` (dentro de su ventana) cambia la ventana del `j+1`. "Cadena más larga" = la secuencia más larga de saltos dependientes | hay que aprender secuencias, no reacciones sueltas |
| **log10 P(ganar al azar)** | probabilidad de ganar apretando 50/50 en cada tick, en escala logarítmica (−3 = uno de cada mil episodios) | al empezar, PPO/A2C juegan casi 50/50 y DQN con ε = 1 juega al azar: es la chance de ver una victoria explorando (sección 2.5) |
| **fases con solución** | de cuántas fases de arranque se puede ganar | todas tienen que tenerla para evaluar |

Cada columna de la tabla es el **peor caso** entre las 10 fases de arranque de la evaluación (`--fases 1`: sólo
`x0 = 2.0`), porque las métricas cambian un poco con la fase (el despegue sólo puede ocurrir en ticks enteros). Con
`--detalle`, por cada salto: dónde está la ventana, cuántas trampas tiene y, si está encadenado, entre cuánto y cuánto
varía la ventana del siguiente. Funciona con cualquier nivel dibujable en ASCII, también con techos, plataformas
flotantes y pinchos colgantes.

### 9.3 Ejemplo: `level_1`

13 saltos; ventanas de 2 a 10 ticks (media 6.3). La más chica es el salto desde la plataforma baja al pozo (x ≈ 78,
2 ticks); los pinchos dobles (x ≈ 23 y x ≈ 65) tienen 3. Tiene 3 saltos con trampas (escalón x ≈ 31, escalera
x ≈ 49, plataforma baja x ≈ 73), una cadena de 3 saltos (pozo → escalera, x ≈ 45–52) y P(ganar al azar) ≈ 10⁻⁴·⁶
(uno de cada ~40 000 episodios: por eso la política aleatoria de los tests no gana nunca).

### 9.4 Cuidados al leerla

- Las ventanas se miden sobre **un** camino óptimo (despegando en el centro de cada ventana): la de cada salto está
  condicionada a ese camino. Justamente eso es lo que mide "encadenado".
- "Sin gastar saltos de más" significa que un despegue fuera de la ventana puede seguir ganando, pero con más saltos
  que el óptimo. La ventana mide precisión **para jugar óptimo**, no para sobrevivir.
- Los cortes entre fácil / medio / difícil no están fijados por el código: la tabla de la sección 6.2 da los números y
  `level_1` sirve de referencia. El experimento E7 (sección 18) contrasta estas métricas con lo que logran los agentes.
- Se prueba en `tests/test_difficulty.py` (saltos iguales a los del oráculo, ventanas y trampas verificadas con el
  oráculo, valores conocidos de `level_1`, nivel sin solución, peor caso entre fases).

---

## 10. Visualización (`gd_env/render.py`)

`record_episode(env, policy, "x.gif", "x_recorrido.png")` juega un episodio con cualquier objeto con `.predict` (un
modelo SB3 o `OraclePolicy`) y genera:

- un **GIF** con cámara que sigue al cubo, la estela, triángulos verdes donde despegó, el progreso y el motivo de
  muerte;
- un **mapa del recorrido** (PNG) con el nivel completo, el camino del cubo y una cruz roja donde murió.

Los pinchos colgantes se dibujan como triángulos invertidos. También funciona `env.render()` con
`render_mode="rgb_array"` (un frame) o `"ansi"` (texto, con `v` para los colgantes).

Las imágenes de [`niveles/png/`](niveles/png/) son mapas estáticos de cada nivel registrado (sin recorrido), con el
inicio en verde y la meta en amarillo; sirven para mostrar los niveles en el informe sin correr nada.

---

## 11. Los agentes: Stable-Baselines3, PPO, A2C, DQN, QR-DQN y las redes

### 11.1 Stable-Baselines3 (SB3)

[SB3](https://stable-baselines3.readthedocs.io/) es una librería de implementaciones confiables de algoritmos de RL
sobre PyTorch. Recibe un entorno Gymnasium, construye la red según `observation_space` y `action_space`, y entrena con
`model.learn(total_timesteps)`. Se usa para no reimplementar los algoritmos y poder concentrarse en el entorno, la
recompensa, los niveles y la evaluación.

Hay cuatro algoritmos (`ALGOS` en `agents/common.py`, elegidos con `--algos`), en dos familias:

| | on-policy, actor-crítico | off-policy, basados en valor |
|---|---|---|
| simple | **A2C** | **DQN** |
| mejorado | **PPO** (A2C + recorte + varias épocas) | **QR-DQN** (DQN + distribución de retornos) |

Así cada algoritmo "mejorado" tiene su control simple, y se puede medir qué aporta cada mejora. QR-DQN viene de
`sb3-contrib` (se importa sólo si se usa; sin esa librería, sus tests se saltean).

Todas las redes terminan en dos capas ocultas de 64 neuronas (`net_arch=[64, 64]`). Lo que cambia según la
observación es lo que va **antes** de esas capas (sección 11.6).

### 11.2 PPO (Proximal Policy Optimization)

PPO es un algoritmo **on-policy de gradiente de política** con arquitectura **actor-crítico**:

- el **actor** es la política `π(a|s)`, que da la probabilidad de saltar;
- el **crítico** estima `V(s)`, que se usa para calcular la ventaja.

Ciclo de PPO:

```
repetir:
  1. ROLLOUT: jugar n_steps=256 pasos en cada uno de los 8 entornos en paralelo → 2048 transiciones
  2. VENTAJAS: calcular A(s,a) con GAE (λ=0.95): "¿esta acción salió mejor o peor de lo que esperaba el crítico?"
  3. ACTUALIZAR: 10 épocas sobre esas 2048 transiciones, en mini-batches de 256
       subir la probabilidad de acciones con ventaja > 0, bajar las de ventaja < 0,
       pero sin cambiar la política más de un 20% por paso (clip_range=0.2)
  4. descartar los datos (on-policy: sólo sirven para la política que los generó)
```

El **recorte (clipping)** es lo que define a PPO: impide actualizaciones demasiado grandes que destruirían lo aprendido.
**GAE** (Generalized Advantage Estimation) combina retornos de distintos horizontes para obtener ventajas con menos
varianza. El bono de **entropía** (`ent_coef=0.01`) premia a la política por mantener algo de aleatoriedad, lo que
sostiene la exploración.

| Hiperparámetro | Valor | Efecto |
|---|---|---|
| `n_steps` × `n_envs` | 256 × 8 = 2048 | datos por actualización |
| `batch_size` | 256 | tamaño del mini-batch |
| `n_epochs` | 10 | pasadas sobre cada rollout |
| `learning_rate` | 3e-4 | tamaño del paso del optimizador |
| `gamma` | 0.99 | horizonte efectivo ~100 ticks |
| `gae_lambda` | 0.95 | sesgo/varianza de la ventaja |
| `clip_range` | 0.2 | cuánto puede cambiar la política por actualización |
| `ent_coef` | 0.01 | presión hacia la exploración |

### 11.3 A2C (Advantage Actor-Critic)

A2C es el actor-crítico "básico" del que PPO es una mejora. Junta un rollout corto (`n_steps=16` × 8 entornos = 128
transiciones), calcula ventajas, hace **un solo paso de gradiente** y descarta los datos. No tiene recorte ni varias
épocas: cada dato se usa una vez y nada impide un paso demasiado grande.

| Hiperparámetro | Valor |
|---|---|
| `n_steps` × `n_envs` | 16 × 8 = 128 |
| `learning_rate` | 7e-4 |
| `gamma` / `gae_lambda` | 0.99 / 0.95 |
| `ent_coef` / `vf_coef` | 0.01 / 0.5 |

Sirve como control: **si PPO no le gana a A2C, el recorte y las épocas no están aportando** en este problema.

### 11.4 DQN (Deep Q-Network)

DQN es un algoritmo **off-policy basado en valor**. Aprende `Q(s, a)` con una red y actúa con `argmax Q`. Se apoya en
la ecuación de Bellman:

```
Q(s, a) ≈ r + γ · maxₐ' Q_target(s', a')
```

Sus tres ingredientes:

- **Replay buffer** (100k transiciones): guarda experiencias pasadas y entrena con muestras al azar. Rompe la
  correlación entre pasos consecutivos y reutiliza cada dato muchas veces (por eso es "off-policy").
- **Red objetivo** (`target_update_interval=500`): una copia congelada de la red para calcular el objetivo de Bellman.
  Sin ella el objetivo se mueve con cada actualización y el entrenamiento se vuelve inestable.
- **ε-greedy**: ε baja linealmente de 1.0 a 0.05 durante el primer 30% del entrenamiento.

| Hiperparámetro | Valor |
|---|---|
| `learning_rate` | 1e-3 |
| `buffer_size` | 100 000 |
| `learning_starts` | 5 000 (pasos al azar antes de empezar a entrenar) |
| `batch_size` | 64 |
| `train_freq` / `gradient_steps` | una actualización cada 4 pasos |
| `exploration_fraction` / `exploration_final_eps` | 0.3 / 0.05 |

DQN usa **un solo entorno** (`n_envs` sólo afecta a PPO y A2C).

### 11.5 QR-DQN (Quantile Regression DQN)

DQN estima el retorno **promedio** de cada acción. QR-DQN estima la **distribución** completa: la red devuelve 50
cuantiles por acción (`n_quantiles=50`) y actúa con su media. Aprender la distribución da una señal de entrenamiento
más rica, sobre todo cuando los retornos tienen forma muy asimétrica, como acá: casi siempre "avanzar un poco", pero a
veces "−10 por morir". Usa **los mismos hiperparámetros que DQN**, así que la diferencia entre los dos aísla el efecto
de la cabeza distribucional.

### 11.6 Las redes: MLP para `features`, CNN para `grid` y `dict`

En SB3 una política es `observación → extractor de características → capas ocultas (net_arch) → acción / valor`.
`build_model` elige todo según `--obs` (`policy_for` y `policy_kwargs_for` en `agents/common.py`):

| `--obs` | política de SB3 | extractor |
|---|---|---|
| `features` | `MlpPolicy` | ninguno: el vector de 28 entra directo a las capas densas |
| `grid` | `CnnPolicy` | `GridCNN` (`agents/cnn.py`) |
| `dict` | `MultiInputPolicy` | `GridVecCNN`: CNN para la grilla + capa lineal para el vector, concatenadas |

**Qué es una CNN.** Una red convolucional pasa pequeños filtros (acá de 3×3) por toda la imagen. Cada filtro aprende a
detectar un patrón local, por ejemplo "pincho justo delante y a la altura del cubo", y como es el mismo filtro en
todas las posiciones, lo detecta esté donde esté. Es la arquitectura natural para entradas con estructura espacial.

**Por qué una CNN propia.** La CNN por defecto de SB3 (NatureCNN, la de los juegos de Atari) usa filtros de 8×8 con
saltos de 4 píxeles y necesita imágenes de al menos 36×36; con la grilla de 7×16 se queda sin imagen antes de la
última capa. `GridCNN` usa dos convoluciones de 3×3 (32 y 64 filtros) con `padding=1`, que **no achican** la grilla,
y una capa lineal que resume todo en 128 números. No hay pooling ni strides: en una grilla tan chica cada celda
importa, y un pincho corrido una columna significa saltar en otro momento.

```
grid (3×7×16) ─▶ conv3×3 (32) ─▶ ReLU ─▶ conv3×3 (64) ─▶ ReLU ─▶ aplanar ─▶ lineal (128) ─▶ [64, 64] ─▶ acción
dict: grid ─▶ (la misma CNN) ─▶ 128 ┐
      vec  ─▶ lineal (32) ─────────▶ 32 ┴─▶ concatenar (160) ─▶ [64, 64] ─▶ acción
```

Para cargar un modelo entrenado con `grid`/`dict`, `--obs` tiene que ser el mismo con el que se entrenó.
`scripts/evaluate.py` lo lee del `config.json` de la corrida.

### 11.7 Comparación

| | PPO | A2C | DQN | QR-DQN |
|---|---|---|---|---|
| Qué aprende | política + V | política + V | Q promedio | distribución de Q |
| Datos | on-policy, 10 épocas | on-policy, 1 paso | off-policy, replay | off-policy, replay |
| Exploración | entropía | entropía | ε-greedy | ε-greedy |
| Entornos en paralelo | 8 | 8 | 1 | 1 |

En los primeros experimentos con `features`, PPO resultó **muy inestable** entre actualizaciones (sección 15) y DQN más
estable pero más lento al principio. Las CNN (`grid` y `dict`) ya tienen corridas con los niveles de Nacho (sección
18, E6); la comparación de los cuatro algoritmos es el experimento E1.

> Los hiperparámetros son un **punto de partida razonable, no están tuneados**.

---

## 12. Cómo se arma el entrenamiento

### 12.1 Entornos vectorizados

`make_train_env` (en `agents/common.py`) crea un **VecEnv**: `n_envs` copias del entorno que avanzan juntas. Con
PPO son 8; cada `step` devuelve 8 transiciones. Las copias no están sincronizadas en el episodio: cuando una muere, se
reinicia sola mientras las otras siguen.

Cada copia está envuelta en un **`Monitor`**, que escribe en `monitor/<i>.monitor.csv` el retorno, la duración, el
éxito, el progreso y el nivel de cada episodio de entrenamiento. Con `--train-phase random`, además se envuelve en
`RandomPhase`, que sortea `x0 ∈ [2.0, 2.99]` en cada `reset`.

```
GeometryDashEnv  ◀── RandomPhase (opcional) ◀── Monitor ◀── DummyVecEnv (8 copias) ◀── PPO
```

### 12.2 ¿Qué niveles ve y cuántos datos?

- Con `--train-phase fixed` (por defecto) y sin `--levels` **ve siempre `level_1` desde x0 = 2.0**. Todos los
  episodios son idénticos salvo por las acciones del agente. Con `--levels`, cada episodio sortea uno de los niveles
  de entrenamiento (sección 12.5); `fixed`/`random` se aplica igual a todos.
- 300 000 pasos de PPO = 147 rollouts de 2048 pasos (301 056 pasos en total, porque el último rollout se completa).
  Un episodio ganado de `level_1` son 520 pasos, así que equivale a unos 580 niveles completos (unos 1 100 en niveles
  de 60 columnas); en la práctica son muchos más episodios, más cortos, porque al principio muere enseguida.
- No hay "batches de niveles": los batches son de **transiciones** (pasos sueltos) mezcladas de distintos episodios,
  entornos y niveles.

### 12.3 Callbacks

SB3 llama a los **callbacks** en cada paso de entrenamiento. Es la forma de engancharse al entrenamiento sin tocar el
algoritmo. `agents/training.py` arma (en `_callbacks`) esta lista:

| Callback | Cuándo | Qué hace |
|---|---|---|
| `CheckpointCallback` (SB3) | cada `--checkpoint-freq` pasos (25k) | guarda `checkpoints/model_<N>_steps.zip` |
| `LevelProgression` | con `progresion` en el `.json` | introduce los niveles por etapas y ajusta sus pesos (sección 12.6) |
| `PerformanceJumpPenaltyRamp` | con `--ramp` | ajusta la penalización por salto según el desempeño (sección 14.4) |
| `BestModelCallback` | cada `--eval-freq` pasos (25k) y al terminar | evalúa y guarda el mejor modelo (sección 15) |

### 12.4 `train(...)`: una corrida completa

Toda la lógica de una corrida está en la función `train` de `agents/training.py`; `scripts/train.py` sólo traduce los
argumentos de la línea de comandos a esa llamada. Así se puede entrenar igual desde Python o desde Colab:

```python
from agents import train
res = train("ppo", seeds=[0], timesteps=200_000, win_bonus=2.0, death_penalty=5.0, jump_penalty=0.05)
res = train(["ppo", "a2c"], obs="grid", reward_fn=mi_recompensa)
res = train("ppo", levels="niveles/santi.json", rewards=["progress", "win", "death", "salto_preciso"])
# res = {"root": carpeta de la corrida, "final": {...}, "best": {...}, "final_val": {...}, "best_val": {...}}
```

Para cada algoritmo y seed: crea el VecEnv con la recompensa pedida, construye el modelo (o lo carga con
`init_model`), entrena con los callbacks, guarda `model.zip`, evalúa el modelo final y el mejor (en los niveles de
entrenamiento y en los de validación) y graba los GIFs. Al final escribe `resumen.md`.

Las combinaciones inválidas se rechazan **antes** de empezar a entrenar: las reglas de los términos de la recompensa
(sección 7.3), rampa sin `jump_penalty`, `survive_rate` sin `jump_penalty` o junto con `survive_bonus`, `init_model`
con más de un algoritmo o que no existe, una `reward_fn` que no carga, niveles que no existen o un bloque
`progresion` mal armado.

### 12.5 Entrenar con varios niveles

Entrenar en un solo nivel enseña **ese** nivel. Para que el agente aprenda a jugar (a reaccionar a pinchos, pozos y
escalones en cualquier orden), se entrena con varios y se mide en niveles que **nunca vio**. Es la misma lógica que la
división entrenamiento/validación del aprendizaje supervisado:

- **Niveles de entrenamiento**: los que juega mientras aprende. De acá sale el mejor modelo.
- **Niveles de validación (held out)**: no se juegan nunca durante el entrenamiento; sólo se miden. La diferencia
  entre el éxito en entrenamiento y en validación es la **brecha de generalización**: si es grande, el agente memorizó
  los niveles en vez de aprender a jugar.

Se configuran con `--levels`, con nombres sueltos (peso 1 cada uno, validación `null`) o con un `.json` (sección 6.3):

```json
{"niveles": {"franco_facil": 1.0, "franco_medio": 1.0, "franco_dificil": 2.0},
 "validacion": ["franco_techo"]}
```

- `niveles`: nivel → **peso** (probabilidad relativa de jugarlo; acá `franco_dificil` sale el doble), o una lista con
  pesos iguales. Un peso 0 deja el nivel cargado sin jugarse.
- `validacion`: nombres fijos; un número `N` (sortea `N` de `niveles` y los saca del entrenamiento, reproducible con
  `--val-seed`); o `null` (no reserva ninguno y valida en uno de los de entrenamiento, al azar: sirve para ver el
  pipeline funcionando, pero no mide generalización, y `resumen.md` lo aclara).
- Las claves que empiezan con `_` se ignoran (sirven de comentario).

El sorteo de validación se hace **una vez por corrida**, no por seed: si cada seed validara en un nivel distinto, el
promedio entre seeds mezclaría niveles más fáciles y más difíciles y el ± dejaría de medir la variación del algoritmo.

Qué cambia en las salidas: `eval_fases.csv` tiene una columna `level`; los niveles de validación van a
`eval_fases_val.csv` (y `_best_val`); hay un GIF por nivel; `eval_history.csv` suma las columnas `val_success`,
`val_progress` y `val_mean_jumps_won`, que dan la **curva de generalización** a lo largo del entrenamiento. Esas
columnas **no** se usan para elegir el mejor modelo: si se usaran, el nivel de validación influiría en la elección y
su resultado dejaría de ser una medida honesta (haría falta un tercer conjunto, de test).

También hay una columna de éxito **por nivel** (`success_<nivel>` y `val_success_<nivel>`), y `monitor/*.csv` anota qué
nivel se jugó en cada episodio de entrenamiento.

### 12.6 Progresión de niveles (curriculum learning)

Si el agente ve desde el principio un nivel muy difícil, casi nunca gana y casi no recibe señal útil (la P(ganar al
azar) de `nacho_muy_dificil` es 10⁻¹²·⁸). La **progresión** (en la literatura, *curriculum learning*) ordena la
experiencia como un plan de estudios: primero los niveles fáciles, y los difíciles se suman cuando ya domina los
anteriores. Es la misma idea que la rampa de la penalización por salto, aplicada a los niveles.

Se define en el `.json` de niveles con un bloque `progresion` (ejemplo: `niveles/nacho_progresion.json`):

```json
"progresion": {"etapas": [["nacho_facil"], ["nacho_medio"], ["nacho_dificil", "nacho_muy_dificil"]],
               "desbloquear": 0.8, "ventana": 50, "piso": 0.2}
```

**El riesgo: olvido catastrófico.** Una red no guarda "lo del nivel fácil" en un lugar aparte: todo está en los mismos
pesos. Si durante mucho tiempo sólo juega el difícil, cada actualización ajusta los pesos para el difícil y nada protege
lo que servía para el fácil. PPO lo sufre más que DQN porque no tiene replay buffer: lo que no juega, no lo repasa. Y
los niveles pueden pedir cosas opuestas (`nacho_facil` premia *no* saltar en las bajadas; `nacho_muy_dificil`, saltar
en cadena).

**Cómo lo evita `LevelProgression`** (`agents/callbacks.py`):

1. Arranca sólo con la primera etapa; las demás tienen peso 0.
2. Lleva el éxito de los últimos `ventana` episodios de entrenamiento de **cada** nivel.
3. Cada nivel activo pesa `peso_base × max(piso, 1 − éxito)`: se practica más lo que más falla. Un nivel recién
   desbloqueado entra con peso máximo; uno dominado baja hasta `piso` pero **nunca a 0** (sigue repasándolo); y si lo
   empieza a olvidar, su éxito baja y su peso vuelve a subir solo. Esta regla de "jugar más donde más se falla" es la
   idea de *prioritized level replay*.
4. Desbloquea la etapa siguiente cuando todos los niveles activos ganan al menos `desbloquear` de su ventana.

`desbloquear`, `ventana` y `piso` son opcionales (por defecto 0.8, 50 y 0.2). Si `validacion` saca un nivel que estaba
en una etapa, se lo quita de ella. El nombre de la corrida suma `_prog`.

`progresion.csv` registra cada cambio de etapa y la evolución de pesos y éxito por nivel; junto con las columnas
`success_<nivel>` de `eval_history.csv`, permite ver si algún nivel se olvida. El mejor modelo se sigue eligiendo con
**todos** los niveles de entrenamiento (incluidos los todavía bloqueados), así que un modelo que olvidó el fácil no puede
ganar esa comparación.

Detalle de implementación: los entornos se crean ya con los pesos de la primera etapa, porque SB3 resetea los entornos
antes de llamar a los callbacks y, si no, los primeros episodios podrían caer en niveles bloqueados. La progresión se
combina con cualquier recompensa (`--rewards`, `--jump-penalty`): los términos valen igual en todas las etapas.

---

## 13. Evaluación: fases de arranque y métricas

### 13.1 Por qué fases

Entrenando siempre desde `x0 = 2.0`, un agente podría memorizar "saltar en el tick 56, 106, ..." sin mirar los
obstáculos. Para detectarlo, `evaluate_policy_phases` juega, en cada nivel, **un episodio por cada una de 10 fases**
`x0 ∈ {2.0, 2.105, ..., 2.95}` (`EVAL_PHASES`). Desde cada fase cambian los ticks exactos en que hay que saltar, pero
el nivel es el mismo. Todas tienen solución comprobada por el oráculo en los 23 niveles.

La evaluación es **determinista** (`predict(..., deterministic=True)`): el agente siempre toma la acción más probable
(PPO, A2C) o la de mayor Q (DQN, QR-DQN). Por eso las tasas de éxito de evaluación son más altas que las del
entrenamiento, donde hay exploración.

### 13.2 Qué se mide

Cada fila (una por nivel y fase) tiene `level, x0, success, progress, return, ticks, status, n_jumps, oracle_jumps, death_x`.
Las métricas agregadas (`metrics`):

- **éxito**: fracción de (nivel, fase) ganados;
- **progreso medio**: qué tan lejos llega (0 a 1), útil cuando no gana;
- **saltos medios en las fases ganadas**, comparados con los del oráculo (`saltos / oráculo`; 1.00 = óptimo).

`summarize` arma la tabla entre seeds (media ± desvío) que queda en `resumen.md`. Se usan **varias seeds** porque en
RL el resultado de una sola corrida depende mucho del azar: una seed puede ganar todo y otra nada con la misma
configuración.

---

## 14. El costo de saltar: penalización, rampa, bonus por seguir vivo y salto preciso

### 14.1 El problema: saltar sin parar

Sin penalización, saltar no cuesta nada, y estar en el aire es seguro casi siempre (pasa por arriba de los pinchos).
El agente aprende entonces a **saltar todo el tiempo**: ~27 saltos contra 13 del oráculo en `level_1`. Gana, pero
juega "sucio".

### 14.2 La solución: costo por salto (`jump_penalty`)

`--jump-penalty c` resta `c` en cada salto **efectivo** (acción 1 con el cubo apoyado; apretar en el aire no cuesta,
porque no hace nada). Como el agente maximiza el retorno **total**, sigue pagando los saltos necesarios: no saltar un
pincho cuesta `−death_penalty` y todo el progreso restante, mucho más que `c`. Sólo deja de hacer los saltos
innecesarios. El retorno del oráculo en `level_1` con penalización es `2 − 13·c`.

### 14.3 Por qué una penalización grande puede romper el aprendizaje

Con `c = 0.1` y `death_penalty = 1`, PPO **no aprende a ganar nunca**. El análisis de valores mostró que,
matemáticamente, saltar sigue siendo siempre mejor que morir (el margen mínimo es +0.80, en el pozo de x = 45.8).
Entonces el problema **no es la economía de la recompensa sino la exploración**:

1. Al principio la política es casi aleatoria y salta mucho.
2. Cada salto cuesta `c` de inmediato, pero el beneficio de un salto bien hecho (cruzar el pozo) llega más tarde y sólo
   si el salto estaba en el momento justo.
3. El gradiente aprende rápido "saltar es caro" y la política deja de saltar **antes** de descubrir que cruzar el pozo
   vale la pena. Sin saltos, ya no puede descubrirlo: la exploración colapsa.

### 14.4 Curriculum: subir `c` de a poco

La idea de **curriculum learning** es enseñar primero la tarea fácil (ganar, con `c = 0`) y después la difícil (ganar
con pocos saltos, con `c` alto). Hay dos modos:

| Modo | Comando | Comportamiento |
|---|---|---|
| Fijo | `--jump-penalty 0.05` | `c` constante todo el entrenamiento (sufijo `_jp0.05`) |
| Rampa | `--jump-penalty 0.1 --ramp` | `c` arranca en 0 y sube hasta 0.1 sólo cuando el agente ya gana seguido (sufijo `_ramp`) |

**Por qué la rampa mira el desempeño y no el tiempo.** Primero se probó una rampa por tiempo (`c = 0` hasta el 30%
del entrenamiento y subida lineal hasta el 60%). No mira al agente, y en las pruebas PPO recién empezaba a ganar
entre los 150k y 250k pasos: la penalización ya estaba subiendo antes de que supiera ganar (sección 14.7). Por eso se
reemplazó por la rampa por desempeño y la rampa por tiempo ya no está en el código.

**La rampa** (`PerformanceJumpPenaltyRamp`) mira los últimos `--ramp-window` (20) episodios de
entrenamiento.

```
si éxito ≥ ramp-up (80%)   → c += ramp-step (0.01), sin pasar de --jump-penalty
si éxito ≤ ramp-down (40%) → c −= ramp-step, sin bajar de 0
si no                      → no cambia
después de cada cambio se vacía la ventana: la próxima decisión usa episodios jugados con el c nuevo
```

El umbral es 80% y no 100% porque los episodios de entrenamiento incluyen exploración y fallan más que la evaluación
determinista. La banda entre 40% y 80% evita que `c` oscile en cada episodio (histéresis). Combinada con la progresión
de niveles, cada nivel nuevo baja la tasa de éxito y la rampa puede aflojar `c` un rato.

**Detalle de implementación:** para cambiar `c` en los 8 entornos en caliente se usa
`env_method("set_wrapper_attr", "jump_penalty", c)`, que atraviesa los wrappers hasta `GeometryDashEnv`.
`VecEnv.set_attr` **no** sirve: dejaría el atributo en el `Monitor` sin efecto alguno (hay un test que lo verifica).

La rampa registra cada cambio en `jump_penalty.csv` (paso, `c`, tasa de éxito que lo motivó).

### 14.5 Bonus por seguir vivo (`survive`)

Con varios niveles de distinto largo aparece un problema de escala. El progreso está normalizado (cruzar el nivel vale
`progress_scale` sea cual sea su largo), pero el costo total de saltar crece con el largo: en un nivel largo con
muchos saltos, lo que queda por pagar en saltos puede superar lo que queda por ganar en progreso, y entonces **morir
conviene**. Como el cubo avanza solo, un bonus por tick vivo no premia "esperar" sino **cada tramo del nivel que
sobrevive**, y crece con el largo igual que los saltos.

| Parámetros | Qué suma |
|---|---|
| `--survive-bonus b --survive-interval N` | `b` cada `N` ticks vivo (en los ticks `t = N, 2N, ...`, no en el que muere). Tasa por tick = `b / N` |
| `--survive-rate r --survive-interval N` | `r · c · N` cada `N` ticks, con `c` = la `jump_penalty` **vigente** (sigue a la rampa). Tasa por tick = `r · c` |

`survive_rate` existe para no tener que recalcular el bonus al cambiar de nivel o de `c`: la condición para que morir
nunca convenga es que la tasa por tick supere `c × saltos_por_tick` del nivel, o sea `r > saltos_por_tick`. En los
niveles registrados el máximo es ~0.056 saltos por tick (`muy_dificil`: 15 saltos en 270 ticks; `level_1`: 13 en 520),
así que `r = 0.1` deja margen en todos. Ejemplo: `--jump-penalty 0.05 --survive-rate 0.1 --survive-interval 20` da
`0.1 × 0.05 × 20 = 0.1` cada 20 ticks.

### 14.6 Salto preciso (`salto_preciso`, Santi)

La penalización fija cobra igual un salto necesario que uno inútil; el agente sólo aprende la diferencia a través de
la muerte, que llega tarde. La recompensa de **salto preciso** cobra el salto según **lo que hay adelante**: mira las
columnas que cubre el salto desde donde despega (`JUMP_SPAN` = 3.2 tiles, lo que avanza un salto en 16 ticks) y suma,
**sólo en los ticks con salto efectivo** (`salto_preciso_term` en `gd_env/rewards.py`):

| Lo que hay en el tramo del salto | Ajuste |
|---|---|
| nada (ni pincho, ni pozo, ni cambio de altura) | `−0.06`: salto innecesario |
| un solo peligro, o sólo un escalón | `0`: el salto sale gratis |
| 2 o más peligros (pinchos o columnas de pozo, p. ej. `^^`, `^^^`, un pozo ancho) | `+0.03` por cada peligro |

Es un término más de la recompensa, así que se combina con progreso, meta y muerte:

```bash
python scripts/train.py --algos ppo --levels niveles/santi.json --rewards progress win death salto_preciso
```

Es **excluyente** con `jump_penalty` (y con `--ramp`): los dos definen el costo de saltar. La versión original de
Santi sigue disponible como recompensa suelta, `--reward-fn ejemplos/reward_santi.py:salto_preciso` (recompensa por
defecto con `win_bonus = 2` más este término).

Cuidados al interpretarla:

- No es shaping "neutral" (no es basado en potencial): premiar saltos sobre muchos peligros puede cambiar qué solución
  prefiere el agente, no sólo acelerar el aprendizaje. Por eso se compara aparte (E5).
- Fue pensada para niveles con obstáculos **en el piso**, como los de Santi. Mira la altura del bloque más alto de cada
  columna y si hay algún pincho: con techos o plataformas flotantes una columna con techo cuenta como "escalón" (el
  salto nunca se castiga), un pozo bajo un techo no cuenta como peligro y un pincho colgante cuenta como peligro
  aunque saltar debajo mate.

### 14.7 Primeras corridas (preliminares, `level_1`)

Estas son las corridas con las que se diseñó la penalización y la rampa. Todas con `features`, `level_1` fijo
(`x0 = 2.0` en entrenamiento) y 300k pasos; "saltos/oráculo" es sobre las fases ganadas (1.00 = óptimo, 13 saltos).
Las que tienen carpeta en `runs/` lo indican; el resto se corrió con versiones anteriores del código y sus carpetas no
están en el repositorio. Las de 1 seed son indicativas.

| Corrida | Algoritmo | seeds | éxito (fases) | saltos / oráculo | en `runs/` |
|---|---|---|---|---|---|
| sin penalización, muerte −1 | PPO | 1 | 100% | 2.09 (~27 saltos) | `features_fixed` |
| `c = 0.02`, muerte −1 | PPO | 1 | 100% | 1.96 | `features_fixed_jp0.02` |
| `c = 0.05`, muerte −1 | PPO | 1 | 100% | 1.46 | `features_fixed_jp0.05` |
| `c = 0.1`, muerte −1 | PPO | 1 | **0%** (43% de progreso) | — | `features_fixed_jp0.1` |
| `c = 0.05`, muerte −1 | PPO | 3 | 93 ± 9% | 1.40 (~19 saltos) | no |
| `c = 0.1`, muerte −1 | PPO | 3 | **0%** | — | no |
| `c = 0.1` rampa por tiempo 0.3–0.6, muerte −1 | PPO | 3 | 53 ± 37% | 1.36 | no |
| `c = 0.1`, muerte −10 (mejor modelo) | PPO | 1 | 100% | 1.66 | no |
| `c = 0.5`, muerte −10 (mejor modelo) | PPO | 1 | 100% | 1.62 | no |
| `c = 0.5`, muerte −10 | DQN | 1 | 100% | **1.09** | no |

Lecturas:

- La penalización por salto **funciona**: baja de ~27 a ~19 saltos con `c = 0.05`, sin perder éxito.
- Con muerte −1, `c = 0.1` fijo colapsa por falta de exploración (sección 14.3). La rampa por tiempo (que ya no
  está en el código) lo rescata sólo en parte y con mucha varianza entre seeds.
- Con muerte −10 (el valor por defecto actual) el agente tolera penalizaciones más grandes: morir pasa a ser mucho peor
  que cualquier cantidad razonable de saltos, y la señal "no saltar me mata" es más fuerte.
- El mejor resultado en saltos lo obtuvo DQN con `c = 0.5`: 1.09 veces el óptimo.

---

## 15. Guardado del mejor modelo

### 15.1 Por qué hace falta

PPO resultó **muy inestable** en este problema. En una corrida, el checkpoint de 300 000 pasos ganaba 0 de 10 fases y
el modelo final (301 056 pasos, **una sola actualización después**) ganaba 8 de 10 con 13 saltos (el óptimo).
Reportar sólo el último modelo es una lotería: depende de en qué actualización se cortó.

Es un fenómeno conocido en RL: la política se apoya en decisiones de timing muy finas (ventanas de 1 a 3 ticks en los
saltos más precisos, sección 9), y un cambio chico en los pesos puede mover un salto un tick y hacer que muera.

### 15.2 Cómo funciona (`BestModelCallback`)

Cada `--eval-freq` pasos (25k; `0` lo desactiva) y al terminar:

1. evalúa el modelo en las 10 fases de cada nivel de entrenamiento (determinista, con la misma recompensa que el
   entrenamiento);
2. lo compara con el mejor hasta ahora por un criterio **lexicográfico**: más fases ganadas → más progreso → menos
   saltos;
3. si es mejor, guarda `best_model.zip` y `best_model.json` (paso y métricas, también por nivel y de validación);
4. siempre agrega una fila a `eval_history.csv` (con `is_best = 1` cuando mejoró el récord), que es una **curva de
   aprendizaje determinista**, mucho menos ruidosa que la de los `monitor/`. Si hay niveles de validación, también los
   evalúa y los anota en las columnas `val_*`, pero **no** cuentan para elegir.

Al final `train` evalúa por separado el modelo final y el mejor, y `resumen.md` trae una tabla de cada uno (en
entrenamiento y en validación).

> Elegir el mejor modelo con las mismas fases con que se reporta introduce un sesgo optimista leve. Para un informe
> riguroso conviene re-evaluar `best_model.zip` con más fases: `scripts/evaluate.py --run runs/<corrida> --best
> --n-phases 20`.

---

## 16. Continuar entrenando un modelo

`--init-model <ruta.zip>` carga un modelo (final, mejor o checkpoint) y sigue entrenándolo:

```bash
# aprender a ganar sin penalización y después afinar con penalización
python scripts/train.py --algos ppo --seeds 0 --timesteps 200000 \
    --init-model runs/features_fixed/ppo/seed0/best_model.zip --jump-penalty 0.05
```

Es otra forma de curriculum, manual: dos etapas con recompensas distintas.

- `--algos` tiene que tener **un solo** algoritmo, el del modelo.
- Se conservan los **pesos y los hiperparámetros** guardados en el `.zip`.
- `--timesteps` son pasos **adicionales**; la recompensa, los niveles y `--train-phase` son los de la corrida nueva.
- **DQN y QR-DQN**: el replay buffer no se guarda en el `.zip`, así que arranca vacío. Además SB3 reiniciaría ε en 1.0
  (jugar al azar el 30% del entrenamiento y tirar lo aprendido), así que `load_for_training` fija ε en su valor final
  (0.05).
- PPO y A2C pueden continuar con otra cantidad de entornos en paralelo.
- Con `--ramp`, `c` vuelve a arrancar en 0; con `progresion`, se vuelve a la primera etapa.
- La corrida nueva se guarda con el sufijo `_continuado` y `config.json` registra `init_model`.

---

## 17. Scripts y archivos de salida

| Script | Uso típico |
|---|---|
| `scripts/run_oracle.py --ascii [--x0 2.5]` | imprime `level_1`, lo resuelve con el oráculo y lo reproduce |
| `scripts/difficulty.py [NIVEL ...] [--detalle] [--ascii] [--fases N]` | tabla de dificultad de los niveles (sección 9) |
| `scripts/train.py --algos ppo a2c dqn qrdqn --seeds 0 1 2` | entrena, evalúa y graba GIFs (llama a `agents.train`) |
| `scripts/evaluate.py --run runs/<corrida> [--best] [--gif] [--n-phases 20]` | re-evalúa modelos guardados (lee la recompensa, los niveles y `--obs` del `config.json`) |
| `scripts/record.py [--model ... --algo PPO --obs ...]` | GIF de un episodio del oráculo o de un modelo |

Opciones de `train.py`, por grupo (la lista completa, con ejemplos, está en el [README](README.md)):

- **generales**: `--algos {ppo,a2c,dqn,qrdqn}`, `--seeds`, `--timesteps`, `--obs {features,grid,dict}`,
  `--train-phase {fixed,random}`, `--n-envs`, `--run-name`, `--out`, `--no-gif`;
- **niveles**: `--levels NIVEL|ARCHIVO.json ...`, `--val-seed`;
- **recompensa**: `--rewards TERMINO ...`, `--jump-penalty`, `--death-penalty`, `--win-bonus`, `--progress-scale`,
  `--survive-bonus`, `--survive-interval`, `--survive-rate`, `--reward-fn ARCHIVO.py:FUNCION`;
- **rampa**: `--ramp` (sí/no), `--ramp-step`, `--ramp-window`, `--ramp-up`, `--ramp-down`;
- **guardado**: `--checkpoint-freq` (25k), `--eval-freq` (25k), `--init-model`.

**Nombre de la corrida.** Si no se pasa `--run-name`, la carpeta `runs/<nombre>/` se arma con la configuración, en
este orden (`run_name_from` en `agents/training.py`):

```
<obs>_<fase>[_lv-<conjunto>][_prog][_jp<c>][_ramp][_rw-<términos>][_dp..][_wb..][_ps..][_sb..][_si..][_sr..][_rf_<función>][_continuado]
```

`lv-` lleva el nombre del `.json` (sin extensión) o los nombres sueltos unidos con `+`; `prog` aparece si el `.json`
tiene `progresion`; `rw-` sólo si se pasó `--rewards`; `dp`/`wb`/`ps`/`sb`/`si`/`sr` sólo para los parámetros de
recompensa que se pasaron explícitamente. Ejemplos: `grid_fixed_lv-nacho_progresion_prog_jp0.05`,
`features_fixed_lv-santi_rw-progress+win+death+salto_preciso`, `features_fixed_jp0.05_si20_sr0.1`.
**Nunca se pisa una corrida**: si la carpeta existe, se agrega fecha y hora (`<nombre>_<AAAAMMDD-HHMMSS>`).

```
runs/<corrida>/
├── config.json              todos los parámetros, lo pedido y lo resuelto (niveles, validación, progresión); se escribe al empezar
├── resumen.md               tablas entre seeds: modelo final y mejor, en niveles de entrenamiento y de validación
└── <algo>/seed<N>/
    ├── model.zip            modelo al terminar
    ├── best_model.zip/.json mejor modelo de las evaluaciones periódicas
    ├── eval_history.csv     todas las evaluaciones periódicas (success_<nivel>; val_*: niveles de validación)
    ├── eval_fases.csv       evaluación del modelo final  (eval_fases_best.csv: del mejor; *_val.csv: validación)
    ├── checkpoints/         modelos intermedios
    ├── monitor/             episodios de entrenamiento, un CSV por entorno (con el nivel jugado)
    ├── jump_penalty.csv     evolución de c (sólo con --ramp)
    ├── progresion.csv       etapas, pesos y éxito por nivel (sólo con "progresion" en el .json)
    └── episodio_<nivel>_x0_*.gif, *_recorrido.png   del modelo final (mejor_*: del mejor; val_*: validación)
```

Las corridas viejas (sin niveles en el `config.json`) siguen funcionando con `evaluate.py`: se evalúan en `level_1`.

---

## 18. Resultados experimentales

> Sección en construcción: los números se completan a medida que terminan las corridas. Salvo que se indique otra
> cosa, cada experimento se reporta con el mejor modelo (`best_model.zip`), media ± desvío entre seeds, en éxito
> (% de fases), progreso y saltos / oráculo, en niveles de entrenamiento y de validación.

### E1. Algoritmos

Configuración: PPO, A2C, DQN y QR-DQN con la misma observación (`features`), la recompensa por defecto, 3 seeds y
300k pasos cada uno (`--run-name e1_algoritmos`).

Resultados: pendiente.

### E2. Observaciones

Configuración: el mismo algoritmo, seeds y pasos con `--obs features`, `grid` y `dict`, para medir cuánto cuesta
aprender la representación y si la limitación de `features` con techos (sección 7.2) se nota.

Resultados: pendiente.

### E3. Penalización por salto fija vs. rampa

Configuración: `--jump-penalty c` fijo contra `--jump-penalty c --ramp` (misma `c` máxima), mismas seeds y pasos;
se mira éxito y saltos / oráculo.

Resultados: pendiente.

### E4. Fase fija vs. aleatoria

Configuración: `--train-phase fixed` (siempre `x0 = 2.0`) contra `--train-phase random` (`x0` sorteado en cada
episodio), evaluando en las 10 fases: mide cuánto memoriza ticks el agente entrenado con fase fija.

Resultados: pendiente.

### E5. Recompensa `salto_preciso` en los niveles de Santi

Configuración: `--levels niveles/santi.json` (validación `level_1`) con `--rewards progress win death salto_preciso`
contra la recompensa con `jump_penalty` fija, mismas seeds y pasos.

Resultados: pendiente.

### E6. Progresión sí / no en los niveles de Nacho

Configuración: `--levels niveles/nacho_progresion.json` (con etapas) contra `--levels niveles/nacho.json` (los mismos
4 niveles, todos desde el principio), validación `level_1` en los dos casos.

Resultados: pendiente (falta la corrida sin progresión para comparar).

**Corrida anterior (sólo con progresión).** Hay dos corridas previas en `runs/`, ambas PPO, 2 seeds (0 y 1), 300k
pasos, `--jump-penalty 0.05`, fase fija y `niveles/nacho_progresion.json`:
[`runs/grid_fixed_lv-nacho_progresion_prog_jp0.05`](runs/grid_fixed_lv-nacho_progresion_prog_jp0.05/resumen.md) y
[`runs/dict_fixed_lv-nacho_progresion_prog_jp0.05`](runs/dict_fixed_lv-nacho_progresion_prog_jp0.05/resumen.md).
Según sus `resumen.md` (éxito = % de fases ganadas, media ± desvío entre las 2 seeds):

| Observación | Modelo | Entrenamiento (4 niveles de Nacho) | Validación (`level_1`) |
|---|---|---|---|
| `grid` | final | 50 ± 0% | 25 ± 25% |
| `grid` | mejor | 65 ± 12% | 25 ± 25% |
| `dict` | final | 69 ± 6% | 50 ± 50% |
| `dict` | mejor | 76 ± 1% | 0 ± 0% |

Lo que muestran los archivos de cada seed:

- La progresión **llegó a la última etapa en las 4 seeds** (`progresion.csv`: etapa 3/3 entre los ~97k y ~230k pasos).
- Los mejores modelos ganan `nacho_facil` y `nacho_medio` en todas las fases, `nacho_dificil` en 3 de las 4 seeds y
  `nacho_muy_dificil` casi nunca (0–10% de las fases; `success_<nivel>` de `best_model.json`).
- La **brecha de generalización** es grande: ~65–76% de éxito en los niveles de entrenamiento contra 0–25% en
  `level_1` con el mejor modelo. Además, el mejor modelo de `dict` (elegido con los niveles de entrenamiento) valida
  peor que el modelo final (0% contra 50 ± 50%): elegir por entrenamiento no garantiza generalizar mejor.
- Son 2 seeds y una sola configuración: sin la corrida sin progresión todavía no se puede decir cuánto aporta la
  progresión.

### E7. Dificultad vs. desempeño

Configuración: cruzar las métricas de dificultad de la sección 6.2 (ventana mínima, trampas, cadena más larga,
log10 P(ganar al azar)) con el éxito por nivel (`success_<nivel>`) de los agentes entrenados, para ver qué métrica
predice mejor qué niveles se aprenden.

Resultados: pendiente.

---

## 19. Tests

`pytest tests` junta 112 casos (contando los parametrizados) en cuatro archivos. Los de QR-DQN se saltean si no está
instalado `sb3-contrib`. Cubren:

- **física y entorno** (`test_env.py`): que cumpla la API de Gymnasium en los tres `obs_mode`, la trayectoria del
  salto, el determinismo, que las políticas triviales y la aleatoria no ganen, que el cubo muerto quede quieto
  (también al chocar de costado), el contenido de la grilla, el render y la conversión a ASCII;
- **oráculo**: que gane `level_1` con 13 saltos, que sea mínimo comparado con fuerza bruta en un nivel chico y que
  resuelva todas las fases de evaluación;
- **recompensa**: el valor por defecto de la muerte, los parámetros por defecto y cambiados, el bonus cada N ticks y
  `survive_rate` siguiendo a `jump_penalty`, las dos firmas de `reward_fn`, cargarla desde `archivo.py:funcion` y
  reconstruirla desde un `config.json`; la penalización sólo cuenta saltos efectivos y el retorno del oráculo con
  penalización es `2 − 13·c`;
- **términos de la recompensa**: que sin `--rewards` la recompensa no cambie, apagar términos, los errores de
  validación (desconocidos, vacíos, excluyentes, parámetros de términos apagados, antes de entrenar), los valores de
  `salto_preciso` y que los términos viajen en `config.json`;
- **callbacks**: que `c` llegue realmente al entorno interno, la lógica de la ventana de la rampa y su lectura de los
  episodios del `Monitor`, el guardado del mejor modelo y el orden del criterio de comparación;
- **continuar entrenando**: los modelos cargan los mismos pesos, los basados en valor con ε constante, y se puede
  cambiar la cantidad de entornos;
- **varios niveles**: que cada nivel registrado se gane con el oráculo desde las 10 fases, que el sorteo respete los
  pesos (y que un peso 0 lo excluya), forzar un nivel, `max_steps` por nivel, el oráculo siguiendo los cambios de
  nivel, la resolución de `--levels` (lista, número, `null`, sorteo reproducible, errores), la evaluación por
  (nivel, fase) y las columnas `val_*` y por nivel del historial; los niveles de Santi y que `franco_sin_saltar` sólo
  se gane sin saltar;
- **progresión**: el bloque `progresion` del `.json` (etapas válidas e inválidas, niveles que pasan a validación), la
  lógica de `LevelProgression` (desbloqueo, piso, peso máximo sin datos, recuperación de un nivel olvidado) y que los
  pesos lleguen al entorno de adentro;
- **CNN** (`test_cnn.py`): la política correcta para cada `obs_mode`, la forma de salida de `GridCNN` y `GridVecCNN`,
  y entrenar, guardar, cargar y evaluar con cada combinación de algoritmo y observación;
- **dificultad** (`test_difficulty.py`): sección 9.4;
- **pinchos colgantes** (`test_spike_down.py`): ASCII de ida y vuelta, pasar por debajo sin saltar gana y saltar mata,
  la hitbox espejada (la base mata, la punta no), que el pincho común se comporte igual que antes, el canal de la
  grilla, que `features` los vea, y que sólo los niveles de techo los usen.

---

## 20. Limitaciones y trabajo futuro

- **Progresión sin comparación todavía**: hay corridas con progresión (sección 18, E6) pero falta compararlas contra
  entrenar con todos los niveles desde el principio. Con `--init-model` vuelve a empezar desde la primera etapa.
- **Generalización baja**: en las corridas de Nacho el éxito en `level_1` (validación) es mucho menor que en los
  niveles de entrenamiento. Más niveles y más variados (ya hay 23) es la vía natural para achicar la brecha.
- **Elegir el mejor modelo con validación**: hoy se elige con los niveles de entrenamiento. Usar validación requeriría
  un tercer conjunto de niveles de test para seguir teniendo una medida honesta.
- **`features` con techos**: no distingue techo de piso ni pincho colgante de pincho común (sección 7.2); `salto_preciso`
  tiene la misma limitación (sección 14.6).
- **Correr los experimentos pendientes** con 3+ seeds (sección 18).
- **Hiperparámetros sin tunear**: un barrido (learning rate, `ent_coef`, `n_steps`) podría reducir la inestabilidad
  de PPO.
- **Aprovechar el oráculo**: clonado de comportamiento (pre-entrenar imitando al oráculo) o DAgger, y recompensas de
  timing basadas en `plan.jump_x` o en las ventanas de la medida de dificultad.

---

## 21. Glosario

| Término | Significado |
|---|---|
| **Tick** | un paso de simulación = un paso del entorno. El cubo avanza 0.2 tiles por tick |
| **Episodio** | una partida, desde `reset` hasta ganar, morir o agotar el tiempo |
| **Política** `π` | la regla con la que el agente elige acciones |
| **Retorno** | suma descontada de recompensas futuras |
| **γ (gamma)** | factor de descuento; fija el horizonte (~100 ticks con 0.99) |
| **On-policy / off-policy** | si el algoritmo aprende sólo de datos de la política actual (PPO, A2C) o también de datos viejos (DQN, QR-DQN) |
| **Rollout** | tanda de experiencia que PPO junta antes de actualizar (2048 pasos) |
| **Ventaja** | cuánto mejor (o peor) fue una acción que lo esperado |
| **Replay buffer** | memoria de transiciones pasadas de DQN y QR-DQN |
| **Actor-crítico** | método con dos redes (o dos cabezas): la política (actor) y el valor `V` (crítico) |
| **Cuantil** | valor por debajo del cual cae una fracción dada de los retornos; QR-DQN aprende 50 por acción |
| **CNN** | red convolucional: filtros chicos que recorren la imagen buscando patrones locales |
| **Extractor de características** | la parte de la red que convierte la observación en un vector (CNN para `grid`) |
| **`Transition`** | lo que pasó en un tick (estados, resultado, si saltó); es la entrada de una `reward_fn` y de los términos |
| **Término de recompensa** | cada pieza de la recompensa (`progress`, `win`, `death`, `survive`, `jump_penalty`, `salto_preciso`) que se prende o apaga con `--rewards` |
| **Salto preciso** | término de Santi: castiga saltar sin obstáculos adelante y premia saltar sobre varios peligros |
| **ε-greedy** | elegir al azar con probabilidad ε, si no la mejor acción conocida |
| **Reward shaping** | agregar recompensas intermedias para guiar el aprendizaje |
| **Curriculum** | ordenar la dificultad de la tarea de fácil a difícil durante el entrenamiento |
| **Fase de arranque** `x0` | posición inicial del cubo; cambia el timing de los saltos sin cambiar el nivel |
| **Salto efectivo** | acción 1 con el cubo apoyado (la única que realmente salta) |
| **Pincho colgante** (`v`) | pincho que cuelga de un bloque y apunta hacia abajo; con hueco de 1 tile, saltar debajo mata |
| **Oráculo** | planificador exacto que conoce la física; da la solución de mínimos saltos |
| **Ventana (de un salto)** | cantidad de ticks de despegue que siguen ganando sin gastar saltos de más; chica = salto preciso |
| **Trampa** | despegue desde el que el cubo aterriza vivo pero el nivel ya no se puede ganar |
| **Saltos encadenados** | saltos en los que cuándo se despega en uno cambia la ventana del siguiente |
| **P(ganar al azar)** | probabilidad de ganar apretando 50/50 en cada tick; mide cuánto cuesta dar con la solución explorando |
| **Held out** | nivel reservado: nunca se usa para entrenar, sólo para medir generalización |
| **Brecha de generalización** | diferencia de desempeño entre los niveles de entrenamiento y los de validación |
| **Progresión de niveles** | introducir los niveles por etapas, de fácil a difícil (*curriculum learning*) |
| **Olvido catastrófico** | perder lo aprendido en una tarea al entrenar mucho tiempo sólo en otra |
| **Peso de un nivel** | probabilidad relativa de que ese nivel salga en un episodio de entrenamiento |
| **Seed** | semilla aleatoria; distintas seeds = distintas corridas independientes |
| **VecEnv** | varias copias del entorno que avanzan juntas |
| **Callback** | función que SB3 llama durante el entrenamiento para observarlo o modificarlo |
| **Checkpoint** | copia del modelo guardada a mitad del entrenamiento |
