# Presentación: Geometry Dash con Aprendizaje por Refuerzo

Guion por diapositiva, siguiendo los 8 bloques de `Guia_Presentacion_8_Bloques_RL_Geometry_Dash.pdf`, con estas
correcciones respecto de esa guía:

- Los algoritmos son **PPO y QR-DQN** (sacamos A2C y DQN).
- Los resultados salen de las corridas nuevas de `experimentos/`, no de las `features_fixed_jp*`, que son de 1 seed y de
  una versión anterior del código.
- Se agregan **E7** (dificultad contra desempeño) y **E8** (generalización con niveles held out).

Cada diapositiva tiene: lo que se dice, la figura y de dónde sale el dato. Lo marcado **[pendiente]** se completa cuando
terminen las corridas (`python experimentos/graficos.py` arma `figuras/e1` … `figuras/e8`). Los resultados se reportan
con el **mejor modelo** (`best_model.zip`), media ± desvío entre seeds, sobre las 10 fases de arranque.

---

## Bloque 1. El problema

### Diapositiva 1: Por qué Geometry Dash es un desafío para RL
- **Una sola acción, saltar o no, pero con mucha precisión.** Un tick (0.2 bloques) antes o después puede ser la
  diferencia entre ganar (+2) y morir (−10).
- **Crédito diferido.** La decisión se toma al despegar, pero el resultado (aterrizar vivo o pincharse) llega hasta
  16 ticks después.
- **Explorar es difícil.** Una política aleatoria (50/50 en cada tick) gana con probabilidad entre 10⁻² y 10⁻³³ según el
  nivel (`scripts/difficulty.py`).
- **Figura:** GIF del oráculo en `level_1` (`python scripts/record.py --out videos/oraculo`).

### Diapositiva 2: Física (`gd_env/physics.py`)
- **Avance constante:** x = x0 + 0.2·t. `level_1` se juega en 520 ticks.
- **Salto:** velocidad inicial 0.4 y gravedad 0.05. En piso plano dura **16 ticks**, sube unos 1.4 bloques y avanza **3.2 bloques**.
- **Saltar sólo surte efecto con el cubo apoyado.** Apretar en el aire no hace nada.
- **4 formas de morir:** chocar de costado con un bloque, pincho de piso (`^`), pincho colgante (`v`) y caer a un pozo.
- **Figura:** recorrido del oráculo (`videos/oraculo_recorrido.png`): estela y triángulos de despegue.

## Bloque 2. MDP y observaciones (`gd_env/env.py`)

### Diapositiva 3: El MDP
- **Estado:** lo que ve el agente alrededor del cubo. No ve la x absoluta ni el tick, para que no memorice ticks.
- **Acciones:** {0 = nada, 1 = saltar}.
- **Recompensa:** progreso (suma 1 en todo el nivel) + 1 al ganar − 10 al morir − c por cada salto efectivo.

### Diapositiva 4: Tres observaciones
| `--obs` | Qué es | Limitación |
|---|---|---|
| `features` (MLP) | 28 números: [y, vy, en el piso, frac(x)] + 12 columnas × [altura del bloque más alto, hay pincho] | Ve sólo el bloque más alto de cada columna: un techo parece una pared. Es ambiguo en 7 de los 23 niveles |
| `grid` (CNN) | imagen 3×7×16: bloques, pinchos, jugador | No trae la velocidad vertical ni la posición dentro del tile: la imagen no cambia durante **2 o 3 ticks seguidos** en el piso |
| `dict` (CNN + vector) | `grid` + [y, vy, en el piso, frac(x)] | No tiene la limitación de ninguna de las otras dos |

- **Idea clave:** sin `frac(x)` el agente no puede distinguir ticks que se ven iguales. Con una política determinista,
  una ventana de salto de 1 tick sólo se acierta de casualidad.

## Bloque 3. Redes (`agents/cnn.py`)

### Diapositiva 5: Por qué una CNN propia
- **NatureCNN no sirve:** es la de SB3 (pensada para Atari) y exige imágenes de al menos 36×36. Con filtros de 8×8 y
  stride 4, nuestra grilla de 7×16 desaparece.
- **Sin pooling ni stride:** cada celda es un bloque, y achicar la imagen haría perder en qué columna está el pincho.
- **GridCNN** (`grid`): Conv 3×3 (3→32) → Conv 3×3 (32→64), las dos con padding 1 → Flatten (7168) → Linear (128).
- **GridVecCNN** (`dict`): la misma CNN (→128), en paralelo Linear(4→32) para el vector, concatenadas (160).
- **Después de cualquier extractor** van 2 capas ocultas de 64 (política y valor).
- **Figura:** diagrama de GridVecCNN **[a hacer]**.

## Bloque 4. Oráculo y dificultad

### Diapositiva 6: El oráculo (`gd_env/oracle.py`)
- **Programación dinámica:** como x depende sólo del tick, los estados forman un DAG por capas, y se recorre guardando
  la mínima cantidad de saltos.
- **Da la solución óptima:** **13 saltos** en `level_1`. Es la referencia de la métrica **saltos / oráculo**
  (1.0 = óptimo).
- **Los tests comprueban** que los 23 niveles se pueden ganar desde las 10 fases de arranque que se usan para evaluar.

### Diapositiva 7: 23 niveles y su dificultad (`scripts/difficulty.py`)
- **Origen:** `level_1` más los niveles de Nacho, Rugo, Franco y Santi.
- **Métrica**, sin entrenar y en el peor caso entre las 10 fases:
  - **ventana mínima:** cuántos ticks de despegue distintos siguen ganando. 1 tick = salto pixel-perfect;
  - **trampas:** despegues desde los que el cubo aterriza vivo, pero el nivel ya no se puede ganar;
  - **cadenas de saltos:** saltos seguidos donde cuándo despegás en uno cambia la ventana del siguiente;
  - **log10 P(ganar al azar):** de −2.0 (`franco_facil`) a −14.0 (`santi_dificil`) y −33.1 (`muy_muy_dificil`).
- **Figura:** `niveles/png/_todos_los_niveles.png` + la tabla de `python scripts/difficulty.py`.

## Bloque 5. Recompensa: del spam al salto preciso

### Diapositiva 8: Sin costo por saltar, salta todo el tiempo (E3)
- **Comparación:** sin castigo, c = 0.1 fijo y c = 0.1 con rampa (PPO, `features`, `level_1`, 3 seeds).
- **Qué esperamos:**
  - sin castigo, gana saltando de más (en las corridas viejas, unos 27 saltos contra 13: 2.1 veces el óptimo);
  - con c alto y fijo, deja de saltar antes de aprender a cruzar y no gana;
  - la rampa (c arranca en 0 y sube 0.01 cada vez que gana ≥ 80% de los últimos 20 episodios) le da tiempo a aprender
    primero a ganar.
- **Figuras:** `figuras/e3/barras_entrenamiento.png` (éxito y saltos/oráculo) y `penalizacion_salto.png` (cómo sube c).
  Los GIFs de cada corrida, lado a lado.
- **Datos:** **[pendiente: `runs/base_ppo`, `runs/e3_fijo`, `runs/e3_rampa`]**.

### Diapositiva 9: Recompensa `salto_preciso` (E5, opcional si falta tiempo)
- **Qué hace:** saltar sin ningún obstáculo en los 3.2 bloques del salto cuesta 0.06; saltar sobre un peligro es gratis,
  y si cruza 2 o más (`^^`, un pozo ancho) suma 0.03 por cada uno.
- **Comparación:** contra el castigo fijo de 0.05, en los 5 niveles de Santi, validando en `level_1`.
- **Datos:** **[pendiente: `runs/e5_santi_jp`, `runs/e5_santi_salto_preciso`]**.

## Bloque 6. Algoritmos (E1)

### Diapositiva 10: PPO contra QR-DQN
- **PPO (on-policy, actor-crítico):**
  - 8 entornos en paralelo y rollouts de 256 × 8 = 2048 pasos;
  - GAE con λ = 0.95;
  - 10 épocas con minibatches de 256, clip 0.2 y entropía 0.01.
- **QR-DQN (off-policy, distribucional):**
  - buffer de 100k y red objetivo actualizada cada 500 pasos;
  - ε de 1 a 0.05 en el primer 30% del entrenamiento;
  - 50 cuantiles del retorno en vez de un solo valor esperado. El retorno es bimodal (+2 si gana, −10 si muere), y por
    eso lo elegimos sobre DQN.
- **Mejor modelo:** se evalúa cada 25k pasos en las 10 fases. Se queda el que más fases gana; si empatan, el de más
  progreso, y después el de menos saltos.
- **Figuras:** `figuras/e1/curva_exito.png` y `figuras/e1/barras_entrenamiento.png`.
- **Datos:** **[pendiente: `runs/base_ppo`, `runs/e1_qrdqn`]**.

### Diapositiva 11: Observaciones (E2)
- **Comparación:** PPO en `level_1` con `features`, `grid` y `dict`.
- **Qué mirar:**
  - el costo de aprender una representación (la CNN tarda más en arrancar);
  - si `dict` salta más cerca del óptimo que `grid`.
- **Figuras:** `figuras/e2/eval_exito.png` y `figuras/e2/barras_entrenamiento.png`.
- **Datos:** **[pendiente: `runs/base_ppo`, `runs/e2_obs_grid`, `runs/e2_obs_dict`]**.
- **Lo que ya vimos en los niveles de Nacho** (4 niveles, 2 seeds): `dict` gana unos 10 puntos más que `grid` (68% contra
  59% con el modelo final) y resuelve `nacho_dificil` en las 2 seeds. Pero salta igual de más (13–15 saltos por nivel).

## Bloque 7. Generalización

### Diapositiva 12: ¿Memoriza ticks? Fase fija contra aleatoria (E4)
- **Comparación:** entrenar siempre desde x0 = 2.0 contra sortear x0 en cada episodio, evaluando en 10 fases entre 2.0 y
  2.95.
- **Figura:** `figuras/e4/barras_entrenamiento.png`.
- **Datos:** **[pendiente: `runs/base_ppo`, `runs/e4_random`]**.

### Diapositiva 13: Progresión de niveles (E6)
- **Mecanismo (`LevelProgression`):** arranca sólo con `nacho_facil`. Cuando todos los niveles activos ganan ≥ 80% de
  sus últimos 50 episodios, se suma la etapa siguiente (medio, y después difícil y muy difícil).
- **Contra el olvido:** cada nivel pesa base × max(0.2, 1 − éxito). Se practica más lo que más cuesta, sin dejar de
  repasar lo fácil.
- **Resultados** (PPO, `dict`, c = 0.05, 2 seeds, 300k pasos):

| | Entrenamiento, mejor modelo | Entrenamiento, modelo final | `level_1` (held out), final |
|---|---|---|---|
| sin progresión | 82 ± 8% | 68 ± 2% | 0% |
| con progresión | 76 ± 1% | 69 ± 6% | 50 ± 50% (1 de 2 seeds) |
| sin progresión, c = 0.2 | 76 ± 1% (saltos/oráculo 1.38 contra 1.62) | 51 ± 1% | 0% |

- **Lectura honesta:**
  - la progresión llega a la última etapa (a los 97–120k pasos), pero **no mejora** el éxito en entrenamiento;
  - la seed que gana `level_1` lo hace con 26 saltos (el óptimo es 13), y su best_model saca 0%.
- **El olvido existe aun sin progresión:** sin progresión, una seed llegó a ganar el 60% de `nacho_muy_dificil` a los
  125k pasos y en la evaluación siguiente volvió a 0.
- **Figuras:** `figuras/e6/progresion_*.png` (etapas y pesos) y `figuras/e6/barras_niveles.png`.

### Diapositiva 14: Generalización a niveles nuevos (E8)
- **Comparación:** PPO + `dict` entrenado en 20 niveles, validado en 3 que nunca ve: `santi_facil` (fácil), `level_1`
  (medio) y `nacho_dificil` (difícil).
- **Qué mirar:**
  - la **brecha de generalización** (éxito en entrenamiento contra validación);
  - si la brecha crece con la dificultad del nivel held out.
- **Figura:** `figuras/e8/barras_niveles.png` (los 3 niveles de validación aparecen separados abajo).
- **Datos:** **[pendiente: `runs/e8_generalizacion`]**.

## Bloque 8. Resultados y cierre

### Diapositiva 15: ¿La dificultad medida predice lo que le cuesta al agente? (E7)
- **Comparación:** PPO entrenado en un solo nivel, en 12 niveles de distinta dificultad. Correlación de Spearman entre
  cada métrica de dificultad y los pasos hasta la primera victoria, los pasos hasta aprenderlo y el éxito final.
- **Figura:** `figuras/e7/panel.png`, y `figuras/e7/tabla.md` para los números.
- **Datos:** **[pendiente: `runs/e7_nivel_*`]**.

### Diapositiva 16: Conclusiones
**[completar con los resultados]** Lo que ya está respaldado:
1. **Sin costo por saltar, el agente salta casi sin parar.** Es una estrategia segura y nada la castiga. Un costo
   moderado lo recorta; uno alto y fijo impide aprender.
2. **La representación importa.** `features` no puede ver techos; `grid` no puede distinguir ticks dentro de un tile;
   `dict` resuelve las dos cosas y gana más.
3. **Generalizar es lo difícil.** Con 4 niveles de entrenamiento, `level_1` held out casi no se gana, y cuando se gana es
   saltando el doble de lo necesario.
4. **Limitaciones:** 2–3 seeds (desvíos grandes), evaluación determinista (un nivel pasa de 0% a 100% entre una
   evaluación y la siguiente) y el mejor modelo se elige con los niveles de entrenamiento, no con los de validación.

---

## Preguntas probables de los docentes

- **¿Por qué el salto recorre 3.2 bloques?** Dura 16 ticks a 0.2 bloques por tick. En piso plano alcanza para 2 pinchos
  seguidos (`^^`); para 3 (`^^^`) hay que saltar desde un escalón, y así ganar tiempo de vuelo.
- **¿Cuándo conviene la CNN sobre el MLP?** Cuando hay techos, túneles o pinchos colgantes: `features` sólo codifica el
  bloque más alto de cada columna. Sin techos, `features` es más barato y aprende más rápido.
- **¿Por qué no usar directamente un castigo alto por salto?** Porque el agente aprende a no saltar antes de aprender a
  cruzar el primer obstáculo. Por eso la rampa.
- **¿Por qué QR-DQN y no DQN?** El retorno es bimodal (+2 o −10). El promedio esconde ese riesgo y la distribución no.
- **¿El agente memoriza el nivel?** E4 lo mide en un solo nivel (fase fija contra aleatoria) y E8 con niveles que nunca
  vio.
- **¿Por qué el mejor modelo valida peor que el final, a veces?** Porque se elige con los niveles de entrenamiento.
  Elegirlo con validación sería usar la validación para entrenar.

## Pendientes del grupo
- Correr los 4 scripts de `experimentos/` y después `python experimentos/graficos.py`.
- Completar los **[pendiente]** con `runs/<corrida>/resumen.md` y las figuras.
- Hacer el diagrama de GridVecCNN (diapositiva 5) y elegir los GIFs (diapositivas 1 y 8).
- Definir la duración de la exposición, quién presenta cada bloque, y si el póster se arma con este mismo contenido.
