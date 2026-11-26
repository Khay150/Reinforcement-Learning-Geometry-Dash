# Experimentos para la presentación

Cada integrante corre **un** script en su compu. Son entre 1h45 y 2h15 por script, con 4 núcleos y sin GPU. La lista de
experimentos (E1 a E7) es la de la sección 18 de `DOCUMENTACION.md`, más E8 (generalización).

| Script | Corridas | ~Tiempo |
|---|---|---|
| `integrante_1.py` | `e2_obs_grid`, `base_ppo`, `e3_fijo`, `e3_rampa` + 2 niveles de E7 | 111 min |
| `integrante_2.py` | `e2_obs_dict`, `e4_random`, `e5_santi_jp`, `e5_santi_salto_preciso` + 2 niveles de E7 | 111 min |
| `integrante_3.py` | `e1_qrdqn` (necesita `sb3-contrib`) + los 8 niveles más difíciles de E7 | 105 min |
| `integrante_4.py` | `e8_generalizacion` | 135 min |

- **Algoritmos:** sólo PPO y QR-DQN. `base_ppo` (PPO, `features`, `level_1`, recompensa por defecto, fase fija) es la
  referencia de E1 a E4, así que no se repite.
- **E6** (progresión sí o no) ya está en `runs/`.
- **E7** entrena con `features` en un solo nivel, así que usa sólo niveles sin techo ni bloques flotantes: `features`
  muestra el bloque más alto de cada columna y un techo se ve como una pared.
- **E8** entrena en 20 niveles y valida en 3 que nunca ve: `santi_facil` (fácil), `level_1` (medio) y `nacho_dificil`
  (difícil); ver `niveles/generalizacion.json`. Usa `dict`, porque entre los 20 están los niveles con techo.

## Cómo correrlo

```bash
git pull
pip install -r requirements.txt               # con el .venv activo (ver SETUP.md)
python experimentos/integrante_N.py --dry-run # qué va a correr y cuánto falta
python experimentos/integrante_N.py --subir   # entrena y sube cada corrida al terminar
```

- **Si se corta** (se apaga la compu, Ctrl+C), volvé a lanzar el mismo comando. Las corridas terminadas
  (las que tienen `runs/<nombre>/resumen.md`) se saltean.
- **Si una corrida quedó a medias**, su carpeta en `runs/` existe pero no tiene `resumen.md`. El script avisa y no la
  toca: borrá esa carpeta y volvé a correr.
- **`--subir` hace un commit por seed** y un push después de cada uno, porque cada seed pesa entre 15 y 160 MB.
  - Si el push falla porque alguien subió algo antes: `git pull --rebase`, y después `python experimentos/subir.py runs/<nombre>`.
  - Lo que ya estaba subido se saltea.
- **Sin `--subir`**, al final se imprime el comando de `subir.py` para hacerlo a mano.

## Gráficos

Cuando estén todas las corridas (`git pull`):
```bash
python experimentos/graficos.py      # figuras/e1 ... figuras/e8
```
Las corridas que todavía faltan se avisan y se saltean, así que se puede correr antes para ver cómo viene.
