"""Lo que comparten los scripts de ``experimentos/``: correr una lista de entrenamientos y subir las corridas.

Cada entrenamiento es un ``Corrida(nombre, args)``: se ejecuta ``python scripts/train.py <args> --run-name <nombre>`` desde
la raíz del repo (con el mismo Python que corre el script, así sirve con el .venv activo en Windows, Linux o macOS).

* Si ``runs/<nombre>/resumen.md`` ya existe, la corrida está terminada y se saltea: se puede volver a lanzar el script
  después de un corte y sigue donde quedó.
* Si ``runs/<nombre>/`` existe pero sin ``resumen.md``, quedó a medias: no se toca (train.py la guardaría con otro
  nombre y los gráficos no la encontrarían). Se avisa al final; hay que borrar esa carpeta y volver a correr.
* Con ``--subir``, cada corrida que termina se commitea (un commit por seed) y se pushea en el momento.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
RUNS = RAIZ / "runs"

# Base de E1-E4: level_1, observación features, recompensa por defecto, fase fija, 3 seeds.
N = ["--seeds", "0", "1", "2", "--timesteps", "300000"]

# E7 (dificultad vs. desempeño): PPO + features entrenado en un solo nivel, de más fácil a más difícil según
# scripts/difficulty.py. Sólo niveles sin techo ni bloques flotantes, que features no puede representar (por eso no
# están muy_dificil ni los de Franco con techo). Repartidos entre los integrantes 1, 2 y 3.
E7 = ["franco_facil", "nacho_facil", "facil", "santi_intermedio_1", "nacho_medio", "medio", "dificil",
      "nacho_dificil", "santi_intermedio_2", "nacho_muy_dificil", "santi_dificil", "level_1"]


@dataclass
class Corrida:
    nombre: str            # carpeta runs/<nombre>/ (es lo que buscan los gráficos de experimentos/graficos.py)
    args: list[str]        # argumentos de scripts/train.py (sin --run-name)
    minutos: int           # estimación con 4 núcleos y sin GPU, sólo para mostrar el total

    @property
    def carpeta(self) -> Path:
        return RUNS / self.nombre

    def terminada(self) -> bool:
        return (self.carpeta / "resumen.md").exists()


def e7(niveles: list[str]) -> list["Corrida"]:
    """Una corrida de E7 por nivel: PPO, features, 3 seeds, 200k pasos, sólo ese nivel."""
    args = ["--algos", "ppo", "--seeds", "0", "1", "2", "--timesteps", "200000"]
    return [Corrida(f"e7_nivel_{lv}", [*args, "--levels", lv], minutos=15 if lv == "level_1" else 9) for lv in niveles]


def _chequear_dependencias(corridas: list[Corrida]):
    if any("qrdqn" in c.args for c in corridas):
        try:
            import sb3_contrib  # noqa: F401
        except ImportError:
            sys.exit("Este script entrena QR-DQN y falta sb3-contrib: pip install -r requirements.txt")


def correr(titulo: str, corridas: list[Corrida]):
    """Punto de entrada de cada ``integrante_N.py``: corre las corridas que falten, en orden."""
    ap = argparse.ArgumentParser(description=titulo)
    ap.add_argument("--dry-run", action="store_true", help="sólo muestra los comandos, no entrena")
    ap.add_argument("--subir", action="store_true",
                    help="al terminar cada corrida, commitearla (un commit por seed) y pushearla")
    a = ap.parse_args()
    os.chdir(RAIZ)

    pendientes = [c for c in corridas if not c.terminada()]
    print(f"== {titulo}")
    for c in corridas:
        estado = "terminada" if c.terminada() else ("A MEDIAS" if c.carpeta.exists() else f"~{c.minutos} min")
        print(f"  {c.nombre:<36} {estado}")
    print(f"Faltan {len(pendientes)} de {len(corridas)} (~{sum(c.minutos for c in pendientes)} min en total).\n")
    if a.dry_run:
        for c in pendientes:
            print(" ".join(["python", "scripts/train.py", *c.args, "--run-name", c.nombre]))
        return
    _chequear_dependencias(pendientes)

    a_medias, fallidas = [], []
    for i, c in enumerate(pendientes, 1):
        if c.carpeta.exists():
            a_medias.append(c.nombre)
            print(f"[{i}/{len(pendientes)}] {c.nombre}: la carpeta existe pero no terminó; la salteo.\n")
            continue
        cmd = [sys.executable, "scripts/train.py", *c.args, "--run-name", c.nombre]
        print(f"[{i}/{len(pendientes)}] {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        if subprocess.run(cmd).returncode != 0:
            fallidas.append(c.nombre)
            print(f"  -> falló {c.nombre}\n")
            continue
        print(f"  -> {c.nombre} terminada en {(time.time() - t0) / 60:.0f} min\n", flush=True)
        if a.subir:
            subir([c.carpeta])

    print("== Listo.")
    if a_medias:
        print("Quedaron a medias (borrá esas carpetas de runs/ y volvé a correr el script):", ", ".join(a_medias))
    if fallidas:
        print("Fallaron:", ", ".join(fallidas))
    if not a.subir:
        hechas = [c.nombre for c in corridas if c.terminada()]
        print("Para subirlas (un commit por seed):\n  python experimentos/subir.py " +
              " ".join(f"runs/{n}" for n in hechas))


# ------------------------------------------------------------------------------------------------------ subir
def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=RAIZ, check=check, text=True, capture_output=True)


def subir(carpetas: list[Path]):
    """Commitea cada corrida con un commit por seed (``config.json`` y ``resumen.md`` van con la primera) y pushea
    después de cada commit, así ningún push es enorme. Lo que ya está commiteado se saltea."""
    for carpeta in carpetas:
        carpeta = Path(carpeta).resolve()
        if not (carpeta / "resumen.md").exists():
            print(f"{carpeta.name}: no terminó (no hay resumen.md); no la subo.")
            continue
        seeds = sorted(carpeta.glob("*/seed*"))
        for k, seed in enumerate(seeds):
            rutas = [seed] + ([carpeta / "config.json", carpeta / "resumen.md"] if k == 0 else [])
            rutas = [str(p.relative_to(RAIZ)) for p in rutas]
            _git("add", "--", *rutas)
            if _git("diff", "--cached", "--quiet", "--", *rutas, check=False).returncode == 0:
                continue                                    # nada nuevo en esta seed: ya estaba subida
            msg = f"Corrida {carpeta.name}: {seed.parent.name}/{seed.name}"
            _git("commit", "-m", msg, "--", *rutas)         # sólo estas rutas, aunque haya otras cosas en el stage
            print(f"  commit: {msg}", flush=True)
            push = _git("push", check=False)
            if push.returncode != 0:
                sys.exit("El push falló (¿alguien subió algo antes?). Corré `git pull --rebase` y después "
                         f"`python experimentos/subir.py {carpeta.relative_to(RAIZ)}` para seguir.\n" + push.stderr)
