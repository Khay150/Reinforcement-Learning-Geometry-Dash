"""Pinchos colgantes ('v'): hitbox espejada, ASCII, observación y que los niveles anteriores no cambien."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gd_env import LEVELS, GeometryDashEnv, get_level, parse_level, to_ascii
from gd_env import physics as ph

# Piso (fila 0), un hueco de 1 tile (fila 1) y, en las columnas 8-16, una plataforma (fila 3) con pinchos colgando (fila 2).
TECHO = "\n".join([
    "." * 30,
    "." * 8 + "#" * 9 + "." * 13,
    "." * 8 + "v" * 9 + "." * 13,
    "." * 30,
    "#" * 30,
])


CON_COLGANTES = {"franco_techo", "franco_tunel", "franco_pasillos", "franco_sin_saltar"}   # niveles que usan el pincho colgante 'v'


def _run(level, jump_from_x=None):
    """Corre el nivel sin saltar, o saltando (una vez apoyado) desde ``jump_from_x``. Devuelve el status final."""
    p = ph.Physics()
    s, status = ph.initial_state(level, p), ph.OK
    while status == ph.OK:
        jump = jump_from_x is not None and s.x >= jump_from_x and s.on_ground
        s, status = ph.step(level, p, s, 1 if jump else 0)
    return status


def test_ascii_roundtrip_con_pincho_colgante():
    lv = parse_level(TECHO)
    assert (lv.grid == ph.SPIKE_DOWN).sum() == 9
    assert to_ascii(lv) == TECHO


def test_pasar_por_debajo_sin_saltar_gana():
    assert _run(parse_level(TECHO)) == ph.WIN


def test_saltar_debajo_del_pincho_colgante_mata():
    assert _run(parse_level(TECHO), jump_from_x=9.0) == ph.SPIKE_HIT


def test_hitbox_espejada_la_base_mata_y_la_punta_no():
    """Pincho colgante en la fila 2 (ocupa y de 2 a 3): la hitbox es el 0.3 de arriba (2.7 a 3). Un cubo cuya cabeza
    llega a 2.6 (y = 1.6) no lo toca; con la cabeza en 2.8 (y = 1.8) sí."""
    lv, p = parse_level(TECHO), ph.Physics()
    assert not ph._spike_hit(lv, p, 11.0, 1.6)
    assert ph._spike_hit(lv, p, 11.0, 1.8)


def test_el_pincho_hacia_arriba_se_comporta_igual_que_antes():
    lv, p = get_level("level_1"), ph.Physics()
    r, c = sorted(lv._spikes)[0]
    assert ph._spike_hit(lv, p, float(c), float(r))          # apoyado sobre el pincho: muere
    assert not ph._spike_hit(lv, p, float(c), r + 0.4)       # 0.4 por encima de la base: ya no lo toca


def test_grilla_marca_los_colgantes_en_el_canal_de_pinchos_y_conserva_la_forma():
    env = GeometryDashEnv(level=parse_level(TECHO), obs_mode="grid", window=16)
    obs, _ = env.reset()
    assert obs.shape == (3, 5, 16)                           # mismos 3 canales que antes
    assert obs[1].sum() > 0                                  # los colgantes se ven en el canal de pinchos


def test_features_ven_el_pincho_colgante():
    lv = parse_level(TECHO)
    assert lv.has_spike(10) and not lv.has_spike(2)


@pytest.mark.parametrize("name", list(LEVELS))
def test_solo_los_niveles_de_techo_tienen_pinchos_colgantes(name):
    """Los niveles anteriores al tile 'v' no tienen ninguno, así que su física es exactamente la de antes."""
    assert bool(get_level(name)._spikes_down) == (name in CON_COLGANTES)
