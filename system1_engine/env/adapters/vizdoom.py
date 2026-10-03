import math
from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np

# Nomes de entidades do ViZDoom que não devem ser tratadas como monstros
NON_MONSTER_NAMES = {
    "DoomPlayer",
    "BulletPuff",
    "Blood",
    "BloodSplatter",
    "TeleportFog",
    "Clip",
    "ClipBox",
    "HealthBonus",
    "ArmorBonus",
    "Medikit",
    "Stimpack",
    "GreenArmor",
    "BlueArmor",
    "Soulsphere",
    "Megasphere",
    "InvulnerabilitySphere",
    "Infrared",
    "RadSuit",
}


class VizdoomAimRewardWrapper(gym.Wrapper):
    """Reward Shaping Universal para ambientes ViZDoom.

    Suporta dinamicamente:
      1. Cenários Laterais / Strafe (ex: basic.cfg, simpler_basic.cfg):
         - Botões: MOVE_LEFT, MOVE_RIGHT, ATTACK.
         - Alinhamento no eixo Y com relação ao monstro alvo (|dy| <= 35.0).
         - Recompensa aproximação física e disparo alinhado (+5.0).
         - Penaliza inércia (-3.0), colisão contra paredes (-3.0) e tiro no vazio (-5.0).
      2. Cenários Rotacionais 360° (ex: defend_the_center.cfg, defend_the_line.cfg):
         - Botões: TURN_LEFT, TURN_RIGHT, ATTACK.
         - Radar angular 360° contra múltiplas ameaças convergentes (Demon, Marine, etc.).
         - Priorização automática do monstro mais próximo (menor distância euclidiana).
         - Alinhamento de mira (|delta_theta| <= 8.5°).
         - Recompensa disparo quando alinhado (+5.0) e rotação em direção à ameaça (+2.0).
         - Penalidade severa por tiro cego no escuro (-5.0) e por inércia (-2.0),
           eliminando o exploit de 'Beyblade / Aspersor' (girar e atirar sem mirar).
      3. Bônus por Eliminação de Inimigo:
         - +50.0 em término por vitória ou +25.0 por monstro abatido.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.is_visual = getattr(env, "is_visual", True)
        self.prev_yp: Optional[float] = None
        self.prev_angle: Optional[float] = None
        self._button_names: Optional[List[str]] = None

    def _get_engine(self) -> Any:
        return getattr(self.unwrapped, "_engine_instance", None)

    def _resolve_buttons(self) -> None:
        if self._button_names is not None:
            return
        engine = self._get_engine()
        if engine is not None:
            try:
                self._button_names = [b.name for b in engine.get_available_buttons()]
            except Exception:
                self._button_names = []
        else:
            self._button_names = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.prev_yp = None
        self.prev_angle = None
        self._resolve_buttons()

        engine = self._get_engine()
        if engine is not None:
            try:
                state = engine.get_state()
                if state and getattr(state, "objects", None):
                    for o in state.objects:
                        if o.name == "DoomPlayer" or getattr(o, "category", "") == "Self":
                            self.prev_yp = float(o.position_y)
                            self.prev_angle = float(o.angle)
                            break
            except Exception:
                pass
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        obs, r, term, trunc, info = self.env.step(action)
        shaped_r = float(r)

        # 1. Normaliza custo de tempo para -1.0 em vez das penalidades brutas do Doom
        if shaped_r < -7.0:
            shaped_r = -1.0
        elif shaped_r < 0:
            shaped_r = -1.0

        self._resolve_buttons()
        buttons = self._button_names or []

        # Mapeamento dinâmico de ações por nome do botão no cenário
        attack_idx = buttons.index("ATTACK") if "ATTACK" in buttons else 2
        turn_left_idx = buttons.index("TURN_LEFT") if "TURN_LEFT" in buttons else None
        turn_right_idx = buttons.index("TURN_RIGHT") if "TURN_RIGHT" in buttons else None
        move_left_idx = buttons.index("MOVE_LEFT") if "MOVE_LEFT" in buttons else 0
        move_right_idx = buttons.index("MOVE_RIGHT") if "MOVE_RIGHT" in buttons else 1

        is_rotational = turn_left_idx is not None and "MOVE_LEFT" not in buttons
        is_strafe = "MOVE_LEFT" in buttons or "MOVE_RIGHT" in buttons

        act_idx = int(action) if np.isscalar(action) else -1

        # 2. Rastreamento inteligente via coordenadas do motor
        engine = self._get_engine()
        if engine is not None:
            try:
                state = engine.get_state()
                if state and getattr(state, "objects", None):
                    player = None
                    monsters = []
                    for o in state.objects:
                        cat = getattr(o, "category", "")
                        if o.name == "DoomPlayer" or cat == "Self":
                            player = o
                        elif cat == "Monster" or (o.name not in NON_MONSTER_NAMES and cat != "Gibs"):
                            monsters.append(o)

                    if player is not None and monsters:
                        px = float(player.position_x)
                        py = float(player.position_y)
                        p_angle = float(player.angle)

                        if is_rotational:
                            # --- CENÁRIO ROTACIONAL 360° ---
                            # Radar angular contra todas as ameaças ativas
                            threat_data = []
                            for m in monsters:
                                mx = float(m.position_x)
                                my = float(m.position_y)
                                dx = mx - px
                                dy = my - py
                                dist = math.hypot(dx, dy)
                                target_angle = math.degrees(math.atan2(dy, dx))
                                delta_theta = (target_angle - p_angle + 180.0) % 360.0 - 180.0
                                threat_data.append((dist, delta_theta, m))

                            # Verifica se ALGUM monstro está alinhado com a retícula (mira frontal)
                            aligned_threats = [t for t in threat_data if abs(t[1]) <= 8.5]
                            has_target_in_crosshair = len(aligned_threats) > 0

                            # Rastreamento de rotação real do jogador
                            player_rotated = False
                            if self.prev_angle is not None:
                                delta_rot = abs((p_angle - self.prev_angle + 180.0) % 360.0 - 180.0)
                                player_rotated = delta_rot > 0.4
                            self.prev_angle = p_angle

                            if has_target_in_crosshair:
                                # Inimigo na mira: incentiva tiro certeiro
                                if act_idx == attack_idx:
                                    shaped_r += 5.0
                            else:
                                # Mira no vazio:
                                if act_idx == attack_idx and r <= 0:
                                    # Penalidade pesada por atirar no escuro (quebra exploit do aspersor)
                                    shaped_r -= 5.0

                                # Prioriza ameaça mais imediata (combina proximidade e menor rotação necessária)
                                def threat_cost(t):
                                    d, delta, _ = t
                                    if d < 250.0:
                                        return d * 0.5 + abs(delta)
                                    return d + 2.5 * abs(delta)

                                threat_data.sort(key=threat_cost)
                                closest_dist, closest_delta, _ = threat_data[0]

                                desired_turn = turn_left_idx if closest_delta > 0 else turn_right_idx
                                opposite_turn = turn_right_idx if closest_delta > 0 else turn_left_idx

                                if act_idx == desired_turn and player_rotated:
                                    shaped_r += 2.0  # Rastreando ativamente a ameaça
                                elif act_idx == opposite_turn:
                                    shaped_r -= 1.0  # Girando para o lado oposto ao monstro
                                elif not player_rotated and act_idx != attack_idx:
                                    shaped_r -= 2.0  # Inércia: parado enquanto monstros avançam

                        elif is_strafe:
                            # --- CENÁRIO LATERAL / STRAFE (ex: basic.cfg) ---
                            # Monstro prioritário (mais próximo)
                            closest_m = min(monsters, key=lambda m: math.hypot(m.position_x - px, m.position_y - py))
                            ym = float(closest_m.position_y)
                            dy = py - ym
                            aligned = abs(dy) <= 35.0

                            # Detecção de deslocamento físico real no mapa
                            player_moved = False
                            if self.prev_yp is not None:
                                player_moved = abs(py - self.prev_yp) > 0.4
                            self.prev_yp = py

                            # Penalidade por tentar andar mas ficar travado na parede
                            if act_idx in (move_left_idx, move_right_idx) and not player_moved:
                                shaped_r -= 3.0

                            if aligned:
                                # Monstro alinhado: incentiva disparo certeiro
                                if act_idx == attack_idx:
                                    shaped_r += 5.0
                            else:
                                # Monstro fora da mira:
                                target_action = move_left_idx if ym > py else move_right_idx
                                if act_idx == target_action and player_moved:
                                    shaped_r += 4.0  # Bônus por perseguir ativamente
                                elif act_idx == attack_idx and r <= 0:
                                    shaped_r -= 5.0  # Penalidade por tiro cego no vazio
                                elif not player_moved:
                                    shaped_r -= 3.0  # Penalidade direta por inércia

            except Exception:
                pass

        # 3. Bônus de eliminação de inimigo
        if term and r > 0:
            shaped_r += 50.0
        elif r > 0:
            shaped_r += 25.0

        # Sincroniza prev_reward na observação dict do wrapper, se aplicável
        if isinstance(obs, dict) and "prev_reward" in obs:
            obs["prev_reward"] = shaped_r

        return obs, shaped_r, term, trunc, info
