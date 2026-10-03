from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import itertools
import json
import socket
import time
import gymnasium as gym
import numpy as np
from PIL import Image

from system1_engine.env.adapters.base import BaseGameAdapter


class NativeEngineEnv(BaseGameAdapter):
    """Nível 3: Adaptador de Motores Nativos e Emuladores (Lock-Step & Headless).

    Suporta conexões diretas com simuladores de alto desempenho:
      - ViZDoom, Gym-Retro, Godot RL, Unity ML-Agents ou Sockets IPC (Unix/TCP).
      - Regime estritamente síncrono (Lock-Step): o motor só avança quando step() é chamado.
      - Execução Headless em alta velocidade (capaz de 10.000+ FPS).
      - Formato padronizado de observação (Visual [C, 84, 84] ou Vetorial [obs_dim]).
      - Espaço de ações compostas ampliado (Produto Cartesiano de eixos ortogonais), permitindo
        manobras de combate complexas como 'TURN_LEFT + MOVE_RIGHT + ATTACK' (circle-strafe atirando).
    """

    def __init__(
        self,
        engine_type: str = "native_sim",  # "vizdoom", "retro", "godot", "ipc", "native_sim"
        scenario_path: Optional[str] = None,
        args: Optional[Dict[str, Any]] = None,
        is_visual: bool = True,
        channels: int = 1,
        obs_dim: int = 4,
        action_dim: int = 2,
        is_action_discrete: bool = True,
        socket_address: Optional[Union[str, Tuple[str, int]]] = None,
        custom_step_fn: Optional[Callable[[Any], Tuple[Any, float, bool, bool, Dict[str, Any]]]] = None,
        custom_reset_fn: Optional[Callable[[], Tuple[Any, Dict[str, Any]]]] = None,
    ) -> None:
        self.engine_type = engine_type.lower()
        self.scenario_path = scenario_path
        self.args = args or {}
        self.channels = channels
        self.obs_dim = obs_dim
        self.socket_address = socket_address
        self.custom_step_fn = custom_step_fn
        self.custom_reset_fn = custom_reset_fn

        # Configuração de Espaço de Observação
        if is_visual:
            obs_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(channels, 84, 84),
                dtype=np.float32,
            )
        else:
            obs_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )

        # Configuração de Espaço de Ações
        if is_action_discrete:
            act_space = gym.spaces.Discrete(action_dim)
        else:
            act_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
            )

        super().__init__(
            observation_space=obs_space,
            action_space=act_space,
            is_visual=is_visual,
            target_fps=None,  # Lock-step não tem limite de FPS artificial
        )

        # Recursos de motor / IPC
        self._engine_instance = None
        self._ipc_socket: Optional[socket.socket] = None
        self.action_table: Optional[List[List[int]]] = None
        self.action_descriptions: Optional[List[str]] = None
        self.button_names: Optional[List[str]] = None
        self._game_var_names: List[str] = []
        self._prev_hp: float = 100.0
        self._prev_dmg: float = 0.0
        self._prev_kill: float = 0.0
        self.is_attacking: bool = False
        self._init_engine()

        # Estado do simulador nativo integrado (fallback de ultra-alta velocidade)
        self._sim_step: int = 0
        self._sim_state: np.ndarray = np.zeros(obs_dim, dtype=np.float32)

    @staticmethod
    def _build_composite_actions(
        button_names: List[str],
    ) -> Tuple[List[List[int]], List[str]]:
        """Gera sistematicamente o espaço de ações combinadas (Produto Cartesiano de eixos ortogonais).

        Eixos modelados:
          1. Longitudinal (Surge): [None, MOVE_FORWARD, MOVE_BACKWARD]
          2. Lateral (Strafe): [None, MOVE_LEFT, MOVE_RIGHT]
          3. Yaw (Rotação): [None, TURN_LEFT, TURN_RIGHT]
          4. Pitch (Olhar vertical): [None, LOOK_UP, LOOK_DOWN]
          5. Seleção de Armas: [None, SELECT_WEAPON1, SELECT_WEAPON2, ...]
          6. Ações independentes (Toggles): ATTACK, USE, SPEED, CROUCH, JUMP, etc.

        Permite derivações ricas e completas de combate como:
          - 'MOVE_RIGHT + TURN_LEFT + ATTACK' (circle-strafe atirando)
          - 'MOVE_FORWARD + MOVE_RIGHT + TURN_LEFT + ATTACK'
          - 'MOVE_BACKWARD + ATTACK'
          - 'MOVE_FORWARD + ATTACK'
        eliminando simultaneamente contradições físicas (como avançar e retroceder no mesmo instante).
        """
        btn_set = set(button_names)
        axes: List[List[Optional[str]]] = []

        # 1. Eixo Longitudinal
        fwd_bwd = [b for b in ["MOVE_FORWARD", "MOVE_BACKWARD"] if b in btn_set]
        if fwd_bwd:
            axes.append([None] + fwd_bwd)

        # 2. Eixo Lateral (Strafe)
        strafe = [b for b in ["MOVE_LEFT", "MOVE_RIGHT"] if b in btn_set]
        if strafe:
            axes.append([None] + strafe)

        # 3. Eixo Yaw (Rotação)
        yaw = [b for b in ["TURN_LEFT", "TURN_RIGHT"] if b in btn_set]
        if yaw:
            axes.append([None] + yaw)

        # 4. Eixo Pitch (Vertical)
        pitch = [b for b in ["LOOK_UP", "LOOK_DOWN"] if b in btn_set]
        if pitch:
            axes.append([None] + pitch)

        # 5. Seleção de Armas (mutuamente exclusivas)
        weapons = [b for b in button_names if b.startswith("SELECT_WEAPON") or b.startswith("WEAPON")]
        if weapons:
            axes.append([None] + weapons)

        # 6. Ações independentes (Toggles como ATTACK, USE, SPEED, JUMP, etc.)
        handled = set(fwd_bwd + strafe + yaw + pitch + weapons)
        for b in button_names:
            if b not in handled:
                axes.append([None, b])

        # Produto cartesiano de todos os eixos
        combos = list(itertools.product(*axes))

        # Teto de segurança para cenários exóticos com muitos botões
        if len(combos) > 128:
            combos = combos[:128]

        action_table: List[List[int]] = []
        action_descriptions: List[str] = []
        for combo in combos:
            pressed = [b for b in combo if b is not None]
            vec = [1 if b in pressed else 0 for b in button_names]
            action_table.append(vec)
            desc = "+".join(pressed) if pressed else "NO_OP"
            action_descriptions.append(desc)

        return action_table, action_descriptions

    def _read_game_variables(self, state: Any) -> Tuple[float, float, float]:
        """Extrai valores numéricos de telemetria universal do estado do motor."""
        if state is None or not hasattr(state, "game_variables") or state.game_variables is None:
            return 100.0, 0.0, 0.0

        if not self._game_var_names and self._engine_instance is not None and hasattr(self._engine_instance, "get_available_game_variables"):
            try:
                self._game_var_names = [v.name for v in self._engine_instance.get_available_game_variables()]
            except Exception:
                self._game_var_names = []

        var_map: Dict[str, float] = {}
        for i, val in enumerate(state.game_variables):
            if i < len(self._game_var_names):
                var_map[self._game_var_names[i]] = float(val)

        hp = float(var_map.get("HEALTH", 100.0))
        dmg = float(var_map.get("DAMAGECOUNT", 0.0))
        kill = float(var_map.get("KILLCOUNT", 0.0))
        return hp, dmg, kill

    def _init_engine(self) -> None:
        """Inicializa bindings específicos ou conexões IPC."""
        if self.engine_type == "ipc" and self.socket_address is not None:
            try:
                if isinstance(self.socket_address, str):
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.socket_address)
                    self._ipc_socket = s
                elif isinstance(self.socket_address, tuple):
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.connect(self.socket_address)
                    self._ipc_socket = s
            except Exception:
                self._ipc_socket = None

        elif self.engine_type == "vizdoom":
            # Tenta carregar vizdoom se instalado
            try:
                import os
                import vizdoom as vzd
                game = vzd.DoomGame()
                scen = self.scenario_path or "basic.cfg"
                if not os.path.isabs(scen) and not os.path.exists(scen):
                    scen = os.path.join(vzd.scenarios_path, scen)
                game.load_config(scen)
                game.set_window_visible(not self.args.get("headless", True))
                # Sempre use RGB24 para que o buffer de tela seja colorido para o HUD
                game.set_screen_format(vzd.ScreenFormat.RGB24)
                game.set_objects_info_enabled(True)

                # Injeta variáveis universais de objetivo de combate se disponíveis
                for var in [
                    vzd.GameVariable.HEALTH,
                    vzd.GameVariable.DAMAGECOUNT,
                    vzd.GameVariable.KILLCOUNT,
                    vzd.GameVariable.HITCOUNT,
                    vzd.GameVariable.AMMO2,
                ]:
                    try:
                        game.add_available_game_variable(var)
                    except Exception:
                        pass

                game.init()
                self._engine_instance = game
                self.button_names = [b.name for b in game.get_available_buttons()]
                self.available_buttons_count = len(self.button_names)
                try:
                    self._game_var_names = [v.name for v in game.get_available_game_variables()]
                except Exception:
                    self._game_var_names = []

                use_combined = self.args.get("combined_actions", True)
                if use_combined and self.available_buttons_count > 1:
                    self.action_table, self.action_descriptions = self._build_composite_actions(self.button_names)
                    self.action_space = gym.spaces.Discrete(len(self.action_table))
                else:
                    self.action_table = [
                        [1 if i == j else 0 for j in range(self.available_buttons_count)]
                        for i in range(self.available_buttons_count)
                    ]
                    self.action_descriptions = self.button_names
                    self.action_space = gym.spaces.Discrete(max(1, self.available_buttons_count))
            except ImportError:
                # Usa motor nativo simulado se pacote não estiver compilado no host
                self._engine_instance = None

        elif self.engine_type == "retro":
            try:
                import retro
                game_name = self.args.get("game", "Airstriker-Genesis")
                self._engine_instance = retro.make(game=game_name)
            except ImportError:
                self._engine_instance = None

    def _format_visual_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Padroniza frames nativos para (C, 84, 84) normalizado [0.0, 1.0]."""
        # Suporta entradas (C, H, W) e (H, W, C) e (H, W)
        frame = raw_frame
        if frame.ndim == 3:
            if frame.shape[0] in (1, 3) and frame.shape[2] not in (1, 3):
                # Formato CHW (ex.: ViZDoom buffer) -> Transpõe para HWC para PIL
                if frame.shape[0] == 3:
                    frame = np.transpose(frame, (1, 2, 0))
                else:
                    frame = frame.squeeze(0)
            elif frame.shape[2] == 1:
                frame = frame.squeeze(-1)

        pil_img = Image.fromarray(frame)
        if self.channels == 1:
            pil_img = pil_img.convert("L").resize((84, 84), Image.Resampling.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            return np.expand_dims(arr, axis=0)
        else:
            pil_img = pil_img.convert("RGB").resize((84, 84), Image.Resampling.BILINEAR)
            arr = np.asarray(pil_img, dtype=np.float32) / 255.0
            return np.transpose(arr, (2, 0, 1))

    def _reset_impl(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> Tuple[Any, Dict[str, Any]]:
        self._sim_step = 0
        self.is_attacking = False
        self._prev_hp = 100.0
        self._prev_dmg = 0.0
        self._prev_kill = 0.0

        if self.custom_reset_fn is not None:
            return self.custom_reset_fn()

        if self._engine_instance is not None:
            if hasattr(self._engine_instance, "reset"):
                res = self._engine_instance.reset()
                raw_obs = res[0] if isinstance(res, tuple) else res
                self._last_render_frame = raw_obs
                obs = self._format_visual_frame(raw_obs) if self.is_visual else raw_obs
                return obs, {"engine": self.engine_type}
            elif hasattr(self._engine_instance, "new_episode"):
                self._engine_instance.new_episode()
                state = self._engine_instance.get_state()
                raw_obs = state.screen_buffer if state else np.zeros((240, 320, 3), dtype=np.uint8)
                self._last_render_frame = raw_obs
                self._prev_hp, self._prev_dmg, self._prev_kill = self._read_game_variables(state)
                obs = self._format_visual_frame(raw_obs)
                return obs, {"engine": self.engine_type}

        # Simulação interna lock-step (Ultra-alta velocidade > 10.000 FPS)
        if self.is_visual:
            obs = np.zeros((self.channels, 84, 84), dtype=np.float32)
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            self._sim_state = obs.copy()

        return obs, {"engine": "native_fast_sim", "lock_step": True}

    def _step_impl(
        self, action: Any
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        self._sim_step += 1

        if self.custom_step_fn is not None:
            return self.custom_step_fn(action)

        if self._ipc_socket is not None:
            # Protocolo síncrono IPC: Envia ação -> Lê estado em lock-step
            try:
                payload = json.dumps({"action": action}).encode("utf-8") + b"\n"
                self._ipc_socket.sendall(payload)
                data = self._ipc_socket.recv(4096).decode("utf-8")
                response = json.loads(data)
                obs = np.array(response["obs"], dtype=np.float32)
                return obs, float(response["reward"]), bool(response["done"]), False, response.get("info", {})
            except Exception:
                pass

        if self._engine_instance is not None:
            if hasattr(self._engine_instance, "make_action"):
                # ViZDoom nativo com avanço por botões compostos e frame_skip
                if self.action_table is not None and isinstance(action, (int, np.integer)):
                    act_idx = int(action)
                    if 0 <= act_idx < len(self.action_table):
                        act_vector = self.action_table[act_idx]
                        act_name = self.action_descriptions[act_idx] if self.action_descriptions else str(act_idx)
                    else:
                        act_vector = [0] * getattr(self, "available_buttons_count", 3)
                        act_name = "UNKNOWN"
                elif isinstance(action, (list, np.ndarray)):
                    n_b = getattr(self, "available_buttons_count", len(action))
                    act_vector = [int(x) for x in action]
                    if len(act_vector) < n_b:
                        act_vector += [0] * (n_b - len(act_vector))
                    elif len(act_vector) > n_b:
                        act_vector = act_vector[:n_b]
                    act_name = "CUSTOM_VECTOR"
                else:
                    n_buttons = getattr(self, "available_buttons_count", 3)
                    act_vector = [0] * n_buttons
                    act_idx = int(action) if np.isscalar(action) else 0
                    if 0 <= act_idx < n_buttons:
                        act_vector[act_idx] = 1
                    act_name = str(act_idx)

                # Flag de disparo para telemetria e HUD
                self.is_attacking = False
                if "ATTACK" in getattr(self, "button_names", []):
                    atk_i = self.button_names.index("ATTACK")
                    if 0 <= atk_i < len(act_vector) and act_vector[atk_i] == 1:
                        self.is_attacking = True

                frame_skip = int(self.args.get("frame_skip", 4))
                raw_reward = float(self._engine_instance.make_action(act_vector, frame_skip))
                terminated = self._engine_instance.is_episode_finished()

                if not terminated:
                    state = self._engine_instance.get_state()
                    raw_obs = state.screen_buffer if state else np.zeros((240, 320, 3), dtype=np.uint8)
                    self._last_render_frame = raw_obs
                else:
                    raw_obs = getattr(self, "_last_render_frame", None)
                    if raw_obs is None:
                        raw_obs = np.zeros((240, 320, 3), dtype=np.uint8)

                obs = self._format_visual_frame(raw_obs)
                truncated = False

                # Telemetria universal de combate sem cheats/aimbot:
                reward = raw_reward
                if self.args.get("reward_shaping", True):
                    is_dead = getattr(self._engine_instance, "is_player_dead", lambda: False)()
                    if not terminated:
                        state = self._engine_instance.get_state()
                        cur_hp, cur_dmg, cur_kill = self._read_game_variables(state)
                    else:
                        cur_hp = 0.0 if is_dead else self._prev_hp
                        cur_dmg = self._prev_dmg
                        cur_kill = self._prev_kill

                    delta_hp = min(0.0, cur_hp - self._prev_hp)
                    delta_dmg = max(0.0, cur_dmg - self._prev_dmg)
                    delta_kill = max(0.0, cur_kill - self._prev_kill)

                    # Bônus por combate e preservação de integridade:
                    # +0.1 por ponto de dano causado aos inimigos
                    # +15.0 por inimigo abatido
                    # +0.1 por ponto de vida preservada (delta_hp <= 0)
                    reward += 0.1 * delta_dmg + 15.0 * delta_kill + 0.1 * delta_hp

                    # Bônus adicional ao completar a missão vivo (ex: alcançar armadura no deadly_corridor)
                    if terminated and not is_dead and raw_reward >= 0:
                        reward += 50.0

                    self._prev_hp = cur_hp
                    self._prev_dmg = cur_dmg
                    self._prev_kill = cur_kill

                info = {
                    "engine": "vizdoom",
                    "action_vector": act_vector,
                    "action_name": act_name,
                    "raw_reward": raw_reward,
                }
                return obs, reward, terminated, truncated, info

            elif hasattr(self._engine_instance, "step"):
                obs_raw, reward, term, trunc, info = self._engine_instance.step(action)
                self._last_render_frame = obs_raw
                obs = self._format_visual_frame(obs_raw) if self.is_visual else obs_raw
                return obs, float(reward), bool(term), bool(trunc), info

        # Motor simulado lock-step de alta performance
        if self.is_visual:
            # Renderização procedural ultra-leve de baixa alocação
            obs = np.full((self.channels, 84, 84), fill_value=0.1, dtype=np.float32)
            # Modifica um pixel com base na ação para simular feedback visual
            act_idx = int(action) if np.isscalar(action) else 0
            obs[0, 10 + (act_idx % 60), 10 + (act_idx % 60)] = 1.0
            reward = 1.0
            terminated = self._sim_step >= 500
            truncated = False
            return obs, reward, terminated, truncated, {"fps_mode": "lock_step_unlocked"}
        else:
            # Dinâmica vetorial simples
            self._sim_state += 0.05
            obs = self._sim_state.copy()
            reward = 1.0
            terminated = self._sim_step >= 500
            truncated = False
            return obs, reward, terminated, truncated, {"fps_mode": "lock_step_unlocked"}

    def _render_impl(self) -> Optional[np.ndarray]:
        if self._engine_instance is not None:
            if hasattr(self._engine_instance, "get_state"):
                state = self._engine_instance.get_state()
                if state is not None and getattr(state, "screen_buffer", None) is not None:
                    self._last_render_frame = state.screen_buffer
                return getattr(self, "_last_render_frame", None)
            elif hasattr(self._engine_instance, "render"):
                return self._engine_instance.render()
        if hasattr(self, "_last_render_frame"):
            return self._last_render_frame
        if self.is_visual:
            return np.zeros((84, 84, 3), dtype=np.uint8)
        return None

    def _close_impl(self) -> None:
        if self._ipc_socket is not None:
            try:
                self._ipc_socket.close()
            except Exception:
                pass
            self._ipc_socket = None

        if self._engine_instance is not None:
            if hasattr(self._engine_instance, "close"):
                self._engine_instance.close()
            self._engine_instance = None
