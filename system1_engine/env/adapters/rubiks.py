"""Módulo de Simulação e Ambientes de Aprendizado por Reforço para Cubo Mágico (Rubik's Cube 3x3).

Implementa:
1. `RubiksCubeCore`: Modelo matemático 3D com permutações pré-calculadas ultrarrápidas (< 1 µs/passo).
2. `RubiksCubeEnv`: Ambiente Gymnasium para movimentos atômicos (12 giros elementares).
3. `RubiksCubeMacroEnv`: Ambiente Gymnasium para macro-ações e algoritmos pré-definidos (Hierarchical RL).
4. Renderização visual 2D estilizada (Net/Desdobramento) compatível com o streaming MJPEG do HUD.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw


# =====================================================================
# MATEMÁTICA E PERMUTAÇÕES 3D DO CUBO MÁGICO
# =====================================================================

ROT_F = np.array([[ 0,  1,  0], [-1,  0,  0], [ 0,  0,  1]], dtype=np.int32)
ROT_B = np.array([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0,  1]], dtype=np.int32)
ROT_R = np.array([[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]], dtype=np.int32)
ROT_L = np.array([[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]], dtype=np.int32)
ROT_U = np.array([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]], dtype=np.int32)
ROT_D = np.array([[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]], dtype=np.int32)

BASE_MOVES = {
    "F": (ROT_F, lambda p: p[2] == 1),
    "F_prime": (ROT_F.T, lambda p: p[2] == 1),
    "B": (ROT_B, lambda p: p[2] == -1),
    "B_prime": (ROT_B.T, lambda p: p[2] == -1),
    "R": (ROT_R, lambda p: p[0] == 1),
    "R_prime": (ROT_R.T, lambda p: p[0] == 1),
    "L": (ROT_L, lambda p: p[0] == -1),
    "L_prime": (ROT_L.T, lambda p: p[0] == -1),
    "U": (ROT_U, lambda p: p[1] == 1),
    "U_prime": (ROT_U.T, lambda p: p[1] == 1),
    "D": (ROT_D, lambda p: p[1] == -1),
    "D_prime": (ROT_D.T, lambda p: p[1] == -1),
    # Rotações do cubo inteiro
    "Y": (ROT_U, lambda p: True),
    "Y_prime": (ROT_U.T, lambda p: True),
}


def _build_initial_stickers() -> List[Dict[str, Any]]:
    stickers = []
    # Face 0: U (y=1, norm=(0,1,0))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([c - 1, 1, 1 - r]), "norm": np.array([0, 1, 0]), "color": 0})
    # Face 1: D (y=-1, norm=(0,-1,0))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([c - 1, -1, r - 1]), "norm": np.array([0, -1, 0]), "color": 1})
    # Face 2: F (z=1, norm=(0,0,1))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([c - 1, 1 - r, 1]), "norm": np.array([0, 0, 1]), "color": 2})
    # Face 3: B (z=-1, norm=(0,0,-1))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([1 - c, 1 - r, -1]), "norm": np.array([0, 0, -1]), "color": 3})
    # Face 4: R (x=1, norm=(1,0,0))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([1, 1 - r, 1 - c]), "norm": np.array([1, 0, 0]), "color": 4})
    # Face 5: L (x=-1, norm=(-1,0,0))
    for r in range(3):
        for c in range(3):
            stickers.append({"pos": np.array([-1, 1 - r, c - 1]), "norm": np.array([-1, 0, 0]), "color": 5})
    return stickers


def _get_slot_index(pos: np.ndarray, norm: np.ndarray) -> int:
    x, y, z = pos[0], pos[1], pos[2]
    nx, ny, nz = norm[0], norm[1], norm[2]
    if ny == 1:
        face, r, c = 0, 1 - z, x + 1
    elif ny == -1:
        face, r, c = 1, z + 1, x + 1
    elif nz == 1:
        face, r, c = 2, 1 - y, x + 1
    elif nz == -1:
        face, r, c = 3, 1 - y, 1 - x
    elif nx == 1:
        face, r, c = 4, 1 - y, 1 - z
    elif nx == -1:
        face, r, c = 5, 1 - y, z + 1
    else:
        raise ValueError(f"Normal inválida: {norm}")
    return int(face * 9 + (r * 3 + c))


def _precompute_permutations() -> Dict[str, np.ndarray]:
    perms = {}
    for m, (rot_mat, cond) in BASE_MOVES.items():
        st = _build_initial_stickers()
        for i, s in enumerate(st):
            s["color"] = _get_slot_index(s["pos"], s["norm"])
        new_st = []
        for s in st:
            pos, norm, col = s["pos"], s["norm"], s["color"]
            if cond(pos):
                new_st.append({"pos": rot_mat @ pos, "norm": rot_mat @ norm, "color": col})
            else:
                new_st.append({"pos": pos.copy(), "norm": norm.copy(), "color": col})
        perm = np.zeros(54, dtype=np.int32)
        for s in new_st:
            dest_slot = _get_slot_index(s["pos"], s["norm"])
            src_slot = s["color"]
            perm[dest_slot] = src_slot
        perms[m] = perm
    return perms


PERMUTATIONS = _precompute_permutations()


# =====================================================================
# CORE DO CUBO MÁGICO
# =====================================================================

class RubiksCubeCore:
    """Simulador vetorial de alta velocidade para o Cubo Mágico 3x3x3."""

    # 12 Movimentos Atômicos
    ATOMIC_MOVES = [
        "U", "U_prime", "D", "D_prime",
        "F", "F_prime", "B", "B_prime",
        "R", "R_prime", "L", "L_prime",
    ]

    # 12 Macro-Ações Pré-Definidas (Hierarchical / Speedcubing)
    MACRO_ACTIONS = {
        "SEXY_MOVE_R": ["R", "U", "R_prime", "U_prime"],
        "SEXY_MOVE_L": ["L_prime", "U_prime", "L", "U"],
        "SUNE": ["R", "U", "R_prime", "U", "R", "U", "U", "R_prime"],
        "ANTI_SUNE": ["R", "U", "U", "R_prime", "U_prime", "R", "U_prime", "R_prime"],
        "T_PERM": [
            "R", "U", "R_prime", "U_prime", "R_prime", "F",
            "R", "R", "U_prime", "R_prime", "U_prime", "R", "U", "R_prime", "F_prime"
        ],
        "INSERT_EDGE_R": ["U", "R", "U_prime", "R_prime", "U_prime", "F_prime", "U", "F"],
        "INSERT_EDGE_L": ["U_prime", "L_prime", "U", "L", "U", "F", "U_prime", "F_prime"],
        "YELLOW_CROSS": ["F", "R", "U", "R_prime", "U_prime", "F_prime"],
        "ROTATE_Y": ["Y"],
        "ROTATE_Y_PRIME": ["Y_prime"],
        "U_TURN": ["U"],
        "U_PRIME_TURN": ["U_prime"],
    }
    MACRO_NAMES = list(MACRO_ACTIONS.keys())

    # Paleta de cores oficial (Dark UI / Cyber Style)
    PALETTE = {
        0: (245, 245, 250),  # U: Branco
        1: (250, 204, 21),   # D: Amarelo
        2: (34, 197, 94),    # F: Verde
        3: (59, 130, 246),   # B: Azul
        4: (239, 68, 68),    # R: Vermelho
        5: (249, 115, 22),   # L: Laranja
    }

    # Coordenadas do desdobramento 2D (Net) no viewport (480x360)
    # Layout em Cruz:
    #       [ U ]
    # [ L ] [ F ] [ R ] [ B ]
    #       [ D ]
    FACE_LAYOUT = {
        0: (164, 46),   # U
        5: (88, 122),   # L
        2: (164, 122),  # F
        4: (240, 122),  # R
        3: (316, 122),  # B
        1: (164, 198),  # D
    }

    def __init__(self) -> None:
        # Estado resolvido: 6 faces x 9 facetas com a cor da face (0..5)
        self.state = np.repeat(np.arange(6, dtype=np.int32), 9)

    def reset(self) -> None:
        """Restaura o cubo para o estado perfeitamente resolvido."""
        self.state = np.repeat(np.arange(6, dtype=np.int32), 9)

    def apply_atomic(self, move_name: str) -> None:
        """Executa um movimento atômico via permutação de arrays."""
        if move_name in PERMUTATIONS:
            self.state = self.state[PERMUTATIONS[move_name]]

    def apply_macro(self, macro_name: str) -> None:
        """Executa uma macro-ação (sequência atômica)."""
        if macro_name in self.MACRO_ACTIONS:
            for m in self.MACRO_ACTIONS[macro_name]:
                self.apply_atomic(m)

    def scramble(self, depth: int = 4, use_macros: bool = False, rng: Optional[np.random.Generator] = None) -> List[str]:
        """Embaralha o cubo a partir do estado resolvido com uma profundidade definida."""
        self.reset()
        if rng is None:
            rng = np.random.default_rng()
        applied = []
        if use_macros:
            choices = self.MACRO_NAMES
            for _ in range(depth):
                act = str(rng.choice(choices))
                self.apply_macro(act)
                applied.append(act)
        else:
            choices = self.ATOMIC_MOVES
            for _ in range(depth):
                act = str(rng.choice(choices))
                self.apply_atomic(act)
                applied.append(act)
        return applied

    def get_aligned_count(self) -> int:
        """Retorna o número de facetas que combinam com a cor do centro da sua respectiva face (6 a 54)."""
        count = 0
        for f in range(6):
            center_color = self.state[f * 9 + 4]
            count += int(np.sum(self.state[f * 9 : f * 9 + 9] == center_color))
        return count

    def get_score(self) -> float:
        """Retorna a métrica de resolução normalizada em [0.0, 1.0]."""
        aligned = self.get_aligned_count()
        # 6 centros sempre coincidem, sobram 48 facetas
        return float(max(0.0, min(1.0, (aligned - 6) / 48.0)))

    def is_solved(self) -> bool:
        """Verifica se o cubo está 100% resolvido."""
        return self.get_aligned_count() == 54

    def get_one_hot(self) -> np.ndarray:
        """Retorna a observação vetorial one-hot: shape (324,) em [0.0, 1.0]."""
        one_hot = np.zeros((54, 6), dtype=np.float32)
        one_hot[np.arange(54), self.state] = 1.0
        return one_hot.flatten()

    def render_net(
        self,
        width: int = 480,
        height: int = 320,
        last_action: str = "INITIAL",
        steps: int = 0,
    ) -> np.ndarray:
        """Gera imagem RGB (H, W, 3) com estética cibernética dark mode para o HUD."""
        img = Image.new("RGB", (width, height), color=(11, 15, 25))
        draw = ImageDraw.Draw(img)

        # Moldura estética
        draw.rectangle([6, 6, width - 7, height - 7], outline=(31, 41, 55), width=2)
        draw.rectangle([10, 10, width - 11, height - 11], outline=(17, 24, 39), width=1)

        # Cabeçalho
        aligned = self.get_aligned_count()
        score = self.get_score() * 100.0
        draw.text((20, 16), "⚡ SYSTEM 1 HUD | RUBIK'S CUBE", fill=(56, 189, 248))
        status_txt = f"Passos: {steps} | Alinhamento: {aligned}/54 ({score:.1f}%)"
        draw.text((20, 30), status_txt, fill=(148, 163, 184))

        # Desenho das 6 faces no formato de Cruz
        for f_idx, (fx, fy) in self.FACE_LAYOUT.items():
            for r in range(3):
                for c in range(3):
                    sticker_idx = f_idx * 9 + (r * 3 + c)
                    col_id = int(self.state[sticker_idx])
                    color = self.PALETTE.get(col_id, (255, 255, 255))
                    x1 = fx + c * 24
                    y1 = fy + r * 24
                    x2 = x1 + 22
                    y2 = y1 + 22
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline=(15, 20, 30), width=1)

        # Rodapé com ação recente
        action_color = (74, 222, 128) if self.is_solved() else (226, 232, 240)
        footer_label = "STATUS: RESOLVIDO 🏆" if self.is_solved() else f"AÇÃO ATIVA: {last_action}"
        draw.text((20, height - 28), footer_label, fill=action_color)

        return np.array(img, dtype=np.uint8)


# =====================================================================
# AMBIENTES GYMNASIUM
# =====================================================================

class RubiksCubeEnv(gym.Env):
    """Ambiente Gymnasium para Cubo Mágico com Ações Atômicas (12 giros simples)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        scramble_depth: int = 3,
        max_steps: int = 40,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.scramble_depth = scramble_depth
        self.max_steps = max_steps
        self.core = RubiksCubeCore()

        # 12 Ações Atômicas: U, U', D, D', F, F', B, B', R, R', L, L'
        self.action_space = gym.spaces.Discrete(len(self.core.ATOMIC_MOVES))
        # 54 facetas x 6 cores = 324 floats em [0, 1]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(324,), dtype=np.float32)

        self._steps = 0
        self._last_action_name = "NONE"
        self._prev_score = 0.0
        self.rng = np.random.default_rng()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        depth = options.get("scramble_depth", self.scramble_depth) if options else self.scramble_depth
        self.core.scramble(depth=depth, use_macros=False, rng=self.rng)
        self._steps = 0
        self._last_action_name = "RESET"
        self._prev_score = self.core.get_score()

        obs = self.core.get_one_hot()
        info = {
            "aligned_stickers": self.core.get_aligned_count(),
            "score": self._prev_score,
            "is_solved": self.core.is_solved(),
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._steps += 1
        act_idx = int(action)
        action_name = self.core.ATOMIC_MOVES[act_idx]
        self._last_action_name = action_name

        self.core.apply_atomic(action_name)
        cur_score = self.core.get_score()
        delta_score = cur_score - self._prev_score
        self._prev_score = cur_score

        # Recompensa Densa:
        # 1. Delta de progresso (avanço em direção ao alinhamento)
        # 2. Penalidade de tempo (-0.02) para incentivar trajetórias curtas
        # 3. Grande bônus ao resolver (+10.0)
        reward = (delta_score * 5.0) - 0.02
        is_solved = self.core.is_solved()
        if is_solved:
            reward += 10.0

        terminated = is_solved
        truncated = self._steps >= self.max_steps

        obs = self.core.get_one_hot()
        info = {
            "aligned_stickers": self.core.get_aligned_count(),
            "score": cur_score,
            "is_solved": is_solved,
            "action_name": action_name,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self.core.render_net(
                last_action=self._last_action_name,
                steps=self._steps,
            )
        return None


class RubiksCubeMacroEnv(gym.Env):
    """Ambiente Gymnasium para Cubo Mágico com Macro-Ações e Algoritmos Pré-Definidos.
    
    Ações (Discrete 12):
    0: SEXY_MOVE_R    (R U R' U')
    1: SEXY_MOVE_L    (L' U' L U)
    2: SUNE           (R U R' U R U2 R')
    3: ANTI_SUNE      (R U2 R' U' R U' R')
    4: T_PERM         (R U R' U' R' F R2 U' R' U' R U R' F')
    5: INSERT_EDGE_R  (U R U' R' U' F' U F)
    6: INSERT_EDGE_L  (U' L' U L U F U' F')
    7: YELLOW_CROSS   (F R U R' U' F')
    8: ROTATE_Y       (Giro do cubo todo no eixo Y)
    9: ROTATE_Y_PRIME (Giro reverso do cubo no eixo Y)
    10: U_TURN        (U - Giro superior)
    11: U_PRIME_TURN  (U' - Giro superior inverso)
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        scramble_depth: int = 2,
        max_steps: int = 25,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.scramble_depth = scramble_depth
        self.max_steps = max_steps
        self.core = RubiksCubeCore()

        # 12 Macro-Ações
        self.action_space = gym.spaces.Discrete(len(self.core.MACRO_NAMES))
        # 54 facetas x 6 cores = 324 floats em [0, 1]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(324,), dtype=np.float32)

        self._steps = 0
        self._last_action_name = "NONE"
        self._prev_score = 0.0
        self.rng = np.random.default_rng()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        depth = options.get("scramble_depth", self.scramble_depth) if options else self.scramble_depth
        self.core.scramble(depth=depth, use_macros=True, rng=self.rng)
        self._steps = 0
        self._last_action_name = "RESET"
        self._prev_score = self.core.get_score()

        obs = self.core.get_one_hot()
        info = {
            "aligned_stickers": self.core.get_aligned_count(),
            "score": self._prev_score,
            "is_solved": self.core.is_solved(),
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._steps += 1
        act_idx = int(action)
        macro_name = self.core.MACRO_NAMES[act_idx]
        self._last_action_name = macro_name

        self.core.apply_macro(macro_name)
        cur_score = self.core.get_score()
        delta_score = cur_score - self._prev_score
        self._prev_score = cur_score

        # Recompensa com sub-metas e bônus de resolução
        reward = (delta_score * 8.0) - 0.02
        is_solved = self.core.is_solved()
        if is_solved:
            reward += 15.0

        terminated = is_solved
        truncated = self._steps >= self.max_steps

        obs = self.core.get_one_hot()
        info = {
            "aligned_stickers": self.core.get_aligned_count(),
            "score": cur_score,
            "is_solved": is_solved,
            "action_name": macro_name,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self.core.render_net(
                last_action=self._last_action_name,
                steps=self._steps,
            )
        return None


# =====================================================================
# REGISTRO NO GYMNASIUM
# =====================================================================

def register_rubiks_environments() -> None:
    """Registra os ambientes no Gymnasium se ainda não estiverem presentes."""
    try:
        if "RubiksCube-v0" not in gym.envs.registry:
            gym.register(
                id="RubiksCube-v0",
                entry_point="system1_engine.env.adapters.rubiks:RubiksCubeEnv",
                max_episode_steps=50,
            )
        if "RubiksCubeMacro-v0" not in gym.envs.registry:
            gym.register(
                id="RubiksCubeMacro-v0",
                entry_point="system1_engine.env.adapters.rubiks:RubiksCubeMacroEnv",
                max_episode_steps=30,
            )
    except Exception:
        pass


# Executa o registro imediatamente ao importar o módulo
register_rubiks_environments()
