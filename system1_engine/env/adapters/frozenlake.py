"""Módulo de Simulação e Curriculum Learning para o FrozenLake-v1.

Implementa `FrozenLakeCurriculumWrapper`:
1. Reverse Curriculum (Goal Proximity):
   - Nível 1: Spawn a 1 passo do Goal (tiles 14 ou 10) -> Aprende a transição imediata para a vitória.
   - Nível 2: Spawn a 2 passos do Goal (tiles 13, 9, 6) -> Aprende a evitar buracos adjacentes (11 e 12).
   - Nível 3: Spawn a 3 passos do Goal (tiles 8, 4, 2) -> Aprende a contornar o buraco 5.
   - Nível 4: Spawn oficial no início (tile 0) -> Rota global completa do topo-esquerdo ao objetivo.
2. Reward Shaping e Janela de Promoção:
   - Monitoramento contínuo da taxa móvel de resolução (vitórias nos últimos 20 episódios).
   - Promoção automática de nível quando taxa >= target_success_rate (default: 85%).
   - Penalidade leve de tempo (-0.01) para incentivar trajetórias geodésicas ótimas.
   - Penalidade de queda em buraco (-0.50) e reforço de objetivo (+2.0).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
import numpy as np


class FrozenLakeCurriculumWrapper(gym.Wrapper):
    """Wrapper de Curriculum Learning para o FrozenLake-v1."""

    # Mapeamento de tiles de spawn por nível de distância até o Goal (15)
    LEVEL_SPAWNS: Dict[int, List[int]] = {
        1: [14, 10],       # 1 passo do Goal
        2: [13, 9, 6],     # 2 passos do Goal
        3: [8, 4, 2],      # 3 passos do Goal
        4: [0],            # Ponto de partida oficial (0,0)
    }

    def __init__(
        self,
        env: gym.Env,
        curriculum: bool = True,
        min_level: int = 1,
        max_level: int = 4,
        target_success_rate: float = 0.85,
        curriculum_window: int = 20,
        time_penalty: float = -0.01,
        hole_penalty: float = -0.50,
        goal_reward: float = 2.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        self.curriculum = curriculum
        self.min_level = min_level
        self.max_level = max_level
        self.target_success_rate = target_success_rate
        self.curriculum_window = curriculum_window
        self.time_penalty = time_penalty
        self.hole_penalty = hole_penalty
        self.goal_reward = goal_reward

        self.current_level = min_level if curriculum else max_level
        self.recent_successes: deque[float] = deque(maxlen=curriculum_window)
        self.curriculum_promotions: int = 0
        self.rng = np.random.default_rng(seed)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options and "level" in options:
            self.current_level = int(options["level"])
        elif self.curriculum:
            if len(self.recent_successes) >= self.curriculum_window:
                rate = float(np.mean(self.recent_successes))
                if rate >= self.target_success_rate and self.current_level < self.max_level:
                    old_lvl = self.current_level
                    self.current_level += 1
                    self.curriculum_promotions += 1
                    self.recent_successes.clear()
                    print(
                        f"\n🚀 [CURRICULUM FROZENLAKE] Taxa {rate*100:.1f}% >= {self.target_success_rate*100:.0f}%! "
                        f"Nível promovido: {old_lvl} -> {self.current_level}/{self.max_level}"
                    )

        if self.curriculum and self.current_level < self.max_level:
            spawns = self.LEVEL_SPAWNS.get(self.current_level, [0])
            spawn_s = int(self.rng.choice(spawns))
            self.unwrapped.s = spawn_s
            obs = spawn_s

        info["current_level"] = self.current_level
        info["curriculum_rate"] = (
            float(np.mean(self.recent_successes)) if len(self.recent_successes) > 0 else 0.0
        )
        info["spawn_tile"] = int(self.unwrapped.s)
        return obs, info

    def step(self, action: int) -> Tuple[int, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        is_goal = bool(obs == 15 and terminated)
        is_hole = bool(terminated and not is_goal)

        shaped_reward = self.time_penalty
        if is_goal:
            shaped_reward += self.goal_reward
        elif is_hole:
            shaped_reward += self.hole_penalty

        if (terminated or truncated) and self.curriculum:
            self.recent_successes.append(1.0 if is_goal else 0.0)

        info["current_level"] = self.current_level
        info["is_goal"] = is_goal
        info["is_hole"] = is_hole
        info["curriculum_rate"] = (
            float(np.mean(self.recent_successes)) if len(self.recent_successes) > 0 else 0.0
        )
        return obs, shaped_reward, terminated, truncated, info
