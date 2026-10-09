"""Multi-Scenario Quality Ensemble for System 1.

Executes multiple parallel/simultaneous scenario rollouts under different calibration regimes
(e.g., deterministic, calibrated, homeostatic-auto, and exploratory), evaluates trajectory
quality (success, steps-to-solve, cumulative reward, critic latent value, and action purity),
and selects the best high-quality outcome.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import time
from dataclasses import dataclass, field


@dataclass
class TrajectoryResult:
    """Resultado detalhado da execução de uma trajetória com um regime de calibração."""
    scenario_id: int
    calibration: Any
    solved: bool
    steps: int
    cumulative_reward: float
    mean_latent_value: float
    purity_score: float  # [0.0, 1.0] penaliza reversões e oscilações
    actions: List[int]
    decisions: List[Any]
    quality_score: float
    elapsed_ms: float

    def summary(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "calibration": str(self.calibration),
            "solved": self.solved,
            "steps": self.steps,
            "cumulative_reward": round(self.cumulative_reward, 3),
            "mean_latent_value": round(self.mean_latent_value, 3),
            "purity_score": round(self.purity_score, 3),
            "quality_score": round(self.quality_score, 3),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


class QualityEnsembleEvaluator:
    """Avaliador e seletor de trajetórias para garantia de qualidade em múltiplos cenários simultâneos."""

    DEFAULT_CALIBRATIONS = [0.0, 0.15, "auto", 0.35]

    @staticmethod
    def compute_purity_score(actions: List[int], is_atomic_cube: bool = True) -> float:
        """Calcula a pureza da trajetória penalizando reversões imediatas e oscilações redundantes."""
        if len(actions) <= 1:
            return 1.0
        penalties = 0.0
        for i in range(1, len(actions)):
            a_prev = actions[i - 1]
            a_curr = actions[i]
            if is_atomic_cube:
                inv_prev = (a_prev + 1) if (a_prev % 2 == 0) else (a_prev - 1)
                if a_curr == inv_prev:
                    penalties += 1.0
            if i >= 2 and actions[i] == actions[i - 2]:
                penalties += 0.5
        max_possible = max(1.0, float(len(actions) - 1))
        return float(np.clip(1.0 - (penalties / max_possible), 0.0, 1.0))

    @classmethod
    def evaluate_scenarios(
        cls,
        agent: Any,
        env: Any,
        max_steps: int = 40,
        calibrations: Optional[List[Any]] = None,
        is_atomic_cube: bool = True,
    ) -> Tuple[TrajectoryResult, List[TrajectoryResult]]:
        """Executa múltiplos cenários sobre o mesmo ponto de partida com diferentes calibrações e seleciona o melhor."""
        if calibrations is None:
            calibrations = cls.DEFAULT_CALIBRATIONS

        results: List[TrajectoryResult] = []

        # Determina como clonar o ambiente a partir do estado atual
        can_clone = hasattr(env, "clone") and callable(getattr(env, "clone"))

        for s_id, calib in enumerate(calibrations):
            t0 = time.perf_counter()

            # Cria instância independente do ambiente
            if can_clone:
                sim_env = env.clone()
                obs_dict = {
                    "obs": sim_env.core.get_one_hot() if hasattr(sim_env, "core") else sim_env.reset()[0],
                    "prev_action": 0,
                    "prev_reward": 0.0,
                    "delta_obs": None,
                }
            elif hasattr(env, "core"):
                # Fallback para ambientes de Cubo sem clone explícito
                from system1_engine.env.adapters.rubiks import RubiksCubeEnv
                sim_env = RubiksCubeEnv(
                    render_mode=getattr(env, "render_mode", "rgb_array"),
                    scramble_depth=getattr(env, "scramble_depth", 4),
                    max_steps=max_steps,
                )
                sim_env.reset()
                sim_env.core.state = env.core.state.copy()
                sim_env._steps = 0
                sim_env._prev_score = env.core.get_score()
                obs_dict = {
                    "obs": sim_env.core.get_one_hot(),
                    "prev_action": 0,
                    "prev_reward": 0.0,
                    "delta_obs": None,
                }
            else:
                raise ValueError("Ambiente não suporta clonagem para avaliação em múltiplos cenários.")

            # Reseta estado neural e histórico para cada cenário
            agent.reset_memory()

            actions: List[int] = []
            decisions: List[Any] = []
            cumulative_reward = 0.0
            latent_values: List[float] = []
            solved = False

            curr_obs = obs_dict

            for step in range(max_steps):
                decision = agent.act_fast(
                    curr_obs,
                    calibration=calib,
                    return_decision=True,
                    return_value=True,
                    quality_filter=True,
                )
                act = int(decision.action)
                actions.append(act)
                decisions.append(decision)
                if decision.latent_value is not None:
                    latent_values.append(decision.latent_value)

                next_obs, rew, terminated, truncated, info = sim_env.step(act)
                cumulative_reward += float(rew)

                is_solved = bool(info.get("is_solved", False) or terminated)
                if is_solved:
                    solved = True
                    break
                if truncated:
                    break

                curr_obs = {
                    "obs": next_obs,
                    "prev_action": act,
                    "prev_reward": float(rew),
                    "delta_obs": None,
                }

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            purity = cls.compute_purity_score(actions, is_atomic_cube=is_atomic_cube)
            mean_val = float(np.mean(latent_values)) if latent_values else 0.0
            n_steps = len(actions)

            # Função de pontuação de qualidade
            if solved:
                # Prioridade absoluta para resolução, premiando menos passos e alta pureza
                quality_score = 10000.0 - (n_steps * 20.0) + (cumulative_reward * 5.0) + (purity * 100.0)
            else:
                # Caso não resolvido, premia progresso de recompensa, valor latente e pureza
                quality_score = (cumulative_reward * 10.0) + (mean_val * 5.0) + (purity * 50.0) - (n_steps * 2.0)

            traj = TrajectoryResult(
                scenario_id=s_id,
                calibration=calib,
                solved=solved,
                steps=n_steps,
                cumulative_reward=cumulative_reward,
                mean_latent_value=mean_val,
                purity_score=purity,
                actions=actions,
                decisions=decisions,
                quality_score=quality_score,
                elapsed_ms=elapsed_ms,
            )
            results.append(traj)

        # Ordena pelo quality_score decrescente
        sorted_results = sorted(results, key=lambda x: x.quality_score, reverse=True)
        best_result = sorted_results[0]

        return best_result, sorted_results
