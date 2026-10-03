import gymnasium as gym
import numpy as np


class MountainCarNormalizedWrapper(gym.Wrapper):
    """Normaliza a velocidade do MountainCar para a faixa [-1.0, 1.0].
    
    A velocidade bruta do MountainCar é [-0.07, 0.07], enquanto a posição é [-1.2, 0.6].
    Essa diferença de escala de 14x faz com que os encoders neurais ignorem a velocidade.
    Esta normalização equaliza a magnitude das features para que a rede neural consiga
    detectar a direção do movimento instantaneamente.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Ajusta o observation_space para refletir a nova escala da velocidade
        low = np.array([env.observation_space.low[0], -1.0], dtype=np.float32)
        high = np.array([env.observation_space.high[0], 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        norm_obs = np.array([obs[0], obs[1] * 14.2857], dtype=np.float32)
        return norm_obs, r, term, trunc, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        norm_obs = np.array([obs[0], obs[1] * 14.2857], dtype=np.float32)
        return norm_obs, info


class MountainCarEnergyRewardWrapper(gym.Wrapper):
    """Reward Shaping Físico baseado em Potência Mecânica e Energia Potencial.
    
    Transforma a recompensa esparsa do MountainCar (-1.0 fixo) em gradiente contínuo:
    1. Potência Mecânica (F * v): Recompensa aceleração na direção do movimento (embalar)
       e penaliza aceleração contrária (frear).
    2. Energia Potencial: Bônus pela altura alcançada em qualquer uma das encostas.
    3. Bônus de Vitória: Reforço positivo expressivo ao atingir o mastro (position >= 0.5).
    """

    def __init__(
        self,
        env: gym.Env,
        power_scale: float = 20.0,
        height_scale: float = 5.0,
        goal_bonus: float = 50.0,
    ) -> None:
        super().__init__(env)
        self.power_scale = power_scale
        self.height_scale = height_scale
        self.goal_bonus = goal_bonus

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        if hasattr(self.unwrapped, "state") and self.unwrapped.state is not None:
            pos = float(self.unwrapped.state[0])
            vel = float(self.unwrapped.state[1])
        else:
            pos = float(obs[0])
            norm_vel = float(obs[1])
            vel = norm_vel / 14.2857 if abs(norm_vel) > 0.08 else norm_vel

        # Potência mecânica: força * velocidade
        force = float(action - 1)  # 0 -> -1 (ré), 1 -> 0, 2 -> +1 (frente)
        power_bonus = self.power_scale * (force * vel)

        # Altura potencial no vale de seno: sin(3*x) varia de -1 (fundo) a ~1 (topo)
        height = np.sin(3.0 * pos) + 1.0  # [0.0, 2.0]
        height_bonus = self.height_scale * height

        shaped_reward = float(r) + power_bonus + height_bonus
        if term and pos >= 0.5:
            shaped_reward += self.goal_bonus

        return obs, shaped_reward, term, trunc, info
