from typing import Any, Dict
import gymnasium as gym

from system1_engine.env.wrapper import UniversalS1Wrapper


def make_game_env(adapter_type: str, config: Dict[str, Any]) -> UniversalS1Wrapper:
    """Instancia o ambiente adequado para qualquer um dos 3 níveis de integração.

    Garante que o retorno seja sempre empacotado no UniversalS1Wrapper com os
    canais diferenciais (Δs_t), histórico causal (a_{t-1}, r_{t-1}) e barramento
    compatível com o UniversalS1Agent.

    Parâmetros:
      - adapter_type: "window" (Nível 1), "memory" (Nível 2) ou "native" (Nível 3).
      - config: Dicionário contendo os parâmetros de inicialização específicos.

    Retorna:
      - Instância de UniversalS1Wrapper pronta para treino ou inferência com o System 1.
    """
    adapter_key = adapter_type.strip().lower()

    if adapter_key == "window":
        from system1_engine.env.adapters.window_adapter import WindowCaptureEnv

        raw_env = WindowCaptureEnv(
            window_bbox=config.get("window_bbox"),
            actions_map=config.get("actions_map"),
            target_fps=config.get("target_fps", 30),
            grayscale=config.get("grayscale", True),
            reward_fn=config.get("reward_fn"),
            done_fn=config.get("done_fn"),
            reset_action=config.get("reset_action"),
            mock_capture_source=config.get("mock_capture_source"),
            key_press_duration=config.get("key_press_duration", 0.02),
        )

    elif adapter_key == "memory":
        from system1_engine.env.adapters.memory_adapter import MemoryHookEnv

        raw_env = MemoryHookEnv(
            process_name=config.get("process_name"),
            pid=config.get("pid"),
            memory_schema=config.get("memory_schema"),
            actions_map=config.get("actions_map"),
            capture_screen=config.get("capture_screen", False),
            window_bbox=config.get("window_bbox"),
            target_fps=config.get("target_fps", 30),
            memory_backend=config.get("memory_backend"),
            reward_calculator=config.get("reward_calculator"),
            base_address=config.get("base_address", 0),
        )

    elif adapter_key == "native":
        from system1_engine.env.adapters.native_adapter import NativeEngineEnv

        raw_env = NativeEngineEnv(
            engine_type=config.get("engine_type", "native_sim"),
            scenario_path=config.get("scenario_path"),
            args=config.get("args", {}),
            is_visual=config.get("is_visual", True),
            channels=config.get("channels", 1),
            obs_dim=config.get("obs_dim", 4),
            action_dim=config.get("action_dim", 2),
            is_action_discrete=config.get("is_action_discrete", True),
            socket_address=config.get("socket_address"),
            custom_step_fn=config.get("custom_step_fn"),
            custom_reset_fn=config.get("custom_reset_fn"),
        )

    else:
        raise ValueError(
            f"Tipo de adaptador desconhecido: '{adapter_type}'. "
            f"Use 'window' (Nível 1), 'memory' (Nível 2) ou 'native' (Nível 3)."
        )

    # Retorna o ambiente com o wrapper de barramento latente unificado
    return UniversalS1Wrapper(raw_env, is_visual=raw_env.is_visual)
