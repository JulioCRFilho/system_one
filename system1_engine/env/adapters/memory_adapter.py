from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import ctypes
import os
import struct
import sys
import gymnasium as gym
import numpy as np

from system1_engine.env.adapters.base import BaseGameAdapter
from system1_engine.env.adapters.window_adapter import WindowCaptureEnv


@dataclass
class MemoryField:
    """Definição de um campo na memória RAM do processo alvo."""
    name: str
    offset: int
    dtype: str = "float32"  # "int32", "uint32", "float32", "int64", "float64", "byte"
    scale: float = 1.0
    is_reward: bool = False
    is_terminal: bool = False  # Termina se <= 0 (ex.: HP)


class MemoryReaderBackend:
    """Interface abstrata para leitura de memória de processos do SO."""

    def read_bytes(self, address: int, size: int) -> bytes:
        raise NotImplementedError

    def close(self) -> None:
        pass


class BufferMemoryBackend(MemoryReaderBackend):
    """Backend de memória simulada/compartilhada para testes e emulação controlada."""

    def __init__(self, buffer_size: int = 4096) -> None:
        self.buffer = bytearray(buffer_size)

    def write_value(self, offset: int, value: Union[int, float], dtype: str = "float32") -> None:
        fmt = self._dtype_to_format(dtype)
        packed = struct.pack(fmt, value)
        self.buffer[offset : offset + len(packed)] = packed

    def read_bytes(self, address: int, size: int) -> bytes:
        return bytes(self.buffer[address : address + size])

    @staticmethod
    def _dtype_to_format(dtype: str) -> str:
        formats = {
            "int32": "<i",
            "uint32": "<I",
            "float32": "<f",
            "int64": "<q",
            "float64": "<d",
            "byte": "<b",
            "uint8": "<B",
        }
        return formats.get(dtype, "<f")


class ProcessMemoryBackend(MemoryReaderBackend):
    """Backend nativo para leitura de memória de processos reais no Linux e Windows."""

    def __init__(self, pid: int, base_address: int = 0) -> None:
        self.pid = pid
        self.base_address = base_address
        self._mem_file = None

        if sys.platform.startswith("linux"):
            mem_path = f"/proc/{pid}/mem"
            if os.path.exists(mem_path):
                try:
                    self._mem_file = open(mem_path, "rb", buffering=0)
                except PermissionError:
                    self._mem_file = None

    def read_bytes(self, address: int, size: int) -> bytes:
        abs_address = self.base_address + address
        if self._mem_file is not None:
            self._mem_file.seek(abs_address)
            return self._mem_file.read(size)
        return bytes(size)

    def close(self) -> None:
        if self._mem_file is not None:
            self._mem_file.close()
            self._mem_file = None


class MemoryHookEnv(BaseGameAdapter):
    """Nível 2: Adaptador de Engenharia Reversa de Estado (Memory Hooking).

    Suporta dois modos operacionais de alta fidelidade:
      1. Modo Vetorial Puro: Lê diretamente variáveis de dinâmica da RAM
         ([x, y, z, vx, vy, hp, stamina]) e produz observações Box estruturadas.
      2. Modo Híbrido: O agente recebe os frames visuais (C, 84, 84), mas a recompensa
         e o status de Game Over são computados com precisão matemática da RAM (HP, Score).
    """

    def __init__(
        self,
        process_name: Optional[str] = None,
        pid: Optional[int] = None,
        memory_schema: Optional[Union[List[MemoryField], Dict[str, Dict[str, Any]]]] = None,
        actions_map: Optional[Dict[int, Any]] = None,
        capture_screen: bool = False,
        window_bbox: Optional[Dict[str, int]] = None,
        target_fps: int = 30,
        memory_backend: Optional[MemoryReaderBackend] = None,
        reward_calculator: Optional[Callable[[Dict[str, float], Dict[str, float]], float]] = None,
        base_address: int = 0,
    ) -> None:
        self.process_name = process_name
        self.pid = pid
        self.capture_screen = capture_screen
        self.actions_map = actions_map or {0: None}
        self.base_address = base_address

        # Backend de memória
        if memory_backend is not None:
            self.backend = memory_backend
        elif pid is not None:
            self.backend = ProcessMemoryBackend(pid=pid, base_address=base_address)
        else:
            # Fallback seguro para mock/buffer testável
            self.backend = BufferMemoryBackend()

        # Parse defensivo do schema de memória
        self.fields: List[MemoryField] = self._parse_schema(memory_schema or {})

        # Sub-adaptador de tela para modo híbrido
        self.screen_adapter: Optional[WindowCaptureEnv] = None
        if self.capture_screen:
            self.screen_adapter = WindowCaptureEnv(
                window_bbox=window_bbox,
                actions_map=self.actions_map,
                target_fps=target_fps,
                grayscale=True,
            )
            obs_space = self.screen_adapter.observation_space
            is_visual = True
        else:
            # Modo vetorial: vetor numérico com os valores extraídos
            num_fields = max(1, len(self.fields))
            obs_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(num_fields,),
                dtype=np.float32,
            )
            is_visual = False

        action_space = gym.spaces.Discrete(len(self.actions_map))

        super().__init__(
            observation_space=obs_space,
            action_space=action_space,
            is_visual=is_visual,
            target_fps=float(target_fps),
        )

        self.reward_calculator = reward_calculator
        self._prev_ram_values: Dict[str, float] = {}

    def _parse_schema(
        self, schema: Union[List[Union[MemoryField, Dict[str, Any]]], Dict[str, Dict[str, Any]]]
    ) -> List[MemoryField]:
        if isinstance(schema, list):
            parsed: List[MemoryField] = []
            for item in schema:
                if isinstance(item, MemoryField):
                    parsed.append(item)
                elif isinstance(item, dict):
                    parsed.append(
                        MemoryField(
                            name=str(item.get("name", "field")),
                            offset=int(item.get("offset", 0)),
                            dtype=str(item.get("dtype", "float32")),
                            scale=float(item.get("scale", 1.0)),
                            is_reward=bool(item.get("is_reward", False)),
                            is_terminal=bool(item.get("is_terminal", False)),
                        )
                    )
            return parsed

        parsed_dict: List[MemoryField] = []
        for name, spec in schema.items():
            parsed_dict.append(
                MemoryField(
                    name=name,
                    offset=int(spec.get("offset", 0)),
                    dtype=str(spec.get("dtype", "float32")),
                    scale=float(spec.get("scale", 1.0)),
                    is_reward=bool(spec.get("is_reward", False)),
                    is_terminal=bool(spec.get("is_terminal", False)),
                )
            )
        return parsed_dict

    def read_ram(self) -> Dict[str, float]:
        """Lê os valores atuais dos campos na memória do processo."""
        results: Dict[str, float] = {}
        type_sizes = {
            "int32": (4, "<i"),
            "uint32": (4, "<I"),
            "float32": (4, "<f"),
            "int64": (8, "<q"),
            "float64": (8, "<d"),
            "byte": (1, "<b"),
            "uint8": (1, "<B"),
        }

        for f in self.fields:
            size, fmt = type_sizes.get(f.dtype, (4, "<f"))
            raw = self.backend.read_bytes(f.offset, size)
            if len(raw) == size:
                val = struct.unpack(fmt, raw)[0]
                results[f.name] = float(val) * f.scale
            else:
                results[f.name] = 0.0

        return results

    def _reset_impl(
        self, seed: Optional[int], options: Optional[Dict[str, Any]]
    ) -> Tuple[Any, Dict[str, Any]]:
        self._prev_ram_values = self.read_ram()

        if self.screen_adapter is not None:
            obs, info = self.screen_adapter.reset(seed=seed, options=options)
            info["ram"] = self._prev_ram_values
            return obs, info
        else:
            # Modo vetorial: valores numéricos em numpy float32
            vector = np.array(
                [self._prev_ram_values.get(f.name, 0.0) for f in self.fields],
                dtype=np.float32,
            )
            return vector, {"ram": self._prev_ram_values}

    def _step_impl(
        self, action: int
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        # 1. Se em modo híbrido, avança o adaptador de tela
        screen_info: Dict[str, Any] = {}
        if self.screen_adapter is not None:
            screen_obs, _, _, _, screen_info = self.screen_adapter.step(action)
            obs = screen_obs
        else:
            # Modo vetorial puro (avanço lógico ou emulação de tecla)
            obs = None

        # 2. Leitura cirúrgica da RAM
        curr_ram = self.read_ram()

        # 3. Cálculo de Recompensa
        reward = 0.0
        if self.reward_calculator is not None:
            reward = self.reward_calculator(curr_ram, self._prev_ram_values)
        else:
            # Recompensa heurística padrão a partir do delta dos campos marcados como 'is_reward'
            for f in self.fields:
                if f.is_reward:
                    delta = curr_ram.get(f.name, 0.0) - self._prev_ram_values.get(f.name, 0.0)
                    reward += delta

        # 4. Critério de Término (Game Over)
        terminated = False
        for f in self.fields:
            if f.is_terminal and curr_ram.get(f.name, 0.0) <= 0.0:
                terminated = True
                break

        truncated = False
        self._prev_ram_values = curr_ram.copy()

        # Montagem de observação vetorial caso não seja híbrido
        if obs is None:
            obs = np.array(
                [curr_ram.get(f.name, 0.0) for f in self.fields],
                dtype=np.float32,
            )

        info = {**screen_info, "ram": curr_ram}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _close_impl(self) -> None:
        if self.backend is not None:
            self.backend.close()
        if self.screen_adapter is not None:
            self.screen_adapter.close()
