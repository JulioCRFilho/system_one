"""Módulo de resolução e instalação automática de dependências em runtime para ambientes Gymnasium.

Permite que qualquer ambiente selecionado pelo usuário (MuJoCo, Box2D, Atari, MiniGrid, etc.)
seja detectado, tenha suas bibliotecas instaladas sob demanda no ambiente virtual ativo e seja
inicializado transparentemente sem interromper a execução do HUD ou do CLI.
"""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Callable, List, Optional

import gymnasium as gym
from gymnasium.error import DependencyNotInstalled, NamespaceNotFound

# Mapeamento estático para suites populares do Gymnasium
SUITE_PATTERNS = {
    "box2d": {
        "keywords": ["lunarlander", "bipedalwalker", "carracing"],
        "packages": ["swig", "gymnasium[box2d]"],
    },
    "mujoco": {
        "keywords": [
            "ant-",
            "halfcheetah-",
            "hopper-",
            "humanoid-",
            "humanoidstandup-",
            "invertedpendulum-",
            "inverteddoublependulum-",
            "reacher-",
            "swimmer-",
            "walker2d-",
            "pusher-",
        ],
        "packages": ["gymnasium[mujoco]"],
    },
    "atari": {
        "keywords": [
            "ale/",
            "breakout",
            "pong",
            "spaceinvaders",
            "seaquest",
            "beamrider",
            "qbert",
            "mspacman",
            "asteroids",
            "centipede",
            "freeway",
            "enduro",
            "defender",
            "alien",
        ],
        "packages": ["gymnasium[atari]", "ale-py"],
    },
    "minigrid": {
        "keywords": ["minigrid-", "babyai-"],
        "packages": ["minigrid"],
    },
}


def find_uv_binary() -> Optional[str]:
    """Localiza o binário do gerenciador ultra-rápido 'uv' no sistema."""
    uv = shutil.which("uv")
    if uv and os.path.isfile(uv) and os.access(uv, os.X_OK):
        return uv
    candidates = [
        "/opt/homebrew/bin/uv",
        os.path.expanduser("~/.local/bin/uv"),
        os.path.expanduser("~/.cargo/bin/uv"),
        "/usr/local/bin/uv",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def resolve_missing_packages(env_id: str, exc: Optional[Exception] = None) -> List[str]:
    """Identifica quais pacotes pip precisam ser instalados para executar o ambiente."""
    packages: List[str] = []
    env_lower = env_id.lower()
    err_str = str(exc) if exc is not None else ""
    err_lower = err_str.lower()

    # 1. Extração dinâmica via regex na mensagem de erro do Gymnasium (ex: pip install "gymnasium[mujoco]")
    if err_str:
        matches = re.findall(
            r"pip install\s+[\"'\`]?([a-zA-Z0-9_\-\[\]\.\>\=\<\~]+)[\"'\`]?",
            err_str,
        )
        for m in matches:
            pkg = m.strip("\"'`")
            if pkg and pkg not in packages:
                packages.append(pkg)

    # 2. Heurística baseada no erro ou no namespace
    if "mujoco" in err_lower or "namespace mujoco" in err_lower:
        if "gymnasium[mujoco]" not in packages:
            packages.append("gymnasium[mujoco]")

    if "box2d" in err_lower:
        for p in ["swig", "gymnasium[box2d]"]:
            if p not in packages:
                packages.append(p)

    if "ale" in err_lower or "namespace ale" in err_lower or "atari" in err_lower:
        for p in ["gymnasium[atari]", "ale-py"]:
            if p not in packages:
                packages.append(p)

    if "minigrid" in err_lower or "namespace minigrid" in err_lower:
        if "minigrid" not in packages:
            packages.append("minigrid")

    # 3. Fallback para catálogo conhecido pelo nome do ambiente caso a lista esteja vazia
    if not packages:
        for suite, data in SUITE_PATTERNS.items():
            if any(kw in env_lower for kw in data["keywords"]):
                for p in data["packages"]:
                    if p not in packages:
                        packages.append(p)
                break

    return packages


def install_packages(
    packages: List[str],
    log_fn: Optional[Callable[[str], None]] = None,
) -> bool:
    """Executa a instalação em segundo plano usando 'uv' (quando disponível) ou 'pip'."""
    if not packages:
        return True

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(msg, flush=True)

    uv_bin = find_uv_binary()
    if uv_bin:
        cmd = [uv_bin, "pip", "install", "--python", sys.executable, *packages]
        tool_name = "uv"
    else:
        cmd = [sys.executable, "-m", "pip", "install", *packages]
        tool_name = "pip"

    pkgs_label = ", ".join(packages)
    _log(f"📦 [Auto-Installer] Instalando pacotes necessários via {tool_name}: {pkgs_label}...")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout:
            for line in proc.stdout:
                line_str = line.strip()
                if line_str and not line_str.startswith("Looking in indexes"):
                    _log(f"   [Auto-Installer] {line_str}")
        proc.wait()

        if proc.returncode == 0:
            _log(f"✅ [Auto-Installer] Dependências instaladas com sucesso: {pkgs_label}")
            return True

        _log(f"⚠️ [Auto-Installer] {tool_name} retornou código {proc.returncode}. Tentando fallback...")
        if uv_bin:
            fallback_cmd = [sys.executable, "-m", "pip", "install", *packages]
            _log(f"📦 [Auto-Installer] Executando fallback via pip: {pkgs_label}...")
            fb_proc = subprocess.Popen(
                fallback_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if fb_proc.stdout:
                for line in fb_proc.stdout:
                    line_str = line.strip()
                    if line_str:
                        _log(f"   [Auto-Installer] {line_str}")
            fb_proc.wait()
            if fb_proc.returncode == 0:
                _log(f"✅ [Auto-Installer] Dependências instaladas via pip fallback: {pkgs_label}")
                return True

        return False
    except Exception as e:
        _log(f"❌ [Auto-Installer] Erro ao invocar instalador: {e}")
        return False


def make_gym_env_with_auto_install(
    env_id: str,
    render_mode: Optional[str] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Any:
    """Cria o ambiente Gymnasium, detectando e instalando automaticamente dependências ausentes.

    Se o ambiente exigir pacotes como MuJoCo, Box2D, Atari ou MiniGrid e estes não estiverem
    instalados, a exceção é interceptada, os pacotes são baixados e instalados em runtime,
    e a inicialização é realizada novamente com sucesso.
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(msg, flush=True)

    # 0. Verificação antecipada para suites que exigem importação de plugin (ex: ale-py para namespace ALE)
    if env_id.lower().startswith("ale/"):
        try:
            import ale_py  # noqa: F401
        except ImportError:
            _log(f"📦 [Auto-Installer] Ambiente Atari detectado ('{env_id}'). Instalando ale-py...")
            install_packages(["gymnasium[atari]", "ale-py"], log_fn=_log)
            importlib.invalidate_caches()
            try:
                import ale_py  # noqa: F401
            except ImportError:
                pass

    try:
        return gym.make(env_id, render_mode=render_mode, **kwargs)
    except (DependencyNotInstalled, NamespaceNotFound, ModuleNotFoundError, ImportError) as exc:
        _log(f"📦 [Auto-Installer] Dependência ausente detectada para '{env_id}': {exc}")
        packages = resolve_missing_packages(env_id, exc)

        if not packages:
            _log(f"❌ [Auto-Installer] Não foi possível deduzir automaticamente os pacotes para '{env_id}'.")
            raise

        ok = install_packages(packages, log_fn=_log)
        if not ok:
            raise RuntimeError(
                f"Falha ao instalar automaticamente as dependências {packages} para o ambiente '{env_id}'. "
                f"Por favor, execute manualmente: pip install {' '.join(packages)}"
            ) from exc

        # Atualiza caches do importlib e plugins
        importlib.invalidate_caches()
        if any("ale" in p.lower() for p in packages):
            try:
                import ale_py  # noqa: F401
            except ImportError:
                pass
        if any("minigrid" in p.lower() for p in packages):
            try:
                import minigrid  # noqa: F401
            except ImportError:
                pass

        _log(f"🔄 [Auto-Installer] Recarregando ambiente '{env_id}' após instalação...")
        return gym.make(env_id, render_mode=render_mode, **kwargs)
