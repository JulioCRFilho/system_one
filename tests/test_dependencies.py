import unittest
from unittest.mock import MagicMock, patch

from gymnasium.error import DependencyNotInstalled, NamespaceNotFound
from system1_engine.env.dependencies import (
    find_uv_binary,
    install_packages,
    make_gym_env_with_auto_install,
    resolve_missing_packages,
)
from system1_engine.hud.worker import build_hud_env


class TestDependencies(unittest.TestCase):
    def test_find_uv_binary(self):
        uv_path = find_uv_binary()
        # uv may or may not be installed in any environment, but if present must be a non-empty string
        if uv_path is not None:
            self.assertIsInstance(uv_path, str)
            self.assertTrue(len(uv_path) > 0)

    def test_resolve_missing_packages_by_env_name(self):
        # MuJoCo
        mujoco_pkgs = resolve_missing_packages("Ant-v5")
        self.assertIn("gymnasium[mujoco]", mujoco_pkgs)

        # Atari
        atari_pkgs = resolve_missing_packages("ALE/Breakout-v5")
        self.assertIn("gymnasium[atari]", atari_pkgs)
        self.assertIn("ale-py", atari_pkgs)

        # Box2D
        box2d_pkgs = resolve_missing_packages("LunarLander-v3")
        self.assertIn("gymnasium[box2d]", box2d_pkgs)
        self.assertIn("swig", box2d_pkgs)

        # MiniGrid
        minigrid_pkgs = resolve_missing_packages("MiniGrid-Empty-5x5-v0")
        self.assertIn("minigrid", minigrid_pkgs)

    def test_resolve_missing_packages_from_error_message(self):
        err_mujoco = DependencyNotInstalled('MuJoCo is not installed, run `pip install "gymnasium[mujoco]"`')
        pkgs = resolve_missing_packages("SomeUnknownEnv-v1", err_mujoco)
        self.assertEqual(pkgs, ["gymnasium[mujoco]"])

        err_box2d = DependencyNotInstalled(
            'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
        )
        pkgs = resolve_missing_packages("SomeUnknownEnv-v1", err_box2d)
        self.assertIn("swig", pkgs)
        self.assertIn("gymnasium[box2d]", pkgs)

        err_ale = NamespaceNotFound("Namespace ALE not found. Have you installed the proper package for ALE?")
        pkgs = resolve_missing_packages("ALE/Pong-v5", err_ale)
        self.assertIn("gymnasium[atari]", pkgs)
        self.assertIn("ale-py", pkgs)

    def test_install_packages_empty(self):
        self.assertTrue(install_packages([]))

    @patch("subprocess.Popen")
    def test_install_packages_success_mock(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout = ["Collecting package\n", "Successfully installed\n"]
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        logs = []
        res = install_packages(["some-package"], log_fn=logs.append)
        self.assertTrue(res)
        self.assertTrue(any("some-package" in log for log in logs))
        self.assertTrue(any("sucesso" in log for log in logs))

    def test_make_gym_env_standard(self):
        env = make_gym_env_with_auto_install("CartPole-v1")
        self.assertIsNotNone(env)
        obs, _ = env.reset()
        self.assertIsNotNone(obs)
        env.close()

    @patch("gymnasium.make")
    @patch("system1_engine.env.dependencies.install_packages")
    def test_make_gym_env_auto_install_retry(self, mock_install, mock_gym_make):
        mock_install.return_value = True

        mock_created_env = MagicMock()
        # First call raises DependencyNotInstalled, second call succeeds
        mock_gym_make.side_effect = [
            DependencyNotInstalled("MuJoCo is not installed, run `pip install \"gymnasium[mujoco]\"`"),
            mock_created_env,
        ]

        logs = []
        env = make_gym_env_with_auto_install("Ant-v5", log_fn=logs.append)
        self.assertEqual(env, mock_created_env)
        self.assertEqual(mock_gym_make.call_count, 2)
        mock_install.assert_called_once()
        self.assertIn("gymnasium[mujoco]", mock_install.call_args[0][0])

    def test_build_hud_env_with_standard_and_installed(self):
        env = build_hud_env("CartPole-v1", render_mode="in_browser")
        obs, _ = env.reset()
        self.assertIn("obs", obs)
        env.close()


if __name__ == "__main__":
    unittest.main()
