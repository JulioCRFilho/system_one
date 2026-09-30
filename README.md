# Universal System 1 RL Agent Engine

Modular, ultra-lightweight Python/PyTorch library implementing an amortized reflex agent trained via pure Recurrent Proximal Policy Optimization (PPO).

Unlike deliberative "System 2" frameworks (LLMs with Chain-of-Thought, Monte Carlo tree search), this agent operates by **amortized reflex**: executing a single deterministic forward pass with sub-millisecond latency on CPU (`act_fast() <= 0.8 ms`, measured at **~0.15 ms**).

---

## ⚡ Key Characteristics

1. **Extreme Lightweightness**:
   - **Vector mode**: ~725k parameters (~2.8 MB float32, < 15 MB RAM).
   - **Visual mode (IMPALA)**: ~2.02M parameters (~7.7 MB float32, < 25 MB RAM).
   - Rigid parameter and memory budget strictly respected (< 35 MB RAM limit).
2. **Sub-Millisecond Reflex Latency**:
   - `agent.act_fast()` executes in **0.150 ms** on standard CPU (5.3x below the 0.8 ms limit).
   - Visual inference with IMPALA takes **~1.17 ms** (well below the 5.0 ms visual budget).
3. **Confidence Gating & System 2 Arbitration**:
   - `agent.act_with_confidence()` extracts Shannon/differential entropy, Top-1 confidence, and margin in **+1.3 µs** overhead.
   - Automatically raises `is_uncertain = True` when the policy distribution collapses towards uniform, providing an amortized zero-overhead trigger to call a deliberative System 2 planner or human supervisor.
   - Robust continuous uncertainty normalization immune to negative differential entropy ($H < 0$) and variance underflow.
4. **Transfer without Catastrophic Forgetting**:
   - Strict decoupling between perception front-ends, universal reflexive trunk (`System1Trunk`), and action/value heads.
   - Cross-scenario transfer freezes the recurrent trunk (`requires_grad = False`) and trains only new front-end adapters and heads (verified with bitwise invariance $\Delta = 0.0000$).
5. **Universal 3-Level Environment Integration**:
   - **Level 1 (WindowCaptureEnv)**: Pure black-box screen capture (`mss`) and OS input injection (`pynput`) with real-time frame pacing.
   - **Level 2 (MemoryHookEnv)**: Direct RAM state reads for surgical reward and Game Over detection, with hybrid mode (pixels + RAM feedback) eliminating 100% of computer vision false positives.
   - **Level 3 (NativeEngineEnv)**: Synchronous lock-step headless execution (> 10,000 FPS) with native binaries (ViZDoom, Gym-Retro, Godot RL, Unity).
6. **Real-Time Asynchronous Telemetry Bus**:
   - `LiveStatsTracker`: RAM-based circular ring buffers ($O(1)$ deque) with **~96 ns** insertion overhead (20x faster than the 2 µs ceiling).
   - **Rich Terminal Dashboard (`S1LiveDashboard`)**: Live 3-phase multi-panel operational view in the terminal.
   - **Web Streaming Dashboard (`TelemetryServer`)**: Pure Python standard library HTTP/SSE streaming server serving a dark, responsive dashboard on `http://localhost:8050` with Chart.js sliding charts.
7. **Pure Implementation**:
   - Zero high-level RL libraries (no Stable-Baselines3, no Ray/RLlib, no TRL). 100% pure PyTorch 2.2+, Gymnasium, and NumPy.

---

## 📐 Unified Tensor Flow Architecture

```
MODO VETORIAL:
[ s_t ] (Estado Atual)               ──► [ VectorEncoder ]   ──► z_s  ∈ ℝ^256 ┐
[ Δs_t = s_t - s_{t-1} ] (Derivada)   ──► [ DeltaEncoder ]    ──► z_Δ  ∈ ℝ^64  │
                                                                               ├─► Concat (320)
MODO VISUAL:                                                                  │
[ s_t, s_{t-1} ] (Canais: 2*C)       ──► [ ImpalaEncoder ]   ──► z_vis ∈ ℝ^320 ┘
                                                                               │
FEEDBACK PASSADO:                                                              │
[ a_{t-1} ] (Ação Anterior)          ──► [ ActionEncoder ]   ──► z_a  ∈ ℝ^16 ──┤
[ r_{t-1} ] (Recompensa Anterior)    ──► [ RewardEncoder ]   ──► z_r  ∈ ℝ^1  ──┘
                                                                               │
                                                           (Concatenação Total)▼
                                                                  z_in ∈ ℝ^337
                                                                               │
                                                                      [ LayerNorm(337) ]
                                                                               │
                                                                               ▼
                                                         ╔════════════════════════════════╗
                                                         ║   System 1 Recurrent Trunk     ║
                                                         ║   - nn.GRU (337 -> 256)        ║
                                                         ║   - 2x ResMLP Blocks (256)     ║
                                                         ╚════════════════════════════════╝
                                                                               │
                                                                               ▼
                                                                    h_t ∈ ℝ^256 (Latente)
                                                                    ┌──────────┴──────────┐
                                                                    ▼                     ▼
                                                            [ Policy Head ]        [ Critic Head ]
                                                            Dist. Ações (π)         Valor V(s)
```

---

## 🗂️ Directory Structure

```
system_one/
├── system1_engine/                 # Core engine package
│   ├── __init__.py
│   ├── core/                       # Latent representation & amortized reflex
│   │   ├── encoders.py             # VectorFrontEnd, ImpalaVisualFrontEnd, Action/Reward Adapters
│   │   ├── trunk.py                # System1Trunk (nn.GRU 337->256 + 2x ResMLP 256)
│   │   ├── heads.py                # CategoricalPolicyHead, GaussianPolicyHead, ValueHead
│   │   └── agent.py                # UniversalS1Agent (act_fast, act_with_confidence)
│   ├── env/                        # State ingestion & environment adapters
│   │   ├── wrapper.py              # UniversalS1Wrapper (Δs, past action/reward, causal buffers)
│   │   └── adapters/               # 3-Level Integration Layer (common gym.Env)
│   │       ├── base.py             # BaseGameAdapter (abstract class)
│   │       ├── window_adapter.py   # Level 1: WindowCaptureEnv (mss + pynput)
│   │       ├── memory_adapter.py   # Level 2: MemoryHookEnv (RAM offsets + hybrid mode)
│   │       ├── native_adapter.py   # Level 3: NativeEngineEnv (Lock-Step Headless >10k FPS)
│   │       └── factory.py          # make_game_env(...)
│   ├── training/                   # Recurrent Reinforcement Learning
│   │   ├── buffer.py               # RecurrentRolloutBuffer (chunked BPTT)
│   │   └── ppo.py                  # RecurrentPPOTrainer with GAE & episode boundary masking
│   ├── transfer/                   # Knowledge transfer & isolation
│   │   └── manager.py              # KnowledgeTransferManager (strict trunk freezing)
│   ├── telemetry/                  # High-frequency real-time observability
│   │   ├── tracker.py              # LiveStatsTracker with O(1) circular ring buffers
│   │   ├── dashboard.py            # S1LiveDashboard (Rich terminal live layout)
│   │   └── server.py               # TelemetryServer (HTTP/SSE live streaming on port 8050)
│   └── cli.py                      # Command-line interface (--mode train/run/benchmark)
├── examples/                       # Ready-to-run operational task scripts
│   ├── 01_benchmark_latency.py     # Task 1: CPU latency benchmark (<= 0.8 ms)
│   ├── 02_evaluate_cartpole.py     # Task 2: Pretrained checkpoint evaluation
│   ├── 03_train_cartpole.py        # Task 3: Training from scratch with Recurrent PPO
│   ├── 04_transfer_learning_acrobot.py # Task 4: Transfer learning with frozen trunk
│   ├── 05_continuous_action_pendulum.py # Task 5: Continuous control (GaussianPolicyHead)
│   ├── 06_visual_observation_impala.py # Task 6: Visual perception with IMPALA
│   ├── 07_run_all_tasks.py         # Task 7: Batch sequential execution of all tasks
│   ├── 08_adapters_three_levels.py # Task 8: 3 Integration Levels (Window/Mem/Native)
│   ├── 09_confidence_gating.py     # Task 9: Uncertainty mechanism & System 2 trigger
│   ├── 10_vizdoom_visual_transfer.py # Task 10: Visual transfer test on ViZDoom
│   ├── 11_live_telemetry_dashboard.py # Task 11: Real-time telemetry & Rich dashboard
│   └── 12_web_telemetry_streaming.py  # Task 12: Web telemetry streaming (SSE + Chart.js)
├── tests/                          # Automated test suite (32 unit tests)
│   ├── test_dimensions.py          # Tensor contracts and ℝ^337 bus validation
│   ├── test_wrapper.py             # Buffer resets and delta computation
│   ├── test_latency.py             # Sub-millisecond CPU latency budget
│   ├── test_transfer.py            # Bitwise invariance with freeze_trunk=True
│   ├── test_convergence.py         # Proven CartPole-v1 convergence (>= 475.0)
│   ├── test_adapters.py            # Window, Memory, and Native/ViZDoom adapters
│   ├── test_confidence.py          # Discrete and continuous Confidence Gating
│   ├── test_telemetry.py           # O(1) tracker overhead and Rich dashboard
│   └── test_telemetry_server.py    # HTTP lifecycle, REST API, and SSE stream
├── MANUAL_DE_USO.md                # In-depth operational user manual (Portuguese)
└── README.md                       # Main documentation & quickstart (English)
```

---

## 🚀 Ready-to-Run Tasks Catalog

| Task | Script | Description |
| :--- | :--- | :--- |
| **Task 1** | `examples/01_benchmark_latency.py` | CPU latency verification (1000 steps, P50/P99, budget $\le 0.8\text{ ms}$). |
| **Task 2** | `examples/02_evaluate_cartpole.py` | Evaluation of trained `s1_cartpole.pt` checkpoint on CartPole-v1. |
| **Task 3** | `examples/03_train_cartpole.py` | Train agent from scratch using Recurrent PPO with chunked BPTT ($T=16$). |
| **Task 4** | `examples/04_transfer_learning_acrobot.py` | Transfer trunk weights to Acrobot-v1 with `freeze_trunk=True`. |
| **Task 5** | `examples/05_continuous_action_pendulum.py` | Continuous control on Pendulum-v1 via `GaussianPolicyHead`. |
| **Task 6** | `examples/06_visual_observation_impala.py` | Visual perception with IMPALA CNN ($2 \times C, 84, 84 \to \mathbb{R}^{320}$). |
| **Task 7** | `examples/07_run_all_tasks.py` | Batch sequential execution of all operational tasks with status summary. |
| **Task 8** | `examples/08_adapters_three_levels.py` | Demonstration of WindowCapture, MemoryHook, and NativeEngine adapters. |
| **Task 9** | `examples/09_confidence_gating.py` | Uncertainty measurement, entropy, and System 2 arbitration trigger. |
| **Task 10** | `examples/10_vizdoom_visual_transfer.py` | Visual transfer test on ViZDoom (`basic.cfg`) comparing transfer vs scratch. |
| **Task 11** | `examples/11_live_telemetry_dashboard.py` | High-frequency telemetry with terminal Rich multi-panel dashboard. |
| **Task 12** | `examples/12_web_telemetry_streaming.py` | HTTP/SSE real-time web telemetry streaming dashboard (`localhost:8050`). |

---

## 💻 CLI Usage

The CLI module provides a comprehensive command-line interface:

### 1. Benchmark CPU Latency
```bash
python -m system1_engine.cli --mode benchmark --steps 1000
```

### 2. Train on CartPole-v1
```bash
# Standard training
python -m system1_engine.cli --mode train --env CartPole-v1 --steps 40000 --save s1_cartpole.pt --target-return 475.0

# With live terminal dashboard
python -m system1_engine.cli --mode train --env CartPole-v1 --steps 40000 --live-stats

# With real-time web panel on port 8050 (opens browser automatically)
python -m system1_engine.cli --mode train --env CartPole-v1 --steps 40000 --web-panel --port 8050

# Without auto-opening the browser (headless/remote)
python -m system1_engine.cli --mode train --env CartPole-v1 --steps 40000 --web-panel --no-browser
```

### 3. Evaluate Pretrained Checkpoint
```bash
# Standard evaluation
python -m system1_engine.cli --mode run --env CartPole-v1 --load s1_cartpole.pt --episodes 5

# With live terminal telemetry
python -m system1_engine.cli --mode run --env CartPole-v1 --load s1_cartpole.pt --live-stats

# With real-time web dashboard (opens browser automatically, paces at 50 FPS & stays alive until Ctrl+C)
python -m system1_engine.cli --mode run --env CartPole-v1 --load s1_cartpole.pt --web-panel --port 8050
```

---

## 🧪 Test Suite

Run the full automated test suite:

```bash
pytest tests/ -v
```

**All 32 unit tests pass in ~24s**:
1. `test_confidence_gating_discrete`: Uncertainty and gating in discrete space.
2. `test_confidence_gating_continuous`: Uncertainty and gating in continuous Box space.
3. `test_confidence_gating_continuous_low_sigma`: Numerical stability and strictly positive uncertainty for $\sigma < 0.242$ ($H < 0$).
4. `test_confidence_gating_continuous_extreme_small_sigma`: Defensive clamping for extreme small std ($\sigma \approx 3 \times 10^{-7}$).
5. `test_confidence_gating_continuous_high_sigma`: Trigger activation for high dispersion ($\sigma = 2.0$).
6. `test_act_fast_latency_with_confidence`: Confidence calculation latency overhead (< 0.8 ms).
7. `test_livestats_tracker_initialization_and_ring_buffers`: $O(1)$ ring buffer initialization and memory boundedness.
8. `test_livestats_tracker_record_inference_overhead`: Verifies insertion overhead $< 2.0\text{ µs}$ (measured ~96 ns).
9. `test_livestats_tracker_env_step_and_fps`: Episode transitions, return accumulation, and physical FPS.
10. `test_livestats_tracker_training_epoch_and_snapshot`: Statistical percentiles (P50/P99) and module gradient norms.
11. `test_s1_live_dashboard_generate_view_and_render`: 3-phase Rich layout tree rendering.
12. `test_trainer_integration_with_telemetry`: Automatic telemetry population during PPO training.
13. `test_telemetry_server_lifecycle_and_http_get`: Server lifecycle and HTTP GET on `/`.
14. `test_telemetry_server_api_metrics_endpoint`: REST `/api/metrics` endpoint returning JSON snapshot.
15. `test_telemetry_server_sse_stream`: Non-blocking SSE streaming on `/stream` (`data: {...}\n\n`).
16. `test_telemetry_server_context_manager_and_idempotency`: Context manager usage (`with server:`) and clean shutdown.
17. `test_window_capture_env_mock_and_agent_pipeline`: Level 1 adapter with frame pacing and $(2 \times C, 84, 84)$ preprocessing.
18. `test_memory_hook_env_vector_mode`: Level 2 adapter with RAM offsets for score and terminal conditions.
19. `test_memory_hook_env_hybrid_mode`: Level 2 hybrid mode (visual pixels + RAM reward).
20. `test_native_engine_env_lock_step_and_speed`: Level 3 lock-step engine throughput (> 5,000 FPS).
21. `test_native_engine_vizdoom_real`: Direct lock-step integration with native ViZDoom binary.
22. `test_factory_make_game_env`: `make_game_env` factory validation across all 3 levels.
23. `test_cartpole_convergence`: Proves PPO converges to score $\ge 475.0$ in $< 40,000$ steps.
24. `test_vector_frontend_dimensions`: Rigid $\mathbb{R}^{337}$ input bus dimensional validation.
25. `test_visual_frontend_dimensions`: IMPALA convolutional front-end $\mathbb{R}^{337}$ bus validation.
26. `test_system1_trunk_dimensions_and_hx`: GRU latent state continuity and shape correctness.
27. `test_parameter_counts_and_memory`: Parameter counts and memory budget strictly $< 35\text{ MB}$.
28. `test_act_fast_latency_budget`: Discrete action CPU latency budget $\le 0.8\text{ ms}$.
29. `test_act_fast_continuous_action`: Continuous action CPU latency budget $\le 0.8\text{ ms}$.
30. `test_transfer_without_catastrophic_forgetting`: Strict trunk freezing and parameter immutabilidade ($\Delta = 0.0000$).
31. `test_wrapper_reset_robustness`: Temporal buffer flushing and cross-trajectory state isolation.
32. `test_wrapper_delta_computation`: Causal computation of $\Delta s$, $a_{t-1}$, and $r_{t-1}$.

---

## 📖 In-Depth Portuguese Manual

For exhaustive architectural specifications, mathematical foundations, implementation notes, and code recipes in Portuguese, refer to:
👉 **[`MANUAL_DE_USO.md`](MANUAL_DE_USO.md)**
