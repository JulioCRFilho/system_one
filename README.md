# Universal System 1 RL Agent Engine

Modular, lightweight Python/PyTorch library implementing an amortized reflex agent trained via pure Recurrent Proximal Policy Optimization (PPO).

Unlike deliberative "System 2" frameworks (LLMs with Chain-of-Thought, tree search), this agent operates by **amortized reflex**: executing a single deterministic forward pass with sub-millisecond latency on CPU (`act_fast() <= 0.8 ms`).

---

## Key Characteristics

1. **Extreme Lightweightness**:
   - **Vector mode**: ~725k parameters (~2.8 MB in float32, < 35 MB RAM).
   - **Visual mode**: ~2.02M parameters (~8.1 MB in float32, < 35 MB RAM).
2. **Transfer without Catastrophic Forgetting**:
   - Strict decoupling between perception front-ends, universal reflexive trunk (`System1Trunk`), and action/value heads.
   - Cross-scenario transfer freezes the recurrent trunk (`requires_grad = False`) and trains only new front-end adapters and heads.
3. **Resilience to Cyclic Attractors**:
   - Causal state augmentation $(s_t, \Delta s_t, a_{t-1}, r_{t-1})$.
   - Recurrence in latent space via `nn.GRU` and residual blocks (`ResMLPBlock`).
   - Temporal masking on episode boundaries prevents cross-trajectory state leakage.

---

## Unified Tensor Flow Architecture

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

## Directory Structure

```
system1_engine/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── encoders.py       # VectorFrontEnd, ImpalaVisualFrontEnd, Action/Reward Adapters
│   ├── trunk.py          # System1Trunk (nn.GRU + ResMLP)
│   ├── heads.py          # CategoricalPolicyHead, GaussianPolicyHead, ValueHead
│   └── agent.py          # UniversalS1Agent unificado
├── env/
│   ├── __init__.py
│   ├── wrapper.py        # UniversalS1Wrapper com buffers delta e meta
│   └── adapters/         # Camada de 3 Níveis de Integração
│       ├── __init__.py
│       ├── base.py       # BaseGameAdapter (gym.Env comum)
│       ├── window_adapter.py  # Nível 1: WindowCaptureEnv (mss + pynput)
│       ├── memory_adapter.py  # Nível 2: MemoryHookEnv (RAM Offsets + Híbrido)
│       ├── native_adapter.py  # Nível 3: NativeEngineEnv (Lock-Step Headless >10k FPS)
│       └── factory.py         # make_game_env(...)
├── training/
│   ├── __init__.py
│   ├── buffer.py         # RecurrentRolloutBuffer (Chunks para BPTT)
│   └── ppo.py            # RecurrentPPOTrainer com suporte a GRU e masking
├── transfer/
│   ├── __init__.py
│   └── manager.py        # KnowledgeTransferManager (isolamento e freeze)
└── cli.py                # Entrypoint CLI (train, run, benchmark)
tests/
├── test_dimensions.py    # Validação de tensores e shapes
├── test_wrapper.py       # Testes de isolamento de episódios e deltas
├── test_latency.py       # Benchmark de latência CPU (<= 0.8 ms)
├── test_transfer.py      # Transferência com congelamento estrito de tronco
├── test_convergence.py   # Convergência comprovada em CartPole-v1 (>= 475)
└── test_adapters.py      # Validação dos 3 níveis de adaptadores e factory
```

---

## CLI Usage

### 1. Benchmark CPU Inference Latency
```bash
python -m system1_engine.cli --mode benchmark --steps 1000
```
Output:
```
=== Running act_fast() CPU Latency Benchmark ===
Benchmark over 1000 sequential steps:
  Average Latency : 0.1506 ms (Budget: <= 0.8000 ms)
  Median Latency  : 0.1480 ms
  P95 Latency     : 0.1647 ms
  P99 Latency     : 0.1853 ms
[PASS] Latency budget satisfied (<= 0.8 ms).
```

### 2. Train on CartPole-v1
```bash
python -m system1_engine.cli --mode train --env CartPole-v1 --steps 40000 --save s1_cartpole.pt --target-return 475.0
```

### 3. Evaluate Checkpoint
```bash
python -m system1_engine.cli --mode run --env CartPole-v1 --load s1_cartpole.pt --episodes 5
```

---

## Running the Test Suite

```bash
pytest tests/ -v
```

All 16 tests pass:
- **3 Integration Levels**: Full validation for WindowCaptureEnv, MemoryHookEnv (Vector & Hybrid), and NativeEngineEnv (Lock-step > 5,000 FPS).
- **Dimensionality validation**: Exact `[B, T, 337]` input and `[B, T, 256]` latent representations.
- **CartPole-v1 Convergence**: Reaches moving average score of **491.90** in < 40,000 steps.
- **Zero Catastrophic Forgetting**: Trunk weights transferred to Acrobot-v1 remain 100% frozen and bitwise identical.
- **CPU Latency**: 0.15 ms per step (5x below the 0.8 ms limit).
- **Environment Wrapper Flush**: Zero residual leakage across episode resets.
