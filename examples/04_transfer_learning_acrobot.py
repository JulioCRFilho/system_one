#!/usr/bin/env python3
"""Task 4: Transfer Learning com Tronco Congelado (CartPole -> Acrobot).

Demonstra a transferência de conhecimento entre domínios distintos:
- Origem: CartPole-v1 (Obs: 4D, Ações: 2)
- Destino: Acrobot-v1 (Obs: 6D, Ações: 3)

O System1Trunk (~719k parâmetros recorrentes) é transferido e congelado
(requires_grad = False). Apenas os adaptadores perceptivos e as cabeças
de decisão são treinados no novo cenário, eliminando o esquecimento catastrófico.
"""

from pathlib import Path
import sys

# Garante import do system1_engine mesmo se executado de qualquer diretório
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import torch

from system1_engine.core.agent import UniversalS1Agent
from system1_engine.env.wrapper import UniversalS1Wrapper
from system1_engine.training.ppo import RecurrentPPOTrainer
from system1_engine.transfer.manager import KnowledgeTransferManager


def run_transfer_learning(
    checkpoint_source: str = "s1_cartpole.pt",
    target_env_id: str = "Acrobot-v1",
) -> None:
    print("=" * 70)
    print("🧠 TASK 4: TRANSFER LEARNING COM CONGELAMENTO ESTÁTICO DE TRONCO")
    print("=" * 70)

    # 1. Configuração do Novo Cenário (Acrobot-v1: 6D obs, 3 ações)
    raw_env = gym.make(target_env_id)
    env = UniversalS1Wrapper(raw_env)

    print(f"• Ambiente de Destino : {target_env_id}")
    print(f"  - Espaço de Estados : {env.env.observation_space.shape} (vs 4D em CartPole)")
    print(f"  - Espaço de Ações   : {env.action_space.n} ações discretas (vs 2 em CartPole)")

    # 2. Instanciação do Novo Agente
    agent = UniversalS1Agent(
        obs_space=env.env.observation_space,
        action_space=env.action_space,
    )

    # 3. Transferência de Pesos Cognitivos do Tronco
    ckpt_path = Path(checkpoint_source)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint de origem '{checkpoint_source}' não encontrado!")
        sys.exit(1)

    print(f"• Carregando núcleos transferíveis de: {checkpoint_source}")
    loaded_keys = KnowledgeTransferManager.load_transferable_weights(
        agent=agent,
        checkpoint_path=checkpoint_source,
        freeze_trunk=True,
    )
    print(f"  - Tensores transferidos com sucesso: {len(loaded_keys)} chaves de 'trunk.*'")

    # Snapshot inicial dos pesos do tronco para auditoria bitwise pós-treino
    trunk_snapshot = {
        name: param.clone().detach()
        for name, param in agent.trunk.named_parameters()
    }

    # 4. Auditoria de Gradientes e Parâmetros Congelados
    frozen_params = [p for p in agent.trunk.parameters() if not p.requires_grad]
    trainable_params = [p for p in agent.parameters() if p.requires_grad]

    print("-" * 70)
    print("🔒 AUDITORIA DE CONGELAMENTO (Zero Catastrophic Forgetting):")
    print(f"  - Parâmetros do Tronco Congelados : {sum(p.numel() for p in frozen_params):,} (requires_grad = False)")
    print(f"  - Parâmetros Adaptativos Livres   : {sum(p.numel() for p in trainable_params):,} (Front-end + Heads)")
    assert len(frozen_params) == len(list(agent.trunk.parameters())), "Todos os pesos do tronco devem estar congelados!"

    # 5. Execução de Treinamento Rápido no Novo Cenário
    print("-" * 70)
    print("⚡ Treinando adaptadores no Acrobot-v1 (3 iterações de demonstração)...")
    trainer = RecurrentPPOTrainer(
        agent=agent,
        env=env,
        learning_rate=1e-3,
        rollout_steps=512,
        chunk_length=16,
        chunk_batch_size=8,
        n_epochs=2,
    )

    trainer.train(max_steps=1536, verbose=True)

    # 6. Verificação Criptográfica/Numérica Bitwise dos Pesos do Tronco
    print("-" * 70)
    print("🔬 VERIFICAÇÃO BITWISE DOS PESOS DO TRONCO PÓS-GRADIENTES:")
    for name, param in agent.trunk.named_parameters():
        initial = trunk_snapshot[name]
        diff = torch.max(torch.abs(param - initial)).item()
        assert diff == 0.0, f"Violação de congelamento! Peso '{name}' foi modificado!"
    print("✅ [CONFIRMADO] Tronco permaneceu 100% inalterado (Diferença absoluta máxima = 0.0000).")
    print("✅ Os adaptadores do front-end e cabeças aprenderam sem corromper a memória motora!")
    print("=" * 70)


if __name__ == "__main__":
    run_transfer_learning()
