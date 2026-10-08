use crate::prng::FastRng;

#[derive(Debug, Clone)]
pub struct ReflexDecision {
    pub action: usize,
    pub confidence: f32,
    pub uncertainty: f32,
    pub margin: f32,
    pub is_uncertain: bool,
    pub entropy: f32,
    pub calibration: f32,
    pub ranked_actions: Vec<usize>,
    pub ranked_probs: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct AgentDecisionEngine {
    pub action_history: Vec<usize>,
}

impl AgentDecisionEngine {
    pub fn new() -> Self {
        Self {
            action_history: Vec::with_capacity(8),
        }
    }

    pub fn reset(&mut self) {
        self.action_history.clear();
    }

    pub fn decide_step(
        &mut self,
        logits: &[f32],
        calib: f32,
        _is_macro: bool,
        rng: &mut FastRng,
    ) -> ReflexDecision {
        let n = logits.len();
        if n == 0 {
            return ReflexDecision {
                action: 0,
                confidence: 0.0,
                uncertainty: 1.0,
                margin: 0.0,
                is_uncertain: true,
                entropy: 0.0,
                calibration: calib,
                ranked_actions: vec![],
                ranked_probs: vec![],
            };
        }

        // 1. Softmax numericamente estável
        let mut max_logit = f32::NEG_INFINITY;
        for &l in logits {
            if l > max_logit {
                max_logit = l;
            }
        }

        let mut sum_exp = 0.0f32;
        let mut probs = vec![0.0f32; n];
        for i in 0..n {
            probs[i] = (logits[i] - max_logit).exp();
            sum_exp += probs[i];
        }
        if sum_exp > 0.0 {
            for p in &mut probs {
                *p /= sum_exp;
            }
        }

        // 2. Entropia de Shannon e Incerteza Normalizada
        let mut entropy = 0.0f32;
        for &p in &probs {
            if p > 1e-9 {
                entropy -= p * p.ln();
            }
        }
        let max_entropy = (n as f32).ln();
        let uncertainty = if max_entropy > 0.0 {
            (entropy / max_entropy).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 3. Ranqueamento das Ações
        let mut ranked: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_action = ranked[0].0;
        let confidence = ranked[0].1;
        let margin = if n > 1 {
            (ranked[0].1 - ranked[1].1).max(0.0)
        } else {
            1.0
        };

        // 4. Quebrador de Ciclos (AvoidSet)
        let mut avoid_set = std::collections::HashSet::new();
        if let Some(&last) = self.action_history.last() {
            // Em pares simétricos (atômicos ou macros), o inverso é (last ^ 1)
            let inverse = if last % 2 == 0 { last + 1 } else { last - 1 };
            avoid_set.insert(inverse);

            // Evita 2-ciclos idênticos
            if self.action_history.len() >= 2 {
                let second_last = self.action_history[self.action_history.len() - 2];
                if second_last == last {
                    avoid_set.insert(last);
                }
            }
        }

        // 5. Amostragem Estocástica Calibrada com Temperatura
        let mut chosen_action = best_action;
        if calib > 0.01 && rng.next_f32() < calib {
            let t = 1.0 + 0.5 * calib;
            let mut sum_exp_t = 0.0f32;
            let mut sampled_probs = vec![0.0f32; n];
            for i in 0..n {
                sampled_probs[i] = ((logits[i] / t) - (max_logit / t)).exp();
                sum_exp_t += sampled_probs[i];
            }
            if sum_exp_t > 0.0 {
                for p in &mut sampled_probs {
                    *p /= sum_exp_t;
                }
            }
            let rand_val = rng.next_f32();
            let mut cum = 0.0f32;
            for (i, &p) in sampled_probs.iter().enumerate() {
                cum += p;
                if rand_val <= cum || i == n - 1 {
                    chosen_action = i;
                    break;
                }
            }
        }

        // 6. Imposição Estrita do Quebrador de Ciclos
        if avoid_set.contains(&chosen_action) && ranked.len() > 1 {
            if let Some(&(alt_action, _)) = ranked.iter().find(|(act, _)| !avoid_set.contains(act)) {
                chosen_action = alt_action;
            } else {
                chosen_action = ranked[1].0;
            }
        }

        // 7. Atualização do Histórico (Janela deslizante de 8)
        self.action_history.push(chosen_action);
        if self.action_history.len() > 8 {
            self.action_history.remove(0);
        }

        let is_uncertain = uncertainty > 0.70 || confidence < 0.50;

        ReflexDecision {
            action: chosen_action,
            confidence,
            uncertainty,
            margin,
            is_uncertain,
            entropy,
            calibration: calib,
            ranked_actions: ranked.iter().map(|(a, _)| *a).collect(),
            ranked_probs: ranked.iter().map(|(_, p)| *p).collect(),
        }
    }
}
