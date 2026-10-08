use crate::permutations::*;
use crate::prng::FastRng;

pub const MACRO_NAMES: [&str; 12] = [
    "SEXY_MOVE_R",
    "SEXY_MOVE_R_PRIME",
    "SEXY_MOVE_L",
    "SEXY_MOVE_L_PRIME",
    "SUNE",
    "ANTI_SUNE",
    "YELLOW_CROSS",
    "YELLOW_CROSS_PRIME",
    "ROTATE_Y",
    "ROTATE_Y_PRIME",
    "U_TURN",
    "U_PRIME_TURN",
];

pub const SCRAMBLE_MACRO_NAMES: [&str; 10] = [
    "SEXY_MOVE_R",
    "SEXY_MOVE_R_PRIME",
    "SEXY_MOVE_L",
    "SEXY_MOVE_L_PRIME",
    "SUNE",
    "ANTI_SUNE",
    "YELLOW_CROSS",
    "YELLOW_CROSS_PRIME",
    "U_TURN",
    "U_PRIME_TURN",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RubiksCubeCore {
    // 54 facetas indexadas: face * 9 + (r * 3 + c), cada valor 0..5
    pub state: [u8; 54],
}

impl Default for RubiksCubeCore {
    fn default() -> Self {
        Self::new()
    }
}

impl RubiksCubeCore {
    #[inline]
    pub fn new() -> Self {
        let mut state = [0u8; 54];
        for f in 0..6 {
            for i in 0..9 {
                state[f * 9 + i] = f as u8;
            }
        }
        Self { state }
    }

    #[inline]
    pub fn reset(&mut self) {
        for f in 0..6 {
            for i in 0..9 {
                self.state[f * 9 + i] = f as u8;
            }
        }
    }

    #[inline]
    pub fn apply_perm(&mut self, perm: &[usize; 54]) {
        let old = self.state;
        for i in 0..54 {
            self.state[i] = old[perm[i]];
        }
    }

    #[inline]
    pub fn apply_atomic(&mut self, move_name: &str) {
        match move_name {
            "U" => self.apply_perm(&PERM_U),
            "U_prime" => self.apply_perm(&PERM_U_PRIME),
            "D" => self.apply_perm(&PERM_D),
            "D_prime" => self.apply_perm(&PERM_D_PRIME),
            "F" => self.apply_perm(&PERM_F),
            "F_prime" => self.apply_perm(&PERM_F_PRIME),
            "B" => self.apply_perm(&PERM_B),
            "B_prime" => self.apply_perm(&PERM_B_PRIME),
            "R" => self.apply_perm(&PERM_R),
            "R_prime" => self.apply_perm(&PERM_R_PRIME),
            "L" => self.apply_perm(&PERM_L),
            "L_prime" => self.apply_perm(&PERM_L_PRIME),
            "Y" => self.apply_perm(&PERM_Y),
            "Y_prime" => self.apply_perm(&PERM_Y_PRIME),
            _ => {}
        }
    }

    #[inline]
    pub fn apply_atomic_idx(&mut self, idx: usize) {
        match idx {
            0 => self.apply_perm(&PERM_U),
            1 => self.apply_perm(&PERM_U_PRIME),
            2 => self.apply_perm(&PERM_D),
            3 => self.apply_perm(&PERM_D_PRIME),
            4 => self.apply_perm(&PERM_F),
            5 => self.apply_perm(&PERM_F_PRIME),
            6 => self.apply_perm(&PERM_B),
            7 => self.apply_perm(&PERM_B_PRIME),
            8 => self.apply_perm(&PERM_R),
            9 => self.apply_perm(&PERM_R_PRIME),
            10 => self.apply_perm(&PERM_L),
            11 => self.apply_perm(&PERM_L_PRIME),
            _ => {}
        }
    }

    #[inline]
    pub fn apply_macro(&mut self, macro_name: &str) {
        match macro_name {
            "SEXY_MOVE_R" => {
                self.apply_perm(&PERM_R);
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_R_PRIME);
                self.apply_perm(&PERM_U_PRIME);
            }
            "SEXY_MOVE_R_PRIME" => {
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_R);
                self.apply_perm(&PERM_U_PRIME);
                self.apply_perm(&PERM_R_PRIME);
            }
            "SEXY_MOVE_L" => {
                self.apply_perm(&PERM_L_PRIME);
                self.apply_perm(&PERM_U_PRIME);
                self.apply_perm(&PERM_L);
                self.apply_perm(&PERM_U);
            }
            "SEXY_MOVE_L_PRIME" => {
                self.apply_perm(&PERM_U_PRIME);
                self.apply_perm(&PERM_L_PRIME);
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_L);
            }
            "SUNE" => {
                self.apply_perm(&PERM_R);
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_R_PRIME);
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_R);
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_R_PRIME);
            }
            "ANTI_SUNE" => {
                self.apply_perm(&PERM_R);
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_R_PRIME);
                self.apply_perm(&PERM_U_PRIME);
                self.apply_perm(&PERM_R);
                self.apply_perm(&PERM_U_PRIME);
                self.apply_perm(&PERM_R_PRIME);
            }
            "YELLOW_CROSS" => {
                self.apply_perm(&PERM_F);
                self.apply_perm(&PERM_R);
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_R_PRIME);
                self.apply_perm(&PERM_U_PRIME);
                self.apply_perm(&PERM_F_PRIME);
            }
            "YELLOW_CROSS_PRIME" => {
                self.apply_perm(&PERM_F);
                self.apply_perm(&PERM_U);
                self.apply_perm(&PERM_R);
                self.apply_perm(&PERM_U_PRIME);
                self.apply_perm(&PERM_R_PRIME);
                self.apply_perm(&PERM_F_PRIME);
            }
            "ROTATE_Y" => self.apply_perm(&PERM_Y),
            "ROTATE_Y_PRIME" => self.apply_perm(&PERM_Y_PRIME),
            "U_TURN" => self.apply_perm(&PERM_U),
            "U_PRIME_TURN" => self.apply_perm(&PERM_U_PRIME),
            _ => {}
        }
    }

    #[inline]
    pub fn apply_macro_idx(&mut self, idx: usize) {
        if idx < MACRO_NAMES.len() {
            self.apply_macro(MACRO_NAMES[idx]);
        }
    }

    #[inline]
    pub fn get_aligned_count(&self) -> usize {
        let mut count = 0;
        for f in 0..6 {
            let center = self.state[f * 9 + 4];
            for i in 0..9 {
                if self.state[f * 9 + i] == center {
                    count += 1;
                }
            }
        }
        count
    }

    #[inline]
    pub fn get_score(&self) -> f32 {
        let aligned = self.get_aligned_count();
        if aligned <= 6 {
            0.0
        } else {
            ((aligned - 6) as f32 / 48.0).clamp(0.0, 1.0)
        }
    }

    #[inline]
    pub fn is_solved(&self) -> bool {
        self.get_aligned_count() == 54
    }

    #[inline]
    pub fn get_one_hot(&self) -> [f32; 324] {
        let mut one_hot = [0.0f32; 324];
        for i in 0..54 {
            let color = self.state[i] as usize;
            if color < 6 {
                one_hot[i * 6 + color] = 1.0;
            }
        }
        one_hot
    }

    pub fn scramble(&mut self, depth: usize, use_macros: bool, rng: &mut FastRng) -> Vec<String> {
        self.reset();
        let target_depth = depth.max(1);
        let mut applied = Vec::with_capacity(target_depth);
        let mut last_act: Option<usize> = None;
        let max_attempts = target_depth.saturating_mul(10).max(100);
        let mut attempts = 0;

        if use_macros {
            while (applied.len() < target_depth || self.is_solved()) && attempts < max_attempts {
                attempts += 1;
                let mut choices = Vec::new();
                for (idx, &m) in SCRAMBLE_MACRO_NAMES.iter().enumerate() {
                    let inverse_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
                    if Some(inverse_idx) != last_act {
                        choices.push((idx, m));
                    }
                }
                if choices.is_empty() {
                    break;
                }
                let choice_idx = rng.gen_range(choices.len());
                let (act_idx, act_name) = choices[choice_idx];
                self.apply_macro(act_name);
                applied.push(act_name.to_string());
                last_act = Some(act_idx);
            }
        } else {
            while (applied.len() < target_depth || self.is_solved()) && attempts < max_attempts {
                attempts += 1;
                let mut choices = Vec::new();
                for (idx, &m) in ATOMIC_MOVE_NAMES.iter().enumerate() {
                    let inverse_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
                    if Some(inverse_idx) != last_act {
                        choices.push((idx, m));
                    }
                }
                if choices.is_empty() {
                    break;
                }
                let choice_idx = rng.gen_range(choices.len());
                let (act_idx, act_name) = choices[choice_idx];
                self.apply_atomic(act_name);
                applied.push(act_name.to_string());
                last_act = Some(act_idx);
            }
        }

        applied
    }
}
