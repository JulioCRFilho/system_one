pub mod permutations;
pub mod prng;
pub mod rubiks;
pub mod decision;

pub use rubiks::RubiksCubeCore;
pub use decision::{AgentDecisionEngine, ReflexDecision};
pub use prng::FastRng;

// =====================================================================
// BINDINGS PYTHON (PyO3)
// =====================================================================

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub type PyDecisionTuple = (usize, f32, f32, f32, bool, f32, Vec<usize>, Vec<f32>);

#[cfg(feature = "python")]
#[pyclass(name = "RubiksCubeCore")]
pub struct RubiksCubeCorePy {
    inner: rubiks::RubiksCubeCore,
    rng: prng::FastRng,
}

#[cfg(feature = "python")]
impl Default for RubiksCubeCorePy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl RubiksCubeCorePy {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: rubiks::RubiksCubeCore::new(),
            rng: prng::FastRng::new(0x42A5C9E1),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn apply_atomic(&mut self, move_name: &str) {
        self.inner.apply_atomic(move_name);
    }

    pub fn apply_atomic_idx(&mut self, idx: usize) {
        self.inner.apply_atomic_idx(idx);
    }

    pub fn apply_macro(&mut self, macro_name: &str) {
        self.inner.apply_macro(macro_name);
    }

    pub fn apply_macro_idx(&mut self, idx: usize) {
        self.inner.apply_macro_idx(idx);
    }

    #[pyo3(signature = (depth, use_macros, seed=None))]
    pub fn scramble(
        &mut self,
        depth: usize,
        use_macros: bool,
        seed: Option<u64>,
    ) -> Vec<String> {
        if let Some(s) = seed {
            self.rng = prng::FastRng::new(s);
        }
        self.inner.scramble(depth, use_macros, &mut self.rng)
    }

    pub fn get_aligned_count(&self) -> usize {
        self.inner.get_aligned_count()
    }

    pub fn get_score(&self) -> f32 {
        self.inner.get_score()
    }

    pub fn is_solved(&self) -> bool {
        self.inner.is_solved()
    }

    pub fn get_state(&self) -> Vec<u8> {
        self.inner.state.to_vec()
    }

    pub fn get_one_hot(&self) -> Vec<f32> {
        self.inner.get_one_hot().to_vec()
    }

    pub fn step_atomic(
        &mut self,
        action_idx: usize,
        prev_score: f32,
    ) -> (Vec<f32>, f32, bool, f32, usize) {
        self.inner.apply_atomic_idx(action_idx);
        let cur_score = self.inner.get_score();
        let delta = cur_score - prev_score;
        let mut reward = (delta * 5.0) - 0.02;
        let is_solved = self.inner.is_solved();
        if is_solved {
            reward += 10.0;
        }
        let obs = self.inner.get_one_hot().to_vec();
        (obs, reward, is_solved, cur_score, self.inner.get_aligned_count())
    }

    pub fn step_macro(
        &mut self,
        action_idx: usize,
        prev_score: f32,
    ) -> (Vec<f32>, f32, bool, f32, usize) {
        self.inner.apply_macro_idx(action_idx);
        let cur_score = self.inner.get_score();
        let delta = cur_score - prev_score;
        let mut reward = (delta * 8.0) - 0.02;
        let is_solved = self.inner.is_solved();
        if is_solved {
            reward += 15.0;
        }
        let obs = self.inner.get_one_hot().to_vec();
        (obs, reward, is_solved, cur_score, self.inner.get_aligned_count())
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "AgentDecisionEngine")]
pub struct AgentDecisionEnginePy {
    inner: decision::AgentDecisionEngine,
    rng: prng::FastRng,
}

#[cfg(feature = "python")]
impl Default for AgentDecisionEnginePy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl AgentDecisionEnginePy {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: decision::AgentDecisionEngine::new(),
            rng: prng::FastRng::new(0x7F9B1E34),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn decide_step(
        &mut self,
        logits: Vec<f32>,
        calib: f32,
        is_macro: bool,
    ) -> PyDecisionTuple {
        let dec = self.inner.decide_step(&logits, calib, is_macro, &mut self.rng);
        (
            dec.action,
            dec.confidence,
            dec.uncertainty,
            dec.margin,
            dec.is_uncertain,
            dec.entropy,
            dec.ranked_actions,
            dec.ranked_probs,
        )
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn system1_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RubiksCubeCorePy>()?;
    m.add_class::<AgentDecisionEnginePy>()?;
    Ok(())
}

// =====================================================================
// BINDINGS WEBASSEMBLY (wasm-bindgen)
// =====================================================================

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct RubiksCubeCoreWasm {
    inner: rubiks::RubiksCubeCore,
    rng: prng::FastRng,
}

#[cfg(feature = "wasm")]
impl Default for RubiksCubeCoreWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl RubiksCubeCoreWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: rubiks::RubiksCubeCore::new(),
            rng: prng::FastRng::new(0xABC12345),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn apply_atomic(&mut self, move_name: &str) {
        self.inner.apply_atomic(move_name);
    }

    pub fn apply_atomic_idx(&mut self, idx: usize) {
        self.inner.apply_atomic_idx(idx);
    }

    pub fn apply_macro(&mut self, macro_name: &str) {
        self.inner.apply_macro(macro_name);
    }

    pub fn apply_macro_idx(&mut self, idx: usize) {
        self.inner.apply_macro_idx(idx);
    }

    pub fn scramble(&mut self, depth: usize, use_macros: bool) -> js_sys::Array {
        let moves = self.inner.scramble(depth, use_macros, &mut self.rng);
        let arr = js_sys::Array::new();
        for m in moves {
            arr.push(&JsValue::from_str(&m));
        }
        arr
    }

    pub fn get_aligned_count(&self) -> usize {
        self.inner.get_aligned_count()
    }

    pub fn get_score(&self) -> f32 {
        self.inner.get_score()
    }

    pub fn is_solved(&self) -> bool {
        self.inner.is_solved()
    }

    pub fn get_state(&self) -> js_sys::Uint8Array {
        js_sys::Uint8Array::from(&self.inner.state[..])
    }

    pub fn get_one_hot(&self) -> js_sys::Float32Array {
        let one_hot = self.inner.get_one_hot();
        js_sys::Float32Array::from(&one_hot[..])
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct AgentDecisionEngineWasm {
    inner: decision::AgentDecisionEngine,
    rng: prng::FastRng,
}

#[cfg(feature = "wasm")]
impl Default for AgentDecisionEngineWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl AgentDecisionEngineWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            inner: decision::AgentDecisionEngine::new(),
            rng: prng::FastRng::new(0x9876FEDC),
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn decide_step(
        &mut self,
        logits: &[f32],
        calib: f32,
        is_macro: bool,
    ) -> js_sys::Object {
        let dec = self.inner.decide_step(logits, calib, is_macro, &mut self.rng);
        let obj = js_sys::Object::new();
        
        js_sys::Reflect::set(&obj, &JsValue::from_str("action"), &JsValue::from_f64(dec.action as f64)).unwrap();
        js_sys::Reflect::set(&obj, &JsValue::from_str("confidence"), &JsValue::from_f64(dec.confidence as f64)).unwrap();
        js_sys::Reflect::set(&obj, &JsValue::from_str("uncertainty"), &JsValue::from_f64(dec.uncertainty as f64)).unwrap();
        js_sys::Reflect::set(&obj, &JsValue::from_str("margin"), &JsValue::from_f64(dec.margin as f64)).unwrap();
        js_sys::Reflect::set(&obj, &JsValue::from_str("isUncertain"), &JsValue::from_bool(dec.is_uncertain)).unwrap();
        js_sys::Reflect::set(&obj, &JsValue::from_str("entropy"), &JsValue::from_f64(dec.entropy as f64)).unwrap();
        js_sys::Reflect::set(&obj, &JsValue::from_str("calibration"), &JsValue::from_f64(dec.calibration as f64)).unwrap();
        
        let ranked_arr = js_sys::Array::new();
        for i in 0..dec.ranked_actions.len() {
            let item = js_sys::Object::new();
            js_sys::Reflect::set(&item, &JsValue::from_str("action"), &JsValue::from_f64(dec.ranked_actions[i] as f64)).unwrap();
            js_sys::Reflect::set(&item, &JsValue::from_str("prob"), &JsValue::from_f64(dec.ranked_probs[i] as f64)).unwrap();
            ranked_arr.push(&item);
        }
        js_sys::Reflect::set(&obj, &JsValue::from_str("ranked"), &ranked_arr).unwrap();

        obj
    }
}
