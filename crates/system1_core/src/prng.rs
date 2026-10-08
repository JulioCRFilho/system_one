// Fast, zero-dependency, deterministic 64-bit PRNG (SplitMix64)
// Perfectly compatible with both native x86/ARM64 and WebAssembly

#[derive(Debug, Clone)]
pub struct FastRng {
    state: u64,
}

impl FastRng {
    #[inline]
    pub fn new(seed: u64) -> Self {
        let mut rng = Self { state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } };
        // Warm up state
        rng.next_u64();
        rng
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // [0.0, 1.0)
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    #[inline]
    pub fn gen_range(&mut self, upper: usize) -> usize {
        if upper == 0 {
            return 0;
        }
        (self.next_u64() % (upper as u64)) as usize
    }
}
