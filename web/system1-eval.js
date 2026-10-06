/**
 * System 1 Engine — Standalone Web Evaluation Runtime
 * Permite que qualquer portfólio importe e execute o agente neural diretamente no navegador via ONNX Runtime Web.
 * 
 * GitHub: https://github.com/JulioCRFilho/system_one
 */

// =====================================================================
// SIMULADOR DO CUBO MÁGICO 3X3 EM JAVASCRIPT PURO
// =====================================================================

class RubiksCubeSim {
  static ATOMIC_MOVES = [
    "U", "U_prime", "D", "D_prime",
    "F", "F_prime", "B", "B_prime",
    "R", "R_prime", "L", "L_prime"
  ];

  static MACRO_ACTIONS = {
    "SEXY_MOVE_R": ["R", "U", "R_prime", "U_prime"],
    "SEXY_MOVE_L": ["L_prime", "U_prime", "L", "U"],
    "SUNE": ["R", "U", "R_prime", "U", "R", "U", "U", "R_prime"],
    "ANTI_SUNE": ["R", "U", "U", "R_prime", "U_prime", "R", "U_prime", "R_prime"],
    "T_PERM": [
      "R", "U", "R_prime", "U_prime", "R_prime", "F",
      "R", "R", "U_prime", "R_prime", "U_prime", "R", "U", "R_prime", "F_prime"
    ],
    "INSERT_EDGE_R": ["U", "R", "U_prime", "R_prime", "U_prime", "F_prime", "U", "F"],
    "INSERT_EDGE_L": ["U_prime", "L_prime", "U", "L", "U", "F", "U_prime", "F_prime"],
    "YELLOW_CROSS": ["F", "R", "U", "R_prime", "U_prime", "F_prime"],
    "ROTATE_Y": ["Y"],
    "ROTATE_Y_PRIME": ["Y_prime"],
    "U_TURN": ["U"],
    "U_PRIME_TURN": ["U_prime"]
  };
  static MACRO_NAMES = Object.keys(RubiksCubeSim.MACRO_ACTIONS);

  static INVERSE_ATOMIC = {
    "U": "U_prime", "U_prime": "U",
    "D": "D_prime", "D_prime": "D",
    "F": "F_prime", "F_prime": "F",
    "B": "B_prime", "B_prime": "B",
    "R": "R_prime", "R_prime": "R",
    "L": "L_prime", "L_prime": "L"
  };

  static INVERSE_MACROS = {
    "SUNE": "ANTI_SUNE", "ANTI_SUNE": "SUNE",
    "U_TURN": "U_PRIME_TURN", "U_PRIME_TURN": "U_TURN",
    "ROTATE_Y": "ROTATE_Y_PRIME", "ROTATE_Y_PRIME": "ROTATE_Y"
  };

  static SCRAMBLE_MACRO_NAMES = [
    "SEXY_MOVE_R", "SEXY_MOVE_L",
    "SUNE", "ANTI_SUNE",
    "T_PERM",
    "INSERT_EDGE_R", "INSERT_EDGE_L",
    "YELLOW_CROSS",
    "U_TURN", "U_PRIME_TURN"
  ];

  static PALETTE = {
    0: "#F5F5FA", // U: Branco
    1: "#FACC15", // D: Amarelo
    2: "#22C55E", // F: Verde
    3: "#3B82F6", // B: Azul
    4: "#EF4444", // R: Vermelho
    5: "#F97316"  // L: Laranja
  };

  static FACE_LAYOUT = {
    0: { x: 164, y: 46 },  // U
    5: { x: 88,  y: 122 }, // L
    2: { x: 164, y: 122 }, // F
    4: { x: 240, y: 122 }, // R
    3: { x: 316, y: 122 }, // B
    1: { x: 164, y: 198 }  // D
  };

  constructor() {
    this.permutations = this._initPermutations();
    this.scrambleSequence = [];
    this.reset();
  }

  _initPermutations() {
    // Rotações 3D matemáticas idênticas ao RubiksCubeCore
    const rotF = [ [0, 1, 0], [-1, 0, 0], [0, 0, 1] ];
    const rotB = [ [0, -1, 0], [1, 0, 0], [0, 0, 1] ];
    const rotR = [ [1, 0, 0], [0, 0, -1], [0, 1, 0] ];
    const rotL = [ [1, 0, 0], [0, 0, 1], [0, -1, 0] ];
    const rotU = [ [0, 0, -1], [0, 1, 0], [1, 0, 0] ];
    const rotD = [ [0, 0, 1], [0, 1, 0], [-1, 0, 0] ];

    const transpose = (m) => [
      [m[0][0], m[1][0], m[2][0]],
      [m[0][1], m[1][1], m[2][1]],
      [m[0][2], m[1][2], m[2][2]]
    ];

    const moves = {
      "F": { m: rotF, cond: (p) => p[2] === 1 },
      "F_prime": { m: transpose(rotF), cond: (p) => p[2] === 1 },
      "B": { m: rotB, cond: (p) => p[2] === -1 },
      "B_prime": { m: transpose(rotB), cond: (p) => p[2] === -1 },
      "R": { m: rotR, cond: (p) => p[0] === 1 },
      "R_prime": { m: transpose(rotR), cond: (p) => p[0] === 1 },
      "L": { m: rotL, cond: (p) => p[0] === -1 },
      "L_prime": { m: transpose(rotL), cond: (p) => p[0] === -1 },
      "U": { m: rotU, cond: (p) => p[1] === 1 },
      "U_prime": { m: transpose(rotU), cond: (p) => p[1] === 1 },
      "D": { m: rotD, cond: (p) => p[1] === -1 },
      "D_prime": { m: transpose(rotD), cond: (p) => p[1] === -1 },
      "Y": { m: rotU, cond: () => true },
      "Y_prime": { m: transpose(rotU), cond: () => true }
    };

    const getSlot = (pos, norm) => {
      const [x, y, z] = pos;
      const [nx, ny, nz] = norm;
      let face, r, c;
      if (ny === 1) { face = 0; r = 1 - z; c = x + 1; }
      else if (ny === -1) { face = 1; r = z + 1; c = x + 1; }
      else if (nz === 1) { face = 2; r = 1 - y; c = x + 1; }
      else if (nz === -1) { face = 3; r = 1 - y; c = 1 - x; }
      else if (nx === 1) { face = 4; r = 1 - y; c = 1 - z; }
      else if (nx === -1) { face = 5; r = 1 - y; c = z + 1; }
      return face * 9 + (r * 3 + c);
    };

    const perms = {};
    for (const [mName, { m: rotMat, cond }] of Object.entries(moves)) {
      const stickers = [];
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [c - 1, 1, 1 - r], norm: [0, 1, 0] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [c - 1, -1, r - 1], norm: [0, -1, 0] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [c - 1, 1 - r, 1], norm: [0, 0, 1] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [1 - c, 1 - r, -1], norm: [0, 0, -1] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [1, 1 - r, 1 - c], norm: [1, 0, 0] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [-1, 1 - r, c - 1], norm: [-1, 0, 0] });

      for (let i = 0; i < stickers.length; i++) stickers[i].srcSlot = getSlot(stickers[i].pos, stickers[i].norm);

      const mul = (mat, v) => [
        mat[0][0]*v[0] + mat[0][1]*v[1] + mat[0][2]*v[2],
        mat[1][0]*v[0] + mat[1][1]*v[1] + mat[1][2]*v[2],
        mat[2][0]*v[0] + mat[2][1]*v[1] + mat[2][2]*v[2]
      ];

      const perm = new Int32Array(54);
      for (const s of stickers) {
        let p = s.pos, n = s.norm;
        if (cond(p)) { p = mul(rotMat, p); n = mul(rotMat, n); }
        const destSlot = getSlot(p, n);
        perm[destSlot] = s.srcSlot;
      }
      perms[mName] = perm;
    }
    return perms;
  }

  reset() {
    this.state = new Int32Array(54);
    for (let f = 0; f < 6; f++) {
      for (let i = 0; i < 9; i++) this.state[f * 9 + i] = f;
    }
    this.lastAction = "RESET";
    this.scrambleSequence = [];
    this.steps = 0;
  }

  applyAtomic(moveName) {
    const perm = this.permutations[moveName];
    if (!perm) return;
    const nextState = new Int32Array(54);
    for (let i = 0; i < 54; i++) nextState[i] = this.state[perm[i]];
    this.state = nextState;
  }

  applyMacro(macroName) {
    const seq = RubiksCubeSim.MACRO_ACTIONS[macroName];
    if (!seq) return;
    for (const m of seq) this.applyAtomic(m);
    this.lastAction = macroName;
    this.steps++;
  }

  scrambleAtomic(depth = 3) {
    this.reset();
    const applied = [];
    let lastMove = null;
    let attempts = 0;
    const targetDepth = Math.max(1, depth);
    while ((applied.length < targetDepth || this.isSolved()) && attempts < 60) {
      attempts++;
      const candidates = RubiksCubeSim.ATOMIC_MOVES.filter(m => m !== RubiksCubeSim.INVERSE_ATOMIC[lastMove]);
      const move = candidates[Math.floor(Math.random() * candidates.length)];
      this.applyAtomic(move);
      applied.push(move);
      lastMove = move;
    }
    this.lastAction = "EMBARALHADO";
    this.scrambleSequence = applied;
    this.steps = 0;
    return applied;
  }

  scrambleMacro(depth = 2) {
    this.reset();
    const applied = [];
    let lastMove = null;
    let attempts = 0;
    const targetDepth = Math.max(1, depth);
    while ((applied.length < targetDepth || this.isSolved()) && attempts < 60) {
      attempts++;
      const candidates = RubiksCubeSim.SCRAMBLE_MACRO_NAMES.filter(m => m !== RubiksCubeSim.INVERSE_MACROS[lastMove]);
      const move = candidates[Math.floor(Math.random() * candidates.length)];
      this.applyMacro(move);
      applied.push(move);
      lastMove = move;
    }
    this.lastAction = "EMBARALHADO";
    this.scrambleSequence = applied;
    this.steps = 0;
    return applied;
  }

  scramble(depth = 3, mode = "atomic") {
    if (mode === "macro") {
      return this.scrambleMacro(depth || 2);
    }
    return this.scrambleAtomic(depth || 3);
  }

  getAlignedCount() {
    let count = 0;
    for (let f = 0; f < 6; f++) {
      const center = this.state[f * 9 + 4];
      for (let i = 0; i < 9; i++) {
        if (this.state[f * 9 + i] === center) count++;
      }
    }
    return count;
  }

  getScore() {
    const aligned = this.getAlignedCount();
    return Math.max(0.0, Math.min(1.0, (aligned - 6) / 48.0));
  }

  isSolved() {
    return this.getAlignedCount() === 54;
  }

  getOneHot() {
    const arr = new Float32Array(324);
    for (let i = 0; i < 54; i++) {
      const color = this.state[i];
      arr[i * 6 + color] = 1.0;
    }
    return arr;
  }

  render(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Fundo Cyber Dark
    ctx.fillStyle = '#0B0F19';
    ctx.fillRect(0, 0, width, height);

    // Moldura
    ctx.strokeStyle = '#1F2937';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, width - 12, height - 12);

    // Cabeçalho
    const aligned = this.getAlignedCount();
    const scorePct = ((aligned - 6) / 48 * 100).toFixed(1);
    ctx.fillStyle = '#38BDF8';
    ctx.font = 'bold 12px monospace';
    ctx.fillText("⚡ SYSTEM 1 HUD | CUBO MÁGICO 3X3", 20, 26);

    ctx.fillStyle = '#94A3B8';
    ctx.font = '11px monospace';
    let statusLine = `Passos: ${this.steps} | Alinhamento: ${aligned}/54 (${scorePct}%)`;
    if (this.scrambleSequence && this.scrambleSequence.length > 0 && this.steps === 0) {
      statusLine += ` | Scramble: ${this.scrambleSequence.join(" ")}`;
    }
    ctx.fillText(statusLine, 20, 42);

    // Desenho das 6 faces em Cruz
    for (const [fIdxStr, { x: fx, y: fy }] of Object.entries(RubiksCubeSim.FACE_LAYOUT)) {
      const fIdx = parseInt(fIdxStr);
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const stickerIdx = fIdx * 9 + (r * 3 + c);
          const colId = this.state[stickerIdx];
          ctx.fillStyle = RubiksCubeSim.PALETTE[colId] || '#FFFFFF';
          const x1 = fx + c * 24;
          const y1 = fy + r * 24;
          ctx.fillRect(x1, y1, 22, 22);
          ctx.strokeStyle = '#0F172A';
          ctx.lineWidth = 1;
          ctx.strokeRect(x1, y1, 22, 22);
        }
      }
    }

    // Rodapé
    ctx.font = 'bold 11px monospace';
    ctx.fillStyle = this.isSolved() ? '#4ADE80' : '#E2E8F0';
    ctx.fillText(this.isSolved() ? "STATUS: RESOLVIDO 🏆" : `AÇÃO ATIVA: ${this.lastAction}`, 20, height - 16);
  }
}

// =====================================================================
// SIMULADOR CARTPOLE-V1 EM JAVASCRIPT PURO
// =====================================================================

class CartPoleSim {
  constructor() {
    this.gravity = 9.8;
    this.masscart = 1.0;
    this.masspole = 0.1;
    this.total_mass = this.masscart + this.masspole;
    this.length = 0.5;
    this.polemass_length = this.masspole * this.length;
    this.force_mag = 10.0;
    this.tau = 0.02;
    this.reset();
  }

  reset() {
    this.x = (Math.random() - 0.5) * 0.1;
    this.x_dot = (Math.random() - 0.5) * 0.1;
    this.theta = (Math.random() - 0.5) * 0.1;
    this.theta_dot = (Math.random() - 0.5) * 0.1;
    this.steps = 0;
  }

  getState() {
    return new Float32Array([this.x, this.x_dot, this.theta, this.theta_dot]);
  }

  step(action) {
    const force = action === 1 ? this.force_mag : -this.force_mag;
    const costh = Math.cos(this.theta);
    const sinth = Math.sin(this.theta);

    const temp = (force + this.polemass_length * this.theta_dot * this.theta_dot * sinth) / this.total_mass;
    const thetaacc = (this.gravity * sinth - costh * temp) / (this.length * (4.0 / 3.0 - this.masspole * costh * costh / this.total_mass));
    const xacc = temp - this.polemass_length * thetaacc * costh / this.total_mass;

    this.x += this.tau * this.x_dot;
    this.x_dot += this.tau * xacc;
    this.theta += this.tau * this.theta_dot;
    this.theta_dot += this.tau * thetaacc;
    this.steps++;

    const done = Math.abs(this.x) > 2.4 || Math.abs(this.theta) > 0.2095;
    return { done, steps: this.steps };
  }

  render(canvas) {
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#0B0F19';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#1F2937';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, width - 12, height - 12);

    ctx.fillStyle = '#38BDF8';
    ctx.font = 'bold 12px monospace';
    ctx.fillText("⚡ SYSTEM 1 HUD | CARTPOLE-V1", 20, 26);
    ctx.fillStyle = '#94A3B8';
    ctx.font = '11px monospace';
    ctx.fillText(`Passos Equilibrados: ${this.steps}`, 20, 42);

    // Linha do solo
    const groundY = height - 80;
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(20, groundY);
    ctx.lineTo(width - 20, groundY);
    ctx.stroke();

    // Carrinho
    const scale = width / 4.8;
    const cartX = width / 2 + this.x * scale;
    const cartW = 60, cartH = 30;
    ctx.fillStyle = '#38BDF8';
    ctx.fillRect(cartX - cartW/2, groundY - cartH, cartW, cartH);

    // Haste
    const poleLen = 90;
    const poleEndX = cartX + Math.sin(this.theta) * poleLen;
    const poleEndY = (groundY - cartH) - Math.cos(this.theta) * poleLen;
    ctx.strokeStyle = '#F59E0B';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cartX, groundY - cartH);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();
  }
}

// =====================================================================
// RUNTIME DO AGENTE SYSTEM 1 (ONNX RUNTIME WEB)
// =====================================================================

class System1AgentWeb {
  constructor(session) {
    this.session = session;
    this.hx = new Float32Array(256);
    this.prevObs = null;
    this.prevAction = 0;
    this.prevReward = 0.0;
    this.dynamicCalibration = 0.0;
    this.stagnationCount = 0;
    this.rewardEma = 0.0;
  }

  resetMemory() {
    this.hx.fill(0.0);
    this.prevObs = null;
    this.prevAction = 0;
    this.prevReward = 0.0;
    this.dynamicCalibration = 0.0;
    this.stagnationCount = 0;
    this.rewardEma = 0.0;
  }

  async actWithConfidence(obsArr, prevReward = 0.0, options = {}) {
    const obsDim = obsArr.length;
    let deltaArr = new Float32Array(obsDim);
    if (this.prevObs) {
      for (let i = 0; i < obsDim; i++) deltaArr[i] = obsArr[i] - this.prevObs[i];
    }
    this.prevObs = new Float32Array(obsArr);
    this.prevReward = typeof prevReward === 'number' ? prevReward : 0.0;

    const t0 = performance.now();

    const tObs = new ort.Tensor('float32', obsArr, [1, 1, obsDim]);
    const tDelta = new ort.Tensor('float32', deltaArr, [1, 1, obsDim]);
    const tAct = new ort.Tensor('int64', BigInt64Array.from([BigInt(this.prevAction)]), [1, 1]);
    const tRew = new ort.Tensor('float32', new Float32Array([this.prevReward]), [1, 1, 1]);
    const tHx = new ort.Tensor('float32', this.hx, [1, 1, 256]);

    const feeds = {
      obs: tObs,
      delta_obs: tDelta,
      prev_action: tAct,
      prev_reward: tRew,
      hx: tHx,
    };

    const results = await this.session.run(feeds);
    const latencyMs = performance.now() - t0;

    const logits = results.logits.data;
    this.hx.set(results.next_hx.data);

    // Softmax & Entropia de Shannon (Gating nos logits originais)
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }

    let sumExp = 0;
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += probs[i];
    }

    let entropy = 0;
    let bestAction = 0;
    let maxProb = 0;
    const ranked = [];
    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumExp;
      ranked.push({ action: i, prob: probs[i], logit: logits[i] });
      if (probs[i] > maxProb) { maxProb = probs[i]; bestAction = i; }
      if (probs[i] > 1e-9) entropy -= probs[i] * Math.log(probs[i]);
    }
    ranked.sort((a, b) => b.prob - a.prob);

    // Calibração Contínua de Ação [0.0 = Determinístico/argmax, 1.0 = Estocástico Total] ou Auto-Calibração Homeostática
    let calib = 0.5;
    if (options.autoCalibrate || options.calibration === 'auto') {
      const r = this.prevReward;
      const deltaR = r - this.rewardEma;
      this.rewardEma = 0.9 * this.rewardEma + 0.1 * r;

      if (r > 0.01 || deltaR > 0.01) {
        this.stagnationCount = 0;
        this.dynamicCalibration = Math.max(0.0, this.dynamicCalibration - 0.20);
      } else {
        this.stagnationCount++;
        if (this.stagnationCount >= 2) {
          this.dynamicCalibration = Math.min(0.80, this.dynamicCalibration + 0.10);
        }
      }
      calib = this.dynamicCalibration;
    } else {
      const calibration = options.calibration !== undefined 
        ? options.calibration 
        : (options.temperature !== undefined ? options.temperature : (options.deterministic ? 0.0 : 0.5));
      calib = Math.min(1.0, Math.max(0.0, typeof calibration === 'number' ? calibration : 0.5));
    }

    let chosenAction = bestAction;
    if (calib > 0.01) {
      const T = Math.max(0.05, calib);
      let maxScaled = -Infinity;
      for (let i = 0; i < logits.length; i++) {
        if (logits[i] / T > maxScaled) maxScaled = logits[i] / T;
      }
      let sumExpT = 0;
      const sampledProbs = new Float32Array(logits.length);
      for (let i = 0; i < logits.length; i++) {
        sampledProbs[i] = Math.exp((logits[i] / T) - maxScaled);
        sumExpT += sampledProbs[i];
      }
      for (let i = 0; i < logits.length; i++) sampledProbs[i] /= sumExpT;

      const r = Math.random();
      let cum = 0;
      for (let i = 0; i < sampledProbs.length; i++) {
        cum += sampledProbs[i];
        if (r <= cum || i === sampledProbs.length - 1) {
          chosenAction = i;
          break;
        }
      }
    }

    // Fallback defensivo de cycle breaking se avoidAction for especificado explicitamente
    if (options.avoidAction !== undefined && options.avoidAction === chosenAction && ranked.length > 1) {
      chosenAction = ranked[1].action;
    }

    const maxEntropy = Math.log(logits.length);
    const uncertainty = Math.min(1.0, Math.max(0.0, entropy / maxEntropy));

    this.prevAction = chosenAction;

    return {
      action: chosenAction,
      confidence: maxProb,
      uncertainty: uncertainty,
      calibration: calib,
      isUncertain: uncertainty > 0.70 || maxProb < 0.50,
      latencyMs: latencyMs,
      ranked: ranked,
      value: results.value ? results.value.data[0] : 0.0
    };
  }
}

// Expõe globalmente para uso no portfólio
window.System1 = {
  RubiksCubeSim,
  CartPoleSim,
  System1AgentWeb,
};
