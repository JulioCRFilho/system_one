/* tslint:disable */
/* eslint-disable */

export class AgentDecisionEngineWasm {
    free(): void;
    [Symbol.dispose](): void;
    decide_step(logits: Float32Array, calib: number, is_macro: boolean): object;
    constructor();
    reset(): void;
}

export class RubiksCubeCoreWasm {
    free(): void;
    [Symbol.dispose](): void;
    apply_atomic(move_name: string): void;
    apply_atomic_idx(idx: number): void;
    apply_macro(macro_name: string): void;
    apply_macro_idx(idx: number): void;
    get_aligned_count(): number;
    get_one_hot(): Float32Array;
    get_score(): number;
    get_state(): Uint8Array;
    is_solved(): boolean;
    constructor();
    reset(): void;
    scramble(depth: number, use_macros: boolean): Array<any>;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_agentdecisionenginewasm_free: (a: number, b: number) => void;
    readonly __wbg_rubikscubecorewasm_free: (a: number, b: number) => void;
    readonly agentdecisionenginewasm_decide_step: (a: number, b: number, c: number, d: number, e: number) => any;
    readonly agentdecisionenginewasm_new: () => number;
    readonly agentdecisionenginewasm_reset: (a: number) => void;
    readonly rubikscubecorewasm_apply_atomic: (a: number, b: number, c: number) => void;
    readonly rubikscubecorewasm_apply_atomic_idx: (a: number, b: number) => void;
    readonly rubikscubecorewasm_apply_macro: (a: number, b: number, c: number) => void;
    readonly rubikscubecorewasm_apply_macro_idx: (a: number, b: number) => void;
    readonly rubikscubecorewasm_get_aligned_count: (a: number) => number;
    readonly rubikscubecorewasm_get_one_hot: (a: number) => any;
    readonly rubikscubecorewasm_get_score: (a: number) => number;
    readonly rubikscubecorewasm_get_state: (a: number) => any;
    readonly rubikscubecorewasm_is_solved: (a: number) => number;
    readonly rubikscubecorewasm_new: () => number;
    readonly rubikscubecorewasm_reset: (a: number) => void;
    readonly rubikscubecorewasm_scramble: (a: number, b: number, c: number) => any;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
