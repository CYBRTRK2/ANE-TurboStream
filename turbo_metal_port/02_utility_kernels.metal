// ===== TurboQuant4 bulk dequant to fp16 (for prefill FA) =====
// Dequants turbo4 blocks → half buffer. Dispatch before f16 FA during prefill.
// Each thread processes one 128-element block.
kernel void kernel_turbo4_dequant_f16(
        device const block_turbo4_0 * src [[buffer(0)]],
        device       half           * dst [[buffer(1)]],
        constant     uint           & n_blocks [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const uint blk_idx = tgpig * ntg + tiitg;
    if (blk_idx >= n_blocks) return;

    device const block_turbo4_0 & blk = src[blk_idx];
    device half * out = dst + blk_idx * QK_TURBO4;
    const half norm_h = blk.norm;

    // 4-bit nibble unpack → centroid → scale by norm → write fp16
    for (int j = 0; j < QK_TURBO4; j += 2) {
        const uint8_t qb = blk.qs[j / 2];
        out[j    ] = turbo_centroids_4bit_h[(qb     ) & 0xF] * norm_h;
        out[j + 1] = turbo_centroids_4bit_h[(qb >> 4) & 0xF] * norm_h;
    }
}

// ===== TurboQuant Walsh-Hadamard Transform kernel =====
// O(d log d) rotation for 128-element groups. Replaces dense 128x128 matmul.
// Each thread processes one 128-element group using half4 vectorized butterfly.
// Uses the same WHT signs already defined (turbo_wht_signs1/2, turbo_wht_signs1_h4/2_h4).

kernel void kernel_turbo_wht(
        constant ggml_metal_kargs_turbo_wht & args,
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    // Each thread handles one 128-element group
    const int64_t group_idx = tgpig * ntg + tiitg;
    const int64_t n_groups = args.n_elements / 128;
    if (group_idx >= n_groups) return;

    const device float * in = src + group_idx * 128;
    device float * out = dst + group_idx * 128;

    // Load into half4 vectors for fast butterfly
    half4 v[32];
    const bool is_inverse = (args.direction == 1);

    // Apply first signs (s1 for fwd, s2 for inv)
    for (int i = 0; i < 32; i++) {
        float4 f = float4(in[i*4], in[i*4+1], in[i*4+2], in[i*4+3]);
        half4 s = is_inverse ? turbo_wht_signs2_h4[i] : turbo_wht_signs1_h4[i];
        v[i] = half4(f) * s;
    }

    // WHT butterfly (7 stages, vectorized half4)
    // h=1: within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    }
    // h=2: within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    }
    // h=4..64: between half4 vectors
    for (int h = 4; h < 128; h *= 2) {
        int vec_stride = h / 4;
        for (int i = 0; i < 32; i++) {
