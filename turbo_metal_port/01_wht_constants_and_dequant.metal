// ----- TurboQuant quantize/dequantize with Fast Walsh-Hadamard rotation -----
// Uses O(d log d) WHT instead of O(d²) dense matvec (18× fewer operations)
// 512 bytes of sign arrays instead of 256KB of dense matrices
// ===== INLINED turbo-wht.h =====
// TurboQuant Fast Walsh-Hadamard rotation for Metal
// Replaces 256KB dense matrices with 512 bytes of sign arrays + O(d log d) butterfly
// Generated with seed=42 (rotation) and seed=1042 (QJL)

// --- Rotation sign arrays ---
constant float turbo_wht_signs1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
constant float turbo_wht_signs2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f};

// --- Pre-packed half4 sign arrays for vectorized WHT (eliminates float→half conversion) ---
constant half4 turbo_wht_signs1_h4[32] = {
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, 1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, -1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(1.0h, 1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, -1.0h, 1.0h), half4(1.0h, 1.0h, -1.0h, -1.0h),
    half4(1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, 1.0h, 1.0h, 1.0h), half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(-1.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h), half4(1.0h, 1.0h, -1.0h, 1.0h)
};
constant half4 turbo_wht_signs2_h4[32] = {
    half4(1.0h, 1.0h, 1.0h, 1.0h), half4(-1.0h, 1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, -1.0h), half4(1.0h, -1.0h, -1.0h, -1.0h),
    half4(1.0h, 1.0h, -1.0h, -1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, 1.0h), half4(-1.0h, 1.0h, 1.0h, 1.0h),
    half4(1.0h, 1.0h, -1.0h, -1.0h), half4(-1.0h, 1.0h, -1.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h, -1.0h), half4(1.0h, 1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, -1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, -1.0h), half4(-1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h, 1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h), half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, -1.0h), half4(-1.0h, -1.0h, 1.0h, -1.0h)
};

// --- QJL sign arrays ---
constant float turbo_qjl_wht_signs1[128] = {
    1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f};
constant float turbo_qjl_wht_signs2[128] = {
    1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f};

// --- Fast Walsh-Hadamard Transform (in-place, normalized) ---
// O(n log n) = 896 operations for n=128, vs O(n²) = 16384 for dense matvec
static void turbo_fwht_128(thread float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    // Normalize by 1/sqrt(128)
    const float inv_sqrt_128 = 0.08838834764831845f; // 1/sqrt(128)
    for (int i = 0; i < 128; i++) {
        x[i] *= inv_sqrt_128;
    }
}

// --- Forward rotation: signs1 → FWHT → signs2 ---
static void turbo_rotate_forward(thread float * x, constant float * s1, constant float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
}

// --- Inverse rotation: signs2 → FWHT → signs1 (FWHT is its own inverse) ---
static void turbo_rotate_inverse(thread float * x, constant float * s1, constant float * s2) {
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
}

// ===== END turbo-wht.h =====

// 2-bit centroids for d=128 (scaled by 1/sqrt(128))
constant float turbo_centroids_2bit[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };
// 3-bit centroids for d=128
constant float turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
// Midpoints for 2-bit nearest centroid lookup
constant float turbo_mid_2bit[3] = { -0.086728f, 0.0f, 0.086728f };
// Midpoints for 3-bit
constant float turbo_mid_3bit[7] = { -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f };

// 4-bit PolarQuant centroids (16 levels) — optimal for N(0, 1/sqrt(128))
constant float turbo_centroids_4bit[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};
constant float turbo_mid_4bit[15] = {
    -0.145560f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013963f,  0.000000f,
     0.013963f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145560f
};

// Half-precision 4-bit centroid LUT for vec path
constant half turbo_centroids_4bit_h[16] = {
    -0.173926h, -0.117195h, -0.089527h, -0.068756h,
    -0.051262h, -0.035597h, -0.020989h, -0.006938h,
     0.006938h,  0.020989h,  0.035597h,  0.051262h,
     0.068756h,  0.089527h,  0.117195h,  0.173926h
};

// 8-entry magnitude LUT for 4-bit (positive half, ascending)
// idx 8-15 are positive: mag = centroids_4bit[idx]
// idx 0-7 are negative: mag = centroids_4bit[15 - idx] with sign flip
// sign = (idx >> 3) ? +1 : -1
// mag_idx = (idx & 7) for positive, (7 - (idx & 7)) for negative — but
// since centroids are symmetric ascending, just use: mag[idx & 7] for idx>=8,
// mag[7 - (idx & 7)] for idx<8. Simpler: mag[idx >= 8 ? idx & 7 : 7 - (idx & 7)]
constant half turbo_mag_4bit_h[8] = {
    0.006938h, 0.020989h, 0.035597h, 0.051262h,
    0.068756h, 0.089527h, 0.117195h, 0.173926h
};

// Half-precision 2-bit centroid LUT for vec path
constant half turbo_centroids_2bit_h[4] = {
    -0.133462h, -0.039994h, 0.039994h, 0.133462h
};

// Quantize 32 elements into one block_turbo2_0 (NO rotation — rotation happens
// at the 128-element group level in kernel_set_rows_turbo)
void quantize_turbo2_0(device const float * src, device block_turbo2_0 & dst) {
#pragma METAL fp math_mode(safe)
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO2; j++) norm_sq += src[j] * src[j];
    float norm = sqrt(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst.norm = half(norm);

    for (int j = 0; j < QK_TURBO2 / 4; j++) dst.qs[j] = 0;

    for (int j = 0; j < QK_TURBO2; j++) {
        float val = src[j] * inv_norm;
        uint8_t idx;
        if      (val < turbo_mid_2bit[0]) idx = 0;
        else if (val < turbo_mid_2bit[1]) idx = 1;
        else if (val < turbo_mid_2bit[2]) idx = 2;
        else                              idx = 3;

        dst.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
    }
}

// Quantize 32 elements into one block_turbo3_0 (NO rotation — rotation happens
// at the 128-element group level in kernel_set_rows_turbo)
void quantize_turbo3_0(device const float * src, device block_turbo3_0 & dst) {
#pragma METAL fp math_mode(safe)
    // Compute norm for this 32-element sub-block
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3; j++) norm_sq += src[j] * src[j];
    float norm = sqrt(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst.norm = half(norm);

    // Quantize to 3-bit centroids
    for (int j = 0; j < QK_TURBO3 / 4; j++) dst.qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) dst.signs[j] = 0;

    for (int j = 0; j < QK_TURBO3; j++) {
        float val = src[j] * inv_norm;
        uint8_t idx;
        if      (val < turbo_mid_3bit[0]) idx = 0;
        else if (val < turbo_mid_3bit[1]) idx = 1;
        else if (val < turbo_mid_3bit[2]) idx = 2;
        else if (val < turbo_mid_3bit[3]) idx = 3;
        else if (val < turbo_mid_3bit[4]) idx = 4;
        else if (val < turbo_mid_3bit[5]) idx = 5;
        else if (val < turbo_mid_3bit[6]) idx = 6;
        else                              idx = 7;

        dst.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) {
            dst.signs[j / 8] |= (1 << (j % 8));
        }
    }
}

void quantize_turbo4_0(device const float * src, device block_turbo4_0 & dst) {
#pragma METAL fp math_mode(safe)
    // 4-bit PolarQuant: normalize → rotate → quantize to 16 centroids → nibble pack
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float grp_norm = sqrt(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

    for (int j = 0; j < QK_TURBO4 / 2; j++) dst.qs[j] = 0;

    float recon_norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        float val = x[j];
        uint8_t idx;
        if      (val < turbo_mid_4bit[ 0]) idx = 0;
        else if (val < turbo_mid_4bit[ 1]) idx = 1;
        else if (val < turbo_mid_4bit[ 2]) idx = 2;
        else if (val < turbo_mid_4bit[ 3]) idx = 3;
        else if (val < turbo_mid_4bit[ 4]) idx = 4;
        else if (val < turbo_mid_4bit[ 5]) idx = 5;
        else if (val < turbo_mid_4bit[ 6]) idx = 6;
        else if (val < turbo_mid_4bit[ 7]) idx = 7;
        else if (val < turbo_mid_4bit[ 8]) idx = 8;
        else if (val < turbo_mid_4bit[ 9]) idx = 9;
        else if (val < turbo_mid_4bit[10]) idx = 10;
        else if (val < turbo_mid_4bit[11]) idx = 11;
        else if (val < turbo_mid_4bit[12]) idx = 12;
        else if (val < turbo_mid_4bit[13]) idx = 13;
        else if (val < turbo_mid_4bit[14]) idx = 14;
        else                               idx = 15;

        dst.qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
        recon_norm_sq += turbo_centroids_4bit[idx] * turbo_centroids_4bit[idx];
    }

    dst.rnorm = half(0.0f);
    float recon_norm = sqrt(recon_norm_sq);
    dst.norm = half((recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm);
}

// ----- turbo3 dequantize with per-thread block cache -----
// The rotation requires all 128 elements. Flash attention calls dequantize
// up to 32× per block (once per 4-element chunk). We cache the full
// dequantized block per thread and only recompute when the block pointer changes.

// turbo3 dequant — full block dequantize with inverse rotation
// Must process all 128 elements to apply WHT inverse rotation
// Half-precision vectorized WHT for faster dequant.
// Uses half4 vectors for 4-wide SIMD throughput on Apple GPU.
// Centroids fit in fp16 (max |val| = 0.19), butterfly stays in range.
static void turbo_fwht_128_half4(thread half4 * v) {
    // 32 half4 vectors = 128 elements
    // Stage h=1: butterfly between elements 0,1 and 2,3 within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    }
    // Stage h=2: butterfly between elements 0,2 and 1,3 within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    }
    // Stages h=4,8,16,32,64: butterfly between half4 vectors
    for (int h = 4; h < 128; h *= 2) {
        int vec_stride = h / 4;  // distance in half4 units
        for (int i = 0; i < 32; i++) {
            int group_pos = i % (2 * vec_stride);
            if (group_pos < vec_stride) {
                int partner = i + vec_stride;
                half4 a = v[i];
                half4 b = v[partner];
                v[i]       = a + b;
                v[partner] = a - b;
            }
        }
    }
    // Normalize
    const half4 inv_sqrt_128 = half4(0.08838834764831845h);
    for (int i = 0; i < 32; i++) {
        v[i] *= inv_sqrt_128;
    }
}

// ----- turbo2 dequantize -----
// 2-bit indices (4 centroids), no signs byte. Simpler than turbo3.
// Block size 32, nl=2 for non-vec (32/16), nl=8 for vec (32/4).

// Non-vec: 16 elements per call (il ∈ {0,1}), returns type4x4
template <typename type4x4>
void dequantize_turbo2_0(device const block_turbo2_0 * xb, short il, thread type4x4 & reg) {
    const float norm = float(xb->norm);
    // il=0 → elements 0-15 (qs bytes 0-3)
    // il=1 → elements 16-31 (qs bytes 4-7)
    const int qs_off = il * 4;
    float4x4 reg_f;
    for (int g = 0; g < 4; g++) {
        const uint8_t qb = xb->qs[qs_off + g];
        reg_f[g] = float4(
            turbo_centroids_2bit[(qb      ) & 0x03] * norm,
            turbo_centroids_2bit[(qb >> 2) & 0x03] * norm,
            turbo_centroids_2bit[(qb >> 4) & 0x03] * norm,
            turbo_centroids_2bit[(qb >> 6)       ] * norm
        );
    }
    reg = (type4x4) reg_f;
}

// Vec: 4 elements per call (il ∈ {0..7}), returns type4
template <typename type4>
void dequantize_turbo2_0_t4(device const block_turbo2_0 * xb, short il, thread type4 & reg) {
    const float norm = float(xb->norm);
    // il selects which byte of qs (each byte has 4 x 2-bit values)
    const uint8_t qb = xb->qs[il];
    reg = type4(float4(
        float(turbo_centroids_2bit_h[(qb      ) & 0x03]) * norm,
        float(turbo_centroids_2bit_h[(qb >> 2) & 0x03]) * norm,
        float(turbo_centroids_2bit_h[(qb >> 4) & 0x03]) * norm,
        float(turbo_centroids_2bit_h[(qb >> 6)       ]) * norm
    ));
}

// Block-32 dequant: no WHT needed (graph handles rotation). Just centroid lookup + norm scale.
// With QK_TURBO3=32: nl=2 for non-vec FA (32/16), nl=8 for vec FA (32/4).
// Much less redundant work than block-128.

// Optimized turbo3 dequant: batch byte reads, unrolled index extraction.
// Non-vec: 16 elements per call (il ∈ {0,1}), returns type4x4
template <typename type4x4>
void dequantize_turbo3_0(device const block_turbo3_0 * xb, short il, thread type4x4 & reg) {
    const float norm = float(xb->norm);
    // il=0 → elements 0-15 (qs bytes 0-3, signs bytes 0-1)
    // il=1 → elements 16-31 (qs bytes 4-7, signs bytes 2-3)
    const int qs_off = il * 4;
    float4x4 reg_f;
    for (int g = 0; g < 4; g++) {
        // g iterates over 4 groups of 4 elements within our 16
        // element index within block: il*16 + g*4 + k, k=0..3
        const uint8_t qb = xb->qs[qs_off + g];
        // signs byte index: (il*16 + g*4) / 8 = il*2 + g/2
        const uint8_t sb = xb->signs[il * 2 + g / 2];
        const int sshift = (g & 1) * 4;

        reg_f[g] = float4(
            turbo_centroids_3bit[(qb & 0x03)        | (((sb >> (sshift + 0)) & 1) << 2)] * norm,
            turbo_centroids_3bit[((qb >> 2) & 0x03) | (((sb >> (sshift + 1)) & 1) << 2)] * norm,
            turbo_centroids_3bit[((qb >> 4) & 0x03) | (((sb >> (sshift + 2)) & 1) << 2)] * norm,
            turbo_centroids_3bit[((qb >> 6) & 0x03) | (((sb >> (sshift + 3)) & 1) << 2)] * norm
        );
    }
    reg = (type4x4) reg_f;
}

// Half-precision centroid LUT for vec path — reduces constant cache pressure at long context.
// Register LUT (cn[8] = centroid*norm in thread registers) was tested but caused register
// spill on Metal, making it slower. Constant half LUT + float norm broadcast remains the
// fastest approach on Apple Silicon. On CUDA, register LUT works better (see @spiritbuun).
constant half turbo_centroids_3bit_h[8] = {
    -0.190685h, -0.117832h, -0.065717h, -0.021460h,
     0.021460h,  0.065717h,  0.117832h,  0.190685h
};

// 4-entry magnitude LUT (positive values only, ascending order)
// Used with ALU sign application to halve constant cache divergence
constant half turbo_mag_3bit_h[4] = {
    0.021460h, 0.065717h, 0.117832h, 0.190685h
};

// 2-entry PAIR LUT: each entry is a half2 containing two adjacent magnitudes.
// Only 2 possible constant addresses per lookup (vs 4 for mag LUT, 8 for full).
// bit1 selects the pair, bit0 selects within the pair via ternary.
constant half2 turbo_mag_pairs_h[2] = {
    half2(0.021460h, 0.065717h),   // pair 0: mag indices 0,1
    half2(0.117832h, 0.190685h),   // pair 1: mag indices 2,3
};

// Vec: 4 elements per call (il ∈ {0..7}), returns type4
// Experiment: batched byte reads (ported from @spiritbuun's CUDA impl).
// Read qs + signs bytes with minimal device memory accesses.
// The signs byte covers 8 elements — read once, shift for each element.
// Constant half[8] LUT for centroid lookup (proven fastest on Apple Silicon).
template <typename type4>
void dequantize_turbo3_0_t4(device const block_turbo3_0 * xb, short il, thread type4 & reg) {
    // PROFILING MODE: controlled by TURBO_PROFILE_MODE compile flag
    // 0 = full dequant (batched extract)
    // 1 = no-op (return zeros) — decode ceiling without dequant cost
    // 2 = norm only (read norm, return constant) — isolate norm read
    // 3 = norm + qs only (skip signs) — isolate signs byte cost
    // 4 = full dequant, skip LUT (use constant centroid) — isolate LUT cost
#ifndef TURBO_PROFILE_MODE
#define TURBO_PROFILE_MODE 0
#endif

#if TURBO_PROFILE_MODE == 1
    // NO-OP: decode speed ceiling
    reg = type4(0.0f);
#elif TURBO_PROFILE_MODE == 2
    // NORM ONLY: just read norm, return it as all 4 values
    const float norm = float(xb->norm);
    reg = type4(norm);
#elif TURBO_PROFILE_MODE == 3
    // NORM + QS: read norm and qs byte, skip signs
    const float norm = float(xb->norm);
    const uint8_t qb = xb->qs[il];
    const uint8_t q0 = (qb      ) & 0x03;
    const uint8_t q1 = (qb >> 2) & 0x03;
    const uint8_t q2 = (qb >> 4) & 0x03;
    const uint8_t q3 = (qb >> 6);
    // Use qs without signs — just positive centroids
    reg = type4(float4(
        float(turbo_centroids_3bit_h[q0 + 4]),
        float(turbo_centroids_3bit_h[q1 + 4]),
        float(turbo_centroids_3bit_h[q2 + 4]),
        float(turbo_centroids_3bit_h[q3 + 4])
    ) * norm);
#elif TURBO_PROFILE_MODE == 4
    // SKIP LUT: read all bytes but use constant centroid value
    const float norm = float(xb->norm);
    const uint8_t qb = xb->qs[il];
    const uint8_t sb = xb->signs[il >> 1];
    // Pretend all elements are centroid 0 — isolates LUT indexing cost
    reg = type4(float4(float(turbo_centroids_3bit_h[0])) * norm);
#else
    // MODE 0: 4-entry magnitude LUT + ALU sign (halves constant cache divergence)
    // Only 4 possible constant addresses per lookup (vs 8 in full LUT).
    // Sign applied via select() — no branch, just conditional negate.
    // Correct sign mapping: sign=1 → +mag[qs], sign=0 → -mag[3-qs] (reversed)
    const float norm = float(xb->norm);
    const uint8_t qb = xb->qs[il];
    const uint8_t sb = xb->signs[il >> 1];
    const int sshift = (il & 1) << 2;

    const uint8_t q0 = (qb      ) & 0x03;
    const uint8_t q1 = (qb >> 2) & 0x03;
    const uint8_t q2 = (qb >> 4) & 0x03;
    const uint8_t q3 = (qb >> 6);
    const uint8_t s0 = (sb >> (sshift    )) & 1;
    const uint8_t s1 = (sb >> (sshift + 1)) & 1;
    const uint8_t s2 = (sb >> (sshift + 2)) & 1;
    const uint8_t s3 = (sb >> (sshift + 3)) & 1;

    // Auto-selected dequant path based on hardware.
    // TURBO_USE_4MAG=1 (pre-M5): 4-entry magnitude LUT + XOR sign (+38-45% on M2)
    // TURBO_USE_4MAG=0 (M5+): 8-entry full LUT (best on M5, 0.905x q8_0)
#if TURBO_USE_4MAG
    // 4-mag LUT (proven +38-45% on M2). See decode-speed-hardware-analysis.md
    // for the full 12-approach experiment log.
    const uint8_t mi0 = q0 ^ (s0 ? 0u : 0x3u);
    const uint8_t mi1 = q1 ^ (s1 ? 0u : 0x3u);
    const uint8_t mi2 = q2 ^ (s2 ? 0u : 0x3u);
    const uint8_t mi3 = q3 ^ (s3 ? 0u : 0x3u);

    const float v0 = float(turbo_mag_3bit_h[mi0]) * norm;
    const float v1 = float(turbo_mag_3bit_h[mi1]) * norm;
    const float v2 = float(turbo_mag_3bit_h[mi2]) * norm;
    const float v3 = float(turbo_mag_3bit_h[mi3]) * norm;

    reg = type4(float4(
        s0 ? v0 : -v0,
        s1 ? v1 : -v1,
        s2 ? v2 : -v2,
        s3 ? v3 : -v3
    ));
#else
    // 8-entry full LUT: best on M5 Max (0.905x q8_0, 77.4 tok/s)
    reg = type4(float4(
        float(turbo_centroids_3bit_h[q0 | (s0 << 2)]),
        float(turbo_centroids_3bit_h[q1 | (s1 << 2)]),
        float(turbo_centroids_3bit_h[q2 | (s2 << 2)]),
        float(turbo_centroids_3bit_h[q3 | (s3 << 2)])
    ) * norm);
#endif
#endif
}

// ----- turbo4 dequantize with per-thread block cache -----

static void turbo4_dequantize_full_block(device const block_turbo4_0 * xb, thread float * cache) {
    const float norm = float(xb->norm);

    // 4-bit nibble unpack — 2 elements per byte, simple and fast
    for (int j = 0; j < 128; j++) {
        uint8_t idx = (xb->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
        cache[j] = turbo_centroids_4bit[idx] * norm;
    }
}

template <typename type4x4>
void dequantize_turbo4_0(device const block_turbo4_0 * xb, short il, thread type4x4 & reg) {
    // Direct 16-element extraction — 4-bit nibble unpack
    const float norm = float(xb->norm);
    const int base = il * 16;
    float4x4 reg_f;

    for (int g = 0; g < 4; g++) {
        for (int k = 0; k < 4; k++) {
            const int j = base + g * 4 + k;
            uint8_t idx = (xb->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
            reg_f[g][k] = turbo_centroids_4bit[idx] * norm;
        }
    }
    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_turbo4_0_t4(device const block_turbo4_0 * xb, short il, thread type4 & reg) {
    // Direct 16-entry half LUT — fastest on M5 Max (constant cache not the bottleneck)
    // 8-mag LUT tested: -3% on M5 due to ternary branch overhead. Keep for M1/M2 if needed.
    const float norm = float(xb->norm);
    const device uint8_t * qs = xb->qs + il * 2;
    const uint8_t qb0 = qs[0];
    const uint8_t qb1 = qs[1];

    reg = type4(float4(
        float(turbo_centroids_4bit_h[(qb0     ) & 0xF]) * norm,
        float(turbo_centroids_4bit_h[(qb0 >> 4) & 0xF]) * norm,
        float(turbo_centroids_4bit_h[(qb1     ) & 0xF]) * norm,
        float(turbo_centroids_4bit_h[(qb1 >> 4) & 0xF]) * norm
    ));
}

template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
