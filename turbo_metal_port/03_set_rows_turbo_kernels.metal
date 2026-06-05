kernel void kernel_set_rows_q32(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;

    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;

    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) {
        return;
    }

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float   * src_row = (const device float   *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    for (int ind = tiitg%tptg.x; ind < args.nk0; ind += tptg.x) {
        quantize_func(src_row + 32*ind, dst_row[ind]);
    }
}

// TurboQuant set_rows kernel — block size 128 (QK_TURBO3/QK_TURBO4)
// TurboQuant SET_ROWS kernel — processes QK_TURBO3_GROUP (128) elements per iteration,
// writes QK_TURBO3_GROUP/QK_TURBO3 (4) blocks per iteration.
// The rotation operates on 128 elements, then results are split into 32-element blocks.
template<typename TI, typename block_q, int QK, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_turbo(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float   * src_row = (const device float   *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Process in groups of 4 blocks (128 elements) for rotation
    const int blocks_per_group = QK_TURBO3_GROUP / QK;  // 128/32 = 4
    const int n_groups = args.nk0 / blocks_per_group;

    for (int grp = tiitg%tptg.x; grp < n_groups; grp += tptg.x) {
        const device float * grp_src = src_row + QK_TURBO3_GROUP * grp;

        // Normalize and rotate the full 128-element group
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO3_GROUP; j++) norm_sq += grp_src[j] * grp_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = grp_src[j] * inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        // Split into 4 blocks of 32 elements each
        // All blocks store the SAME group norm — centroids are in normalized space
        // Norm correction (ported from @spiritbuun's CUDA implementation):
        // Accumulate ||centroid_vector||^2 across all 128 elements, then store
        // grp_norm / ||centroid_vector|| instead of raw grp_norm. This makes
        // dequantized vectors have the exact original L2 norm at zero decode cost.
        float recon_norm_sq = 0.0f;

        for (int b = 0; b < blocks_per_group; b++) {
            device block_q & blk = dst_row[grp * blocks_per_group + b];
            const int off = b * QK;

            for (int j = 0; j < QK / 4; j++) blk.qs[j] = 0;
            for (int j = 0; j < QK / 8; j++) blk.signs[j] = 0;

            // Quantize rotated values to 3-bit centroids
            for (int j = 0; j < QK; j++) {
                float rv = x[off + j];  // rotated, normalized value
                uint8_t idx;
                if      (rv < turbo_mid_3bit[0]) idx = 0;
                else if (rv < turbo_mid_3bit[1]) idx = 1;
                else if (rv < turbo_mid_3bit[2]) idx = 2;
                else if (rv < turbo_mid_3bit[3]) idx = 3;
                else if (rv < turbo_mid_3bit[4]) idx = 4;
                else if (rv < turbo_mid_3bit[5]) idx = 5;
                else if (rv < turbo_mid_3bit[6]) idx = 6;
                else                              idx = 7;

                blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) blk.signs[j / 8] |= (1 << (j % 8));

                // Accumulate centroid reconstruction norm for norm correction
                float c = turbo_centroids_3bit[idx];
                recon_norm_sq += c * c;
            }
        }

        // Norm correction: store corrected norm so dequant(x) has exact original L2 norm.
        // Zero decode cost — dequant already multiplies by stored norm.
        float recon_norm = sqrt(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            dst_row[grp * blocks_per_group + b].norm = half(corrected_norm);
        }
    }
}

// TurboQuant2 SET_ROWS kernel — 2-bit PolarQuant, 4 centroids, no signs byte.
// Same 128-element group WHT rotation as turbo3, but simpler quantization.
template<typename TI>
kernel void kernel_set_rows_turbo2(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo2_0 * dst_row = (      device block_turbo2_0 *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float           * src_row = (const device float           *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Process in groups of 4 blocks (128 elements) for rotation
    const int blocks_per_group = QK_TURBO2_GROUP / QK_TURBO2;  // 128/32 = 4
    const int n_groups = args.nk0 / blocks_per_group;

    for (int grp = tiitg%tptg.x; grp < n_groups; grp += tptg.x) {
        const device float * grp_src = src_row + QK_TURBO2_GROUP * grp;

        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO2_GROUP; j++) norm_sq += grp_src[j] * grp_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = grp_src[j] * inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        float recon_norm_sq = 0.0f;

        for (int b = 0; b < blocks_per_group; b++) {
            device block_turbo2_0 & blk = dst_row[grp * blocks_per_group + b];
            const int off = b * QK_TURBO2;

            for (int j = 0; j < QK_TURBO2 / 4; j++) blk.qs[j] = 0;

            for (int j = 0; j < QK_TURBO2; j++) {
                float rv = x[off + j];
                uint8_t idx;
                if      (rv < turbo_mid_2bit[0]) idx = 0;
                else if (rv < turbo_mid_2bit[1]) idx = 1;
                else if (rv < turbo_mid_2bit[2]) idx = 2;
                else                              idx = 3;

                blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);

                float c = turbo_centroids_2bit[idx];
                recon_norm_sq += c * c;
            }
        }

        float recon_norm = sqrt(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            dst_row[grp * blocks_per_group + b].norm = half(corrected_norm);
        }
    }
}

// TurboQuant4 SET_ROWS kernel — processes 128 elements per block.
// Unlike turbo3 (4x32 blocks), turbo4 uses single 128-element blocks
// with packed 3-bit indices and QJL signs.
template<typename TI>
kernel void kernel_set_rows_turbo4(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo4_0 * dst_row = (      device block_turbo4_0 *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float           * src_row = (const device float         *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Each block is one 128-element group (nk0 = ne0 / QK_TURBO4)
    const int n_blocks = args.nk0;

    for (int blk_idx = tiitg%tptg.x; blk_idx < n_blocks; blk_idx += tptg.x) {
        const device float * blk_src = src_row + QK_TURBO4 * blk_idx;
        device block_turbo4_0 & blk = dst_row[blk_idx];

        // Step 1: Compute norm + normalize
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) norm_sq += blk_src[j] * blk_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        float normalized[128];
        for (int j = 0; j < 128; j++) {
            normalized[j] = blk_src[j] * inv_norm;
            x[j] = normalized[j];
        }

        // Step 2: WHT rotate in-place
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        // Step 3: 4-bit PolarQuant — nibble packing (2 indices per byte)
        for (int j = 0; j < QK_TURBO4 / 2; j++) blk.qs[j] = 0;
        // qs[64] covers full nibble range — no signs field in 4-bit struct

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

            // 4-bit nibble pack: 2 elements per byte
            blk.qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);

            float c = turbo_centroids_4bit[idx];
            recon_norm_sq += c * c;
        }

        blk.rnorm = half(0.0f);  // reserved field, unused in 4-bit mode

        // Norm correction
        float recon_norm = sqrt(recon_norm_sq);
        blk.norm = half((recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm);
    }
}

template<typename T, typename TI>
kernel void kernel_set_rows_f(
