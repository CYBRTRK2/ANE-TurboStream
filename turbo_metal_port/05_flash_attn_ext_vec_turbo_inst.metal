template [[host_name("kernel_flash_attn_ext_vec_kbf16_vbf16_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 128, 128, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_kq4_0_vq4_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq4_1_vq4_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_0_vq5_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_1_vq5_1_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vq8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 1>;

template [[host_name("kernel_flash_attn_ext_vec_kf32_vf32_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kf16_vf16_dk192_dv192")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 192, 2>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_kbf16_vbf16_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 192, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_kq4_0_vq4_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq4_1_vq4_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_0_vq5_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_1_vq5_1_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vq8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 192, 2>;

template [[host_name("kernel_flash_attn_ext_vec_kf32_vf32_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kf16_vf16_dk192_dv128")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  192, 128, 2>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_kbf16_vbf16_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 192, 128, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_kq4_0_vq4_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq4_1_vq4_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_0_vq5_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_1_vq5_1_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vq8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 128, 2>;

template [[host_name("kernel_flash_attn_ext_vec_kf32_vf32_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kf16_vf16_dk256_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  256, 256, 1>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_kbf16_vbf16_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 256, 256, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_kq4_0_vq4_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq4_1_vq4_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_0_vq5_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_1_vq5_1_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vq8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 256, 256, 1>;
// TurboQuant flash attention - dk256_dv256

template [[host_name("kernel_flash_attn_ext_vec_kf32_vf32_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kf16_vf16_dk320_dv256")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  320, 256, 2>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_kbf16_vbf16_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 320, 256, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_kq4_0_vq4_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq4_1_vq4_1_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_0_vq5_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_1_vq5_1_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vq8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 320, 256, 2>;

template [[host_name("kernel_flash_attn_ext_vec_kf32_vf32_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kf16_vf16_dk512_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  512, 512, 1>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_kbf16_vbf16_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 512, 512, 1>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_kq4_0_vq4_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq4_1_vq4_1_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_0_vq5_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_1_vq5_1_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vq8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 512, 512, 1>;

template [[host_name("kernel_flash_attn_ext_vec_kf32_vf32_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES_F32, float4,     1, dequantize_f32_t4,  float4,      1, dequantize_f32_t4,  576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kf16_vf16_dk576_dv512")]]  kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     half4,      1, dequantize_f16_t4,  half4,       1, dequantize_f16_t4,  576, 512, 2>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_flash_attn_ext_vec_kbf16_vbf16_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     bfloat4,    1, dequantize_bf16_t4, bfloat4,     1, dequantize_bf16_t4, 576, 512, 2>;
#endif
template [[host_name("kernel_flash_attn_ext_vec_kq4_0_vq4_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_0, 8, dequantize_q4_0_t4, block_q4_0,  8, dequantize_q4_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq4_1_vq4_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q4_1, 8, dequantize_q4_1_t4, block_q4_1,  8, dequantize_q4_1_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_0_vq5_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_0, 8, dequantize_q5_0_t4, block_q5_0,  8, dequantize_q5_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq5_1_vq5_1_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q5_1, 8, dequantize_q5_1_t4, block_q5_1,  8, dequantize_q5_1_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vq8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES,     block_q8_0, 8, dequantize_q8_0_t4, block_q8_0,  8, dequantize_q8_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo3_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo4_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 576, 512, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo2_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 576, 512, 2>;

// Asymmetric K/V TurboQuant vec flash attention — turbo2 K, turbo3 V
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo3_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 576, 512, 2>;

// Asymmetric K/V TurboQuant vec flash attention — turbo3 K, turbo2 V
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo2_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 576, 512, 2>;

// Asymmetric K/V TurboQuant vec flash attention — turbo2 K, turbo4 V
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vturbo4_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 576, 512, 2>;

// Asymmetric K/V TurboQuant vec flash attention — turbo4 K, turbo2 V
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo2_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 576, 512, 2>;

// Asymmetric K/V TurboQuant vec flash attention — turbo3 K, turbo4 V
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vturbo4_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 576, 512, 2>;

// Asymmetric K/V TurboQuant vec flash attention — turbo4 K, turbo3 V
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vturbo3_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 576, 512, 2>;


// Asymmetric q8_0 K, turbo2 V — vec flash attention
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo2_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, 576, 512, 2>;

// Asymmetric turbo2 K, q8_0 V — vec flash attention
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo2_vq8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo2_0, NL_TURBO2_VEC, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 576, 512, 2>;

// Asymmetric q8_0 K, turbo3 V — vec flash attention
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo3_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, 576, 512, 2>;

// Asymmetric turbo3 K, q8_0 V — vec flash attention
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo3_vq8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo3_0, NL_TURBO3_VEC, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 576, 512, 2>;

// Asymmetric q8_0 K, turbo4 V — vec flash attention
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 576, 512, 2>;

// Asymmetric turbo4 K, q8_0 V — vec flash attention
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk32_dv32")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 32, 32, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk64_dv64")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 64, 64, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk96_dv96")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 96, 96, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk128_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 128, 128, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk192_dv192")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 192, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk192_dv128")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 192, 128, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk256_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 256, 256, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk320_dv256")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 320, 256, 2>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk512_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 512, 512, 1>;
template [[host_name("kernel_flash_attn_ext_vec_kturbo4_vq8_0_dk576_dv512")]] kernel flash_attn_ext_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4, 576, 512, 2>;

#undef FA_TYPES
#undef FA_TYPES_F32

