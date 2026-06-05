// TurboQuant non-vec flash attention (nl=NL_TURBO3=QK_TURBO3/16)
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo3_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo3_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo3_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo3_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo3_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo3_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo3_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo3_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo3_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo3_vturbo3_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, block_turbo3_0, NL_TURBO3, dequantize_turbo3_0, 576, 512>;

// TurboQuant2 non-vec flash attention (nl=NL_TURBO2=QK_TURBO2/16)
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo2_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo2_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo2_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo2_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo2_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo2_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo2_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo2_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo2_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo2_vturbo2_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, block_turbo2_0, NL_TURBO2, dequantize_turbo2_0, 576, 512>;

// TurboQuant4 non-vec flash attention (block size 128, nl=8)
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo4_dk32_dv32"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 32,  32>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo4_dk64_dv64"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 64,  64>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo4_dk96_dv96"  )]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 96,  96>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo4_dk128_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 128, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo4_dk192_dv192")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 192, 192>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo4_dk192_dv128")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 192, 128>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo4_dk256_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 256, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo4_dk320_dv256")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 320, 256>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo4_dk512_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 512, 512>;
template [[host_name("kernel_flash_attn_ext_kturbo4_vturbo4_dk576_dv512")]] kernel flash_attn_ext_t kernel_flash_attn_ext<FA_TYPES, block_turbo4_0, 8, dequantize_turbo4_0, block_turbo4_0, 8, dequantize_turbo4_0, 576, 512>;

