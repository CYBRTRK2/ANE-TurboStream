template [[host_name("kernel_set_rows_iq4_nl_i64")]] kernel set_rows_q32_t kernel_set_rows_q32<int64_t, block_iq4_nl, quantize_iq4_nl>;
template [[host_name("kernel_set_rows_iq4_nl_i32")]] kernel set_rows_q32_t kernel_set_rows_q32<int32_t, block_iq4_nl, quantize_iq4_nl>;

// TurboQuant3 set_rows instantiations (4x32-element blocks per 128-element group)
typedef decltype(kernel_set_rows_turbo<int64_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>) set_rows_turbo3_t;

template [[host_name("kernel_set_rows_turbo3_i64")]] kernel set_rows_turbo3_t kernel_set_rows_turbo<int64_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>;
template [[host_name("kernel_set_rows_turbo3_i32")]] kernel set_rows_turbo3_t kernel_set_rows_turbo<int32_t, block_turbo3_0, QK_TURBO3, quantize_turbo3_0>;

// TurboQuant2 set_rows instantiations (dedicated kernel, 4x32-element blocks, no signs)
typedef decltype(kernel_set_rows_turbo2<int64_t>) set_rows_turbo2_t;

template [[host_name("kernel_set_rows_turbo2_i64")]] kernel set_rows_turbo2_t kernel_set_rows_turbo2<int64_t>;
template [[host_name("kernel_set_rows_turbo2_i32")]] kernel set_rows_turbo2_t kernel_set_rows_turbo2<int32_t>;

// TurboQuant4 set_rows instantiations (dedicated kernel, 128-element blocks with QJL)
typedef decltype(kernel_set_rows_turbo4<int64_t>) set_rows_turbo4_t;

template [[host_name("kernel_set_rows_turbo4_i64")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4<int64_t>;
template [[host_name("kernel_set_rows_turbo4_i32")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4<int32_t>;

//
// matrix-matrix multiplication
