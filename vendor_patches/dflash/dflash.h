// dflash.h - DFlash block-diffusion speculative decoding for anemll-flash-llama.cpp
// Based on arXiv:2602.06036 (DFlash: Block-Diffusion Speculative Decoding)
// Port from dflash-mlx (Python/MLX) to C++/Metal for llama.cpp

#ifndef DFLASH_H
#define DFLASH_H

#include "ggml.h"
#include "llama.h"
#include <vector>
#include <map>
#include <cstdint>

// ============================================================================
// DFlash Configuration
// ============================================================================

struct dflash_params {
    int32_t block_size      = 16;      // Number of draft tokens per block
    int32_t n_draft_layers  = 8;       // Number of layers in draft model
    int32_t n_kv_heads      = 4;       // KV heads in draft model
    int32_t n_attn_heads    = 32;      // Attention heads in draft model
    int32_t hidden_size     = 2048;    // Hidden size of draft model
    int32_t intermediate_size = 6144;  // Intermediate (MLP) size of draft model
    int32_t head_dim        = 128;     // Head dimension of draft model
    int32_t max_ctx         = 262144;  // Max context length
    int32_t n_threads       = 4;       // Threads for draft model
    float   rms_norm_eps    = 1e-6f;   // RMS norm epsilon
    float   rope_theta      = 1e7f;    // RoPE theta
    int32_t mask_token_id   = 0;       // Mask token ID for block diffusion

    // Target layer IDs where hidden states are captured
    // Default: [1, 10, 19, 28, 37] for Qwen3.5-35B-A3B (40 layers)
    std::vector<int32_t> target_layer_ids = {1, 10, 19, 28, 37};

    // Verify chunking: if > 0, verify in chunks of this many tokens
    int32_t verify_chunk_tokens = 0;   // 0 = single pass

    std::string draft_model_path;      // Path to draft model GGUF
    bool use_target_as_draft = true;   // Use target model as draft (for initial testing)
    int32_t n_gpu_layers = 99;         // GPU layers for draft model
};

// ============================================================================
// GatedDeltaNet Innovation Tape
// Records (innovation_tape, k, g, qkv) during verify for rollback
// ============================================================================

struct dflash_gdn_tape {
    // Per-layer tape data for a single verify pass
    ggml_tensor * innovation_tape;  // [B, accepted_steps+1, h_v, d_v] - the delta values
    ggml_tensor * tape_k;           // [B, accepted_steps+1, h_k, d_k]
    ggml_tensor * tape_g;           // [B, accepted_steps+1, h_k] - gate values
    ggml_tensor * tape_qkv;         // [B, steps, conv_dim] - conv input for state rebuild

    // Snapshot of GDN state before verify (for rollback)
    ggml_tensor * state_snapshot;    // [B, h_v, d_v, d_k]
    ggml_tensor * conv_snapshot;      // [B, conv_kernel_size-1, conv_dim]

    bool armed = false;  // Whether tape recording is active
    int32_t prefix_len = 0;  // Position before verify started
};

// ============================================================================
// DFlash Draft Model State
// ============================================================================

struct dflash_draft_model {
    // The draft model loaded as a separate llama_model + llama_context
    llama_model  * model  = nullptr;
    llama_context * ctx    = nullptr;

    // DFlash-specific weights (loaded from GGUF as custom tensors)
    ggml_tensor * fc_weight;        // [hidden_size, n_target_layers * hidden_size]
    ggml_tensor * hidden_norm_weight; // [hidden_size]

    // FC and norm weights extracted as floats for projection
    std::vector<float> fc_weight_data;
    std::vector<float> hidden_norm_data;
    int32_t fc_out_dim = 0;
    int32_t fc_in_dim = 0;

    // Layer IDs to capture from target
    std::vector<int32_t> target_layer_ids;

    // Block size for draft generation
    int32_t block_size = 16;
    int32_t mask_token_id = 0;

    // KV cache for draft model (context-only, sliding window)
    struct draft_kv_cache {
        ggml_tensor * keys;    // [n_kv_heads, sink_size+window_size, head_dim]
        ggml_tensor * values;   // [n_kv_heads, sink_size+window_size, head_dim]
        int32_t offset = 0;
        int32_t sink_size = 64;
        int32_t window_size = 1024;
    };
    std::vector<draft_kv_cache> layer_caches;
};

// ============================================================================
// DFlash Rollback Cache per GDN layer
// ============================================================================

struct dflash_rollback_cache {
    ggml_tensor * state_snapshot;   // Snapshot of GDN state before verify
    ggml_tensor * conv_snapshot;    // Snapshot of conv state before verify
    ggml_tensor * innovation_tape;  // Recorded during verify
    ggml_tensor * tape_k;
    ggml_tensor * tape_g;
    ggml_tensor * tape_qkv;
    bool armed = false;
};

// ============================================================================
// DFlash Acceptance Result
// ============================================================================

struct dflash_accept_result {
    int32_t accepted_length;      // How many draft tokens were accepted (0 to block_size-1)
    std::vector<llama_token> accepted_tokens;  // The accepted token IDs
    std::vector<float> target_logits;           // Target logits at last accepted position
};

// ============================================================================
// DFlash Context (main orchestrator)
// ============================================================================

struct dflash_context {
    dflash_params params;

    // Target model
    llama_model  * target_model  = nullptr;
    llama_context * target_ctx    = nullptr;

    // Draft model
    dflash_draft_model draft;

    // Rollback caches (one per GDN layer in target)
    std::vector<dflash_rollback_cache> rollback_caches;

    // Hidden state capture buffers
    // Maps: target_layer_id -> captured hidden state tensor
    std::map<int32_t, ggml_tensor *> captured_hidden_states;

    // Projected hidden states for draft model (used in KV-injection)
    std::vector<float> projected_hidden;
    int32_t projected_dim = 0;

    // Stats
    int64_t total_draft_ns     = 0;
    int64_t total_verify_ns    = 0;
    int64_t total_replay_ns    = 0;
    int64_t total_cycles       = 0;
    int64_t total_accepted     = 0;
    int64_t total_generated    = 0;

    // Sequence position tracking for correct batch.pos in verify
    int32_t n_past = 0;
};

// ============================================================================
// DFlash API Functions
// ============================================================================

// Initialize DFlash context with target and draft models
dflash_context * dflash_init(
    const dflash_params & params,
    llama_model  * target_model,
    llama_context * target_ctx
);

// Free DFlash context
void dflash_free(dflash_context * dflash);

// Prefill: run target model on prompt, capture hidden states
bool dflash_prefill(
    dflash_context * dflash,
    const llama_token * prompt_tokens,
    int32_t n_prompt,
    llama_token & first_generated_token
);

// Draft: generate block_size-1 candidate tokens using draft model + target hidden states
int32_t dflash_draft(
    dflash_context * dflash,
    llama_token first_token,
    llama_token * draft_tokens,  // Output: block_size-1 draft tokens
    int32_t max_draft
);

// Verify: run target model on draft tokens in batched mode
// Returns acceptance length (number of consecutive matching tokens)
dflash_accept_result dflash_verify(
    dflash_context * dflash,
    const llama_token * draft_tokens,
    int32_t n_draft,
    llama_token first_token
);

// Rollback: restore target model KV/GDN caches to accepted position
void dflash_rollback(
    dflash_context * dflash,
    int32_t accepted_length,
    int32_t n_drafted
);

// Main generation loop: draft-verify-accept cycle
int32_t dflash_generate(
    dflash_context * dflash,
    const llama_token * prompt_tokens,
    int32_t n_prompt,
    llama_token * output_tokens,
    int32_t max_new_tokens
);

// Get acceptance ratio
float dflash_acceptance_ratio(const dflash_context * dflash);

// Print stats
void dflash_print_stats(const dflash_context * dflash);

// Helper: extract FC and norm weights from draft model for projection
bool dflash_load_draft_weights(dflash_context * dflash);

// Helper: project hidden states using fc + norm
void dflash_project_hidden(dflash_context * dflash, const float * concatenated_hidden, int32_t n_tokens);

#endif // DFLASH_H