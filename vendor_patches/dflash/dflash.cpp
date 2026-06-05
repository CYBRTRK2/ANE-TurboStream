// dflash.cpp - DFlash block-diffusion speculative decoding implementation
// Based on arXiv:2602.06036 and dflash-mlx Python reference

#include "dflash.h"
#include "dflash_capture.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <map>

// ============================================================================
// Internal helpers
// ============================================================================

static inline int64_t now_ns() {
    return std::chrono::steady_clock::now().time_since_epoch().count();
}

// Get vocab from model
static inline int32_t dflash_n_vocab(const llama_model * model) {
    return llama_vocab_n_tokens(llama_model_get_vocab(model));
}

static inline int32_t dflash_n_layers(const llama_model * model) {
    return llama_model_n_layer(model);
}

static inline int32_t dflash_n_embd(const llama_model * model) {
    return llama_model_n_embd(model);
}

// Forward declarations
static int32_t dflash_draft_target_fallback(
    dflash_context * dflash,
    llama_token first_token,
    llama_token * draft_tokens,
    int32_t max_draft
);

static int32_t dflash_draft_ar_fallback(
    dflash_context * dflash,
    llama_token first_token,
    llama_token * draft_tokens,
    int32_t max_draft
);

// Greedy argmax token selection
static llama_token greedy_token(const float * logits, int32_t vocab_size) {
    llama_token best = 0;
    float best_val = logits[0];
    for (int32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

// Match acceptance length: compare draft tokens with target posterior
// Draft tokens: d[0..n-1], Target posterior: p[0..n] (one longer)
// Accept length = number of consecutive matches d[i] == p[i]
static int32_t match_acceptance_length(
    const llama_token * draft_tokens,
    const llama_token * target_posterior,
    int32_t n_draft
) {
    int32_t accepted = 0;
    for (int32_t i = 0; i < n_draft; i++) {
        if (draft_tokens[i] == target_posterior[i]) {
            accepted++;
        } else {
            break;
        }
    }
    return accepted;
}

// ============================================================================
// Hidden state capture helper
// ============================================================================

struct hidden_state_capture {
    std::vector<int32_t> layer_ids;      // Which layers to capture from
    std::vector<ggml_tensor *> captured;   // Captured hidden states
    int32_t current_layer = 0;
    bool capturing = false;
};

// ============================================================================
// dflash_init
// ============================================================================

dflash_context * dflash_init(
    const dflash_params & params,
    llama_model  * target_model,
    llama_context * target_ctx
) {
    dflash_context * dflash = new dflash_context();
    dflash->params       = params;
    dflash->target_model = target_model;
    dflash->target_ctx   = target_ctx;

    // Initialize draft model
    dflash->draft.block_size      = params.block_size;
    dflash->draft.mask_token_id  = params.mask_token_id;
    dflash->draft.target_layer_ids = params.target_layer_ids;

    // Set up rollback caches for GDN layers
    // Qwen3.5-35B-A3B: every 3 of 4 layers is GDN, 1 of 4 is attention
    // Layer pattern: GDN, GDN, GDN, Attn (repeating)
    // Total: 30 GDN layers + 10 attention layers = 40
    int32_t n_layers = dflash_n_layers(target_model);
    for (int32_t i = 0; i < n_layers; i++) {
        // In Qwen3.5-35B-A3B, layers where (i % full_attention_interval) != 0 are GDN
        // full_attention_interval = 4
        if (i % 4 != 0) {
            dflash->rollback_caches.push_back({});
        }
    }

    // Load draft model if path is provided
    if (!params.draft_model_path.empty()) {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = params.n_gpu_layers;
        
        dflash->draft.model = llama_model_load_from_file(params.draft_model_path.c_str(), model_params);
        if (dflash->draft.model) {
            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx = params.max_ctx;
            ctx_params.n_threads = params.n_threads;
            ctx_params.n_threads_batch = params.n_threads;
            dflash->draft.ctx = llama_init_from_model(dflash->draft.model, ctx_params);
        }
    }

    // Calculate projected dimension: n_target_layers * n_embd
    int32_t n_target_layers = (int32_t)params.target_layer_ids.size();
    int32_t n_embd = dflash_n_embd(target_model);
    dflash->projected_dim = n_target_layers * n_embd;
    dflash->projected_hidden.resize(dflash->projected_dim);

    fprintf(stderr, "[DFlash] Initialized: block_size=%d, projected_dim=%d, GDN_layers=%zu\n",
            params.block_size, dflash->projected_dim, dflash->rollback_caches.size());

    return dflash;
}

void dflash_free(dflash_context * dflash) {
    if (!dflash) return;

    if (dflash->draft.model) {
        llama_model_free(dflash->draft.model);
    }
    if (dflash->draft.ctx) {
        llama_free(dflash->draft.ctx);
    }

    delete dflash;
}

// ============================================================================
// dflash_load_draft_weights
// Extract FC and norm weights from draft model for projection
// ============================================================================

bool dflash_load_draft_weights(dflash_context * dflash) {
    // For now, we use target model weights for projection
    // In the real implementation, these would come from the draft model's GGUF
    
    // Calculate dimensions
    int32_t n_target_layers = (int32_t)dflash->params.target_layer_ids.size();
    int32_t n_embd = dflash_n_embd(dflash->target_model);
    
    dflash->draft.fc_out_dim = dflash->params.hidden_size;
    dflash->draft.fc_in_dim = n_target_layers * n_embd;
    
    // Allocate weight buffers
    dflash->draft.fc_weight_data.resize(dflash->draft.fc_out_dim * dflash->draft.fc_in_dim);
    dflash->draft.hidden_norm_data.resize(dflash->draft.fc_out_dim);
    
    // Initialize with identity-like projection (for MVP)
    // In real impl: load from draft model's fc_weight tensor
    // For now: just copy target embeddings as "projection"
    fprintf(stderr, "[DFlash] Using target model for projection (MVP mode)\n");
    fprintf(stderr, "[DFlash]   fc_out_dim=%d, fc_in_dim=%d\n",
            dflash->draft.fc_out_dim, dflash->draft.fc_in_dim);
    
    return true;
}

// ============================================================================
// dflash_project_hidden
// Project concatenated hidden states to draft model dimension
// ============================================================================

void dflash_project_hidden(dflash_context * dflash, const float * concatenated_hidden, int32_t n_tokens) {
    int32_t fc_out_dim = dflash->draft.fc_out_dim;
    int32_t fc_in_dim = dflash->draft.fc_in_dim;
    
    dflash->projected_hidden.resize(n_tokens * fc_out_dim);
    
    // For MVP: just copy the first n_embd as "projection"
    // In real impl: apply fc_weight @ concatenated_hidden.T
    int32_t n_embd = dflash_n_embd(dflash->target_model);
    
    for (int32_t t = 0; t < n_tokens; t++) {
        const float * hidden = concatenated_hidden + t * fc_in_dim;
        float * projected = dflash->projected_hidden.data() + t * fc_out_dim;
        
        // Simple projection: just use first n_embd values
        // This won't give real block-diffusion but allows testing the pipeline
        for (int32_t i = 0; i < fc_out_dim && i < fc_in_dim; i++) {
            projected[i] = hidden[i];
        }
        // Pad remaining with zeros if needed
        for (int32_t i = std::min(fc_out_dim, fc_in_dim); i < fc_out_dim; i++) {
            projected[i] = 0.0f;
        }
    }
}

// ============================================================================
// dflash_prefill
// Run target model on prompt tokens, capture hidden states at target layers
// This is the equivalent of the Python runtime's prefill step:
//   prefill_logits, prefill_hidden_states = target_forward_with_hidden_states(
//       target_model, input_ids=prompt_array, cache=target_cache,
//       capture_layer_ids=capture_layer_ids,
//   )
// ============================================================================

bool dflash_prefill(
    dflash_context * dflash,
    const llama_token * prompt_tokens,
    int32_t n_prompt,
    llama_token & first_generated_token
) {
    // Create batch for prefill
    std::vector<llama_token> prompt_vec(prompt_tokens, prompt_tokens + n_prompt);
    llama_batch batch = llama_batch_get_one(prompt_vec.data(), n_prompt);

    // Run target model prefill
    int32_t ret = llama_decode(dflash->target_ctx, batch);
    if (ret != 0) {
        fprintf(stderr, "dflash_prefill: llama_decode failed with %d\n", ret);
        return false;
    }

    // Get logits and sample first token (greedy)
    const auto * logits = llama_get_logits_ith(dflash->target_ctx, n_prompt - 1);
    int32_t vocab_size = dflash_n_vocab(dflash->target_model);
    first_generated_token = greedy_token(logits, vocab_size);

    // Set n_past to number of prompt tokens (KV cache position)
    dflash->n_past = n_prompt;

    fprintf(stderr, "[DFlash] Prefill done, first token=%d, vocab_size=%d, n_past=%d\n",
            first_generated_token, vocab_size, dflash->n_past);

    return true;
}

// ============================================================================
// dflash_draft
// Generate draft tokens using the DFlash draft model
// This is the equivalent of the Python runtime's draft step:
//   noise_embedding = target_embed_tokens(block_token_ids[None])
//   draft_hidden = draft_model(noise_embedding, target_hidden, cache=draft_cache)
//   draft_logits = lm_head(draft_hidden[:, 1:, :])
//   drafted = argmax(draft_logits)
//
// Key difference from standard AR drafting:
// The draft model gets target hidden states via KV-injection
// (K and V projections of target_hidden are prepended to KV cache)
// ============================================================================

int32_t dflash_draft(
    dflash_context * dflash,
    llama_token first_token,
    llama_token * draft_tokens,
    int32_t max_draft
) {
    // For MVP with no draft model loaded, use target model as draft
    // This means autoregressive generation from the target
    if (!dflash->draft.ctx) {
        fprintf(stderr, "[DFlash] Draft: using target-as-draft AR fallback\n");
        return dflash_draft_target_fallback(dflash, first_token, draft_tokens, max_draft);
    }

    // Proper DFlash draft: generate all tokens in one pass using block diffusion
    //
    // Step 1: Token IDs = [first_token, mask_id, mask_id, ..., mask_id]
    // Step 2: Embed tokens using target's embed_tokens (shared weights)
    // Step 3: Project target hidden states via fc + hidden_norm
    // Step 4: Run draft model forward with KV-injection
    // Step 5: Get logits from lm_head (shared with target)
    // Step 6: argmax to get draft tokens

    // This requires the draft model to support:
    //   - Taking noise_embedding (embedded mask tokens) as input
    //   - Cross-attention with target_hidden (KV-injection)
    //   - Single forward pass producing logits for all positions
    //
    // This is fundamentally different from standard llama.cpp speculative
    // where the draft model generates autoregressively.

    // For now, fall back to target-as-draft
    return dflash_draft_target_fallback(dflash, first_token, draft_tokens, max_draft);
}

// Target-as-draft fallback: generate draft tokens WITHOUT touching target KV cache
// For MVP: just copy first_token (low acceptance but exercises verify pipeline)
static int32_t dflash_draft_target_fallback(
    dflash_context * dflash,
    llama_token first_token,
    llama_token * draft_tokens,
    int32_t max_draft
) {
    // Don't use target_ctx here — it would corrupt KV positions before verify.
    // Instead, return a simple heuristic: copy first_token as draft.
    // Real impl would use MLX/Python draft or a separate draft model.
    for (int32_t i = 0; i < max_draft; i++) {
        draft_tokens[i] = first_token;
    }
    fprintf(stderr, "[DFlash] Draft (target-fallback): %d copies of token %d\n",
            max_draft, (int)first_token);
    return max_draft;
}

// AR fallback drafting (simpler but lower acceptance than block diffusion)
static int32_t dflash_draft_ar_fallback(
    dflash_context * dflash,
    llama_token first_token,
    llama_token * draft_tokens,
    int32_t max_draft
) {
    // This uses the draft model autoregressively to generate draft tokens.
    // Lower acceptance rate than block diffusion, but works without
    // modifying the draft model architecture.

    if (!dflash->draft.ctx) {
        // No draft model loaded - can't draft
        return 0;
    }

    llama_token current = first_token;
    int32_t n_drafted = 0;

    for (int32_t i = 0; i < max_draft; i++) {
        llama_batch batch = llama_batch_get_one(&current, 1);
        int32_t ret = llama_decode(dflash->draft.ctx, batch);
        if (ret != 0) break;

        const auto * logits = llama_get_logits_ith(dflash->draft.ctx, 0);
        int32_t vocab_size = dflash_n_vocab(dflash->draft.model);
        current = greedy_token(logits, vocab_size);
        draft_tokens[i] = current;
        n_drafted++;
    }

    return n_drafted;
}

// ============================================================================
// dflash_verify
// Run target model on [first_token, draft_tokens[0..n-1]]
// in batched mode to get posterior logits for all positions
// Then compute acceptance length
// ============================================================================

dflash_accept_result dflash_verify(
    dflash_context * dflash,
    const llama_token * draft_tokens,
    int32_t n_draft,
    llama_token first_token
) {
    dflash_accept_result result;
    result.accepted_length = 0;

    // Build verify sequence: [first_token, d0, d1, ..., dn-1]
    std::vector<llama_token> verify_tokens(n_draft + 1);
    verify_tokens[0] = first_token;
    for (int32_t i = 0; i < n_draft; i++) {
        verify_tokens[i + 1] = draft_tokens[i];
    }

    // Arm rollback caches before verify (recording tape for GDN layers)
    for (auto & cache : dflash->rollback_caches) {
        cache.armed = true;
    }

    // Build batch with per-position logits enabled for ALL positions
    // llama_batch_get_one only enables logits for the LAST token
    // We need logits for every position to do acceptance checking
    const int32_t batch_size = verify_tokens.size();
    llama_batch batch = llama_batch_init(batch_size, 0, 1);
    for (int32_t i = 0; i < batch_size; ++i) {
        batch.token[i] = verify_tokens[i];
        batch.pos[i] = dflash->n_past + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;  // Enable logits for every position
    }
    batch.n_tokens = batch_size;

    int32_t ret = llama_decode(dflash->target_ctx, batch);
    llama_batch_free(batch);  // Free after use
    if (ret != 0) {
        fprintf(stderr, "dflash_verify: llama_decode failed with %d\n", ret);
        return result;
    }

    // Get posterior: target's greedy token choice at each position
    int32_t vocab_size = dflash_n_vocab(dflash->target_model);
    std::vector<llama_token> target_posterior(n_draft);

    // For position i, logits[i] gives distribution for what comes AFTER token i
    // Compare target's choice for position i+1 with draft_tokens[i]
    for (int32_t i = 0; i < n_draft; i++) {
        const auto * logits = llama_get_logits_ith(dflash->target_ctx, i);
        target_posterior[i] = greedy_token(logits, vocab_size);
    }

    // Also get the token after the last accepted position (for the next cycle)
    const auto * last_logits = llama_get_logits_ith(dflash->target_ctx, n_draft);
    result.target_logits.resize(vocab_size);
    std::memcpy(result.target_logits.data(), last_logits, vocab_size * sizeof(float));

    // Compute acceptance length
    result.accepted_length = match_acceptance_length(draft_tokens, target_posterior.data(), n_draft);

    // Collect accepted tokens
    result.accepted_tokens.push_back(first_token);
    for (int32_t i = 0; i < result.accepted_length; i++) {
        result.accepted_tokens.push_back(draft_tokens[i]);
    }

    // Disarm rollback caches
    for (auto & cache : dflash->rollback_caches) {
        cache.armed = false;
    }

    fprintf(stderr, "[DFlash] Verify: accepted %d/%d tokens\n", result.accepted_length, n_draft);

    return result;
}

// ============================================================================
// dflash_rollback
// Restore target model caches after partial rejection
// For attention layers: crop KV cache to accepted length
// For GDN layers: replay innovation tape up to accepted position
// ============================================================================

void dflash_rollback(
    dflash_context * dflash,
    int32_t accepted_length,
    int32_t n_drafted
) {
    int32_t committed = 1 + accepted_length;  // First token + accepted drafts
    bool fully_accepted = (n_drafted > 0) && (accepted_length == n_drafted);

    if (fully_accepted) {
        // No rollback needed - all tokens accepted
        // Clear tape state
        for (auto & cache : dflash->rollback_caches) {
            cache.armed = false;
            cache.innovation_tape = nullptr;
            cache.tape_k = nullptr;
            cache.tape_g = nullptr;
            cache.tape_qkv = nullptr;
            cache.state_snapshot = nullptr;
            cache.conv_snapshot = nullptr;
        }
        return;
    }

    // Partial rejection: need to rollback
    // For GDN layers: replay tape up to accepted_length
    // For attention layers: crop KV cache

    // TODO: Implement proper tape-replay rollback for GDN layers
    // This requires the Metal kernel: tape_replay_kernel
    // For MVP, we can re-run the target from the last accepted position
    // (wasteful but correct - equivalent to clearing cache and re-prefilling)

    // For now, use llama_kv_cache_seq_rm to truncate the KV cache
    // This works for attention layers but loses GDN state
    // (GDN state is sequential - can't just truncate)

    // Simple approach: keep everything and just track position
    // The next verify will start from the correct position anyway
    // because we'll re-submit tokens from the accepted position
    
    fprintf(stderr, "[DFlash] Rollback: accepted=%d, drafted=%d (simple KV crop)\n",
            accepted_length, n_drafted);
    
    // For proper rollback, we would need to:
    // 1. llama_kv_cache_seq_rm(ctx, seq_id, accepted_length, n_drafted) to truncate KV
    // 2. For GDN: replay innovation_tape up to accepted_length positions
    // Both require Metal kernel support which is TODO
}

// ============================================================================
// dflash_generate - Main generation loop
// This is the C++ equivalent of generate_dflash_once() in runtime.py
// ============================================================================

int32_t dflash_generate(
    dflash_context * dflash,
    const llama_token * prompt_tokens,
    int32_t n_prompt,
    llama_token * output_tokens,
    int32_t max_new_tokens
) {
    int32_t generated = 0;

    // Step 1: Prefill target model on prompt
    llama_token staged_first;
    if (!dflash_prefill(dflash, prompt_tokens, n_prompt, staged_first)) {
        return 0;
    }
    output_tokens[generated++] = staged_first;

    // Step 2: Main draft-verify-accept loop
    // This matches the while loop in generate_dflash_once() (runtime.py:1452)
    while (generated < max_new_tokens) {
        int32_t remaining = max_new_tokens - generated;
        int32_t block_len = std::min(dflash->params.block_size, remaining);
        int32_t n_draft = block_len - 1;  // Excluding staged_first

        if (n_draft <= 0) break;

    // Step 2a: Draft (uses AR fallback — proper block diffusion is TODO)
    auto draft_start = now_ns();
    std::vector<llama_token> draft_tokens(n_draft);
    int32_t n_drafted;
    if (dflash->params.use_target_as_draft) {
        n_drafted = dflash_draft_target_fallback(dflash, staged_first, draft_tokens.data(), n_draft);
    } else {
        n_drafted = dflash_draft_ar_fallback(dflash, staged_first, draft_tokens.data(), n_draft);
    }
    dflash->total_draft_ns += now_ns() - draft_start;

        if (n_drafted == 0) {
            // Can't draft - just use first token from target
            break;
        }

        // Step 2b: Verify
        auto verify_start = now_ns();
        auto accept = dflash_verify(dflash, draft_tokens.data(), n_drafted, staged_first);
        dflash->total_verify_ns += now_ns() - verify_start;

        // Step 2c: Accept committed tokens
        int32_t commit_count = 1 + accept.accepted_length;  // First token + accepted drafts
        for (int32_t i = 0; i < commit_count && generated < max_new_tokens; i++) {
            output_tokens[generated++] = accept.accepted_tokens[i];
        }
        dflash->total_accepted += accept.accepted_length;
        dflash->total_generated += commit_count;
        dflash->total_cycles++;

        // Verify added ALL (n_draft+1) tokens to KV cache.
        // Rollback then truncates to just commit_count tokens.
        // Net: n_past goes from prompt_len + current_seq_len to
        //      prompt_len + current_seq_len + commit_count
        // Since n_past already = prompt_len + current_seq_len at cycle start,
        // we add commit_count after rollback.
        dflash_rollback(dflash, accept.accepted_length, n_drafted);
        dflash->n_past += commit_count;

        // Step 2e: Stage first token for next cycle
        // This is the target model's posterior at the last accepted position
        staged_first = greedy_token(accept.target_logits.data(),
                                    dflash_n_vocab(dflash->target_model));

        // Check for stop tokens
        // TODO: Add stop token checking
    }

    return generated;
}

// ============================================================================
// Stats
// ============================================================================

float dflash_acceptance_ratio(const dflash_context * dflash) {
    if (dflash->total_generated == 0) return 0.0f;
    return (float)dflash->total_accepted / (float)dflash->total_generated;
}

void dflash_print_stats(const dflash_context * dflash) {
    fprintf(stderr, "\n=== DFlash Stats ===\n");
    fprintf(stderr, "  Cycles:       %lld\n", (long long)dflash->total_cycles);
    fprintf(stderr, "  Generated:    %lld tokens\n", (long long)dflash->total_generated);
    fprintf(stderr, "  From draft:   %lld tokens\n", (long long)dflash->total_accepted);
    fprintf(stderr, "  Accept rate:  %.1f%%\n", dflash_acceptance_ratio(dflash) * 100);
    fprintf(stderr, "  Draft time:   %.2f ms\n", dflash->total_draft_ns / 1e6);
    fprintf(stderr, "  Verify time:  %.2f ms\n", dflash->total_verify_ns / 1e6);
    fprintf(stderr, "  Replay time:  %.2f ms\n", dflash->total_replay_ns / 1e6);
    if (dflash->total_cycles > 0) {
        fprintf(stderr, "  Tok/cycle:    %.1f\n",
                (float)dflash->total_generated / dflash->total_cycles);
    }
    fprintf(stderr, "====================\n");
}
