// K2-Horizon model weights. Minimal host-side weight container for GGUF loading.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vt/tensor.h"

namespace vllm {

struct K2HorizonLayerWeights {
  // Attention.
  OwnedTensor attn_norm_weight;     // [hidden_size]
  OwnedTensor attn_q_weight;       // [hidden_size, n_heads * head_dim]
  OwnedTensor attn_k_weight;       // [hidden_size, n_kv_heads * head_dim]
  OwnedTensor attn_v_weight;       // [hidden_size, n_kv_heads * head_dim] (dense)
  OwnedTensor attn_output_weight;  // [n_heads * head_dim, hidden_size]
  // MoVA (value routing).
  OwnedTensor attn_gate_weight;    // [n_heads * head_dim, hidden_size] (softplus gate)
  OwnedTensor attn_v_gate_weight;  // [n_value_experts, hidden_size]
  OwnedTensor attn_v_gate_bias;    // [n_value_experts]
  OwnedTensor attn_v_exps_weight;  // [n_value_experts, n_kv_heads * head_dim, hidden_size]
  // FFN.
  OwnedTensor ffn_norm_weight;     // [hidden_size]
  OwnedTensor ffn_gate_weight;     // [intermediate_size, hidden_size]
  OwnedTensor ffn_up_weight;       // [intermediate_size, hidden_size]
  OwnedTensor ffn_down_weight;     // [hidden_size, intermediate_size]
  // MoE FFN.
  OwnedTensor ffn_gate_inp_weight;
  OwnedTensor ffn_gate_exps_weight;
  OwnedTensor ffn_up_exps_weight;
  OwnedTensor ffn_down_exps_weight;
  OwnedTensor ffn_gate_shexp_weight;
  OwnedTensor ffn_up_shexp_weight;
  OwnedTensor ffn_down_shexp_weight;
  OwnedTensor exp_probs_b_bias;
};

struct K2HorizonWeights {
  OwnedTensor embed_tokens;
  OwnedTensor lm_head;
  OwnedTensor final_norm_weight;
  OwnedTensor rope_cos_sin;  // bf16 [max_pos, rotary_dim]
  std::vector<K2HorizonLayerWeights> layers;
  int64_t n_norm_groups = 1;
  int64_t n_value_experts = 0;
  int64_t n_value_expert_used = 0;
  int64_t leading_dense_block_count = 0;
  int64_t moe_every_n_layers = 0;
  int64_t expert_count = 0;
  int64_t expert_used_count = 0;
  int64_t expert_shared_count = 0;
  int64_t expert_ffn_size = 0;    // expert_feed_forward_length (per-expert MoE intermediate)
  int64_t shared_ffn_size = 0;    // expert_shared_feed_forward_length (shared-expert intermediate)
  int64_t expert_gating_func = 1; // llm_expert_gating_func_type: 1=softmax, 2=sigmoid
  bool expert_weights_norm = false;
  float expert_weights_scale = 1.0f;
  float rms_norm_eps = 1e-6f;
};

K2HorizonWeights LoadK2HorizonGgufWeights(const class GgufFile& gguf,
                                          const struct HfConfig& config);

// CPU bring-up forward: returns host logits directly.
std::vector<float> K2HorizonForward(const K2HorizonWeights& weights,
                                    const struct HfConfig& config,
                                    const struct ModelForwardInput& input);

}  // namespace vllm
