// K2-Horizon GGUF weight loader.
#include "vllm/model_executor/models/k2horizon.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/layers/rotary_embedding/base.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm {

namespace {

int64_t KvInt(const GgufValue& v, const std::string& key) {
  switch (v.TypeId()) {
    case kGgufU8: return std::get<uint8_t>(v.v);
    case kGgufI8: return std::get<int8_t>(v.v);
    case kGgufU16: return std::get<uint16_t>(v.v);
    case kGgufI16: return std::get<int16_t>(v.v);
    case kGgufU32: return std::get<uint32_t>(v.v);
    case kGgufI32: return std::get<int32_t>(v.v);
    case kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case kGgufI64: return std::get<int64_t>(v.v);
    case kGgufBool: return std::get<bool>(v.v) ? 1 : 0;
    default:
      throw std::runtime_error("k2-horizon gguf: key " + key +
                               " is not an integer");
  }
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? KvInt(*v, key) : dflt;
}

float OptFloat(const GgufFile& g, const std::string& key, float dflt) {
  const GgufValue* v = g.FindKv(key);
  if (v == nullptr) return dflt;
  switch (v->TypeId()) {
    case kGgufF32: return std::get<float>(v->v);
    case kGgufF64: return static_cast<float>(std::get<double>(v->v));
    default:
      // Fall back to the integer reader for a value stored as an integer.
      return static_cast<float>(KvInt(*v, key));
  }
}

OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

int64_t ShapeNumel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

// Dequantize a GGUF tensor to bf16. For 2-D weights the GGUF stores
// torch [N, K] (out, in) layout, which IS MatmulBT's expected [N, K]
// layout — no transpose needed.  For 1-D tensors, just dequant to bf16.
OwnedTensor LoadWeightBf16(const GgufFile& g, const std::string& name) {
  const GgufTensorInfo& t = g.Get(name);
  const int64_t numel = ShapeNumel(t.shape);
  std::vector<uint16_t> dq =
      DequantGgufRowToBf16(t.ggml_type, t.data, numel, 1.0f);
  OwnedTensor o = MakeOwned(vt::DType::kBF16,
                            std::vector<int64_t>(t.shape.begin(), t.shape.end()));
  std::memcpy(o.bytes.data(), dq.data(), dq.size() * sizeof(uint16_t));
  return o;
}

}  // namespace

K2HorizonWeights LoadK2HorizonGgufWeights(const GgufFile& gguf,
                                          const HfConfig& config) {
  K2HorizonWeights w;
  const int64_t H = config.hidden_size;
  const int64_t Dh = config.head_dim;
  const int64_t I = config.intermediate_size;
  const int64_t n_layers = config.num_hidden_layers;

  w.n_norm_groups = OptInt(gguf, "k2-horizon.attention.group_norm_groups", 1);
  w.n_value_experts = OptInt(gguf, "k2-horizon.attention.value_expert_count", 0);
  w.n_value_expert_used = OptInt(gguf, "k2-horizon.attention.value_expert_used_count", 0);
  w.moe_every_n_layers = OptInt(gguf, "k2-horizon.moe_every_n_layers", 0);
  w.leading_dense_block_count =
      OptInt(gguf, "k2-horizon.leading_dense_block_count", 0);
  w.expert_count = OptInt(gguf, "k2-horizon.expert_count", 0);
  w.expert_used_count = OptInt(gguf, "k2-horizon.expert_used_count", 0);
  w.expert_shared_count = OptInt(gguf, "k2-horizon.expert_shared_count", 0);
  w.expert_ffn_size = OptInt(gguf, "k2-horizon.expert_feed_forward_length", 0);
  w.shared_ffn_size = OptInt(gguf, "k2-horizon.expert_shared_feed_forward_length", 0);
  w.expert_gating_func = OptInt(gguf, "k2-horizon.expert_gating_func", 1);
  w.expert_weights_norm = OptInt(gguf, "k2-horizon.expert_weights_norm", 0) != 0;
  w.expert_weights_scale = OptFloat(gguf, "k2-horizon.expert_weights_scale", 1.0f);
  w.rms_norm_eps = config.rms_norm_eps;

  // Global tensors. Embedding and lm_head stay in GGUF's [V, H] layout;
  // Embedding looks up rows directly and lm_head uses MatmulBT.
  w.embed_tokens = LoadWeightBf16(gguf, "token_embd.weight");
  w.lm_head = LoadWeightBf16(gguf, "output.weight");
  w.final_norm_weight = LoadWeightBf16(gguf, "output_norm.weight");

  // Build rope cos/sin cache through the shared get_rope() factory so YaRN
  // checkpoints (1B) get the scaled cache and plain ones (4B/7B/36B) the
  // default. The cache always spans the full context: default rows = ctx,
  // YaRN rows = orig_ctx * factor (8192 * 16 = 131072 = the 1B context).
  // The cache is bf16 to match q/k: vt::RopeFromCache requires q, k and cache
  // to share a dtype, and the attention activations are bf16 (as commandr is).
  {
    auto rope = get_rope(Dh, config.max_position_embeddings,
                         /*is_neox_style=*/true, config.rope_parameters,
                         vt::DType::kBF16);
    const vt::Tensor cache = rope->cos_sin_cache();
    VT_CHECK(cache.shape[1] == config.rotary_dim,
             "k2-horizon: rope cache must be [rows, rotary_dim]");
    w.rope_cos_sin = MakeOwned(vt::DType::kBF16,
                               {cache.shape[0], cache.shape[1]});
    std::memcpy(w.rope_cos_sin.bytes.data(), cache.data,
                w.rope_cos_sin.bytes.size());
  }

  // Per-layer tensors.
  w.layers.resize(static_cast<size_t>(n_layers));
  for (int64_t il = 0; il < n_layers; ++il) {
    const std::string blk = "blk." + std::to_string(il) + ".";
    auto& L = w.layers[static_cast<size_t>(il)];

    // Layer class, matching the reference (build_k2horizon.cpp
    // `is_moe_layer = n_expert > 0 && il >= n_layer_dense_lead`): the leading
    // blocks are dense, every block at or after them routes experts. Computed
    // up front because it decides both the V projection (dense vs MoVA) and
    // the FFN (dense vs MoE) below.
    const bool is_dense_layer =
        w.expert_count == 0 || il < w.leading_dense_block_count;
    const bool is_mova_layer = !is_dense_layer && w.n_value_experts > 0;

    L.attn_norm_weight = LoadWeightBf16(gguf, blk + "attn_norm.weight");
    L.attn_q_weight = LoadWeightBf16(gguf, blk + "attn_q.weight");
    L.attn_k_weight = LoadWeightBf16(gguf, blk + "attn_k.weight");
    // Dense layers carry a plain `attn_v`. MoVA layers replace it with routed
    // value experts (`attn_v_exps` + `attn_v_gate`), so the plain projection is
    // absent there and must not be required.
    if (!is_mova_layer) {
      L.attn_v_weight = LoadWeightBf16(gguf, blk + "attn_v.weight");
    }
    L.attn_output_weight =
        LoadWeightBf16(gguf, blk + "attn_output.weight");

    try {
      L.attn_gate_weight = LoadWeightBf16(gguf, blk + "attn_gate.weight");
    } catch (...) {
    }

    if (is_mova_layer) {
      L.attn_v_gate_weight = LoadWeightBf16(gguf, blk + "attn_v_gate.weight");
      L.attn_v_gate_bias = LoadWeightBf16(gguf, blk + "attn_v_gate.bias");
      // Kept in its natural [E, Hkv*Dh, H] torch layout so the forward can
      // Slice dim-0 per value expert.
      L.attn_v_exps_weight = LoadWeightBf16(gguf, blk + "attn_v_exps.weight");
    }

    // Optional: a partial checkpoint may omit a layer's FFN entirely, which
    // the forward detects via the (empty) norm and skips that FFN.
    try {
      L.ffn_norm_weight = LoadWeightBf16(gguf, blk + "ffn_norm.weight");
    } catch (...) {
    }

    if (is_dense_layer) {
      // Try merged gate_up first, fall back to separate gate + up.
      bool loaded = false;
      try {
        L.ffn_gate_weight =
            LoadWeightBf16(gguf, blk + "ffn_gate_up.weight");
        loaded = true;
      } catch (...) {
      }
      if (!loaded) {
        try {
          auto gate = LoadWeightBf16(gguf, blk + "ffn_gate.weight");
          auto up = LoadWeightBf16(gguf, blk + "ffn_up.weight");
          // gate is [I, H], up is [I, H]. Concat to [2*I, H].
          const int64_t g_cols = gate.shape[gate.rank > 1 ? 1 : 0];
          const int64_t u_cols = up.shape[up.rank > 1 ? 1 : 0];
          L.ffn_gate_weight = MakeOwned(vt::DType::kBF16, std::vector<int64_t>{2 * I, H});
          auto* dst =
              reinterpret_cast<uint16_t*>(L.ffn_gate_weight.bytes.data());
          const auto* gs =
              reinterpret_cast<const uint16_t*>(gate.bytes.data());
          const auto* us =
              reinterpret_cast<const uint16_t*>(up.bytes.data());
          for (int64_t r = 0; r < I; ++r) {
            std::memcpy(dst + r * H, gs + r * g_cols,
                        static_cast<size_t>(H) * sizeof(uint16_t));
            std::memcpy(dst + (I + r) * H, us + r * u_cols,
                        static_cast<size_t>(H) * sizeof(uint16_t));
          }
        } catch (...) {
        }
      }
      L.ffn_down_weight =
          LoadWeightBf16(gguf, blk + "ffn_down.weight");
    }

    if (!is_dense_layer) {
      try {
        L.ffn_gate_inp_weight =
            LoadWeightBf16(gguf, blk + "ffn_gate_inp.weight");
        L.ffn_gate_exps_weight =
            LoadWeightBf16(gguf, blk + "ffn_gate_exps.weight");
        L.ffn_up_exps_weight =
            LoadWeightBf16(gguf, blk + "ffn_up_exps.weight");
        L.ffn_down_exps_weight =
            LoadWeightBf16(gguf, blk + "ffn_down_exps.weight");
      } catch (...) {
      }
      try {
        L.ffn_gate_shexp_weight =
            LoadWeightBf16(gguf, blk + "ffn_gate_shexp.weight");
        L.ffn_up_shexp_weight =
            LoadWeightBf16(gguf, blk + "ffn_up_shexp.weight");
        L.ffn_down_shexp_weight =
            LoadWeightBf16(gguf, blk + "ffn_down_shexp.weight");
      } catch (...) {
      }
      try {
        L.exp_probs_b_bias =
            LoadWeightBf16(gguf, blk + "exp_probs_b.bias");
      } catch (...) {
      }
    }
  }

  return w;
}

}  // namespace vllm
