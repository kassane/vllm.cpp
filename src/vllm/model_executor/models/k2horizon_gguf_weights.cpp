// K2-Horizon GGUF config builder implementation.
#include "vllm/model_executor/models/k2horizon_gguf_weights.h"

#include <cstdint>
#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

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

double KvFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(KvInt(v, key));
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? KvInt(*v, key) : dflt;
}

double OptFloat(const GgufFile& g, const std::string& key, double dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? KvFloat(*v, key) : dflt;
}

std::string OptStr(const GgufFile& g, const std::string& key,
                   std::string dflt) {
  const GgufValue* v = g.FindKv(key);
  if (v != nullptr && v->TypeId() == kGgufString) {
    return std::get<std::string>(v->v);
  }
  return dflt;
}

// k2-horizon GGUFs use a custom prefix, not "llama.". The reference llama.cpp fork
// writes "k2-horizon.<key>" (e.g. "k2-horizon.block_count").
constexpr const char* kPrefix = "k2-horizon.";

}  // namespace

HfConfig K2HorizonHfConfigFromGguf(const GgufFile& gguf) {
  HfConfig config;

  auto P = [](const char* suffix) -> std::string {
    return std::string(kPrefix) + suffix;
  };

  const int64_t hidden = OptInt(gguf, P("embedding_length"), 2560);
  const int64_t n_heads = OptInt(gguf, P("attention.head_count"), 32);
  const int64_t n_kv = OptInt(gguf, P("attention.head_count_kv"), 8);
  const int64_t ctx_len = OptInt(gguf, P("context_length"), 8192);
  // vocab_size is not a declared KV in k2-horizon GGUFs. Derive it from the
  // embedding table rows (torch [V, H]) rather than a fixed default, so each
  // checkpoint resolves its own vocabulary length.
  int64_t vocab = OptInt(gguf, P("vocab_size"), 0);
  if (vocab <= 0) {
    vocab = 0;
    for (const GgufTensorInfo& t : gguf.Tensors()) {
      if (t.name == "token_embd.weight" && !t.shape.empty()) {
        vocab = t.shape[0];
        break;
      }
    }
  }
  VT_CHECK(vocab > 0, "k2-horizon gguf: cannot resolve vocab_size");
  const int64_t block_count = OptInt(gguf, P("block_count"), 36);
  const int64_t head_dim = OptInt(
      gguf, P("attention.key_length"),
      OptInt(gguf, P("attention.value_length"), 128));
  const int64_t intermediate = OptInt(gguf, P("feed_forward_length"), 10240);
  const double rope_theta =
      OptFloat(gguf, P("rope.freq_base"), 10000000.0);
  const double norm_eps = OptFloat(
      gguf, P("attention.layer_norm_rms_epsilon"), 1e-6);

  config.architectures = {"K2HorizonForCausalLM"};
  config.hidden_size = static_cast<int>(hidden);
  config.num_attention_heads = static_cast<int>(n_heads);
  config.num_key_value_heads = static_cast<int>(n_kv);
  config.head_dim = static_cast<int>(head_dim);
  config.intermediate_size = static_cast<int>(intermediate);
  config.num_hidden_layers = static_cast<int>(block_count);
  config.vocab_size = static_cast<int>(vocab);
  config.max_position_embeddings = static_cast<int>(ctx_len);
  config.rope_theta = rope_theta;
  config.rms_norm_eps = norm_eps;
  config.model_type = "k2-horizon";
  // rotary_dim is the number of rotated dims (rope.dimension_count), which
  // equals head_dim for every k2-horizon checkpoint but is the correct source.
  const int64_t rotary_dim =
      OptInt(gguf, P("rope.dimension_count"), head_dim);
  config.rotary_dim = static_cast<int>(rotary_dim);

  // RoPE parameters feed the shared get_rope() cache builder. Every
  // k2-horizon variant uses NeoX RoPE; the 1B ships YaRN
  // (rope.scaling.type=yarn, factor 16, orig ctx 8192) while the 4B/7B/36B
  // ship plain RoPE. The reference build passes the context RoPE params (n_ctx_orig,
  // freq_base, factor, attn factor, betas) into ggml_rope_ext
  // (build_k2horizon.cpp:199-201), so mirror them here.
  config.rope_parameters.rope_theta = rope_theta;
  config.rope_parameters.rope_dim = rotary_dim;
  const std::string scaling = OptStr(gguf, P("rope.scaling.type"), "default");
  VT_CHECK(scaling == "default" || scaling == "none" || scaling == "yarn",
           "k2-horizon gguf: unsupported rope.scaling.type '" + scaling + "'");
  if (scaling == "yarn") {
    config.rope_parameters.rope_type = "yarn";
    config.rope_parameters.factor =
        OptFloat(gguf, P("rope.scaling.factor"), 1.0);
    config.rope_parameters.original_max_position_embeddings =
        OptInt(gguf, P("rope.scaling.original_context_length"), ctx_len);
    // llama.cpp folds yarn_attn_factor and (1 + 0.1*ln(factor)) into one
    // mscale; this repo computes yarn_get_mscale(factor) * attn_factor, so
    // passing the stored yarn_attn_factor reproduces the reference mscale.
    config.rope_parameters.attn_factor =
        OptFloat(gguf, P("rope.scaling.yarn_attn_factor"), 1.0);
    config.rope_parameters.beta_fast = static_cast<int64_t>(
        OptFloat(gguf, P("rope.scaling.yarn_beta_fast"), 32.0));
    config.rope_parameters.beta_slow = static_cast<int64_t>(
        OptFloat(gguf, P("rope.scaling.yarn_beta_slow"), 1.0));
  }
  config.has_rope_parameters = true;

  return config;
}

}  // namespace vllm
