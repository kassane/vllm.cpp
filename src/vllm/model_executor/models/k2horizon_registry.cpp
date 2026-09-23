// K2-Horizon (K2HorizonForCausalLM) registry TU.
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/k2horizon.h"
#include "vllm/model_executor/models/k2horizon_gguf_weights.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_common.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

inline constexpr ModelInfo kK2HorizonInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

class K2HorizonLoadedModel final : public LoadedModel {
 public:
  K2HorizonLoadedModel(const ModelRegistration& registration,
                       K2HorizonWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}
  const K2HorizonWeights& weights() const { return weights_; }

 private:
  K2HorizonWeights weights_;
};

std::unique_ptr<LoadedModel> LoadK2HorizonForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kGguf) {
    throw std::runtime_error(
        "K2HorizonForCausalLM currently only supports GGUF weights");
  }
  if (source.gguf == nullptr) {
    throw std::runtime_error("GGUF model source is empty");
  }
  return std::make_unique<K2HorizonLoadedModel>(
      registration, LoadK2HorizonGgufWeights(*source.gguf, config));
}

void ParseK2HorizonConfig(const HfConfig& config) { (void)config; }

void PrepareK2Horizon(LoadedModel& model, const HfConfig& config,
                      vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardK2Horizon(LoadedModel& model,
                               const ModelForwardInput& input) {
  const auto& km =
      ModelAs<K2HorizonLoadedModel>(model, "K2HorizonForCausalLM");
  std::vector<float> host =
      K2HorizonForward(km.weights(), input.config, input);
  return HostLogits(std::move(host), input.config.vocab_size);
}

v1::KVCacheConfig MakeKvCacheK2Horizon(const HfConfig& config, int block_size,
                                       int num_blocks) {
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(
          block_size, num_kv_heads, head_dim, v1::ResolveKvCacheDType()));
  return kv;
}

const ModelFactory kK2HorizonFactory{
    .parse_config = &ParseK2HorizonConfig,
    .load_weights = &LoadK2HorizonForCausalLM,
    .prepare = &PrepareK2Horizon,
    .forward = &ForwardK2Horizon,
    .make_kv_cache = &MakeKvCacheK2Horizon,
    .is_dense_model = true,
};

REGISTER_VLLM_MODEL(k2_horizon, "K2HorizonForCausalLM", kK2HorizonFactory,
                     kK2HorizonInfo)

}  // namespace
}  // namespace vllm
