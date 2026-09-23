// K2-Horizon GGUF config builder. Maps llama.cpp's `k2-horizon` architecture
// metadata keys to the HfConfig vllm.cpp's model registry consumes.
#pragma once

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

HfConfig K2HorizonHfConfigFromGguf(const GgufFile& gguf);

}  // namespace vllm
