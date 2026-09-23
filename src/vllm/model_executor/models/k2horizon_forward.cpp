// K2-Horizon forward pass: dense path, softplus attention gate, MoVA routed
// value, and MoE FFN (the 36B; the dense 0.9B/3.7B/7B use the plain arms).
#include "vllm/model_executor/models/k2horizon.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "vllm/model_executor/layers/linear.h"  // UnquantizedMlpGateUpMethod seam
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_common.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using namespace dense_attn;

// The reference's softplus-gate constants (build_k2horizon.cpp: scale by ln 2
// before softplus, then by 1/ln 2). Kept as doubles to match the literals; the
// MulScalar kernel rounds to f32 for the compute, as ggml_scale does.
constexpr double kGateLn2 = 0.6931471805599453;
constexpr double kGateInvLn2 = 1.4426950408889634;
// The value router's division guard (build_k2horizon.cpp:91).
constexpr float kRouteMinSum = 6.103515625e-5f;

bool HasWeight(const OwnedTensor& t) { return !t.bytes.empty(); }

// Download a contiguous device tensor to a host vector of `n` elements
// (the pattern qwen3_5.cpp's MoeBlock uses: backend Copy + Synchronize).
template <typename T>
std::vector<T> DownloadToHost(Dev d, const Tensor& t, int64_t n) {
  std::vector<T> h(static_cast<size_t>(n));
  d.b.Copy(d.q, h.data(), t.data, static_cast<size_t>(n) * sizeof(T));
  d.b.Synchronize(d.q);
  return h;
}

inline float SiluF32(float x) { return x / (1.0f + std::exp(-x)); }

// ── Host top-k expert routing ───────────────────────────────────────
// Mirrors llm_build_moe_ffn / k2_horizon_routed_value selection: gate the
// logits (softmax or sigmoid), select top-k by the BIASED score, then read the
// weights from the UNBIASED probabilities at the selected ids, optionally
// renormalize (the value router clamps the row sum; k2's MoE FFN does not) and
// scale. The device MoeRouterTopK hardcodes softmax and rejects a bias on the
// ungrouped path, so this runs on the host. `bias` may be null. Row-major
// weights [T*K] f32 and ids [T*K] i32 (slot-major within a token).
void HostRouteTopK(const std::vector<float>& logits, int64_t T, int64_t E,
                   int64_t K, int gating, const float* bias, bool renorm,
                   bool clamp_row_sum, float scale, bool apply_scale,
                   std::vector<float>& weights, std::vector<int32_t>& ids) {
  weights.assign(static_cast<size_t>(T) * static_cast<size_t>(K), 0.0f);
  ids.assign(static_cast<size_t>(T) * static_cast<size_t>(K), 0);
  std::vector<float> probs(static_cast<size_t>(E));
  std::vector<float> choice(static_cast<size_t>(E));
  std::vector<char> taken(static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t) {
    const float* lg = logits.data() + t * E;
    if (gating == 1) {  // softmax
      float mx = -std::numeric_limits<float>::infinity();
      for (int64_t e = 0; e < E; ++e) mx = std::max(mx, lg[e]);
      float sum = 0.0f;
      for (int64_t e = 0; e < E; ++e) {
        probs[e] = std::exp(lg[e] - mx);
        sum += probs[e];
      }
      for (int64_t e = 0; e < E; ++e)
        probs[e] = sum > 0.0f ? probs[e] / sum : 0.0f;
    } else {  // sigmoid (k2's expert_gating_func == 2), stable in both tails
      for (int64_t e = 0; e < E; ++e) {
        const float x = lg[e];
        probs[e] = x >= 0.0f ? 1.0f / (1.0f + std::exp(-x))
                             : std::exp(x) / (1.0f + std::exp(x));
      }
    }
    // Selection uses biased scores; the weights come from the unbiased probs.
    for (int64_t e = 0; e < E; ++e)
      choice[e] = probs[e] + (bias != nullptr ? bias[e] : 0.0f);
    std::fill(taken.begin(), taken.end(), 0);
    float row_sum = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
      int64_t best = -1;
      float best_v = -std::numeric_limits<float>::infinity();
      for (int64_t e = 0; e < E; ++e) {
        if (taken[e]) continue;
        if (choice[e] > best_v) {  // strict `>` : lowest index wins a tie
          best_v = choice[e];
          best = e;
        }
      }
      taken[best] = 1;
      const float w = probs[best];  // unbiased weight
      weights[t * K + k] = w;
      ids[t * K + k] = static_cast<int32_t>(best);
      row_sum += w;
    }
    if (renorm) {
      float denom = row_sum;
      if (clamp_row_sum) denom = std::max(denom, kRouteMinSum);
      if (denom > 0.0f)
        for (int64_t k = 0; k < K; ++k) weights[t * K + k] /= denom;
    }
    if (apply_scale)
      for (int64_t k = 0; k < K; ++k) weights[t * K + k] *= scale;
  }
}

// Bucket each (token, slot) pair by its selected expert.
std::vector<std::vector<std::pair<int64_t, int64_t>>> BucketByExpert(
    const std::vector<int32_t>& ids, int64_t T, int64_t K, int64_t E) {
  std::vector<std::vector<std::pair<int64_t, int64_t>>> lists(
      static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t)
    for (int64_t k = 0; k < K; ++k)
      lists[static_cast<size_t>(ids[static_cast<size_t>(t * K + k)])]
          .push_back({t, k});
  return lists;
}

// ── MoVA routed value ───────────────────────────────────────────────
// Replaces the dense V projection with a top-k value-expert mixture
// (build_k2horizon.cpp k2_horizon_routed_value). Returns [T, kdim] bf16.
DBuf K2HorizonMovaValue(Dev d, const K2HorizonLayerWeights& w,
                        const K2HorizonWeights& gw, const HfConfig& cfg,
                        const Tensor& dhn, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t kdim = cfg.num_key_value_heads * cfg.head_dim;
  const int64_t Ev = gw.n_value_experts;
  const int64_t Kv = gw.n_value_expert_used;
  VT_CHECK(Ev > 0 && Kv > 0 && HasWeight(w.attn_v_gate_weight),
           "k2-horizon: MoVA layer missing value-expert router");

  // Router logits [T, Ev] f32.
  std::vector<float> logits(static_cast<size_t>(T) * static_cast<size_t>(Ev));
  {
    DBuf dlog(d, DType::kF32, {T, Ev});
    Tensor wg = ResidentWeight(d, w.attn_v_gate_weight, {Ev, H});
    vt::MatmulBT(d.q, dlog.t(), dhn, wg);
    dlog.Download(d, logits.data());
  }

  // Selection bias (unbiased weights).
  std::vector<float> bias_host;
  const float* bias = nullptr;
  if (HasWeight(w.attn_v_gate_bias)) {
    const auto* b =
        reinterpret_cast<const uint16_t*>(w.attn_v_gate_bias.bytes.data());
    bias_host.resize(static_cast<size_t>(Ev));
    for (int64_t e = 0; e < Ev; ++e) bias_host[e] = vt::BF16ToF32(b[e]);
    bias = bias_host.data();
  }

  std::vector<float> route_w;
  std::vector<int32_t> route_id;
  // Value router: renorm clamps the row sum; scale applies when not {0,1}.
  HostRouteTopK(logits, T, Ev, Kv, static_cast<int>(gw.expert_gating_func),
                bias, gw.expert_weights_norm, /*clamp_row_sum=*/true,
                gw.expert_weights_scale,
                gw.expert_weights_scale != 0.0f &&
                    gw.expert_weights_scale != 1.0f,
                route_w, route_id);

  // Hidden states on host for the expert gather.
  std::vector<uint16_t> xh = DownloadToHost<uint16_t>(d, dhn, T * H);
  auto lists = BucketByExpert(route_id, T, Kv, Ev);

  // Stacked value experts, viewed [Ev*kdim, H] so each expert is a row Slice.
  Tensor wexp = ResidentWeight(d, w.attn_v_exps_weight, {Ev * kdim, H});

  std::vector<uint16_t> expert_out(
      static_cast<size_t>(T) * static_cast<size_t>(Kv) * static_cast<size_t>(kdim),
      0);
  for (int64_t e = 0; e < Ev; ++e) {
    const auto& slots = lists[static_cast<size_t>(e)];
    if (slots.empty()) continue;
    const int64_t n = static_cast<int64_t>(slots.size());
    // Gather the rows routed to this expert: [n, H].
    std::vector<uint16_t> xg(static_cast<size_t>(n) * static_cast<size_t>(H));
    for (int64_t i = 0; i < n; ++i) {
      const int64_t t = slots[static_cast<size_t>(i)].first;
      std::memcpy(xg.data() + i * H, xh.data() + t * H,
                  static_cast<size_t>(H) * sizeof(uint16_t));
    }
    // value = silu(x_gathered @ W_e^T), W_e = [kdim, H].
    DBuf dx(d, DType::kBF16, {n, H}, xg.data());
    DBuf dy(d, DType::kBF16, {n, kdim});
    vt::MatmulBT(d.q, dy.t(), dx.t(), wexp.Slice(0, e * kdim, (e + 1) * kdim));
    std::vector<uint16_t> y(static_cast<size_t>(n) * static_cast<size_t>(kdim));
    dy.Download(d, y.data());
    for (int64_t i = 0; i < n; ++i) {
      const int64_t t = slots[static_cast<size_t>(i)].first;
      const int64_t k = slots[static_cast<size_t>(i)].second;
      const uint16_t* row = y.data() + i * kdim;
      uint16_t* dst = expert_out.data() +
                      (static_cast<size_t>(t) * static_cast<size_t>(Kv) +
                       static_cast<size_t>(k)) *
                          static_cast<size_t>(kdim);
      for (int64_t j = 0; j < kdim; ++j)
        dst[j] = vt::F32ToBF16(SiluF32(vt::BF16ToF32(row[j])));
    }
  }

  // value_out[t,:] = sum_k w[t,k] * expert_out[t,k,:].
  DBuf deo(d, DType::kBF16,
           {T, Kv, kdim}, expert_out.data());
  DBuf drw(d, DType::kF32, {T, Kv}, route_w.data());
  DBuf out(d, DType::kBF16, {T, kdim});
  vt::MoeCombine(d.q, out.t(), deo.t(), drw.t(), /*shared=*/nullptr,
                 /*routed_scale=*/1.0f);
  return out;
}

// ── Attention block ─────────────────────────────────────────────────
DBuf K2HorizonAttnBlock(Dev d, const K2HorizonLayerWeights& w,
                        const K2HorizonWeights& gw, const Tensor& rope_cache,
                        const HfConfig& cfg, const Tensor& dhn,
                        const StepInputs& si,
                        const CommonAttentionMetadata& meta,
                        const PagedKvCache& kv, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t Hq = cfg.num_attention_heads;
  const int64_t Hkv = cfg.num_key_value_heads;
  const int64_t Dh = cfg.head_dim;
  const int64_t qdim = Hq * Dh;
  const int64_t kdim = Hkv * Dh;

  DBuf q(d, DType::kBF16, {T, qdim});
  {
    Tensor wq = ResidentWeight(d, w.attn_q_weight, {qdim, H});
    vt::MatmulBT(d.q, q.t(), dhn, wq);
  }

  DBuf k(d, DType::kBF16, {T, kdim});
  {
    Tensor wk = ResidentWeight(d, w.attn_k_weight, {kdim, H});
    vt::MatmulBT(d.q, k.t(), dhn, wk);
  }

  // V projection: dense (plain attn_v) or MoVA (routed value experts). The
  // loader leaves attn_v_exps present only on MoVA layers.
  DBuf v = [&]() -> DBuf {
    if (HasWeight(w.attn_v_exps_weight))
      return K2HorizonMovaValue(d, w, gw, cfg, dhn, T);
    DBuf vd(d, DType::kBF16, {T, kdim});
    Tensor wv = ResidentWeight(d, w.attn_v_weight, {kdim, H});
    vt::MatmulBT(d.q, vd.t(), dhn, wv);
    return vd;
  }();

  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  {
    vt::RopeArgs ra;
    ra.rotary_dim = static_cast<int>(Dh);
    // k2-horizon uses NeoX-style RoPE (llama.cpp:9680).
    ra.is_neox_style = true;
    Tensor k3v = k3;
    vt::RopeFromCache(d.q, q3, &k3v, si.positions.t(), rope_cache, ra);
  }

  Tensor kw = k3;
  Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});
  Tensor vw = v3;
  DBuf kcast(d, kv.dtype, {T, Hkv, Dh});
  DBuf vcast(d, kv.dtype, {T, Hkv, Dh});
  if (kv.dtype != DType::kBF16) {
    vt::CastF32(d.q, kcast.t(), k3);
    vt::CastF32(d.q, vcast.t(), v3);
    kw = kcast.t();
    vw = vcast.t();
  }
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, si.slot_mapping.t());

  DBuf attn(d, DType::kBF16, {T, Hq, Dh});
  const float scale = 1.0F / std::sqrt(static_cast<float>(Dh));
  vt::PagedAttentionArgs pa{scale, meta.causal};
  pa.query_start_loc_host = meta.query_start_loc.data();
  pa.max_seq_len = meta.max_seq_len;
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, si.block_table.t(),
                     si.seq_lens.t(), si.query_start_loc.t(), pa);

  Tensor o_in = Reshape(attn.t(), {T, Hq * Dh});
  // Softplus attention gate: gate = softplus(x·ln2)/ln2 over the normed input,
  // applied elementwise to the attention output before the wo projection
  // (build_k2horizon.cpp:229-236). Present only on layers that carry
  // attn_gate.weight — every 36B block, no dense 0.9B/3.7B/7B block.
  Tensor src = o_in;
  DBuf gated(d, DType::kBF16, {T, qdim});
  if (HasWeight(w.attn_gate_weight)) {
    DBuf g(d, DType::kBF16, {T, qdim});
    {
      Tensor wg = ResidentWeight(d, w.attn_gate_weight, {qdim, H});
      vt::MatmulBT(d.q, g.t(), dhn, wg);
    }
    vt::MulScalar(d.q, g.t(), g.t(), kGateLn2);
    vt::Softplus(d.q, g.t(), g.t());
    vt::MulScalar(d.q, g.t(), g.t(), kGateInvLn2);
    vt::Mul(d.q, gated.t(), o_in, g.t());
    src = gated.t();
  }

  DBuf o(d, DType::kBF16, {T, H});
  {
    Tensor wo = ResidentWeight(d, w.attn_output_weight, {H, Hq * Dh});
    vt::MatmulBT(d.q, o.t(), src, wo);
  }
  return o;
}

// ── SwiGLU FFN (dense path) ────────────────────────────────────────
DBuf K2HorizonDenseFfnBlock(Dev d, const K2HorizonLayerWeights& w,
                            const HfConfig& cfg, const Tensor& dhn,
                            int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t I = cfg.intermediate_size;

  // Shared merged-GEMM seam: same merged [2I, H] operand and byte-for-byte the
  // same MatmulBT + SiluAndMul the inline path issued.
  DBuf act =
      layers::UnquantizedMlpGateUpMethod(&w.ffn_gate_weight, I).Apply(d, dhn);

  DBuf out(d, DType::kBF16, {T, H});
  {
    Tensor wd = ResidentWeight(d, w.ffn_down_weight, {H, I});
    vt::MatmulBT(d.q, out.t(), act.t(), wd);
  }
  return out;
}

// ── MoE FFN + shared expert ────────────────────────────────────────
// Top-k routed experts with a host gather (build_k2horizon.cpp's MoE branch /
// llm_build_moe_ffn) plus the dense shared expert, combined through
// MoeCombine. Returns [T, H] bf16.
DBuf K2HorizonMoeFfnBlock(Dev d, const K2HorizonLayerWeights& w,
                          const K2HorizonWeights& gw, const HfConfig& cfg,
                          const Tensor& ffn_normed, int64_t T) {
  const int64_t H = cfg.hidden_size;
  const int64_t E = gw.expert_count;
  const int64_t K = gw.expert_used_count;
  const int64_t Ie = gw.expert_ffn_size;
  const int64_t Is = gw.shared_ffn_size;
  VT_CHECK(E > 0 && K > 0 && Ie > 0 && HasWeight(w.ffn_gate_inp_weight),
           "k2-horizon: MoE layer missing router or expert intermediate size");
  VT_CHECK(HasWeight(w.ffn_gate_exps_weight) &&
               HasWeight(w.ffn_up_exps_weight) &&
               HasWeight(w.ffn_down_exps_weight),
           "k2-horizon: MoE layer missing expert weights");

  // Router logits [T, E] f32.
  std::vector<float> logits(static_cast<size_t>(T) * static_cast<size_t>(E));
  {
    DBuf dlog(d, DType::kF32, {T, E});
    Tensor wr = ResidentWeight(d, w.ffn_gate_inp_weight, {E, H});
    vt::MatmulBT(d.q, dlog.t(), ffn_normed, wr);
    dlog.Download(d, logits.data());
  }

  std::vector<float> bias_host;
  const float* bias = nullptr;
  if (HasWeight(w.exp_probs_b_bias)) {
    const auto* b =
        reinterpret_cast<const uint16_t*>(w.exp_probs_b_bias.bytes.data());
    bias_host.resize(static_cast<size_t>(E));
    for (int64_t e = 0; e < E; ++e) bias_host[e] = vt::BF16ToF32(b[e]);
    bias = bias_host.data();
  }

  std::vector<float> route_w;
  std::vector<int32_t> route_id;
  // k2 is not LAGUNA/BAILING/STEP35: the MoE row sum is NOT clamped, and its
  // call passes scale_w=true gated on |scale-1|>1e-5.
  HostRouteTopK(logits, T, E, K, static_cast<int>(gw.expert_gating_func), bias,
                gw.expert_weights_norm, /*clamp_row_sum=*/false,
                gw.expert_weights_scale,
                std::fabs(gw.expert_weights_scale - 1.0f) > 1e-5f, route_w,
                route_id);

  // Hidden states on host for the expert gather.
  std::vector<uint16_t> xh = DownloadToHost<uint16_t>(d, ffn_normed, T * H);
  auto lists = BucketByExpert(route_id, T, K, E);

  // Stacked expert weights, viewed so each expert is a contiguous row Slice.
  Tensor wgate = ResidentWeight(d, w.ffn_gate_exps_weight, {E * Ie, H});
  Tensor wup = ResidentWeight(d, w.ffn_up_exps_weight, {E * Ie, H});
  Tensor wdown = ResidentWeight(d, w.ffn_down_exps_weight, {E * H, Ie});

  std::vector<uint16_t> expert_out(
      static_cast<size_t>(T) * static_cast<size_t>(K) * static_cast<size_t>(H),
      0);
  for (int64_t e = 0; e < E; ++e) {
    const auto& slots = lists[static_cast<size_t>(e)];
    if (slots.empty()) continue;
    const int64_t n = static_cast<int64_t>(slots.size());
    // Gather the rows routed to this expert: [n, H].
    std::vector<uint16_t> xg(static_cast<size_t>(n) * static_cast<size_t>(H));
    for (int64_t i = 0; i < n; ++i) {
      const int64_t t = slots[static_cast<size_t>(i)].first;
      std::memcpy(xg.data() + i * H, xh.data() + t * H,
                  static_cast<size_t>(H) * sizeof(uint16_t));
    }
    DBuf dx(d, DType::kBF16, {n, H}, xg.data());
    DBuf dg(d, DType::kF32, {n, Ie});
    DBuf du(d, DType::kF32, {n, Ie});
    vt::MatmulBT(d.q, dg.t(), dx.t(), wgate.Slice(0, e * Ie, (e + 1) * Ie));
    vt::MatmulBT(d.q, du.t(), dx.t(), wup.Slice(0, e * Ie, (e + 1) * Ie));
    // SwiGLU: silu(gate) * up.
    DBuf da(d, DType::kF32, {n, Ie});
    vt::MoeSiluMul(d.q, da.t(), dg.t(), du.t());
    DBuf dy(d, DType::kF32, {n, H});
    vt::MatmulBT(d.q, dy.t(), da.t(), wdown.Slice(0, e * H, (e + 1) * H));
    std::vector<float> y(static_cast<size_t>(n) * static_cast<size_t>(H));
    dy.Download(d, y.data());
    for (int64_t i = 0; i < n; ++i) {
      const int64_t t = slots[static_cast<size_t>(i)].first;
      const int64_t k = slots[static_cast<size_t>(i)].second;
      uint16_t* dst = expert_out.data() +
                      (static_cast<size_t>(t) * static_cast<size_t>(K) +
                       static_cast<size_t>(k)) *
                          static_cast<size_t>(H);
      for (int64_t j = 0; j < H; ++j)
        dst[j] = vt::F32ToBF16(y[static_cast<size_t>(i) * H +
                                 static_cast<size_t>(j)]);
    }
  }

  // Shared expert: dense SwiGLU at the (smaller) shared size, on device. When
  // a checkpoint carries no shared-expert weights, combine with no shared term.
  DBuf shared(d, DType::kBF16, {T, H});
  const bool has_shared = HasWeight(w.ffn_gate_shexp_weight) &&
                          HasWeight(w.ffn_up_shexp_weight) &&
                          HasWeight(w.ffn_down_shexp_weight);
  if (has_shared) {
    DBuf g(d, DType::kF32, {T, Is});
    DBuf u(d, DType::kF32, {T, Is});
    {
      Tensor wgs = ResidentWeight(d, w.ffn_gate_shexp_weight, {Is, H});
      Tensor wus = ResidentWeight(d, w.ffn_up_shexp_weight, {Is, H});
      vt::MatmulBT(d.q, g.t(), ffn_normed, wgs);
      vt::MatmulBT(d.q, u.t(), ffn_normed, wus);
    }
    DBuf a(d, DType::kF32, {T, Is});
    vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
    Tensor wds = ResidentWeight(d, w.ffn_down_shexp_weight, {H, Is});
    vt::MatmulBT(d.q, shared.t(), a.t(), wds);
  }

  DBuf deo(d, DType::kBF16, {T, K, H}, expert_out.data());
  DBuf drw(d, DType::kF32, {T, K}, route_w.data());
  DBuf out(d, DType::kBF16, {T, H});
  Tensor shared_t = shared.t();
  const Tensor* shared_p = has_shared ? &shared_t : nullptr;
  vt::MoeCombine(d.q, out.t(), deo.t(), drw.t(), shared_p,
                 /*routed_scale=*/1.0f);
  return out;
}

// ── Grouped RmsNorm helper ──────────────────────────────────────────
// k2-horizon normalizes each consecutive chunk of H/n_groups elements
// independently (build_k2horizon.cpp:19-23 reshapes
// (H,T)->(H/G,G,T) then rms_norms over ne[0]). vt::RmsNormGroup reduces over
// `group_size` consecutive elements, so group_size = H/n_groups. n_groups==1
// degenerates to a whole-row norm, which is what vt::RmsNorm already does.
void RmsNormInto(Dev d, DBuf& out, const Tensor& x, const OwnedTensor& wt,
                 float eps, int64_t H, int64_t n_groups) {
  Tensor w = ResidentWeight(d, wt, {H});
  VT_CHECK(n_groups > 0 && H % n_groups == 0,
           "k2-horizon: hidden_size must be divisible by n_norm_groups");
  vt::RmsNormGroupArgs ga;
  ga.eps = eps;
  ga.group_size = H / n_groups;
  vt::RmsNormGroup(d.q, out.t(), x, w, ga);
}

// ── Full forward body ───────────────────────────────────────────────
std::vector<float> ForwardBody(Dev d, const std::vector<int32_t>& token_ids,
                               const std::vector<int32_t>& positions,
                               const CommonAttentionMetadata& attn_meta,
                               const std::vector<PagedKvCache>& attn_kv,
                               const K2HorizonWeights& weights,
                               const HfConfig& config, float eps) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = config.hidden_size;
  const int64_t vocab = config.vocab_size;
  VT_CHECK(static_cast<int64_t>(positions.size()) == T,
           "k2-horizon: positions length must match token_ids");
  VT_CHECK(static_cast<int64_t>(attn_kv.size()) == config.num_hidden_layers,
           "k2-horizon: one PagedKvCache per layer required");

  Tensor rope_cache = ResidentWeight(d, weights.rope_cos_sin);

  DBuf hidden(d, DType::kBF16, {T, H});
  {
    Tensor dtab = ResidentWeight(d, weights.embed_tokens, {vocab, H});
    DBuf dids(d, DType::kI32, {T}, token_ids.data());
    vt::Embedding(d.q, hidden.t(), dtab, dids.t());
  }

  StepInputs si = BuildStepInputs(d, positions, attn_meta, config);

  for (int64_t il = 0; il < config.num_hidden_layers; ++il) {
    const auto& L = weights.layers[static_cast<size_t>(il)];

    DBuf normed(d, DType::kBF16, {T, H});
    RmsNormInto(d, normed, hidden.t(), L.attn_norm_weight, eps, H,
                weights.n_norm_groups);

    DBuf attn_out = K2HorizonAttnBlock(d, L, weights, rope_cache, config,
                                       normed.t(), si, attn_meta,
                                       attn_kv[static_cast<size_t>(il)], T);

    vt::Add(d.q, hidden.t(), hidden.t(), attn_out.t());

    // A partial checkpoint may omit a layer's FFN entirely (the APEX-mini's
    // blk.3 has no ffn tensors); skip it when the norm is absent rather than
    // refusing the file. Otherwise dense vs MoE by layer class, matching the
    // reference (n_expert>0 && il>=leading_dense).
    if (!L.ffn_norm_weight.bytes.empty()) {
      DBuf ffn_normed(d, DType::kBF16, {T, H});
      RmsNormInto(d, ffn_normed, hidden.t(), L.ffn_norm_weight, eps, H,
                  weights.n_norm_groups);
      const bool is_moe = weights.expert_count > 0 &&
                          il >= weights.leading_dense_block_count;
      DBuf ffn_out = [&]() -> DBuf {
        if (is_moe)
          return K2HorizonMoeFfnBlock(d, L, weights, config, ffn_normed.t(), T);
        return K2HorizonDenseFfnBlock(d, L, config, ffn_normed.t(), T);
      }();
      vt::Add(d.q, hidden.t(), hidden.t(), ffn_out.t());
    }
  }

  DBuf dnorm(d, DType::kBF16, {T, H});
  RmsNormInto(d, dnorm, hidden.t(), weights.final_norm_weight, eps, H,
              weights.n_norm_groups);

  // LM head — download to host logits directly.
  const int64_t n_out = T;
  Tensor lm = ResidentWeight(d, weights.lm_head, {vocab, H});
  DBuf logits(d, DType::kF32, {n_out, vocab});
  vt::MatmulBT(d.q, logits.t(), dnorm.t(), lm);

  std::vector<float> host(static_cast<size_t>(n_out * vocab));
  logits.Download(d, host.data());
  return host;
}

}  // namespace

std::vector<float> K2HorizonForward(const K2HorizonWeights& weights,
                                    const HfConfig& config,
                                    const ModelForwardInput& input) {
  Dev d{vt::GetBackend(input.queue.device.type), input.queue};
  const float eps = static_cast<float>(config.rms_norm_eps);
  return ForwardBody(d, input.token_ids, input.positions, input.attn_meta,
                     input.attn_kv, weights, config, eps);
}

}  // namespace vllm
