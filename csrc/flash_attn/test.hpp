#pragma once

#include <string>
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>    // for at::cuda::CUDAGuard
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/command_line.h>

#include "dtype_mapping.hpp"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_launch.hpp"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


namespace test_fa{

enum AttnBiasType {
  SHARE_BATCH_HEAD = 0,
  SHARE_HEAD = 1,
  PER_HEAD = 2
};


std::ostream& operator<<(std::ostream& out, const AttnBiasType& type) {
  switch (type) {
    case AttnBiasType::SHARE_BATCH_HEAD:
      out << "SHARE_BATCH_HEAD";
      break;
    case AttnBiasType::SHARE_HEAD:
      out << "SHARE_HEAD";
      break;
    case AttnBiasType::PER_HEAD:
      out << "PER_HEAD";
      break;
    default:
      out << "Unknown AttnBiasType";
  }
  return out;
}

std::istream& operator>>(std::istream& in,  AttnBiasType& type) {
  std::string token;
  in >> token;
  if (token == "SHARE_BATCH_HEAD") {
    type = AttnBiasType::SHARE_BATCH_HEAD;
  } else if (token == "SHARE_HEAD") {
    type = AttnBiasType::SHARE_HEAD;
  } else if (token == "PER_HEAD") {
    type = AttnBiasType::PER_HEAD;
  } else {
    throw std::runtime_error("Unknown AttnBiasType: " + token);
  }
  return in;
}


struct Options {

  bool help = false;
  bool error = false;

  int num_runs = 1;
  int batch_size = 32;
  int seq_length = 2048;
  int num_heads = 12;
  int head_size = 64;

  bool var_len = false;
  // bool cross_attn = false;
  bool causal = false;

  // attention bias configuration
  bool attn_bias = false;
  bool pad_bias_to_multiple_of_8 = false;
  AttnBiasType attn_bias_type = AttnBiasType::SHARE_BATCH_HEAD;

  bool check_ref = false;

  Options() { }

  // Parses the command line
  void parse(int argc, const char **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("num_runs", num_runs, 1);
    cmd.get_cmd_line_argument("batch_size", batch_size, 32);
    cmd.get_cmd_line_argument("seq_length", seq_length, 2048);
    cmd.get_cmd_line_argument("num_heads", num_heads, 12);
    cmd.get_cmd_line_argument("head_size", head_size, 64);

    cmd.get_cmd_line_argument("var_len", var_len, false);
    // cmd.get_cmd_line_argument("cross_attn", cross_attn, false);
    cmd.get_cmd_line_argument("causal", causal, false);
    cmd.get_cmd_line_argument("attn_bias", attn_bias, false);
    cmd.get_cmd_line_argument("pad_bias_to_multiple_of_8", pad_bias_to_multiple_of_8, false);
    cmd.get_cmd_line_argument("attn_bias_type", attn_bias_type, AttnBiasType::SHARE_BATCH_HEAD);

    cmd.get_cmd_line_argument("check_ref", check_ref, true);
  }

  void print_usage() const {

    std::cout << "TestFA\n\n"
      << "Options:\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --num_runs=<int>            Number of runs to average performance (default: --num_runs=1)\n"
      << "  --batch_size=<int>          Batch size in multi-head attention (default: --batch_size=16)\n"
      << "  --seq_length=<int>          Sequence length in multi-head attention for Q (default: --seq_length=1024)\n"
      << "  --num_heads=<int>         Head number in multi-head attention (default: --num_heads=12)\n"
      << "  --head_size=<int>           Head size in multi-head attention (default: --head_size=64)\n"
      << "  --var_len                   If specified, use variable length sequences (default: false)\n"
      // << "  --cross_attn                If specified, use cross attention (default: false)\n"
      << "  --causal                    If specified, use causal attention (default: false)\n"
      << "  --attn_bias                 If specified, use custom attention bias (default: false)\n"
      << "  --pad_bias_to_multiple_of_8 If specified, pad attention bias to multiple of 8 (default: false)\n"
      << "  --attn_bias_type=<type>     Attention bias type: SHARE_BATCH_HEAD, SHARE_HEAD, PER_HEAD (default: SHARE_BATCH_HEAD)\n"
      << "  --check_ref                 If specified, check results against reference implementation (default: false)\n"
      << std::endl;

  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {
    return double{};
  }

  friend std::ostream& operator<<(std::ostream& out, const Options& options) {
    out << "Options:\n"
        << "  num_runs: " << options.num_runs << "\n"
        << "  batch_size: " << options.batch_size << "\n"
        << "  seq_length: " << options.seq_length << "\n"
        << "  num_heads: " << options.num_heads << "\n"
        << "  head_size: " << options.head_size << "\n"
        << "  var_len: " << (options.var_len ? "true" : "false") << "\n"
        // << "  cross_attn: " << (options.cross_attn ? "true" : "false") << "\n"
        << "  causal: " << (options.causal ? "true" : "false") << "\n"
        << "  attn_bias: " << (options.attn_bias ? "true" : "false") << "\n"
        << "  pad_bias_to_multiple_of_8: " << (options.pad_bias_to_multiple_of_8 ? "true" : "false") << "\n"
        << "  attn_bias_type: " << options.attn_bias_type << "\n"
        << "  check_ref: " << (options.check_ref ? "true" : "false") << "\n";
    return out;
  }
};


template<
  typename scalar_t_,
  typename scalar_ref_t_,
  typename accum_t_
>
class DataBatch {
public:
  using scalar_t = scalar_t_;
  using scalar_ref_t = scalar_ref_t_;  // when using torch, need this ref type tensor
  using accum_t = accum_t_;
  static constexpr auto torch_scalar_t = CutlassToTorch<scalar_t>::value;
  static constexpr auto torch_scalar_ref_t = CutlassToTorch<scalar_ref_t>::value;
  static constexpr auto torch_accum_t = CutlassToTorch<accum_t>::value;

  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;
  torch::Tensor q_ref;
  torch::Tensor k_ref;
  torch::Tensor v_ref;
  torch::Tensor cu_seqlens;
  int max_seqlen;
  int total_seqlen;
  torch::Tensor bias;
  torch::Tensor bias_ref;
  torch::Tensor o_ref;

  DataBatch(const Options& options, torch::DeviceType device) {

    if (options.var_len) {
      // Generate random sequence lengths
      auto seqlens = torch::randint(1, options.seq_length + 1, {options.batch_size}, torch::kInt32);
      cu_seqlens = torch::zeros({options.batch_size + 1}, torch::dtype(torch::kInt32));
      cu_seqlens.slice(0, 1, options.batch_size + 1) = torch::cumsum(seqlens, 0);
      max_seqlen = seqlens.max().item<int>();

      total_seqlen = cu_seqlens[options.batch_size].item<int>();

      cu_seqlens = cu_seqlens.to(device);

      // q, k, v: [total_seqlen, num_heads, head_size]
      init_qkv_(options.check_ref, {total_seqlen, options.num_heads, options.head_size}, device);
    } else {
      // q, k, v: [batch_size, seq_length, num_heads, head_size]
      init_qkv_(options.check_ref, {options.batch_size, options.seq_length, options.num_heads, options.head_size}, device);
    }

    if (options.attn_bias) {
      int bias_length = options.var_len ? max_seqlen : options.seq_length;
      if (options.pad_bias_to_multiple_of_8) {
        bias_length = ((bias_length + 7) / 8) * 8;
      }
      init_bias_(
        bias_length, options.attn_bias_type, options.batch_size, options.num_heads,
        device, options.check_ref
      );
    }

    if (options.check_ref) {
      o_ref = torch::zeros_like(q_ref, torch::dtype(torch_scalar_ref_t));
      // compute o_ref, TODO
    }
  }

private:
  void init_qkv_(bool init_ref, at::IntArrayRef size, torch::DeviceType device) {
      init_tensor_with_ref_(q, q_ref, size, device, torch_scalar_t, torch_scalar_ref_t, init_ref);
      init_tensor_with_ref_(k, k_ref, size, device, torch_scalar_t, torch_scalar_ref_t, init_ref);
      init_tensor_with_ref_(v, v_ref, size, device, torch_scalar_t, torch_scalar_ref_t, init_ref);
  }

  void init_bias_(
    int bias_length,
    AttnBiasType bias_type,
    int batch_size,
    int num_heads,
    torch::DeviceType device,
    bool with_ref = true
  ) {
    at::IntArrayRef bias_size;
    switch (bias_type) {
      case AttnBiasType::SHARE_BATCH_HEAD:
        bias_size = {1, 1, bias_length, bias_length};
        break;
      case AttnBiasType::SHARE_HEAD:
        bias_size = {1, num_heads, bias_length, bias_length};
        break;
      case AttnBiasType::PER_HEAD:
        bias_size = {batch_size, num_heads, bias_length, bias_length};
        break;
      default:
        throw std::runtime_error("Unknown AttnBiasType");
    }
    init_tensor_with_ref_(
      bias,
      bias_ref,
      bias_size,
      device,
      torch_accum_t,
      torch_scalar_ref_t,
      with_ref
    );
  }

  void init_tensor_with_ref_(
    torch::Tensor& tensor,
    torch::Tensor& tensor_ref,
    at::IntArrayRef size,
    torch::DeviceType device,
    torch::ScalarType dtype = torch_scalar_t,
    torch::ScalarType ref_dtype = torch_scalar_ref_t,
    bool with_ref = true
  ) {
    if (with_ref) {
      tensor_ref = torch::randn(
        size,
        torch::device(device).dtype(ref_dtype)
      );
      tensor = tensor_ref.to(dtype);
    } else {
      tensor = torch::randn(
        size,
        torch::device(device).dtype(dtype)
      );
    }
  }
};


template<
  typename scalar_t_ = cutlass::half_t,
  typename scalar_ref_t_ = cutlass::tfloat32_t,
  typename accum_t_ = cutlass::tfloat32_t
>
class TestFA {
public:
  using scalar_t = scalar_t_;
  using scalar_ref_t = scalar_ref_t_;         // when using torch, need this ref type tensor
  using accum_t = accum_t_;

  TestFA(const Options& options_) : options(options_) {
    initialize();
  }

  void initialize() {
    auto device = torch::kCPU;
    if (torch::cuda::is_available()) {
      device = torch::kCUDA;
    } else {
      std::cout << "Not detecting any cuda device, using cpu instead" << std::endl;
    }
    for (int i = 0; i < options.num_runs; ++i) {
      data.emplace_back(options, device);
    }
  }

  std::vector<DataBatch<scalar_t, scalar_ref_t, accum_t>> data;

private:
  Options options;


};



class FlashKernel {
public:
  FlashKernel() {}

  std::vector<at::Tensor> forward(
    at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const std::optional<at::Tensor> &cu_seqlens,  // b+1
    std::optional<int> max_seqlen,
    std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const std::optional<float> softmax_scale_,
    bool is_causal
  ) {
    bool is_varlen = cu_seqlens.has_value();

    // checks
    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};

    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "FlashAttention only supports Ampere GPUs or newer.");

    CHECK_DEVICE(q);
    CHECK_DEVICE(k);
    CHECK_DEVICE(v);
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    if (is_varlen) {
      TORCH_CHECK(cu_seqlens.value().dtype() == torch::kInt32, "cu_seqlens must have dtype int32");
    }

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    if (is_varlen) {
      CHECK_CONTIGUOUS(cu_seqlens.value());
    }

    int batch_size = 0, seqlen = 0, num_heads = 0, head_size = 0;

    if (is_varlen) {
      batch_size = cu_seqlens.value().size(0) - 1;
      seqlen = max_seqlen.value();
      num_heads = q.size(1);
      head_size = q.size(2);
    } else {
      batch_size = q.size(0);
      seqlen = q.size(1);
      num_heads = q.size(2);
      head_size = q.size(3);
    }
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (is_varlen) {
            CHECK_SHAPE(out, cu_seqlens.value()[-1].item<int>(), num_heads, head_size);
        } else {
            CHECK_SHAPE(out, batch_size, seqlen, num_heads, head_size);
        }
    } else {
        out = torch::empty_like(q);
    }

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
    const int seqlen_rounded = round_multiple(seqlen, 128);

    float softmax_scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    if (softmax_scale_.has_value()) {
        softmax_scale = softmax_scale_.value();
    }

    torch::TensorOptions opts = q.options();
    at::Tensor softmax_lse;
    if (is_varlen) {
      softmax_lse = torch::empty({num_heads, q.size(0)}, opts.dtype(at::kFloat));
    } else {
      softmax_lse = torch::empty({batch_size, num_heads, seqlen}, opts.dtype(at::kFloat));
    }

    // assemble fwd params
    FLASH_NAMESPACE::Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen, seqlen,
                     seqlen_rounded, seqlen_rounded,
                     num_heads, num_heads,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     is_varlen ? cu_seqlens.value().data_ptr() : nullptr,
                     is_varlen ? cu_seqlens.value().data_ptr() : nullptr,
                     /*seqused_k*/nullptr,
                     /*p_d, softmax_o*/nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout*/0.f,
                     softmax_scale,
                     /*window_size_left*/-1,
                     /*window_size_right*/-1,
                     /*softcap*/0.f,
                     /*seqlenq_ngroups_swapped*/false,
                     /*unpadded_lse*/true);

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    auto rng_state = torch::empty(
      {2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)
    );
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

    // launch kernel
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    run_mha_fwd(params, stream);

    return {out, softmax_lse, rng_state};

  }

  // void backward() {}


private:
  static void set_params_fprop(
    FLASH_NAMESPACE::Flash_fwd_params &params,
    // sizes
    const size_t b,
    const size_t seqlen_q,
    const size_t seqlen_k,
    const size_t seqlen_q_rounded,
    const size_t seqlen_k_rounded,
    const size_t h,
    const size_t h_k,
    const size_t d,
    const size_t d_rounded,
    // device pointers
    const at::Tensor q,
    const at::Tensor k,
    const at::Tensor v,
    at::Tensor out,
    void *cu_seqlens_q_d,
    void *cu_seqlens_k_d,
    void *seqused_k,
    void *p_d,
    void *softmax_lse_d,
    float p_dropout,
    float softmax_scale,
    int window_size_left,
    int window_size_right,
    const float softcap,
    bool seqlenq_ngroups_swapped=false,
    const bool unpadded_lse=false
  ) {

      // Reset the parameters
      params = {};

      params.is_bf16 = q.dtype() == torch::kBFloat16;

      // Set the pointers and strides.
      params.q_ptr = q.data_ptr();
      params.k_ptr = k.data_ptr();
      params.v_ptr = v.data_ptr();
      // All stride are in elements, not bytes.
      params.q_row_stride = q.stride(-3);
      params.k_row_stride = k.stride(-3);
      params.v_row_stride = v.stride(-3);
      params.q_head_stride = q.stride(-2);
      params.k_head_stride = k.stride(-2);
      params.v_head_stride = v.stride(-2);
      params.o_ptr = out.data_ptr();
      params.o_row_stride = out.stride(-3);
      params.o_head_stride = out.stride(-2);

      if (cu_seqlens_q_d == nullptr) {
          params.q_batch_stride = q.stride(0);
          params.k_batch_stride = k.stride(0);
          params.v_batch_stride = v.stride(0);
          params.o_batch_stride = out.stride(0);
          if (seqlenq_ngroups_swapped) {
              params.q_batch_stride *= seqlen_q;
              params.o_batch_stride *= seqlen_q;
          }
      }

      params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
      params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
      params.seqused_k = static_cast<int *>(seqused_k);

      // P = softmax(QK^T)
      params.p_ptr = p_d;

      // Softmax sum
      params.softmax_lse_ptr = softmax_lse_d;

      // Set the dimensions.
      params.b = b;
      params.h = h;
      params.h_k = h_k;
      params.h_h_k_ratio = h / h_k;
      params.seqlen_q = seqlen_q;
      params.seqlen_k = seqlen_k;
      params.seqlen_q_rounded = seqlen_q_rounded;
      params.seqlen_k_rounded = seqlen_k_rounded;
      params.d = d;
      params.d_rounded = d_rounded;

      // Set the different scale values.
      #ifdef FLASHATTENTION_DISABLE_SOFTCAP
          TORCH_CHECK(softcap <= 0.0, "This flash attention build does not support softcap.");
      #endif
      if (softcap > 0.0) {
          params.softcap = softmax_scale / softcap;
          params.scale_softmax = softcap;
          params.scale_softmax_log2 = softcap * M_LOG2E;
      } else {
          // Remove potential NaN
          params.softcap = 0.0;
          params.scale_softmax = softmax_scale;
          params.scale_softmax_log2 = softmax_scale * M_LOG2E;
      }

      // Set this to probability of keeping an element to simplify things.
      params.p_dropout = 1.f - p_dropout;
      // Convert p from float to int so we don't have to convert the random uint to float to compare.
      // [Minor] We want to round down since when we do the comparison we use <= instead of <
      // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
      // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
      params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
      params.rp_dropout = 1.f / params.p_dropout;
      params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
      TORCH_CHECK(p_dropout < 1.f);
      #ifdef FLASHATTENTION_DISABLE_DROPOUT
          TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
      #endif

      // Causal is the special case where window_size_right == 0 and window_size_left < 0.
      // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
      params.is_causal = window_size_left < 0 && window_size_right == 0;

      if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
      if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
      params.window_size_left = window_size_left;
      params.window_size_right = window_size_right;

      #ifdef FLASHATTENTION_DISABLE_LOCAL
          TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
              "This flash attention build does not support local attention.");
      #endif

      params.is_seqlens_k_cumulative = true;

      #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
          TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
      #endif

      params.unpadded_lse = unpadded_lse;
      params.seqlenq_ngroups_swapped = seqlenq_ngroups_swapped;
  }

};


}  // namespace test_fa
