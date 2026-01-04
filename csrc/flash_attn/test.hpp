#include <string>
#include <torch/torch.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/command_line.h>
#include "dtype_mapping.hpp"

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
    in.setstate(std::ios::failbit);
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

  bool check_ref = true;

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
      << "  --num_heads=<int>         Head number in multi-head attention (default: --num_heads=12)\n"
      << "  --batch_size=<int>          Batch size in multi-head attention (default: --batch_size=16)\n"
      << "  --head_size=<int>           Head size in multi-head attention (default: --head_size=64)\n"
      << "  --seq_length=<int>          Sequence length in multi-head attention for Q (default: --seq_length=1024)\n"
      << "  --var_len                   If specified, use variable length sequences (default: false)\n"
      // << "  --cross_attn                If specified, use cross attention (default: false)\n"
      << "  --causal                    If specified, use causal attention (default: false)\n"
      << "  --attn_bias                 If specified, use custom attention bias (default: false)\n"
      << "  --pad_bias_to_multiple_of_8 If specified, pad attention bias to multiple of 8 (default: false)\n"
      << "  --attn_bias_type=<type>     Attention bias type: SHARE_BATCH_HEAD, SHARE_HEAD, PER_HEAD (default: SHARE_BATCH_HEAD)\n"
      << "  --check_ref                 If specified, check results against reference implementation (default: true)\n"
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
      bias_length = options.var_len ? max_seqlen : options.seq_length;
      if (options.pad_bias_to_multiple_of_8) {
        bias_length = ((bias_length + 7) / 8) * 8;
      }
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

  void init_bias_(int bias_length, int batch_size, int num_heads) {
    at::IntArrayRef bias_size;
    switch (options.attn_bias_type) {
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
      options.check_ref
    );
  }

  void init_tensor_with_ref_(
    torch::Tensor& tensor,
    torch::Tensor& tensor_ref,
    at::IntArrayRef size,
    torch::DeviceType device,
    torch::ScalarType dtype = torch_scalar_t,
    torch::ScalarType ref_dtype = torch_scalar_ref_t,
    with_ref = true
  ) {
    tensor_ref = torch::randn(
      size,
      torch::device(device).dtype(ref_dtype)
    );
    tensor = tensor_ref.to(dtype);
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

}  // namespace test_fa
