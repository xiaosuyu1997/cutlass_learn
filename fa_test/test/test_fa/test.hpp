#pragma once

#include <torch/torch.h>

namespace test_fa{

enum AttnBiasType {
  SHARE_BATCH_HEAD = 0,
  SHARE_HEAD = 1,
  PER_HEAD = 2
};


std::istream& operator>>(std::istream& in,  AttnBiasType& type);


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
  void parse(int argc, const char **args);

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


class DataBatch {
public:
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

  torch::ScalarType torch_scalar_t;
  torch::ScalarType torch_scalar_ref_t;
  torch::ScalarType torch_accum_t;

  DataBatch(
    const Options& options,
    torch::DeviceType device,
    torch::ScalarType torch_scalar_t_ = torch::kFloat16,
    torch::ScalarType torch_scalar_ref_t_ = torch::kFloat16,
    torch::ScalarType torch_accum_t_ = torch::kFloat32
  );

private:
  void init_qkv_(bool init_ref, at::IntArrayRef size, torch::DeviceType device);

  void init_bias_(
    int bias_length,
    AttnBiasType bias_type,
    int batch_size,
    int num_heads,
    torch::DeviceType device,
    bool with_ref = true
  );

  void init_tensor_with_ref_(
    torch::Tensor& tensor,
    torch::Tensor& tensor_ref,
    at::IntArrayRef size,
    torch::DeviceType device,
    torch::ScalarType dtype,
    torch::ScalarType ref_dtype,
    bool with_ref = true
  );
};


class TestFA {
public:
  TestFA(const Options& options_) : options(options_) {
    initialize();
  }

  void initialize();

  std::vector<DataBatch> data;

private:
  Options options;


};

}  // namespace test_fa
