// #include "dtype_mapping.hpp"
#include <cutlass/util/command_line.h>

#include "test.hpp"
#include "fa_kernels/interface.hpp"


namespace test_fa {

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

void Options::parse(int argc, const char **args) {
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


DataBatch::DataBatch(
  const Options& options,
  torch::DeviceType device,
  torch::ScalarType torch_scalar_t_,
  torch::ScalarType torch_scalar_ref_t_,
  torch::ScalarType torch_accum_t_
) : torch_scalar_t(torch_scalar_t_),
    torch_scalar_ref_t(torch_scalar_ref_t_),
    torch_accum_t(torch_accum_t_)
{
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

void DataBatch::init_qkv_(bool init_ref, at::IntArrayRef size, torch::DeviceType device) {
    init_tensor_with_ref_(q, q_ref, size, device, torch_scalar_t, torch_scalar_ref_t, init_ref);
    init_tensor_with_ref_(k, k_ref, size, device, torch_scalar_t, torch_scalar_ref_t, init_ref);
    init_tensor_with_ref_(v, v_ref, size, device, torch_scalar_t, torch_scalar_ref_t, init_ref);
}

void DataBatch::init_bias_(
  int bias_length,
  AttnBiasType bias_type,
  int batch_size,
  int num_heads,
  torch::DeviceType device,
  bool with_ref
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

void DataBatch::init_tensor_with_ref_(
  torch::Tensor& tensor,
  torch::Tensor& tensor_ref,
  at::IntArrayRef size,
  torch::DeviceType device,
  torch::ScalarType dtype,
  torch::ScalarType ref_dtype,
  bool with_ref
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


void TestFA::initialize() {
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

} // namespace test_fa
