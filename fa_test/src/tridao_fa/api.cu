#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>


#include "fa_kernels/interface.hpp"   // public header
#include "api.h"											// private headers
#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_launch.hpp"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace fa_kernels {

std::vector<at::Tensor> TriDaoKernel::forward(
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
  set_params_fprop(
    params,
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
    /*unpadded_lse*/true
	);

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

void TriDaoKernel::set_params_fprop(
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
	bool seqlenq_ngroups_swapped,
	const bool unpadded_lse
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

} // namespace fa_kernels
