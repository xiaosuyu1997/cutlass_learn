#pragma once

#include "fa_kernels/interface.hpp"
#include "flash.h"

namespace fa_kernels {
class TriDaoKernel : public Interface {
public:
  TriDaoKernel() {}

  std::vector<at::Tensor> forward(
    at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const std::optional<at::Tensor> &cu_seqlens,  // b+1
    std::optional<int> max_seqlen,
    std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const std::optional<float> softmax_scale_,
    bool is_causal
  ) override;

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
  );

};

} // namespace fa_kernels
