#pragma once

#include <torch/torch.h>


namespace fa_kernels {

class Interface {
public:
  Interface() {}

  virtual std::vector<at::Tensor> forward(
    at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
    const std::optional<at::Tensor> &cu_seqlens,  // b+1
    std::optional<int> max_seqlen,
    std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
    const std::optional<float> softmax_scale_,
    bool is_causal
  ) = 0;

  // virtual std::vector<at::Tensor> backward() = 0;

};


} // namespace fa_kernels
