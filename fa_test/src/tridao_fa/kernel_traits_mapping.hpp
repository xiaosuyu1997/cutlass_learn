#pragma once

#include "namespace_config.h"
#include "kernel_traits.h"

namespace FLASH_NAMESPACE {

template<int hdim, bool is_dropout, typename scalar_t>
struct Params2KernelTraits {
  using Traits = Flash_fwd_kernel_traits<hdim, 128, 128, 8, false, false, scalar_t>;
};

template<typename scalar_t>
struct Params2KernelTraits<64, false, scalar_t> {
  using Traits = Flash_fwd_kernel_traits<64, 128, 128, 4, false, false, scalar_t>;
  // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
  // Using block size (64 x 256) is 27% slower for seqlen=2k
  // Using block size (256 x 64) is 85% slower for seqlen=2k, because of register spilling
  // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
  // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
  // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
};

template<typename scalar_t>
struct Params2KernelTraits<64, true, scalar_t> {
  using Traits = Flash_fwd_kernel_traits<64, 128, 64, 4, false, false, scalar_t>;
  // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
  // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
  // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
  // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
};


}  // namespace FLASH_NAMESPACE
