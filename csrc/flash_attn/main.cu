#include <torch/torch.h>
// #include <torch/python.h>
// #include <torch/nn/functional.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <c10/cuda/CUDAStream.h>
// #include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::Generator and at::PhiloxCudaState

// #include "philox_unpack.cuh"  // For at::cuda::philox::unpack
// #include "flash_fwd_kernel.h"

#include "test.hpp"


int main(int argc, const char** argv) {
  std::cout << "FlashAttention forward test" << std::endl;

  // torch::Device device(torch::kCPU);
  // if (torch::cuda::is_available()) {
  //   device = torch::Device(torch::kCUDA);
  // } else {
  //   std::cout << "Not detecting any cuda device, using cpu instead" << std::endl;
  // }

  // int batch_size = 32, seqlen = 1024, num_heads = 16, head_size = 64;
  // torch::Tensor q = torch::randn({batch_size, seqlen, num_heads, head_size}, device);
  // torch::Tensor k = torch::randn({batch_size, seqlen, num_heads, head_size}, device);

  // // slice [0, 0, :10, :10] of q
  // std::cout << q.index({
  //   0,
  //   0,
  //   torch::indexing::Slice(0, 10),
  //   torch::indexing::Slice(0, 10)
  // }) << std::endl;

  auto options = test_fa::Options();
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage();
    return 0;
  }

  // use options to init test
  std::cout << "Running test with parameters: " << std::endl
            << options << std::endl;

  return 0;
}

