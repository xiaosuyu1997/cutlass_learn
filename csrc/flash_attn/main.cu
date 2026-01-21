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

  test_fa::Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage();
    return 0;
  }

  // use options to init test
  std::cout << "Running test with parameters: " << std::endl
            << options << std::endl;

  // test suite
  test_fa::TestFA test{options};

  // print data
  std::cout << test.data.size() << " data batches initialized." << std::endl;
  std::cout << "First batch q:" << std::endl;
            // << test.data[0].q << std::endl;

  return 0;
}

