#include <torch/torch.h>
#include <iostream>

int main() {
    // Check if CUDA is available
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    } else {
        std::cout << "CUDA is not available. Training on CPU." << std::endl;
    }

    // Define matrix dimensions
    const int64_t M = 128;
    const int64_t K = 64;
    const int64_t N = 128;

    std::cout << "Performing Matrix Multiplication (M=" << M << ", K=" << K << ", N=" << N << ")" << std::endl;

    // Create tensors
    // Ensure we use the correct device (CPU by default here, can be moved to CUDA)
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::Device(torch::kCUDA);
    }

    torch::Tensor A = torch::randn({M, K}, device);
    torch::Tensor B = torch::randn({K, N}, device);

    // Perform matrix multiplication
    // torch::matmul is the high-level API that dispatches to ATen's mm/bmm/etc.
    torch::Tensor C = torch::matmul(A, B);

    std::cout << "Multiplication complete." << std::endl;
    std::cout << "Result shape: " << C.sizes() << std::endl;

    // Print a small slice of the result to verify
    std::cout << "Result slice (top-left 3x3):" << std::endl;
    std::cout << C.slice(0, 0, 3).slice(1, 0, 3) << std::endl;

    return 0;
}
