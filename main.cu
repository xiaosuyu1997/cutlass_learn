#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;

template <
  typename TiledMMA,
  typename SmemLayout
>
__global__ void test_kernel()
{
  extern __shared__ char smem[];

  TiledMMA tiled_mma;

  Tensor sA = make_tensor(make_smem_ptr(smem), SmemLayout{});   // (BLK_M,BLK_K,PIPE)

  ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCrA = thr_mma.partition_fragment_A(sA);               // (MMA,MMA_M,MMA_K,PIPE)

  if (thread0()) {
    print("tCsA : "); print(tCsA); print("\n");
    print("tCrA : "); print(tCrA); print("\n");
  }

}




int main() {
  // std::cout << "Hello, World!" << std::endl;
  // MMA_Atom mma_atom = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>{};
  // MMA_Atom mma_atom = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>{};
  // print_latex(mma_atom);

  // MMA
  auto tiled_mma = make_tiled_mma(
    MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>{},
    Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
    Tile<_32,_16,_8>{}
  );      // 32x32x16 Tiled MMA for LDSM
  // print_latex(tiled_mma);

  // S2R Copy
  // Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_a;
  // Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_b;
  // TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, tiled_mma);
  // TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, tiled_mma);
  // print_latex(s2r_copy_a);
  // print_latex(s2r_copy_b);

  // TiledCopy g2s_copy_a = make_tiled_copy(
  //   Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
  //   Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
  //   Layout<Shape< _1,_8>>{}                 // Val layout  1x8 k-major
  // );
  // print_latex(copyA);

  // smem layout
  auto bM = Int<64>{};
  auto bK = Int<32>{};
  auto bP = Int<3>{};
  auto smem_atom = make_layout(
    make_shape(bM, bK),
    make_stride(bK, _1{})
  );
  auto smem_layout = tile_to_shape(
    smem_atom,
    make_shape(bM, bK, bP)
  );
  print("smem layout : "); print(smem_layout); print("\n");

  std::cout << "start flag\n";
  test_kernel<decltype(tiled_mma), decltype(smem_layout)>
    <<<1, dim3(size(tiled_mma)), size(smem_layout) * sizeof(half_t)>>>();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return -1;
  }
  std::cout << "end flag\n";

  // make_shape(512, 128)
  // auto A = make_identity_tensor(
  //   make_shape(16, 8)
  // );
  // print_latex(A);


  // auto swizzle_atom = Layout<Shape <_8,Shape <_8, _8>>,
  //                            Stride<_8,Stride<_1,_64>>>{};

  // Define the smem layouts (static)
  // Swizzles for LDSM and 128b k-major loads
  // auto swizzle_atom = composition(Swizzle<3,3,3>{},
  //                                 Layout<Shape <_8,Shape <_8, _8>>,
  //                                        Stride<_8,Stride<_1,_64>>>{});
  // print_latex(swizzle_atom);

  // auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
  // auto a = Layout<Shape<_8,_8>, Stride<_8,_1>>{};
  // print_latex(a);
  // auto s_a = composition(Swizzle<2,0,-3>{}, a);
  // print_latex(s_a);

  return 0;
}
