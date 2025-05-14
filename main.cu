#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

int main() {
  using namespace cute;

  // std::cout << "Hello, World!" << std::endl;

  // MMA
  TiledMMA mma = make_tiled_mma(SM80_16x8x8_F16F16F16F16_TN{},
                                Layout<Shape<_1,_1>>{},    // 2x2x1 MMA Atoms
                                Tile<_32,_32,_8>{});      // 32x32x16 Tiled MMA for LDSM
  // print_latex(mma);

  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_a;
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  print_latex(s2r_copy_a);
  return 0;
}
