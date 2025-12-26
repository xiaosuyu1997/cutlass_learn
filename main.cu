#include <iostream>
#include <cute/tensor.hpp>

using namespace cute;


int main() {
  // MMA
  auto tiled_mma = make_tiled_mma(
    MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>{},
    Layout<Shape<_4,_1,_1>>{},  // 4x1x1 or 8x1x1 thread group
    Tile<_64, _16, _16>{}
  );      // 32x32x16 Tiled MMA for LDSM
  print_latex(tiled_mma);

  return 0;
}
