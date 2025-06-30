#include <iostream>
#include <cute/tensor.hpp>


int main() {
  using namespace cute;

  // std::cout << "Hello, World!" << std::endl;
  MMA_Atom mma_atom = MMA_Atom<SM80_16x8x8_F16F16F16F16_TN>{};
  // MMA_Atom mma_atom = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>{};
  // print_latex(mma_atom);

  // MMA
  TiledMMA tiled_mma = make_tiled_mma(
    mma_atom,
    Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
    Tile<_32,_32,_16>{}
  );      // 32x32x16 Tiled MMA for LDSM
  // print_latex(tiled_mma);

  // S2R Copy
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_a;
  Copy_Atom<SM75_U32x4_LDSM_N, half_t> s2r_atom_b;
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, tiled_mma);
  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, tiled_mma);
  // print_latex(s2r_copy_a);
  // print_latex(s2r_copy_b);

  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, cute::half_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 k-major
  // print_latex(copyA);

  // smem layout
  int bM = Int<64>{};
  int bK = Int<32>{};
  // auto swizzle_atom = Layout<Shape <_8,Shape <_8, _8>>,
  //                            Stride<_8,Stride<_1,_64>>>{};

  // Define the smem layouts (static)
  // Swizzles for LDSM and 128b k-major loads
  auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8,Shape <_8, _8>>,
                                         Stride<_8,Stride<_1,_64>>>{});
  // print_latex(swizzle_atom);

  // auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
  auto a = Layout<Shape<_8,_8>, Stride<_8,_1>>{};
  // print_latex(a);
  auto s_a = composition(Swizzle<2,0,-3>{}, a);
  print_latex(s_a);

  return 0;
}
