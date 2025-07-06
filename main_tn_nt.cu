#include <iostream>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>

int main () {
  using namespace cute;
  MMA_Atom mma_atom = MMA_Atom<SM70_8x8x4_F16F16F16F16_NT>{};
  MMA_Atom mma_atom2 = MMA_Atom<SM70_8x8x4_F16F16F16F16_TN>{};
  print_latex(mma_atom);
  print_latex(mma_atom2);

  return 0;
}
