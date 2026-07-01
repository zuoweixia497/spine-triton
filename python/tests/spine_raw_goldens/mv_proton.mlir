module {
func.func @k_mv_proton(
    %B : memref<*xf16, #ptr.generic_space>,
    %A : memref<*xf16, #ptr.generic_space>,
    %col : index,
    %M : index,
    %nk : index,
    %C : memref<*xf32, #ptr.generic_space>
) {
  %zero_f32 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %zero_f16 = arith.constant 0.000000e+00 : f16
  proton.record start "alloc"
  %b0 = memref.alloc() {alignment = 64 : i64} : memref<32x64xf16>
  proton.record end "alloc"
  %a0_init = vector.broadcast %zero_f32 : f32 to vector<1x64xf32>
  %a0 = scf.for %kb = %c0 to %nk step %c1 iter_args(%a0_in = %a0_init) -> (vector<1x64xf32>) {
    %koff = arith.muli %kb, %c32 : index
    %ranked = memref.cast %B : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
    %lhs = memref.reinterpret_cast %ranked to offset: [%koff], sizes: [1, 32], strides: [32, 1] : memref<?xf16, #ptr.generic_space> to memref<1x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>
    proton.record start "pack"
    %ranked_1 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
    %boff = arith.muli %col, %M : index
    %boff_2 = arith.addi %boff, %koff : index
    %Aview = memref.reinterpret_cast %ranked_1 to offset: [%boff_2], sizes: [64, 32], strides: [%M, 1] : memref<?xf16, #ptr.generic_space> to memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%Aview : memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>) outs(%b0 : memref<32x64xf16>) {
    ^bb0(%a: f16, %_: f16):
      linalg.yield %a : f16
    }
    proton.record end "pack"
    %view2d = memref.reinterpret_cast %b0 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<32x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    %r0 = vector.transfer_read %view2d[%c0, %c0], %zero_f16 {in_bounds = [true, true]} : memref<32x64xf16, strided<[64, 1]>>, vector<32x64xf16>
    %a0_upd = "vector_ext.batch_macc"(%lhs, %r0, %a0_in) : (memref<1x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>, vector<32x64xf16>, vector<1x64xf32>) -> vector<1x64xf32>
    scf.yield %a0_upd : vector<1x64xf32>
  }
  %ranked_3 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  %boff_4 = arith.muli %col, %c1 : index
  %view2d_5 = memref.reinterpret_cast %ranked_3 to offset: [%boff_4], sizes: [1, 64], strides: [64, 1] : memref<?xf32, #ptr.generic_space> to memref<1x64xf32, strided<[64, 1], offset: ?>, #ptr.generic_space>
  vector.transfer_write %a0, %view2d_5[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #ptr.generic_space>
  memref.dealloc %b0 : memref<32x64xf16>
  return
}
}
