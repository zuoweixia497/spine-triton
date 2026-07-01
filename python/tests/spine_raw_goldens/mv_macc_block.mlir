module {
func.func @k_mv_macc_block(
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
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %c192 = arith.constant 192 : index
  %zero_f16 = arith.constant 0.000000e+00 : f16
  %buf0 = memref.alloc() {alignment = 64 : i64} : memref<32x64xf16>
  %buf1 = memref.alloc() {alignment = 64 : i64} : memref<32x64xf16>
  %buf2 = memref.alloc() {alignment = 64 : i64} : memref<32x64xf16>
  %buf3 = memref.alloc() {alignment = 64 : i64} : memref<32x64xf16>
  %acc0_init = vector.broadcast %zero_f32 : f32 to vector<1x64xf32>
  %acc1_init = vector.broadcast %zero_f32 : f32 to vector<1x64xf32>
  %acc2_init = vector.broadcast %zero_f32 : f32 to vector<1x64xf32>
  %acc3_init = vector.broadcast %zero_f32 : f32 to vector<1x64xf32>
  %acc0, %acc1, %acc2, %acc3 = scf.for %kb = %c0 to %nk step %c1 iter_args(%acc0_in = %acc0_init, %acc1_in = %acc1_init, %acc2_in = %acc2_init, %acc3_in = %acc3_init) -> (vector<1x64xf32>, vector<1x64xf32>, vector<1x64xf32>, vector<1x64xf32>) {
    %koff = arith.muli %kb, %c32 : index
    %ranked = memref.cast %B : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
    %lhs = memref.reinterpret_cast %ranked to offset: [%koff], sizes: [1, 32], strides: [32, 1] : memref<?xf16, #ptr.generic_space> to memref<1x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>
    %ranked_1 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
    %boff = arith.muli %col, %M : index
    %boff_2 = arith.addi %boff, %koff : index
    %Aview = memref.reinterpret_cast %ranked_1 to offset: [%boff_2], sizes: [64, 32], strides: [%M, 1] : memref<?xf16, #ptr.generic_space> to memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%Aview : memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>) outs(%buf0 : memref<32x64xf16>) {
    ^bb0(%a: f16, %_: f16):
      linalg.yield %a : f16
    }
    %t = arith.addi %col, %c64 : index
    %ranked_3 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
    %boff_4 = arith.muli %t, %M : index
    %boff_5 = arith.addi %boff_4, %koff : index
    %Aview_6 = memref.reinterpret_cast %ranked_3 to offset: [%boff_5], sizes: [64, 32], strides: [%M, 1] : memref<?xf16, #ptr.generic_space> to memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%Aview_6 : memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>) outs(%buf1 : memref<32x64xf16>) {
    ^bb0(%a: f16, %_: f16):
      linalg.yield %a : f16
    }
    %t_7 = arith.addi %col, %c128 : index
    %ranked_8 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
    %boff_9 = arith.muli %t_7, %M : index
    %boff_10 = arith.addi %boff_9, %koff : index
    %Aview_11 = memref.reinterpret_cast %ranked_8 to offset: [%boff_10], sizes: [64, 32], strides: [%M, 1] : memref<?xf16, #ptr.generic_space> to memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%Aview_11 : memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>) outs(%buf2 : memref<32x64xf16>) {
    ^bb0(%a: f16, %_: f16):
      linalg.yield %a : f16
    }
    %t_12 = arith.addi %col, %c192 : index
    %ranked_13 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
    %boff_14 = arith.muli %t_12, %M : index
    %boff_15 = arith.addi %boff_14, %koff : index
    %Aview_16 = memref.reinterpret_cast %ranked_13 to offset: [%boff_15], sizes: [64, 32], strides: [%M, 1] : memref<?xf16, #ptr.generic_space> to memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%Aview_16 : memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>) outs(%buf3 : memref<32x64xf16>) {
    ^bb0(%a: f16, %_: f16):
      linalg.yield %a : f16
    }
    %view2d = memref.reinterpret_cast %buf0 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<32x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    %r0 = vector.transfer_read %view2d[%c0, %c0], %zero_f16 {in_bounds = [true, true]} : memref<32x64xf16, strided<[64, 1]>>, vector<32x64xf16>
    %view2d_17 = memref.reinterpret_cast %buf1 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<32x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    %r1 = vector.transfer_read %view2d_17[%c0, %c0], %zero_f16 {in_bounds = [true, true]} : memref<32x64xf16, strided<[64, 1]>>, vector<32x64xf16>
    %view2d_18 = memref.reinterpret_cast %buf2 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<32x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    %r2 = vector.transfer_read %view2d_18[%c0, %c0], %zero_f16 {in_bounds = [true, true]} : memref<32x64xf16, strided<[64, 1]>>, vector<32x64xf16>
    %view2d_19 = memref.reinterpret_cast %buf3 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<32x64xf16> to memref<32x64xf16, strided<[64, 1]>>
    %r3 = vector.transfer_read %view2d_19[%c0, %c0], %zero_f16 {in_bounds = [true, true]} : memref<32x64xf16, strided<[64, 1]>>, vector<32x64xf16>
    %acc0_upd = "vector_ext.batch_macc"(%lhs, %r0, %acc0_in) : (memref<1x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>, vector<32x64xf16>, vector<1x64xf32>) -> vector<1x64xf32>
    %acc1_upd = "vector_ext.batch_macc"(%lhs, %r1, %acc1_in) : (memref<1x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>, vector<32x64xf16>, vector<1x64xf32>) -> vector<1x64xf32>
    %acc2_upd = "vector_ext.batch_macc"(%lhs, %r2, %acc2_in) : (memref<1x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>, vector<32x64xf16>, vector<1x64xf32>) -> vector<1x64xf32>
    %acc3_upd = "vector_ext.batch_macc"(%lhs, %r3, %acc3_in) : (memref<1x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>, vector<32x64xf16>, vector<1x64xf32>) -> vector<1x64xf32>
    scf.yield %acc0_upd, %acc1_upd, %acc2_upd, %acc3_upd : vector<1x64xf32>, vector<1x64xf32>, vector<1x64xf32>, vector<1x64xf32>
  }
  %ranked_20 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  %boff_21 = arith.muli %col, %c1 : index
  %view2d_22 = memref.reinterpret_cast %ranked_20 to offset: [%boff_21], sizes: [1, 64], strides: [64, 1] : memref<?xf32, #ptr.generic_space> to memref<1x64xf32, strided<[64, 1], offset: ?>, #ptr.generic_space>
  vector.transfer_write %acc0, %view2d_22[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #ptr.generic_space>
  %t_23 = arith.addi %col, %c64 : index
  %ranked_24 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  %boff_25 = arith.muli %t_23, %c1 : index
  %view2d_26 = memref.reinterpret_cast %ranked_24 to offset: [%boff_25], sizes: [1, 64], strides: [64, 1] : memref<?xf32, #ptr.generic_space> to memref<1x64xf32, strided<[64, 1], offset: ?>, #ptr.generic_space>
  vector.transfer_write %acc1, %view2d_26[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #ptr.generic_space>
  %t_27 = arith.addi %col, %c128 : index
  %ranked_28 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  %boff_29 = arith.muli %t_27, %c1 : index
  %view2d_30 = memref.reinterpret_cast %ranked_28 to offset: [%boff_29], sizes: [1, 64], strides: [64, 1] : memref<?xf32, #ptr.generic_space> to memref<1x64xf32, strided<[64, 1], offset: ?>, #ptr.generic_space>
  vector.transfer_write %acc2, %view2d_30[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #ptr.generic_space>
  %t_31 = arith.addi %col, %c192 : index
  %ranked_32 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  %boff_33 = arith.muli %t_31, %c1 : index
  %view2d_34 = memref.reinterpret_cast %ranked_32 to offset: [%boff_33], sizes: [1, 64], strides: [64, 1] : memref<?xf32, #ptr.generic_space> to memref<1x64xf32, strided<[64, 1], offset: ?>, #ptr.generic_space>
  vector.transfer_write %acc3, %view2d_34[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #ptr.generic_space>
  return
}
}
