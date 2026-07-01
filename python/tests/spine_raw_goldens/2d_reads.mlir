module {
func.func @k_2d_reads(
    %A : memref<*xf16, #ptr.generic_space>,
    %C : memref<*xf32, #ptr.generic_space>,
    %rb : index,
    %off : index,
    %M : index,
    %co : index
) {
  %c0 = arith.constant 0 : index
  %zero_f16 = arith.constant 0.000000e+00 : f16
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  %ranked = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
  %v0 = memref.reinterpret_cast %ranked to offset: [0], sizes: [1, 32], strides: [32, 1] : memref<?xf16, #ptr.generic_space> to memref<1x32xf16, strided<[32, 1]>, #ptr.generic_space>
  %view2d = memref.reinterpret_cast %v0 to offset: [0], sizes: [1, 32], strides: [32, 1] : memref<1x32xf16, strided<[32, 1]>, #ptr.generic_space> to memref<1x32xf16, strided<[32, 1]>, #ptr.generic_space>
  %ld2d = vector.transfer_read %view2d[%c0, %c0], %zero_f16 {in_bounds = [true, true]} : memref<1x32xf16, strided<[32, 1]>, #ptr.generic_space>, vector<1x32xf16>
  %ranked_1 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  %view2d_2 = memref.reinterpret_cast %ranked_1 to offset: [0], sizes: [1, 32], strides: [32, 1] : memref<?xf32, #ptr.generic_space> to memref<1x32xf16, strided<[32, 1]>, #ptr.generic_space>
  vector.transfer_write %ld2d, %view2d_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf16>, memref<1x32xf16, strided<[32, 1]>, #ptr.generic_space>
  %ranked_3 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
  %boff = arith.muli %off, %c64 : index
  %view2d_4 = memref.reinterpret_cast %ranked_3 to offset: [%boff], sizes: [64, 32], strides: [32, 1] : memref<?xf16, #ptr.generic_space> to memref<64x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>
  %la = vector.transfer_read %view2d_4[%c0, %c0], %zero_f16 {in_bounds = [true, true]} : memref<64x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>, vector<64x32xf16>
  %ranked_5 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  %boff_6 = arith.muli %off, %c64 : index
  %view2d_7 = memref.reinterpret_cast %ranked_5 to offset: [%boff_6], sizes: [64, 32], strides: [32, 1] : memref<?xf32, #ptr.generic_space> to memref<64x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>
  vector.transfer_write %la, %view2d_7[%c0, %c0] {in_bounds = [true, true]} : vector<64x32xf16>, memref<64x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>
  %ranked_8 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
  %boff_9 = arith.muli %rb, %c128 : index
  %viewT = memref.reinterpret_cast %ranked_8 to offset: [%boff_9], sizes: [32, 64], strides: [1, 128] : memref<?xf16, #ptr.generic_space> to memref<32x64xf16, strided<[1, 128], offset: ?>, #ptr.generic_space>
  %lt_static = vector.transfer_read %viewT[%c0, %c0], %zero_f16 {in_bounds = [true, true]} : memref<32x64xf16, strided<[1, 128], offset: ?>, #ptr.generic_space>, vector<32x64xf16>
  %ranked_10 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  %view2d_11 = memref.reinterpret_cast %ranked_10 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<?xf32, #ptr.generic_space> to memref<32x64xf16, strided<[64, 1]>, #ptr.generic_space>
  vector.transfer_write %lt_static, %view2d_11[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf16>, memref<32x64xf16, strided<[64, 1]>, #ptr.generic_space>
  %ranked_12 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
  %boff_13 = arith.muli %rb, %M : index
  %boff_14 = arith.addi %boff_13, %co : index
  %viewT_15 = memref.reinterpret_cast %ranked_12 to offset: [%boff_14], sizes: [32, 64], strides: [1, %M] : memref<?xf16, #ptr.generic_space> to memref<32x64xf16, strided<[1, ?], offset: ?>, #ptr.generic_space>
  %lt_dyn = vector.transfer_read %viewT_15[%c0, %c0], %zero_f16 {in_bounds = [true, true]} : memref<32x64xf16, strided<[1, ?], offset: ?>, #ptr.generic_space>, vector<32x64xf16>
  %ranked_16 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  %view2d_17 = memref.reinterpret_cast %ranked_16 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<?xf32, #ptr.generic_space> to memref<32x64xf16, strided<[64, 1]>, #ptr.generic_space>
  vector.transfer_write %lt_dyn, %view2d_17[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf16>, memref<32x64xf16, strided<[64, 1]>, #ptr.generic_space>
  %ranked_18 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
  %boff_19 = arith.muli %rb, %M : index
  %boff_20 = arith.addi %boff_19, %co : index
  %Aview = memref.reinterpret_cast %ranked_18 to offset: [%boff_20], sizes: [64, 32], strides: [%M, 1] : memref<?xf16, #ptr.generic_space> to memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>
  %buf = memref.alloca() {alignment = 64 : i64} : memref<32x64xf16>
  linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%Aview : memref<64x32xf16, strided<[?, 1], offset: ?>, #ptr.generic_space>) outs(%buf : memref<32x64xf16>) {
  ^bb0(%a: f16, %_: f16):
    linalg.yield %a : f16
  }
  %pk = vector.transfer_read %buf[%c0, %c0], %zero_f16 {in_bounds = [true, true]} : memref<32x64xf16>, vector<32x64xf16>
  %ranked_21 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  %view2d_22 = memref.reinterpret_cast %ranked_21 to offset: [0], sizes: [32, 64], strides: [64, 1] : memref<?xf32, #ptr.generic_space> to memref<32x64xf16, strided<[64, 1]>, #ptr.generic_space>
  vector.transfer_write %pk, %view2d_22[%c0, %c0] {in_bounds = [true, true]} : vector<32x64xf16>, memref<32x64xf16, strided<[64, 1]>, #ptr.generic_space>
  return
}
}
