module {
func.func @k_matmul(
    %A : memref<*xf16, #ptr.generic_space>,
    %C : memref<*xf32, #ptr.generic_space>,
    %i : index,
    %rb : index,
    %cb : index
) {
  %zero_f16 = arith.constant 0.000000e+00 : f16
  %zero_f32 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %ranked = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
  %lhs = vector.transfer_read %ranked[%i], %zero_f16 {in_bounds = [true]} : memref<?xf16, #ptr.generic_space>, vector<32xf16>
  %ranked_1 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
  %rhs = vector.transfer_read %ranked_1[%i], %zero_f16 {in_bounds = [true]} : memref<?xf16, #ptr.generic_space>, vector<2048xf16>
  %accv = vector.broadcast %zero_f32 : f32 to vector<64xf32>
  %out = "vector_ext.matmul"(%lhs, %rhs, %accv) <{m = 1 : i64, n = 64 : i64, k = 32 : i64}> : (vector<32xf16>, vector<2048xf16>, vector<64xf32>) -> vector<64xf32>
  %e0 = vector.extract %out[0] : f32 from vector<64xf32>
  %ei = vector.extractelement %out[%i : index] : vector<64xf32>
  %ranked_2 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  memref.store %e0, %ranked_2[%i] : memref<?xf32, #ptr.generic_space>
  %ranked_3 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  memref.store %ei, %ranked_3[%rb] : memref<?xf32, #ptr.generic_space>
  %padded = vector.broadcast %zero_f16 : f16 to vector<2048xf16>
  %padded_4 = vector.insert_strided_slice %lhs, %padded {offsets = [0], strides = [1]} : vector<32xf16> into vector<2048xf16>
  %out2 = "vector_ext.matmul"(%lhs, %padded_4, %accv) <{m = 1 : i64, n = 64 : i64, k = 32 : i64}> : (vector<32xf16>, vector<2048xf16>, vector<64xf32>) -> vector<64xf32>
  %ranked_5 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  vector.transfer_write %out2, %ranked_5[%cb] {in_bounds = [true]} : vector<64xf32>, memref<?xf32, #ptr.generic_space>
  %ranked_6 = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
  %sz1d = memref.dim %ranked_6, %c0 : memref<?xf16, #ptr.generic_space>
  %view2d = memref.reinterpret_cast %ranked_6 to offset: [0], sizes: [%sz1d, 32], strides: [32, 1] : memref<?xf16, #ptr.generic_space> to memref<?x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>
  %tile = vector.transfer_read %view2d[%rb, %cb], %zero_f16 {in_bounds = [true, true]} : memref<?x32xf16, strided<[32, 1], offset: ?>, #ptr.generic_space>, vector<1x32xf16>
  %tile_7 = vector.shape_cast %tile : vector<1x32xf16> to vector<32xf16>
  %ranked_8 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  vector.transfer_write %tile_7, %ranked_8[%i] {in_bounds = [true]} : vector<32xf16>, memref<?xf32, #ptr.generic_space>
  return
}
}
