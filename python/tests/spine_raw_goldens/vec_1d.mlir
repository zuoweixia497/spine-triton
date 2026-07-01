module {
func.func @k_vec_1d(
    %A : memref<*xf16, #ptr.generic_space>,
    %B : memref<*xf16, #ptr.generic_space>,
    %C : memref<*xf32, #ptr.generic_space>,
    %i : index,
    %n : index
) {
  %zero_f32 = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero_f16 = arith.constant 0.000000e+00 : f16
  %acc_init = vector.broadcast %zero_f32 : f32 to vector<32xf32>
  %acc = scf.for %k = %c0 to %n step %c1 iter_args(%acc_in = %acc_init) -> (vector<32xf32>) {
    %ranked = memref.cast %A : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
    %a = vector.transfer_read %ranked[%i], %zero_f16 {in_bounds = [true]} : memref<?xf16, #ptr.generic_space>, vector<32xf16>
    %ranked_1 = memref.cast %B : memref<*xf16, #ptr.generic_space> to memref<?xf16, #ptr.generic_space>
    %b = vector.transfer_read %ranked_1[%i], %zero_f16 {in_bounds = [true]} : memref<?xf16, #ptr.generic_space>, vector<32xf16>
    %af = arith.extf %a : vector<32xf16> to vector<32xf32>
    %bf = arith.extf %b : vector<32xf16> to vector<32xf32>
    %acc_upd = math.fma %af, %bf, %acc_in : vector<32xf32>
    scf.yield %acc_upd : vector<32xf32>
  }
  %s = vector.reduction <add>, %acc : vector<32xf32> into f32
  %ranked_2 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  memref.store %s, %ranked_2[%i] : memref<?xf32, #ptr.generic_space>
  %ranked_3 = memref.cast %C : memref<*xf32, #ptr.generic_space> to memref<?xf32, #ptr.generic_space>
  vector.transfer_write %acc, %ranked_3[%i] {in_bounds = [true]} : vector<32xf32>, memref<?xf32, #ptr.generic_space>
  return
}
}
