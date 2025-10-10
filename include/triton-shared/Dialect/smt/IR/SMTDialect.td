#ifndef SMT_DIALECT
#define SMT_DIALECT

include "mlir/IR/OpBase.td"

def SMT_Dialect : Dialect {
  let name = "smt";
  let cppNamespace = "::mlir::smt";

  let description = [{
    SMT Dialect.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::smt::SMTDialect",
  ];
}

#endif
