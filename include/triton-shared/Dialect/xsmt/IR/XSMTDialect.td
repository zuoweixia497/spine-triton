#ifndef XSMT_DIALECT
#define XSMT_DIALECT

include "mlir/IR/OpBase.td"

def XSMT_Dialect : Dialect {
  let name = "xsmt";
  let cppNamespace = "::mlir::xsmt";

  let description = [{
    XSMT Dialect.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::xsmt::XSMTDialect",
  ];
}

#endif
