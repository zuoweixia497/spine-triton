#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"

// clang-format off
#include "triton-shared/Dialect/xsmt/IR/XSMTDialect.h"
#include "triton-shared/Dialect/xsmt/IR/XSMTDialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::xsmt;

void mlir::xsmt::XSMTDialect::initialize() {

  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/xsmt/IR/XSMTOps.cpp.inc"
      >();
  addInterfaces<mlir::triton::TritonInlinerInterface>();
}
