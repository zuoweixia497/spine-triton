#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"

// clang-format off
#include "triton-shared/Dialect/smt/IR/SMTDialect.h"
#include "triton-shared/Dialect/smt/IR/SMTDialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::smt;

void mlir::smt::SMTDialect::initialize() {

  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/smt/IR/SMTOps.cpp.inc"
      >();
  addInterfaces<mlir::triton::TritonInlinerInterface>();
}
