#ifndef TRITON_DIALECT_SMT_IR_DIALECT_H_
#define TRITON_DIALECT_SMT_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"


#include "triton-shared/Dialect/smt/IR/SMTDialect.h.inc"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/smt/IR/SMTOps.h.inc"

namespace mlir {
namespace smt {

} // namespace smt
} // namespace mlir

#endif // TRITON_DIALECT_SMT_IR_DIALECT_H_
