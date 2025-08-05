//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_ConvertScan_H
#define TRITON_CONVERSION_TRITONTOLINALG_ConvertScan_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertScanOpPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_ConvertScan_H
