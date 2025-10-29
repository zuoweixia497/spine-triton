//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_XSMTTOLINALG_XSMTTOLINALG_H
#define TRITON_CONVERSION_XSMTTOLINALG_XSMTTOLINALG_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/XsmtToLinalg/Passes.h.inc"

void fillToMemrefConversionPatterns(RewritePatternSet &patterns);
void ForToForallConversionPatterns(RewritePatternSet &patterns);
void populateXsmtToLinalgConversionPatterns(RewritePatternSet &patterns);
void MMT4DOpConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createXsmtToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_XSMTTOLINALG_XSMTTOLINALG_H
