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
#include "ViewOpInfoStorage.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/XSMTToLinalg/Passes.h.inc"

void TransposeEliminationConversionPatterns(RewritePatternSet &patterns);
void fillToMemrefConversionPatterns(RewritePatternSet &patterns);
void ForToForallConversionPatterns(RewritePatternSet &patterns);
void populateXSMTToLinalgConversionPatterns(RewritePatternSet &patterns, ViewOpInfoStorage &storage);
void DescriptorLoadViewOpConversionPatterns(RewritePatternSet &patterns, ViewOpInfoStorage &storage);
void MMT4DOpConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createXSMTToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_XSMTTOLINALG_XSMTTOLINALG_H
