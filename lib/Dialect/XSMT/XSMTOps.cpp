//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/XSMT/IR/XSMTOps.cpp.inc"

namespace mlir {
namespace xsmt {
void DescriptorLoadOp::build(OpBuilder &builder, OperationState &state,
                               Value base, ValueRange offsets,
                               ArrayRef<int32_t> shape,
                               ArrayRef<int32_t> micro_size) {
  auto pointerType = cast<mlir::triton::PointerType>(base.getType());
  auto pointeeType = pointerType.getPointeeType();

  Type elementType = pointeeType;
  while (auto tensorType = dyn_cast<RankedTensorType>(elementType)) {
    elementType = tensorType.getElementType();
  }

  if (isa<RankedTensorType>(elementType)) {
    llvm::report_fatal_error("Cannot determine scalar element type for tensor");
  }

  SmallVector<int64_t> resultShape;
  resultShape.push_back(shape[0] / micro_size[0]);
  resultShape.push_back(shape[1] / micro_size[1]);
  resultShape.push_back(micro_size[0]);
  resultShape.push_back(micro_size[1]);

  auto resultType = RankedTensorType::get(resultShape, elementType);

  return build(builder, state, resultType, base, offsets,
               builder.getDenseI32ArrayAttr(shape),
               builder.getDenseI32ArrayAttr(micro_size));
}


void ViewOp::build(OpBuilder &builder, OperationState &state,
                   Value base, ValueRange offsets,
                   ArrayRef<int32_t> shape,
                   ArrayRef<int32_t> micro_size) {
  auto baseType = cast<RankedTensorType>(base.getType());
  Type elementType = baseType.getElementType();
  bool allOnes = true;
  for (int32_t size : micro_size) {
    if (size != 1) {
      allOnes = false;
      break;
    }
  }

  bool allZeros = true;
  for (int32_t size : micro_size) {
    if (size != 0) {
      allZeros = false;
      break;
    }
  }

  if(allOnes){
    SmallVector<int64_t> resultShape;
    resultShape.push_back(shape[0]);
    resultShape.push_back(shape[1]);
    auto resultType = RankedTensorType::get(resultShape, elementType);
    return build(builder, state, resultType, base, offsets,
               builder.getDenseI32ArrayAttr(shape),
               builder.getDenseI32ArrayAttr(micro_size));
  }else if (allZeros){
    auto baseShape = baseType.getShape();
    std::vector<int32_t> actualMicroSize = {
      static_cast<int32_t>(baseShape[baseShape.size() - 2]),
      static_cast<int32_t>(baseShape[baseShape.size() - 1])
    };

    SmallVector<int64_t> resultShape;
    resultShape.push_back(shape[0] / actualMicroSize[0]);
    resultShape.push_back(shape[1] / actualMicroSize[1]);
    resultShape.push_back(actualMicroSize[0]);
    resultShape.push_back(actualMicroSize[1]);

    auto resultType = RankedTensorType::get(resultShape, elementType);
    return build(builder, state, resultType, base, offsets,
               builder.getDenseI32ArrayAttr(shape),
               builder.getDenseI32ArrayAttr(actualMicroSize));
  }else{
    SmallVector<int64_t> resultShape;
    resultShape.push_back(shape[0] / micro_size[0]);
    resultShape.push_back(shape[1] / micro_size[1]);
    resultShape.push_back(micro_size[0]);
    resultShape.push_back(micro_size[1]);
    auto resultType = RankedTensorType::get(resultShape, elementType);

    return build(builder, state, resultType, base, offsets,
                builder.getDenseI32ArrayAttr(shape),
                builder.getDenseI32ArrayAttr(micro_size));
  }
}

} // namespace xsmt
} // namespace mlir
