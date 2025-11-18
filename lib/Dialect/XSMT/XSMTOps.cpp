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
#include "triton-shared/Dialect/XSMT/IR/OpsEnums.cpp.inc"

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

void DescriptorLoadViewOp::build(OpBuilder &builder, OperationState &state,
                                 Value base, ValueRange offsets, ArrayRef<int32_t> shape,
                                 ArrayRef<int32_t> micro_size, Value destination) {
  auto resultType = destination.getType();
  return build(builder, state, resultType, base, offsets,
               builder.getDenseI32ArrayAttr(shape),
               builder.getDenseI32ArrayAttr(micro_size), destination);
}


void ViewOp::build(OpBuilder &builder, OperationState &state,
                   Value base, ValueRange offsets,
                   ArrayRef<int32_t> shape,
                   ArrayRef<int32_t> micro_size) {
  auto baseType = cast<RankedTensorType>(base.getType());
  bool allOnes = true;
  for (int32_t size : micro_size) {
    if (size != 1) {
      allOnes = false;
      break;
    }
  }
  if(allOnes){
    auto baseShape = baseType.getShape();
    auto actualMicroSize = {
      static_cast<int32_t>(baseShape[baseShape.size() - 2]),
      static_cast<int32_t>(baseShape[baseShape.size() - 1])
    };
    return build(builder, state, baseType, base, offsets,
               builder.getDenseI32ArrayAttr(shape),
               builder.getDenseI32ArrayAttr(actualMicroSize));
  }else{
    Type elementType = baseType.getElementType();

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

void AllocOp::build(OpBuilder &builder, OperationState &state,
                    ArrayRef<int32_t> shape,
                    ArrayRef<int32_t> micro_size,
                    Type elementType) {
  SmallVector<int64_t> resultShape;
  resultShape.push_back(shape[0] / micro_size[0]);
  resultShape.push_back(shape[1] / micro_size[1]);
  resultShape.push_back(micro_size[0]);
  resultShape.push_back(micro_size[1]);

  auto resultType = RankedTensorType::get(resultShape, elementType);
  build(builder, state, resultType,
        builder.getDenseI32ArrayAttr(shape),
        builder.getDenseI32ArrayAttr(micro_size));
}

} // namespace xsmt
} // namespace mlir
