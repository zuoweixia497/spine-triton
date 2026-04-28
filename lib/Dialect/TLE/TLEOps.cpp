//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/TLE/IR/TLEDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Types.h"

// clang-format off
#include "triton-shared/Dialect/TLE/IR/TLEDialect.cpp.inc"
#define GET_OP_CLASSES
#include "triton-shared/Dialect/TLE/IR/TLEOps.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::tle;

static RankedTensorType getTensorPointeeType(Type type) {
  auto ptrTy = dyn_cast<triton::PointerType>(type);
  if (!ptrTy)
    return {};
  return dyn_cast<RankedTensorType>(ptrTy.getPointeeType());
}

// ============================================================================
// TLE Dialect initialization
// ============================================================================
void mlir::tle::TLEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TLE/IR/TLEOps.cpp.inc"
      >();
}

// ============================================================================
// ToTensorOp Builder / Verification
// ============================================================================
void ToTensorOp::build(OpBuilder &builder, OperationState &state,
                       Value buffer) {
  auto resultType = getTensorPointeeType(buffer.getType());
  state.addOperands(buffer);
  state.addTypes(resultType);
}

LogicalResult ToTensorOp::verify() {
  auto pointeeTy = getTensorPointeeType(getBuffer().getType());
  if (!pointeeTy)
    return emitOpError("buffer must be a pointer to ranked tensor");

  auto resultTy = cast<RankedTensorType>(getResult().getType());
  if (resultTy != pointeeTy)
    return emitOpError("result type must match buffer pointee tensor type");

  return success();
}

// ============================================================================
// ToBufferOp Builder / Verification
// ============================================================================
void ToBufferOp::build(OpBuilder &builder, OperationState &state, Value tensor,
                       Value buffer) {
  state.addOperands(tensor);
  state.addOperands(buffer);
  state.addTypes(buffer.getType());
}

LogicalResult ToBufferOp::verify() {
  auto tensorTy = cast<RankedTensorType>(getTensor().getType());
  auto pointeeTy = getTensorPointeeType(getBuffer().getType());
  if (!pointeeTy)
    return emitOpError("buffer must be a pointer to ranked tensor");

  if (tensorTy != pointeeTy)
    return emitOpError("tensor type must match buffer pointee tensor type");

  if (getResult().getType() != getBuffer().getType())
    return emitOpError("result type must match buffer type");

  return success();
}

// ============================================================================
// LocalPtrOp Builder / Verification / Assembly
// ============================================================================
LogicalResult LocalPtrOp::verify() {
  auto bufferTy = getTensorPointeeType(getBuffer().getType());
  if (!bufferTy)
    return emitOpError("buffer must be a pointer to ranked tensor");

  auto indices = getIndices();
  int64_t rank = bufferTy.getRank();

  // No indices: result must be a tensor of pointers matching buffer shape.
  if (indices.empty()) {
    auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
    if (!resultTy)
      return emitOpError("no-index mode requires tensor-of-pointers result");
    if (resultTy.getShape() != bufferTy.getShape())
      return emitOpError("no-index result shape must match buffer shape");
    return success();
  }

  // Index count must match buffer rank.
  if (static_cast<int64_t>(indices.size()) != rank)
    return emitOpError("expected ")
           << rank << " indices, got " << indices.size();

  // Classify: all scalar or all tensor (same shape).
  bool allScalar = true;
  bool allTensor = true;
  ArrayRef<int64_t> tensorShape;
  for (auto idx : indices) {
    if (auto tensorTy = dyn_cast<RankedTensorType>(idx.getType())) {
      allScalar = false;
      if (tensorShape.empty()) {
        tensorShape = tensorTy.getShape();
      } else if (tensorTy.getShape() != tensorShape) {
        return emitOpError("all tensor indices must have identical shapes");
      }
    } else {
      allTensor = false;
    }
  }

  if (!allScalar && !allTensor)
    return emitOpError("indices must be either all scalar or all tensor");

  if (allScalar) {
    // Result must be a scalar pointer.
    if (isa<RankedTensorType>(getResult().getType()))
      return emitOpError("scalar indices require scalar pointer result");
  } else {
    // Result must be a tensor of pointers with matching shape.
    auto resultTy = dyn_cast<RankedTensorType>(getResult().getType());
    if (!resultTy)
      return emitOpError("tensor indices require tensor-of-pointers result");
    if (resultTy.getShape() != tensorShape)
      return emitOpError("result shape must match index tensor shape");
  }

  return success();
}

void LocalPtrOp::print(OpAsmPrinter &printer) {
  printer << " " << getBuffer();
  if (!getIndices().empty()) {
    printer << "[";
    llvm::interleaveComma(getIndices(), printer);
    printer << "]";
  }
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getBuffer().getType() << " -> " << getResult().getType();
}

ParseResult LocalPtrOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand buffer;
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
  Type bufferType, resultType;

  if (parser.parseOperand(buffer))
    return failure();

  // Optional [indices].
  if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseOperandList(indices) || parser.parseRSquare())
      return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(bufferType) || parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();

  // Resolve buffer operand.
  if (parser.resolveOperand(buffer, bufferType, result.operands))
    return failure();

  // Resolve index operands: infer types from result.
  if (!indices.empty()) {
    auto ptrTy = dyn_cast<triton::PointerType>(bufferType);
    if (!ptrTy)
      return failure();
    auto tensorTy = dyn_cast<RankedTensorType>(ptrTy.getPointeeType());
    if (!tensorTy)
      return failure();

    // Check if result is tensor (tensor indices) or scalar (scalar indices).
    if (auto resTensorTy = dyn_cast<RankedTensorType>(resultType)) {
      // Tensor indices: each index is tensor<shape x i32>.
      auto idxTy = RankedTensorType::get(
          resTensorTy.getShape(), IntegerType::get(parser.getContext(), 32));
      for (auto &idx : indices) {
        if (parser.resolveOperand(idx, idxTy, result.operands))
          return failure();
      }
    } else {
      // Scalar indices: each index is i32.
      auto idxTy = IntegerType::get(parser.getContext(), 32);
      for (auto &idx : indices) {
        if (parser.resolveOperand(idx, idxTy, result.operands))
          return failure();
      }
    }
  }

  result.addTypes(resultType);
  return success();
}

// ============================================================================
// ExtractTileOp Builder
// ============================================================================
void ExtractTileOp::build(OpBuilder &builder, OperationState &state, Value src,
                          Value index, ArrayRef<int64_t> tileShape) {
  auto srcType = cast<RankedTensorType>(src.getType());
  auto resultType = RankedTensorType::get(tileShape, srcType.getElementType());
  state.addOperands(src);
  state.addOperands(index);
  state.addAttribute("tile_shape", builder.getDenseI64ArrayAttr(tileShape));
  state.addTypes(resultType);
}

// ============================================================================
// ExtractTileOp Verification
// ============================================================================
LogicalResult ExtractTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());
  auto srcShape = srcTy.getShape();
  auto dstShape = dstTy.getShape();

  // Get tile_shape attribute
  auto tileShapeRawAttr = getOperation()->getAttr("tile_shape");
  SmallVector<int64_t> tileShape;
  if (auto denseArray64 =
          mlir::dyn_cast<mlir::DenseI64ArrayAttr>(tileShapeRawAttr)) {
    for (auto v : denseArray64.asArrayRef())
      tileShape.push_back(v);
  }

  // Check 1: element types must match
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitError("result element type must match source element type");

  // Check 2: rank must match
  if (srcTy.getRank() != dstTy.getRank())
    return emitError("result rank must equal source rank");

  // Check 3: tile_shape rank must match source rank
  if (tileShape.size() != srcShape.size())
    return emitOpError("tile_shape rank must match source rank");

  // Check 4: tile_shape positive, divisible, dst shape == tile_shape
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (tileShape[i] <= 0)
      return emitOpError("tile_shape must be positive at dimension ") << i;
    if (srcShape[i] % tileShape[i] != 0)
      return emitOpError(
                 "source shape must be divisible by tile_shape at dimension ")
             << i << " (source=" << srcShape[i] << ", tile=" << tileShape[i]
             << ")";
    if (dstShape[i] != tileShape[i])
      return emitOpError("result shape must equal tile_shape at dimension ")
             << i;
  }

  // Determine if index is a static constant
  auto indexConstOp =
      getOperation()->getOperand(1).getDefiningOp<arith::ConstantOp>();

  if (!indexConstOp) {
    // Dynamic index: skip out-of-bounds checks
    return success();
  }

  // Full checks for static index
  int64_t index =
      mlir::cast<mlir::IntegerAttr>(indexConstOp.getValue()).getInt();

  int64_t totalTiles = 1;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    totalTiles *= srcShape[i] / tileShape[i];
  }

  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  return success();
}

// ============================================================================
// InsertTileOp Type Inference
// ============================================================================
LogicalResult InsertTileOp::inferReturnTypes(
    [[maybe_unused]] MLIRContext *context,
    [[maybe_unused]] std::optional<Location> location, ValueRange operands,
    [[maybe_unused]] DictionaryAttr attributes,
    [[maybe_unused]] OpaqueProperties properties,
    [[maybe_unused]] RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  if (operands.size() < 3)
    return failure();

  auto srcTy = dyn_cast<RankedTensorType>(operands[0].getType());
  auto tileTy = dyn_cast<RankedTensorType>(operands[1].getType());
  if (!srcTy || !tileTy)
    return failure();

  if (srcTy.getElementType() != tileTy.getElementType() ||
      srcTy.getRank() != tileTy.getRank())
    return failure();

  inferredReturnTypes.clear();
  inferredReturnTypes.push_back(srcTy);
  return success();
}

// ============================================================================
// InsertTileOp Verification
// ============================================================================
LogicalResult InsertTileOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto tileTy = cast<RankedTensorType>(getTile().getType());
  auto dstTy = cast<RankedTensorType>(getResult().getType());

  auto srcShape = srcTy.getShape();
  auto tileShape = tileTy.getShape();
  auto dstShape = dstTy.getShape();

  // Check 1: element types must match
  if (srcTy.getElementType() != tileTy.getElementType())
    return emitOpError("tile element type must match source element type");
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError("result element type must match source element type");

  // Check 2: rank must match
  if (srcTy.getRank() != tileTy.getRank())
    return emitOpError("tile rank must equal source rank");
  if (srcTy.getRank() != dstTy.getRank())
    return emitOpError("result rank must equal source rank");

  // Check 3: result shape must equal source shape
  if (dstShape != srcShape)
    return emitOpError("result shape must equal source shape");

  // Check 4: tile_shape positive and divides source shape
  SmallVector<int64_t> logicalGridShape(srcShape.size(), 0);
  int64_t totalTiles = 1;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (tileShape[i] <= 0)
      return emitOpError("tile shape must be positive at dimension ") << i;
    if (srcShape[i] % tileShape[i] != 0)
      return emitOpError(
                 "source shape must be divisible by tile shape at dimension ")
             << i << " (source=" << srcShape[i] << ", tile=" << tileShape[i]
             << ")";
    logicalGridShape[i] = srcShape[i] / tileShape[i];
    totalTiles *= logicalGridShape[i];
  }

  // Determine if index is a static constant
  auto idxDef =
      getOperation()->getOperand(2).getDefiningOp<arith::ConstantOp>();
  if (!idxDef) {
    // Dynamic index: skip out-of-bounds checks
    return success();
  }

  // Full checks for static index
  int64_t index = mlir::cast<mlir::IntegerAttr>(idxDef.getValue()).getInt();
  if (index < 0 || index >= totalTiles)
    return emitOpError("index out of bounds for tile grid: index=")
           << index << ", total_tiles=" << totalTiles;

  return success();
}
