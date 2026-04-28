//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
// TLE → Linalg/Tensor lowering patterns:
//   tle.to_tensor    → bufferization.to_tensor
//   tle.to_buffer    → bufferization.materialize_in_destination
//   tle.extract_tile → tensor.extract_slice
//   tle.insert_tile  → tensor.insert_slice
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TLEToLinalg/TLEToLinalg.h"
#include "triton-shared/Dialect/TLE/IR/TLEDialect.h"
#include "triton-shared/Dialect/TLE/IR/TLEOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define DEBUG_TYPE "tle-to-linalg"

using namespace mlir;

namespace {

MemRefType getMemRefTypeFromTensorPtr(Type type) {
  auto ptrTy = dyn_cast<triton::PointerType>(type);
  if (!ptrTy)
    return {};

  auto tensorTy = dyn_cast<RankedTensorType>(ptrTy.getPointeeType());
  if (!tensorTy)
    return {};

  return MemRefType::get(tensorTy.getShape(), tensorTy.getElementType(), {},
                         ptrTy.getAddressSpace());
}

// ============================================================================
// ToTensorOp → bufferization.to_tensor
// ============================================================================
struct ToTensorOpPattern : public OpRewritePattern<mlir::tle::ToTensorOp> {
  using OpRewritePattern<mlir::tle::ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tle::ToTensorOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value buffer = op.getBuffer();
    auto memrefTy = getMemRefTypeFromTensorPtr(buffer.getType());
    if (!memrefTy)
      return failure();

    auto cast =
        UnrealizedConversionCastOp::create(rewriter, loc, memrefTy, buffer);
    auto toTensor = bufferization::ToTensorOp::create(
        rewriter, loc, op.getResult().getType(), cast.getResult(0),
        /*restrict=*/true, /*writable=*/true);

    rewriter.replaceOp(op, toTensor.getResult());
    return success();
  }
};

// ============================================================================
// ToBufferOp → bufferization.materialize_in_destination
// ============================================================================
struct ToBufferOpPattern : public OpRewritePattern<mlir::tle::ToBufferOp> {
  using OpRewritePattern<mlir::tle::ToBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tle::ToBufferOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value buffer = op.getBuffer();
    auto memrefTy = getMemRefTypeFromTensorPtr(buffer.getType());
    if (!memrefTy)
      return failure();

    auto cast =
        UnrealizedConversionCastOp::create(rewriter, loc, memrefTy, buffer);
    auto materialize = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, op.getTensor(), cast.getResult(0));
    materialize.setWritable(true);

    auto resultCast = UnrealizedConversionCastOp::create(
        rewriter, loc, op.getResult().getType(), materialize.getResult());
    rewriter.replaceOp(op, resultCast.getResult(0));
    return success();
  }
};

// ============================================================================
// LocalPtrOp lowering
//
// Converts tle.local_ptr into memref-level operations:
//   - No indices:     bufferization.to_tensor (full view)
//   - Scalar indices: memref.load (scalar element access)
//   - Tensor indices: linalg.generic (gather-style element access)
//
// At this point in the pipeline (after StructuredToMemref), the buffer
// operand has been converted from TT_TensorPtr to memref via an
// UnrealizedConversionCast.  We look through the cast to get the memref.
// ============================================================================
struct LocalPtrOpPattern : public OpRewritePattern<mlir::tle::LocalPtrOp> {
  using OpRewritePattern<mlir::tle::LocalPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tle::LocalPtrOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value buffer = op.getBuffer();
    auto memrefTy = getMemRefTypeFromTensorPtr(buffer.getType());
    if (!memrefTy)
      return failure();

    auto cast =
        UnrealizedConversionCastOp::create(rewriter, loc, memrefTy, buffer);
    Value memrefVal = cast.getResult(0);
    auto indices = op.getIndices();

    if (indices.empty()) {
      // No-index mode: full-view → bufferization.to_tensor.
      auto tensorTy =
          RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());
      auto toTensor = bufferization::ToTensorOp::create(
          rewriter, loc, tensorTy, memrefVal,
          /*restrict=*/true, /*writable=*/true);

      // Bridge back to the original result type (pointer tensor) via cast.
      auto resultCast = UnrealizedConversionCastOp::create(
          rewriter, loc, op.getResult().getType(), toTensor.getResult());
      rewriter.replaceOp(op, resultCast.getResult(0));
      return success();
    }

    // Classify: all scalar or all tensor.
    bool allScalar = true;
    for (auto idx : indices) {
      if (isa<RankedTensorType>(idx.getType())) {
        allScalar = false;
        break;
      }
    }

    if (allScalar) {
      // Scalar indices → memref.load.
      SmallVector<Value> indexVals;
      for (auto idx : indices) {
        Value idxVal = idx;
        if (!idxVal.getType().isIndex())
          idxVal = arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getIndexType(), idxVal);
        indexVals.push_back(idxVal);
      }
      auto loadOp = memref::LoadOp::create(rewriter, loc, memrefVal, indexVals);

      // Bridge scalar element back to pointer result type via cast.
      auto resultCast = UnrealizedConversionCastOp::create(
          rewriter, loc, op.getResult().getType(), loadOp.getResult());
      rewriter.replaceOp(op, resultCast.getResult(0));
      return success();
    }

    // Tensor indices → linalg.generic gather.
    auto resultTensorTy = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTensorTy)
      return failure();

    auto elemTy = memrefTy.getElementType();
    auto outputTensorTy =
        RankedTensorType::get(resultTensorTy.getShape(), elemTy);
    int64_t rank = resultTensorTy.getRank();

    // Create init tensor for the output.
    auto emptyOp = tensor::EmptyOp::create(rewriter, loc,
                                           outputTensorTy.getShape(), elemTy);

    // Build indexing maps: each index tensor is identity-mapped.
    SmallVector<AffineMap> indexingMaps;
    auto identityMap = rewriter.getMultiDimIdentityMap(rank);
    for (size_t i = 0; i < indices.size(); ++i)
      indexingMaps.push_back(identityMap);
    indexingMaps.push_back(identityMap); // output

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // Cast index tensors to index type if needed.
    SmallVector<Value> indexTensors;
    auto indexTensorTy = RankedTensorType::get(resultTensorTy.getShape(),
                                               rewriter.getIndexType());
    for (auto idx : indices) {
      Value idxVal = idx;
      auto idxTy = dyn_cast<RankedTensorType>(idxVal.getType());
      if (!idxTy)
        return failure();
      if (!idxTy.getElementType().isIndex()) {
        // Use linalg.generic to cast element-wise, or arith.index_cast on
        // tensor.  For simplicity, use arith.index_cast (tensor-level).
        idxVal =
            arith::IndexCastOp::create(rewriter, loc, indexTensorTy, idxVal);
      }
      indexTensors.push_back(idxVal);
    }

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, outputTensorTy, /*inputs=*/indexTensors,
        /*outputs=*/ValueRange{emptyOp.getResult()}, indexingMaps,
        iteratorTypes, [&](OpBuilder &b, Location loc, ValueRange args) {
          // args[0..rank-1] are index values from index tensors.
          // args[rank] is the output element (unused init).
          SmallVector<Value> loadIndices;
          for (size_t i = 0; i < indices.size(); ++i)
            loadIndices.push_back(args[i]);
          auto loaded = memref::LoadOp::create(b, loc, memrefVal, loadIndices);
          linalg::YieldOp::create(b, loc, loaded.getResult());
        });

    // Bridge tensor result back to pointer-tensor result type via cast.
    auto resultCast = UnrealizedConversionCastOp::create(
        rewriter, loc, op.getResult().getType(), genericOp.getResult(0));
    rewriter.replaceOp(op, resultCast.getResult(0));
    return success();
  }
};

// ============================================================================
// Helper: delinearize a tile index into per-dimension offsets.
//
// Given:
//   srcShape  = [S0, S1, ..., Sn]
//   tileShape = [T0, T1, ..., Tn]
//   grid      = [S0/T0, S1/T1, ..., Sn/Tn]
//
// For a linear tile index `idx`, compute:
//   offsets[i] = (idx / stride_i % grid[i]) * tileShape[i]
//
// where stride_i = product(grid[i+1:]).
// ============================================================================
SmallVector<OpFoldResult> delinearizeIndex(Location loc, Value index,
                                           ArrayRef<int64_t> srcShape,
                                           ArrayRef<int64_t> tileShape,
                                           PatternRewriter &rewriter) {
  int64_t rank = srcShape.size();
  SmallVector<int64_t> grid(rank);
  for (int64_t i = 0; i < rank; ++i)
    grid[i] = srcShape[i] / tileShape[i];

  // Try to extract a static constant from the index value.
  std::optional<int64_t> staticIdx;
  if (auto constOp = index.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      staticIdx = intAttr.getInt();
  }

  SmallVector<OpFoldResult> offsets;
  if (staticIdx.has_value()) {
    // Static path: compute offsets at compile time.
    int64_t remain = *staticIdx;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t stride = 1;
      for (int64_t j = i + 1; j < rank; ++j)
        stride *= grid[j];
      int64_t tileIdx = remain / stride;
      remain %= stride;
      offsets.push_back(rewriter.getIndexAttr(tileIdx * tileShape[i]));
    }
  } else {
    // Dynamic path: emit arith ops to compute offsets at runtime.
    // Ensure index is index type.
    if (!index.getType().isIndex())
      index = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                         index);
    for (int64_t i = 0; i < rank; ++i) {
      int64_t stride = 1;
      for (int64_t j = i + 1; j < rank; ++j)
        stride *= grid[j];

      // tileIdx = (index / stride) % grid[i]
      Value tileIdx = index;
      if (stride != 1) {
        Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
        tileIdx = arith::DivUIOp::create(rewriter, loc, tileIdx, strideVal);
      }
      if (grid[i] != 1) {
        // Only need modulo if not the first dimension (or grid > 1)
        Value gridVal = arith::ConstantIndexOp::create(rewriter, loc, grid[i]);
        tileIdx = arith::RemUIOp::create(rewriter, loc, tileIdx, gridVal);
      }
      // offset = tileIdx * tileShape[i]
      Value tileSizeVal =
          arith::ConstantIndexOp::create(rewriter, loc, tileShape[i]);
      Value offset = arith::MulIOp::create(rewriter, loc, tileIdx, tileSizeVal);
      offsets.push_back(offset);
    }
  }
  return offsets;
}

// ============================================================================
// ExtractTileOp → tensor.extract_slice
// ============================================================================
struct ExtractTileOpPattern
    : public OpRewritePattern<mlir::tle::ExtractTileOp> {
  using OpRewritePattern<mlir::tle::ExtractTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tle::ExtractTileOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = op.getSrc();
    Value index = op.getIndex();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(op.getResult().getType());
    auto srcShape = srcTy.getShape();
    auto tileShape = dstTy.getShape();
    int64_t rank = srcShape.size();

    // Compute per-dimension offsets from the linear tile index.
    SmallVector<OpFoldResult> offsets =
        delinearizeIndex(loc, index, srcShape, tileShape, rewriter);

    // Sizes = tileShape (all static).
    SmallVector<OpFoldResult> sizes;
    for (int64_t i = 0; i < rank; ++i)
      sizes.push_back(rewriter.getIndexAttr(tileShape[i]));

    // Strides = all 1.
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto extractSlice = tensor::ExtractSliceOp::create(
        rewriter, loc, dstTy, src, offsets, sizes, strides);

    rewriter.replaceOp(op, extractSlice.getResult());
    return success();
  }
};

// ============================================================================
// InsertTileOp → tensor.insert_slice
// ============================================================================
struct InsertTileOpPattern : public OpRewritePattern<mlir::tle::InsertTileOp> {
  using OpRewritePattern<mlir::tle::InsertTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tle::InsertTileOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = op.getSrc();
    Value tile = op.getTile();
    Value index = op.getIndex();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto tileTy = cast<RankedTensorType>(tile.getType());
    auto srcShape = srcTy.getShape();
    auto tileShape = tileTy.getShape();
    int64_t rank = srcShape.size();

    // Compute per-dimension offsets from the linear tile index.
    SmallVector<OpFoldResult> offsets =
        delinearizeIndex(loc, index, srcShape, tileShape, rewriter);

    // Sizes = tileShape (all static).
    SmallVector<OpFoldResult> sizes;
    for (int64_t i = 0; i < rank; ++i)
      sizes.push_back(rewriter.getIndexAttr(tileShape[i]));

    // Strides = all 1.
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto insertSlice = tensor::InsertSliceOp::create(rewriter, loc, tile, src,
                                                     offsets, sizes, strides);

    rewriter.replaceOp(op, insertSlice.getResult());
    return success();
  }
};

} // namespace

// ============================================================================
// Pattern registration
// ============================================================================
void mlir::triton::populateTLEToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ToTensorOpPattern>(patterns.getContext());
  patterns.add<ToBufferOpPattern>(patterns.getContext());
  patterns.add<LocalPtrOpPattern>(patterns.getContext());
  patterns.add<ExtractTileOpPattern>(patterns.getContext());
  patterns.add<InsertTileOpPattern>(patterns.getContext());
}
