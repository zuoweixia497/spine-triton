//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/XsmtToLinalg/XsmtToLinalg.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "xsmt-to-linalg"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-shared/Dialect/xsmt/IR/XSMTDialect.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::memref;
using namespace mlir::tensor;
using namespace mlir::bufferization;
using namespace mlir::xsmt;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/XsmtToLinalg/Passes.h.inc"


Value ensureIndexType(Location loc, Value value, PatternRewriter &rewriter) {
    auto indexType = rewriter.getIndexType();
    if (value.getType() == indexType) {
      return value;
    }
    return rewriter.create<arith::IndexCastOp>(loc, indexType, value);
  }

Value createZeroConstant(PatternRewriter &rewriter, Location loc, Type elementType) {
    if (elementType.isF32()) {
      return rewriter.create<arith::ConstantFloatOp>(
          loc, APFloat(0.0f), rewriter.getF32Type());
    } else if (elementType.isF64()) {
      return rewriter.create<arith::ConstantFloatOp>(
          loc, APFloat(0.0), rewriter.getF64Type());
    } else if (elementType.isF16()) {
      return rewriter.create<arith::ConstantFloatOp>(
          loc, APFloat::getZero(rewriter.getF16Type().getFloatSemantics()),
          rewriter.getF16Type());
    } else if (elementType.isBF16()) {
      return rewriter.create<arith::ConstantFloatOp>(
          loc, APFloat::getZero(rewriter.getBF16Type().getFloatSemantics()),
          rewriter.getBF16Type());
    } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
      return rewriter.create<arith::ConstantIntOp>(
          loc, 0, intType.getWidth());
    }
    return nullptr;
  }


tensor::ExtractSliceOp CheckExtractSliceOpUser(Value value) {
    SmallVector<tensor::ExtractSliceOp> ExtractSliceOps;
    for (Operation *user : value.getUsers()) {
        if (auto ExtractSliceOp = dyn_cast<tensor::ExtractSliceOp>(user)) {
          ExtractSliceOps.push_back(ExtractSliceOp);
        }
    }
    if (ExtractSliceOps.size() == 1) {
      return ExtractSliceOps[0];
    }
    return nullptr;
}

Value createCeilDivUI(PatternRewriter &rewriter, Location loc, Value dividend, Value divisor) {
  auto type = dividend.getType();
  auto one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(type, 1));
  auto sum = rewriter.create<arith::AddIOp>(loc, dividend, divisor);
  auto numerator = rewriter.create<arith::SubIOp>(loc, sum, one);
  return rewriter.create<arith::DivUIOp>(loc, numerator, divisor);
}

struct DescriptorLoadPattern : public OpRewritePattern<DescriptorLoadOp> {
  DescriptorLoadPattern(MLIRContext *context)
      : OpRewritePattern<DescriptorLoadOp>(context) {}

  LogicalResult matchAndRewrite(DescriptorLoadOp descriptorLoadOp,
                                PatternRewriter &rewriter) const override {
    Value result = descriptorLoadOp.getResult();

    bool hasTranspose = false;
    linalg::TransposeOp transposeOp;
    for (auto &use : result.getUses()) {
      if (auto transpose = dyn_cast<linalg::TransposeOp>(use.getOwner())) {
        if (hasTranspose) {
          return failure();
        }
        hasTranspose = true;
        transposeOp = transpose;
      }
    }

    Value sourcePtr = descriptorLoadOp.getBase();
    auto unrealizedCast = sourcePtr.getDefiningOp<UnrealizedConversionCastOp>();
    if (!unrealizedCast) return failure();

    Value reinterpretCast = unrealizedCast.getOperand(0);
    auto reinterpretCastOp = reinterpretCast.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretCastOp) return failure();

    auto sizes = reinterpretCastOp.getSizes();
    if (sizes.size() != 2) {
      return failure();
    }
    Value dim0Value = sizes[0];
    Value dim1Value = sizes[1];

    auto microSizeAttr = descriptorLoadOp.getMicroSize();
    if (microSizeAttr.size() != 2) {
      return failure();
    }
    int32_t tile0 = microSizeAttr[0];
    int32_t tile1 = microSizeAttr[1];

    Location loc = descriptorLoadOp.getLoc();
    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    Value tile0Value = rewriter.create<ConstantIndexOp>(loc, tile0);
    Value tile1Value = rewriter.create<ConstantIndexOp>(loc, tile1);

    auto offsets = descriptorLoadOp.getOffsets();
    Value offset0 = offsets[0];
    Value offset1 = offsets[1];
    Value rankedMemRef = reinterpretCast;

    Value indexOffset0 = ensureIndexType(loc, offset0, rewriter);
    Value indexOffset1 = ensureIndexType(loc, offset1, rewriter);
    Value indexDim0Value = ensureIndexType(loc, dim0Value, rewriter);
    Value indexDim1Value = ensureIndexType(loc, dim1Value, rewriter);

    auto shapes = descriptorLoadOp.getShape();
    Value size_m1 = rewriter.create<SubIOp>(loc, indexDim0Value, indexOffset0);
    Value shape0 = rewriter.create<ConstantIndexOp>(loc, shapes[0]);
    Value size_m2 = rewriter.create<MinSIOp>(loc, size_m1, shape0);

    Value size_k1 = rewriter.create<SubIOp>(loc, indexDim1Value, indexOffset1);
    Value shape1 = rewriter.create<ConstantIndexOp>(loc, shapes[1]);
    Value size_k2 = rewriter.create<MinSIOp>(loc, size_k1, shape1);

    Value subview = rewriter.create<SubViewOp>(
        loc,
        rankedMemRef,
        ArrayRef<OpFoldResult>{indexOffset0, indexOffset1},
        ArrayRef<OpFoldResult>{size_m2, size_k2},
        ArrayRef<OpFoldResult>{c1, c1}
    );

    auto subviewType = cast<MemRefType>(subview.getType());
    auto tensorType = RankedTensorType::get(subviewType.getShape(), subviewType.getElementType());
    Value tensor = rewriter.create<ToTensorOp>(loc, tensorType, subview,
                                              /*restrict=*/rewriter.getUnitAttr());

    if (!hasTranspose) {
      return convertWithoutTranspose(descriptorLoadOp, tensor, size_m2, size_k2, tile0, tile1, rewriter);
    }

    auto permutation = transposeOp.getPermutation();
    if (permutation.size() != 4 ||
        permutation[0] != 1 || permutation[1] != 0 ||
        permutation[2] != 3 || permutation[3] != 2) {
      return failure();
    }

    return convertWithTranspose(descriptorLoadOp, transposeOp, tensor, size_m2, size_k2, tile0, tile1, rewriter);
  }

private:
  LogicalResult convertWithoutTranspose(DescriptorLoadOp descriptorLoadOp,
                                        Value tensor,
                                        Value dim0, Value dim1,
                                        int32_t tile0, int32_t tile1,
                                        PatternRewriter &rewriter) const {
    Location loc = descriptorLoadOp.getLoc();
    Value tile0Value = rewriter.create<ConstantIndexOp>(loc, tile0);
    Value tile1Value = rewriter.create<ConstantIndexOp>(loc, tile1);
    Value packedRows = createCeilDivUI(rewriter, loc, dim0, tile0Value);
    Value packedCols = createCeilDivUI(rewriter, loc, dim1, tile1Value);

    SmallVector<int64_t, 4> shapeDims;
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(tile0);
    shapeDims.push_back(tile1);

    auto tensorType = cast<RankedTensorType>(tensor.getType());
    Type elementType = tensorType.getElementType();

    auto emptyType = RankedTensorType::get(shapeDims, elementType);

    Value emptyTensor = rewriter.create<EmptyOp>(
        loc, emptyType, ValueRange{packedRows, packedCols});
    Value paddingValue = createZeroConstant(rewriter, loc, elementType);

    Value packedResult = rewriter.create<PackOp>(
        loc,
        tensor,
        emptyTensor,
        ArrayRef<int64_t>{0, 1},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile0), rewriter.getIndexAttr(tile1)},
        paddingValue,
        ArrayRef<int64_t>{0, 1}
    );

    auto originalResultType = cast<RankedTensorType>(descriptorLoadOp.getResult().getType());
    Value castResult = rewriter.create<tensor::CastOp>(loc, originalResultType, packedResult);

    rewriter.replaceOp(descriptorLoadOp, castResult);
    cleanupUnusedOps(descriptorLoadOp, rewriter);

    return success();
  }

  LogicalResult convertWithTranspose(DescriptorLoadOp descriptorLoadOp,
                                     linalg::TransposeOp transposeOp,
                                     Value tensor,
                                     Value dim0, Value dim1,
                                     int32_t tile0, int32_t tile1,
                                     PatternRewriter &rewriter) const {
    Location loc = descriptorLoadOp.getLoc();
    Value tile0Value = rewriter.create<ConstantIndexOp>(loc, tile0);
    Value tile1Value = rewriter.create<ConstantIndexOp>(loc, tile1);
    Value packedRows = createCeilDivUI(rewriter, loc, dim1, tile1Value);
    Value packedCols = createCeilDivUI(rewriter, loc, dim0, tile0Value);

    auto tensorType = cast<RankedTensorType>(tensor.getType());
    Type elementType = tensorType.getElementType();

    SmallVector<int64_t, 4> shapeDims;
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(tile1);
    shapeDims.push_back(tile0);

    auto emptyType = RankedTensorType::get(shapeDims, elementType);

    Value emptyTensor = rewriter.create<EmptyOp>(
        loc, emptyType, ValueRange{packedRows, packedCols});

    Value paddingValue = createZeroConstant(rewriter, loc, elementType);

    Value packedResult = rewriter.create<PackOp>(
        loc,
        tensor,
        emptyTensor,
        ArrayRef<int64_t>{1, 0},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile1), rewriter.getIndexAttr(tile0)},
        paddingValue,
        ArrayRef<int64_t>{1, 0}
    );

    auto transposeResultType = cast<RankedTensorType>(transposeOp->getResult(0).getType());
    Value castResult = rewriter.create<tensor::CastOp>(loc, transposeResultType , packedResult);

    rewriter.replaceOp(transposeOp, castResult);
    rewriter.eraseOp(descriptorLoadOp);
    cleanupUnusedOps(descriptorLoadOp, rewriter);

    return success();
  }

  void cleanupUnusedOps(DescriptorLoadOp descriptorLoadOp,
                        PatternRewriter &rewriter) const {
    Value sourcePtr = descriptorLoadOp.getBase();
    if (auto unrealizedCast = sourcePtr.getDefiningOp<UnrealizedConversionCastOp>()) {
      Value reinterpretCast = unrealizedCast.getOperand(0);
      if (unrealizedCast->getUses().empty()) {
        rewriter.eraseOp(unrealizedCast);
        if (auto reinterpretCastOp = reinterpretCast.getDefiningOp<memref::ReinterpretCastOp>()) {
          if (reinterpretCastOp->getUses().empty()) {
            rewriter.eraseOp(reinterpretCastOp);
          }
        }
      }
    }
  }
};


struct FillOpPattern : public OpRewritePattern<linalg::FillOp> {
public:
  FillOpPattern(MLIRContext *context)
      : OpRewritePattern<linalg::FillOp>(context) {}

  LogicalResult matchAndRewrite(linalg::FillOp fillOp,
                                PatternRewriter &rewriter) const override {

    Location loc = fillOp.getLoc();
    Value fillValue = fillOp.getInputs()[0];
    Value fillOutput = fillOp.getOutputs()[0];

    auto outputType = dyn_cast<TensorType>(fillOutput.getType());
    if (!outputType) {
      return rewriter.notifyMatchFailure(fillOp, "FAIL: Output is not a tensor type");
    }

    SmallVector<int64_t> shape(outputType.getShape().begin(),
                              outputType.getShape().end());

    auto allocOp = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(shape, outputType.getElementType()));
    PatternRewriter::InsertionGuard guard(rewriter);

    if (shape.size() == 2) {
      Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      Value dim0 = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
      Value dim1 = rewriter.create<arith::ConstantIndexOp>(loc, shape[1]);
      auto outerLoop = rewriter.create<scf::ForOp>(loc, c0, dim0, c1);

      rewriter.setInsertionPointToStart(outerLoop.getBody());
      auto innerLoop = rewriter.create<scf::ForOp>(loc, c0, dim1, c1);
      rewriter.setInsertionPointToStart(innerLoop.getBody());

      Value i = outerLoop.getInductionVar();
      Value j = innerLoop.getInductionVar();
      auto storeOp = rewriter.create<memref::StoreOp>(loc, fillValue, allocOp, ValueRange{i, j});
      rewriter.setInsertionPointAfter(outerLoop);
    } else {
      return rewriter.notifyMatchFailure(fillOp, "FAIL: Not 2D tensor");
    }

    auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
        loc, outputType, allocOp, /*restrict=*/true);

    rewriter.replaceOp(fillOp, toTensorOp.getResult());
    return success();
  }
};

struct ExtractSliceMaterializePattern : public OpRewritePattern<tensor::ExtractSliceOp> {
public:
  ExtractSliceMaterializePattern(MLIRContext *context)
      : OpRewritePattern<tensor::ExtractSliceOp>(context) {}

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extractSliceOp,
                                PatternRewriter &rewriter) const override {
    if (!extractSliceOp->hasOneUse()) {
      return rewriter.notifyMatchFailure(extractSliceOp, "FAIL: extractSliceOp has more than one user");
    }

    auto materializeOp = dyn_cast<bufferization::MaterializeInDestinationOp>(
        *extractSliceOp->getUsers().begin());
    if (!materializeOp) {
      return rewriter.notifyMatchFailure(extractSliceOp, "FAIL: extractSliceOp user is not materialize_in_destination");
    }

    return convertExtractSliceToSubView(extractSliceOp, materializeOp, rewriter);
  }

private:
  LogicalResult convertExtractSliceToSubView(
      tensor::ExtractSliceOp extractSliceOp,
      bufferization::MaterializeInDestinationOp materializeOp,
      PatternRewriter &rewriter) const {
    Location loc = extractSliceOp.getLoc();

    Value source = extractSliceOp.getSource();
    auto toTensorOp = source.getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp) {
      return rewriter.notifyMatchFailure(extractSliceOp, "FAIL: Source is not a ToTensorOp");
    }

    Value memrefSource = toTensorOp->getOperand(0);
    Value destMemref = materializeOp.getDest();

    rewriter.setInsertionPointAfter(materializeOp);
    auto subViewOp = rewriter.create<memref::SubViewOp>(
        loc,
        memrefSource,
        extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides()
    );
    auto copyOp = rewriter.create<memref::CopyOp>(loc, subViewOp, destMemref);
    rewriter.eraseOp(materializeOp);
    rewriter.eraseOp(extractSliceOp);
    return success();
  }
};

struct ViewOpPattern : public OpRewritePattern<xsmt::ViewOp> {
public:
  ViewOpPattern(MLIRContext *context)
      : OpRewritePattern<xsmt::ViewOp>(context) {}

  LogicalResult matchAndRewrite(xsmt::ViewOp viewOp,
                                PatternRewriter &rewriter) const override {

    Location loc = viewOp.getLoc();
    Value base = viewOp.getBase();
    ValueRange offsets = viewOp.getOffsets();
    ArrayRef<int32_t> shape = viewOp.getShape();
    ArrayRef<int32_t> microSize = viewOp.getMicroSize();

    auto resultType = dyn_cast<TensorType>(viewOp.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(viewOp, "Result is not a tensor type");
    }
    Type elementType = resultType.getElementType();
    if (shape.size() != 2 || microSize.size() != 2) {
      return rewriter.notifyMatchFailure(viewOp, "Expected shape and micro_size to have 2 elements each");
    }
    if (offsets.size() != 2) {
      return rewriter.notifyMatchFailure(viewOp, "Expected exactly 2 offset values");
    }
    auto toTensorOp = base.getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp) {
      return rewriter.notifyMatchFailure(viewOp, "Base is not a ToTensorOp result");
    }

    Value memrefBaseValue = toTensorOp->getOperand(0);
    SmallVector<OpFoldResult> sliceSizes;
    SmallVector<OpFoldResult> offsetValues;
    for (auto offset : offsets) {
      offsetValues.push_back(ensureIndexType(loc, offset, rewriter));
    }

    if (auto ExtractSliceOp = CheckExtractSliceOpUser(base)) {
        sliceSizes = ExtractSliceOp.getMixedSizes();
    } else {
        llvm::errs() << "Error: There are multiple ExtractSliceOp users\n";
        return failure();
    }

    Value c1 = rewriter.create<ConstantIndexOp>(loc, 1);
    auto size_m1 = subOFRs(sliceSizes[0], offsetValues[0], loc, rewriter);
    Value size_m2 = ofrToIndexValue(size_m1, loc, rewriter);
    Value shape0 = rewriter.create<ConstantIndexOp>(loc, shape[0]);
    Value size_m3 = rewriter.create<MinSIOp>(loc, size_m2, shape0);

    auto size_n1 = subOFRs(sliceSizes[1], offsetValues[1], loc, rewriter);
    Value size_n2 = ofrToIndexValue(size_n1, loc, rewriter);
    Value shape1 = rewriter.create<ConstantIndexOp>(loc, shape[1]);
    Value size_n3 = rewriter.create<MinSIOp>(loc, size_n2, shape1);

    auto subview = rewriter.create<memref::SubViewOp>(
        loc,
        memrefBaseValue,
        offsetValues,
        ArrayRef<OpFoldResult>{size_m3, size_n3},
        ArrayRef<OpFoldResult>{c1, c1}
    );
    auto subviewType = cast<MemRefType>(subview.getType());
    auto tensorType = RankedTensorType::get(subviewType.getShape(), subviewType.getElementType());
    Value toTensorOp2 = rewriter.create<ToTensorOp>(loc, tensorType, subview,
                                              /*restrict=*/rewriter.getUnitAttr());

    Value tile0Value = rewriter.create<ConstantIndexOp>(loc, microSize[0]);
    Value tile1Value = rewriter.create<ConstantIndexOp>(loc, microSize[1]);
    Value packedRows = createCeilDivUI(rewriter, loc, size_m3, tile0Value);
    Value packedCols = createCeilDivUI(rewriter, loc, size_n3, tile1Value);

    SmallVector<int64_t, 4> shapeDims;
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(microSize[0]);
    shapeDims.push_back(microSize[1]);
    auto emptyType = RankedTensorType::get(shapeDims, tensorType.getElementType());
    Value emptyTensor = rewriter.create<EmptyOp>(
        loc, emptyType, ValueRange{packedRows, packedCols});

    Value paddingValue = createZeroConstant(rewriter, loc, elementType);
    if (!paddingValue) {
      return rewriter.notifyMatchFailure(viewOp, "Unsupported element type for padding");
    }

    Value packedResult = rewriter.create<linalg::PackOp>(
        loc,
        toTensorOp2,
        emptyTensor,
        ArrayRef<int64_t>{0, 1},
        ArrayRef<OpFoldResult>{
        rewriter.getIndexAttr(microSize[0]),
        rewriter.getIndexAttr(microSize[1])
        },
        paddingValue,
        ArrayRef<int64_t>{0, 1});
    auto originalResultType = cast<RankedTensorType>(viewOp.getResult().getType());
    Value castResult = rewriter.create<tensor::CastOp>(loc, originalResultType, packedResult);
    rewriter.replaceOp(viewOp, castResult);
    return success();
  }
};

struct LowerXSMTMMT4D : public OpRewritePattern<xsmt::MMT4DOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmt::MMT4DOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value a = op.getA();
    Value b = op.getB();
    Value c = op.getC();

    if (auto cast = a.getDefiningOp<tensor::CastOp>())
      a = cast.getSource();
    if (auto cast = b.getDefiningOp<tensor::CastOp>())
      b = cast.getSource();
    if (auto cast = c.getDefiningOp<tensor::CastOp>())
      c = cast.getSource();

    auto cPackOp = c.getDefiningOp<linalg::PackOp>();

    Value packInput = cPackOp.getSource();
    auto toTensorOp = packInput.getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp)
      return rewriter.notifyMatchFailure(op, "Packed input must come from to_tensor");

    Value subview = toTensorOp->getOperand(0);
    auto mmt4dOp = rewriter.create<linalg::Mmt4DOp>(
        loc, ValueRange{a, b}, ValueRange{cPackOp});
    ArrayRef<int64_t> staticInnerTiles = cPackOp.getStaticInnerTiles();
    if (staticInnerTiles.empty()) {
      return rewriter.notifyMatchFailure(op, "static_inner_tiles is empty");
    }

    SmallVector<OpFoldResult> innerTiles;
    for (int64_t tile : staticInnerTiles) {
      innerTiles.push_back(rewriter.getIndexAttr(tile));
    }
    auto unpackOp = rewriter.create<linalg::UnPackOp>(
        loc,
        mmt4dOp.getResult(0),
        toTensorOp,
        cPackOp.getInnerDimsPos(),
        innerTiles,
        cPackOp.getOuterDimsPerm());
    rewriter.replaceOp(op, unpackOp.getResult());
    return success();
  }
};

class ForToForallPattern : public OpRewritePattern<scf::ForOp> {
public:
  ForToForallPattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto bindSubBlockAttr = forOp->getAttrOfType<BoolAttr>("bind_sub_block");
    if (!bindSubBlockAttr || !bindSubBlockAttr.getValue()) {
      return failure();
    }

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value step = forOp.getStep();
    Value indVar = forOp.getInductionVar();
    auto convertToIndex = [&](Value val) -> Value {
      if (val.getType().isIndex()) return val;
      return rewriter.create<arith::IndexCastOp>(
          forOp.getLoc(), rewriter.getIndexType(), val);
    };

    SmallVector<OpFoldResult> lbs = {convertToIndex(lb)};
    SmallVector<OpFoldResult> ubs = {convertToIndex(ub)};
    SmallVector<OpFoldResult> steps = {convertToIndex(step)};
    SmallVector<Value> outputs;
    auto initArgs = forOp.getInitArgs();
    for (auto initArg : initArgs) {
      outputs.push_back(initArg);
    }

    auto forallOp = rewriter.create<scf::ForallOp>(
        forOp.getLoc(), lbs, ubs, steps, outputs, std::nullopt);

    Block &forallBlock = forallOp.getRegion().front();
    Block &forBody = *forOp.getBody();
    auto inParallelOp = cast<scf::InParallelOp>(forallBlock.getTerminator());
    rewriter.setInsertionPoint(inParallelOp);
    IRMapping mapping;

    Value forallIndVar = forallBlock.getArgument(0);
    auto castIndVar = rewriter.create<arith::IndexCastOp>(
        forOp.getLoc(), indVar.getType(), forallIndVar);
    mapping.map(indVar, castIndVar);

    unsigned numInitArgs = initArgs.size();
    for (unsigned i = 0; i < numInitArgs; ++i) {
      mapping.map(forBody.getArgument(i + 1), forallBlock.getArgument(i + 1));
    }

    for (auto &op : forBody.without_terminator()) {
      rewriter.clone(op, mapping);
    }
    rewriter.replaceOp(forOp, forallOp.getResults());
    return success();
  }
};

void mlir::triton::fillToMemrefConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<FillOpPattern>(patterns.getContext());
}

void mlir::triton::populateXsmtToLinalgConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ExtractSliceMaterializePattern>(patterns.getContext());
  patterns.add<ViewOpPattern>(patterns.getContext());
  patterns.add<DescriptorLoadPattern>(patterns.getContext());
}

void mlir::triton::MMT4DOpConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerXSMTMMT4D>(patterns.getContext());
}

void mlir::triton::ForToForallConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ForToForallPattern>(patterns.getContext());
}