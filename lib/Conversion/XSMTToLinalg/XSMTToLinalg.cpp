//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/XSMTToLinalg/XSMTToLinalg.h"
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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "xsmt-to-linalg"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::memref;
using namespace mlir::tensor;
using namespace mlir::bufferization;
using namespace mlir::xsmt;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/XSMTToLinalg/Passes.h.inc"


Value ensureIndexType(Location loc, Value value, PatternRewriter &rewriter) {
    auto indexType = rewriter.getIndexType();
    if (value.getType() == indexType) {
      return value;
    }
    return arith::IndexCastOp::create(rewriter, loc, indexType, value);
  }

Value ofrToIndexValue(const Location loc, const OpFoldResult ofr,
                      PatternRewriter &rewriter) {
  if (Value val = dyn_cast<Value>(ofr)) {
    assert(val.getType().isIntOrIndex());
    if (!val.getType().isIndex()) {
      val = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), val);
    }
    return val;
  }

  auto intVal = getIntAttr(ofr);
  if (intVal.has_value()) {
    return arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(intVal.value()));
  }
  llvm_unreachable("Unexpected OpFoldResult state");
  return nullptr;
}

Value createZeroConstant(PatternRewriter &rewriter, Location loc, Type elementType) {
    if (elementType.isF32()) {
      return arith::ConstantFloatOp::create(rewriter, loc, rewriter.getF32Type(), APFloat(0.0f));
    } else if (elementType.isF64()) {
      return arith::ConstantFloatOp::create(rewriter, loc, rewriter.getF64Type(), APFloat(0.0));
    } else if (elementType.isF16()) {
      return arith::ConstantFloatOp::create(rewriter, loc, rewriter.getF16Type(), APFloat::getZero(rewriter.getF16Type().getFloatSemantics()));
    } else if (elementType.isBF16()) {
      return arith::ConstantFloatOp::create(rewriter, loc, rewriter.getBF16Type(), APFloat::getZero(rewriter.getBF16Type().getFloatSemantics()));
    } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
      return arith::ConstantIntOp::create(rewriter, loc, 0, intType.getWidth());
    }
    return nullptr;
  }

Value createCeilDivUI(PatternRewriter &rewriter, Location loc, Value dividend, Value divisor) {
  auto indexType = rewriter.getIndexType();
  Value dividendIndex = ensureIndexType(loc, dividend, rewriter);
  Value divisorIndex = ensureIndexType(loc, divisor, rewriter);

  auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto sum = arith::AddIOp::create(rewriter, loc, dividendIndex, divisorIndex);
  auto numerator = arith::SubIOp::create(rewriter, loc, sum, one);
  return arith::DivUIOp::create(rewriter, loc, numerator, divisorIndex);
}

inline bool isInsideLoop(mlir::Operation *op) {
  mlir::Operation *parent = op->getParentOp();
  while (parent) {
    if (llvm::isa<mlir::scf::ForOp,
                  mlir::scf::WhileOp,
                  mlir::scf::ParallelOp>(parent)) {
      return true;
    }
    parent = parent->getParentOp();
  }
  return false;
}

class TransposeEliminationPattern : public OpRewritePattern<linalg::TransposeOp> {
public:
  TransposeEliminationPattern(MLIRContext *context)
      : OpRewritePattern<linalg::TransposeOp>(context) {}

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto permutation = transposeOp.getPermutation();
    SmallVector<int64_t> expectedPerm = {1, 0, 3, 2};
    if (!permutation.equals(expectedPerm)) {
      return failure();
    }

    Value transposeInput = transposeOp.getInput();
    auto viewOp = transposeInput.getDefiningOp<xsmt::ViewOp>();
    if (!viewOp) return failure();
    auto allocOp = viewOp.getBase().getDefiningOp<xsmt::AllocOp>();
    if (!allocOp) return failure();

    xsmt::DescriptorLoadViewOp descriptorLoadViewOp = nullptr;
    for (auto user : viewOp.getResult().getUsers()) {
      if (auto loadOp = dyn_cast<xsmt::DescriptorLoadViewOp>(user)) {
        descriptorLoadViewOp = loadOp;
        break;
      }
    }
    if (!descriptorLoadViewOp) return failure();

    Value transposeOutput = transposeOp->getResult(0);
    if (!transposeOutput.hasOneUse()) return failure();

    auto mmt4dUser = *transposeOutput.user_begin();
    auto mmt4dOp = dyn_cast<xsmt::MMT4DOp>(mmt4dUser);
    if (!mmt4dOp) return failure();

    rewriter.setInsertionPoint(allocOp);
    auto oldAllocType = dyn_cast<RankedTensorType>(allocOp.getResult().getType());
    if (!oldAllocType) return failure();
    auto oldShape = oldAllocType.getShape();
    if (oldShape.size() != 4) return failure();

    SmallVector<int64_t> newAllocShape;
    newAllocShape.push_back(oldShape[1]);
    newAllocShape.push_back(oldShape[0]);
    newAllocShape.push_back(oldShape[3]);
    newAllocShape.push_back(oldShape[2]);

    auto newAllocType = RankedTensorType::get(newAllocShape, oldAllocType.getElementType());
    auto oldShapeAttr = allocOp.getShape();
    auto oldMicroSizeAttr = allocOp.getMicroSize();

    SmallVector<int32_t> newShapeVec = {oldShapeAttr[1], oldShapeAttr[0]};
    SmallVector<int32_t> newMicroSizeVec = {oldMicroSizeAttr[1], oldMicroSizeAttr[0]};

    auto newAlloc = xsmt::AllocOp::create(rewriter, allocOp.getLoc(), newAllocType,
        rewriter.getDenseI32ArrayAttr(newShapeVec),
        rewriter.getDenseI32ArrayAttr(newMicroSizeVec));

    rewriter.setInsertionPoint(viewOp);
    auto oldOffsets = viewOp.getOffsets();
    auto oldShapeView = viewOp.getShape();
    auto oldMicroSizeView = viewOp.getMicroSize();
    SmallVector<Value> newOffsets = {oldOffsets[1], oldOffsets[0]};

    SmallVector<int32_t> newShapeViewVec = {oldShapeView[1], oldShapeView[0]};
    SmallVector<int32_t> newMicroSizeViewVec = {oldMicroSizeView[1], oldMicroSizeView[0]};

    auto oldViewType = dyn_cast<RankedTensorType>(viewOp.getResult().getType());
    if (!oldViewType) return failure();
    auto oldViewShape = oldViewType.getShape();

    SmallVector<int64_t> newViewShape;
    newViewShape.push_back(oldViewShape[1]);
    newViewShape.push_back(oldViewShape[0]);
    newViewShape.push_back(oldViewShape[3]);
    newViewShape.push_back(oldViewShape[2]);

    auto newViewType = RankedTensorType::get(newViewShape, oldViewType.getElementType());

    auto newView = xsmt::ViewOp::create(rewriter, viewOp.getLoc(), newViewType, newAlloc, newOffsets,
        rewriter.getDenseI32ArrayAttr(newShapeViewVec),
        rewriter.getDenseI32ArrayAttr(newMicroSizeViewVec));

    rewriter.setInsertionPoint(descriptorLoadViewOp);

    auto newDescriptorLoad = xsmt::DescriptorLoadViewOp::create(rewriter, descriptorLoadViewOp.getLoc(), newViewType,
        descriptorLoadViewOp.getBase(), newOffsets, newShapeViewVec, newMicroSizeViewVec, newView);
    newDescriptorLoad->setAttr("transpose", rewriter.getBoolAttr(true));
    rewriter.replaceAllUsesWith(transposeOp.getResult(), newView.getResult());

    rewriter.eraseOp(transposeOp);
    if (auto emptyOp = transposeOp.getDpsInitOperand(0)->get().getDefiningOp<tensor::EmptyOp>()) {
      if (emptyOp->use_empty()) {
        rewriter.eraseOp(emptyOp);
      }
    }
    if (descriptorLoadViewOp->use_empty()) {
      rewriter.eraseOp(descriptorLoadViewOp);
    }
    if (viewOp->use_empty()) {
      rewriter.eraseOp(viewOp);
    }
    if (allocOp->use_empty()) {
      rewriter.eraseOp(allocOp);
    }

    return success();
  }
};

struct MMT4DBindFusionPattern : public OpRewritePattern<xsmt::BindOp> {
  MMT4DBindFusionPattern(MLIRContext *context)
      : OpRewritePattern<xsmt::BindOp>(context) {}

  LogicalResult matchAndRewrite(xsmt::BindOp bindOp,
                                PatternRewriter &rewriter) const override {

    Value mmt4dResult = bindOp.getDest();
    Value accumulator = bindOp.getSrc();

    auto mmt4dOp = mmt4dResult.getDefiningOp<xsmt::MMT4DOp>();
    if (!mmt4dOp) {
      return failure();
    }

    if (mmt4dOp.getC()) {
      return failure();
    }

    rewriter.setInsertionPoint(bindOp);

    auto newMMT4DOp = xsmt::MMT4DOp::create(rewriter, bindOp.getLoc(),
        bindOp.getResult().getType(),
        mmt4dOp.getA(),
        mmt4dOp.getB(),
        accumulator
    );

    rewriter.replaceOp(bindOp, newMMT4DOp.getResult());

    if (mmt4dOp->use_empty()) {
        rewriter.eraseOp(mmt4dOp);
    }

    return success();
  }
};

template <typename OpTy>
struct MBarrierLoopCheckPattern : public OpRewritePattern<OpTy> {
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    Operation *currentParent = op->getParentOp();

    while (currentParent) {
      if (isInsideLoop(op.getOperation())) {
        return op.emitError()
        << "'" << op->getName() << "' operation cannot be used inside a loop. ";
      }
      if (isa<FunctionOpInterface>(currentParent)) {
        break;
      }
      currentParent = currentParent->getParentOp();
    }
    return failure();
  }
};

template <typename OpTy>
struct CheckGlobalMBarrierInLoopPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op, PatternRewriter &rewriter) const override {
    Value barrier = op.getMbarrier();

    Operation *definingOp = barrier.getDefiningOp();
    if (!definingOp) {
      return failure();
    }

    if (!llvm::isa<xsmt::GlobalMBarrierInitOp>(definingOp)) {
      return failure();
    }

    if (!isInsideLoop(op.getOperation())) {
      return failure();
    }

    op.emitError()
        << "'" << op->getName() << "' operation with barrier defined by "
        << "'xsmt.global_mbarrier_init' cannot be used inside a loop. "
        << "Global barriers are not designed for loop-level synchronization.";

    return failure();
  }
};

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
    if (!reinterpretCastOp){
      if(auto castop = reinterpretCast.getDefiningOp<memref::CastOp>()){
        reinterpretCast = castop.getOperand();
        if(auto reinterpretCastOp1 = reinterpretCast.getDefiningOp<memref::ReinterpretCastOp>()){
          reinterpretCastOp = reinterpretCastOp1;
        }else{
          return failure();
        }
      }else{
        return failure();
      }
    }

    auto sizes = reinterpretCastOp.getMixedSizes();
    if (sizes.size() != 2) {
      return failure();
    }

    Location loc = descriptorLoadOp.getLoc();
    Value dim0Value = ofrToIndexValue(loc, sizes[0], rewriter);
    Value dim1Value = ofrToIndexValue(loc, sizes[1], rewriter);

    auto microSizeAttr = descriptorLoadOp.getMicroSize();
    if (microSizeAttr.size() != 2) {
      return failure();
    }

    int32_t tile0 = microSizeAttr[0];
    int32_t tile1 = microSizeAttr[1];

    Value c1 = ConstantIndexOp::create(rewriter, loc, 1);
    Value tile0Value = ConstantIndexOp::create(rewriter, loc, tile0);
    Value tile1Value = ConstantIndexOp::create(rewriter, loc, tile1);

    auto offsets = descriptorLoadOp.getOffsets();
    Value offset0 = offsets[0];
    Value offset1 = offsets[1];
    Value rankedMemRef = reinterpretCast;

    Value indexOffset0 = ensureIndexType(loc, offset0, rewriter);
    Value indexOffset1 = ensureIndexType(loc, offset1, rewriter);
    Value indexDim0Value = ensureIndexType(loc, dim0Value, rewriter);
    Value indexDim1Value = ensureIndexType(loc, dim1Value, rewriter);

    auto shapes = descriptorLoadOp.getShape();
    Value size_m1 = SubIOp::create(rewriter, loc, indexDim0Value, indexOffset0);
    Value shape0 = ConstantIndexOp::create(rewriter, loc, shapes[0]);
    Value size_m2 = MinSIOp::create(rewriter, loc, size_m1, shape0);

    Value size_k1 = SubIOp::create(rewriter, loc, indexDim1Value, indexOffset1);
    Value shape1 = ConstantIndexOp::create(rewriter, loc, shapes[1]);
    Value size_k2 = MinSIOp::create(rewriter, loc, size_k1, shape1);

    Value subview = SubViewOp::create(rewriter, loc,
        rankedMemRef,
        ArrayRef<OpFoldResult>{indexOffset0, indexOffset1},
        ArrayRef<OpFoldResult>{size_m2, size_k2},
        ArrayRef<OpFoldResult>{c1, c1});

    auto subviewType = cast<MemRefType>(subview.getType());
    auto tensorType = RankedTensorType::get(subviewType.getShape(), subviewType.getElementType());
    Value tensor = ToTensorOp::create(rewriter, loc, tensorType, subview,
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
    Value tile0Value = ConstantIndexOp::create(rewriter, loc, tile0);
    Value tile1Value = ConstantIndexOp::create(rewriter, loc, tile1);
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

    Value emptyTensor = EmptyOp::create(rewriter, loc, emptyType, ValueRange{packedRows, packedCols});
    Value paddingValue = createZeroConstant(rewriter, loc, elementType);

    Value packedResult = PackOp::create(rewriter, loc,
        tensor,
        emptyTensor,
        ArrayRef<int64_t>{0, 1},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile0), rewriter.getIndexAttr(tile1)},
        paddingValue,
        ArrayRef<int64_t>{0, 1}
    );

    auto originalResultType = cast<RankedTensorType>(descriptorLoadOp.getResult().getType());
    Value castResult = tensor::CastOp::create(rewriter, loc, originalResultType, packedResult);

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
    Value tile0Value = ConstantIndexOp::create(rewriter, loc, tile0);
    Value tile1Value = ConstantIndexOp::create(rewriter, loc, tile1);
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

    Value emptyTensor = EmptyOp::create(rewriter, loc, emptyType, ValueRange{packedRows, packedCols});

    Value paddingValue = createZeroConstant(rewriter, loc, elementType);

    Value packedResult = PackOp::create(rewriter, loc,
        tensor,
        emptyTensor,
        ArrayRef<int64_t>{1, 0},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile1), rewriter.getIndexAttr(tile0)},
        paddingValue,
        ArrayRef<int64_t>{1, 0}
    );

    auto transposeResultType = cast<RankedTensorType>(transposeOp->getResult(0).getType());
    Value castResult = tensor::CastOp::create(rewriter, loc, transposeResultType , packedResult);

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



struct DescriptorLoadViewOpPattern : public OpRewritePattern<DescriptorLoadViewOp> {
public:
  DescriptorLoadViewOpPattern(MLIRContext *context)
      : OpRewritePattern<xsmt::DescriptorLoadViewOp>(context) {}

  LogicalResult matchAndRewrite(DescriptorLoadViewOp DescriptorLoadViewOp,
                                PatternRewriter &rewriter) const override {
    Value result = DescriptorLoadViewOp.getResult();

    Value sourcePtr = DescriptorLoadViewOp.getBase();
    auto unrealizedCast = sourcePtr.getDefiningOp<UnrealizedConversionCastOp>();
    if (!unrealizedCast) return failure();

    Value reinterpretCast = unrealizedCast.getOperand(0);
    auto reinterpretCastOp = reinterpretCast.getDefiningOp<memref::ReinterpretCastOp>();
    if (!reinterpretCastOp) return failure();

    auto sizes = reinterpretCastOp.getSizes();
    if (sizes.size() != 2) {
      return failure();
    }
    bool hasTranspose = DescriptorLoadViewOp->hasAttr("transpose") &&
                        DescriptorLoadViewOp->getAttrOfType<BoolAttr>("transpose").getValue();
    if (!hasTranspose){
      return failure();
    }

    Value  dim0Value = sizes[1];
    Value  dim1Value = sizes[0];

    ValueRange offsets = DescriptorLoadViewOp.getOffsets();
    ArrayRef<int32_t> shapes = DescriptorLoadViewOp.getShape();
    ArrayRef<int32_t> microSizeAttr = DescriptorLoadViewOp.getMicroSize();

    int32_t tile0 = microSizeAttr[0];
    int32_t tile1 = microSizeAttr[1];

    Location loc = DescriptorLoadViewOp.getLoc();
    Value c1 = ConstantIndexOp::create(rewriter, loc, 1);
    Value tile0Value = ConstantIndexOp::create(rewriter, loc, tile0);
    Value tile1Value = ConstantIndexOp::create(rewriter, loc, tile1);

    Value offset0 = offsets[0];
    Value offset1 = offsets[1];
    Value rankedMemRef = reinterpretCast;

    Value indexOffset0 = ensureIndexType(loc, offset0, rewriter);
    Value indexOffset1 = ensureIndexType(loc, offset1, rewriter);
    Value indexDim0Value = ensureIndexType(loc, dim0Value, rewriter);
    Value indexDim1Value = ensureIndexType(loc, dim1Value, rewriter);

    Value size_m1 = SubIOp::create(rewriter, loc, indexDim0Value, indexOffset0);
    Value shape0 = ConstantIndexOp::create(rewriter, loc, shapes[0]);
    Value dim0 = MinSIOp::create(rewriter, loc, size_m1, shape0);

    Value size_k1 = SubIOp::create(rewriter, loc, indexDim1Value, indexOffset1);
    Value shape1 = ConstantIndexOp::create(rewriter, loc, shapes[1]);
    Value dim1 = MinSIOp::create(rewriter, loc, size_k1, shape1);

    Value subview = SubViewOp::create(rewriter, loc,
        rankedMemRef,
        ArrayRef<OpFoldResult>{indexOffset1, indexOffset0},
        ArrayRef<OpFoldResult>{dim1, dim0},
        ArrayRef<OpFoldResult>{c1, c1});

    auto subviewType = cast<MemRefType>(subview.getType());
    auto tensorType = RankedTensorType::get(subviewType.getShape(), subviewType.getElementType());
    Value tensor = ToTensorOp::create(rewriter, loc, tensorType, subview,
                                              /*restrict=*/rewriter.getUnitAttr());
    Value packedRows = createCeilDivUI(rewriter, loc, dim0, tile0Value);
    Value packedCols = createCeilDivUI(rewriter, loc, dim1, tile1Value);

    SmallVector<int64_t, 4> shapeDims;
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(tile0);
    shapeDims.push_back(tile1);
    Type elementType = tensorType.getElementType();

    auto emptyType = RankedTensorType::get(shapeDims, elementType);

    Value emptyTensor = EmptyOp::create(rewriter, loc, emptyType, ValueRange{packedRows, packedCols});
    Value paddingValue = createZeroConstant(rewriter, loc, elementType);

    auto packOp = PackOp::create(rewriter, loc,
          tensor,
          emptyTensor,
          ArrayRef<int64_t>{1, 0},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile0), rewriter.getIndexAttr(tile1)},
          paddingValue,
          ArrayRef<int64_t>{1, 0});

    Value desSubview;
    Value castValue = DescriptorLoadViewOp->getOperand(3);
    if (auto subviewValue = findSubview(castValue)) {
      desSubview = subviewValue;
    } else {
      return failure();
    }

    Value c0 = ConstantIndexOp::create(rewriter, loc, 0);
    Value c16 = ConstantIndexOp::create(rewriter, loc, 16);
    Value c8 = ConstantIndexOp::create(rewriter, loc, 8);

    auto destMemRefType = dyn_cast<MemRefType>(desSubview.getType());
    if (!destMemRefType || destMemRefType.getRank() < 4) {
      return rewriter.notifyMatchFailure(DescriptorLoadViewOp, "Destination memref must be at least 4D");
    }

    Value destSubview = SubViewOp::create(rewriter, loc,
        desSubview,
        ArrayRef<OpFoldResult>{c0, c0, c0, c0},
        ArrayRef<OpFoldResult>{packedRows, packedCols, c16, c8},
        ArrayRef<OpFoldResult>{c1, c1, c1, c1});

    auto materializeOp = bufferization::MaterializeInDestinationOp::create(rewriter, packOp.getLoc(), packOp.getResult(), destSubview);
    materializeOp.setWritable(true);
    rewriter.eraseOp(DescriptorLoadViewOp);

    return success();
  }

private:
  Value findSubview(mlir::Value castValue) const {
    if (auto castOp = castValue.getDefiningOp<tensor::CastOp>()) {
      Value toTensorValue = castOp.getSource();
      if (auto toTensorOp = toTensorValue.getDefiningOp<bufferization::ToTensorOp>()) {
        Value subviewValue = toTensorOp->getOperand(0);
        return subviewValue;
        }
      }
    return nullptr;
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

    auto allocOp = memref::AllocOp::create(rewriter, loc, MemRefType::get(shape, outputType.getElementType()));
    PatternRewriter::InsertionGuard guard(rewriter);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    SmallVector<Value> ivs;

    scf::ForOp outermostLoop;

    for (size_t i = 0; i < shape.size(); ++i) {
      Value dimLimit = arith::ConstantIndexOp::create(rewriter, loc, shape[i]);
      auto loop = scf::ForOp::create(rewriter, loc, c0, dimLimit, c1);

      if (i == 0) {
        outermostLoop = loop;
      }

      ivs.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    memref::StoreOp::create(rewriter, loc, fillValue, allocOp, ivs);
    if (outermostLoop) {
      rewriter.setInsertionPointAfter(outermostLoop);
    }

    auto toTensorOp = bufferization::ToTensorOp::create(rewriter, loc, outputType, allocOp, /*restrict=*/true);

    rewriter.replaceOp(fillOp, toTensorOp.getResult());
    return success();
  }
};


class ConvertXSMTAllocToMemRef : public OpRewritePattern<xsmt::AllocOp> {
public:
  using OpRewritePattern<xsmt::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmt::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Location loc = allocOp.getLoc();
    TensorType resultType = cast<TensorType>(allocOp.getResult().getType());

    MemRefType memrefType = MemRefType::get(
        resultType.getShape(),
        resultType.getElementType()
    );

    Value memref = memref::AllocOp::create(rewriter, loc, memrefType);

    Value newTensor = bufferization::ToTensorOp::create(rewriter, loc, resultType, memref);

    rewriter.replaceOp(allocOp, newTensor);

    return success();
  }
};

struct ViewOpPattern : public OpRewritePattern<xsmt::ViewOp> {
public:
  ViewOpPattern(MLIRContext *context)
      : OpRewritePattern<xsmt::ViewOp>(context){}

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
    auto baseType = dyn_cast<TensorType>(base.getType());
    if (!baseType) {
      return rewriter.notifyMatchFailure(viewOp, "Base is not a tensor type");
    }
    auto memrefType = MemRefType::get(baseType.getShape(), baseType.getElementType());
    Value memref = memref::AllocOp::create(rewriter, loc, memrefType);

    auto materialize = bufferization::MaterializeInDestinationOp::create(rewriter, loc, base, memref);
    materialize.setWritable(true);

    bool hasMicroSize = false;
    bool isSameMicroSize = false;
    ArrayRef<int32_t> preMicroSize;

    if (auto allocOp = base.getDefiningOp<xsmt::AllocOp>()) {
        preMicroSize = allocOp.getMicroSize();
    } else if (auto loadOp = base.getDefiningOp<xsmt::DescriptorLoadOp>()) {
        preMicroSize = loadOp.getMicroSize();
    } else if (auto ViewOp = base.getDefiningOp<xsmt::ViewOp>()) {
        preMicroSize = ViewOp.getMicroSize();
    }

    if(!preMicroSize.empty()){
        hasMicroSize = true;
        if(preMicroSize == microSize){
            isSameMicroSize = true;
        }
    }

    SmallVector<OpFoldResult> sizes;
    if (auto ViewOp = base.getDefiningOp<xsmt::ViewOp>()){
      auto shapes = ViewOp.getShape();
      for (int64_t shape : shapes) {
        sizes.push_back(rewriter.getIndexAttr(shape));
      }
    }else if (auto AllocOp = base.getDefiningOp<xsmt::AllocOp>()){
      auto shapes = AllocOp.getShape();
      for (int64_t shape : shapes) {
        sizes.push_back(rewriter.getIndexAttr(shape));
      }
    }else if (auto FillOp = base.getDefiningOp<linalg::FillOp>()){
      auto tensorType = cast<RankedTensorType>(FillOp.getResult(0).getType());
      auto shapes = tensorType.getShape();
      for (int64_t shape : shapes) {
        sizes.push_back(rewriter.getIndexAttr(shape));
      }
    }else{
      if (baseType.getRank() == 2) {
        auto shapes = baseType.getShape();
        for (int64_t s : shapes) {
          sizes.push_back(rewriter.getIndexAttr(s));
        }
        return convertNormal(viewOp, memref, sizes, rewriter);
      } else if (baseType.getRank() == 4){
        return convertIsDiffMicroSize(viewOp, memref, rewriter);
      } else{
        return failure();
      }
    }


    if(!hasMicroSize && !isSameMicroSize) {
      return convertNormal(viewOp, memref, sizes, rewriter);
    }else if(hasMicroSize && isSameMicroSize){
      return convertIsSameMicroSize(viewOp, memref, sizes, rewriter);
    }
    else if(hasMicroSize && !isSameMicroSize){
      return convertIsDiffMicroSize(viewOp, memref, rewriter);
    }else{
      return failure();
    }
}

private:
    LogicalResult convertNormal(xsmt::ViewOp viewOp, Value memrefBaseValue, SmallVector<OpFoldResult> sizes, PatternRewriter &rewriter) const {
    Location loc = viewOp.getLoc();
    Value base = viewOp.getBase();
    ValueRange offsets = viewOp.getOffsets();
    ArrayRef<int32_t> shape = viewOp.getShape();
    ArrayRef<int32_t> microSize = viewOp.getMicroSize();

    auto resultType = cast<TensorType>(viewOp.getResult().getType());
    Type elementType = resultType.getElementType();

    SmallVector<OpFoldResult> offsetValues;

    for (auto offset : offsets) {
      offsetValues.push_back(ensureIndexType(loc, offset, rewriter));
    }

    Value c1 = ConstantIndexOp::create(rewriter, loc, 1);
    auto size_m1 = subOFRs(sizes[0], offsetValues[0], loc, rewriter);
    Value size_m2 = ofrToIndexValue(size_m1, loc, rewriter);
    Value shape0 = ConstantIndexOp::create(rewriter, loc, shape[0]);
    Value size_m3 = MinSIOp::create(rewriter, loc, size_m2, shape0);

    auto size_n1 = subOFRs(sizes[1], offsetValues[1], loc, rewriter);
    Value size_n2 = ofrToIndexValue(size_n1, loc, rewriter);
    Value shape1 = ConstantIndexOp::create(rewriter, loc, shape[1]);
    Value size_n3 = MinSIOp::create(rewriter, loc, size_n2, shape1);

    auto subview = memref::SubViewOp::create(rewriter, loc,
        memrefBaseValue,
        offsetValues,
        ArrayRef<OpFoldResult>{size_m3, size_n3},
        ArrayRef<OpFoldResult>{c1, c1});

    auto subviewType = cast<MemRefType>(subview.getType());
    auto tensorType = RankedTensorType::get(subviewType.getShape(), subviewType.getElementType());
    Value toTensorOp2 = bufferization::ToTensorOp::create(rewriter, loc, tensorType, subview,
                                              /*restrict=*/rewriter.getUnitAttr());

    Value tile0Value = ConstantIndexOp::create(rewriter, loc, microSize[0]);
    Value tile1Value = ConstantIndexOp::create(rewriter, loc, microSize[1]);
    Value packedRows = createCeilDivUI(rewriter, loc, size_m3, tile0Value);
    Value packedCols = createCeilDivUI(rewriter, loc, size_n3, tile1Value);

    SmallVector<int64_t, 4> shapeDims;
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(ShapedType::kDynamic);
    shapeDims.push_back(microSize[0]);
    shapeDims.push_back(microSize[1]);
    auto emptyType = RankedTensorType::get(shapeDims, tensorType.getElementType());
    Value emptyTensor = EmptyOp::create(rewriter, loc, emptyType, ValueRange{packedRows, packedCols});

    Value paddingValue = createZeroConstant(rewriter, loc, elementType);
    if (!paddingValue) {
      return rewriter.notifyMatchFailure(viewOp, "Unsupported element type for padding");
    }

    Value packedResult = linalg::PackOp::create(rewriter, loc,
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
    Value castResult = tensor::CastOp::create(rewriter, loc, originalResultType, packedResult);
    rewriter.replaceOp(viewOp, castResult);
    return success();
  }

  LogicalResult convertIsSameMicroSize(xsmt::ViewOp viewOp, Value memrefBaseValue, SmallVector<OpFoldResult> sizes, PatternRewriter &rewriter) const {

    Location loc = viewOp.getLoc();
    Value base = viewOp.getBase();
    ValueRange offsets = viewOp.getOffsets();
    ArrayRef<int32_t> shape = viewOp.getShape();
    ArrayRef<int32_t> microSize = viewOp.getMicroSize();

    auto resultType = cast<TensorType>(viewOp.getResult().getType());
    Type elementType = resultType.getElementType();

    SmallVector<OpFoldResult> offsetValues;

    for (auto offset : offsets) {
      offsetValues.push_back(ensureIndexType(loc, offset, rewriter));
    }

    Value c1 = ConstantIndexOp::create(rewriter, loc, 1);
    auto size_m1 = subOFRs(sizes[0], offsetValues[0], loc, rewriter);
    Value size_m2 = ofrToIndexValue(size_m1, loc, rewriter);
    Value shape0 = ConstantIndexOp::create(rewriter, loc, shape[0]);
    Value size_m3 = MinSIOp::create(rewriter, loc, size_m2, shape0);

    auto size_n1 = subOFRs(sizes[1], offsetValues[1], loc, rewriter);
    Value size_n2 = ofrToIndexValue(size_n1, loc, rewriter);
    Value shape1 = ConstantIndexOp::create(rewriter, loc, shape[1]);
    Value size_n3 = MinSIOp::create(rewriter, loc, size_n2, shape1);

    Value tile0Value = ConstantIndexOp::create(rewriter, loc, microSize[0]);
    Value tile1Value = ConstantIndexOp::create(rewriter, loc, microSize[1]);
    Value size0 = createCeilDivUI(rewriter, loc, size_m3, tile0Value);
    Value size1 = createCeilDivUI(rewriter, loc, size_n3, tile1Value);

    Value offset0 = createCeilDivUI(rewriter, loc, offsets[0], tile0Value);
    Value offset1 = createCeilDivUI(rewriter, loc, offsets[1], tile1Value);

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);

    auto subview = memref::SubViewOp::create(rewriter, loc,
        memrefBaseValue,
        ArrayRef<OpFoldResult>{offset0, offset1, c0, c0},
        ArrayRef<OpFoldResult>{size0, size1, tile0Value, tile1Value},
        ArrayRef<OpFoldResult>{c1, c1, c1, c1});

    auto subviewMemRefType = subview.getType();
    auto subviewTensorType = RankedTensorType::get(
        subviewMemRefType.getShape(),
        subviewMemRefType.getElementType()
    );

    Value newTensor = bufferization::ToTensorOp::create(rewriter, loc,
        subviewTensorType,
        subview.getResult(),
        /*restrict=*/rewriter.getUnitAttr()
    );

    Value castResult = tensor::CastOp::create(rewriter, loc, resultType, newTensor);
    rewriter.replaceOp(viewOp, castResult);

    return success();
  }

  LogicalResult convertIsDiffMicroSize(xsmt::ViewOp viewOp,
                                     Value memrefBaseValue,
                                     PatternRewriter &rewriter) const {
    Location loc = viewOp.getLoc();
    Value base = viewOp.getBase();
    ValueRange offsets = viewOp.getOffsets();
    ArrayRef<int32_t> shape = viewOp.getShape();
    ArrayRef<int32_t> microSize = viewOp.getMicroSize();

    auto resultType = cast<RankedTensorType>(viewOp.getResult().getType());
    Type elementType = resultType.getElementType();

    bool other = false;

    ArrayRef<int32_t> preMicroSize;
    if (auto allocOp = base.getDefiningOp<xsmt::AllocOp>()) {
      preMicroSize = allocOp.getMicroSize();
    } else if (auto loadOp = base.getDefiningOp<xsmt::DescriptorLoadOp>()) {
      preMicroSize = loadOp.getMicroSize();
    } else if (auto prevViewOp = base.getDefiningOp<xsmt::ViewOp>()) {
      preMicroSize = prevViewOp.getMicroSize();
    } else {
      other = true;
    }

    Value packedTensor = memrefBaseValue;
    ShapedType shapedTy = cast<ShapedType>(memrefBaseValue.getType());
    if (!isa<RankedTensorType>(memrefBaseValue.getType())) {
      auto tensorTy = RankedTensorType::get(shapedTy.getShape(), shapedTy.getElementType());
      packedTensor = bufferization::ToTensorOp::create(rewriter, loc, tensorTy, memrefBaseValue, rewriter.getUnitAttr());  // restrict=true
    }


    if (microSize[0] == 1 && microSize[1] == 1) {
      Value flatTensor = packedTensor;

      Value dim0 = tensor::DimOp::create(rewriter, loc, packedTensor, 0);
      Value dim1 = tensor::DimOp::create(rewriter, loc, packedTensor, 1);

      int64_t dim2 = shapedTy.getDimSize(2);
      int64_t dim3 = shapedTy.getDimSize(3);

      if(other){
        Value flatRows = arith::MulIOp::create(rewriter, loc, dim0, arith::ConstantIndexOp::create(rewriter, loc, dim2));
        Value flatCols = arith::MulIOp::create(rewriter, loc, dim1, arith::ConstantIndexOp::create(rewriter, loc, dim3));

        auto flatTy = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
        Value empty = tensor::EmptyOp::create(rewriter, loc, flatTy, ValueRange{flatRows, flatCols});

        flatTensor = linalg::UnPackOp::create(rewriter, loc, packedTensor, empty,
            ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{
                rewriter.getIndexAttr(dim2),
                rewriter.getIndexAttr(dim3)
            });
      }else{
        Value flatRows = arith::MulIOp::create(rewriter, loc, dim0, arith::ConstantIndexOp::create(rewriter, loc, preMicroSize[0]));
        Value flatCols = arith::MulIOp::create(rewriter, loc, dim1, arith::ConstantIndexOp::create(rewriter, loc, preMicroSize[1]));

        auto flatTy = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
        Value empty = tensor::EmptyOp::create(rewriter, loc, flatTy, ValueRange{flatRows, flatCols});

        flatTensor = linalg::UnPackOp::create(rewriter, loc, packedTensor, empty,
            ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{
                rewriter.getIndexAttr(preMicroSize[0]),
                rewriter.getIndexAttr(preMicroSize[1])
        });
      }

      Value off0 = ensureIndexType(loc, offsets[0], rewriter);
      Value off1 = ensureIndexType(loc, offsets[1], rewriter);
      Value reqRows = arith::ConstantIndexOp::create(rewriter, loc, shape[0]);
      Value reqCols = arith::ConstantIndexOp::create(rewriter, loc, shape[1]);

      Value flatRows = tensor::DimOp::create(rewriter, loc, flatTensor, 0);
      Value flatCols = tensor::DimOp::create(rewriter, loc, flatTensor, 1);
      Value availRows = arith::SubIOp::create(rewriter, loc, flatRows, off0);
      Value availCols = arith::SubIOp::create(rewriter, loc, flatCols, off1);
      Value actualRows = arith::MinUIOp::create(rewriter, loc, reqRows, availRows);
      Value actualCols = arith::MinUIOp::create(rewriter, loc, reqCols, availCols);

      Value slice = tensor::ExtractSliceOp::create(rewriter, loc, flatTensor,
          ArrayRef<Value>{off0, off1},
          ArrayRef<Value>{actualRows, actualCols},
          ArrayRef<Value>{arith::ConstantIndexOp::create(rewriter, loc, 1),
                          arith::ConstantIndexOp::create(rewriter, loc, 1)});

      Value result = tensor::CastOp::create(rewriter, loc, resultType, slice);
      rewriter.replaceOp(viewOp, result);
      return success();
    }

    Value dim0 = tensor::DimOp::create(rewriter, loc, packedTensor, 0);
    Value dim1 = tensor::DimOp::create(rewriter, loc, packedTensor, 1);

    int64_t dim2 = shapedTy.getDimSize(2);
    int64_t dim3 = shapedTy.getDimSize(3);

    Value flatRows = arith::MulIOp::create(rewriter, loc, dim0, arith::ConstantIndexOp::create(rewriter, loc, dim2));
    Value flatCols = arith::MulIOp::create(rewriter, loc, dim1, arith::ConstantIndexOp::create(rewriter, loc, dim3));

    auto flatTy = RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
    Value flatEmpty = tensor::EmptyOp::create(rewriter, loc, flatTy, ValueRange{flatRows, flatCols});

    Value flatTensor = linalg::UnPackOp::create(rewriter, loc, packedTensor, flatEmpty,
        ArrayRef<int64_t>{0, 1},
        ArrayRef<OpFoldResult>{
            rewriter.getIndexAttr(dim2),
            rewriter.getIndexAttr(dim3)
        });

    Value off0 = ensureIndexType(loc, offsets[0], rewriter);
    Value off1 = ensureIndexType(loc, offsets[1], rewriter);
    Value reqRows = arith::ConstantIndexOp::create(rewriter, loc, shape[0]);
    Value reqCols = arith::ConstantIndexOp::create(rewriter, loc, shape[1]);

    Value availRows = arith::SubIOp::create(rewriter, loc,
        tensor::DimOp::create(rewriter, loc, flatTensor, 0), off0);
    Value availCols = arith::SubIOp::create(rewriter, loc,
        tensor::DimOp::create(rewriter, loc, flatTensor, 1), off1);
    Value actualRows = arith::MinUIOp::create(rewriter, loc, reqRows, availRows);
    Value actualCols = arith::MinUIOp::create(rewriter, loc, reqCols, availCols);

    Value slice = tensor::ExtractSliceOp::create(rewriter, loc, flatTensor,
        ArrayRef<Value>{off0, off1},
        ArrayRef<Value>{actualRows, actualCols},
        ArrayRef<Value>{arith::ConstantIndexOp::create(rewriter, loc, 1),
                        arith::ConstantIndexOp::create(rewriter, loc, 1)});

    Value tile0 = arith::ConstantIndexOp::create(rewriter, loc, microSize[0]);
    Value tile1 = arith::ConstantIndexOp::create(rewriter, loc, microSize[1]);
    Value outerRows = createCeilDivUI(rewriter, loc, actualRows, tile0);
    Value outerCols = createCeilDivUI(rewriter, loc, actualCols, tile1);

    SmallVector<int64_t> packedShape = {ShapedType::kDynamic, ShapedType::kDynamic,
                                        microSize[0], microSize[1]};
    auto packedTy = RankedTensorType::get(packedShape, elementType);
    Value packedEmpty = tensor::EmptyOp::create(rewriter, loc, packedTy, ValueRange{outerRows, outerCols});

    Value padding = createZeroConstant(rewriter, loc, elementType);
    if (!padding)
      return rewriter.notifyMatchFailure(viewOp, "Cannot create padding value");

    Value packed = linalg::PackOp::create(rewriter, loc, slice, packedEmpty,
        ArrayRef<int64_t>{0, 1},
        ArrayRef<OpFoldResult>{
            rewriter.getIndexAttr(microSize[0]),
            rewriter.getIndexAttr(microSize[1])},
        padding);

    Value result = tensor::CastOp::create(rewriter, loc, resultType, packed);
    rewriter.replaceOp(viewOp, result);
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

    bool unpack = false;

    if (c) {
      if(c.hasOneUse() && op.getResult().use_empty())
        unpack = true;
      if(unpack){
        if (auto cast = c.getDefiningOp<tensor::CastOp>())
          c = cast.getSource();
        auto cPackOp = c.getDefiningOp<linalg::PackOp>();
        if (!cPackOp)
          return rewriter.notifyMatchFailure(op, "C must be result of linalg.pack");

        Value packInput = cPackOp.getSource();
        auto toTensorOp = packInput.getDefiningOp<bufferization::ToTensorOp>();
        if (!toTensorOp)
          return rewriter.notifyMatchFailure(op, "Packed input must come from to_tensor");

        Value subview = toTensorOp->getOperand(0);


        auto cType = dyn_cast<RankedTensorType>(c.getType());
        auto zeroAttr = FloatAttr::get(cType.getElementType(), 0.0);
        auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);

        auto fillOp = linalg::FillOp::create(rewriter, loc, TypeRange{c.getType()},
            ValueRange{zero},
            ValueRange{c}
        );

        auto mmt4dOp = linalg::Mmt4DOp::create(rewriter, loc, ValueRange{a, b}, ValueRange{fillOp.getResult(0)});

        RankedTensorType unpackedType = cast<RankedTensorType>(packInput.getType());

        SmallVector<OpFoldResult> mixedSizes =
            tensor::getMixedSizes(rewriter, loc, packInput);
        Value emptyUnpack = tensor::EmptyOp::create(rewriter, loc, mixedSizes, unpackedType.getElementType());
        ArrayRef<int64_t> staticInnerTiles = cPackOp.getStaticInnerTiles();
        if (staticInnerTiles.empty()) {
          return rewriter.notifyMatchFailure(op, "static_inner_tiles is empty");
        }

        SmallVector<OpFoldResult> innerTiles;
        for (int64_t tile : staticInnerTiles) {
          innerTiles.push_back(rewriter.getIndexAttr(tile));
        }
        auto unpackOp = linalg::UnPackOp::create(rewriter, loc,
            mmt4dOp.getResult(0),
            emptyUnpack,
            cPackOp.getInnerDimsPos(),
            innerTiles,
            cPackOp.getOuterDimsPerm());

        Value destination = traceDestinationFromOutput(c);

        auto materializeOp = bufferization::MaterializeInDestinationOp::create(rewriter, unpackOp.getLoc(), unpackOp.getResult(), destination);
        materializeOp.setWritable(true);

        rewriter.replaceOp(op, unpackOp.getResult());
        return success();
      }else{
      if (auto cast = c.getDefiningOp<tensor::CastOp>())
        c = cast.getSource();
      auto toTensorOp = c.getDefiningOp<bufferization::ToTensorOp>();
      if (!toTensorOp)
        return rewriter.notifyMatchFailure(op, "c must come from to_tensor");

      auto subview = toTensorOp->getOperand(0);

      auto cType = dyn_cast<RankedTensorType>(c.getType());
      auto zeroAttr = FloatAttr::get(cType.getElementType(), 0.0);
      auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);

      auto fillOp = linalg::FillOp::create(rewriter, loc, TypeRange{c.getType()},
          ValueRange{zero},
          ValueRange{c}
      );

      auto mmt4dOp = linalg::Mmt4DOp::create(rewriter, loc, ValueRange{a, b}, ValueRange{fillOp.getResult(0)});

      auto materializeOp = bufferization::MaterializeInDestinationOp::create(rewriter, mmt4dOp.getLoc(), mmt4dOp.getResult(0), subview);
        materializeOp.setWritable(true);
      rewriter.replaceOp(op, mmt4dOp.getResult(0));
      return success();
      }
    }else{
      auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
      Value initTensor = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(), resultType.getElementType());
      auto zeroAttr = FloatAttr::get(resultType.getElementType(), 0.0);
      auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);

      auto fillOp = linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{initTensor});

      auto mmt4dOp = linalg::Mmt4DOp::create(rewriter, loc, ValueRange{a, b}, ValueRange{fillOp.getResult(0)});
      rewriter.replaceOp(op, mmt4dOp.getResult(0));
      return success();
    }
  }
    Value traceDestinationFromOutput(Value output) const{
    if (auto packOp = output.getDefiningOp<linalg::PackOp>()) {
      Value packInput = packOp.getSource();
      if (auto toTensorOp = packInput.getDefiningOp<bufferization::ToTensorOp>()) {
        return toTensorOp->getOperand(0);
      }

      return packInput;
    }
    if (auto toTensorOp = output.getDefiningOp<bufferization::ToTensorOp>()) {
      return toTensorOp->getOperand(0);
    }
    return Value();
  }
};

struct ConvertMMT4DAddPattern : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp, PatternRewriter &rewriter) const override {
    for (auto it : genericOp.getIteratorTypesArray()) {
      if (it != utils::IteratorType::parallel)
        return rewriter.notifyMatchFailure(genericOp, "non-parallel iterators");
    }

    auto maps = genericOp.getIndexingMaps();
    if (maps.size() != 3)
      return rewriter.notifyMatchFailure(genericOp, "wrong number of maps");

    auto &block = genericOp.getRegion().front();
    if (!llvm::hasSingleElement(block.without_terminator())) {
      return rewriter.notifyMatchFailure(genericOp, "multiple block operations");
    }

    auto addOp = dyn_cast<arith::AddFOp>(*block.begin());
    if (!addOp)
      return rewriter.notifyMatchFailure(genericOp, "not an addf operation");
    if (addOp.getLhs() != block.getArgument(0) ||
        addOp.getRhs() != block.getArgument(1)) {
      return rewriter.notifyMatchFailure(genericOp, "addf not using block args");
    }

    Value mmt4dResult;
    Value otherInput;
    if (auto mmt4d = genericOp.getInputs()[0].getDefiningOp<MMT4DOp>()) {
      mmt4dResult = genericOp.getInputs()[0];
      otherInput = genericOp.getInputs()[1];
    } else if (auto mmt4d = genericOp.getInputs()[1].getDefiningOp<MMT4DOp>()) {
      mmt4dResult = genericOp.getInputs()[1];
      otherInput = genericOp.getInputs()[0];
    } else {
      return rewriter.notifyMatchFailure(genericOp, "no xsmt.mmt4d input");
    }

    auto mmt4dOp = cast<MMT4DOp>(mmt4dResult.getDefiningOp());
    if (mmt4dOp.getNumOperands() != 2) {
      return rewriter.notifyMatchFailure(genericOp, "mmt4d has accumulator");
    }

    Value accTensor = genericOp.getOutputs()[0];
    if (accTensor != otherInput) {
      return rewriter.notifyMatchFailure(genericOp, "output not matching accumulation target");
    }

    auto resType = dyn_cast<RankedTensorType>(mmt4dResult.getType());
    if (!resType || resType.getRank() != 4 || !resType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(genericOp, "non-static 4D result type");
    }
    auto shape = resType.getShape();

    Location loc = genericOp.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(genericOp);

    Value a = mmt4dOp.getA();
    Value b = mmt4dOp.getB();

    if (auto cast = a.getDefiningOp<tensor::CastOp>())
      a = cast.getSource();
    if (auto cast = b.getDefiningOp<tensor::CastOp>())
      b = cast.getSource();

    SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes;

    Value dim0 = tensor::DimOp::create(rewriter, loc, a, 0);
    Value dim1 = tensor::DimOp::create(rewriter, loc, b, 0);
    sizes.push_back(dim0);
    sizes.push_back(dim1);
    sizes.push_back(rewriter.getIndexAttr(shape[2]));
    sizes.push_back(rewriter.getIndexAttr(shape[3]));

    SmallVector<int64_t> sliceShape = {
        ShapedType::kDynamic, ShapedType::kDynamic,
        shape[2], shape[3]
    };
    auto sliceType = RankedTensorType::get(sliceShape, resType.getElementType());

    auto extractOp = tensor::ExtractSliceOp::create(rewriter, loc, sliceType, accTensor, offsets, sizes, strides);

    auto mmt4dResType = RankedTensorType::get(sliceShape, resType.getElementType());
    auto newMmt4dOp = linalg::Mmt4DOp::create(rewriter, loc, mmt4dResType,
        ValueRange{a, b},
        ValueRange{extractOp.getResult()}
    );

    auto insertOp = tensor::InsertSliceOp::create(rewriter, loc, newMmt4dOp.getResult(0), accTensor,
        offsets, sizes, strides
    );

    rewriter.replaceOp(genericOp, insertOp.getResult());
    rewriter.eraseOp(mmt4dOp);

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
      return arith::IndexCastOp::create(rewriter, forOp.getLoc(), rewriter.getIndexType(), val);
    };

    SmallVector<OpFoldResult> lbs = {convertToIndex(lb)};
    SmallVector<OpFoldResult> ubs = {convertToIndex(ub)};
    SmallVector<OpFoldResult> steps = {convertToIndex(step)};
    SmallVector<Value> outputs;
    auto initArgs = forOp.getInitArgs();
    for (auto initArg : initArgs) {
      outputs.push_back(initArg);
    }

    auto forallOp = scf::ForallOp::create(rewriter, forOp.getLoc(), lbs, ubs, steps, outputs, std::nullopt);

    Block &forallBlock = forallOp.getRegion().front();
    Block &forBody = *forOp.getBody();
    auto inParallelOp = cast<scf::InParallelOp>(forallBlock.getTerminator());
    rewriter.setInsertionPoint(inParallelOp);
    IRMapping mapping;

    Value forallIndVar = forallBlock.getArgument(0);
    auto castIndVar = arith::IndexCastOp::create(rewriter, forOp.getLoc(), indVar.getType(), forallIndVar);
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

class ReplaceRedundantMaterializePattern1 : public OpRewritePattern<bufferization::MaterializeInDestinationOp> {
public:
  using OpRewritePattern<bufferization::MaterializeInDestinationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::MaterializeInDestinationOp materializeOp,
                                PatternRewriter &rewriter) const override {

    auto toTensorOp = materializeOp.getSource().getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorOp) return failure();

    auto destAlloc = materializeOp.getDest().getDefiningOp<memref::AllocOp>();
    if (!destAlloc) return failure();

    auto srcAlloc = toTensorOp->getOperand(0).getDefiningOp<memref::AllocOp>();
    if (!srcAlloc) return failure();

    if (srcAlloc.getType() != destAlloc.getType()) {
      return failure();
    }

    Value srcMemref = srcAlloc.getResult();
    Value destMemref = destAlloc.getResult();

    bool hasOtherUses = false;
    for (OpOperand &use : llvm::make_early_inc_range(destMemref.getUses())) {
      if (use.getOwner() == materializeOp) continue;

      rewriter.modifyOpInPlace(use.getOwner(), [&]() {
        use.set(srcMemref);
      });
      hasOtherUses = true;
    }

    rewriter.eraseOp(materializeOp);

    if (!hasOtherUses && destAlloc->use_empty()) {
      rewriter.eraseOp(destAlloc);
    }

    return success();
  }
};
struct ReplaceRedundantMaterializePattern2
    : public OpRewritePattern<bufferization::MaterializeInDestinationOp> {
  using OpRewritePattern<bufferization::MaterializeInDestinationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::MaterializeInDestinationOp matOp,
                                PatternRewriter &rewriter) const override {
    auto allocOp = matOp.getDest().getDefiningOp<memref::AllocOp>();
    if (!allocOp) return failure();

    auto loop = dyn_cast<scf::ForOp>(matOp->getParentOp());
    if (!loop) return failure();

    Value sourceTensor = matOp.getSource();
    if (!sourceTensor.getDefiningOp() || loop->isAncestor(sourceTensor.getDefiningOp()))
      return failure();

    MemRefType memrefType = allocOp.getType();
    if (memrefType.getRank() != 4 || !memrefType.hasStaticShape())
      return failure();

    rewriter.setInsertionPoint(loop);
    auto hoistedAlloc = memref::AllocOp::create(rewriter, loop.getLoc(), memrefType);
    auto hoistMat = bufferization::MaterializeInDestinationOp::create(rewriter, loop.getLoc(), sourceTensor, hoistedAlloc.getResult());
    hoistMat->setAttr("writable", rewriter.getUnitAttr());

    Value hoistedBuffer = hoistedAlloc.getResult();

    matOp.getDest().replaceAllUsesWith(hoistedBuffer);

    rewriter.eraseOp(matOp);
    rewriter.eraseOp(allocOp);

    SmallVector<Operation*> deadOps;
    for (auto &use : llvm::make_early_inc_range(sourceTensor.getUses())) {
      auto redundant = dyn_cast<bufferization::MaterializeInDestinationOp>(use.getOwner());
      if (!redundant) continue;
      if (redundant.getSource() != sourceTensor) continue;
      if (loop->isAncestor(redundant)) continue;

      Value deadMemref = redundant.getDest();
      if (auto deadAlloc = deadMemref.getDefiningOp<memref::AllocOp>()) {
        for (auto *user : llvm::make_early_inc_range(deadMemref.getUsers())) {
          if (user == redundant) continue;
          rewriter.modifyOpInPlace(user, [&] {
            for (unsigned i = 0; i < user->getNumOperands(); ++i) {
              if (user->getOperand(i) == deadMemref) {
                user->setOperand(i, hoistedBuffer);
              }
            }
          });
        }
      }
    }
    return success();
  }
};

struct RemoveUnusedAllocPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    Value memrefResult = allocOp.getMemref();

    SmallVector<Operation *> opsToErase;

    for (Operation *user : memrefResult.getUsers()) {

      if (auto matOp = dyn_cast<MaterializeInDestinationOp>(user)) {
        if (matOp.getDest() == memrefResult) {
          opsToErase.push_back(user);
          continue;
        }
      }

      if (isa<memref::DeallocOp>(user)) {
        opsToErase.push_back(user);
        continue;
      }

      return failure();
    }
    for (Operation *op : opsToErase) {
      rewriter.eraseOp(op);
    }
    rewriter.eraseOp(allocOp);

    return success();
  }
};

struct RemoveRedundantBufferizationRoundtrip : public OpRewritePattern<ToTensorOp> {
  using OpRewritePattern<ToTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToTensorOp toTensorOp, PatternRewriter &rewriter) const override {
    Value memref = toTensorOp->getOperand(0);

    auto allocOp = memref.getDefiningOp<memref::AllocOp>();
    if (!allocOp) {
      return failure();
    }

    MaterializeInDestinationOp matOp;
    Operation *deallocOp = nullptr;

    for (Operation *user : memref.getUsers()) {
      if (user == toTensorOp) continue;

      if (auto m = dyn_cast<MaterializeInDestinationOp>(user)) {
        if (m.getDest() == memref) {
          if (matOp) {
            return failure();
          }
          matOp = m;
          continue;
        }
      }

      if (isa<memref::DeallocOp>(user)) {
        deallocOp = user;
        continue;
      }
      return failure();
    }

    if (!matOp) {
      return failure();
    }

    Value sourceTensor = matOp.getSource();

    if (sourceTensor.getType() != toTensorOp.getResult().getType()) {
      return failure();
    }

    rewriter.replaceOp(toTensorOp, sourceTensor);
    if (deallocOp) {
      rewriter.eraseOp(deallocOp);
    }
    rewriter.eraseOp(matOp);
    rewriter.eraseOp(allocOp);

    return success();
  }
};


struct GetThreadOpToLLVMCallPattern : public OpRewritePattern<xsmt::GetThreadOp> {
  using OpRewritePattern<xsmt::GetThreadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmt::GetThreadOp getThreadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = getThreadOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = getThreadOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return failure();
    }

    StringRef funcName = "spine_get_stream_threads";
    Type resultType = rewriter.getI32Type();

    auto llvmFuncOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!llvmFuncOp) {
      auto funcType = LLVM::LLVMFunctionType::get(resultType, /*params=*/{}, /*isVarArg=*/false);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      llvmFuncOp = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcType);
      llvmFuncOp.setLinkage(LLVM::Linkage::External);
    }

    auto callOp = LLVM::CallOp::create(rewriter, loc,
        resultType,
        funcName,
        ValueRange{});

    rewriter.replaceOp(getThreadOp, callOp.getResults());

    return success();
  }
};


struct InsertMBarrierReleasePattern : public mlir::OpRewritePattern<xsmt_async::MBarrierAllocOp> {
  using OpRewritePattern<xsmt_async::MBarrierAllocOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(xsmt_async::MBarrierAllocOp op,
                                      mlir::PatternRewriter &rewriter) const override {
    mlir::Value barrier = op.getResult();
    mlir::Block *parentBlock = op->getBlock();

    for (mlir::Operation *user : barrier.getUsers()) {
      if (mlir::isa<xsmt_async::MBarrierReleaseOp>(user)) {
        return mlir::failure();
      }
    }

    mlir::Operation *lastUser = op.getOperation();
    bool usedInReturn = false;

    for (mlir::Operation *user : barrier.getUsers()) {
      mlir::Operation *ancestor = parentBlock->findAncestorOpInBlock(*user);

      if (!ancestor || ancestor->hasTrait<mlir::OpTrait::IsTerminator>()) {
         usedInReturn = true;
         break;
      }

      if (lastUser->isBeforeInBlock(ancestor)) {
        lastUser = ancestor;
      }
    }

    if (usedInReturn) {
      return mlir::failure();
    }

    rewriter.setInsertionPointAfter(lastUser);
    xsmt_async::MBarrierReleaseOp::create(rewriter, op.getLoc(), barrier);
    return mlir::success();
  }
};


void mlir::triton::populateXSMTOptimizationAndValidationPatterns(RewritePatternSet &patterns) {
  patterns.add<TransposeEliminationPattern>(patterns.getContext());
  patterns.add<MMT4DBindFusionPattern>(patterns.getContext());
  patterns.add<MBarrierLoopCheckPattern<xsmt_async::MBarrierAllocOp>>(patterns.getContext());
  patterns.add<MBarrierLoopCheckPattern<xsmt::GlobalMBarrierInitOp>>(patterns.getContext());
  patterns.add<CheckGlobalMBarrierInLoopPattern<xsmt_async::MBarrierArriveOp>>(patterns.getContext());
  patterns.add<CheckGlobalMBarrierInLoopPattern<xsmt_async::MBarrierWaitOp>>(patterns.getContext());
}

void mlir::triton::populateXSMTToLinalgConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<FillOpPattern>(patterns.getContext());
  patterns.add<ConvertXSMTAllocToMemRef>(patterns.getContext());
  patterns.add<ViewOpPattern>(patterns.getContext());
  patterns.add<DescriptorLoadPattern>(patterns.getContext());
  patterns.add<DescriptorLoadViewOpPattern>(patterns.getContext());
  patterns.add<InsertMBarrierReleasePattern>(patterns.getContext());
  patterns.add<GetThreadOpToLLVMCallPattern>(patterns.getContext());
  }

void mlir::triton::ConvertMMT4DAddConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ConvertMMT4DAddPattern>(patterns.getContext());
}

void mlir::triton::MMT4DOpConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<LowerXSMTMMT4D>(patterns.getContext());
}

void mlir::triton::LoopParallelizationConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ForToForallPattern>(patterns.getContext());
  patterns.add<ReplaceRedundantMaterializePattern1>(patterns.getContext());
  patterns.add<ReplaceRedundantMaterializePattern2>(patterns.getContext());
}

void mlir::triton::BufferizationCleanupConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<RemoveUnusedAllocPattern>(patterns.getContext());
  patterns.add<RemoveRedundantBufferizationRoundtrip>(patterns.getContext());
}
