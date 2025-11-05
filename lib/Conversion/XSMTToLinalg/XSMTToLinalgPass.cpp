//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "triton-shared/Conversion/XSMTToLinalg/XSMTToLinalg.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "xsmt-to-linalg"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_XSMTTOLINALG
#include "triton-shared/Conversion/XSMTToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class XSMTToLinalgPass
    : public triton::impl::XSMTToLinalgBase<XSMTToLinalgPass> {
  using XSMTToLinalgBase<XSMTToLinalgPass>::XSMTToLinalgBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                triton::TritonDialect, ttx::TritonTilingExtDialect,
                tts::TritonStructuredDialect, mlir::LLVM::LLVMDialect, mlir::xsmt::XSMTDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    bool hasXSMTUser = false;
    moduleOp.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("xsmt.")) {
        hasXSMTUser = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasXSMTUser) {
      RewritePatternSet patterns(&getContext());
      triton::ForToForallConversionPatterns(patterns);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        signalPassFailure();
      }
      return;
    }

    RewritePatternSet patterns1(&getContext());
    RewritePatternSet patterns2(&getContext());
    RewritePatternSet patterns3(&getContext());
    RewritePatternSet patterns4(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
        cf::ControlFlowDialect, tensor::TensorDialect,
        bufferization::BufferizationDialect, ttx::TritonTilingExtDialect,
        memref::MemRefDialect>();


    target.addIllegalOp<linalg::FillOp>();
    triton::fillToMemrefConversionPatterns(patterns1);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns1)))) {
      signalPassFailure();
    }

    target.addIllegalOp<bufferization::MaterializeInDestinationOp, tensor::ExtractSliceOp, xsmt::DescriptorLoadViewOp, xsmt::ViewOp>();
    triton::populateXSMTToLinalgConversionPatterns(patterns2);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns2)))) {
      signalPassFailure();
    }

    target.addIllegalOp<xsmt::MMT4DOp>();
    triton::MMT4DOpConversionPatterns(patterns3);
    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns3)))) {
      signalPassFailure();
    }


    moduleOp.walk([&](linalg::UnPackOp unpackOp) {
      auto mmt4dOp = unpackOp.getSource().getDefiningOp<linalg::Mmt4DOp>();
      if (!mmt4dOp) return;
      if (mmt4dOp->getNumOperands() < 3) {
        llvm::dbgs() << "MMT4D operation has fewer than 3 operands\n";
        return;
      }
      Value outputOperand = mmt4dOp->getOperand(mmt4dOp->getNumOperands() - 1);
      Value destination = traceDestinationFromOutput(outputOperand);
      if (!destination) {
        llvm::dbgs() << "Destination not found\n";
        return;
      }

      OpBuilder builder(unpackOp);
      builder.setInsertionPointAfter(unpackOp);
      auto materializeOp = builder.create<bufferization::MaterializeInDestinationOp>(
          unpackOp.getLoc(), unpackOp.getResult(), destination);
      materializeOp.setWritable(true);
    });

    triton::ForToForallConversionPatterns(patterns4);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns4)))) {
      signalPassFailure();
    }
  }

  Value traceDestinationFromOutput(Value output) {
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

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createXSMTToLinalgPass() {
  return std::make_unique<XSMTToLinalgPass>();
}